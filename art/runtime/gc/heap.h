/*
 * Copyright (C) 2008 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ART_RUNTIME_GC_HEAP_H_
#define ART_RUNTIME_GC_HEAP_H_

#include <iosfwd>
#include <string>
#include <unordered_set>
#include <vector>

#include <android-base/logging.h>

#include "allocator_type.h"
#include "base/atomic.h"
#include "base/histogram.h"
#include "base/macros.h"
#include "base/mutex.h"
#include "base/runtime_debug.h"
#include "base/safe_map.h"
#include "base/time_utils.h"
#include "gc/collector/gc_type.h"
#include "gc/collector/iteration.h"
#include "gc/collector_type.h"
#include "gc/gc_cause.h"
#include "gc/space/large_object_space.h"
#include "handle.h"
#include "obj_ptr.h"
#include "offsets.h"
#include "process_state.h"
#include "read_barrier_config.h"
#include "runtime_globals.h"
#include "verify_object.h"

namespace art {

class ConditionVariable;
enum class InstructionSet;
class IsMarkedVisitor;
class Mutex;
class ReflectiveValueVisitor;
class RootVisitor;
class StackVisitor;
class Thread;
class ThreadPool;
class TimingLogger;
class VariableSizedHandleScope;

namespace mirror {
class Class;
class Object;
}  // namespace mirror

namespace gc {

class AllocationListener;
class AllocRecordObjectMap;
class GcPauseListener;
class HeapTask;
class ReferenceProcessor;
class TaskProcessor;
class Verification;

namespace accounting {
template <typename T> class AtomicStack;
using ObjectStack = AtomicStack<mirror::Object>;
class CardTable;
class HeapBitmap;
class ModUnionTable;
class ReadBarrierTable;
class RememberedSet;
}  // namespace accounting

namespace collector {
class ConcurrentCopying;
class GarbageCollector;
class MarkSweep;
class SemiSpace;
}  // namespace collector

namespace allocator {
class RosAlloc;
}  // namespace allocator

namespace space {
class AllocSpace;
class BumpPointerSpace;
class ContinuousMemMapAllocSpace;
class DiscontinuousSpace;
class DlMallocSpace;
class ImageSpace;
class LargeObjectSpace;
class MallocSpace;
class RegionSpace;
class RosAllocSpace;
class Space;
class ZygoteSpace;
}  // namespace space

enum HomogeneousSpaceCompactResult {
  // Success.
  kSuccess,
  // Reject due to disabled moving GC.
  kErrorReject,
  // Unsupported due to the current configuration.
  kErrorUnsupported,
  // System is shutting down.
  kErrorVMShuttingDown,
};

// If true, use rosalloc/RosAllocSpace instead of dlmalloc/DlMallocSpace
static constexpr bool kUseRosAlloc = true;

// If true, use thread-local allocation stack.
static constexpr bool kUseThreadLocalAllocationStack = true;

class Heap {
 public:
  // How much we grow the TLAB if we can do it.
  static constexpr size_t kPartialTlabSize = 16 * KB;
  static constexpr bool kUsePartialTlabs = true;

  static constexpr size_t kDefaultStartingSize = kPageSize;
  static constexpr size_t kDefaultInitialSize = 2 * MB;
  static constexpr size_t kDefaultMaximumSize = 256 * MB;
  static constexpr size_t kDefaultNonMovingSpaceCapacity = 64 * MB;
  static constexpr size_t kDefaultMaxFree = 2 * MB;
  static constexpr size_t kDefaultMinFree = kDefaultMaxFree / 4;
  static constexpr size_t kDefaultLongPauseLogThreshold = MsToNs(5);
  static constexpr size_t kDefaultLongPauseLogThresholdGcStress = MsToNs(50);
  static constexpr size_t kDefaultLongGCLogThreshold = MsToNs(100);
  static constexpr size_t kDefaultLongGCLogThresholdGcStress = MsToNs(1000);
  static constexpr size_t kDefaultTLABSize = 32 * KB;
  static constexpr double kDefaultTargetUtilization = 0.75;
  static constexpr double kDefaultHeapGrowthMultiplier = 2.0;
  // Primitive arrays larger than this size are put in the large object space.
  static constexpr size_t kMinLargeObjectThreshold = 3 * kPageSize;
  static constexpr size_t kDefaultLargeObjectThreshold = kMinLargeObjectThreshold;
  // Whether or not parallel GC is enabled. If not, then we never create the thread pool.
  static constexpr bool kDefaultEnableParallelGC = false;
  static uint8_t* const kPreferredAllocSpaceBegin;

  // Whether or not we use the free list large object space. Only use it if USE_ART_LOW_4G_ALLOCATOR
  // since this means that we have to use the slow msync loop in MemMap::MapAnonymous.
  static constexpr space::LargeObjectSpaceType kDefaultLargeObjectSpaceType =
      USE_ART_LOW_4G_ALLOCATOR ?
          space::LargeObjectSpaceType::kFreeList
        : space::LargeObjectSpaceType::kMap;

  // Used so that we don't overflow the allocation time atomic integer.
  static constexpr size_t kTimeAdjust = 1024;

  // Client should call NotifyNativeAllocation every kNotifyNativeInterval allocations.
  // Should be chosen so that time_to_call_mallinfo / kNotifyNativeInterval is on the same order
  // as object allocation time. time_to_call_mallinfo seems to be on the order of 1 usec
  // on Android.
#ifdef __ANDROID__
  static constexpr uint32_t kNotifyNativeInterval = 64;
#else
  // Some host mallinfo() implementations are slow. And memory is less scarce.
  static constexpr uint32_t kNotifyNativeInterval = 384;
#endif

  // RegisterNativeAllocation checks immediately whether GC is needed if size exceeds the
  // following. kCheckImmediatelyThreshold * kNotifyNativeInterval should be small enough to
  // make it safe to allocate that many bytes between checks.
  static constexpr size_t kCheckImmediatelyThreshold = 300000;

  // How often we allow heap trimming to happen (nanoseconds).
  static constexpr uint64_t kHeapTrimWait = MsToNs(5000);
  // How long we wait after a transition request to perform a collector transition (nanoseconds).
  static constexpr uint64_t kCollectorTransitionWait = MsToNs(5000);
  // Whether the transition-wait applies or not. Zero wait will stress the
  // transition code and collector, but increases jank probability.
  DECLARE_RUNTIME_DEBUG_FLAG(kStressCollectorTransition);

  // Create a heap with the requested sizes. The possible empty
  // image_file_names names specify Spaces to load based on
  // ImageWriter output.
  Heap(size_t initial_size,
       size_t growth_limit,
       size_t min_free,
       size_t max_free,
       double target_utilization,
       double foreground_heap_growth_multiplier,
       size_t stop_for_native_allocs,
       size_t capacity,
       size_t non_moving_space_capacity,
       const std::vector<std::string>& boot_class_path,
       const std::vector<std::string>& boot_class_path_locations,
       const std::vector<int>& boot_class_path_fds,
       const std::vector<int>& boot_class_path_image_fds,
       const std::vector<int>& boot_class_path_vdex_fds,
       const std::vector<int>& boot_class_path_oat_fds,
       const std::vector<std::string>& image_file_names,
       InstructionSet image_instruction_set,
       CollectorType foreground_collector_type,
       CollectorType background_collector_type,
       space::LargeObjectSpaceType large_object_space_type,
       size_t large_object_threshold,
       size_t parallel_gc_threads,
       size_t conc_gc_threads,
       bool low_memory_mode,
       size_t long_pause_threshold,
       size_t long_gc_threshold,
       bool ignore_target_footprint,
       bool always_log_explicit_gcs,
       bool use_tlab,
       bool verify_pre_gc_heap,
       bool verify_pre_sweeping_heap,
       bool verify_post_gc_heap,
       bool verify_pre_gc_rosalloc,
       bool verify_pre_sweeping_rosalloc,
       bool verify_post_gc_rosalloc,
       bool gc_stress_mode,
       bool measure_gc_performance,
       bool use_homogeneous_space_compaction,
       bool use_generational_cc,
       uint64_t min_interval_homogeneous_space_compaction_by_oom,
       bool dump_region_info_before_gc,
       bool dump_region_info_after_gc);

  ~Heap();

  // Allocates and initializes storage for an object instance.
  template <bool kInstrumented = true, typename PreFenceVisitor>
  mirror::Object* AllocObject(Thread* self,
                              ObjPtr<mirror::Class> klass,
                              size_t num_bytes,
                              const PreFenceVisitor& pre_fence_visitor)
      REQUIRES_SHARED(Locks::mutator_lock_)
      REQUIRES(!*gc_complete_lock_,
               !*pending_task_lock_,
               !*backtrace_lock_,
               !process_state_update_lock_,
               !Roles::uninterruptible_) {
    return AllocObjectWithAllocator<kInstrumented>(self,
                                                   klass,
                                                   num_bytes,
                                                   GetCurrentAllocator(),
                                                   pre_fence_visitor);
  }

  template <bool kInstrumented = true, typename PreFenceVisitor>
  mirror::Object* AllocNonMovableObject(Thread* self,
                                        ObjPtr<mirror::Class> klass,
                                        size_t num_bytes,
                                        const PreFenceVisitor& pre_fence_visitor)
      REQUIRES_SHARED(Locks::mutator_lock_)
      REQUIRES(!*gc_complete_lock_,
               !*pending_task_lock_,
               !*backtrace_lock_,
               !process_state_update_lock_,
               !Roles::uninterruptible_) {
    mirror::Object* obj = AllocObjectWithAllocator<kInstrumented>(self,
                                                                  klass,
                                                                  num_bytes,
                                                                  GetCurrentNonMovingAllocator(),
                                                                  pre_fence_visitor);
    // Java Heap Profiler check and sample allocation.
    JHPCheckNonTlabSampleAllocation(self, obj, num_bytes);
    return obj;
  }

  template <bool kInstrumented = true, bool kCheckLargeObject = true, typename PreFenceVisitor>
  ALWAYS_INLINE mirror::Object* AllocObjectWithAllocator(Thread* self,
                                                         ObjPtr<mirror::Class> klass,
                                                         size_t byte_count,
                                                         AllocatorType allocator,
                                                         const PreFenceVisitor& pre_fence_visitor)
      REQUIRES_SHARED(Locks::mutator_lock_)
      REQUIRES(!*gc_complete_lock_,
               !*pending_task_lock_,
               !*backtrace_lock_,
               !process_state_update_lock_,
               !Roles::uninterruptible_);

  AllocatorType GetCurrentAllocator() const {
    return current_allocator_;
  }

  AllocatorType GetCurrentNonMovingAllocator() const {
    return current_non_moving_allocator_;
  }

  AllocatorType GetUpdatedAllocator(AllocatorType old_allocator) {
    return (old_allocator == kAllocatorTypeNonMoving) ?
        GetCurrentNonMovingAllocator() : GetCurrentAllocator();
  }

  // Visit all of the live objects in the heap.
  template <typename Visitor>
  ALWAYS_INLINE void VisitObjects(Visitor&& visitor)
      REQUIRES_SHARED(Locks::mutator_lock_)
      REQUIRES(!Locks::heap_bitmap_lock_, !*gc_complete_lock_);
  template <typename Visitor>
  ALWAYS_INLINE void VisitObjectsPaused(Visitor&& visitor)
      REQUIRES(Locks::mutator_lock_, !Locks::heap_bitmap_lock_, !*gc_complete_lock_);

  void VisitReflectiveTargets(ReflectiveValueVisitor* visitor)
      REQUIRES(Locks::mutator_lock_, !Locks::heap_bitmap_lock_, !*gc_complete_lock_);

  void CheckPreconditionsForAllocObject(ObjPtr<mirror::Class> c, size_t byte_count)
      REQUIRES_SHARED(Locks::mutator_lock_);

  // Inform the garbage collector of a non-malloc allocated native memory that might become
  // reclaimable in the future as a result of Java garbage collection.
  void RegisterNativeAllocation(JNIEnv* env, size_t bytes)
      REQUIRES(!*gc_complete_lock_, !*pending_task_lock_, !process_state_update_lock_);
  void RegisterNativeFree(JNIEnv* env, size_t bytes);

  // Notify the garbage collector of malloc allocations that might be reclaimable
  // as a result of Java garbage collection. Each such call represents approximately
  // kNotifyNativeInterval such allocations.
  void NotifyNativeAllocations(JNIEnv* env)
      REQUIRES(!*gc_complete_lock_, !*pending_task_lock_, !process_state_update_lock_);

  uint32_t GetNotifyNativeInterval() {
    return kNotifyNativeInterval;
  }

  // Change the allocator, updates entrypoints.
  void ChangeAllocator(AllocatorType allocator)
      REQUIRES(Locks::mutator_lock_, !Locks::runtime_shutdown_lock_);

  // Change the collector to be one of the possible options (MS, CMS, SS). Only safe when no
  // concurrent accesses to the heap are possible.
  void ChangeCollector(CollectorType collector_type)
      REQUIRES(Locks::mutator_lock_, !*gc_complete_lock_);

  // The given reference is believed to be to an object in the Java heap, check the soundness of it.
  // TODO: NO_THREAD_SAFETY_ANALYSIS since we call this everywhere and it is impossible to find a
  // proper lock ordering for it.
  void VerifyObjectBody(ObjPtr<mirror::Object> o) NO_THREAD_SAFETY_ANALYSIS;

  // Consistency check of all live references.
  void VerifyHeap() REQUIRES(!Locks::heap_bitmap_lock_);
  // Returns how many failures occured.
  size_t VerifyHeapReferences(bool verify_referents = true)
      REQUIRES(Locks::mutator_lock_, !*gc_complete_lock_);
  bool VerifyMissingCardMarks()
      REQUIRES(Locks::heap_bitmap_lock_, Locks::mutator_lock_);

  // A weaker test than IsLiveObject or VerifyObject that doesn't require the heap lock,
  // and doesn't abort on error, allowing the caller to report more
  // meaningful diagnostics.
  bool IsValidObjectAddress(const void* obj) const REQUIRES_SHARED(Locks::mutator_lock_);

  // Faster alternative to IsHeapAddress since finding if an object is in the large object space is
  // very slow.
  bool IsNonDiscontinuousSpaceHeapAddress(const void* addr) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  // Returns true if 'obj' is a live heap object, false otherwise (including for invalid addresses).
  // Requires the heap lock to be held.
  bool IsLiveObjectLocked(ObjPtr<mirror::Object> obj,
                          bool search_allocation_stack = true,
                          bool search_live_stack = true,
                          bool sorted = false)
      REQUIRES_SHARED(Locks::heap_bitmap_lock_, Locks::mutator_lock_);

  // Returns true if there is any chance that the object (obj) will move.
  bool IsMovableObject(ObjPtr<mirror::Object> obj) const REQUIRES_SHARED(Locks::mutator_lock_);

  // Enables us to compacting GC until objects are released.
  void IncrementDisableMovingGC(Thread* self) REQUIRES(!*gc_complete_lock_);
  void DecrementDisableMovingGC(Thread* self) REQUIRES(!*gc_complete_lock_);

  // Temporarily disable thread flip for JNI critical calls.
  void IncrementDisableThreadFlip(Thread* self) REQUIRES(!*thread_flip_lock_);
  void DecrementDisableThreadFlip(Thread* self) REQUIRES(!*thread_flip_lock_);
  void ThreadFlipBegin(Thread* self) REQUIRES(!*thread_flip_lock_);
  void ThreadFlipEnd(Thread* self) REQUIRES(!*thread_flip_lock_);

  // Clear all of the mark bits, doesn't clear bitmaps which have the same live bits as mark bits.
  // Mutator lock is required for GetContinuousSpaces.
  void ClearMarkedObjects()
      REQUIRES(Locks::heap_bitmap_lock_)
      REQUIRES_SHARED(Locks::mutator_lock_);

  // Initiates an explicit garbage collection. Guarantees that a GC started after this call has
  // completed.
  void CollectGarbage(bool clear_soft_references, GcCause cause = kGcCauseExplicit)
      REQUIRES(!*gc_complete_lock_, !*pending_task_lock_, !process_state_update_lock_);

  // Does a concurrent GC, provided the GC numbered requested_gc_num has not already been
  // completed. Should only be called by the GC daemon thread through runtime.
  void ConcurrentGC(Thread* self, GcCause cause, bool force_full, uint32_t requested_gc_num)
      REQUIRES(!Locks::runtime_shutdown_lock_, !*gc_complete_lock_,
               !*pending_task_lock_, !process_state_update_lock_);

  // Implements VMDebug.countInstancesOfClass and JDWP VM_InstanceCount.
  // The boolean decides whether to use IsAssignableFrom or == when comparing classes.
  void CountInstances(const std::vector<Handle<mirror::Class>>& classes,
                      bool use_is_assignable_from,
                      uint64_t* counts)
      REQUIRES(!Locks::heap_bitmap_lock_, !*gc_complete_lock_)
      REQUIRES_SHARED(Locks::mutator_lock_);

  // Removes the growth limit on the alloc space so it may grow to its maximum capacity. Used to
  // implement dalvik.system.VMRuntime.clearGrowthLimit.
  void ClearGrowthLimit() REQUIRES(!*gc_complete_lock_);

  // Make the current growth limit the new maximum capacity, unmaps pages at the end of spaces
  // which will never be used. Used to implement dalvik.system.VMRuntime.clampGrowthLimit.
  void ClampGrowthLimit() REQUIRES(!Locks::heap_bitmap_lock_);

  // Target ideal heap utilization ratio, implements
  // dalvik.system.VMRuntime.getTargetHeapUtilization.
  double GetTargetHeapUtilization() const {
    return target_utilization_;
  }

  // Data structure memory usage tracking.
  void RegisterGCAllocation(size_t bytes);
  void RegisterGCDeAllocation(size_t bytes);

  // Set the heap's private space pointers to be the same as the space based on it's type. Public
  // due to usage by tests.
  void SetSpaceAsDefault(space::ContinuousSpace* continuous_space)
      REQUIRES(!Locks::heap_bitmap_lock_);
  void AddSpace(space::Space* space)
      REQUIRES(!Locks::heap_bitmap_lock_)
      REQUIRES(Locks::mutator_lock_);
  void RemoveSpace(space::Space* space)
    REQUIRES(!Locks::heap_bitmap_lock_)
    REQUIRES(Locks::mutator_lock_);

  double GetPreGcWeightedAllocatedBytes() const {
    return pre_gc_weighted_allocated_bytes_;
  }

  double GetPostGcWeightedAllocatedBytes() const {
    return post_gc_weighted_allocated_bytes_;
  }

  void CalculatePreGcWeightedAllocatedBytes();
  void CalculatePostGcWeightedAllocatedBytes();
  uint64_t GetTotalGcCpuTime();

  uint64_t GetProcessCpuStartTime() const {
    return process_cpu_start_time_ns_;
  }

  uint64_t GetPostGCLastProcessCpuTime() const {
    return post_gc_last_process_cpu_time_ns_;
  }

  // Set target ideal heap utilization ratio, implements
  // dalvik.system.VMRuntime.setTargetHeapUtilization.
  void SetTargetHeapUtilization(float target);

  // For the alloc space, sets the maximum number of bytes that the heap is allowed to allocate
  // from the system. Doesn't allow the space to exceed its growth limit.
  // Set while we hold gc_complete_lock or collector_type_running_ != kCollectorTypeNone.
  void SetIdealFootprint(size_t max_allowed_footprint);

  // Blocks the caller until the garbage collector becomes idle and returns the type of GC we
  // waited for. Only waits for running collections, ignoring a requested but unstarted GC. Only
  // heuristic, since a new GC may have started by the time we return.
  collector::GcType WaitForGcToComplete(GcCause cause, Thread* self) REQUIRES(!*gc_complete_lock_);

  // Update the heap's process state to a new value, may cause compaction to occur.
  void UpdateProcessState(ProcessState old_process_state, ProcessState new_process_state)
      REQUIRES(!*pending_task_lock_, !*gc_complete_lock_, !process_state_update_lock_);

  bool HaveContinuousSpaces() const NO_THREAD_SAFETY_ANALYSIS {
    // No lock since vector empty is thread safe.
    return !continuous_spaces_.empty();
  }

  const std::vector<space::ContinuousSpace*>& GetContinuousSpaces() const
      REQUIRES_SHARED(Locks::mutator_lock_) {
    return continuous_spaces_;
  }

  const std::vector<space::DiscontinuousSpace*>& GetDiscontinuousSpaces() const {
    return discontinuous_spaces_;
  }

  const collector::Iteration* GetCurrentGcIteration() const {
    return &current_gc_iteration_;
  }
  collector::Iteration* GetCurrentGcIteration() {
    return &current_gc_iteration_;
  }

  // Enable verification of object references when the runtime is sufficiently initialized.
  void EnableObjectValidation() {
    verify_object_mode_ = kVerifyObjectSupport;
    if (verify_object_mode_ > kVerifyObjectModeDisabled) {
      VerifyHeap();
    }
  }

  // Disable object reference verification for image writing.
  void DisableObjectValidation() {
    verify_object_mode_ = kVerifyObjectModeDisabled;
  }

  // Other checks may be performed if we know the heap should be in a healthy state.
  bool IsObjectValidationEnabled() const {
    return verify_object_mode_ > kVerifyObjectModeDisabled;
  }

  // Returns true if low memory mode is enabled.
  bool IsLowMemoryMode() const {
    return low_memory_mode_;
  }

  // Returns the heap growth multiplier, this affects how much we grow the heap after a GC.
  // Scales heap growth, min free, and max free.
  double HeapGrowthMultiplier() const;

  // Freed bytes can be negative in cases where we copy objects from a compacted space to a
  // free-list backed space.
  void RecordFree(uint64_t freed_objects, int64_t freed_bytes);

  // Record the bytes freed by thread-local buffer revoke.
  void RecordFreeRevoke();

  accounting::CardTable* GetCardTable() const {
    return card_table_.get();
  }

  accounting::ReadBarrierTable* GetReadBarrierTable() const {
    return rb_table_.get();
  }

  void AddFinalizerReference(Thread* self, ObjPtr<mirror::Object>* object);

  // Returns the number of bytes currently allocated.
  // The result should be treated as an approximation, if it is being concurrently updated.
  size_t GetBytesAllocated() const {
    return num_bytes_allocated_.load(std::memory_order_relaxed);
  }

  bool GetUseGenerationalCC() const {
    return use_generational_cc_;
  }

  // Returns the number of objects currently allocated.
  size_t GetObjectsAllocated() const
      REQUIRES(!Locks::heap_bitmap_lock_);

  // Returns the total number of objects allocated since the heap was created.
  uint64_t GetObjectsAllocatedEver() const;

  // Returns the total number of bytes allocated since the heap was created.
  uint64_t GetBytesAllocatedEver() const;

  // Returns the total number of objects freed since the heap was created.
  // With default memory order, this should be viewed only as a hint.
  uint64_t GetObjectsFreedEver(std::memory_order mo = std::memory_order_relaxed) const {
    return total_objects_freed_ever_.load(mo);
  }

  // Returns the total number of bytes freed since the heap was created.
  // With default memory order, this should be viewed only as a hint.
  uint64_t GetBytesFreedEver(std::memory_order mo = std::memory_order_relaxed) const {
    return total_bytes_freed_ever_.load(mo);
  }

  space::RegionSpace* GetRegionSpace() const {
    return region_space_;
  }

  // Implements java.lang.Runtime.maxMemory, returning the maximum amount of memory a program can
  // consume. For a regular VM this would relate to the -Xmx option and would return -1 if no Xmx
  // were specified. Android apps start with a growth limit (small heap size) which is
  // cleared/extended for large apps.
  size_t GetMaxMemory() const {
    // There are some race conditions in the allocation code that can cause bytes allocated to
    // become larger than growth_limit_ in rare cases.
    return std::max(GetBytesAllocated(), growth_limit_);
  }

  // Implements java.lang.Runtime.totalMemory, returning approximate amount of memory currently
  // consumed by an application.
  size_t GetTotalMemory() const;

  // Returns approximately how much free memory we have until the next GC happens.
  size_t GetFreeMemoryUntilGC() const {
    return UnsignedDifference(target_footprint_.load(std::memory_order_relaxed),
                              GetBytesAllocated());
  }

  // Returns approximately how much free memory we have until the next OOME happens.
  size_t GetFreeMemoryUntilOOME() const {
    return UnsignedDifference(growth_limit_, GetBytesAllocated());
  }

  // Returns how much free memory we have until we need to grow the heap to perform an allocation.
  // Similar to GetFreeMemoryUntilGC. Implements java.lang.Runtime.freeMemory.
  size_t GetFreeMemory() const {
    return UnsignedDifference(GetTotalMemory(),
                              num_bytes_allocated_.load(std::memory_order_relaxed));
  }

  // Get the space that corresponds to an object's address. Current implementation searches all
  // spaces in turn. If fail_ok is false then failing to find a space will cause an abort.
  // TODO: consider using faster data structure like binary tree.
  space::ContinuousSpace* FindContinuousSpaceFromObject(ObjPtr<mirror::Object>, bool fail_ok) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  space::ContinuousSpace* FindContinuousSpaceFromAddress(const mirror::Object* addr) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  space::DiscontinuousSpace* FindDiscontinuousSpaceFromObject(ObjPtr<mirror::Object>,
                                                              bool fail_ok) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  space::Space* FindSpaceFromObject(ObjPtr<mirror::Object> obj, bool fail_ok) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  space::Space* FindSpaceFromAddress(const void* ptr) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  std::string DumpSpaceNameFromAddress(const void* addr) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  void DumpForSigQuit(std::ostream& os) REQUIRES(!*gc_complete_lock_);

  // Do a pending collector transition.
  void DoPendingCollectorTransition()
      REQUIRES(!*gc_complete_lock_, !*pending_task_lock_, !process_state_update_lock_);

  // Deflate monitors, ... and trim the spaces.
  void Trim(Thread* self) REQUIRES(!*gc_complete_lock_);

  void RevokeThreadLocalBuffers(Thread* thread);
  void RevokeRosAllocThreadLocalBuffers(Thread* thread);
  void RevokeAllThreadLocalBuffers();
  void AssertThreadLocalBuffersAreRevoked(Thread* thread);
  void AssertAllBumpPointerSpaceThreadLocalBuffersAreRevoked();
  void RosAllocVerification(TimingLogger* timings, const char* name)
      REQUIRES(Locks::mutator_lock_);

  accounting::HeapBitmap* GetLiveBitmap() REQUIRES_SHARED(Locks::heap_bitmap_lock_) {
    return live_bitmap_.get();
  }

  accounting::HeapBitmap* GetMarkBitmap() REQUIRES_SHARED(Locks::heap_bitmap_lock_) {
    return mark_bitmap_.get();
  }

  accounting::ObjectStack* GetLiveStack() REQUIRES_SHARED(Locks::heap_bitmap_lock_) {
    return live_stack_.get();
  }

  void PreZygoteFork() NO_THREAD_SAFETY_ANALYSIS;

  // Mark and empty stack.
  void FlushAllocStack()
      REQUIRES_SHARED(Locks::mutator_lock_)
      REQUIRES(Locks::heap_bitmap_lock_);

  // Revoke all the thread-local allocation stacks.
  void RevokeAllThreadLocalAllocationStacks(Thread* self)
      REQUIRES(Locks::mutator_lock_, !Locks::runtime_shutdown_lock_, !Locks::thread_list_lock_);

  // Mark all the objects in the allocation stack in the specified bitmap.
  // TODO: Refactor?
  void MarkAllocStack(accounting::SpaceBitmap<kObjectAlignment>* bitmap1,
                      accounting::SpaceBitmap<kObjectAlignment>* bitmap2,
                      accounting::SpaceBitmap<kLargeObjectAlignment>* large_objects,
                      accounting::ObjectStack* stack)
      REQUIRES_SHARED(Locks::mutator_lock_)
      REQUIRES(Locks::heap_bitmap_lock_);

  // Mark the specified allocation stack as live.
  void MarkAllocStackAsLive(accounting::ObjectStack* stack)
      REQUIRES_SHARED(Locks::mutator_lock_)
      REQUIRES(Locks::heap_bitmap_lock_);

  // Unbind any bound bitmaps.
  void UnBindBitmaps()
      REQUIRES(Locks::heap_bitmap_lock_)
      REQUIRES_SHARED(Locks::mutator_lock_);

  // Returns the boot image spaces. There may be multiple boot image spaces.
  const std::vector<space::ImageSpace*>& GetBootImageSpaces() const {
    return boot_image_spaces_;
  }

  bool ObjectIsInBootImageSpace(ObjPtr<mirror::Object> obj) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  bool IsInBootImageOatFile(const void* p) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  // Get the start address of the boot images if any; otherwise returns 0.
  uint32_t GetBootImagesStartAddress() const {
    return boot_images_start_address_;
  }

  // Get the size of all boot images, including the heap and oat areas.
  uint32_t GetBootImagesSize() const {
    return boot_images_size_;
  }

  // Check if a pointer points to a boot image.
  bool IsBootImageAddress(const void* p) const {
    return reinterpret_cast<uintptr_t>(p) - boot_images_start_address_ < boot_images_size_;
  }

  space::DlMallocSpace* GetDlMallocSpace() const {
    return dlmalloc_space_;
  }

  space::RosAllocSpace* GetRosAllocSpace() const {
    return rosalloc_space_;
  }

  // Return the corresponding rosalloc space.
  space::RosAllocSpace* GetRosAllocSpace(gc::allocator::RosAlloc* rosalloc) const
      REQUIRES_SHARED(Locks::mutator_lock_);

  space::MallocSpace* GetNonMovingSpace() const {
    return non_moving_space_;
  }

  space::LargeObjectSpace* GetLargeObjectsSpace() const {
    return large_object_space_;
  }

  // Returns the free list space that may contain movable objects (the
  // one that's not the non-moving space), either rosalloc_space_ or
  // dlmalloc_space_.
  space::MallocSpace* GetPrimaryFreeListSpace() {
    if (kUseRosAlloc) {
      DCHECK(rosalloc_space_ != nullptr);
      // reinterpret_cast is necessary as the space class hierarchy
      // isn't known (#included) yet here.
      return reinterpret_cast<space::MallocSpace*>(rosalloc_space_);
    } else {
      DCHECK(dlmalloc_space_ != nullptr);
      return reinterpret_cast<space::MallocSpace*>(dlmalloc_space_);
    }
  }

  void DumpSpaces(std::ostream& stream) const REQUIRES_SHARED(Locks::mutator_lock_);
  std::string DumpSpaces() const REQUIRES_SHARED(Locks::mutator_lock_);

  // GC performance measuring
  void DumpGcPerformanceInfo(std::ostream& os)
      REQUIRES(!*gc_complete_lock_);
  void ResetGcPerformanceInfo() REQUIRES(!*gc_complete_lock_);

  // Thread pool.
  void CreateThreadPool();
  void DeleteThreadPool();
  ThreadPool* GetThreadPool() {
    return thread_pool_.get();
  }
  size_t GetParallelGCThreadCount() const {
    return parallel_gc_threads_;
  }
  size_t GetConcGCThreadCount() const {
    return conc_gc_threads_;
  }
  accounting::ModUnionTable* FindModUnionTableFromSpace(space::Space* space);
  void AddModUnionTable(accounting::ModUnionTable* mod_union_table);

  accounting::RememberedSet* FindRememberedSetFromSpace(space::Space* space);
  void AddRememberedSet(accounting::RememberedSet* remembered_set);
  // Also deletes the remebered set.
  void RemoveRememberedSet(space::Space* space);

  bool IsCompilingBoot() const;
  bool HasBootImageSpace() const {
    return !boot_image_spaces_.empty();
  }

  ReferenceProcessor* GetReferenceProcessor() {
    return reference_processor_.get();
  }
  TaskProcessor* GetTaskProcessor() {
    return task_processor_.get();
  }

  bool HasZygoteSpace() const {
    return zygote_space_ != nullptr;
  }

  // Returns the active concurrent copying collector.
  collector::ConcurrentCopying* ConcurrentCopyingCollector() {
    collector::ConcurrentCopying* active_collector =
            active_concurrent_copying_collector_.load(std::memory_order_relaxed);
    if (use_generational_cc_) {
      DCHECK((active_collector == concurrent_copying_collector_) ||
             (active_collector == young_concurrent_copying_collector_))
              << "active_concurrent_copying_collector: " << active_collector
              << " young_concurrent_copying_collector: " << young_concurrent_copying_collector_
              << " concurrent_copying_collector: " << concurrent_copying_collector_;
    } else {
      DCHECK_EQ(active_collector, concurrent_copying_collector_);
    }
    return active_collector;
  }

  CollectorType CurrentCollectorType() {
    return collector_type_;
  }

  bool IsGcConcurrentAndMoving() const {
    if (IsGcConcurrent() && IsMovingGc(collector_type_)) {
      // Assume no transition when a concurrent moving collector is used.
      DCHECK_EQ(collector_type_, foreground_collector_type_);
      return true;
    }
    return false;
  }

  bool IsMovingGCDisabled(Thread* self) REQUIRES(!*gc_complete_lock_) {
    MutexLock mu(self, *gc_complete_lock_);
