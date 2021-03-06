/*
 * Copyright (C) 2014 The Android Open Source Project
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

#ifndef ART_LIBARTBASE_BASE_ARENA_OBJECT_H_
#define ART_LIBARTBASE_BASE_ARENA_OBJECT_H_

#include <android-base/logging.h>

#include "arena_allocator.h"
#include "macros.h"
#include "scoped_arena_allocator.h"

namespace art {

// Parent for arena allocated objects giving appropriate new and delete operators.
template<enum ArenaAllocKind kAllocKind>
class ArenaObject {
 public:
  // Allocate a new ArenaObject of 'size' bytes in the Arena.
  void* operator new(size_t size, ArenaAllocator* allocator) {
    return allocator->Alloc(size, kAllocKind);
  }

  static void* operator new(size_t size, ScopedArenaAllocator* allocator) {
    return allocator->Alloc(size, kAllocKind);
  }

  void operator delete(void*, size_t) {
    LOG(FATAL) << "UNREACHABLE";
    UNREACHABLE();
  }

  // NOTE: Providing placement new (and matching delete) for constructing container elements.
  ALWAYS_INLINE void* operator new(size_t, void* ptr) noexcept { return ptr; }
  ALWAYS_INLINE void operator delete(void*, void*) noexcept { }
};


// Parent for arena allocated objects that get deleted, gives appropriate new and delete operators.
template<enum ArenaAllocKind kAllocKind>
class DeletableArenaObject {
 public:
  // Allocate a new ArenaObject of 'size' bytes in the Arena.
  void* operator new(size_t size, ArenaAllocator* allocator) {
    return allocator->Alloc(size, kAllocKind);
  }

  static void* operator new(size_t size, ScopedArenaAllocator* allocator) {
    return allocator->Alloc(size, kAllocKind);
  }

  void operator delete(void*, size_t) {
    // Nop.
  }
};

}  // namespace art

#endif  // ART_LIBARTBASE_BASE_ARENA_OBJECT_H_