/* Copyright (C) 2016 The Android Open Source Project
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This file implements interfaces from the file jvmti.h. This implementation
 * is licensed under the same terms as the file jvmti.h.  The
 * copyright and license information for the file jvmti.h follows.
 *
 * Copyright (c) 2003, 2011, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

#ifndef ART_OPENJDKJVMTI_OBJECT_TAGGING_H_
#define ART_OPENJDKJVMTI_OBJECT_TAGGING_H_

#include <unordered_map>

#include "base/globals.h"
#include "base/mutex.h"
#include "jvmti.h"
#include "jvmti_weak_table.h"
#include "mirror/object.h"

namespace openjdkjvmti {

struct ArtJvmTiEnv;
class EventHandler;

class ObjectTagTable final : public JvmtiWeakTable<jlong> {
 public:
  ObjectTagTable(EventHandler* event_handler, ArtJvmTiEnv* env)
      : lock_("Object tag table lock", art::LockLevel::kGenericBottomLock),
        event_handler_(event_handler),
        jvmti_env_(env) {}

  // Denotes that weak-refs are visible on all threads. Used by semi-space.
  void Allow() override
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(!allow_disallow_lock_);
  // Used by cms and the checkpoint system.
  void Broadcast(bool broadcast_for_checkpoint) override
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(!allow_disallow_lock_);

  bool Set(art::ObjPtr<art::mirror::Object> obj, jlong tag) override
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(!allow_disallow_lock_);
  bool SetLocked(art::ObjPtr<art::mirror::Object> obj, jlong tag) override
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(allow_disallow_lock_);

  jlong GetTagOrZero(art::ObjPtr<art::mirror::Object> obj)
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(!allow_disallow_lock_) {
    jlong tmp = 0;
    GetTag(obj, &tmp);
    return tmp;
  }
  jlong GetTagOrZeroLocked(art::ObjPtr<art::mirror::Object> obj)
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(allow_disallow_lock_) {
    jlong tmp = 0;
    GetTagLocked(obj, &tmp);
    return tmp;
  }

 protected:
  bool DoesHandleNullOnSweep() override;
  void HandleNullSweep(jlong tag) override;

 private:
  void SendDelayedFreeEvents()
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(!allow_disallow_lock_);

  void SendSingleFreeEvent(jlong tag)
      REQUIRES_SHARED(art::Locks::mutator_lock_)
      REQUIRES(!allow_disallow_lock_, !lock_);

  art::Mutex lock_ BOTTOM_MUTEX_ACQUIRED_AFTER;
  std::vector<jlong> null_tags_ GUARDED_BY(lock_);
  EventHandler* event_handler_;
  ArtJvmTiEnv* jvmti_env_;
};

}  // namespace openjdkjvmti

#endif  // ART_OPENJDKJVMTI_OBJECT_TAGGING_H_