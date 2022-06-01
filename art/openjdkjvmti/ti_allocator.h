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

#ifndef ART_OPENJDKJVMTI_TI_ALLOCATOR_H_
#define ART_OPENJDKJVMTI_TI_ALLOCATOR_H_

#include "jni.h"
#include "jvmti.h"

#include <atomic>
#include <memory>

namespace openjdkjvmti {

template<typename T>
class JvmtiAllocator;

class AllocUtil {
 public:
  static jvmtiError Allocate(jvmtiEnv* env, jlong size, unsigned char** mem_ptr);
  static jvmtiError Deallocate(jvmtiEnv* env, unsigned char* mem);
  static jvmtiError GetGlobalJvmtiAllocationState(jvmtiEnv* env, jlong* total_allocated);

 private:
  static void DeallocateImpl(unsigned char* mem);
  static unsigned char* AllocateImpl(jlong size);

  static std::atomic<jlong> allocated;

  template <typename T>
  friend class JvmtiAllocator;
};

}  // namespace openjdkjvmti

#endif  // ART_OPENJDKJVMTI_TI_ALLOCATOR_H_