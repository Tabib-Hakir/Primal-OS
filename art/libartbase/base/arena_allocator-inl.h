/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ART_LIBARTBASE_BASE_ARENA_ALLOCATOR_INL_H_
#define ART_LIBARTBASE_BASE_ARENA_ALLOCATOR_INL_H_

#include "arena_allocator.h"

namespace art {
namespace arena_allocator {

static constexpr bool kArenaAllocatorPreciseTracking = kArenaAllocatorCountAllocations;

static constexpr size_t kArenaDefaultSize = kArenaAllocatorPreciseTracking
                                                ? 32
                                                : 128 * KB;

}  // namespace arena_allocator
}  // namespace art

#endif  // ART_LIBARTBASE_BASE_ARENA_ALLOCATOR_INL_H_