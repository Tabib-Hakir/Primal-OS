/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef ART_LIBELFFILE_ELF_XZ_UTILS_H_
#define ART_LIBELFFILE_ELF_XZ_UTILS_H_

#include <vector>

#include "base/array_ref.h"
#include "base/bit_utils.h"

namespace art {

constexpr size_t kXzDefaultBlockSize = 16 * KB;

void XzCompress(ArrayRef<const uint8_t> src,
                std::vector<uint8_t>* dst,
                int level = 1 /* speed */,
                size_t block_size = kXzDefaultBlockSize);

void XzDecompress(ArrayRef<const uint8_t> src, std::vector<uint8_t>* dst);

}  // namespace art

#endif  // ART_LIBELFFILE_ELF_XZ_UTILS_H_