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

#ifndef ART_LIBELFFILE_STREAM_VECTOR_OUTPUT_STREAM_H_
#define ART_LIBELFFILE_STREAM_VECTOR_OUTPUT_STREAM_H_

#include "output_stream.h"

#include <string.h>
#include <string>
#include <vector>

namespace art {

class VectorOutputStream final : public OutputStream {
 public:
  VectorOutputStream(const std::string& location, std::vector<uint8_t>* vector);

  ~VectorOutputStream() override {}

  bool WriteFully(const void* buffer, size_t byte_count) override {
    if (static_cast<size_t>(offset_) == vector_->size()) {
      const uint8_t* start = reinterpret_cast<const uint8_t*>(buffer);
      vector_->insert(vector_->end(), &start[0], &start[byte_count]);
      offset_ += byte_count;
    } else {
      off_t new_offset = offset_ + byte_count;
      EnsureCapacity(new_offset);
      memcpy(&(*vector_)[offset_], buffer, byte_count);
      offset_ = new_offset;
    }
    return true;
  }

  off_t Seek(off_t offset, Whence whence) override;

  bool Flush() override {
    return true;
  }

 private:
  void EnsureCapacity(off_t new_offset) {
    if (new_offset > static_cast<off_t>(vector_->size())) {
      vector_->resize(new_offset);
    }
  }

  off_t offset_;
  std::vector<uint8_t>* const vector_;

  DISALLOW_COPY_AND_ASSIGN(VectorOutputStream);
};

}  // namespace art

#endif  // ART_LIBELFFILE_STREAM_VECTOR_OUTPUT_STREAM_H_