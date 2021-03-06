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

#ifndef ART_LIBELFFILE_STREAM_BUFFERED_OUTPUT_STREAM_H_
#define ART_LIBELFFILE_STREAM_BUFFERED_OUTPUT_STREAM_H_

#include <memory>

#include "output_stream.h"

#include "base/globals.h"

namespace art {

class BufferedOutputStream final : public OutputStream {
 public:
  explicit BufferedOutputStream(std::unique_ptr<OutputStream> out);

  ~BufferedOutputStream() override;

  bool WriteFully(const void* buffer, size_t byte_count) override;

  off_t Seek(off_t offset, Whence whence) override;

  bool Flush() override;

 private:
  static const size_t kBufferSize = 8 * KB;

  bool FlushBuffer();

  std::unique_ptr<OutputStream> const out_;
  uint8_t buffer_[kBufferSize];
  size_t used_;

  DISALLOW_COPY_AND_ASSIGN(BufferedOutputStream);
};

}  // namespace art

#endif  // ART_LIBELFFILE_STREAM_BUFFERED_OUTPUT_STREAM_H_