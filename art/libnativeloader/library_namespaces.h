/*
 * Copyright (C) 2019 The Android Open Source Project
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

#ifndef ART_LIBNATIVELOADER_LIBRARY_NAMESPACES_H_
#define ART_LIBNATIVELOADER_LIBRARY_NAMESPACES_H_

#if !defined(ART_TARGET_ANDROID)
#error "Not available for host or linux target"
#endif

#define LOG_TAG "nativeloader"

#include "native_loader_namespace.h"

#include <list>
#include <string>

#include <android-base/result.h>
#include <jni.h>

namespace android::nativeloader {

using android::base::Result;

// LibraryNamespaces is a singleton object that manages NativeLoaderNamespace
// objects for an app process. Its main job is to create (and configure) a new
// NativeLoaderNamespace object for a Java ClassLoader, and to find an existing
// object for a given ClassLoader.
class LibraryNamespaces {
 public:
  LibraryNamespaces() : initialized_(false), app_main_namespace_(nullptr) {}

  LibraryNamespaces(LibraryNamespaces&&) = default;
  LibraryNamespaces(const LibraryNamespaces&) = delete;
  LibraryNamespaces& operator=(const LibraryNamespaces&) = delete;

  void Initialize();
  void Reset() {
    namespaces_.clear();
    initialized_ = false;
    app_main_namespace_ = nullptr;
  }
  Result<NativeLoaderNamespace*> Create(JNIEnv* env, uint32_t target_sdk_version,
                                        jobject class_loader, bool is_shared, jstring dex_path,
                                        jstring java_library_path, jstring java_permitted_path,
                                        jstring uses_library_list);
  NativeLoaderNamespace* FindNamespaceByClassLoader(JNIEnv* env, jobject class_loader);

 private:
  Result<void> InitPublicNamespace(const char* library_path);
  NativeLoaderNamespace* FindParentNamespaceByClassLoader(JNIEnv* env, jobject class_loader);

  bool initialized_;
  NativeLoaderNamespace* app_main_namespace_;
  std::list<std::pair<jweak, NativeLoaderNamespace>> namespaces_;
};

Result<std::string> FindApexNamespaceName(const std::string& location);

}  // namespace android::nativeloader

#endif  // ART_LIBNATIVELOADER_LIBRARY_NAMESPACES_H_