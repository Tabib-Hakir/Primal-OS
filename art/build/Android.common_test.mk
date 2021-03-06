#
# Copyright (C) 2011 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ifndef ART_ANDROID_COMMON_TEST_MK
ART_ANDROID_COMMON_TEST_MK = true

include art/build/Android.common_path.mk

# Directory used for temporary test files on the host.
# TMPDIR is always provided by the build system as $OUT_DIR-unique temporary directory.
ART_HOST_TEST_DIR := $(TMPDIR)/test-art

# List of known broken tests that we won't attempt to execute. The test name must be the full
# rule name such as test-art-host-oat-optimizing-HelloWorld64.
ART_TEST_KNOWN_BROKEN :=

# List of known failing tests that when executed won't cause test execution to not finish.
# The test name must be the full rule name such as test-art-host-oat-optimizing-HelloWorld64.
ART_TEST_KNOWN_FAILING :=

# Keep going after encountering a test failure?
ART_TEST_KEEP_GOING ?= true

# Do you want run-test to be quieter? run-tests will only show output if they fail.
ART_TEST_QUIET ?= true

# Define the command run on test failure. $(1) is the name of the test. Executed by the shell.
# If the test was a top-level make target (e.g. `test-art-host-gtest-codegen_test64`), the command
# fails with exit status 1 (returned by the last `grep` statement below).
# Otherwise (e.g., if the test was run as a prerequisite of a compound test command, such as
# `test-art-host-gtest-codegen_test`), the command does not fail, as this would break rules running
# ART_TEST_PREREQ_FINISHED as one of their actions, which expects *all* prerequisites *not* to fail.
define ART_TEST_FAILED
  ( [ -f $(ART_HOST_TEST_DIR)/skipped/$(1) ] || \
    (mkdir -p $(ART_HOST_TEST_DIR)/failed/ && touch $(ART_HOST_TEST_DIR)/failed/$(1) && \
      echo $(ART_TEST_KNOWN_FAILING) | grep -q $(1) \
        && (echo -e "$(1) \e[91mKNOWN FAILURE\e[0m") \
        || (echo -e "$(1) \e[91mFAILED\e[0m" >&2; echo $(MAKECMDGOALS) | grep -q -v $(1))))
endef

ifeq ($(ART_TEST_QUIET),true)
  ART_TEST_ANNOUNCE_PASS := ( true )
  ART_TEST_ANNOUNCE_RUN := ( true )
  ART_TEST_ANNOUNCE_SKIP_FAILURE := ( true )
  ART_TEST_ANNOUNCE_SKIP_BROKEN := ( true )
else
  # Note the use of '=' and not ':=' is intentional since these are actually functions.
  ART_TEST_ANNOUNCE_PASS = ( echo -e "$(1) \e[92mPASSED\e[0m" )
  ART_TEST_ANNOUNCE_RUN = ( echo -e "$(1) \e[95mRUNNING\e[0m")
  ART_TEST_ANNOUNCE_SKIP_FAILURE = ( echo -e "$(1) \e[93mSKIPPING DUE TO EARLIER FAILURE\e[0m" )
  ART_TEST_ANNOUNCE_SKIP_BROKEN = ( echo -e "$(1) \e[93mSKIPPING BROKEN TEST\e[0m" )
endif

# Define the command run on test success. $(1) is the name of the test. Executed by the shell.
# The command checks prints "PASSED" then checks to see if this was a top-level make target (e.g.
# "mm test-art-host-oat-HelloWorld32"), if it was then it does nothing, otherwise it creates a file
# to be printed in the passing test summary.
define ART_TEST_PASSED
  ( $(call ART_TEST_ANNOUNCE_PASS,$(1)) && \
    (echo $(MAKECMDGOALS) | grep -q $(1) || \
      (mkdir -p $(ART_HOST_TEST_DIR)/passed/ && touch $(ART_HOST_TEST_DIR)/passed/$(1))))
endef

# Define the command run on test success of multiple prerequisites. $(1) is the name of the test.
# When the test is a top-level make target then a summary of the ran tests is produced. Executed by
# the shell.
define ART_TEST_PREREQ_FINISHED
  (echo -e "$(1) \e[32mCOMPLETE\e[0m" && \
    (echo $(MAKECMDGOALS) | grep -q -v $(1) || \
      (([ -d $(ART_HOST_TEST_DIR)/passed/ ] \
        && (echo -e "\e[92mPASSING TESTS\e[0m" && ls -1 $(ART_HOST_TEST_DIR)/passed/) \
        || (echo -e "\e[91mNO TESTS PASSED\e[0m")) && \
      ([ -d $(ART_HOST_TEST_DIR)/skipped/ ] \
        && (echo -e "\e[93mSKIPPED TESTS\e[0m" && ls -1 $(ART_HOST_TEST_DIR)/skipped/) \
        || (echo -e "\e[92mNO TESTS SKIPPED\e[0m")) && \
      ([ -d $(ART_HOST_TEST_DIR)/failed/ ] \
        && (echo -e "\e[91mFAILING TESTS\e[0m" >&2 && ls -1 $(ART_HOST_TEST_DIR)/failed/ >&2) \
        || (echo -e "\e[92mNO TESTS FAILED\e[0m")) \
      && ([ ! -d $(ART_HOST_TEST_DIR)/failed/ ] && rm -r $(ART_HOST_TEST_DIR) \
          || (rm -r $(ART_HOST_TEST_DIR) && false)))))
endef

# Define the command executed by the shell ahead of running an art test. $(1) is the name of the
# test.
define ART_TEST_SKIP
  ((echo $(ART_TEST_KNOWN_BROKEN) | grep -q -v $(1) \
     && ([ ! -d $(ART_HOST_TEST_DIR)/failed/ ] || [ $(ART_TEST_KEEP_GOING) = true ])\
     && $(call ART_TEST_ANNOUNCE_RUN,$(1)) ) \
   || ((mkdir -p $(ART_HOST_TEST_DIR)/skipped/ && touch $(ART_HOST_TEST_DIR)/skipped/$(1) \
     && ([ -d $(ART_HOST_TEST_DIR)/failed/ ] \
       && $(call ART_TEST_ANNOUNCE_SKIP_FAILURE,$(1)) ) \
     || $(call ART_TEST_ANNOUNCE_SKIP_BROKEN,$(1)) ) && false))
endef

endif # ART_ANDROID_COMMON_TEST_MK