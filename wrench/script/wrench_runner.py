#!/usr/bin/python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from subprocess import call

for x in range(20):
    call(["cargo", "run", "--features", "vulkan", "--release", "reftest"])