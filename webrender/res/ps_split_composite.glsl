//#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        vec3 vUv : vUv;
        flat vec4 vUvTaskBounds : vUvTaskBounds;
        flat vec4 vUvSampleBounds : vUvSampleBounds;
    };
#else
varying vec3 vUv;
flat varying vec4 vUvTaskBounds;
flat varying vec4 vUvSampleBounds;
#endif //WR_DX11
