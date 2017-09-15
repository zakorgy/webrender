/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 gl_Position : SV_Position;
        vec3 vUv : vUv;
        flat vec4 vUvBounds : vUvBounds;
        flat float vAmount : vAmount;
        flat int vOp : vOp;
    };
#else
varying vec3 vUv;
flat varying vec4 vUvBounds;
flat varying float vAmount;
flat varying int vOp;
#endif //WR_DX11
