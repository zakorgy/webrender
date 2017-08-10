/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        vec3 vUv0 : vUv0;
        vec3 vUv1 : vUv1;
        flat int vOp : vOp;
    };
#else
varying vec3 vUv0;
varying vec3 vUv1;
flat varying int vOp;
#endif //WR_DX11
