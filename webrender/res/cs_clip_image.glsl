//#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        vec3 vPos : vPos;
        flat vec4 vLocalBounds : vLocalBounds;
        vec3 vClipMaskUv : vClipMaskUv;
        flat vec4 vClipMaskUvRect : vClipMaskUvRect;
        flat vec4 vClipMaskUvInnerRect : vClipMaskUvInnerRect;
    };
#else
varying vec3 vPos;
flat varying vec4 vClipMaskUvRect;
flat varying vec4 vClipMaskUvInnerRect;
#endif //WR_DX11
