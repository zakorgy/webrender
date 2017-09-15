/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared,clip_shared

#ifdef WR_DX11
    struct v2p {
        vec4 gl_Position : SV_Position;
        vec3 vPos : vPos;

        flat vec2 vClipCenter : vClipCenter;

        flat vec4 vPoint_Tangent0 : vPoint_Tangent0;
        flat vec4 vPoint_Tangent1 : vPoint_Tangent1;
        flat vec3 vDotParams : vDotParams;
        flat vec2 vAlphaMask : vAlphaMask;
    };
#else
varying vec3 vPos;

flat varying vec2 vClipCenter;

flat varying vec4 vPoint_Tangent0;
flat varying vec4 vPoint_Tangent1;
flat varying vec3 vDotParams;
flat varying vec2 vAlphaMask;
#endif //WR_DX11
