//#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        vec3 vPos : vPos;
        flat vec4 vLocalBounds : vLocalBounds;
        flat float vClipMode : vClipMode;
        flat vec4 vClipCenter_Radius_TL : vClipCenter_Radius_TL;
        flat vec4 vClipCenter_Radius_TR : vClipCenter_Radius_TR;
        flat vec4 vClipCenter_Radius_BL : vClipCenter_Radius_BL;
        flat vec4 vClipCenter_Radius_BR : vClipCenter_Radius_BR;
    };
#else
varying vec3 vPos;
flat varying float vClipMode;
flat varying vec4 vClipCenter_Radius_TL;
flat varying vec4 vClipCenter_Radius_TR;
flat varying vec4 vClipCenter_Radius_BL;
flat varying vec4 vClipCenter_Radius_BR;
#endif //WR_DX11
