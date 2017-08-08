/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : POSITION0;
        vec3 vClipMaskUv : POSITION1;
        flat vec4 vColor : COLOR0;
        vec2 vUv: POSITION2;
        flat vec4 vUvBorder: POSITION3;
    #ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : POSITION4;
        flat vec4 vLocalBounds : POSITION5;
    #endif
    };
#else

flat varying vec4 vColor;
varying vec2 vUv;
flat varying vec4 vUvBorder;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
#endif
#endif
