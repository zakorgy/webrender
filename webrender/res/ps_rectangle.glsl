/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : POSITION0;
        vec3 vClipMaskUv : POSITION1;
        vec4 vColor : COLOR;
    #ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : POSITION2;
        flat vec4 vLocalBounds : POSITION3;
    #endif
    };
#else

    varying vec4 vColor;
#ifdef WR_FEATURE_TRANSFORM
    varying vec3 vLocalPos;
#endif

#endif
