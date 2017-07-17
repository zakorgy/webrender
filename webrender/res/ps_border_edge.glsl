/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : POSITION0;
        vec3 vClipMaskUv : POSITION1;
        flat vec4 vColor0 : COLOR0;
        flat vec4 vColor1 : COLOR1;
        flat vec2 vEdgeDistance : POSITION2;
        flat float vAxisSelect : PSIZE0;
        flat float vAlphaSelect : PSIZE1;
        flat vec4 vClipParams : POSITION3;
        flat float vClipSelect : PSIZE2;
    #ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : POSITION4;
    #else
        vec2 vLocalPos : POSITION4;
    #endif
    };
#else

flat varying vec4 vColor0;
flat varying vec4 vColor1;
flat varying vec2 vEdgeDistance;
flat varying float vAxisSelect;
flat varying float vAlphaSelect;
flat varying vec4 vClipParams;
flat varying float vClipSelect;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
#else
varying vec2 vLocalPos;
#endif
#endif
