//#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

 #ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : POSITION0;
        vec3 vClipMaskUv : POSITION1;

        // Edge color transition
        flat vec4 vColor00 : COLOR0;
        flat vec4 vColor01 : COLOR1;
        flat vec4 vColor10 : COLOR2;
        flat vec4 vColor11 : COLOR3;
        flat vec4 vColorEdgeLine : POSITION2;

        // Border radius
        flat vec2 vClipCenter : POSITION3;
        flat vec4 vRadii0 : POSITION4;
        flat vec4 vRadii1 : POSITION5;
        flat vec2 vClipSign : POSITION6;
        flat vec4 vEdgeDistance : POSITION7;
        flat float vSDFSelect : PSIZE0;

        // Border style
        flat float vAlphaSelect : PSIZE1;
#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : POSITION8;
#else
        vec2 vLocalPos : POSITION8;
#endif
     };
#else

// Edge color transition
flat varying vec4 vColor00;
flat varying vec4 vColor01;
flat varying vec4 vColor10;
flat varying vec4 vColor11;
flat varying vec4 vColorEdgeLine;

// Border radius
flat varying vec2 vClipCenter;
flat varying vec4 vRadii0;
flat varying vec4 vRadii1;
flat varying vec2 vClipSign;
flat varying vec4 vEdgeDistance;
flat varying float vSDFSelect;

// Border style
flat varying float vAlphaSelect;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
#else
varying vec2 vLocalPos;
#endif
#endif
