//#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

 #ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : vClipMaskUvBounds;
        vec3 vClipMaskUv : vClipMaskUv;

        // Edge color transition
        flat vec4 vColor00 : vColor00;
        flat vec4 vColor01 : vColor01;
        flat vec4 vColor10 : vColor10;
        flat vec4 vColor11 : vColor11;
        flat vec4 vColorEdgeLine : vColorEdgeLine;

        // Border radius
        flat vec2 vClipCenter : vClipCenter;
        flat vec4 vRadii0 : vRadii0;
        flat vec4 vRadii1 : vRadii1;
        flat vec2 vClipSign : vClipSign;
        flat vec4 vEdgeDistance : vEdgeDistance;
        flat float vSDFSelect : vSDFSelect;

        // Border style
        flat float vAlphaSelect : vAlphaSelect;
#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : vLocalPos;
        flat vec4 vLocalBounds : vLocalBounds;
#else
        vec2 vLocalPos : vLocalPos;
#endif //WR_FEATURE_TRANSFORM
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
#endif //WR_FEATURE_TRANSFORM
#endif //WR_DX11
