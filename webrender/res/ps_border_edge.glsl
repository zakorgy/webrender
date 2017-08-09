/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : vClipMaskUvBounds;
        vec3 vClipMaskUv : vClipMaskUv;
        flat vec4 vColor0 : vColor0;
        flat vec4 vColor1 : vColor1;
        flat vec2 vEdgeDistance : vEdgeDistance;
        flat float vAxisSelect : vAxisSelect;
        flat float vAlphaSelect : vAlphaSelect;
        flat vec4 vClipParams : vClipParams;
        flat float vClipSelect : vClipSelect;
#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : vLocalPos;
        flat vec4 vLocalBounds : vLocalBounds;
#else
        vec2 vLocalPos : vLocalPos;
#endif //WR_FEATURE_TRANSFORM
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
#endif //WR_FEATURE_TRANSFORM
#endif //WR_DX11
