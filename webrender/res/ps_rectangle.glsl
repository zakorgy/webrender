/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
#ifdef WR_FEATURE_CLIP
        flat vec4 vClipMaskUvBounds : vClipMaskUvBounds;
        vec3 vClipMaskUv : vClipMaskUv;
#endif //WR_FEATURE_CLIP
        vec4 vColor : vColor;
#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : vLocalPos;
        flat vec4 vLocalBounds : vLocalBounds;
#endif //WR_FEATURE_TRANSFORM
    };
#else

    varying vec4 vColor;
#ifdef WR_FEATURE_TRANSFORM
    varying vec3 vLocalPos;
#endif //WR_FEATURE_TRANSFORM

#endif //WR_DX11
