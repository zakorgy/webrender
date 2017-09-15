/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : vClipMaskUvBounds;
        vec3 vClipMaskUv : vClipMaskUv;
        flat vec4 vColor : vColor;

        vec3 vUv : vUv;
        flat vec2 vMirrorPoint : vMirrorPoint;
        flat vec4 vCacheUvRectCoords : vCacheUvRectCoords;
    };
#else
flat varying vec4 vColor;

varying vec3 vUv;
flat varying vec2 vMirrorPoint;
flat varying vec4 vCacheUvRectCoords;
#endif //WR_DX11

