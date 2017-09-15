/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : vClipMaskUvBounds;
        vec3 vClipMaskUv : vClipMaskUv;
        flat vec2 vTextureOffsetY : vTextureOffsetY; // Offset of the y plane into the texture atlas.
        flat vec2 vTextureOffsetU : vTextureOffsetU; // Offset of the u plane into the texture atlas.
        flat vec2 vTextureOffsetV : vTextureOffsetV; // Offset of the v plane into the texture atlas.
        flat vec2 vTextureSizeY : vTextureSizeY;   // Size of the y plane in the texture atlas.
        flat vec2 vTextureSizeUv : vTextureSizeUv;  // Size of the u and v planes in the texture atlas.
        flat vec2 vStretchSize : vStretchSize;
        flat vec2 vHalfTexelY : vHalfTexelY;     // Normalized length of the half of a Y texel.
        flat vec2 vHalfTexelUv : vHalfTexelUv;    // Normalized length of the half of u and v texels.
        flat vec3 vLayers : vLayers;

#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos: vLocalPos;
        flat vec4 vLocalBounds : vLocalBounds;
#else
        vec2 vLocalPos: vLocalPos;
#endif //WR_FEATURE_TRANSFORM
    };
#else
flat varying vec2 vTextureOffsetY; // Offset of the y plane into the texture atlas.
flat varying vec2 vTextureOffsetU; // Offset of the u plane into the texture atlas.
flat varying vec2 vTextureOffsetV; // Offset of the v plane into the texture atlas.
flat varying vec2 vTextureSizeY;   // Size of the y plane in the texture atlas.
flat varying vec2 vTextureSizeUv;  // Size of the u and v planes in the texture atlas.
flat varying vec2 vStretchSize;
flat varying vec2 vHalfTexelY;     // Normalized length of the half of a Y texel.
flat varying vec2 vHalfTexelUv;    // Normalized length of the half of u and v texels.
flat varying vec3 vLayers;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
#else
varying vec2 vLocalPos;
#endif //WR_FEATURE_TRANSFORM
#endif //WR_DX11
