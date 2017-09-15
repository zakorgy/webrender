/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : vClipMaskUvBounds;
        vec3 vClipMaskUv : vClipMaskUv;
        flat vec2 vTextureOffset: vTextureOffset; // Offset of this image into the texture atlas.
        flat vec2 vTextureSize: vTextureSize;   // Size of the image in the texture atlas.
        flat vec2 vTileSpacing: vTileSpacing;   // Amount of space between tiled instances of this image.
        flat vec4 vStRect: vStRect;        // Rectangle of valid texture rect.
        flat float vLayer: vLayer;
        flat vec2 vStretchSize: vStretchSize;

#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos: vLocalPos;
        flat vec4 vLocalBounds : vLocalBounds;
#else
        vec2 vLocalPos: vLocalPos;
#endif //WR_FEATURE_TRANSFORM
    };
#else

// If this is in WR_FEATURE_TEXTURE_RECT mode, the rect and size use non-normalized
// texture coordinates. Otherwise, it uses normalized texture coordinates. Please
// check GL_TEXTURE_RECTANGLE.
flat varying vec2 vTextureOffset; // Offset of this image into the texture atlas.
flat varying vec2 vTextureSize;   // Size of the image in the texture atlas.
flat varying vec2 vTileSpacing;   // Amount of space between tiled instances of this image.
flat varying vec4 vStRect;        // Rectangle of valid texture rect.
flat varying float vLayer;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
#else
varying vec2 vLocalPos;
#endif //WR_FEATURE_TRANSFORM
flat varying vec2 vStretchSize;
#endif //WR_DX11
