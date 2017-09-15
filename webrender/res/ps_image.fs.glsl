/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec4 vClipMaskUvBounds = IN.vClipMaskUvBounds;
    vec3 vClipMaskUv = IN.vClipMaskUv;
    vec2 vTextureOffset = IN.vTextureOffset;
    vec2 vTextureSize = IN.vTextureSize;
    vec2 vTileSpacing = IN.vTileSpacing;
    vec4 vStRect = IN.vStRect;
    float vLayer = IN.vLayer;
    vec2 vStretchSize = IN.vStretchSize;
#endif //WR_DX11
#ifdef WR_FEATURE_TRANSFORM
    float alpha = 0.0;
#ifdef WR_DX11
    vec3 vLocalPos = IN.vLocalPos;
    vec4 vLocalBounds = IN.vLocalBounds;
#endif //WR_DX11
    vec2 pos = init_transform_fs(vLocalPos, vLocalBounds, alpha);

    // We clamp the texture coordinate calculation here to the local rectangle boundaries,
    // which makes the edge of the texture stretch instead of repeat.
    vec2 relative_pos_in_rect = clamp(pos, vLocalBounds.xy, vLocalBounds.zw) - vLocalBounds.xy;
#else
    float alpha = 1.0;
#ifdef WR_DX11
    vec2 vLocalPos = IN.vLocalPos;
#endif //WR_DX11
    vec2 relative_pos_in_rect = vLocalPos;
#endif //WR_FEATURE_TRANSFORM

    alpha = min(alpha, do_clip(vClipMaskUvBounds, vClipMaskUv));

    // We calculate the particular tile this fragment belongs to, taking into
    // account the spacing in between tiles. We only paint if our fragment does
    // not fall into that spacing.
#ifdef WR_DX11
    vec2 stretch_tile = vStretchSize + vTileSpacing;
    vec2 position_in_tile = vec2(mod(relative_pos_in_rect.x, stretch_tile.x),
                                 mod(relative_pos_in_rect.y, stretch_tile.y));
#else
    vec2 position_in_tile = mod(relative_pos_in_rect, vStretchSize + vTileSpacing);
#endif //WR_DX11
    vec2 st = vTextureOffset + ((position_in_tile / vStretchSize) * vTextureSize);
    st = clamp(st, vStRect.xy, vStRect.zw);

    alpha = alpha * float(all(bvec2(step(position_in_tile, vStretchSize))));

    vec4 color = TEX_SAMPLE(sColor0, vec3(st, vLayer));
    SHADER_OUT(Target0, vec4(alpha, alpha, alpha, alpha) * color);
}

