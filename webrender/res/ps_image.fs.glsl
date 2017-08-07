//#line 1

/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
#endif
#ifdef WR_FEATURE_TRANSFORM
    float alpha = 0.0;
#ifdef WR_DX11
    vec3 vLocalPos = IN.vLocalPos;
#endif
    //TODO: Uncomment vLocalBounds and fix init_transform_fs!
    /*vec2 pos = init_transform_fs(vLocalPos, alpha);

    // We clamp the texture coordinate calculation here to the local rectangle boundaries,
    // which makes the edge of the texture stretch instead of repeat.
    vec2 relative_pos_in_rect = clamp(pos, vLocalBounds.xy, vLocalBounds.zw) - vLocalBounds.xy;*/
    vec2 relative_pos_in_rect = vec2(0.0, 0.0);
#else
    float alpha = 1.0;
#ifdef WR_DX11
    vec2 vLocalPos = IN.vLocalPos;
#endif
    vec2 relative_pos_in_rect = vLocalPos;
#endif

#ifdef WR_DX11
    vec4 vClipMaskUvBounds = IN.vClipMaskUvBounds;
    vec3 vClipMaskUv = IN.vClipMaskUv;
    vec2 vTextureOffset = IN.vTextureOffset;
    vec2 vTextureSize = IN.vTextureSize;
    vec2 vTileSpacing = IN.vTileSpacing;
    vec4 vStRect = IN.vStRect;
    vec2 vStretchSize = IN.vStretchSize;
#endif

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
#endif
    vec2 st = vTextureOffset + ((position_in_tile / vStretchSize) * vTextureSize);
    st = clamp(st, vStRect.xy, vStRect.zw);

    alpha = alpha * float(all(bvec2(step(position_in_tile, vStretchSize))));

#ifdef WR_DX11
    //TODO: Add the Sample for the WR_FEATURE_TRANSFORM case.
    OUT.Target0 = vec4(alpha, alpha, alpha, alpha) * vec4(sColor0.Sample(sColor0_, st));
#else
    Target0 = vec4(alpha) * TEX_SAMPLE(sColor0, st);
#endif
}
