/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    ivec4 aDataA = IN.data0;
    ivec4 aDataB = IN.data1;
    int gl_VertexID = IN.vertexId;
#endif //WR_DX11
    Primitive prim = load_primitive(aDataA, aDataB);
    Image image = fetch_image(prim.specific_prim_address);
    ImageResource res = fetch_image_resource(prim.user_data0);

#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(gl_VertexID,
                                                    prim.local_rect,
                                                    prim.local_clip_rect,
                                                    prim.z,
                                                    prim.layer,
                                                    prim.task,
                                                    prim.local_rect
#ifdef WR_DX11
                                                    , OUT.Position
                                                    , OUT.vLocalBounds
#endif //WR_DX11
                                                    );
    SHADER_OUT(vLocalPos, vi.local_pos);

#else
    VertexInfo vi = write_vertex(aPosition,
                                 prim.local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect
#ifdef WR_DX11
                                 , OUT.Position
#endif //WR_DX11
                                 );
    SHADER_OUT(vLocalPos, vi.local_pos - prim.local_rect.p0);
#endif //WR_FEATURE_TEXTURE_RECT

    write_clip(vi.screen_pos,
               prim.clip_area
#ifdef WR_DX11
               , OUT.vClipMaskUvBounds
               , OUT.vClipMaskUv
#endif //WR_DX11
               );

    // If this is in WR_FEATURE_TEXTURE_RECT mode, the rect and size use
    // non-normalized texture coordinates.
#ifdef WR_FEATURE_TEXTURE_RECT
    vec2 texture_size_normalization_factor = vec2(1, 1);
#else
    vec2 texture_size_normalization_factor = vec2(textureSize(sColor0, 0));
#endif //WR_FEATURE_TEXTURE_RECT

    vec2 uv0, uv1;

    if (image.sub_rect.x < 0.0) {
        uv0 = res.uv_rect.xy;
        uv1 = res.uv_rect.zw;
    } else {
        uv0 = res.uv_rect.xy + image.sub_rect.xy;
        uv1 = res.uv_rect.xy + image.sub_rect.zw;
    }

    // vUv will contain how many times this image has wrapped around the image size.
    vec2 st0 = uv0 / texture_size_normalization_factor;
    vec2 st1 = uv1 / texture_size_normalization_factor;

    SHADER_OUT(vLayer, res.layer);
    SHADER_OUT(vTextureSize, st1 - st0);
    SHADER_OUT(vTextureOffset, st0);
    SHADER_OUT(vTileSpacing, image.stretch_size_and_tile_spacing.zw);
    SHADER_OUT(vStretchSize, image.stretch_size_and_tile_spacing.xy);

    // We clamp the texture coordinates to the half-pixel offset from the borders
    // in order to avoid sampling outside of the texture area.
    vec2 half_texel = vec2(0.5, 0.5) / texture_size_normalization_factor;
    SHADER_OUT(vStRect, vec4(min(st0, st1) + half_texel, max(st0, st1) - half_texel));
}
