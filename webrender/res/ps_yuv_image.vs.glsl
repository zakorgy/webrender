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
#endif //WR_FEATURE_TRANSFORM

    write_clip(vi.screen_pos,
               prim.clip_area
#ifdef WR_DX11
               , OUT.vClipMaskUvBounds
               , OUT.vClipMaskUv
#endif //WR_DX11
               );

    ImageResource y_rect = fetch_image_resource(prim.user_data0);
    SHADER_OUT(vLayers, vec3(y_rect.layer, 0.0, 0.0));

#ifndef WR_FEATURE_INTERLEAVED_Y_CB_CR  // only 1 channel
    ImageResource u_rect = fetch_image_resource(prim.user_data1);
    SHADER_OUT(vLayers.y, u_rect.layer);
#ifndef WR_FEATURE_NV12 // 2 channel
    ImageResource v_rect = fetch_image_resource(prim.user_data2);
    SHADER_OUT(vLayers.z, v_rect.layer);
#endif //WR_FEATURE_NV12
#endif //WR_FEATURE_INTERLEAVED_Y_CB_CR

    // If this is in WR_FEATURE_TEXTURE_RECT mode, the rect and size use
    // non-normalized texture coordinates.
#ifdef WR_FEATURE_TEXTURE_RECT
    vec2 y_texture_size_normalization_factor = vec2(1, 1);
#else
    vec2 y_texture_size_normalization_factor = vec2(textureSize(sColor0, 0));
#endif //WR_FEATURE_TEXTURE_RECT
    vec2 y_st0 = y_rect.uv_rect.xy / y_texture_size_normalization_factor;
    vec2 y_st1 = y_rect.uv_rect.zw / y_texture_size_normalization_factor;

    SHADER_OUT(vTextureSizeY, y_st1 - y_st0);
    SHADER_OUT(vTextureOffsetY, y_st0);

#ifndef WR_FEATURE_INTERLEAVED_Y_CB_CR
    // This assumes the U and V surfaces have the same size.
#ifdef WR_FEATURE_TEXTURE_RECT
    vec2 uv_texture_size_normalization_factor = vec2(1, 1);
#else
    vec2 uv_texture_size_normalization_factor = vec2(textureSize(sColor1, 0));
#endif //WR_FEATURE_TEXTURE_RECT
    vec2 u_st0 = u_rect.uv_rect.xy / uv_texture_size_normalization_factor;
    vec2 u_st1 = u_rect.uv_rect.zw / uv_texture_size_normalization_factor;

#ifndef WR_FEATURE_NV12
    vec2 v_st0 = v_rect.uv_rect.xy / uv_texture_size_normalization_factor;
#endif //WR_FEATURE_NV12

    SHADER_OUT(vTextureSizeUv, u_st1 - u_st0);
    SHADER_OUT(vTextureOffsetU, u_st0);
#ifndef WR_FEATURE_NV12
    SHADER_OUT(vTextureOffsetV, v_st0);
#endif //WR_FEATURE_NV12
#endif //WR_FEATURE_INTERLEAVED_Y_CB_CR

    YuvImage image = fetch_yuv_image(prim.specific_prim_address);
    SHADER_OUT(vStretchSize, image.size);

    SHADER_OUT(vHalfTexelY, vec2(0.5, 0.5) / y_texture_size_normalization_factor);
#ifndef WR_FEATURE_INTERLEAVED_Y_CB_CR
    SHADER_OUT(vHalfTexelUv, vec2(0.5, 0.5) / uv_texture_size_normalization_factor);
#endif //WR_FEATURE_INTERLEAVED_Y_CB_CR
}
