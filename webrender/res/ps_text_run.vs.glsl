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
    TextRun text = fetch_text_run(prim.specific_prim_address);

    int glyph_index = prim.user_data0;
    int resource_address = prim.user_data2;
    Glyph glyph = fetch_glyph(prim.specific_prim_address,
                              glyph_index,
                              text.subpx_dir);
    GlyphResource res = fetch_glyph_resource(resource_address);

    vec2 local_pos = glyph.offset +
                     text.offset +
                     vec2(res.offset.x, -res.offset.y) / uDevicePixelRatio;

#ifdef WR_DX11
    RectWithSize local_rect = {local_pos, float2(res.uv_rect.zw - res.uv_rect.xy) / uDevicePixelRatio};
#else
    RectWithSize local_rect = RectWithSize(local_pos,
                                           (res.uv_rect.zw - res.uv_rect.xy) / uDevicePixelRatio);
#endif //WR_DX11

#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(gl_VertexID,
                                                    local_rect,
                                                    prim.local_clip_rect,
                                                    prim.z,
                                                    prim.layer,
                                                    prim.task,
                                                    local_rect
#ifdef WR_DX11
                                                    , OUT.Position
                                                    , OUT.vLocalBounds
#endif //WR_DX11
                                                    );
    SHADER_OUT(vLocalPos, vi.local_pos);
    vec2 f = (vi.local_pos.xy / vi.local_pos.z - local_rect.p0) / local_rect.size;
#else
    VertexInfo vi = write_vertex(aPosition,
                                 local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 local_rect
#ifdef WR_DX11
                                 , OUT.Position
#endif //WR_FEATURE_TRANSFORM
                                 );
    vec2 f = (vi.local_pos - local_rect.p0) / local_rect.size;
#endif //WR_DX11

    write_clip(vi.screen_pos,
               prim.clip_area
#ifdef WR_DX11
               , OUT.vClipMaskUvBounds
               , OUT.vClipMaskUv
#endif //WR_DX11
               );

    vec2 texture_size = vec2(textureSize(sColor0, 0));
    vec2 st0 = res.uv_rect.xy / texture_size;
    vec2 st1 = res.uv_rect.zw / texture_size;

    SHADER_OUT(vColor, text.color);
    SHADER_OUT(vUv, vec3(mix(st0, st1, f), res.layer));
    SHADER_OUT(vUvBorder, (res.uv_rect + vec4(0.5, 0.5, -0.5, -0.5)) / texture_size.xyxy);
}
