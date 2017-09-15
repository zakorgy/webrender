/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#define BS_HEADER_VECS 4

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    ivec4 aDataA = IN.data0;
    ivec4 aDataB = IN.data1;
#endif //WR_DX11
    Primitive prim = load_primitive(aDataA, aDataB);
    BoxShadow bs = fetch_boxshadow(prim.specific_prim_address);
    RectWithSize segment_rect = fetch_instance_geometry(prim.specific_prim_address + BS_HEADER_VECS + prim.user_data0);

    VertexInfo vi = write_vertex(aPosition,
                                 segment_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect
#ifdef WR_DX11
                                 , OUT.Position
#endif //WR_DX11
                                 );

    RenderTaskData child_task = fetch_render_task(prim.user_data1);
    SHADER_OUT(vUv.z, child_task.data1.x);

    // Constant offsets to inset from bilinear filtering border.
    vec2 patch_origin = child_task.data0.xy + vec2(1.0, 1.0);
    vec2 patch_size_device_pixels = child_task.data0.zw - vec2(2.0, 2.0);
    vec2 patch_size = patch_size_device_pixels / uDevicePixelRatio;

    SHADER_OUT(vUv.xy, (vi.local_pos - prim.local_rect.p0) / patch_size);
    SHADER_OUT(vMirrorPoint, 0.5 * prim.local_rect.size / patch_size);

    vec2 texture_size = vec2(textureSize(sCacheRGBA8, 0));
    SHADER_OUT(vCacheUvRectCoords, vec4(patch_origin, patch_origin + patch_size_device_pixels) / texture_size.xyxy);

    SHADER_OUT(vColor, bs.color);

    write_clip(vi.screen_pos,
               prim.clip_area
#ifdef WR_DX11
               , OUT.vClipMaskUvBounds
               , OUT.vClipMaskUv
#endif //WR_DX11
               );
}
