/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Draw a cached primitive (e.g. a blurred text run) from the
// target cache to the framebuffer, applying tile clip boundaries.

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    ivec4 aDataA = IN.data0;
    ivec4 aDataB = IN.data1;
#endif //WR_DX11
    Primitive prim = load_primitive(aDataA, aDataB);

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

    RenderTaskData child_task = fetch_render_task(prim.user_data1);
    SHADER_OUT(vUv.z, child_task.data1.x);

    vec2 texture_size = vec2(textureSize(sCacheRGBA8, 0));
    vec2 uv0 = child_task.data0.xy;
    vec2 uv1 = (child_task.data0.xy + child_task.data0.zw);

    vec2 f = (vi.local_pos - prim.local_rect.p0) / prim.local_rect.size;

    SHADER_OUT(vUv.xy, mix(uv0 / texture_size, uv1 / texture_size, f));
    SHADER_OUT(vUvBounds, vec4(uv0 + vec2(0.5, 0.5), uv1 - vec2(0.5, 0.5)) / texture_size.xyxy);
}
