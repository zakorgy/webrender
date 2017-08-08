//#line 1
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
#endif
    Primitive prim = load_primitive(aDataA, aDataB);
    Gradient gradient = fetch_gradient(prim.specific_prim_address);

    VertexInfo vi = write_vertex(aPosition,
                                 prim.local_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect
#ifdef WR_DX11
                                 , OUT.Position
#endif
                                 );

    SHADER_OUT(vPos, vi.local_pos - prim.local_rect.p0);

    vec2 start_point = gradient.start_end_point.xy;
    vec2 end_point = gradient.start_end_point.zw;
    vec2 dir = end_point - start_point;

    SHADER_OUT(vStartPoint, start_point);
    SHADER_OUT(vScaledDir, dir / dot(dir, dir));

    SHADER_OUT(vTileSize, gradient.tile_size_repeat.xy);
    SHADER_OUT(vTileRepeat, gradient.tile_size_repeat.zw);

    SHADER_OUT(vGradientAddress, prim.specific_prim_address + VECS_PER_GRADIENT);

    // Whether to repeat the gradient instead of clamping.
    SHADER_OUT(vGradientRepeat, float(int(gradient.extend_mode.x) == EXTEND_MODE_REPEAT));
}
