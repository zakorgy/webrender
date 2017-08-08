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
    RadialGradient gradient = fetch_radial_gradient(prim.specific_prim_address);

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

    SHADER_OUT(vStartCenter, gradient.start_end_center.xy);
    SHADER_OUT(vEndCenter, gradient.start_end_center.zw);

    SHADER_OUT(vStartRadius, gradient.start_end_radius_ratio_xy_extend_mode.x);
    SHADER_OUT(vEndRadius, gradient.start_end_radius_ratio_xy_extend_mode.y);

    SHADER_OUT(vTileSize, gradient.tile_size_repeat.xy);
    SHADER_OUT(vTileRepeat, gradient.tile_size_repeat.zw);

    // Transform all coordinates by the y scale so the
    // fragment shader can work with circles
    float ratio_xy = gradient.start_end_radius_ratio_xy_extend_mode.z;

#ifdef WR_DX11
    vec2 vPos = OUT.vPos;
    vec2 vStartCenter  = OUT.vStartCenter;
    vec2 vEndCenter  = OUT.vEndCenter;
    vec2 vTileSize = OUT.vTileSize;
    vec2 vTileRepeat = OUT.vTileRepeat;
#endif
    SHADER_OUT(vPos.y, vPos.y * ratio_xy);
    SHADER_OUT(vStartCenter.y, vStartCenter.y * ratio_xy);
    SHADER_OUT(vEndCenter.y, vEndCenter.y * ratio_xy);
    SHADER_OUT(vTileSize.y, vTileSize.y * ratio_xy);
    SHADER_OUT(vTileRepeat.y, vTileRepeat.y * ratio_xy);

    SHADER_OUT(vGradientAddress, prim.specific_prim_address + VECS_PER_GRADIENT);

    // Whether to repeat the gradient instead of clamping.
    SHADER_OUT(vGradientRepeat, float(int(gradient.start_end_radius_ratio_xy_extend_mode.w) == EXTEND_MODE_REPEAT));
}
