/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        int vGradientAddress : vGradientAddress;
        float vGradientRepeat : vGradientRepeat;

        vec2 vScaledDir : vScaledDir;
        vec2 vStartPoint : vStartPoint;

        vec2 vTileSize : vTileSize;
        vec2 vTileRepeat : vTileRepeat;

        vec2 vPos : vPos;
    };
#else

flat varying int vGradientAddress;
flat varying float vGradientRepeat;

flat varying vec2 vScaledDir;
flat varying vec2 vStartPoint;

flat varying vec2 vTileSize;
flat varying vec2 vTileRepeat;

varying vec2 vPos;
#endif //WR_DX11

#ifdef WR_VERTEX_SHADER
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
#endif //WR_DX11
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
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    int vGradientAddress = IN.vGradientAddress;
    float vGradientRepeat = IN.vGradientRepeat;

    vec2 vScaledDir = IN.vScaledDir;
    vec2 vStartPoint = IN.vStartPoint;

    vec2 vTileSize = IN.vTileSize;
    vec2 vTileRepeat = IN.vTileRepeat;

    vec2 vPos = IN.vPos;

    vec4 gl_FragCoord = IN.Position;
#endif //WR_DX11
    vec2 pos = mod(vPos, vTileRepeat);

    if (pos.x >= vTileSize.x ||
        pos.y >= vTileSize.y) {
        discard;
    }

    float offset = dot(pos - vStartPoint, vScaledDir);

    SHADER_OUT(Target0, sample_gradient(vGradientAddress, offset, vGradientRepeat, gl_FragCoord));
}
#endif //WR_FRAGMENT_SHADER
