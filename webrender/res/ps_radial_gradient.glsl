/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        //flat vec4 vClipMaskUvBounds : POSITION0;
        //vec3 vClipMaskUv : POSITION1;
        flat int vGradientAddress : vGradientAddress;
        flat float vGradientRepeat : vGradientRepeat;

        flat vec2 vStartCenter : vStartCenter;
        flat vec2 vEndCenter : vEndCenter;
        flat float vStartRadius : vStartRadius;
        flat float vEndRadius : vEndRadius;

        flat vec2 vTileSize : vTileSize;
        flat vec2 vTileRepeat : vTileRepeat;

        vec2 vPos : vPos;
    };
#else
flat varying int vGradientAddress;
flat varying float vGradientRepeat;

flat varying vec2 vStartCenter;
flat varying vec2 vEndCenter;
flat varying float vStartRadius;
flat varying float vEndRadius;

flat varying vec2 vTileSize;
flat varying vec2 vTileRepeat;

varying vec2 vPos;
#endif

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
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    int vGradientAddress = IN.vGradientAddress;
    float vGradientRepeat = IN.vGradientRepeat;
    vec2 vStartCenter  = IN.vStartCenter;
    vec2 vEndCenter  = IN.vEndCenter;
    float vStartRadius  = IN.vStartRadius;
    float vEndRadius  = IN.vEndRadius;
    vec2 vTileSize = IN.vTileSize;
    vec2 vTileRepeat = IN.vTileRepeat;
    vec2 vPos = IN.vPos;
    vec4 gl_FragCoord = IN.Position;
#endif
    vec2 pos = mod(vPos, vTileRepeat);

    if (pos.x >= vTileSize.x ||
        pos.y >= vTileSize.y) {
        discard;
    }

    vec2 cd = vEndCenter - vStartCenter;
    vec2 pd = pos - vStartCenter;
    float rd = vEndRadius - vStartRadius;

    // Solve for t in length(t * cd - pd) = vStartRadius + t * rd
    // using a quadratic equation in form of At^2 - 2Bt + C = 0
    float A = dot(cd, cd) - rd * rd;
    float B = dot(pd, cd) + vStartRadius * rd;
    float C = dot(pd, pd) - vStartRadius * vStartRadius;

    float offset;
    if (A == 0.0) {
        // Since A is 0, just solve for -2Bt + C = 0
        if (B == 0.0) {
            discard;
        }
        float t = 0.5 * C / B;
        if (vStartRadius + rd * t >= 0.0) {
            offset = t;
        } else {
            discard;
        }
    } else {
        float discr = B * B - A * C;
        if (discr < 0.0) {
            discard;
        }
        discr = sqrt(discr);
        float t0 = (B + discr) / A;
        float t1 = (B - discr) / A;
        if (vStartRadius + rd * t0 >= 0.0) {
            offset = t0;
        } else if (vStartRadius + rd * t1 >= 0.0) {
            offset = t1;
        } else {
            discard;
        }
    }

    SHADER_OUT(Target0, sample_gradient(vGradientAddress, offset, vGradientRepeat, gl_FragCoord));
}
#endif //WR_FRAGMENT_SHADER
