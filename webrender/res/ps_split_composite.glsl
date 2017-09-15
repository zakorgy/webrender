/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

#ifdef WR_DX11
    struct v2p {
        vec4 gl_Position : SV_Position;
        vec3 vUv : vUv;
        flat vec4 vUvTaskBounds : vUvTaskBounds;
        flat vec4 vUvSampleBounds : vUvSampleBounds;
    };
#else
varying vec3 vUv;
flat varying vec4 vUvTaskBounds;
flat varying vec4 vUvSampleBounds;
#endif //WR_DX11

#ifdef WR_VERTEX_SHADER
struct SplitGeometry {
    vec3 points[4];
};

SplitGeometry fetch_split_geometry(int address) {
    ivec2 uv = get_resource_cache_uv(address);

    vec4 data0 = texelFetchOffset(sResourceCache, uv, 0, ivec2(0, 0));
    vec4 data1 = texelFetchOffset(sResourceCache, uv, 0, ivec2(1, 0));
    vec4 data2 = texelFetchOffset(sResourceCache, uv, 0, ivec2(2, 0));

    SplitGeometry geo;
    geo.points[0] = vec3(data0.xyz);
    geo.points[1] = vec3(data0.w, data1.xy);
    geo.points[2] = vec3(data1.zw, data2.x);
    geo.points[3] = vec3(data2.yzw);
    return geo;
}

vec3 bilerp(vec3 a, vec3 b, vec3 c, vec3 d, float s, float t) {
    vec3 x = mix(a, b, t);
    vec3 y = mix(c, d, t);
    return mix(x, y, s);
}

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    ivec4 aDataA = IN.data0;
    ivec4 aDataB = IN.data1;
#endif //WR_DX11
    CompositeInstance ci = fetch_composite_instance(aDataA, aDataB);
    SplitGeometry geometry = fetch_split_geometry(ci.user_data0);
    AlphaBatchTask src_task = fetch_alpha_batch_task(ci.src_task_index);

    vec3 world_pos = bilerp(geometry.points[0], geometry.points[1],
                            geometry.points[3], geometry.points[2],
                            aPosition.y, aPosition.x);
    vec4 final_pos = vec4(world_pos.xy * uDevicePixelRatio, ci.z, 1.0);

    SHADER_OUT(gl_Position, mul(final_pos, uTransform));
    vec2 uv_origin = src_task.render_target_origin;
    vec2 uv_pos = uv_origin + world_pos.xy - src_task.screen_space_origin;
    vec2 texture_size = vec2(textureSize(sCacheRGBA8, 0));
    SHADER_OUT(vUv, vec3(uv_pos / texture_size, src_task.render_target_layer_index));
    SHADER_OUT(vUvTaskBounds, vec4(uv_origin, uv_origin + src_task.size) / texture_size.xyxy);
    SHADER_OUT(vUvSampleBounds, vec4(uv_origin + 0.5, uv_origin + src_task.size - 0.5) / texture_size.xyxy);
}
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec3 vUv = IN.vUv;
    vec4 vUvTaskBounds = IN.vUvTaskBounds;
    vec4 vUvSampleBounds = IN.vUvSampleBounds;
#endif //WR_DX11
    bvec4 inside = lessThanEqual(vec4(vUvTaskBounds.xy, vUv.xy),
                                 vec4(vUv.xy, vUvTaskBounds.zw));
    if (all(inside)) {
        vec2 uv = clamp(vUv.xy, vUvSampleBounds.xy, vUvSampleBounds.zw);
        SHADER_OUT(Target0, textureLod(sCacheRGBA8, vec3(uv, vUv.z), 0.0));
    } else {
        SHADER_OUT(Target0, vec4(0.0, 0.0, 0.0, 0.0));
    }
}
#endif //WR_FRAGMENT_SHADER
