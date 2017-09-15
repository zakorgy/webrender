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
#endif //WR_DX11
    Primitive prim = load_primitive(aDataA, aDataB);
    CompositeInstance ci = fetch_composite_instance(aDataA, aDataB);
    AlphaBatchTask dest_task = fetch_alpha_batch_task(ci.render_task_index);
    AlphaBatchTask src_task = fetch_alpha_batch_task(ci.src_task_index);

    vec2 dest_origin = dest_task.render_target_origin -
                       dest_task.screen_space_origin +
                       src_task.screen_space_origin;

    vec2 local_pos = mix(dest_origin,
                         dest_origin + src_task.size,
                         aPosition.xy);

    vec2 texture_size = vec2(textureSize(sCacheRGBA8, 0));
    vec2 st0 = src_task.render_target_origin;
    vec2 st1 = src_task.render_target_origin + src_task.size;

    vec2 uv = src_task.render_target_origin + aPosition.xy * src_task.size;
    SHADER_OUT(vUv, vec3(uv / texture_size, src_task.render_target_layer_index));
    SHADER_OUT(vUvBounds, vec4(st0 + 0.5, st1 - 0.5) / texture_size.xyxy);

    SHADER_OUT(vOp, ci.user_data0);
    SHADER_OUT(vAmount, float(ci.user_data1) / 65535.0);
    SHADER_OUT(gl_Position, mul(vec4(local_pos, ci.z, 1.0), uTransform));
}
