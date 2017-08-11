/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

float clip_against_ellipse_if_needed(vec2 pos,
                                     float current_distance,
                                     vec4 ellipse_center_radius,
                                     vec2 sign_modifier,
                                     float afwidth) {
    float ellipse_distance = distance_to_ellipse(pos - ellipse_center_radius.xy,
                                                 ellipse_center_radius.zw);

    return mix(current_distance,
               ellipse_distance + afwidth,
               all(lessThan(sign_modifier * pos, sign_modifier * ellipse_center_radius.xy)));
}

float rounded_rect(vec2 pos,
                   vec4 vClipCenter_Radius_TL,
                   vec4 vClipCenter_Radius_TR,
                   vec4 vClipCenter_Radius_BL,
                   vec4 vClipCenter_Radius_BR) {
    float current_distance = 0.0;

    // Apply AA
    float afwidth = 0.5 * length(fwidth(pos));

    // Clip against each ellipse.
    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_TL,
                                                      vec2(1.0, 1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_TR,
                                                      vec2(-1.0, 1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_BR,
                                                      vec2(-1.0, -1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_BL,
                                                      vec2(1.0, -1.0),
                                                      afwidth);

    return smoothstep(0.0, afwidth, 1.0 - current_distance);
}


#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec3 vPos = IN.vPos;
    vec4 vLocalBounds = IN.vLocalBounds;
    float vClipMode = IN.vClipMode;
    vec4 vClipCenter_Radius_TL = IN.vClipCenter_Radius_TL;
    vec4 vClipCenter_Radius_TR = IN.vClipCenter_Radius_TR;
    vec4 vClipCenter_Radius_BL = IN.vClipCenter_Radius_BL;
    vec4 vClipCenter_Radius_BR = IN.vClipCenter_Radius_BR;
#endif //WR_DX11
    float alpha = 1.f;
    vec2 local_pos = init_transform_fs(vPos, vLocalBounds, alpha);

    float clip_alpha = rounded_rect(local_pos,
                                    vClipCenter_Radius_TL,
                                    vClipCenter_Radius_TR,
                                    vClipCenter_Radius_BL,
                                    vClipCenter_Radius_BR);

    float combined_alpha = min(alpha, clip_alpha);

    // Select alpha or inverse alpha depending on clip in/out.
    float final_alpha = mix(combined_alpha, 1.0 - combined_alpha, vClipMode);

    SHADER_OUT(Target0, vec4(final_alpha, 0.0, 0.0, 1.0));
}
