/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec4 vClipMaskUvBounds = IN.vClipMaskUvBounds;
    vec3 vClipMaskUv = IN.vClipMaskUv;
    vec4 vColor = IN.vColor;
    vec4 gl_FragCoord = IN.Position;
#ifdef WR_FEATURE_TRANSFORM
    vec3 vLocalPos = IN.vLocalPos;
    vec4 vLocalBounds = IN.vLocalBounds;
#else
    vec2 vPos = IN.vPos;
#endif //WR_FEATURE_TRANSFORM
#endif //WR_DX11
#ifdef WR_FEATURE_TRANSFORM
    float alpha = 0.0;
    vec2 local_pos = init_transform_fs(vLocalPos, vLocalBounds, alpha);
#else
    float alpha = 1.0;
    vec2 local_pos = vPos;
#endif //WR_FEATURE_TRANSFORM

    alpha = min(alpha, do_clip(vClipMaskUvBounds, vClipMaskUv));
    vec4 color = dither(vColor * vec4(1.0, 1.0, 1.0, alpha), gl_FragCoord);
    SHADER_OUT(Target0, color);
}
