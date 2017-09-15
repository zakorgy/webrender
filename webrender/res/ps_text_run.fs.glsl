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
    vec3 vUv = IN.vUv;
    vec4 vUvBorder = IN.vUvBorder;
#ifdef WR_FEATURE_TRANSFORM
    vec3 vLocalPos = IN.vLocalPos;
    vec4 vLocalBounds = IN.vLocalBounds;
#endif
#endif //WR_DX11
    vec3 tc = vec3(clamp(vUv.xy, vUvBorder.xy, vUvBorder.zw), vUv.z);
#ifdef WR_FEATURE_SUBPIXEL_AA
    //note: the blend mode is not compatible with clipping
    SHADER_OUT(Target0, texture(sColor0, tc));
#else
    float alpha = texture(sColor0, tc).a;
#ifdef WR_FEATURE_TRANSFORM
    float a = 0.0;
    init_transform_fs(vLocalPos, vLocalBounds, a);
    alpha *= a;
#endif //WR_FEATURE_TRANSFORM
    alpha = min(alpha, do_clip(vClipMaskUvBounds, vClipMaskUv));
    SHADER_OUT(Target0, vec4(vColor.rgb, vColor.a * alpha));
#endif //WR_FEATURE_SUBPIXEL_AA
}
