/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
#ifdef WR_FEATURE_CLIP
    vec4 vClipMaskUvBounds = IN.vClipMaskUvBounds;
    vec3 vClipMaskUv = IN.vClipMaskUv;
#endif //WR_FEATURE_CLIP
    vec4 vColor = IN.vColor;
#endif //WR_DX11
    float alpha = 1.0;
#ifdef WR_FEATURE_TRANSFORM
    alpha = 0.0;
#ifdef WR_DX11
    vec3 vLocalPos = IN.vLocalPos;
    vec4 vLocalBounds = IN.vLocalBounds;
#endif //WR_DX11
    init_transform_fs(vLocalPos, vLocalBounds, alpha);
#endif //WR_FEATURE_TRANSFORM

#ifdef WR_FEATURE_CLIP
    alpha = min(alpha, do_clip(vClipMaskUvBounds, vClipMaskUv));
#endif //WR_FEATURE_CLIP
    SHADER_OUT(Target0, vColor * vec4(1.0, 1.0, 1.0, alpha));
}
