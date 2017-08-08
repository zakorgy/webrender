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
#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos = IN.vLocalPos;
#else
        vec2 vPos = IN.vPos;
#endif
#endif
#ifdef WR_FEATURE_TRANSFORM
    float alpha = 0.0;
    vec2 local_pos = init_transform_fs(vLocalPos, alpha);
#else
    float alpha = 1.0;
    vec2 local_pos = vPos;
#endif

    alpha = min(alpha, do_clip(
#ifdef WR_DX11
                                 vClipMaskUvBounds
                               , vClipMaskUv
#endif
                               ));
    vec4 color = dither(vColor * vec4(1.0, 1.0, 1.0, alpha)
#if defined(WR_DX11) && defined(WR_FEATURE_DITHERING)
                        , IN.Position
#endif
                        );
    SHADER_OUT(Target0, color);
}
