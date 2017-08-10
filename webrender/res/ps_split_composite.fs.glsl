//#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

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
