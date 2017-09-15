/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec3 vUv = IN.vUv;
    vec4 vUvBounds = IN.vUvBounds;
#endif //WR_DX11
	vec2 uv = clamp(vUv.xy, vUvBounds.xy, vUvBounds.zw);
    SHADER_OUT(Target0, texture(sColor0, vec3(uv, vUv.z)));
}
