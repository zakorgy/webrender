/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec3 vUv = IN.vUv;
    vec4 vColor = IN.vColor;
#endif //WR_DX11
    float a = texture(sColor0, vUv).a;
    SHADER_OUT(Target0, vec4(vColor.rgb, vColor.a * a));
}
