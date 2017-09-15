/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,shared_other

#ifdef WR_DX11
struct v2p {
    vec4 gl_Position : SV_Position;
    vec4 vColor : vColor;
};
#else
varying vec4 vColor;
#endif //WR_DX11

#ifdef WR_VERTEX_SHADER
#ifdef WR_DX11
struct a2v_debug {
    vec4 color : aColor;
    vec3 pos : aPosition;
};
#else
in vec4 aColor;
#endif //WR_DX11

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v_debug IN, out v2p OUT) {
    vec4 aColor = IN.color;
    vec3 aPosition = IN.pos;
#endif //WR_DX11
    SHADER_OUT(vColor, aColor);
    vec4 pos = vec4(aPosition, 1.0);
    pos.xy = floor(pos.xy * uDevicePixelRatio + 0.5) / uDevicePixelRatio;
    SHADER_OUT(gl_Position, mul(pos, uTransform));
}
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec4 vColor = IN.vColor;
#endif //WR_DX11
    SHADER_OUT(Target0, vColor);
}
#endif //WR_FRAGMENT_SHADER
