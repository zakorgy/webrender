/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        int vGradientAddress : vGradientAddress;
        float vGradientRepeat : vGradientRepeat;

        vec2 vScaledDir : vScaledDir;
        vec2 vStartPoint : vStartPoint;

        vec2 vTileSize : vTileSize;
        vec2 vTileRepeat : vTileRepeat;

        vec2 vPos : vPos;
    };
#else

flat varying int vGradientAddress;
flat varying float vGradientRepeat;

flat varying vec2 vScaledDir;
flat varying vec2 vStartPoint;

flat varying vec2 vTileSize;
flat varying vec2 vTileRepeat;

varying vec2 vPos;
#endif //WR_DX11
