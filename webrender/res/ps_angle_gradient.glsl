/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        //flat vec4 vClipMaskUvBounds : POSITION0;
        //vec3 vClipMaskUv : POSITION1;
        int vGradientAddress : ADDRESS;
        float vGradientRepeat : PSIZE;

        vec2 vScaledDir : POSITION2;
        vec2 vStartPoint : POSITION3;

        vec2 vTileSize : POSITION4;
        vec2 vTileRepeat : POSITION5;

        vec2 vPos : POSITION6;
    };
#else

flat varying int vGradientAddress;
flat varying float vGradientRepeat;

flat varying vec2 vScaledDir;
flat varying vec2 vStartPoint;

flat varying vec2 vTileSize;
flat varying vec2 vTileRepeat;

varying vec2 vPos;
#endif
