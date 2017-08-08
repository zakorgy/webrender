/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        //flat vec4 vClipMaskUvBounds : POSITION0;
        //vec3 vClipMaskUv : POSITION1;
        flat int vGradientAddress : ADDRESS;
        flat float vGradientRepeat : PSIZE1;

        flat vec2 vStartCenter : POSITION2;
        flat vec2 vEndCenter : POSITION3;
        flat float vStartRadius : PSIZE2;
        flat float vEndRadius : PSIZE3;

        flat vec2 vTileSize : POSITION4;
        flat vec2 vTileRepeat : POSITION5;

        vec2 vPos : POSITION6;
    };
#else
flat varying int vGradientAddress;
flat varying float vGradientRepeat;

flat varying vec2 vStartCenter;
flat varying vec2 vEndCenter;
flat varying float vStartRadius;
flat varying float vEndRadius;

flat varying vec2 vTileSize;
flat varying vec2 vTileRepeat;

varying vec2 vPos;
#endif
