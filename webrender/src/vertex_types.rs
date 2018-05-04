/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct BlurInstance {
    pub aData0: [i32; 4],
    pub aData1: [i32; 4],
    pub aBlurRenderTaskAddress: i32,
    pub aBlurSourceTaskAddress: i32,
    pub aBlurDirection: i32,
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct ClipMaskInstance {
    pub aClipRenderTaskAddress: i32,
    pub aScrollNodeId: i32,
    pub aClipSegment: i32,
    pub aClipDataResourceAddress: [i32; 4],
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct DebugColorVertex {
    aPosition: [f32; 3],
    aColor: [f32; 4],
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct DebugFontVertex {
    aPosition: [f32; 3],
    aColor: [f32; 4],
    aColorTexCoord: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct PrimitiveInstance {
    pub aData0: [i32; 4],
    pub aData1: [i32; 4],
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct Vertex {
    pub aPosition: [f32; 3],
}
