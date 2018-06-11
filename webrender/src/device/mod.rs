/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{DeviceUintSize, ImageFormat, TextureTarget};
#[cfg(not(feature = "gfx"))]
use gleam::gl::{GLuint, DYNAMIC_DRAW, STATIC_DRAW, STREAM_DRAW};
use internal_types::RenderTargetInfo;
#[cfg(feature = "gfx")]
use std::cell::Cell;
use std::ops::Add;
use std::path::PathBuf;
use std::thread;

cfg_if! {
    if #[cfg(feature = "gfx")] {
        mod gfx;
        pub use self::gfx::*;
    } else {
        mod gl;
        pub use self::gl::*;
    }
}

#[cfg(feature = "gfx")]
type IdType = u32;

#[cfg(not(feature = "gfx"))]
type IdType = GLuint;

#[cfg(feature = "debug_renderer")]
pub struct Capabilities {
    pub supports_multisampling: bool,
}

#[derive(Debug, Copy, Clone)]
pub enum ShaderKind {
    Primitive,
    Cache(VertexArrayKind),
    ClipCache,
    Brush,
    Text,
    #[allow(dead_code)]
    VectorStencil,
    #[allow(dead_code)]
    VectorCover,
    #[cfg(feature = "gfx")]
    DebugColor,
    #[cfg(feature = "gfx")]
    DebugFont,
}

#[cfg(feature = "gfx")]
impl ShaderKind {
    fn is_debug(&self) -> bool {
        match *self {
            ShaderKind::DebugFont | ShaderKind::DebugColor => true,
            _ => false,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum VertexArrayKind {
    Primitive,
    Blur,
    Clip,
    DashAndDot,
    VectorStencil,
    VectorCover,
    Border,
}

#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FrameId(usize);

impl FrameId {
    pub fn new(value: usize) -> Self {
        FrameId(value)
    }
}

impl Add<usize> for FrameId {
    type Output = FrameId;

    fn add(self, other: usize) -> FrameId {
        FrameId(self.0 + other)
    }
}

pub struct TextureSlot(pub usize);

#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub enum TextureFilter {
    Nearest,
    Linear,
    Trilinear,
}

#[derive(Debug)]
pub enum VertexAttributeKind {
    F32,
    #[cfg(feature = "debug_renderer")]
    U8Norm,
    U16Norm,
    I32,
    U16,
}

#[derive(Debug)]
pub struct VertexAttribute {
    pub name: &'static str,
    pub count: u32,
    pub kind: VertexAttributeKind,
}

#[derive(Debug)]
pub struct VertexDescriptor {
    pub vertex_attributes: &'static [VertexAttribute],
    pub instance_attributes: &'static [VertexAttribute],
}

#[cfg(not(feature = "gfx"))]
enum FBOTarget {
    Read,
    Draw,
}

/// Method of uploading texel data from CPU to GPU.
#[derive(Debug, Clone)]
pub enum UploadMethod {
    /// Just call `glTexSubImage` directly with the CPU data pointer
    Immediate,
    /// Accumulate the changes in PBO first before transferring to a texture.
    PixelBuffer(VertexUsageHint),
}

/// Plain old data that can be used to initialize a texture.
pub unsafe trait Texel: Copy {}
unsafe impl Texel for u8 {}
unsafe impl Texel for f32 {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReadPixelsFormat {
    Standard(ImageFormat),
    Rgba8,
}

pub trait FileWatcherHandler: Send {
    fn file_changed(&self, path: PathBuf);
}

#[cfg_attr(feature = "replay", derive(Clone))]
pub struct ExternalTexture {
    id: IdType,
    target: IdType,
}


impl ExternalTexture {
    pub fn new(id: u32, target: TextureTarget) -> Self {
        ExternalTexture {
            id,
            #[cfg(feature = "gfx")]
            target: target as _,
            #[cfg(not(feature = "gfx"))]
            target: get_gl_target(target),
        }
    }

    #[cfg(feature = "replay")]
    pub fn internal_id(&self) -> IdType {
        self.id
    }
}

pub struct Texture {
    id: IdType,
    target: IdType,
    layer_count: i32,
    format: ImageFormat,
    width: u32,
    height: u32,
    filter: TextureFilter,
    render_target: Option<RenderTargetInfo>,
    fbo_ids: Vec<FBOId>,
    depth_rb: Option<RBOId>,
    last_frame_used: FrameId,
    #[cfg(feature = "gfx")]
    bound_in_frame: Cell<FrameId>,
}

impl Texture {
    pub fn get_dimensions(&self) -> DeviceUintSize {
        DeviceUintSize::new(self.width, self.height)
    }

    pub fn get_render_target_layer_count(&self) -> usize {
        self.fbo_ids.len()
    }

    pub fn get_layer_count(&self) -> i32 {
        self.layer_count
    }

    pub fn get_format(&self) -> ImageFormat {
        self.format
    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn get_filter(&self) -> TextureFilter {
        self.filter
    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn get_render_target(&self) -> Option<RenderTargetInfo> {
        self.render_target.clone()
    }

    pub fn has_depth(&self) -> bool {
        self.depth_rb.is_some()
    }

    pub fn get_rt_info(&self) -> Option<&RenderTargetInfo> {
        self.render_target.as_ref()
    }

    pub fn used_in_frame(&self, frame_id: FrameId) -> bool {
        self.last_frame_used == frame_id
    }

    #[cfg(feature = "gfx")]
    fn still_in_flight(&self, frame_id: FrameId) -> bool {
        for i in 0..MAX_FRAME_COUNT {
            if self.bound_in_frame.get() == FrameId(frame_id.0 - i) {
                return true
            }
        }
        false
    }

    #[cfg(feature = "replay")]
    pub fn into_external(mut self) -> ExternalTexture {
        let ext = ExternalTexture {
            id: self.id,
            target: self.target,
        };
        self.id = 0; // don't complain, moved out
        ext
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        debug_assert!(thread::panicking() || self.id == 0);
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(IdType);

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
    Stream,
}

#[cfg(not(feature = "gfx"))]
impl VertexUsageHint {
    fn to_gl(&self) -> GLuint {
        match *self {
            VertexUsageHint::Static => STATIC_DRAW,
            VertexUsageHint::Dynamic => DYNAMIC_DRAW,
            VertexUsageHint::Stream => STREAM_DRAW,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error message
    Link(String, String),        // name, error message
}
