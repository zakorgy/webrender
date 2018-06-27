/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source;
use api::{DeviceUintSize, ImageFormat, TextureTarget};
use euclid::Transform3D;
#[cfg(feature = "gleam")]
use gleam::gl::GLuint;
use internal_types::{ORTHO_FAR_PLANE, ORTHO_NEAR_PLANE, RenderTargetInfo};
#[cfg(not(feature = "gleam"))]
use std::cell::Cell;
use std::io::Read;
use std::fs::File;
use std::ops::Add;
use std::path::PathBuf;
use std::thread;

cfg_if! {
    if #[cfg(feature = "gleam")] {
        mod gl;
        pub use self::gl::*;
    } else {
        mod gfx;
        pub use self::gfx::*;
    }
}

const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_IMPORT: &str = "#include ";

#[cfg(not(feature = "gleam"))]
type IdType = u32;

#[cfg(feature = "gleam")]
type IdType = GLuint;

/// Plain old data that can be used to initialize a texture.
pub unsafe trait Texel: Copy {}
unsafe impl Texel for u8 {}
unsafe impl Texel for f32 {}

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
    #[cfg(all(not(feature = "gleam"), feature = "debug_renderer"))]
    DebugColor,
    #[cfg(all(not(feature = "gleam"), feature = "debug_renderer"))]
    DebugFont,
}

#[derive(Debug, Copy, Clone)]
pub enum VertexArrayKind {
    Primitive,
    Blur,
    Clip,
    Border,
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    VectorStencil,
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    VectorCover,
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(IdType);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(IdType);

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

/// Method of uploading texel data from CPU to GPU.
#[derive(Debug, Clone)]
pub enum UploadMethod {
    /// Just call `glTexSubImage` directly with the CPU data pointer
    Immediate,
    /// Accumulate the changes in PBO first before transferring to a texture.
    PixelBuffer(VertexUsageHint),
}

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
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
    target: IdType,
}


impl ExternalTexture {
    pub fn new(id: u32, target: TextureTarget) -> Self {
        ExternalTexture {
            id,
            #[cfg(feature = "gleam")]
            target: get_gl_target(target),
            #[cfg(not(feature = "gleam"))]
            target: target as _,
        }
    }

    #[cfg(feature = "replay")]
    pub fn internal_id(&self) -> IdType {
        self.id
    }
}

pub struct Texture {
    id: IdType,
    #[cfg_attr(not(feature = "gleam"), allow(dead_code))]
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
    #[cfg(not(feature = "gleam"))]
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

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
    Stream,
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error message
    Link(String, String),        // name, error message
}

// Get a shader string by name, from the built in resources or
// an override path, if supplied.
fn get_shader_source(shader_name: &str, base_path: &Option<PathBuf>) -> Option<String> {
    if let Some(ref base) = *base_path {
        let shader_path = base.join(&format!("{}.glsl", shader_name));
        if shader_path.exists() {
            let mut source = String::new();
            File::open(&shader_path)
                .unwrap()
                .read_to_string(&mut source)
                .unwrap();
            return Some(source);
        }
    }

    shader_source::SHADERS
        .get(shader_name)
        .map(|s| s.to_string())
}

// Parse a shader string for imports. Imports are recursively processed, and
// prepended to the list of outputs.
fn parse_shader_source(source: String, base_path: &Option<PathBuf>, output: &mut String) {
    for line in source.lines() {
        if line.starts_with(SHADER_IMPORT) {
            let imports = line[SHADER_IMPORT.len() ..].split(',');

            // For each import, get the source, and recurse.
            for import in imports {
                if let Some(include) = get_shader_source(import, base_path) {
                    parse_shader_source(include, base_path, output);
                }
            }
        } else {
            output.push_str(line);
            output.push_str("\n");
        }
    }
}

pub fn build_shader_strings(
    gl_version_string: &str,
    features: &str,
    base_filename: &str,
    override_path: &Option<PathBuf>,
) -> (String, String) {
    // Construct a list of strings to be passed to the shader compiler.
    let mut vs_source = String::new();
    let mut fs_source = String::new();

    // GLSL requires that the version number comes first.
    vs_source.push_str(gl_version_string);
    fs_source.push_str(gl_version_string);

    // Insert the shader name to make debugging easier.
    let name_string = format!("// {}\n", base_filename);
    vs_source.push_str(&name_string);
    fs_source.push_str(&name_string);

    // Define a constant depending on whether we are compiling VS or FS.
    vs_source.push_str(SHADER_KIND_VERTEX);
    fs_source.push_str(SHADER_KIND_FRAGMENT);

    // Add any defines that were passed by the caller.
    vs_source.push_str(features);
    fs_source.push_str(features);

    // Parse the main .glsl file, including any imports
    // and append them to the list of sources.
    let mut shared_result = String::new();
    if let Some(shared_source) = get_shader_source(base_filename, override_path) {
        parse_shader_source(shared_source, override_path, &mut shared_result);
    }

    vs_source.push_str(&shared_result);
    fs_source.push_str(&shared_result);

    (vs_source, fs_source)
}

pub(crate) fn create_projection(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    main_frame_buffer: bool
) -> Transform3D<f32> {
    let projection = Transform3D::ortho(
        left,
        right,
        bottom,
        top,
        ORTHO_NEAR_PLANE,
        ORTHO_FAR_PLANE,
    );
    if main_frame_buffer && cfg!(not(feature = "gleam")) {
        return projection.post_scale(1.0, -1.0, 1.0);
    }
    projection
}
