/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::ImageFormat;
use internal_types::FastHashMap;
use shader_source;
use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::ops::Add;
use std::path::PathBuf;
use std::rc::Rc;
use std::slice;
use std::sync::Arc;


#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct FrameId(pub(crate) usize);

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

const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_IMPORT: &str = "#include ";

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

/// Plain old data that can be used to initialize a texture.
pub unsafe trait Texel: Copy {}
unsafe impl Texel for u8 {}
unsafe impl Texel for f32 {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReadPixelsFormat {
    Standard(ImageFormat),
    Rgba8,
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

pub trait FileWatcherHandler: Send {
    fn file_changed(&self, path: PathBuf);
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramSources {
    renderer_name: String,
    pub(crate) vs_source: String,
    pub(crate) fs_source: String,
}

impl ProgramSources {
    pub(crate) fn new(renderer_name: String, vs_source: String, fs_source: String) -> Self {
        ProgramSources {
            renderer_name,
            vs_source,
            fs_source,
        }
    }
}

#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramBinary {
    pub(crate) binary: Vec<u8>,
    pub(crate) format: u32,//gl::GLenum,
    #[cfg(feature = "serialize_program")]
    pub(crate) sources: ProgramSources,
}

impl ProgramBinary {
    #[allow(unused_variables)]
    pub(crate) fn new(binary: Vec<u8>,
           format: u32,//gl::GLenum,
           sources: &ProgramSources) -> Self {
        ProgramBinary {
            binary,
            format,
            #[cfg(feature = "serialize_program")]
            sources: sources.clone(),
        }
    }
}

/// The interfaces that an application can implement to handle ProgramCache update
pub trait ProgramCacheObserver {
    fn notify_binary_added(&self, program_binary: &Arc<ProgramBinary>);
    fn notify_program_binary_failed(&self, program_binary: &Arc<ProgramBinary>);
}

pub struct ProgramCache {
    pub(crate) binaries: RefCell<FastHashMap<ProgramSources, Arc<ProgramBinary>>>,

    /// Optional trait object that allows the client
    /// application to handle ProgramCache updating
    /// application to handle ProgramCache updating
    pub(crate) program_cache_handler: Option<Box<ProgramCacheObserver>>,
}

impl ProgramCache {
    pub fn new(program_cache_observer: Option<Box<ProgramCacheObserver>>) -> Rc<Self> {
        Rc::new(
            ProgramCache {
                binaries: RefCell::new(FastHashMap::default()),
                program_cache_handler: program_cache_observer,
            }
        )
    }
    /// Load ProgramBinary to ProgramCache.
    /// The function is typically used to load ProgramBinary from disk.
    #[cfg(feature = "serialize_program")]
    pub fn load_program_binary(&self, program_binary: Arc<ProgramBinary>) {
        let sources = program_binary.sources.clone();
        self.binaries.borrow_mut().insert(sources, program_binary);
    }
}

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
    Stream,
}

#[cfg(feature = "debug_renderer")]
pub struct Capabilities {
    pub supports_multisampling: bool,
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error message
    Link(String, String),        // name, error message
}

pub(crate) fn texels_to_u8_slice<T: Texel>(texels: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(texels.as_ptr() as *const u8, texels.len() * mem::size_of::<T>())
    }
}