/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ImageFormat};
use api::{DeviceIntRect, DeviceUintPoint, DeviceUintRect, DeviceUintSize};
use api::TextureTarget;
#[cfg(any(feature = "debug_renderer", feature="capture"))]
use api::ImageDescriptor;
use euclid::Transform3D;
//use gleam::gl;
use gpu_types;
use internal_types::{FastHashMap, RenderTargetInfo};
use rand::{self, Rng};
use std::cell::Cell;
use std::cmp;
use std::collections::HashMap;
use std::fs::File;
use std::mem;
use std::ops::Add;
use std::path::PathBuf;
use std::rc::Rc;
use std::slice;
use std::sync::Arc;
use std::thread;
use vertex_types::*;

use hal;

// gfx-hal
use hal::pso::{AttributeDesc, DescriptorRangeDesc, DescriptorSetLayoutBinding, VertexBufferDesc};
use hal::pso::{BlendState, BlendOp, Comparison, DepthTest, Factor};
use hal::{Device as BackendDevice, PhysicalDevice, QueueFamily, Surface, Swapchain};
use hal::{Backbuffer, DescriptorPool, FrameSync, Primitive, SwapchainConfig};
use hal::pass::Subpass;
use hal::pso::PipelineStage;
use hal::queue::Submission;

pub const MAX_INSTANCE_COUNT: usize = 1024;
const MAX_DEBUG_COLOR_INDEX_COUNT: usize = 14544;
const MAX_DEBUG_FONT_INDEX_COUNT: usize = 4296;
const MAX_DEBUG_COLOR_VERTEX_COUNT: usize = 9696;
const MAX_DEBUG_FONT_VERTEX_COUNT: usize = 2864;

pub type TextureId = u32;

pub const INVALID_TEXTURE_ID: TextureId = 0;
pub const INVALID_PROGRAM_ID: ProgramId = ProgramId(0);
pub const DEFAULT_READ_FBO: FBOId = FBOId(0);
pub const DEFAULT_DRAW_FBO: FBOId = FBOId(1);

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::COLOR,
    levels: 0 .. 1,
    layers: 0 .. 1,
};
const DEPTH_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::DEPTH,
    levels: 0 .. 1,
    layers: 0 .. 1,
};

const ENTRY_NAME: &str = "main";

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct Locals {
    uTransform: [[f32; 4]; 4],
    uDevicePixelRatio: f32,
    uMode: i32,
}

#[derive(Clone, Deserialize)]
pub struct PipelineRequirements {
    pub attribute_descriptors: Vec<AttributeDesc>,
    pub bindings_map: HashMap<String, u32>,
    pub descriptor_range_descriptors: Vec<DescriptorRangeDesc>,
    pub descriptor_set_layouts: Vec<DescriptorSetLayoutBinding>,
    pub vertex_buffer_descriptors: Vec<VertexBufferDesc>,
}


const QUAD: [Vertex; 6] = [
    Vertex {
        aPosition: [0.0, 0.0, 0.0],
    },
    Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    Vertex {
        aPosition: [1.0, 1.0, 0.0],
    },
];

fn get_shader_source(filename: &str, extension: &str) -> Vec<u8> {
    use std::io::Read;
    let path_str = format!("{}/{}{}", env!("OUT_DIR"), filename, extension);
    let mut file = File::open(path_str).expect(&format!("Unable to open shader file: {}", filename));
    let mut shader = Vec::new();
    file.read_to_end(&mut shader).unwrap();
    shader
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

/*const GL_FORMAT_BGRA_GL: gl::GLuint = gl::BGRA;

const GL_FORMAT_BGRA_GLES: gl::GLuint = gl::BGRA_EXT;

const SHADER_VERSION_GL: &str = "#version 150\n";
const SHADER_VERSION_GLES: &str = "#version 300 es\n";

const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_IMPORT: &str = "#include ";*/

pub struct TextureSlot(pub usize);

// In some places we need to temporarily bind a texture to any slot.
//const DEFAULT_TEXTURE: TextureSlot = TextureSlot(0);

#[repr(u32)]
pub enum DepthFunction {
    #[cfg(feature = "debug_renderer")]
    Less,
    LessEqual,
}

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

pub trait PrimitiveType {
    type Primitive: Clone + Copy;
    fn to_primitive_type(&self) -> Self::Primitive;
}


impl PrimitiveType for gpu_types::BlurInstance {
    type Primitive = BlurInstance;
    fn to_primitive_type(&self) -> BlurInstance {
        BlurInstance {
            aData0: [0,0,0,0],
            aData1: [0,0,0,0],
            aBlurRenderTaskAddress: self.task_address.0 as i32,
            aBlurSourceTaskAddress: self.src_task_address.0 as i32,
            aBlurDirection: self.blur_direction as i32,
        }
    }
}

impl PrimitiveType for gpu_types::BorderInstance {
    type Primitive = BorderInstance;
    fn to_primitive_type(&self) -> BorderInstance {
        BorderInstance {
            aData0: [0,0,0,0],
            aData1: [0,0,0,0],
            aTaskOrigin: [self.task_origin.x, self.task_origin.y],
            aRect: [self.local_rect.origin.x, self.local_rect.origin.y, self.local_rect.size.width, self.local_rect.size.height],
            aColor0: self.color0.to_array(),
            aColor1: self.color1.to_array(),
            aFlags: self.flags,
            aWidths: [self.widths.width, self.widths.height],
            aRadii: [self.radius.width, self.radius.height],
        }
    }
}

impl PrimitiveType for gpu_types::ClipMaskInstance {
    type Primitive = ClipMaskInstance;
    fn to_primitive_type(&self) -> ClipMaskInstance {
        ClipMaskInstance {
            aClipRenderTaskAddress: self.render_task_address.0 as i32,
            aScrollNodeId: self.scroll_node_data_index.0 as i32,
            aClipSegment: self.segment,
            aClipDataResourceAddress: [
                self.clip_data_address.u as i32,
                self.clip_data_address.v as i32,
                self.resource_address.u as i32,
                self.resource_address.v as i32,
            ],
        }
    }
}

impl PrimitiveType for gpu_types::ClipMaskBorderCornerDotDash {
    type Primitive = ClipMaskBorderCornerDotDash;
    fn to_primitive_type(&self) -> ClipMaskBorderCornerDotDash {
        ClipMaskBorderCornerDotDash {
            aClipRenderTaskAddress: self.clip_mask_instance.render_task_address.0 as i32,
            aScrollNodeId: self.clip_mask_instance.scroll_node_data_index.0 as i32,
            aClipSegment: self.clip_mask_instance.segment,
            aClipDataResourceAddress: [
                self.clip_mask_instance.clip_data_address.u as i32,
                self.clip_mask_instance.clip_data_address.v as i32,
                self.clip_mask_instance.resource_address.u as i32,
                self.clip_mask_instance.resource_address.v as i32,
            ],
            aDashOrDot0: [
                self.dot_dash_data[0],
                self.dot_dash_data[1],
                self.dot_dash_data[2],
                self.dot_dash_data[3],
            ],
            aDashOrDot1: [
                self.dot_dash_data[4],
                self.dot_dash_data[5],
                self.dot_dash_data[6],
                self.dot_dash_data[7],
            ]
        }
    }
}

impl PrimitiveType for gpu_types::PrimitiveInstance {
    type Primitive = PrimitiveInstance;
    fn to_primitive_type(&self) -> PrimitiveInstance {
        PrimitiveInstance {
            aData0: [
                self.data[0],
                self.data[1],
                self.data[2],
                self.data[3],
            ],
            aData1: [
                self.data[4],
                self.data[5],
                self.data[6],
                self.data[7],
            ],
        }
    }
}

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
    PixelBuffer,
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
    id: u32,
    _target: TextureTarget,
}

impl ExternalTexture {
    pub fn new(id: u32, _target: TextureTarget) -> Self {
        ExternalTexture {
            id,
            _target,
        }
    }

    #[cfg(feature = "replay")]
    pub fn internal_id(&self) -> u32 {
        self.id
    }
}

pub struct Texture {
    id: TextureId,
    _target: TextureTarget,
    layer_count: i32,
    format: ImageFormat,
    width: u32,
    height: u32,
    filter: TextureFilter,
    render_target: Option<RenderTargetInfo>,
    fbo_ids: Vec<FBOId>,
    depth_rb: Option<RBOId>,
    last_frame_used: FrameId,
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
            _target: self._target,
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
pub struct ProgramId(u32);

pub struct PBO;
pub struct VAO;

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(u32);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(u32);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(u32);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(u32);

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramSources;

#[cfg_attr(feature = "serialize_program", derive(Deserialize, Serialize))]
pub struct ProgramBinary;

pub trait ProgramCacheObserver {
    fn notify_binary_added(&self, program_binary: &Arc<ProgramBinary>);
    fn notify_program_binary_failed(&self, program_binary: &Arc<ProgramBinary>);
}

pub struct ProgramCache;

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

bitflags!(
    pub struct ApiCapabilities: u8 {
        const BLITTING = 0x1;
    }
);

#[derive(Debug, Copy, Clone)]
pub(crate) enum VertexArrayKind {
    Primitive,
    Blur,
    Clip,
    DashAndDot,
    VectorStencil,
    VectorCover,
    Border,
}

pub(crate) enum ShaderKind {
    Primitive,
    Cache(VertexArrayKind),
    ClipCache,
    Brush,
    Text,
    #[allow(dead_code)]
    VectorStencil,
    #[allow(dead_code)]
    VectorCover,
    DebugColor,
    DebugFont,
}

impl ShaderKind {
    fn is_debug(&self) -> bool {
        match *self {
            ShaderKind::DebugFont | ShaderKind::DebugColor => true,
            _ => false,
        }
    }
}

const ALPHA: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::SrcAlpha,
        dst: Factor::OneMinusSrcAlpha,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

const PREMULTIPLIED_DEST_OUT: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcAlpha,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcAlpha,
    },
};

const MAX: BlendState = BlendState::On {
    color: BlendOp::Max,
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

#[cfg(feature = "debug_renderer")]
const MIN: BlendState = BlendState::On {
    color: BlendOp::Min,
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

const SUBPIXEL_PASS0: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcColor,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcAlpha,
    },
};

const SUBPIXEL_PASS1: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
};

const SUBPIXEL_WITH_BG_COLOR_PASS0: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::OneMinusSrcColor,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::One,
    },
};

const SUBPIXEL_WITH_BG_COLOR_PASS1: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::OneMinusDstAlpha,
        dst: Factor::One,
    },
    alpha: BlendOp::Add {
        src: Factor::Zero,
        dst: Factor::One,
    },
};

const SUBPIXEL_WITH_BG_COLOR_PASS2: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::One,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrcAlpha,
    },
};

// This requires blend color to be set
const SUBPIXEL_CONSTANT_TEXT_COLOR: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::ConstColor,
        dst: Factor::OneMinusSrcColor,
    },
    alpha: BlendOp::Add {
        src: Factor::ConstAlpha,
        dst: Factor::OneMinusSrcAlpha,
    },
};

const SUBPIXEL_DUAL_SOURCE: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrc1Color,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrc1Alpha,
    },
};

const LESS_EQUAL_TEST: DepthTest = DepthTest::On {
    fun: Comparison::LessEqual,
    write: false,
};

const LESS_EQUAL_WRITE: DepthTest = DepthTest::On {
    fun: Comparison::LessEqual,
    write: true,
};

pub struct ImageBuffer<B: hal::Backend> {
    pub buffer: CopyBuffer<B>,
    pub offset: u64,
}

impl<B: hal::Backend> ImageBuffer<B> {
    fn new(buffer: CopyBuffer<B>) -> ImageBuffer<B> {
        ImageBuffer {
            buffer,
            offset: 0,
        }
    }

    pub fn update(&mut self, device: &B::Device, data: &[u8], offset_alignment: usize) -> usize {
        self.buffer
            .update(device, self.offset, data.len() as u64, data, offset_alignment)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        self.buffer.deinit(device);
    }
}

pub struct ImageCore<B: hal::Backend> {
    pub image: B::Image,
    pub memory: Option<B::Memory>,
    pub view: B::ImageView,
    pub subresource_range: hal::image::SubresourceRange,
    pub state: Cell<hal::image::State>,
}

impl<B: hal::Backend> ImageCore<B> {
    fn from_image(
        device: &B::Device,
        image: B::Image,
        view_kind: hal::image::ViewKind,
        format: hal::format::Format,
        subresource_range: hal::image::SubresourceRange,
    ) -> Self {
        let view = device.create_image_view(
            &image,
            view_kind,
            format,
            hal::format::Swizzle::NO,
            subresource_range.clone(),
        ).unwrap();
        ImageCore {
            image,
            memory: None,
            view,
            subresource_range,
            state: Cell::new((hal::image::Access::empty(), hal::image::Layout::Undefined)),
        }
    }

    fn create(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        kind: hal::image::Kind,
        view_kind: hal::image::ViewKind,
        mip_levels: hal::image::Level,
        format: hal::format::Format,
        usage: hal::image::Usage,
        subresource_range: hal::image::SubresourceRange,
    ) -> Self {
        let image_unbound = device
            .create_image(kind, mip_levels, format, hal::image::Tiling::Optimal, usage, hal::image::StorageFlags::empty())
            .unwrap();
        let requirements = device.get_image_requirements(&image_unbound);

        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                requirements.type_mask & (1 << id) != 0 &&
                    mem_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let memory = device.allocate_memory(mem_type, requirements.size).unwrap();
        let image = device.bind_image_memory(&memory, 0, image_unbound).unwrap();

        ImageCore {
            memory: Some(memory),
            .. Self::from_image(device, image, view_kind, format, subresource_range)
        }
    }

    fn _reset(&self) {
        self.state.set((hal::image::Access::empty(), hal::image::Layout::Undefined));
    }

    fn deinit(self, device: &B::Device) {
        device.destroy_image_view(self.view);
        if let Some(memory) = self.memory {
            device.destroy_image(self.image);
            device.free_memory(memory);
        }
    }

    fn transit(
        &self,
        access: hal::image::Access,
        layout: hal::image::Layout,
        range: hal::image::SubresourceRange,
    ) -> Option<hal::memory::Barrier<B>> {
        let src_state = self.state.get();
        if src_state == (access, layout) {
            None
        } else {
            self.state.set((access, layout));
            Some(hal::memory::Barrier::Image {
                states: src_state .. (access, layout),
                target: &self.image,
                range,
            })
        }
    }
}

pub struct Image<B: hal::Backend> {
    pub core: ImageCore<B>,
    pub upload_buffer: ImageBuffer<B>,
    pub kind: hal::image::Kind,
    pub format: ImageFormat,
}

impl<B: hal::Backend> Image<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        image_format: ImageFormat,
        image_width: u32,
        image_height: u32,
        image_depth: i32,
        view_kind: hal::image::ViewKind,
        mip_levels: hal::image::Level,
        pitch_alignment: usize,
    ) -> Self {
        let format = match image_format {
            ImageFormat::R8 => hal::format::Format::R8Unorm,
            ImageFormat::RG8 => hal::format::Format::Rg8Unorm,
            ImageFormat::BGRA8 => hal::format::Format::Bgra8Unorm,
            ImageFormat::RGBAF32 => hal::format::Format::Rgba32Float,
        };
        let upload_buffer = CopyBuffer::create(
            device,
            memory_types,
            hal::buffer::Usage::TRANSFER_SRC,
            1, // Data stride is 1, because we receive image data as [u8].
            (image_width * image_format.bytes_per_pixel()) as usize,
            image_height as usize,
            pitch_alignment,
        );

        let kind = hal::image::Kind::D2(
            image_width as _,
            image_height as _,
            image_depth as _,
            1,
        );
        let core = ImageCore::create(
            device,
            memory_types,
            kind,
            view_kind,
            mip_levels,
            format,
            hal::image::Usage::TRANSFER_SRC | hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED | hal::image::Usage::COLOR_ATTACHMENT,
            hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. mip_levels,
                layers: 0 .. image_depth as _,
            },
        );

        Image {
            core,
            upload_buffer: ImageBuffer::new(upload_buffer),
            kind,
            format: image_format,
        }
    }

    pub fn update(
        &mut self,
        device: &mut B::Device,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        rect: DeviceUintRect,
        layer_index: i32,
        image_data: &[u8],
        offset_alignment: usize,
    ) -> hal::command::Submit<B, hal::Graphics, hal::command::MultiShot, hal::command::Primary>
    {
        //let (image_width, image_height, _, _) = self.kind.dimensions();
        let pos = rect.origin;
        let size = rect.size;
        let offset = self.upload_buffer.update(device, image_data, offset_alignment);
        let mut cmd_buffer = cmd_pool.acquire_command_buffer(false);

        let range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: layer_index as _ .. (layer_index + 1) as _,
        };
        if let Some(barrier) = self.core.transit(
            hal::image::Access::TRANSFER_WRITE,
            hal::image::Layout::TransferDstOptimal,
            range.clone(),
        ) {
            cmd_buffer.pipeline_barrier(
                PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        let buffer_width = if self.kind.extent().width == size.width {
            self.upload_buffer.buffer.row_pitch() as u32 / self.format.bytes_per_pixel()
        } else {
            size.width
        };
        cmd_buffer.copy_buffer_to_image(
            &self.upload_buffer.buffer.buffer,
            &self.core.image,
            hal::image::Layout::TransferDstOptimal,
            &[
                hal::command::BufferImageCopy {
                    buffer_offset: self.upload_buffer.offset,
                    buffer_width,
                    buffer_height: size.height,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: layer_index as _ .. (layer_index + 1) as _,
                    },
                    image_offset: hal::image::Offset {
                        x: pos.x as i32,
                        y: pos.y as i32,
                        z: 0,
                    },
                    image_extent: hal::image::Extent {
                        width: size.width as u32,
                        height: size.height as u32,
                        depth: 1,
                    },
                },
            ],
        );

        if let Some(barrier) = self.core.transit(
            hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
            hal::image::Layout::ColorAttachmentOptimal,
            range,
        ) {
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        self.upload_buffer.offset += offset as u64;
        cmd_buffer.finish()
    }

    pub fn deinit(self, device: &B::Device) {
        self.core.deinit(device);
        self.upload_buffer.deinit(device);
    }
}

pub struct Buffer<B: hal::Backend> {
    pub memory: B::Memory,
    pub buffer: B::Buffer,
    pub data_stride: usize,
    pub data_len: usize,
    _state: Cell<hal::buffer::State>,
}

impl<B: hal::Backend> Buffer<B> {
    pub fn create(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        usage: hal::buffer::Usage,
        data_stride: usize,
        data_len: usize,
    ) -> Self {
        let buffer_size = data_stride * data_len;
        let unbound_buffer = device.create_buffer(buffer_size as u64, usage).unwrap();
        let requirements = device.get_buffer_requirements(&unbound_buffer);
        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mt)| {
                requirements.type_mask & (1 << id) != 0 &&
                    mt.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                //&&!mt.properties.contains(memory::Properties::CPU_CACHED)
            })
            .unwrap()
            .into();
        let memory = device
            .allocate_memory(mem_type, requirements.size)
            .unwrap();
        let buffer = device
            .bind_buffer_memory(&memory, 0, unbound_buffer)
            .unwrap();
        Buffer {
            memory,
            buffer,
            data_stride,
            data_len,
            _state: Cell::new(hal::buffer::Access::empty())
        }
    }

    pub fn update<T>(
        &mut self,
        device: &B::Device,
        buffer_offset: u64,
        buffer_width: u64,
        update_data: &[T],
    ) where
        T: Copy,
    {
        //TODO:
        //assert!(self.state.get().contains(hal::buffer::Access::HOST_WRITE));
        let mut data = device
            .acquire_mapping_writer::<T>(
                &self.memory,
                buffer_offset .. (buffer_offset + buffer_width),
            )
            .unwrap();
        assert_eq!(data.len(), update_data.len());
        for (i, d) in update_data.iter().enumerate() {
            data[i] = *d;
        }
        device.release_mapping_writer(data);
    }

    fn _transit(
        &self,
        access: hal::buffer::Access,
    ) -> Option<hal::memory::Barrier<B>> {
        let src_state = self._state.get();
        if src_state == access {
            None
        } else {
            self._state.set(access);
            Some(hal::memory::Barrier::Buffer {
                states: src_state .. access,
                target: &self.buffer,
            })
        }
    }

    pub fn deinit(self, device: &B::Device) {
        device.destroy_buffer(self.buffer);
        device.free_memory(self.memory);
    }
}

pub struct CopyBuffer<B: hal::Backend> {
    pub memory: B::Memory,
    pub buffer: B::Buffer,
    pub data_stride: usize,
    pub data_width: usize,
    pub data_height: usize,
    pub row_pitch_in_bytes: usize,
    state: Cell<hal::buffer::State>,
}

impl<B: hal::Backend> CopyBuffer<B> {
    pub fn create(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        usage: hal::buffer::Usage,
        data_stride: usize,
        data_width: usize, // without stride
        data_height: usize,
        pitch_alignment: usize,
    ) -> Self {
        let mut row_pitch_in_bytes = (data_width * data_stride + pitch_alignment) & !pitch_alignment;
        if row_pitch_in_bytes % data_width > 0 {
            row_pitch_in_bytes = ((data_width + pitch_alignment) & !pitch_alignment) * data_stride;
        }
        let buffer_size = row_pitch_in_bytes * data_height;
        let unbound_buffer = device.create_buffer(buffer_size as u64, usage).unwrap();
        let requirements = device.get_buffer_requirements(&unbound_buffer);
        let mem_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mt)| {
                requirements.type_mask & (1 << id) != 0 &&
                    mt.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                //&&!mt.properties.contains(memory::Properties::CPU_CACHED)
            })
            .unwrap()
            .into();
        let memory = device
            .allocate_memory(mem_type, requirements.size)
            .unwrap();
        let buffer = device
            .bind_buffer_memory(&memory, 0, unbound_buffer)
            .unwrap();
        CopyBuffer {
            memory,
            buffer,
            data_stride,
            data_width,
            data_height,
            row_pitch_in_bytes,
            state: Cell::new(hal::buffer::Access::empty())
        }
    }

    pub fn update<T>(
        &mut self,
        device: &B::Device,
        buffer_offset: u64,
        image_data_width_in_bytes: u64,
        image_data: &[T],
        offset_alignment: usize,
    ) -> usize
        where
            T: Copy,
    {
        //assert!(self.state.get().contains(hal::buffer::Access::HOST_WRITE));
        let buffer_data_width_in_bytes = self.data_width * self.data_stride;
        let mut needed_height = image_data_width_in_bytes / buffer_data_width_in_bytes as u64;
        let last_row_length = image_data_width_in_bytes % buffer_data_width_in_bytes as u64;
        let range = if last_row_length != 0 {
            needed_height += 1;
            buffer_offset ..
                buffer_offset + ((needed_height - 1) as usize * self.row_pitch_in_bytes) as u64 + last_row_length
        } else {
            buffer_offset .. (buffer_offset + (needed_height as usize * self.row_pitch_in_bytes) as u64)
        };

        let mut data = device
            .acquire_mapping_writer::<T>(
                &self.memory,
                range,
            )
            .unwrap();

        for y in 0 .. needed_height as usize {
            let lower_bound = y * self.data_width;
            let upper_bound = cmp::min(lower_bound + self.data_width, image_data.len());
            let row = &(*image_data)[lower_bound .. upper_bound];
            let dest_base = y * self.row_pitch();
            data[dest_base .. dest_base + row.len()].copy_from_slice(row);
        }
        device.release_mapping_writer(data);
        (needed_height as usize * self.row_pitch() + offset_alignment) & !offset_alignment
    }

    fn row_pitch(&self) -> usize {
        assert_eq!(self.row_pitch_in_bytes % self.data_stride, 0);
        self.row_pitch_in_bytes / self.data_stride
    }

    fn transit(
        &self,
        access: hal::buffer::Access,
    ) -> Option<hal::memory::Barrier<B>> {
        let src_state = self.state.get();
        if src_state == access {
            None
        } else {
            self.state.set(access);
            Some(hal::memory::Barrier::Buffer {
                states: src_state .. access,
                target: &self.buffer,
            })
        }
    }

    pub fn deinit(self, device: &B::Device) {
        device.destroy_buffer(self.buffer);
        device.free_memory(self.memory);
    }
}

pub struct InstanceBuffer<B: hal::Backend> {
    pub buffer: Buffer<B>,
    pub size: usize,
    pub offset: usize,
}

impl<B: hal::Backend> InstanceBuffer<B> {
    fn new(buffer: Buffer<B>) -> Self {
        InstanceBuffer {
            buffer,
            size: 0,
            offset: 0,
        }
    }

    fn update<T>(
        &mut self,
        device: &B::Device,
        instances: &[T],
    )where
        T: Copy,
    {
        let data_stride = self.buffer.data_stride;
        self.buffer.update(
            device,
            (self.offset * data_stride) as u64,
            (instances.len() * data_stride) as u64,
            &instances.to_owned(),
        );

        self.size = instances.len();
        self.offset += self.size;
    }

    pub fn reset(&mut self) {
        self.size = 0;
        self.offset = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        self.buffer.deinit(device);
    }
}

pub struct UniformBuffer<B: hal::Backend> {
    pub buffers: Vec<Buffer<B>>,
    pub stride: usize,
    pub memory_types: Vec<hal::MemoryType>,
    pub size: usize,
}

impl<B: hal::Backend> UniformBuffer<B> {
    fn new(stride: usize, memory_types: Vec<hal::MemoryType>) -> UniformBuffer<B> {
        UniformBuffer {
            buffers: vec![],
            stride,
            memory_types,
            size: 0,
        }
    }

    fn add<T>(
        &mut self,
        device: &B::Device,
        instances: &[T],
    ) where T: Copy,
    {
        if self.buffers.len() == self.size {
            let buffer = Buffer::create(
                device,
                &self.memory_types,
                hal::buffer::Usage::UNIFORM,
                self.stride,
                1,
            );
            self.buffers.push(buffer);
        }
        self.buffers[self.size].update(
            device,
            0 as u64,
            (instances.len() * self.stride) as u64,
            &instances.to_owned(),
        );
        self.size += 1;
    }

    pub fn _reset(&mut self) {
        self.size = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        for buffer in self.buffers {
            buffer.deinit(device);
        }
    }
}

pub(crate) struct Program<B: hal::Backend> {
    pub bindings_map: HashMap<String, u32>,
    pub descriptor_set_layout: B::DescriptorSetLayout,
    pub descriptor_pool: B::DescriptorPool,
    pub descriptor_set: B::DescriptorSet,
    pub pipeline_layout: B::PipelineLayout,
    pub pipelines: HashMap<(BlendState, DepthTest), B::GraphicsPipeline>,
    pub vertex_buffer: Buffer<B>,
    pub index_buffer: Option<Buffer<B>>,
    pub instance_buffer: InstanceBuffer<B>,
    pub locals_buffer: UniformBuffer<B>,
    shader_name: String,
}

impl<B: hal::Backend> Program<B> {
    pub fn create(
        pipeline_requirements: PipelineRequirements,
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        shader_name: &str,
        shader_kind: &ShaderKind,
        render_pass: &RenderPass<B>,
    ) -> Program<B> {
        let vs_module = device
            .create_shader_module(get_shader_source(shader_name, ".vert.spv").as_slice())
            .unwrap();
        let fs_module = device
            .create_shader_module(get_shader_source(shader_name, ".frag.spv").as_slice())
            .unwrap();

        let descriptor_set_layout = device.create_descriptor_set_layout(&pipeline_requirements.descriptor_set_layouts);
        let mut descriptor_pool =
            device.create_descriptor_pool(
                1, //The number of descriptor sets
                pipeline_requirements.descriptor_range_descriptors.as_slice(),
            );
        let descriptor_set =
            descriptor_pool.allocate_set(&descriptor_set_layout)
                .expect(&format!("Failed to allocate set with layout: {:?}", descriptor_set_layout));

        let pipeline_layout = device.create_pipeline_layout(Some(&descriptor_set_layout), &[]);

        let pipelines = {
            let (vs_entry, fs_entry) = (
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &vs_module,
                    specialization: &[],
                },
                hal::pso::EntryPoint::<B> {
                    entry: ENTRY_NAME,
                    module: &fs_module,
                    specialization: &[],
                },
            );

            let shader_entries = hal::pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let pipeline_states = match *shader_kind {
                ShaderKind::Brush if shader_name.starts_with("brush_mask") => vec![(BlendState::Off, DepthTest::Off)],
                ShaderKind::Cache(VertexArrayKind::Blur) => vec![(BlendState::Off, DepthTest::Off)],
                ShaderKind::Cache(VertexArrayKind::Primitive) => vec![(BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off)],
                ShaderKind::Cache(VertexArrayKind::DashAndDot) => vec![(MAX, DepthTest::Off), (BlendState::Off, DepthTest::Off)],
                ShaderKind::Cache(VertexArrayKind::Border) => vec![(BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off)],
                ShaderKind::ClipCache => {
                    if shader_name.starts_with("cs_clip_border") {
                        vec![
                            (BlendState::Off, DepthTest::Off),
                            (MAX, DepthTest::Off)
                        ]
                    } else {
                        vec![(BlendState::MULTIPLY, DepthTest::Off)]
                    }
                },
                ShaderKind::Text => vec![
                    (BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off),
                    (BlendState::PREMULTIPLIED_ALPHA, LESS_EQUAL_TEST),
                    (SUBPIXEL_DUAL_SOURCE, DepthTest::Off),
                    (SUBPIXEL_DUAL_SOURCE, LESS_EQUAL_TEST),
                    (SUBPIXEL_CONSTANT_TEXT_COLOR, DepthTest::Off),
                    (SUBPIXEL_CONSTANT_TEXT_COLOR, LESS_EQUAL_TEST),
                    (SUBPIXEL_PASS0, DepthTest::Off),
                    (SUBPIXEL_PASS0, LESS_EQUAL_TEST),
                    (SUBPIXEL_PASS1, DepthTest::Off),
                    (SUBPIXEL_PASS1, LESS_EQUAL_TEST),
                    (SUBPIXEL_WITH_BG_COLOR_PASS0, DepthTest::Off),
                    (SUBPIXEL_WITH_BG_COLOR_PASS0, LESS_EQUAL_TEST),
                    (SUBPIXEL_WITH_BG_COLOR_PASS1, DepthTest::Off),
                    (SUBPIXEL_WITH_BG_COLOR_PASS1, LESS_EQUAL_TEST),
                    (SUBPIXEL_WITH_BG_COLOR_PASS2, DepthTest::Off),
                    (SUBPIXEL_WITH_BG_COLOR_PASS2, LESS_EQUAL_TEST),
                ],
                ShaderKind::DebugColor | ShaderKind::DebugFont => vec![
                    (BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off),
                ],
                _ => vec![
                    (BlendState::Off, DepthTest::Off),
                    (BlendState::Off, LESS_EQUAL_TEST),
                    (BlendState::Off, LESS_EQUAL_WRITE),
                    (ALPHA, DepthTest::Off),
                    (ALPHA, LESS_EQUAL_TEST),
                    (ALPHA, LESS_EQUAL_WRITE),
                    (BlendState::PREMULTIPLIED_ALPHA, DepthTest::Off),
                    (BlendState::PREMULTIPLIED_ALPHA, LESS_EQUAL_TEST),
                    (BlendState::PREMULTIPLIED_ALPHA, LESS_EQUAL_WRITE),
                    (PREMULTIPLIED_DEST_OUT, DepthTest::Off),
                    (PREMULTIPLIED_DEST_OUT, LESS_EQUAL_TEST),
                    (PREMULTIPLIED_DEST_OUT, LESS_EQUAL_WRITE),
                ],
            };
            let format = match *shader_kind {
                ShaderKind::ClipCache | ShaderKind::Cache(VertexArrayKind::DashAndDot) => ImageFormat::R8,
                ShaderKind::Cache(VertexArrayKind::Blur) if shader_name.contains("_alpha_target") => ImageFormat::R8,
                _ => ImageFormat::BGRA8,
            };

            let pipelines_descriptors = pipeline_states.iter().map(|&(blend_state, depth_test)| {
                let subpass = Subpass {
                    index: 0,
                    main_pass: render_pass.get_render_pass(format, depth_test != DepthTest::Off),
                };
                let mut pipeline_descriptor = hal::pso::GraphicsPipelineDesc::new(
                    shader_entries.clone(),
                    Primitive::TriangleList,
                    hal::pso::Rasterizer::FILL,
                    &pipeline_layout,
                    subpass,
                );
                pipeline_descriptor
                    .blender
                    .targets
                    .push(hal::pso::ColorBlendDesc(
                        hal::pso::ColorMask::ALL,
                        blend_state,
                    ));

                pipeline_descriptor.depth_stencil = Some(
                    hal::pso::DepthStencilDesc {
                        depth: depth_test,
                        depth_bounds: false,
                        stencil: hal::pso::StencilTest::Off,
                    }
                );

                pipeline_descriptor.vertex_buffers = pipeline_requirements.vertex_buffer_descriptors.clone();
                pipeline_descriptor.attributes = pipeline_requirements.attribute_descriptors.clone();
                pipeline_descriptor
            }).collect::<Vec<_>>();

            let pipelines = device
                .create_graphics_pipelines(pipelines_descriptors.as_slice())
                .into_iter();

            pipeline_states.iter()
                .cloned()
                .zip(pipelines.map(|pipeline| pipeline.unwrap()))
                .collect::<HashMap<(BlendState, DepthTest), B::GraphicsPipeline>>()
        };

        device.destroy_shader_module(vs_module);
        device.destroy_shader_module(fs_module);

        let (vertex_buffer_stride, vertex_buffer_len) = match *shader_kind {
            ShaderKind::DebugColor => {
                let stride = mem::size_of::<DebugColorVertex>();
                (stride, MAX_DEBUG_COLOR_VERTEX_COUNT * stride)
            },
            ShaderKind::DebugFont => {
                let stride = mem::size_of::<DebugFontVertex>();
                (stride, MAX_DEBUG_FONT_VERTEX_COUNT * stride)
            },
            _ => {
                let stride = mem::size_of::<Vertex>();
                (stride, QUAD.len() * stride)
            },
        };

        let mut vertex_buffer = Buffer::create(
            device,
            memory_types,
            hal::buffer::Usage::VERTEX,
            vertex_buffer_stride,
            vertex_buffer_len,
        );

        let mut index_buffer = None;

        if shader_kind.is_debug() {
            let index_buffer_stride = mem::size_of::<u32>();
            let index_buffer_len = match *shader_kind {
                ShaderKind::DebugColor => MAX_DEBUG_COLOR_INDEX_COUNT * index_buffer_stride,
                ShaderKind::DebugFont => MAX_DEBUG_FONT_INDEX_COUNT * index_buffer_stride,
                _ => unreachable!(),
            };

            index_buffer = Some(Buffer::create(
                device,
                memory_types,
                hal::buffer::Usage::INDEX,
                index_buffer_stride,
                index_buffer_len,
            ));
        } else {
            vertex_buffer.update(device, 0, (QUAD.len() * vertex_buffer_stride) as u64, &QUAD);
        }

        let instance_buffer_stride = match *shader_kind {
            ShaderKind::Primitive |
            ShaderKind::Brush |
            ShaderKind::Text |
            ShaderKind::Cache(VertexArrayKind::Primitive) => mem::size_of::<PrimitiveInstance>(),
            ShaderKind::ClipCache | ShaderKind::Cache(VertexArrayKind::Clip) => mem::size_of::<ClipMaskInstance>(),
            ShaderKind::Cache(VertexArrayKind::Blur) => mem::size_of::<BlurInstance>(),
            ShaderKind::Cache(VertexArrayKind::DashAndDot) => mem::size_of::<ClipMaskBorderCornerDotDash>(),
            ShaderKind::Cache(VertexArrayKind::Border) => mem::size_of::<BorderInstance>(),
            ShaderKind::DebugColor | ShaderKind::DebugFont => 1,
            _ => unreachable!()
        };

        let instance_buffer_len = match *shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => 1,
            _ => MAX_INSTANCE_COUNT * instance_buffer_stride,
        };

        let instance_buffer = Buffer::create(
            device,
            memory_types,
            hal::buffer::Usage::VERTEX,
            instance_buffer_stride,
            instance_buffer_len,
        );

        let locals_buffer_stride = mem::size_of::<Locals>();
        let locals_buffer = UniformBuffer::new(locals_buffer_stride, memory_types.to_vec());

        let bindings_map = pipeline_requirements.bindings_map;

        Program {
            bindings_map,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            pipeline_layout,
            pipelines,
            vertex_buffer,
            index_buffer,
            instance_buffer: InstanceBuffer::new(instance_buffer),
            locals_buffer,
            shader_name: String::from(shader_name),
        }
    }


    pub fn bind_instances<T>(
        &mut self,
        device: &B::Device,
        instances: &[T],
    ) where
        T: Copy,
    {
        if !instances.is_empty() {
            self.instance_buffer.update(
                device,
                instances,
            );
        }
    }

    fn bind_locals(
        &mut self,
        device: &B::Device,
        projection: &Transform3D<f32>,
        device_pixel_ratio: f32,
        u_mode: i32,
    ) {
        let locals_data = vec![
            Locals {
                uTransform: projection.to_row_arrays(),
                uDevicePixelRatio: device_pixel_ratio,
                uMode: u_mode,
            },
        ];
        self.locals_buffer.add(
            device,
            &locals_data,
        );
        device.write_descriptor_sets(vec![
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["Locals"],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Buffer(&self.locals_buffer.buffers[self.locals_buffer.size - 1].buffer, Some(0)..Some(self.locals_buffer.stride as u64))
                ),
            },
        ]);
    }

    pub fn bind_texture(&mut self, device: &B::Device, image: &ImageCore<B>, sampler: &B::Sampler, binding: &'static str) {
        if self.bindings_map.contains_key(&("t".to_owned() + binding)) {
            device.write_descriptor_sets(vec![
                hal::pso::DescriptorSetWrite {
                    set: &self.descriptor_set,
                    binding: self.bindings_map[&("t".to_owned() + binding)],
                    array_offset: 0,
                    descriptors: Some(
                        hal::pso::Descriptor::Image(&image.view, image.state.get().1)
                    ),
                },
                hal::pso::DescriptorSetWrite {
                    set: &self.descriptor_set,
                    binding: self.bindings_map[&("s".to_owned() + binding)],
                    array_offset: 0,
                    descriptors: Some(
                        hal::pso::Descriptor::Sampler(sampler)
                    )
                },
            ]);
        }
    }

    /*pub fn bind<T>(
        &mut self,
        device: &Device<B>,
        projection: &Transform3D<f32>,
        instances: &[T],
    ) where
        T: Copy,
    {
        self.bind_instances(&device.device, instances);
        self.bind_locals(&device.device, &projection, device.program_mode_id);
        self.bind_textures(device);
    }*/

    pub fn submit(
        &mut self,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        viewport: hal::pso::Viewport,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        clear_values: &[hal::command::ClearValue],
        blend_state: BlendState,
        blend_color: ColorF,
        depth_test: DepthTest,
        scissor_rect: Option<DeviceIntRect>,
    ) -> hal::command::Submit<B, hal::Graphics, hal::command::MultiShot, hal::command::Primary> {
        let mut cmd_buffer = cmd_pool.acquire_command_buffer(false);

        cmd_buffer.set_viewports(0, &[viewport.clone()]);
        match scissor_rect {
            Some(r) => cmd_buffer.set_scissors(
                0,
                &[hal::pso::Rect {x: r.origin.x as _, y: r.origin.y as _, w: r.size.width as _, h: r.size.height as _}],
            ),
            None => cmd_buffer.set_scissors(0, &[viewport.rect]),
        }
        cmd_buffer.bind_graphics_pipeline(
            &self.pipelines.get(&(blend_state, depth_test)).expect(&format!("The blend state {:?} with depth test {:?} not found for {} program!", blend_state, depth_test, self.shader_name)));
        cmd_buffer.bind_vertex_buffers(
            0,
            hal::pso::VertexBufferSet(vec![
                (&self.vertex_buffer.buffer, 0),
                (&self.instance_buffer.buffer.buffer, 0),
            ]),
        );

        if let Some(ref index_buffer) = self.index_buffer {
            cmd_buffer.bind_index_buffer(
                hal::buffer::IndexBufferView {
                    buffer: &index_buffer.buffer,
                    offset: 0,
                    index_type: hal::IndexType::U32,
                }
            );
        }

        cmd_buffer.bind_graphics_descriptor_sets(
            &self.pipeline_layout,
            0,
            Some(&self.descriptor_set),
        );

        if blend_state == SUBPIXEL_CONSTANT_TEXT_COLOR {
            cmd_buffer.set_blend_constants(blend_color.to_array());
        }

        {
            let mut encoder = cmd_buffer.begin_render_pass_inline(
                render_pass,
                frame_buffer,
                viewport.rect,
                clear_values,
            );

            if let Some(ref index_buffer) = self.index_buffer {
                encoder.draw_indexed(
                    0 .. (index_buffer.data_len / index_buffer.data_stride) as u32,
                    0,
                    0 .. 1,
                );
            } else {
                encoder.draw(
                    0 .. QUAD.len() as _,
                    (self.instance_buffer.offset - self.instance_buffer.size) as u32 .. self.instance_buffer.offset as u32,
                );
            }
        }

        cmd_buffer.finish()
    }

    pub fn deinit(mut self, device: &B::Device) {
        self.vertex_buffer.deinit(device);
        if let Some(index_buffer) = self.index_buffer {
            index_buffer.deinit(device);
        }
        self.instance_buffer.deinit(device);
        self.locals_buffer.deinit(device);
        device.destroy_descriptor_pool(self.descriptor_pool);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout);
        device.destroy_pipeline_layout(self.pipeline_layout);
        for pipeline in self.pipelines.drain() {
            device.destroy_graphics_pipeline(pipeline.1);
        }
    }
}

pub struct Framebuffer<B: hal::Backend> {
    pub texture: TextureId,
    pub layer_index: u16,
    pub format: ImageFormat,
    pub image_view: B::ImageView,
    pub fbo: B::Framebuffer,
    pub rbo: RBOId,
}

impl<B: hal::Backend> Framebuffer<B> {
    pub fn new(
        device: &B::Device,
        texture: &Texture,
        image: &Image<B>,
        layer_index: u16,
        render_pass: &RenderPass<B>,
        rbo: RBOId,
        depth: Option<&B::ImageView>
    ) -> Self {
        let extent = hal::image::Extent {
            width: texture.width as _,
            height: texture.height as _,
            depth: 1,
        };
        let format = match texture.format {
            ImageFormat::R8 => hal::format::Format::R8Unorm,
            ImageFormat::BGRA8 => hal::format::Format::Bgra8Unorm,
            _ => unimplemented!("TODO image format missing"),
        };
        let image_view = device
            .create_image_view(
                &image.core.image,
                hal::image::ViewKind::D2Array,
                format,
                hal::format::Swizzle::NO,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: 0 .. 1,
                    layers: layer_index .. layer_index+1,
                },
            )
            .unwrap();
        let fbo = if rbo != RBOId(0) {
            device
                .create_framebuffer(render_pass.get_render_pass(texture.format, true), vec![&image_view, depth.unwrap()], extent)
                .unwrap()
        } else {
            device
                .create_framebuffer(render_pass.get_render_pass(texture.format, false), Some(&image_view), extent)
                .unwrap()
        };

        Framebuffer {
            texture: texture.id,
            layer_index,
            format: texture.format,
            image_view,
            fbo,
            rbo,
        }
    }

    pub fn deinit(self, device: &B::Device) {
        device.destroy_framebuffer(self.fbo);
        device.destroy_image_view(self.image_view);
    }
}

pub struct DepthBuffer<B: hal::Backend> {
    pub core: ImageCore<B>
}

impl<B: hal::Backend> DepthBuffer<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        pixel_width: u32,
        pixel_height: u32,
        depth_format: hal::format::Format
    ) -> Self {
        let core = ImageCore::create(
            device,
            memory_types,
            hal::image::Kind::D2(pixel_width, pixel_height, 1, 1),
            hal::image::ViewKind::D2,
            1,
            depth_format,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::DEPTH_STENCIL_ATTACHMENT,
            DEPTH_RANGE,
        );
        DepthBuffer {
            core
        }
    }

    pub fn deinit(self, device: &B::Device) {
        self.core.deinit(device);
    }
}

pub struct RenderPass<B: hal::Backend> {
    pub r8: B::RenderPass,
    pub r8_depth: B::RenderPass,
    pub bgra8: B::RenderPass,
    pub bgra8_depth: B::RenderPass,
}

impl<B: hal::Backend> RenderPass<B> {
    pub fn get_render_pass(&self, format: ImageFormat, depth_enabled: bool) -> &B::RenderPass {
        match format {
            ImageFormat::R8 if depth_enabled => &self.r8_depth,
            ImageFormat::R8 => &self.r8,
            ImageFormat::BGRA8 if depth_enabled => &self.bgra8_depth,
            ImageFormat::BGRA8 => &self.bgra8,
            _ => unimplemented!(),
        }
    }

    pub fn deinit(self, device: &B::Device) {
        device.destroy_render_pass(self.r8);
        device.destroy_render_pass(self.r8_depth);
        device.destroy_render_pass(self.bgra8);
        device.destroy_render_pass(self.bgra8_depth);
    }
}

pub struct Device<B: hal::Backend> {
    pub device: B::Device,
    pub memory_types: Vec<hal::MemoryType>,
    pub upload_memory_type: hal::MemoryTypeId,
    pub download_memory_type: hal::MemoryTypeId,
    pub limits: hal::Limits,
    pub surface_format: hal::format::Format,
    pub depth_format: hal::format::Format,
    pub queue_group: hal::QueueGroup<B, hal::Graphics>,
    pub command_pool: hal::CommandPool<B, hal::Graphics>,
    pub swap_chain: B::Swapchain,
    pub render_pass: RenderPass<B>,
    pub framebuffers: Vec<B::Framebuffer>,
    pub framebuffers_depth: Vec<B::Framebuffer>,
    pub frame_images: Vec<ImageCore<B>>,
    pub frame_depth: DepthBuffer<B>,
    pub viewport: hal::pso::Viewport,
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    pub upload_queue: Vec<hal::command::Submit<B, hal::Graphics, hal::command::MultiShot, hal::command::Primary>>,
    pub current_frame_id: usize,
    current_blend_state: BlendState,
    blend_color: ColorF,
    current_depth_test: DepthTest,
    // device state
    programs: FastHashMap<ProgramId, Program<B>>,
    images: FastHashMap<TextureId, Image<B>>,
    fbos: FastHashMap<FBOId, Framebuffer<B>>,
    rbos: FastHashMap<RBOId, DepthBuffer<B>>,
    // device state
    bound_textures: [u32; 16],
    bound_program: ProgramId,
    bound_sampler: [TextureFilter; 16],
    bound_read_fbo: FBOId,
    bound_draw_fbo: FBOId,
    program_mode_id: i32,
    scissor_rect: Option<DeviceIntRect>,
    //default_read_fbo: FBOId,
    //default_draw_fbo: FBOId,

    device_pixel_ratio: f32,
    upload_method: UploadMethod,

    // HW or API capabilities
    #[cfg(feature = "debug_renderer")]
    capabilities: Capabilities,

    // debug
    inside_frame: bool,

    // resources
    _resource_override_path: Option<PathBuf>,

    max_texture_size: u32,
    _renderer_name: String,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: FrameId,

    // Supported features
    features: hal::Features,
    api_capabilities: ApiCapabilities,
}

impl<B: hal::Backend> Device<B> {
    pub fn new(
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        _cached_programs: Option<Rc<ProgramCache>>,
        adapter: &hal::Adapter<B>,
        surface: &mut <B as hal::Backend>::Surface,
        window_size: (u32, u32),
        api_capabilities: ApiCapabilities,
    ) -> Self {
        let renderer_name = "TODO renderer name".to_owned();
        let features = adapter.physical_device.features();

        let pixel_width = window_size.0;
        let pixel_height = window_size.1;

        let (caps, formats) = surface.capabilities_and_formats(&adapter.physical_device);
        let surface_format = formats
            .map_or(
                hal::format::Format::Bgra8Unorm,
                |formats| {
                    formats
                        .into_iter()
                        .find(|format| {
                            format == &hal::format::Format::Bgra8Unorm
                        })
                        .unwrap()
                },
            );

        let memory_types = adapter
            .physical_device
            .memory_properties()
            .memory_types;
        let limits = adapter
            .physical_device
            .limits();
        let max_texture_size = 4096u32; // TODO use limits after it points to the correct texture size

        let upload_memory_type: hal::MemoryTypeId = memory_types
            .iter()
            .position(|mt| {
                mt.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                //&&!mt.properties.contains(hal::memory::Properties::CPU_CACHED)
            })
            .unwrap()
            .into();
        let download_memory_type = memory_types
            .iter()
            .position(|mt| {
                mt.properties.contains(hal::memory::Properties::CPU_VISIBLE | hal::memory::Properties::CPU_CACHED)
            })
            .unwrap()
            .into();
        info!("upload memory: {:?}", upload_memory_type);
        info!("download memory: {:?}", &download_memory_type);

        let queue_family = adapter.queue_families
            .iter()
            .find(|family| surface.supports_queue_family(family))
            .expect("No queue family is able to render to the surface!");
        let mut gpu = adapter.physical_device
            .open(&[(queue_family, &[1.0])])
            .unwrap();
        let device = gpu.device;
        let queue_group = gpu.queues
            .take(queue_family.id())
            .unwrap();

        let mut command_pool = device.create_command_pool_typed(
            &queue_group,
            hal::pool::CommandPoolCreateFlags::empty(),
            32,
        );
        command_pool.reset();

        println!("{:?}", surface_format);
        assert_eq!(surface_format, hal::format::Format::Bgra8Unorm);
        let min_image_count = caps.image_count.start;
        let swap_config =
            SwapchainConfig::new()
                .with_color(surface_format)
                .with_image_count(min_image_count)
                .with_image_usage(
                    hal::image::Usage::TRANSFER_SRC | hal::image::Usage::TRANSFER_DST | hal::image::Usage::COLOR_ATTACHMENT
                );
        let (swap_chain, backbuffer) = device.create_swapchain(surface, swap_config);
        println!("backbuffer={:?}", backbuffer);
        let depth_format = hal::format::Format::D32Float; //maybe d24s8?

        let render_pass = {
            let attachment_r8 = hal::pass::Attachment {
                format: Some(hal::format::Format::R8Unorm),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::ColorAttachmentOptimal .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_bgra8 = hal::pass::Attachment {
                format: Some(hal::format::Format::Bgra8Unorm),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::ColorAttachmentOptimal .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_depth = hal::pass::Attachment {
                format: Some(depth_format),
                samples: 1,
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::DepthStencilAttachmentOptimal .. hal::image::Layout::DepthStencilAttachmentOptimal,
            };

            let subpass_r8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let subpass_depth_r8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let subpass_bgra8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let subpass_depth_bgra8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            let dependency = hal::pass::SubpassDependency {
                passes: hal::pass::SubpassRef::External .. hal::pass::SubpassRef::Pass(0),
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT
                    .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: hal::image::Access::empty()
                    .. (hal::image::Access::COLOR_ATTACHMENT_READ
                    | hal::image::Access::COLOR_ATTACHMENT_WRITE),
            };

            RenderPass {
                r8: device.create_render_pass(&[attachment_r8.clone()], &[subpass_r8], &[dependency.clone()]),
                r8_depth: device.create_render_pass(&[attachment_r8, attachment_depth.clone()], &[subpass_depth_r8], &[dependency.clone()]),
                bgra8: device.create_render_pass(&[attachment_bgra8.clone()], &[subpass_bgra8], &[dependency.clone()]),
                bgra8_depth: device.create_render_pass(&[attachment_bgra8, attachment_depth], &[subpass_depth_bgra8], &[dependency]),
            }
        };

        let frame_depth = DepthBuffer::new(&device, &memory_types, pixel_width, pixel_height, depth_format);

        // Framebuffer and render target creation
        let (frame_images, framebuffers, framebuffers_depth) = match backbuffer {
            Backbuffer::Images(images) => {
                let extent = hal::image::Extent {
                    width: pixel_width as _,
                    height: pixel_height as _,
                    depth: 1,
                };
                let cores = images
                    .into_iter()
                    .map(|image| {
                        ImageCore::from_image(&device, image, hal::image::ViewKind::D2Array, surface_format, COLOR_RANGE.clone())
                    })
                    .collect::<Vec<_>>();
                let fbos = cores
                    .iter()
                    .map(|core| {
                        device
                            .create_framebuffer(
                                &render_pass.bgra8,
                                Some(&core.view),
                                extent,
                            )
                            .unwrap()
                    })
                    .collect();
                let fbos_depth = cores
                    .iter()
                    .map(|core| {
                        device
                            .create_framebuffer(
                                &render_pass.bgra8_depth,
                                vec![&core.view, &frame_depth.core.view],
                                extent,
                            )
                            .unwrap()
                    })
                    .collect();
                (cores, fbos, fbos_depth)
            }
            // TODO fix depth fbos
            Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo], vec![]),
        };

        // Rendering setup
        let viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: pixel_width as u16,
                h: pixel_height as u16,
            },
            depth: 0.0 .. 1.0,
        };

        // Samplers

        let sampler_linear = device.create_sampler(hal::image::SamplerInfo::new(
            hal::image::Filter::Linear,
            hal::image::WrapMode::Clamp,
        ));

        let sampler_nearest = device.create_sampler(hal::image::SamplerInfo::new(
            hal::image::Filter::Nearest,
            hal::image::WrapMode::Clamp,
        ));

        Device {
            device,
            limits,
            memory_types,
            upload_memory_type,
            download_memory_type,
            surface_format,
            depth_format,
            queue_group,
            command_pool,
            swap_chain,
            render_pass,
            framebuffers,
            framebuffers_depth,
            frame_images,
            frame_depth,
            viewport,
            sampler_linear,
            sampler_nearest,
            upload_queue: Vec::new(),
            current_frame_id: 0,
            current_blend_state: BlendState::Off,
            current_depth_test: DepthTest::Off,
            blend_color: ColorF::new(0.0, 0.0, 0.0, 0.0),
            _resource_override_path: resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            upload_method,
            inside_frame: false,

            #[cfg(feature = "debug_renderer")]
            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },

            programs: FastHashMap::default(),
            images: FastHashMap::default(),
            fbos: FastHashMap::default(),
            rbos: FastHashMap::default(),
            bound_textures: [0; 16],
            bound_program: INVALID_PROGRAM_ID,
            bound_sampler: [TextureFilter::Linear; 16],
            bound_read_fbo: DEFAULT_READ_FBO,
            bound_draw_fbo: DEFAULT_DRAW_FBO,
            program_mode_id: 0,
            scissor_rect: None,

            max_texture_size,
            _renderer_name: renderer_name,
            frame_id: FrameId(0),
            features,
            api_capabilities,
        }
    }

    pub fn set_device_pixel_ratio(&mut self, ratio: f32) {
        self.device_pixel_ratio = ratio;
    }

    pub fn update_program_cache(&mut self, cached_programs: Rc<ProgramCache>) {
        unimplemented!();
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    #[cfg(feature = "debug_renderer")]
    pub fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub fn reset_state(&mut self) {
        self.bound_textures = [0; 16];
        self.bound_program = INVALID_PROGRAM_ID;
        self.bound_sampler = [TextureFilter::Linear; 16];
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
    }

    pub fn reset_image_buffer_offsets(&mut self) {
        for img in self.images.values_mut() {
            img.upload_buffer.reset();
        }
    }

    pub fn reset_program_buffer_offsets(&mut self) {
        for img in self.programs.values_mut() {
            img.instance_buffer.reset();
        }
    }

    pub fn delete_program(&mut self, mut _program: ProgramId) {
        // TODO delete program
        _program = INVALID_PROGRAM_ID;
    }

    pub fn reset_program(&mut self, program: &ProgramId) {
        self.programs.get_mut(program).expect("Program not found.").instance_buffer.reset();
    }

    pub(crate) fn create_program(
        &mut self,
        pipeline_requirements: PipelineRequirements,
        shader_name: &str,
        shader_kind: &ShaderKind,
    ) -> ProgramId {
        let program = Program::create(
            pipeline_requirements,
            &self.device,
            &self.memory_types,
            shader_name,
            shader_kind,
            &self.render_pass,
        );

        let id = self.generate_program_id();
        self.programs.insert(id, program);
        id
    }

    pub fn bind_program(&mut self, program_id: ProgramId) {
        debug_assert!(self.inside_frame);

        if self.bound_program != program_id {
            self.bound_program = program_id;
        }
    }

    pub fn bind_textures(&mut self) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        const SAMPLERS: [(usize, &'static str); 11] = [
            (0, "Color0"),
            (1, "Color1"),
            (2, "Color2"),
            (3, "CacheA8"),
            (4, "CacheRGBA8"),
            (5, "ResourceCache"),
            (6, "ClipScrollNodes"),
            (7, "RenderTasks"),
            (8, "Dither"),
            (9, "SharedCacheA8"),
            (10, "LocalClipRects")
        ];
        let program = self.programs.get_mut(&self.bound_program).expect("Program not found.");
        for &(index, sampler_name) in SAMPLERS.iter() {
            if self.bound_textures[index] != 0 {
                let sampler = match self.bound_sampler[index] {
                    TextureFilter::Linear | TextureFilter::Trilinear => &self.sampler_linear,
                    TextureFilter::Nearest => &self.sampler_nearest,
                };
                program.bind_texture(&self.device, &self.images[&self.bound_textures[index]].core, &sampler, sampler_name);
            }
        }
    }

    #[cfg(feature = "debug_renderer")]
    pub fn update_indices(&mut self, indices: &[u32]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self.programs.get_mut(&self.bound_program).expect("Program not found.");

        if let Some(ref mut index_buffer) = program.index_buffer {
            let index_buffer_len = indices.len() * index_buffer.data_stride;
            index_buffer.update(&self.device, 0, index_buffer_len as u64, &indices);
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    #[cfg(feature = "debug_renderer")]
    pub fn update_vertices<T: Copy>(&mut self, vertices: &[T]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self.programs.get_mut(&self.bound_program).expect("Program not found.");

        if program.shader_name.contains("debug") {
            let vertex_buffer_len = vertices.len() * program.vertex_buffer.data_stride;
            program.vertex_buffer.update(&self.device, 0, vertex_buffer_len as u64, &vertices);
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    pub fn set_uniforms(
        &mut self,
        transform: &Transform3D<f32>,
    ) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        self.programs.get_mut(&self.bound_program).expect("Program not found.").bind_locals(&self.device, transform, self.device_pixel_ratio, self.program_mode_id);
    }

    pub fn update_instances<T>(
        &mut self,
        instances: &[T],
    ) where
        T: Copy
    {
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        self.programs.get_mut(&self.bound_program).expect("Program not found.").bind_instances(&self.device, instances);
    }

    pub fn draw(
        &mut self,
    ) {
        let submit = {
            let (fb, format) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
                (&self.fbos[&self.bound_draw_fbo].fbo, self.fbos[&self.bound_draw_fbo].format)
            } else {
                if self.current_depth_test == DepthTest::Off {
                    (&self.framebuffers[self.current_frame_id], ImageFormat::BGRA8)
                } else {
                    (&self.framebuffers_depth[self.current_frame_id], ImageFormat::BGRA8)
                }
            };
            let rp = self.render_pass.get_render_pass(format, self.current_depth_test != DepthTest::Off);
            self.programs.get_mut(&self.bound_program).expect("Program not found").submit(
                &mut self.command_pool,
                self.viewport.clone(),
                rp,
                &fb,
                &vec![],
                self.current_blend_state,
                self.blend_color,
                self.current_depth_test,
                self.scissor_rect,
            )
        };

        self.upload_queue.push(submit);
        self.submit_to_gpu();
    }

    pub fn begin_frame(&mut self) -> FrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        self.bound_textures = [0; 16];
        self.bound_sampler = [TextureFilter::Linear; 16];
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
        self.program_mode_id = 0;

        self.frame_id
    }

    fn bind_texture_impl(&mut self, slot: TextureSlot, id: u32, sampler: TextureFilter) {
        debug_assert!(self.inside_frame);

        if self.bound_textures[slot.0] != id {
            self.bound_textures[slot.0] = id;
            self.bound_sampler[slot.0] = sampler;
        }
    }

    pub fn bind_texture<S>(&mut self, sampler: S, texture: &Texture)
        where
            S: Into<TextureSlot>,
    {
        self.bind_texture_impl(sampler.into(), texture.id, texture.filter);
    }

    pub fn bind_external_texture<S>(&mut self, sampler: S, external_texture: &ExternalTexture)
        where
            S: Into<TextureSlot>,
    {
        self.bind_texture_impl(sampler.into(), external_texture.id, TextureFilter::Linear);
    }

    pub fn bind_read_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_read_fbo != fbo_id {
            self.bound_read_fbo = fbo_id;
        }
    }

    pub fn bind_read_target(&mut self, texture_and_layer: Option<(&Texture, i32)>) {
        let fbo_id = texture_and_layer.map_or(DEFAULT_READ_FBO, |texture_and_layer| {
            texture_and_layer.0.fbo_ids[texture_and_layer.1 as usize]
        });

        self.bind_read_target_impl(fbo_id)
    }

    fn bind_draw_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
        }
    }

    pub fn bind_draw_target(
        &mut self,
        texture_and_layer: Option<(&Texture, i32)>,
        dimensions: Option<DeviceUintSize>,
    ) {
        let fbo_id = texture_and_layer.map_or(DEFAULT_DRAW_FBO, |texture_and_layer| {
            texture_and_layer.0.fbo_ids[texture_and_layer.1 as usize]
        });

        if let Some((texture, layer_index)) = texture_and_layer {
            let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);
            let rbos = &self.rbos;
            if let Some(barrier) = self.images[&texture.id].core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: 0 .. 1,
                    layers: layer_index as _ .. (layer_index + 1) as _,
                },
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            if let Some(barrier) = texture.depth_rb.and_then(|rbo| rbos[&rbo].core.transit(
                hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                hal::image::Layout::DepthStencilAttachmentOptimal,
                rbos[&rbo].core.subresource_range.clone(),
            )) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::EARLY_FRAGMENT_TESTS,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            self.upload_queue.push(cmd_buffer.finish())
        }

        self.bind_draw_target_impl(fbo_id);

        if let Some(dimensions) = dimensions {
            self.viewport.rect = hal::pso::Rect {
                x: 0,
                y: 0,
                w: dimensions.width as _,
                h: dimensions.height as _,
            };
        }
    }

    pub fn create_fbo_for_external_texture(&mut self, _texture_id: u32) -> FBOId {
        warn!("External texture creation is missing");
        FBOId(0)
    }

    pub fn delete_fbo(&mut self, _fbo: FBOId) {
        warn!("delete fbo is missing");
    }

    pub fn bind_external_draw_target(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
        }
    }

    fn generate_texture_id(&mut self) -> TextureId {
        let mut rng = rand::thread_rng();
        let mut texture_id = INVALID_TEXTURE_ID + 1;
        while self.images.contains_key(&texture_id) {
            texture_id = rng.gen_range::<u32>(INVALID_TEXTURE_ID + 1, u32::max_value());
        }
        texture_id
    }

    fn generate_program_id(&mut self) -> ProgramId {
        let mut rng = rand::thread_rng();
        let mut program_id = ProgramId(INVALID_PROGRAM_ID.0 + 1);
        while self.programs.contains_key(&program_id) {
            program_id = ProgramId(rng.gen_range::<u32>(INVALID_PROGRAM_ID.0 + 1, u32::max_value()));
        }
        program_id
    }

    fn generate_fbo_ids(&mut self, count: i32) -> Vec<FBOId> {
        let mut rng = rand::thread_rng();
        let mut fboids = vec!();
        let mut fbo_id = FBOId(DEFAULT_DRAW_FBO.0 + 1);
        for _ in 0..count {
            while self.fbos.contains_key(&fbo_id) || fboids.contains(&fbo_id) {
                fbo_id = FBOId(rng.gen_range::<u32>(DEFAULT_DRAW_FBO.0 + 1, u32::max_value()));
            }
            fboids.push(fbo_id);
        }
        fboids
    }

    fn generate_rbo_id(&mut self) -> RBOId {
        let mut rng = rand::thread_rng();
        let mut rbo_id = RBOId(1); // 0 is used for invalid
        while self.rbos.contains_key(&rbo_id) {
            rbo_id = RBOId(rng.gen_range::<u32>(1, u32::max_value()));
        }
        rbo_id
    }

    pub fn create_texture(
        &mut self,
        target: TextureTarget,
        format: ImageFormat,
    ) -> Texture {
        Texture {
            id: 0,
            _target: target,
            width: 0,
            height: 0,
            layer_count: 0,
            format,
            filter: TextureFilter::Nearest,
            render_target: None,
            fbo_ids: vec![],
            depth_rb: None,
            last_frame_used: self.frame_id,
        }
    }

    /// Resizes a texture with enabled render target views,
    /// preserves the data by blitting the old texture contents over.
    pub fn resize_renderable_texture(
        &mut self,
        _texture: &mut Texture,
        _new_size: DeviceUintSize,
    ) {
        unimplemented!();
    }

    pub fn init_texture<T: Texel>(
        &mut self,
        texture: &mut Texture,
        mut width: u32,
        mut height: u32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
        pixels: Option<&[T]>,
    ) {
        debug_assert!(self.inside_frame);

        if width > self.max_texture_size || height > self.max_texture_size {
            error!("Attempting to allocate a texture of size {}x{} above the limit, trimming", width, height);
            width = width.min(self.max_texture_size);
            height = height.min(self.max_texture_size);
        }

        let is_resized = texture.width != width || texture.height != height;

        texture.width = width;
        texture.height = height;
        texture.filter = filter;
        texture.layer_count = layer_count;
        texture.render_target = render_target;
        texture.last_frame_used = self.frame_id;

        if texture.id == 0 {
            let id = self.generate_texture_id();
            texture.id = id;
        } else {
            self.free_image(texture);
        }
        assert_eq!(self.images.contains_key(&texture.id), false);
        let (view_kind, mip_levels) = match texture.filter {
            TextureFilter::Nearest => (hal::image::ViewKind::D2, 1),
            TextureFilter::Linear => (hal::image::ViewKind::D2Array, 1),
            TextureFilter::Trilinear => (hal::image::ViewKind::D2Array, (width as f32).max(height as f32).log2().floor() as u8 + 1),
        };
        let img = Image::new(
            &self.device,
            &self.memory_types,
            texture.format,
            texture.width,
            texture.height,
            texture.layer_count,
            view_kind,
            mip_levels,
            (self.limits.min_buffer_copy_pitch_alignment - 1) as usize,
        );

        assert_eq!(texture.fbo_ids.len(), 0);

        {
            let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);

            if let Some(barrier) = img.core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                img.core.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT..PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            self.upload_queue.push(cmd_buffer.finish());
        }

        if let Some(rt_info) = render_target {
            let (depth_rb, allocate_depth) = match texture.depth_rb {
                Some(rbo) => (rbo, is_resized || !rt_info.has_depth),
                None if rt_info.has_depth => {
                    let depth_rb = self.generate_rbo_id();
                    texture.depth_rb = Some(depth_rb);
                    (depth_rb, true)
                },
                None => (RBOId(0), false),
            };

            if allocate_depth {
                if self.rbos.contains_key(&depth_rb) {
                    let old_rbo = self.rbos.remove(&depth_rb).unwrap();
                    old_rbo.deinit(&self.device);
                }
                if rt_info.has_depth {
                    let rbo = DepthBuffer::new(
                        &self.device,
                        &self.memory_types,
                        texture.width,
                        texture.height,
                        self.depth_format
                    );
                    self.rbos.insert(depth_rb, rbo);
                } else {
                    texture.depth_rb = None;
                }
            }

            let new_fbos = self.generate_fbo_ids(texture.layer_count);

            for i in 0..texture.layer_count as u16 {
                let (rbo_id, depth) = match texture.depth_rb {
                    Some(rbo_id) => (rbo_id.clone(), Some(&self.rbos[&rbo_id].core.view)),
                    None => (RBOId(0), None)
                };
                let fbo = Framebuffer::new(&self.device, &texture, &img, i, &self.render_pass, rbo_id.clone(), depth);
                self.fbos.insert(new_fbos[i as usize], fbo);
                texture.fbo_ids.push(new_fbos[i as usize]);
            }
        }

        self.images.insert(texture.id, img);


        if let Some(data) = pixels {
            self.upload_queue
                .push(
                    self.images
                        .get_mut(&texture.id)
                        .expect("Texture not found.")
                        .update(
                            &mut self.device,
                            &mut self.command_pool,
                            DeviceUintRect::new(
                                DeviceUintPoint::new(0, 0),
                                DeviceUintSize::new(texture.width, texture.height),
                            ),
                            0,
                            texels_to_u8_slice(data),
                            (self.limits.min_buffer_copy_offset_alignment - 1) as usize,
                        )
                );
            if texture.filter == TextureFilter::Trilinear {
                self.generate_mipmaps(texture);
            }
        }
    }

    fn generate_mipmaps(&mut self, texture: &Texture) {
        if !self.api_capabilities.contains(ApiCapabilities::BLITTING) {
            warn!("Blitting is not supported!");
            return;
        }

        let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);

        let image = self.images
            .get_mut(&texture.id)
            .expect("Texture not found.");

        let mut mip_width = texture.width;
        let mut mip_height = texture.height;

        let mut half_mip_width =  mip_width / 2;
        let mut half_mip_height =  mip_height / 2;

        if let Some(barrier) = image.core.transit(
            hal::image::Access::TRANSFER_WRITE,
            hal::image::Layout::TransferDstOptimal,
            image.core.subresource_range.clone(),
        ) {
            cmd_buffer.pipeline_barrier(
                PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        for index in 1 .. image.kind.num_levels() {
            if let Some(barrier) = image.core.transit(
                hal::image::Access::TRANSFER_READ,
                hal::image::Layout::TransferSrcOptimal,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: index - 1 .. index,
                    layers: 0 .. 1,
                },
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.blit_image(
                &image.core.image,
                hal::image::Layout::TransferSrcOptimal,
                &image.core.image,
                hal::image::Layout::TransferDstOptimal,
                hal::image::Filter::Linear,
                &[
                    hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index - 1,
                            layers: 0 .. 1,
                        },
                        src_bounds: hal::image::Offset {
                            x: 0,
                            y: 0,
                            z: 0,
                        } .. hal::image::Offset {
                            x: mip_width as i32,
                            y: mip_height as i32,
                            z: 1,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index,
                            layers: 0 .. 1,
                        },
                        dst_bounds: hal::image::Offset {
                            x: 0 as i32,
                            y: 0 as i32,
                            z: 0,
                        } .. hal::image::Offset {
                            x: half_mip_width as i32,
                            y: half_mip_height as i32,
                            z: 1,
                        },
                    }
                ],
            );
            if let Some(barrier) = image.core.transit(
                hal::image::Access::TRANSFER_WRITE,
                hal::image::Layout::TransferDstOptimal,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: index - 1 .. index,
                    layers: 0 .. 1,
                },
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            mip_width = half_mip_width;
            if half_mip_width > 1 {
                half_mip_width /= 2;
            }
            mip_height = half_mip_height;
            if half_mip_height > 1 {
                half_mip_height /= 2;
            }
        }

        if let Some(barrier) = image.core.transit(
            hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
            hal::image::Layout::ColorAttachmentOptimal,
            image.core.subresource_range.clone(),
        ) {
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        self.upload_queue.push(cmd_buffer.finish());
    }

    pub fn blit_render_target(&mut self, src_rect: DeviceIntRect, dest_rect: DeviceIntRect) {
        debug_assert!(self.inside_frame);

        let (src_img, src_layer) = if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture];
            let layer = fbo.layer_index;
            (&img.core, layer)
        } else {
            (&self.frame_images[self.current_frame_id], 0)
        };

        let (dest_img, dest_layer) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture];
            let layer = fbo.layer_index;
            (&img.core, layer)
        } else {
            (&self.frame_images[self.current_frame_id], 0)
        };

        let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);

        let src_range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: src_layer .. src_layer + 1,
        };
        let dest_range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: dest_layer .. dest_layer + 1,
        };
        {
            let mut barriers = Vec::new();
            barriers.extend(
                src_img.transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    src_range.clone(),
                )
            );
            barriers.extend(
                dest_img.transit(
                    hal::image::Access::TRANSFER_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    dest_range.clone(),
                )
            );
            if !barriers.is_empty() {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &barriers,
                );
            }
        }

        if src_rect.size != dest_rect.size {
            // TODO remove this if other platforms are supported
            if !self.api_capabilities.contains(ApiCapabilities::BLITTING) {
                warn!("Blitting is not supported!");
                return;
            }
            cmd_buffer.blit_image(
                &src_img.image,
                hal::image::Layout::TransferSrcOptimal,
                &dest_img.image,
                hal::image::Layout::TransferDstOptimal,
                hal::image::Filter::Linear,
                &[
                    hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: src_layer .. src_layer + 1,
                        },
                        src_bounds: hal::image::Offset {
                            x: src_rect.origin.x as i32,
                            y: src_rect.origin.y as i32,
                            z: 0,
                        } .. hal::image::Offset {
                            x: src_rect.origin.x as i32 + src_rect.size.width as i32,
                            y: src_rect.origin.y as i32 + src_rect.size.height as i32,
                            z: 1,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: dest_layer .. dest_layer + 1,
                        },
                        dst_bounds: hal::image::Offset {
                            x: dest_rect.origin.x as i32,
                            y: dest_rect.origin.y as i32,
                            z: 0,
                        } .. hal::image::Offset {
                            x: dest_rect.origin.x as i32 + dest_rect.size.width as i32,
                            y: dest_rect.origin.y as i32 + dest_rect.size.height as i32,
                            z: 1,
                        },
                    }
                ],
            );
        } else {
            cmd_buffer.copy_image(
                &src_img.image,
                hal::image::Layout::TransferSrcOptimal,
                &dest_img.image,
                hal::image::Layout::TransferDstOptimal,
                &[
                    hal::command::ImageCopy {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: src_layer .. src_layer + 1,
                        },
                        src_offset: hal::image::Offset {
                            x: src_rect.origin.x as i32,
                            y: src_rect.origin.y as i32,
                            z: 0,
                        },
                        dst_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: 0,
                            layers: dest_layer as _ .. (dest_layer + 1) as _,
                        },
                        dst_offset: hal::image::Offset {
                            x: dest_rect.origin.x as i32,
                            y: dest_rect.origin.y as i32,
                            z: 0,
                        },
                        extent: hal::image::Extent {
                            width: src_rect.size.width as u32,
                            height: src_rect.size.height as u32,
                            depth: 1,
                        },
                    }
                ],
            );
        }

        // the blit caller code expects to be able to render to the target
        {
            let mut barriers = Vec::new();
            barriers.extend(
                src_img.transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::ColorAttachmentOptimal,
                    src_range,
                )
            );
            barriers.extend(
                dest_img.transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::ColorAttachmentOptimal,
                    dest_range,
                )
            );
            if !barriers.is_empty() {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &barriers,
                );
            }
        }

        self.upload_queue.push(cmd_buffer.finish());
    }

    pub fn free_texture_storage(&mut self, texture: &mut Texture) {
        debug_assert!(self.inside_frame);
        if texture.width + texture.height == 0 {
            return;
        }

        self.free_image(texture);

        texture.width = 0;
        texture.height = 0;
        texture.layer_count = 0;
        texture.id = 0;
    }

    pub fn free_image(&mut self, texture: &mut Texture) {
        if let Some(depth_rb) = texture.depth_rb.take() {
            let old_rbo = self.rbos.remove(&depth_rb).unwrap();
            old_rbo.deinit(&self.device);
        }

        if !texture.fbo_ids.is_empty() {
            for old in texture.fbo_ids.drain(..) {
                let old_fbo = self.fbos.remove(&old).unwrap();
                old_fbo.deinit(&self.device);
            }
        }

        let image = self.images.remove(&texture.id).expect("Texture not found.");
        image.deinit(&self.device);
    }

    pub fn delete_texture(&mut self, mut texture: Texture) {
        self.free_texture_storage(&mut texture);
    }

    #[cfg(feature = "replay")]
    pub fn delete_external_texture(&mut self, mut external: ExternalTexture) {
        warn!("delete external texture is missing");
        external.id = 0;
    }

    pub fn switch_mode(&mut self, mode: i32) {
        debug_assert!(self.inside_frame);
        self.program_mode_id = mode;
    }

    pub fn create_pbo(&mut self) -> PBO {
        PBO { }
    }

    pub fn delete_pbo(&mut self, _pbo: PBO) {
    }

    pub fn upload_texture<'a>(
        &'a mut self,
        texture: &'a Texture,
        _pbo: &PBO,
        _upload_count: usize,
    ) -> TextureUploader<'a, B> {
        debug_assert!(self.inside_frame);

        match self.upload_method {
            UploadMethod::Immediate => unimplemented!(),
            UploadMethod::PixelBuffer => {
                TextureUploader {
                        device: self,
                        texture,
                }
            },
        }

    }

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    pub fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8> {
        let mut pixels = vec![0; (img_desc.width * img_desc.height * 4) as usize];
        self.read_pixels_into(DeviceUintRect::new(DeviceUintPoint::zero(), DeviceUintSize::new(img_desc.width, img_desc.height)), ReadPixelsFormat::Rgba8, &mut pixels);
        pixels
    }

    /// Read rectangle of pixels into the specified output slice.
    pub fn read_pixels_into(
        &mut self,
        rect: DeviceUintRect,
        format: ReadPixelsFormat,
        output: &mut [u8],
    ) {
        let bytes_per_pixel = match format {
            ReadPixelsFormat::Standard(imf) => imf.bytes_per_pixel(),
            ReadPixelsFormat::Rgba8 => 4,
        };
        let size_in_bytes = (bytes_per_pixel * rect.size.width * rect.size.height) as usize;
        assert_eq!(output.len(), size_in_bytes);
        let (image, layer) = if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture];
            let layer = fbo.layer_index;
            (&img.core, layer)
        } else {
            (&self.frame_images[self.current_frame_id], 0)
        };
        let download_buffer: CopyBuffer<B> = CopyBuffer::create(
            &self.device,
            &self.memory_types,
            hal::buffer::Usage::TRANSFER_DST,
            1,
            (rect.size.width * bytes_per_pixel) as usize,
            rect.size.height as usize,
            (self.limits.min_buffer_copy_pitch_alignment - 1) as usize,
        );

        let copy_submit = {
            let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);
            let mut barriers = Vec::new();
            let range = hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: layer .. layer + 1,
            };
            barriers.extend(download_buffer.transit(hal::buffer::Access::TRANSFER_WRITE));
            barriers.extend(
                image.transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    range.clone(),
                )
            );
            if !barriers.is_empty() {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &barriers
                );
            }

            cmd_buffer.copy_image_to_buffer(
                &image.image,
                hal::image::Layout::TransferSrcOptimal,
                &download_buffer.buffer,
                &[hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: rect.size.width,
                    buffer_height: rect.size.height,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: layer .. layer + 1,
                    },
                    image_offset: hal::image::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    image_extent: hal::image::Extent {
                        width: rect.size.width as _,
                        height: rect.size.height as _,
                        depth: 1 as _,
                    },
                }]);
            if let Some(barrier) = image.transit(
                hal::image::Access::empty(),
                hal::image::Layout::ColorAttachmentOptimal,
                range,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.finish()
        };

        let copy_fence = self.device.create_fence(false);
        let submission = hal::queue::Submission::new()
            .submit(Some(copy_submit));
        self.queue_group.queues[0].submit(submission, Some(&copy_fence));
        //queue.destroy_command_pool(command_pool);
        self.device.wait_for_fence(&copy_fence, !0);
        self.device.destroy_fence(copy_fence);

        if let Ok(reader) = self.device
            .acquire_mapping_reader::<[u8; 4]>(
                &download_buffer.memory,
                0 .. (rect.size.width * rect.size.height * bytes_per_pixel as u32) as u64,
            )
            {
                assert_eq!(reader.len() * 4, output.len());
                let mut offset = 0;
                let (i0, i1, i2, i3) = match self.surface_format.base_format().0 {
                    hal::format::SurfaceType::B8_G8_R8_A8 => (2, 1, 0, 3),
                    //hal::format::SurfaceType::R8_G8_B8_A8 => (0, 1, 2, 3),
                    _ => (0, 1, 2, 3)
                };
                for d in reader.iter() {
                    let data = *d;
                    output[offset + 0] = data[i0];
                    output[offset + 1] = data[i1];
                    output[offset + 2] = data[i2];
                    output[offset + 3] = data[i3];
                    offset += 4;
                }
                self.device.release_mapping_reader(reader);
            } else {
            panic!("Fail to read the download buffer!");
        }

        download_buffer.deinit(&self.device);
    }

    /// Get texels of a texture into the specified output slice.
    #[cfg(feature = "debug_renderer")]
    pub fn get_tex_image_into(
        &mut self,
        texture: &Texture,
        format: ImageFormat,
        output: &mut [u8],
    ) {
        unimplemented!();
    }

    /// Attaches the provided texture to the current Read FBO binding.
    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    fn attach_read_texture_raw(
        &mut self, texture_id: u32, target: TextureTarget, layer_id: i32
    ) {
        unimplemented!();
    }

    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    pub fn attach_read_texture_external(
        &mut self, texture_id: u32, target: TextureTarget, layer_id: i32
    ) {
        self.attach_read_texture_raw(texture_id, target, layer_id)
    }

    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    pub fn attach_read_texture(&mut self, texture: &Texture, layer_id: i32) {
        self.attach_read_texture_raw(texture.id, texture._target, layer_id)
    }

    pub fn bind_vao(&mut self, _vao: &VAO) { }


    fn create_vao_with_vbos(
        &mut self,
        _descriptor: &VertexDescriptor,
        _main_vbo_id: VBOId,
        _instance_vbo_id: VBOId,
        _ibo_id: IBOId,
        _owns_vertices_and_indices: bool,
    ) -> VAO {
        VAO { }
    }

    pub fn create_vao(&mut self, _descriptor: &VertexDescriptor) -> VAO {
        VAO { }
    }

    pub fn delete_vao(&mut self, _vao: VAO) { }

    pub fn create_vao_with_new_instances(
        &mut self,
        _descriptor: &VertexDescriptor,
        _base_vao: &VAO,
    ) -> VAO {
        VAO { }
    }

    pub fn update_vao_main_vertices<V>(
        &mut self,
        _vao: &VAO,
        _vertices: &[V],
        _usage_hint: VertexUsageHint,
    ) { }

    pub fn update_vao_instances<V>(
        &mut self,
        _vao: &VAO,
        instances: &[V],
        _usage_hint: VertexUsageHint,
    )
        where V: PrimitiveType
    {
        let data = instances.iter().map(|pi| pi.to_primitive_type()).collect::<Vec<V::Primitive>>();
        self.update_instances(&data);
    }

    pub fn update_vao_indices<I>(&mut self, _vao: &VAO, _indices: &[I], _usage_hint: VertexUsageHint) { }

    pub fn draw_triangles_u16(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn draw_indexed_triangles_instanced_u16(&mut self, _index_count: i32, _instance_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn end_frame(&mut self) {
        self.bind_draw_target(None, None);
        self.bind_read_target(None);

        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        self.frame_id.0 += 1;
    }

    pub fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<DeviceIntRect>,
    ) {
        let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);

        if let Some(_rect) = rect {
            //TODO handle scissors
        }

        let (img, layer, dimg) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture];
            let dimg = if depth.is_some() {
                Some(&self.rbos[&fbo.rbo].core)
            } else {
                None
            };
            let layer = fbo.layer_index;
            (&img.core, layer, dimg)
        } else {
            (&self.frame_images[self.current_frame_id], 0, Some(&self.frame_depth.core))
        };

        //Note: this function is assumed to be called within an active FBO
        // thus, we bring back the targets into renderable state

        if let Some(color) = color {
            let color_range = hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: layer .. layer + 1,
            };
            if let Some(barrier) = img.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::TransferDstOptimal,
                color_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.clear_image(
                &img.image,
                hal::image::Layout::TransferDstOptimal,
                hal::command::ClearColor::Float([color[0], color[1], color[2], color[3]]),
                hal::command::ClearDepthStencil(0.0, 0),
                Some(color_range.clone()),
            );
            if let Some(barrier) = img.transit(
                hal::image::Access::empty(),
                hal::image::Layout::ColorAttachmentOptimal,
                color_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }

        if let (Some(depth), Some(dimg)) = (depth, dimg) {
            assert_ne!(self.current_depth_test, DepthTest::Off);
            if let Some(barrier) = dimg.transit(
                hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                hal::image::Layout::TransferDstOptimal,
                dimg.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::EARLY_FRAGMENT_TESTS,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.clear_image(
                &dimg.image,
                hal::image::Layout::TransferDstOptimal,
                hal::command::ClearColor::Float([0.0; 4]),
                hal::command::ClearDepthStencil(depth, 0),
                Some(dimg.subresource_range.clone()),
            );
            if let Some(barrier) = dimg.transit(
                hal::image::Access::empty(),
                hal::image::Layout::DepthStencilAttachmentOptimal,
                dimg.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::EARLY_FRAGMENT_TESTS,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
        self.upload_queue.push(cmd_buffer.finish());
    }

    pub fn enable_depth(&mut self) {
        self.current_depth_test = LESS_EQUAL_TEST;
    }

    pub fn disable_depth(&mut self) {
        self.current_depth_test = DepthTest::Off;
    }

    pub fn set_depth_func(&mut self, _depth_func: DepthFunction) {
        // TODO add Less depth function
        //self.current_depth_test = depth_func;
    }

    pub fn enable_depth_write(&mut self) {
        self.current_depth_test = LESS_EQUAL_WRITE;
    }

    pub fn disable_depth_write(&mut self) {
        if self.current_depth_test != DepthTest::Off {
            self.current_depth_test = LESS_EQUAL_TEST;
        }
    }

    pub fn disable_stencil(&self) {
        warn!("disable stencil is missing")
    }

    pub fn set_scissor_rect(&mut self, rect: DeviceIntRect) {
        self.scissor_rect = Some(rect);
    }

    pub fn enable_scissor(&self) {}

    pub fn disable_scissor(&mut self) {
        self.scissor_rect = None;
    }

    pub fn set_blend(&mut self, enable: bool) {
        if !enable {
            self.current_blend_state = BlendState::Off
        }
    }

    pub fn set_blend_mode_alpha(&mut self) {
        self.current_blend_state = ALPHA;
    }

    pub fn set_blend_mode_premultiplied_alpha(&mut self) {
        self.current_blend_state = BlendState::PREMULTIPLIED_ALPHA;
    }

    pub fn set_blend_mode_premultiplied_dest_out(&mut self) {
        self.current_blend_state = PREMULTIPLIED_DEST_OUT;
    }

    pub fn set_blend_mode_multiply(&mut self) {
        self.current_blend_state = BlendState::MULTIPLY;
    }
    pub fn set_blend_mode_max(&mut self) {
        self.current_blend_state = MAX;
    }
    #[cfg(feature = "debug_renderer")]
    pub fn set_blend_mode_min(&mut self) {
        self.current_blend_state = MIN;
    }
    pub fn set_blend_mode_subpixel_pass0(&mut self) {
        self.current_blend_state = SUBPIXEL_PASS0;
    }
    pub fn set_blend_mode_subpixel_pass1(&mut self) {
        self.current_blend_state = SUBPIXEL_PASS1;
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass0(&mut self) {
        self.current_blend_state = SUBPIXEL_WITH_BG_COLOR_PASS0;
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass1(&mut self) {
        self.current_blend_state = SUBPIXEL_WITH_BG_COLOR_PASS1;
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass2(&mut self) {
        self.current_blend_state = SUBPIXEL_WITH_BG_COLOR_PASS2;
    }
    pub fn set_blend_mode_subpixel_constant_text_color(&mut self, color: ColorF) {
        self.current_blend_state = SUBPIXEL_CONSTANT_TEXT_COLOR;
        // color is an unpremultiplied color.
        self.blend_color = ColorF::new(color.r, color.g, color.b, 1.0);
    }
    pub fn set_blend_mode_subpixel_dual_source(&mut self) {
        self.current_blend_state = SUBPIXEL_DUAL_SOURCE;
    }

    pub fn supports_features(&self, features: hal::Features) -> bool {
        self.features.contains(features)
    }

    pub fn set_next_frame_id_and_return_semaphore(&mut self) -> B::Semaphore {
        let mut frame_semaphore = self.device.create_semaphore();
        let frame = self.swap_chain
            .acquire_frame(FrameSync::Semaphore(&mut frame_semaphore));
        self.current_frame_id = frame.id();
        frame_semaphore
    }

    pub fn submit_to_gpu(&mut self) {
        let mut frame_fence = self.device.create_fence(false); // TODO: remove
        {
            self.device.reset_fence(&frame_fence);
            let submission = Submission::new()
                .submit(&self.upload_queue);
            self.queue_group.queues[0].submit(submission, Some(&mut frame_fence));

            // TODO: replace with semaphore
            self.device
                .wait_for_fence(&frame_fence, !0);
        }
        self.upload_queue.clear();
        self.command_pool.reset();
        self.device.destroy_fence(frame_fence);
    }

    pub fn swap_buffers(&mut self, frame_semaphore: B::Semaphore) {
        {
            let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);
            let image = &self.frame_images[self.current_frame_id];
            if let Some(barrier) = image.transit(
                hal::image::Access::empty(),
                hal::image::Layout::Present,
                image.subresource_range.clone(),
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            self.upload_queue.push(cmd_buffer.finish());
        }

        let mut frame_fence = self.device.create_fence(false); // TODO: remove
        {
            self.device.reset_fence(&frame_fence);
            let submission = Submission::new()
                .wait_on(&[(&frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
                .submit(&self.upload_queue);
            self.queue_group.queues[0].submit(submission, Some(&mut frame_fence));

            // TODO: replace with semaphore
            self.device
                .wait_for_fence(&frame_fence, !0);

            // present frame
            self.swap_chain
                .present(&mut self.queue_group.queues[0], &[]);
        }
        self.upload_queue.clear();
        self.command_pool.reset();
        self.reset_state();
        self.reset_image_buffer_offsets();
        self.reset_program_buffer_offsets();

        self.device.destroy_fence(frame_fence);
        self.device.destroy_semaphore(frame_semaphore);
    }

    pub fn deinit(self) {
        self.device.destroy_command_pool(self.command_pool.into_raw());
        for image in self.frame_images {
            image.deinit(&self.device);
        }
        for (_, image) in self.images {
            image.deinit(&self.device);
        }
        for (_, rbo) in self.fbos {
            rbo.deinit(&self.device);
        }
        for (_, rbo) in self.rbos {
            rbo.deinit(&self.device);
        }
        for framebuffer in self.framebuffers {
            self.device.destroy_framebuffer(framebuffer);
        }
        for framebuffer_depth in self.framebuffers_depth {
            self.device.destroy_framebuffer(framebuffer_depth);
        }
        self.frame_depth.deinit(&self.device);
        self.device.destroy_sampler(self.sampler_linear);
        self.device.destroy_sampler(self.sampler_nearest);
        for (_, program) in self.programs {
            program.deinit(&self.device)
        }
        self.render_pass.deinit(&self.device);
        self.device.destroy_swapchain(self.swap_chain);
    }
}

pub struct TextureUploader<'a, B: hal::Backend> {
    device: &'a mut Device<B>,
    texture: &'a Texture,
}

impl<'a, B: hal::Backend> TextureUploader<'a, B> {
    pub fn upload<T>(
        &mut self,
        rect: DeviceUintRect,
        layer_index: i32,
        stride: Option<u32>,
        data: &[T],
    ) {
        let data = unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * mem::size_of::<T>())
        };
        let data_stride: usize = self.texture.format.bytes_per_pixel() as usize;
        let width = rect.size.width as usize;
        let height = rect.size.height as usize;
        let size = width * height * data_stride;
        let mut new_data = vec![0u8; size];
        let data= if stride.is_some() {
            let row_length = (stride.unwrap()) as usize;

            for j in 0..height {
                for i in 0..width {
                    let offset = i * data_stride + j * data_stride * width;
                    let src = &data[j * row_length + i * data_stride ..];
                    assert!(offset + 3 < new_data.len()); // optimization
                    // convert from BGRA
                    new_data[offset + 0] = src[0];
                    new_data[offset + 1] = src[1];
                    new_data[offset + 2] = src[2];
                    new_data[offset + 3] = src[3];
                }
            }

            new_data.as_slice()
        } else {
            data
        };
        assert_eq!(data.len(), width * height * data_stride);
        self.device.upload_queue
            .push(
                self.device.images
                    .get_mut(&self.texture.id)
                    .expect("Texture not found.")
                    .update(
                        &mut self.device.device,
                        &mut self.device.command_pool,
                        rect,
                        layer_index,
                        data,
                        (self.device.limits.min_buffer_copy_offset_alignment - 1) as usize,
                    )
            );

        if self.texture.filter == TextureFilter::Trilinear {
            self.device.generate_mipmaps(self.texture);
        }
    }
}

fn texels_to_u8_slice<T: Texel>(texels: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(texels.as_ptr() as *const u8, texels.len() * mem::size_of::<T>())
    }
}
