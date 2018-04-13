/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source;
use api::{ColorF, ImageDescriptor, ImageFormat};
use api::{DeviceIntPoint, DeviceIntRect, DeviceUintPoint, DeviceUintRect, DeviceUintSize};
//use api::TextureTarget;
use euclid::Transform3D;
//use gleam::gl;
use internal_types::{FastHashMap, RenderTargetInfo};
use rand::{self, Rng};
use serde;
use std::collections::HashMap;
use smallvec::SmallVec;
use std::cell::{Cell, RefCell};
use std::cmp;
use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Range};
use std::path::PathBuf;
use std::thread;

use hal;
use winit;

// gfx-hal
use hal::pso::{AttributeDesc, DescriptorRangeDesc, DescriptorSetLayoutBinding, VertexBufferDesc};
use hal::pso::{BlendState, BlendOp, Comparison, DepthTest, Factor};
use hal::{Device as BackendDevice, PhysicalDevice, QueueFamily, Surface, Swapchain};
use hal::{Backbuffer, DescriptorPool, FrameSync, Primitive, SwapchainConfig};
use hal::format::{ChannelType, Format, Swizzle};
use hal::pass::Subpass;
use hal::pso::PipelineStage;
use hal::queue::Submission;
use ron::de::from_reader;

pub const NODE_TEXTURE_WIDTH: usize = 1017; // 113 * ( 36 / 4)
pub const RENDER_TASK_TEXTURE_WIDTH: usize = 1023; // 341 * ( 12 / 4 )
pub const CLIP_RECTS_TEXTURE_WIDTH: usize = 1024;
pub const TEXTURE_HEIGHT: usize = 8;
pub const MAX_INSTANCE_COUNT: usize = 1024;
const MAX_VERTEX_TEXTURE_WIDTH: usize = 1024;

pub type TextureId = u32;

pub const INVALID_TEXTURE_ID: TextureId = 0;
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
pub struct Vertex {
    aPosition: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
struct Locals {
    uTransform: [[f32; 4]; 4],
    uDevicePixelRatio: f32,
    uMode: i32,
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct PrimitiveInstance {
    pub aData0: [i32; 4],
    pub aData1: [i32; 4],
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct ClipMaskInstance {
    pub aData0: [i32; 4],
    pub aData1: [i32; 4],
    pub aClipRenderTaskAddress: i32,
    pub aScrollNodeId: i32,
    pub aClipSegment: i32,
    pub aClipDataResourceAddress: [i32; 4],
}

#[derive(Debug, Clone, Copy)]
#[allow(non_snake_case)]
pub struct BlurInstance {
    pub aData0: [i32; 4],
    pub aData1: [i32; 4],
    pub aBlurRenderTaskAddress: i32,
    pub aBlurSourceTaskAddress: i32,
    pub aBlurDirection: i32,
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

pub struct TextureSlot(pub usize);

// In some places we need to temporarily bind a texture to any slot.
const DEFAULT_TEXTURE: TextureSlot = TextureSlot(0);

#[repr(u32)]
pub enum DepthFunction {
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
    //target: TextureTarget,
}

impl ExternalTexture {
    pub fn new(id: u32/*, target: TextureTarget*/) -> Self {
        ExternalTexture {
            id,
            //target,
        }
    }

    #[cfg(feature = "replay")]
    pub fn internal_id(&self) -> u32 {
        self.id
    }
}

#[derive(Debug)]
pub struct Texture {
    id: TextureId,
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

    pub fn get_filter(&self) -> TextureFilter {
        self.filter
    }

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

pub struct PBO {
    id: u32,
}

impl Drop for PBO {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called"
        );
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct FBOId(u32);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(u32);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(u32);

#[derive(Debug, Copy, Clone)]
pub enum VertexUsageHint {
    Static,
    Dynamic,
    Stream,
}

pub struct Capabilities {
    pub supports_multisampling: bool,
}

bitflags!(
    pub struct ApiCapabilities: u8 {
        const BLITTING = 0x1;
    }
);

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error mssage
    Link(String, String),        // name, error message
}

#[derive(Debug, Copy, Clone)]
pub enum VertexArrayKind {
    Primitive,
    Blur,
    Clip,
}

pub enum ShaderKind {
    Primitive,
    Cache(VertexArrayKind),
    ClipCache,
    Brush,
    Text,
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
        dst: Factor::OneMinusSrcColor,
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
        src: Factor::ConstColor,
        dst: Factor::OneMinusSrcColor,
    },
};

const SUBPIXEL_DUAL_SOURCE: BlendState = BlendState::On {
    color: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrc1Color,
    },
    alpha: BlendOp::Add {
        src: Factor::One,
        dst: Factor::OneMinusSrc1Color,
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
        format: hal::format::Format,
        usage: hal::image::Usage,
        subresource_range: hal::image::SubresourceRange,
    ) -> Self {
        let image_unbound = device
            .create_image(kind, 1, format, hal::image::Tiling::Optimal, usage, hal::image::StorageFlags::empty())
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

    fn reset(&self) {
        self.state.set((hal::image::Access::empty(), hal::image::Layout::Undefined));
    }

    fn deinit(self, device: &B::Device) {
        device.destroy_image(self.image);
        device.destroy_image_view(self.view);
        if let Some(memory) = self.memory {
            device.free_memory(memory);
        }
    }

    fn transit(
        &self,
        access: hal::image::Access,
        layout: hal::image::Layout,
    ) -> Option<hal::memory::Barrier<B>> {
        let src_state = self.state.get();
        if src_state == (access, layout) {
            None
        } else {
            self.state.set((access, layout));
            Some(hal::memory::Barrier::Image {
                states: src_state .. (access, layout),
                target: &self.image,
                range: self.subresource_range.clone(),
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
        pitch_alignment: usize,
    ) -> Self {
        let (bytes_per_pixel, format) = match image_format {
            ImageFormat::R8 => (1, hal::format::Format::R8Unorm),
            ImageFormat::RG8 => (2, hal::format::Format::Rg8Unorm),
            ImageFormat::BGRA8 => (4, hal::format::Format::Bgra8Unorm),
            _ => unimplemented!("TODO image format missing"),
        };
        let upload_buffer = CopyBuffer::create(
            device,
            memory_types,
            hal::buffer::Usage::TRANSFER_SRC,
            1, // Data stride is 1, because we receive image data as [u8].
            (image_width * bytes_per_pixel) as usize,
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
            hal::image::ViewKind::D2Array,
            format,
            hal::image::Usage::TRANSFER_SRC | hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED | hal::image::Usage::COLOR_ATTACHMENT,
            hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
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

        if let Some(barrier) = self.core.transit(
            hal::image::Access::TRANSFER_WRITE,
            hal::image::Layout::TransferDstOptimal,
        ) {
            cmd_buffer.pipeline_barrier(
                hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT .. hal::pso::PipelineStage::TRANSFER,
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
        ) {
            cmd_buffer.pipeline_barrier(
                hal::pso::PipelineStage::TRANSFER .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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

pub struct VertexDataImage<B: hal::Backend> {
    pub core: ImageCore<B>,
    pub image_upload_buffer: CopyBuffer<B>,
    pub image_stride: usize,
    pub vecs_per_data: usize,
    pub image_width: u32,
    pub image_height: u32,
}

impl<B: hal::Backend> VertexDataImage<B> {
    pub fn create(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        data_stride: usize,
        image_width: u32,
        image_height: u32,
        pitch_alignment: usize,
    ) -> Self {
        let kind = hal::image::Kind::D2(
            image_width as hal::image::Size,
            image_height as hal::image::Size,
            1,
            1,
        );
        let core = ImageCore::create(
            device,
            memory_types,
            kind,
            hal::image::ViewKind::D2,
            hal::format::Format::Rgba32Float,
            hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
            COLOR_RANGE,
        );
        let image_upload_buffer = CopyBuffer::create(
            device,
            memory_types,
            hal::buffer::Usage::TRANSFER_SRC,
            data_stride,
            image_width as usize,
            image_height as usize,
            pitch_alignment,
        );
        let image_stride = 4usize * mem::size_of::<f32>(); // Rgba32Float;

        VertexDataImage {
            core,
            image_upload_buffer,
            image_stride,
            vecs_per_data: data_stride / image_stride,
            image_width,
            image_height,
        }
    }

    pub fn update<T>(
        &mut self,
        device: &mut B::Device,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        image_offset: DeviceUintPoint,
        image_data: &[T],
        offset_alignment: usize,
    ) -> hal::command::Submit<B, hal::Graphics, hal::command::MultiShot, hal::command::Primary>
        where
            T: Copy,
    {
        let image_data_length = image_data.len() as u32;
        let image_row_length = self.image_width / self.vecs_per_data as u32;
        let mut needed_height = image_data_length / image_row_length;
        if image_data_length % image_row_length != 0 { needed_height += 1};
        if needed_height > self.image_height {
            unimplemented!("TODO: implement resize");
        }
        let image_data_width_in_bytes = (image_data_length as usize * self.image_upload_buffer.data_stride) as u64;
        let buffer_offset = (image_offset.y as u64 * self.image_upload_buffer.row_pitch_in_bytes as u64);
        assert_eq!(buffer_offset % (offset_alignment + 1) as u64, 0);
        let _ = self.image_upload_buffer.update(device, buffer_offset, image_data_width_in_bytes, image_data, offset_alignment);

        let mut cmd_buffer = cmd_pool.acquire_command_buffer(false);

        if let Some(barrier) = self.core.transit(
            hal::image::Access::TRANSFER_WRITE,
            hal::image::Layout::TransferDstOptimal
        ) {
            cmd_buffer.pipeline_barrier(
                hal::pso::PipelineStage::VERTEX_SHADER .. hal::pso::PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        cmd_buffer.copy_buffer_to_image(
            &self.image_upload_buffer.buffer,
            &self.core.image,
            hal::image::Layout::TransferDstOptimal,
            &[
                hal::command::BufferImageCopy {
                    buffer_offset,
                    buffer_width: self.image_width as u32,
                    buffer_height: needed_height,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0 .. 1,
                    },
                    image_offset: hal::image::Offset {
                        x: image_offset.x as i32,
                        y: image_offset.y as i32,
                        z: 0,
                    },
                    image_extent: hal::image::Extent {
                        width: self.image_width as u32,
                        height: needed_height,
                        depth: 1,
                    },
                },
            ],
        );

        if let Some(barrier) = self.core.transit(
            hal::image::Access::SHADER_READ,
            hal::image::Layout::ShaderReadOnlyOptimal
        ) {
            cmd_buffer.pipeline_barrier(
                hal::pso::PipelineStage::TRANSFER .. hal::pso::PipelineStage::VERTEX_SHADER,
                hal::memory::Dependencies::empty(),
                &[barrier],
            );
        }

        cmd_buffer.finish()
    }

    pub fn deinit(self, device: &B::Device) {
        self.core.deinit(device);
        self.image_upload_buffer.deinit(device);
    }
}

pub struct Buffer<B: hal::Backend> {
    pub memory: B::Memory,
    pub buffer: B::Buffer,
    pub data_stride: usize,
    state: Cell<hal::buffer::State>,
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
            state: Cell::new(hal::buffer::Access::empty())
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

    pub fn reset(&mut self) {
        self.size = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        for buffer in self.buffers {
            buffer.deinit(device);
        }
    }
}

pub struct Program<B: hal::Backend> {
    pub bindings_map: HashMap<String, u32>,
    pub descriptor_set_layout: B::DescriptorSetLayout,
    pub descriptor_pool: B::DescriptorPool,
    pub descriptor_set: B::DescriptorSet,
    pub pipeline_layout: B::PipelineLayout,
    pub pipelines: HashMap<(BlendState, DepthTest), B::GraphicsPipeline>,
    pub vertex_buffer: Buffer<B>,
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
                ShaderKind::ClipCache => ImageFormat::R8,
                ShaderKind::Cache(VertexArrayKind::Blur) if shader_name.contains("_a8") => ImageFormat::R8,
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

            //device.create_graphics_pipelines(&[pipeline_desc])
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

        let vertex_buffer_stride = mem::size_of::<Vertex>();
        let vertex_buffer_len = QUAD.len() * vertex_buffer_stride;

        let mut vertex_buffer = Buffer::create(
            device,
            memory_types,
            hal::buffer::Usage::VERTEX,
            vertex_buffer_stride,
            vertex_buffer_len,
        );

        vertex_buffer.update(device, 0, vertex_buffer_len as u64, &QUAD);

        let instance_buffer_stride = match *shader_kind {
            ShaderKind::Primitive |
            ShaderKind::Brush |
            ShaderKind::Text |
            ShaderKind::Cache(VertexArrayKind::Primitive) => mem::size_of::<PrimitiveInstance>(),
            ShaderKind::ClipCache | ShaderKind::Cache(VertexArrayKind::Clip) => mem::size_of::<ClipMaskInstance>(),
            ShaderKind::Cache(VertexArrayKind::Blur) => mem::size_of::<BlurInstance>(),
        };
        let instance_buffer_len = MAX_INSTANCE_COUNT * instance_buffer_stride;

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
            instance_buffer: InstanceBuffer::new(instance_buffer),
            locals_buffer,
            shader_name: String::from(shader_name),
        }
    }


    fn bind_instances<T>(
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
        u_mode: i32,
    ) {
        let locals_data = vec![
            Locals {
                uTransform: projection.to_row_arrays(),
                uDevicePixelRatio: 1.0,
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

    fn bind_textures(
        &mut self,
        device: &Device<B, hal::Graphics>,
    ) {
        const samplers: [(usize, &'static str); 6] = [(0, "Color0"), (1, "Color1"), (2, "Color2"), (3, "CacheA8"), (4, "CacheRGBA8"), (9, "SharedCacheA8")];
        for &(index, sampler) in samplers.iter() {
            if device.bound_textures[index] != 0 {
                self.bind_texture(device, &device.bound_textures[index], &device.bound_sampler[index], sampler);
            }
        }
    }

    fn bind_texture(&mut self, device: &Device<B, hal::Graphics>, id: &TextureId, sampler: &TextureFilter, binding: &'static str) {
        let sampler = match *sampler {
            TextureFilter::Linear | TextureFilter::Trilinear => &device.sampler_linear,
            TextureFilter::Nearest => &device.sampler_nearest,
        };
        device.device.write_descriptor_sets(vec![
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map[&("t".to_owned() + binding)],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Image(&device.images[id].core.view, device.images[id].core.state.get().1)
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

    pub fn bind<T>(
        &mut self,
        device: &Device<B, hal::Graphics>,
        projection: &Transform3D<f32>,
        u_mode: i32,
        instances: &[T],
    ) where
        T: Copy,
    {
        self.bind_instances(&device.device, instances);
        self.bind_locals(&device.device, &projection, u_mode);
        self.bind_textures(device);
    }

    pub fn init_vertex_data(
        &mut self,
        device: &B::Device,
        resource_cache: &B::ImageView,
        resource_cache_sampler: &B::Sampler,
        node_data: &B::ImageView,
        node_data_sampler: &B::Sampler,
        render_tasks: &B::ImageView,
        render_tasks_sampler: &B::Sampler,
        local_clip_rects: &B::ImageView,
        local_clip_rects_sampler: &B::Sampler,
    ) {
        device.write_descriptor_sets(vec![
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["tResourceCache"],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Image(resource_cache, hal::image::Layout::Undefined)
                ),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["sResourceCache"],
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Sampler(resource_cache_sampler)),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["tClipScrollNodes"],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Image(node_data, hal::image::Layout::Undefined)
                ),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["sClipScrollNodes"],
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Sampler(node_data_sampler)),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["tRenderTasks"],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Image(render_tasks, hal::image::Layout::Undefined)
                ),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["sRenderTasks"],
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Sampler(render_tasks_sampler)),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["tLocalClipRects"],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Image(local_clip_rects, hal::image::Layout::Undefined)
                ),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["sLocalClipRects"],
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Sampler(local_clip_rects_sampler)),
            },
        ]);
    }

    fn init_dither_data<'a>(
        &mut self,
        device: &B::Device,
        dither: &B::ImageView,
        dither_sampler: &B::Sampler,
    ) {
        device.write_descriptor_sets(vec![
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["tDither"],
                array_offset: 0,
                descriptors: Some(
                    hal::pso::Descriptor::Image(dither, hal::image::Layout::ShaderReadOnlyOptimal)
                ),
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_set,
                binding: self.bindings_map["sDither"],
                array_offset: 0,
                descriptors: Some(hal::pso::Descriptor::Sampler(dither_sampler)),
            },
        ]);
    }

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
    ) -> hal::command::Submit<B, hal::Graphics, hal::command::MultiShot, hal::command::Primary> {
        let mut cmd_buffer = cmd_pool.acquire_command_buffer(false);

        cmd_buffer.set_viewports(&[viewport.clone()]);
        cmd_buffer.set_scissors(&[viewport.rect]);
        cmd_buffer.bind_graphics_pipeline(
            &self.pipelines.get(&(blend_state, depth_test)).expect(&format!("The blend state {:?} with depth test {:?} not found for {} program!", blend_state, depth_test, self.shader_name)));
        cmd_buffer.bind_vertex_buffers(hal::pso::VertexBufferSet(vec![
            (&self.vertex_buffer.buffer, 0),
            (&self.instance_buffer.buffer.buffer, 0),
        ]));
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
            encoder.draw(0 .. 6, (self.instance_buffer.offset - self.instance_buffer.size) as u32 .. self.instance_buffer.offset as u32);
        }

        cmd_buffer.finish()
    }

    pub fn deinit(mut self, device: &B::Device) {
        self.vertex_buffer.deinit(device);
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
                Swizzle::NO,
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

pub struct Device<B: hal::Backend, C> {
    pub device: B::Device,
    pub memory_types: Vec<hal::MemoryType>,
    pub upload_memory_type: hal::MemoryTypeId,
    pub download_memory_type: hal::MemoryTypeId,
    pub limits: hal::Limits,
    pub surface_format: Format,
    pub depth_format: Format,
    pub queue_group: hal::QueueGroup<B, C>,
    pub command_pool: hal::CommandPool<B, C>,
    pub swap_chain: Box<B::Swapchain>,
    pub render_pass: RenderPass<B>,
    pub framebuffers: Vec<B::Framebuffer>,
    pub framebuffers_depth: Vec<B::Framebuffer>,
    pub frame_images: Vec<ImageCore<B>>,
    pub frame_depth: DepthBuffer<B>,
    pub viewport: hal::pso::Viewport,
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    pub resource_cache: VertexDataImage<B>,
    pub render_tasks: VertexDataImage<B>,
    pub local_clip_rects: VertexDataImage<B>,
    pub node_data: VertexDataImage<B>,
    dither_texture: Option<Texture>,
    pub upload_queue: Vec<hal::command::Submit<B, C, hal::command::MultiShot, hal::command::Primary>>,
    pub current_frame_id: usize,
    current_blend_state: BlendState,
    blend_color: ColorF,
    current_depth_test: DepthTest,
    // device state
    images: FastHashMap<TextureId, Image<B>>,
    fbos: FastHashMap<FBOId, Framebuffer<B>>,
    rbos: FastHashMap<RBOId, DepthBuffer<B>>,
    bound_textures: [u32; 16],
    bound_sampler: [TextureFilter; 16],
    bound_program: u32,
    //bound_vao: u32,
    bound_read_fbo: FBOId,
    bound_draw_fbo: FBOId,

    device_pixel_ratio: f32,
    upload_method: UploadMethod,

    // HW or API capabilties
    capabilities: Capabilities,

    // debug
    inside_frame: bool,

    // resources
    resource_override_path: Option<PathBuf>,

    max_texture_size: u32,
    renderer_name: String,
    //cached_programs: Option<Rc<ProgramCache>>,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: FrameId,

    // Supported features
    features: hal::Features,
    api_capabilities: ApiCapabilities,
}

impl<B: hal::Backend> Device<B, hal::Graphics> {
    pub fn new(
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        window: &winit::Window,
        adapter: hal::Adapter<B>,
        surface: &mut <B as hal::Backend>::Surface,
        api_capabilities: ApiCapabilities,
    ) -> Self {
        let renderer_name = "WIP".to_owned();

        let features = adapter.physical_device.features();

        let window_size = window.get_inner_size().unwrap();
        let pixel_width = window_size.0;
        let pixel_height = window_size.1;

        let (caps, formats) = surface.capabilities_and_formats(&adapter.physical_device);
        let surface_format = formats
            .map_or(
                //hal::format::Format::Rgba8Srgb,
                hal::format::Format::Bgra8Unorm,
                |formats| {
                    formats
                        .into_iter()
                        .find(|format| {
                            //format.base_format().1 == ChannelType::Srgb
                            //format.base_format().1 == ChannelType::Unorm
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
        let max_texture_size = 4096u32;

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

        let (device, mut queue_group) =
            adapter.open_with(1, |family| {
                surface.supports_queue_family(family)
            }).unwrap();

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
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::ColorAttachmentOptimal .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_bgra8 = hal::pass::Attachment {
                format: Some(hal::format::Format::Bgra8Unorm),
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::ColorAttachmentOptimal .. hal::image::Layout::ColorAttachmentOptimal,
            };

            let attachment_depth = hal::pass::Attachment {
                format: Some(depth_format),
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::DontCare,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::Layout::Undefined .. hal::image::Layout::DepthStencilAttachmentOptimal,
            };

            let subpass_r8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                preserves: &[],
            };

            let subpass_depth_r8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                preserves: &[],
            };

            let subpass_bgra8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                preserves: &[],
            };

            let subpass_depth_bgra8 = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, hal::image::Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
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

        let resource_cache = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 4]>(),
            MAX_VERTEX_TEXTURE_WIDTH as u32,
            MAX_VERTEX_TEXTURE_WIDTH as u32,
            (limits.min_buffer_copy_pitch_alignment - 1) as usize,
        );

        let render_tasks = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 12]>(),
            RENDER_TASK_TEXTURE_WIDTH as u32,
            TEXTURE_HEIGHT as u32,
            (limits.min_buffer_copy_pitch_alignment - 1) as usize,
        );

        let local_clip_rects = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 4]>(),
            CLIP_RECTS_TEXTURE_WIDTH as u32,
            TEXTURE_HEIGHT as u32,
            (limits.min_buffer_copy_pitch_alignment - 1) as usize,
        );

        let node_data = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 36]>(),
            NODE_TEXTURE_WIDTH as u32,
            TEXTURE_HEIGHT as u32,
            (limits.min_buffer_copy_pitch_alignment - 1) as usize,
        );

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
            swap_chain: Box::new(swap_chain),
            render_pass,
            framebuffers,
            framebuffers_depth,
            frame_images,
            frame_depth,
            viewport,
            sampler_linear,
            sampler_nearest,
            resource_cache,
            render_tasks,
            local_clip_rects,
            node_data,
            dither_texture: None,
            upload_queue: Vec::new(),
            current_frame_id: 0,
            current_blend_state: BlendState::Off,
            current_depth_test: DepthTest::Off,
            blend_color: ColorF::new(0.0, 0.0, 0.0, 0.0),
            resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            upload_method,
            inside_frame: false,

            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },

            images: FastHashMap::default(),
            fbos: FastHashMap::default(),
            rbos: FastHashMap::default(),
            bound_textures: [0; 16],
            bound_sampler: [TextureFilter::Linear; 16],
            bound_program: 0,
            //bound_vao: 0,
            bound_read_fbo: DEFAULT_READ_FBO,
            bound_draw_fbo: DEFAULT_DRAW_FBO,

            max_texture_size,
            renderer_name,
            //cached_programs,
            frame_id: FrameId(0),
            features,
            api_capabilities,
        }
    }

    pub fn update_resource_cache(&mut self, rect: DeviceUintRect, gpu_data: &[[f32; 4]]) {
        debug_assert_eq!(gpu_data.len(), 1024);
        self.upload_queue
            .push(self.resource_cache.update(
                &mut self.device,
                &mut self.command_pool,
                rect.origin,
                gpu_data,
                (self.limits.min_buffer_copy_offset_alignment - 1) as usize,
            ));
    }

    pub fn update_render_tasks(&mut self, task_data: &[[f32; 12]]) {
        self.upload_queue
            .push(self.render_tasks.update(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                task_data,
                (self.limits.min_buffer_copy_offset_alignment - 1) as usize,
            ));
    }

    pub fn update_local_rects(&mut self, local_data: &[[f32; 4]]) {
        self.upload_queue
            .push(self.local_clip_rects.update(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                local_data,
                (self.limits.min_buffer_copy_offset_alignment - 1) as usize,
            ));
    }

    pub fn update_node_data(&mut self, node_data: &[[f32; 36]]) {
        self.upload_queue
            .push(self.node_data.update(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                node_data,
                (self.limits.min_buffer_copy_offset_alignment - 1) as usize,
            ));
    }

    pub fn create_program(
        &mut self,
        pipeline_requirements: PipelineRequirements,
        shader_name: &str,
        shader_kind: &ShaderKind,
    ) -> Program<B> {
        let mut program = Program::create(
            pipeline_requirements,
            &self.device,
            &self.memory_types,
            shader_name,
            shader_kind,
            &self.render_pass,
        );
        program.init_vertex_data(
            &self.device,
            &self.resource_cache.core.view,
            &self.sampler_nearest,
            &self.node_data.core.view,
            &self.sampler_nearest,
            &self.render_tasks.core.view,
            &self.sampler_nearest,
            &self.local_clip_rects.core.view,
            &self.sampler_nearest,
        );

        if shader_name.contains("dithering") {
            if self.dither_texture.is_none() {
                self.dither_texture = Some(self.create_dither_texture());
            }
            let dither_text_id = self.dither_texture.as_ref().unwrap().id;
            program.init_dither_data(
                &self.device,
                &self.images[&dither_text_id].core.view,
                &self.sampler_nearest,
            );
        }
        program
    }

    fn create_dither_texture(&mut self) -> Texture {
        let dither_matrix: [u8; 64] = [
            42,
            26,
            38,
            22,
            41,
            25,
            37,
            21,
            10,
            58,
            06,
            54,
            09,
            57,
            05,
            53,
            34,
            18,
            46,
            30,
            33,
            17,
            45,
            29,
            02,
            50,
            14,
            62,
            01,
            49,
            13,
            61,
            40,
            24,
            36,
            20,
            43,
            27,
            39,
            23,
            08,
            56,
            04,
            52,
            11,
            59,
            07,
            55,
            32,
            16,
            44,
            28,
            35,
            19,
            47,
            31,
            00,
            48,
            12,
            60,
            03,
            51,
            15,
            63
        ];

        let mut texture = self.create_texture(ImageFormat::R8);
        self.init_texture(
            &mut texture,
            8,
            8,
            TextureFilter::Nearest,
            None,
            1,
            Some(&dither_matrix),
        );
        texture
    }

    pub fn draw(
        &mut self,
        program: &mut Program<B>,
        //blend_mode: &BlendMode,
        //enable_depth_write: bool
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
            program.submit(
                &mut self.command_pool,
                self.viewport.clone(),
                rp,
                &fb,
                &vec![],
                self.current_blend_state,
                self.blend_color,
                self.current_depth_test,
            )
        };

        self.upload_queue.push(submit);
        self.submit_to_gpu();
    }

    /*pub fn gl(&self) -> &gl::Gl {
        &*self.gl
    }

    pub fn rc_gl(&self) -> &Rc<gl::Gl> {
        &self.gl
    }*/

    pub fn set_device_pixel_ratio(&mut self, ratio: f32) {
        self.device_pixel_ratio = ratio;
    }

    /*pub fn update_program_cache(&mut self, cached_programs: Rc<ProgramCache>) {
        self.cached_programs = Some(cached_programs);
    }*/

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub fn reset_state(&mut self) {
        self.bound_textures = [0; 16];
        self.bound_sampler = [TextureFilter::Linear; 16];
        //self.bound_vao = 0;
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
        self.reset_image_buffer_offsets();
    }

    pub fn reset_image_buffer_offsets(&mut self) {
        for img in self.images.values_mut() {
            img.upload_buffer.reset();
        }
    }

    pub fn begin_frame(&mut self) -> FrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        // Texture state
        for i in 0 .. self.bound_textures.len() {
            self.bound_textures[i] = 0;
            self.bound_sampler[i] = TextureFilter::Linear;
            //self.gl.active_texture(gl::TEXTURE0 + i as u32);
            //self.gl.bind_texture(gl::TEXTURE_2D, 0);
        }

        // Shader state
        self.bound_program = 0;
        //self.gl.use_program(0);

        // Vertex state
        //self.bound_vao = 0;
        //self.gl.bind_vertex_array(0);

        // FBO state
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;

        // Pixel op state
        //self.gl.pixel_store_i(gl::UNPACK_ALIGNMENT, 1);
        //self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);

        // Default is sampler 0, always
        //self.gl.active_texture(gl::TEXTURE0);

        self.frame_id
    }

    fn bind_texture_impl(&mut self, slot: TextureSlot, id: u32, sampler: TextureFilter) {
        debug_assert!(self.inside_frame);

        if self.bound_textures[slot.0] != id {
            self.bound_textures[slot.0] = id;
            self.bound_sampler[slot.0] = sampler;
            //self.gl.active_texture(gl::TEXTURE0 + slot.0 as u32);
            //self.gl.bind_texture(target, id);
            //self.gl.active_texture(gl::TEXTURE0);
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

        if let Some((texture, _layer)) = texture_and_layer {
            let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);
            let rbos = &self.rbos;
            //TODO: transit only specific layer
            if let Some(barrier) = self.images[&texture.id].core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
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
            /*self.gl.viewport(
                0,
                0,
                dimensions.width as _,
                dimensions.height as _,
            );*/
        }
    }

    pub fn create_fbo_for_external_texture(&mut self, texture_id: u32) -> FBOId {
        /*let fbo = FBOId(self.gl.gen_framebuffers(1)[0]);
        fbo.bind(self.gl(), FBOTarget::Draw);
        self.gl.framebuffer_texture_2d(
            gl::DRAW_FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            gl::TEXTURE_2D,
            texture_id,
            0,
        );
        self.bound_draw_fbo.bind(self.gl(), FBOTarget::Draw);
        fbo*/
        FBOId(0)
    }

    pub fn delete_fbo(&mut self, fbo: FBOId) {
        //self.gl.delete_framebuffers(&[fbo.0]);
    }

    pub fn bind_external_draw_target(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
            //fbo_id.bind(self.gl(), FBOTarget::Draw);
        }
    }

    /*pub fn bind_program(&mut self, program: &Program) {
        debug_assert!(self.inside_frame);

        if self.bound_program != program.id {
            //self.gl.use_program(program.id);
            self.bound_program = program.id;
        }
    }*/
    fn generate_texture_id(&mut self) -> TextureId {
        let mut rng = rand::thread_rng();
        let mut texture_id = INVALID_TEXTURE_ID + 1;
        while self.images.contains_key(&texture_id) {
            texture_id = rng.gen_range::<u32>(INVALID_TEXTURE_ID + 1, u32::max_value());
        }
        texture_id
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

    pub fn create_texture(&mut self, format: ImageFormat) -> Texture {
        Texture {
            id: 0,
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

    fn set_texture_parameters(&mut self, /*target: TextureTarget,*/ filter: TextureFilter) {
        /*let mag_filter = match filter {
            TextureFilter::Nearest => gl::NEAREST,
            TextureFilter::Linear | TextureFilter::Trilinear => gl::LINEAR,
        };

        let min_filter = match filter {
            TextureFilter::Nearest => gl::NEAREST,
            TextureFilter::Linear => gl::LINEAR,
            TextureFilter::Trilinear => gl::LINEAR_MIPMAP_LINEAR,
        };

        self.gl
            .tex_parameter_i(target, gl::TEXTURE_MAG_FILTER, mag_filter as gl::GLint);
        self.gl
            .tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, min_filter as gl::GLint);

        self.gl
            .tex_parameter_i(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
        self.gl
            .tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);*/
    }

    /// Resizes a texture with enabled render target views,
    /// preserves the data by blitting the old texture contents over.
    /*pub fn resize_renderable_texture(
        &mut self,
        texture: &mut Texture,
        new_size: DeviceUintSize,
    ) {
        debug_assert!(self.inside_frame);

        let old_size = texture.get_dimensions();
        let old_fbos = mem::replace(&mut texture.fbo_ids, Vec::new());
        let old_texture_id = mem::replace(&mut texture.id, 0/*self.gl.gen_textures(1)[0]*/);

        texture.width = new_size.width;
        texture.height = new_size.height;
        let rt_info = texture.render_target
            .clone()
            .expect("Only renderable textures are expected for resize here");

        self.bind_texture(DEFAULT_TEXTURE, texture);
        self.set_texture_parameters(/*texture.target,*/ texture.filter);
        self.update_target_storage(texture, &rt_info, true);

        let rect = DeviceIntRect::new(DeviceIntPoint::zero(), old_size.to_i32());
        for (read_fbo, &draw_fbo) in old_fbos.into_iter().zip(&texture.fbo_ids) {
            self.bind_read_target_impl(read_fbo);
            self.bind_draw_target_impl(draw_fbo);
            self.blit_render_target(rect, rect);
            self.delete_fbo(read_fbo);
        }
        //self.gl.delete_textures(&[old_texture_id]);
        self.bind_read_target(None);
    }*/

    pub fn init_texture(
        &mut self,
        texture: &mut Texture,
        mut width: u32,
        mut height: u32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
        pixels: Option<&[u8]>,
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

        self.bind_texture(DEFAULT_TEXTURE, texture);
        self.set_texture_parameters(/*texture.target,*/ filter);

        let rt_info = render_target.unwrap_or(RenderTargetInfo { has_depth: false});

        if texture.id == 0 {
            let id = self.generate_texture_id();
            texture.id = id;
        } else {
            self.free_image(texture);
        }
        assert_eq!(self.images.contains_key(&texture.id), false);
        let img = Image::new(
            &self.device,
            &self.memory_types,
            texture.format,
            texture.width,
            texture.height,
            texture.layer_count,
            (self.limits.min_buffer_copy_pitch_alignment - 1) as usize,
        );

        assert_eq!(texture.fbo_ids.len(), 0);

        let (mut depth_rb, allocate_depth) = match texture.depth_rb {
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
                depth_rb = RBOId(0);
                texture.depth_rb = None;
            }
        }

        let new_fbos = self.generate_fbo_ids(texture.layer_count);

        for i in 0..texture.layer_count as u16 {
            let (rbo_id, depth) = match texture.depth_rb {
                Some(rbo_id) => (rbo_id.clone(), Some(&self.rbos[&rbo_id].core.view)),
                None => (RBOId(0), None)
            };
            let fbo = Framebuffer::new(&self.device, &texture, &img, i, &self.render_pass, rbo_id.clone(),depth);
            self.fbos.insert(new_fbos[i as usize],fbo);
            texture.fbo_ids.push(new_fbos[i as usize]);
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
                            data,
                            (self.limits.min_buffer_copy_offset_alignment - 1) as usize,
                        )
                );
        }
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

        {
            let mut barriers = Vec::new();
            barriers.extend(src_img.transit(hal::image::Access::TRANSFER_READ, hal::image::Layout::TransferSrcOptimal));
            barriers.extend(dest_img.transit(hal::image::Access::TRANSFER_WRITE, hal::image::Layout::TransferDstOptimal));
            if !barriers.is_empty() {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &barriers,
                );
            }
        }
        // TODO remove the first condition if other platforms are supported
        if self.api_capabilities.contains(ApiCapabilities::BLITTING) && src_rect.size != dest_rect.size {
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
            barriers.extend(src_img.transit(hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE, hal::image::Layout::ColorAttachmentOptimal));
            barriers.extend(dest_img.transit(hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE, hal::image::Layout::ColorAttachmentOptimal));
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
        /*if let Some(RBOId(depth_rb)) = texture.depth_rb.take() {
            self.gl.delete_renderbuffers(&[depth_rb]);
        }*/

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

    #[cfg(feature = "capture")]
    pub fn delete_external_texture(&mut self, mut external: ExternalTexture) {
        self.bind_external_texture(DEFAULT_TEXTURE, &external);
        //Note: the format descriptor here doesn't really matter
        /*self.free_texture_storage_impl(external.target, FormatDesc {
            internal: gl::R8 as _,
            external: gl::RED,
            pixel_type: gl::UNSIGNED_BYTE,
        });*/
        //self.gl.delete_textures(&[external.id]);
        external.id = 0;
    }

    pub fn create_pbo(&mut self) -> PBO {
        //let id = self.gl.gen_buffers(1)[0];
        PBO { id: 0 }
    }

    pub fn delete_pbo(&mut self, mut pbo: PBO) {
        //self.gl.delete_buffers(&[pbo.id]);
        pbo.id = 0;
    }

    pub fn upload_texture(
        &mut self,
        texture: &Texture,
        rect: DeviceUintRect,
        layer_index: i32,
        stride: Option<u32>,
        data: &[u8],
    ) {
        let data_stride: usize = match texture.format {
            ImageFormat::R8 => 1,
            ImageFormat::RG8 => 2,
            ImageFormat::BGRA8 => 4,
            _ => unimplemented!("TODO image format missing"),
         };
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
        self.upload_queue
            .push(
                self.images
                    .get_mut(&texture.id)
                    .expect("Texture not found.")
                    .update(
                        &mut self.device,
                        &mut self.command_pool,
                        rect,
                        layer_index,
                        data,
                        (self.limits.min_buffer_copy_offset_alignment - 1) as usize,
                    )
            );
    }

    pub fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8> {
        /*let desc = gl_describe_format(self.gl(), img_desc.format);
        self.gl.read_pixels(
            0, 0,
            img_desc.width as i32,
            img_desc.height as i32,
            desc.external,
            desc.pixel_type,
        )*/
        vec!()
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
        let image = &self.frame_images[self.current_frame_id];
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

            barriers.extend(download_buffer.transit(hal::buffer::Access::TRANSFER_WRITE));
            barriers.extend(image.transit(hal::image::Access::TRANSFER_READ, hal::image::Layout::TransferSrcOptimal));
            if !barriers.is_empty() {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &barriers
                );
            }

            let buffer_width = rect.size.width * bytes_per_pixel as u32;
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
                        layers: 0 .. 1,
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
            ) {
                cmd_buffer.pipeline_barrier(
                    hal::pso::PipelineStage::TRANSFER .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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
            for (i, d) in reader.iter().enumerate() {
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
    pub fn get_tex_image_into(
        &mut self,
        texture: &Texture,
        format: ImageFormat,
        output: &mut [u8],
    ) {
        /*self.bind_texture(DEFAULT_TEXTURE, texture);
        let desc = gl_describe_format(self.gl(), format);
        self.gl.get_tex_image_into_buffer(
            texture.target,
            0,
            desc.external,
            desc.pixel_type,
            output,
        );*/
    }

    /// Attaches the provided texture to the current Read FBO binding.
    fn attach_read_texture_raw(
        &mut self, texture_id: u32, layer_id: i32
    ) {
        /*
        self.gl.framebuffer_texture_layer(
            gl::READ_FRAMEBUFFER,
            gl::COLOR_ATTACHMENT0,
            texture_id,
            0,
            layer_id,
        )
        */
    }

    pub fn attach_read_texture_external(
        &mut self, texture_id: u32, layer_id: i32
    ) {
        self.attach_read_texture_raw(texture_id, layer_id)
    }

    pub fn attach_read_texture(&mut self, texture: &Texture, layer_id: i32) {
        self.attach_read_texture_raw(texture.id, layer_id)
    }

    pub fn update_instances<V>(
        &mut self,
        instances: &[V],
        usage_hint: VertexUsageHint,
    ) {
    }

    pub fn draw_triangles_u16(&mut self, first_vertex: i32, index_count: i32) {
        debug_assert!(self.inside_frame);
        /*self.gl.draw_elements(
            gl::TRIANGLES,
            index_count,
            gl::UNSIGNED_SHORT,
            first_vertex as u32 * 2,
        );*/
    }

    pub fn draw_triangles_u32(&mut self, first_vertex: i32, index_count: i32) {
        debug_assert!(self.inside_frame);
        /*self.gl.draw_elements(
            gl::TRIANGLES,
            index_count,
            gl::UNSIGNED_INT,
            first_vertex as u32 * 4,
        );*/
    }

    pub fn draw_nonindexed_points(&mut self, first_vertex: i32, vertex_count: i32) {
        debug_assert!(self.inside_frame);
        //self.gl.draw_arrays(gl::POINTS, first_vertex, vertex_count);
    }

    pub fn draw_nonindexed_lines(&mut self, first_vertex: i32, vertex_count: i32) {
        debug_assert!(self.inside_frame);
        //self.gl.draw_arrays(gl::LINES, first_vertex, vertex_count);
    }

    pub fn draw_indexed_triangles_instanced_u16(&mut self, index_count: i32, instance_count: i32) {
        debug_assert!(self.inside_frame);
        /*self.gl.draw_elements_instanced(
            gl::TRIANGLES,
            index_count,
            gl::UNSIGNED_SHORT,
            0,
            instance_count,
        );*/
    }

    pub fn end_frame(&mut self) {
        self.bind_draw_target(None, None);
        self.bind_read_target(None);

        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        //self.gl.bind_texture(gl::TEXTURE_2D, 0);
        //self.gl.use_program(0);

        /*for i in 0 .. self.bound_textures.len() {
            self.gl.active_texture(gl::TEXTURE0 + i as u32);
            self.gl.bind_texture(gl::TEXTURE_2D, 0);
        }*/

        //self.gl.active_texture(gl::TEXTURE0);

        self.frame_id.0 += 1;
    }

    pub fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<DeviceIntRect>,
    ) {
        let mut cmd_buffer = self.command_pool.acquire_command_buffer(false);

        if let Some(rect) = rect {
            cmd_buffer.set_scissors(&[
                hal::pso::Rect {
                    x: rect.origin.x as u16,
                    y: rect.origin.y as u16,
                    w: rect.size.width as u16,
                    h: rect.size.height as u16,
                },
            ]);
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
            if let Some(barrier) = img.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::TransferDstOptimal,
            ) {
                cmd_buffer.pipeline_barrier(
                    hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.clear_color_image(
                &img.image,
                hal::image::Layout::TransferDstOptimal,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::COLOR,
                    levels: 0 .. 1,
                    layers: layer .. layer+1,
                },
                hal::command::ClearColor::Float([color[0], color[1], color[2], color[3]]),
            );
            if let Some(barrier) = img.transit(
                hal::image::Access::empty(),
                hal::image::Layout::ColorAttachmentOptimal,
            ) {
                cmd_buffer.pipeline_barrier(
                    hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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
            ) {
                cmd_buffer.pipeline_barrier(
                    hal::pso::PipelineStage::EARLY_FRAGMENT_TESTS .. hal::pso::PipelineStage::EARLY_FRAGMENT_TESTS,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.clear_depth_stencil_image(
                &dimg.image,
                hal::image::Layout::TransferDstOptimal,
                hal::image::SubresourceRange {
                    aspects: hal::format::Aspects::DEPTH,
                    levels: 0 .. 1,
                    layers: 0 .. 1,
                },
                hal::command::ClearDepthStencil(depth, 0)
            );
            if let Some(barrier) = dimg.transit(
                hal::image::Access::empty(),
                hal::image::Layout::DepthStencilAttachmentOptimal,
            ) {
                cmd_buffer.pipeline_barrier(
                    hal::pso::PipelineStage::EARLY_FRAGMENT_TESTS .. hal::pso::PipelineStage::EARLY_FRAGMENT_TESTS,
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

    pub fn set_depth_func(&self, _depth_func: DepthFunction) {
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
        //self.gl.disable(gl::STENCIL_TEST);
    }

    pub fn set_scissor_rect(&self, rect: DeviceIntRect) {
        /*self.gl.scissor(
            rect.origin.x,
            rect.origin.y,
            rect.size.width,
            rect.size.height,
        );*/
    }

    pub fn enable_scissor(&self) {
        //self.gl.enable(gl::SCISSOR_TEST);
    }

    pub fn disable_scissor(&self) {
        //self.gl.disable(gl::SCISSOR_TEST);
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
        self.blend_color = color;
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
            ) {
                cmd_buffer.pipeline_barrier(
                    hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT .. hal::pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
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
        self.device.destroy_fence(frame_fence);
        self.device.destroy_semaphore(frame_semaphore);
    }

    pub fn deinit(self) {
        if let Some(mut texture) = self.dither_texture {
            texture.id = 0;
        }
        self.device
            .destroy_command_pool(self.command_pool.downgrade());
        self.render_pass.deinit(&self.device);
        for framebuffer in self.framebuffers {
            self.device.destroy_framebuffer(framebuffer);
        }
        for image in self.frame_images {
            image.deinit(&self.device);
        }
        for (_, image) in self.images {
            image.deinit(&self.device);
        }
        for (_, rbo) in self.rbos {
            rbo.deinit(&self.device);
        }
        self.resource_cache.deinit(&self.device);
        self.render_tasks.deinit(&self.device);
        self.local_clip_rects.deinit(&self.device);
        self.node_data.deinit(&self.device);
    }
}

