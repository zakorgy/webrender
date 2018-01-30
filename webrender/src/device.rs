/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source;
use api::{ColorF, ImageDescriptor, ImageFormat};
use api::{DeviceIntPoint, DeviceIntRect, DeviceUintPoint, DeviceUintRect, DeviceUintSize};
use api::TextureTarget;
use euclid::Transform3D;
//use gleam::gl;
use internal_types::{FastHashMap, RenderTargetInfo};
use serde;
use std::collections::HashMap;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::marker::PhantomData;
use std::mem;
use std::ops::Add;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;
use std::thread;

use hal;
use winit;
use back;

// gfx-hal
use hal::pso::{AttributeDesc, DescriptorRangeDesc, DescriptorSetLayoutBinding, VertexBufferDesc};
use hal::{Device as BackendDevice, Instance, PhysicalDevice, QueueFamily, Surface, Swapchain};
use hal::{Backbuffer, DescriptorPool, FrameSync, Gpu, Primitive, SwapchainConfig};
use hal::format::{ChannelType, Swizzle};
use hal::pass::Subpass;
use hal::pso::PipelineStage;
use hal::queue::Submission;
use ron::de::from_reader;

pub const NODE_TEXTURE_WIDTH: usize = 1020; // 204 * ( 20 / 4)
pub const RENDER_TASK_TEXTURE_WIDTH: usize = 1023; // 341 * ( 12 / 4 )
pub const CLIP_RECTS_TEXTURE_WIDTH: usize = 1024;
pub const TEXTURE_HEIGHT: usize = 8;
pub const MAX_INSTANCE_COUNT: usize = 1024;

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::AspectFlags::COLOR,
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
    aData0: [i32; 4],
    aData1: [i32; 4],
}

#[derive(Deserialize)]
pub struct PipelineRequirements {
    pub attribute_descriptors: Vec<AttributeDesc>,
    pub bindings_map: HashMap<String, usize>,
    pub descriptor_range_descriptors: Vec<DescriptorRangeDesc>,
    pub descriptor_set_layouts: Vec<DescriptorSetLayoutBinding>,
    pub vertex_buffer_descriptors: Vec<VertexBufferDesc>,
}

impl PrimitiveInstance {
    pub fn new(data: [i32; 8]) -> PrimitiveInstance {
        PrimitiveInstance {
            aData0: [data[0], data[1], data[2], data[3]],
            aData1: [data[4], data[5], data[6], data[7]],
        }
    }
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
    let mut file = File::open(path_str).unwrap();
    let mut shader = Vec::new();
    file.read_to_end(&mut shader).unwrap();
    shader
}

#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
#[cfg_attr(feature = "capture", derive(Deserialize, Serialize))]
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
#[cfg_attr(feature = "capture", derive(Deserialize, Serialize))]
pub enum TextureFilter {
    Nearest,
    Linear,
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

#[cfg_attr(feature = "capture", derive(Clone))]
pub struct ExternalTexture {
    id: u32,
    target: TextureTarget,
}

impl ExternalTexture {
    pub fn new(id: u32, target: TextureTarget) -> Self {
        ExternalTexture {
            id,
            target,
        }
    }

    #[cfg(feature = "capture")]
    pub fn internal_id(&self) -> u32 {
        self.id
    }
}

pub struct Texture {
    id: u32,
    target: TextureTarget,
    layer_count: i32,
    format: ImageFormat,
    width: u32,
    height: u32,
    filter: TextureFilter,
    render_target: Option<RenderTargetInfo>,
    fbo_ids: Vec<FBOId>,
    depth_rb: Option<RBOId>,
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

    #[cfg(feature = "capture")]
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

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error mssage
    Link(String, String),        // name, error message
}

pub struct VertexDataImage<B: hal::Backend> {
    pub image_upload_buffer: Buffer<B>,
    pub image: B::Image,
    pub image_memory: B::Memory,
    pub image_srv: B::ImageView,
    pub image_stride: usize,
    pub mem_stride: usize,
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
    ) -> VertexDataImage<B> {
        let image_upload_buffer = Buffer::create(
            device,
            memory_types,
            hal::buffer::Usage::TRANSFER_SRC,
            data_stride,
            (image_width * image_height) as usize,
        );
        let kind = hal::image::Kind::D2(
            image_width as hal::image::Size,
            image_height as hal::image::Size,
            hal::image::AaMode::Single,
        );
        let image_unbound = device
            .create_image(
                kind,
                1,
                hal::format::Format::Rgba32Float,
                hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED,
            )
            .unwrap(); // TODO: usage
        println!("{:?}", image_unbound);
        let image_req = device.get_image_requirements(&image_unbound);

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                image_req.type_mask & (1 << id) != 0
                    && mem_type
                    .properties
                    .contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let image_memory = device.allocate_memory(device_type, image_req.size).unwrap();

        let image = device
            .bind_image_memory(&image_memory, 0, image_unbound)
            .unwrap();
        let image_srv = device
            .create_image_view(
                &image,
                hal::format::Format::Rgba32Float,
                Swizzle::NO,
                COLOR_RANGE.clone(),
            )
            .unwrap();

        VertexDataImage {
            image_upload_buffer,
            image,
            image_memory,
            image_srv,
            image_stride: 4usize,              // Rgba
            mem_stride: mem::size_of::<f32>(), // Float
            image_width,
            image_height,
        }
    }

    pub fn update_buffer_and_submit_upload<T>(
        &mut self,
        device: &mut B::Device,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        image_offset: DeviceUintPoint,
        image_data: &[T],
    ) -> hal::command::Submit<B, hal::queue::Graphics>
        where
            T: Copy,
    {
        let needed_height = (image_data.len() * self.image_upload_buffer.data_stride)
            / (self.image_width as usize * self.image_stride) + 1;
        if needed_height > self.image_height as usize {
            unimplemented!("TODO: implement resize");
        }
        let buffer_height = needed_height as u64;
        let buffer_width = (image_data.len() * self.image_upload_buffer.data_stride) as u64;
        let buffer_offset = (image_offset.y * buffer_width as u32) as u64;
        self.image_upload_buffer
            .update(device, buffer_offset, buffer_width, image_data);

        let mut cmd_buffer = cmd_pool.acquire_command_buffer();

        let image_barrier = hal::memory::Barrier::Image {
            states: (
                hal::image::Access::TRANSFER_WRITE,
                hal::image::ImageLayout::TransferDstOptimal,
            )
                .. (
                hal::image::Access::TRANSFER_WRITE,
                hal::image::ImageLayout::TransferDstOptimal,
            ),
            target: &self.image,
            range: COLOR_RANGE.clone(),
        };
        cmd_buffer.pipeline_barrier(
            hal::pso::PipelineStage::TOP_OF_PIPE .. hal::pso::PipelineStage::TRANSFER,
            &[image_barrier],
        );

        cmd_buffer.copy_buffer_to_image(
            &self.image_upload_buffer.buffer,
            &self.image,
            hal::image::ImageLayout::TransferDstOptimal,
            &[
                hal::command::BufferImageCopy {
                    buffer_offset,
                    buffer_width: buffer_width as u32,
                    buffer_height: buffer_height as u32,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::AspectFlags::COLOR,
                        level: 0,
                        layers: 0 .. 1,
                    },
                    image_offset: hal::command::Offset {
                        x: image_offset.x as i32,
                        y: image_offset.y as i32,
                        z: 0,
                    },
                    image_extent: hal::device::Extent {
                        width: buffer_width as u32,
                        height: buffer_height as u32,
                        depth: 1,
                    },
                },
            ],
        );

        let image_barrier = hal::memory::Barrier::Image {
            states: (
                hal::image::Access::TRANSFER_WRITE,
                hal::image::ImageLayout::TransferDstOptimal,
            )
                .. (
                hal::image::Access::SHADER_READ,
                hal::image::ImageLayout::ShaderReadOnlyOptimal,
            ),
            target: &self.image,
            range: COLOR_RANGE.clone(),
        };
        cmd_buffer.pipeline_barrier(
            hal::pso::PipelineStage::TRANSFER .. hal::pso::PipelineStage::VERTEX_SHADER,
            &[image_barrier],
        );
        cmd_buffer.finish()
    }

    pub fn deinit(self, device: &B::Device) {
        self.image_upload_buffer.deinit(device);
        device.destroy_image(self.image);
        device.destroy_image_view(self.image_srv);
        device.free_memory(self.image_memory);
    }
}

pub struct Buffer<B: hal::Backend> {
    pub memory: B::Memory,
    pub buffer: B::Buffer,
    pub data_stride: usize,
}

impl<B: hal::Backend> Buffer<B> {
    pub fn create(
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        usage: hal::buffer::Usage,
        data_stride: usize,
        data_len: usize,
    ) -> Buffer<B> {
        let buffer_size = data_stride * data_len;
        let buffer_type: hal::MemoryTypeId = memory_types
            .iter()
            .position(|mt| {
                mt.properties.contains(hal::memory::Properties::CPU_VISIBLE)
                //&&!mt.properties.contains(memory::Properties::CPU_CACHED)
            })
            .unwrap()
            .into();
        let (memory, buffer) = {
            let unbound_buffer = device.create_buffer(buffer_size as u64, usage).unwrap();
            let buffer_req = device.get_buffer_requirements(&unbound_buffer);
            let buffer_memory = device
                .allocate_memory(buffer_type, buffer_req.size)
                .unwrap();
            let buffer = device
                .bind_buffer_memory(&buffer_memory, 0, unbound_buffer)
                .unwrap();
            (buffer_memory, buffer)
        };
        Buffer {
            memory,
            buffer,
            data_stride,
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
        let mut data = device
            .acquire_mapping_writer::<T>(
                &self.buffer,
                buffer_offset .. (buffer_offset + buffer_width),
            )
            .unwrap();
        assert_eq!(data.len(), update_data.len());
        for (i, d) in update_data.iter().enumerate() {
            data[i] = *d;
        }
        device.release_mapping_writer(data);
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
    fn new(buffer: Buffer<B>) -> InstanceBuffer<B> {
        InstanceBuffer {
            buffer,
            size: 0,
            offset: 0,
        }
    }

    pub fn reset(&mut self) {
        self.size = 1;
        self.offset = 0;
    }

    pub fn deinit(self, device: &B::Device) {
        self.buffer.deinit(device);
    }
}

pub struct Program<B: hal::Backend> {
    pub bindings_map: HashMap<String, usize>,
    pub descriptor_set_layout: B::DescriptorSetLayout,
    pub descriptor_pool: B::DescriptorPool,
    pub descriptor_sets: Vec<B::DescriptorSet>,
    pub pipeline_layout: B::PipelineLayout,
    pub pipelines: Vec<B::GraphicsPipeline>,
    pub vertex_buffer: Buffer<B>,
    pub instance_buffer: InstanceBuffer<B>,
    pub locals_buffer: Buffer<B>,
}

impl<B: hal::Backend> Program<B> {
    pub fn create(
        pipeline_requirements: PipelineRequirements,
        device: &B::Device,
        memory_types: &[hal::MemoryType],
        shader_name: &str,
        render_pass: &B::RenderPass,
    ) -> Program<B> {
        #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
        let vs_module = device
            .create_shader_module(get_shader_source(shader_name, ".vert.spv").as_slice())
            .unwrap();
        #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
        let fs_module = device
            .create_shader_module(get_shader_source(shader_name, ".frag.spv").as_slice())
            .unwrap();

        let descriptor_set_layout = device.create_descriptor_set_layout(&pipeline_requirements.descriptor_set_layouts);
        let mut descriptor_pool =
            device.create_descriptor_pool(
                1, //The number of descriptor sets
                pipeline_requirements.descriptor_range_descriptors.as_slice(),
            );
        let descriptor_sets = descriptor_pool.allocate_sets(&[&descriptor_set_layout]);

        let pipeline_layout = device.create_pipeline_layout(&[&descriptor_set_layout], &[]);

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

            let subpass = Subpass {
                index: 0,
                main_pass: render_pass,
            };

            let mut pipeline_descriptor = hal::pso::GraphicsPipelineDesc::new(
                shader_entries,
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
                    hal::pso::BlendState::ALPHA,
                ));

            pipeline_descriptor.vertex_buffers = pipeline_requirements.vertex_buffer_descriptors;
            pipeline_descriptor.attributes = pipeline_requirements.attribute_descriptors;

            //device.create_graphics_pipelines(&[pipeline_desc])
            device
                .create_graphics_pipelines(&[pipeline_descriptor])
                .into_iter()
                .map(|pipeline| pipeline.unwrap())
                .collect()
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

        vertex_buffer.update(device, 0, vertex_buffer_len as u64, &vec![QUAD]);

        let instance_buffer_stride = mem::size_of::<PrimitiveInstance>();
        let instance_buffer_len = MAX_INSTANCE_COUNT * instance_buffer_stride;

        let instance_buffer = Buffer::create(
            device,
            memory_types,
            hal::buffer::Usage::VERTEX,
            instance_buffer_stride,
            instance_buffer_len,
        );

        let locals_buffer_stride = mem::size_of::<Locals>();
        let locals_buffer_len = locals_buffer_stride;

        let locals_buffer = Buffer::create(
            device,
            memory_types,
            hal::buffer::Usage::UNIFORM,
            locals_buffer_stride,
            locals_buffer_len,
        );

        let bindings_map = pipeline_requirements.bindings_map;
        device.update_descriptor_sets(&[
            hal::pso::DescriptorSetWrite {
                set: &descriptor_sets[0],
                binding: bindings_map["Locals"],
                array_offset: 0,
                write: hal::pso::DescriptorWrite::UniformBuffer(vec![
                    (&locals_buffer.buffer, 0 .. mem::size_of::<Locals>() as u64),
                ]),
            },
        ]);

        Program {
            bindings_map,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipelines,
            vertex_buffer,
            instance_buffer: InstanceBuffer::new(instance_buffer),
            locals_buffer,
        }
    }

    pub fn bind(
        &mut self,
        device: &B::Device,
        projection: &Transform3D<f32>,
        u_mode: i32,
        instances: &[PrimitiveInstance],
    ) {
        let data_stride = self.instance_buffer.buffer.data_stride;
        let offset = self.instance_buffer.offset as u64;
        self.instance_buffer.buffer.update(
            device,
            offset,
            (instances.len() * data_stride) as u64,
            &instances.to_owned(),
        );

        self.instance_buffer.size += instances.len();
        let locals_buffer_stride = mem::size_of::<Locals>();
        let locals_data = vec![
            Locals {
                uTransform: projection.post_scale(1.0, -1.0, 1.0).to_row_arrays(),
                uDevicePixelRatio: 1.0,
                uMode: u_mode,
            },
        ];
        self.locals_buffer.update(
            device,
            0,
            (locals_data.len() * locals_buffer_stride) as u64,
            &locals_data,
        );
    }

    pub fn init_vertex_data<'a>(
        &mut self,
        device: &B::Device,
        resource_cache: hal::pso::DescriptorWrite<'a, B>,
        resource_cache_sampler: hal::pso::DescriptorWrite<'a, B>,
        node_data: hal::pso::DescriptorWrite<'a, B>,
        node_data_sampler: hal::pso::DescriptorWrite<'a, B>,
        render_tasks: hal::pso::DescriptorWrite<'a, B>,
        render_tasks_sampler: hal::pso::DescriptorWrite<'a, B>,
        local_clip_rects: hal::pso::DescriptorWrite<'a, B>,
        local_clip_rects_sampler: hal::pso::DescriptorWrite<'a, B>,
    ) {
        device.update_descriptor_sets(&[
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["tResourceCache"],
                array_offset: 0,
                write: resource_cache,
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["sResourceCache"],
                array_offset: 0,
                write: resource_cache_sampler,
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["tClipScrollNodes"],
                array_offset: 0,
                write: node_data,
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["sClipScrollNodes"],
                array_offset: 0,
                write: node_data_sampler,
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["tRenderTasks"],
                array_offset: 0,
                write: render_tasks,
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["sRenderTasks"],
                array_offset: 0,
                write: render_tasks_sampler,
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["tLocalClipRects"],
                array_offset: 0,
                write: local_clip_rects,
            },
            hal::pso::DescriptorSetWrite {
                set: &self.descriptor_sets[0],
                binding: self.bindings_map["sLocalClipRects"],
                array_offset: 0,
                write: local_clip_rects_sampler,
            },
        ]);
    }

    pub fn submit(
        &mut self,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        viewport: hal::command::Viewport,
        render_pass: &B::RenderPass,
        frame_buffer: &B::Framebuffer,
        clear_values: &[hal::command::ClearValue],
    ) -> hal::command::Submit<B, hal::queue::Graphics> {
        let mut cmd_buffer = cmd_pool.acquire_command_buffer();

        cmd_buffer.set_viewports(&[viewport.clone()]);
        cmd_buffer.set_scissors(&[viewport.rect]);
        cmd_buffer.bind_graphics_pipeline(&self.pipelines[0]);
        cmd_buffer.bind_vertex_buffers(hal::pso::VertexBufferSet(vec![
            (&self.vertex_buffer.buffer, 0),
            (&self.instance_buffer.buffer.buffer, 0),
        ]));
        cmd_buffer.bind_graphics_descriptor_sets(
            &self.pipeline_layout,
            0,
            &self.descriptor_sets[0 .. 1],
        );

        {
            let mut encoder = cmd_buffer.begin_renderpass_inline(
                render_pass,
                frame_buffer,
                viewport.rect,
                clear_values,
            );
            encoder.draw(0 .. 6, 0 .. self.instance_buffer.size as u32);
        }

        cmd_buffer.finish()
    }

    pub fn deinit(mut self, device: &B::Device) {
        self.vertex_buffer.deinit(device);
        self.instance_buffer.buffer.deinit(device);
        self.locals_buffer.deinit(device);
        device.destroy_descriptor_pool(self.descriptor_pool);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout);
        device.destroy_pipeline_layout(self.pipeline_layout);
        for pipeline in self.pipelines.drain(..) {
            device.destroy_graphics_pipeline(pipeline);
        }
    }
}

pub struct Device<B: hal::Backend> {
    pub device: B::Device,
    pub memory_types: Vec<hal::MemoryType>,
    pub upload_memory_type: hal::MemoryTypeId,
    pub download_memory_type: hal::MemoryTypeId,
    pub limits: hal::Limits,
    pub queue_group: hal::QueueGroup<B, hal::queue::Graphics>,
    pub command_pool: hal::CommandPool<B, hal::queue::Graphics>,
    pub swap_chain: Box<B::Swapchain>,
    pub render_pass: B::RenderPass,
    pub framebuffers: Vec<B::Framebuffer>,
    pub frame_images: Vec<(B::Image, B::ImageView)>,
    pub viewport: hal::command::Viewport,
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    pub resource_cache: VertexDataImage<B>,
    pub render_tasks: VertexDataImage<B>,
    pub local_clip_rects: VertexDataImage<B>,
    pub node_data: VertexDataImage<B>,
    pub upload_queue: Vec<hal::command::Submit<B, hal::queue::Graphics>>,
    pub current_frame_id: usize,
    // device state
    bound_textures: [u32; 16],
    bound_program: u32,
    //bound_vao: u32,
    bound_read_fbo: FBOId,
    bound_draw_fbo: FBOId,
    default_read_fbo: u32,
    default_draw_fbo: u32,

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

    // GL extensions
    extensions: Vec<String>,
}

impl<B: hal::Backend> Device<B> {
    pub fn new(
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        window: &winit::Window,
        instance: &back::Instance,
        surface: &mut <back::Backend as hal::Backend>::Surface,
    ) -> Device<back::Backend> {
        let max_texture_size = 2048u32;
        let renderer_name = "WIP".to_owned();

        let mut extensions = Vec::new();

        let window_size = window.get_inner_size().unwrap();
        let pixel_width = window_size.0 as u16;
        let pixel_height = window_size.1 as u16;

        // instantiate backend
        let mut adapters = instance.enumerate_adapters();

        for adapter in &adapters {
            println!("{:?}", adapter.info);
        }

        let adapter = adapters.remove(0);
        let surface_format = surface
            .capabilities_and_formats(&adapter.physical_device)
            .1
            .map_or(
                //hal::format::Format::Rgba8Srgb,
                hal::format::Format::Rgba8Unorm,
                |formats| {
                    formats
                        .into_iter()
                        .find(|format| {
                            //format.base_format().1 == ChannelType::Srgb
                            format.base_format().1 == ChannelType::Unorm
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
            .get_limits();

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

        let Gpu {
            device,
            mut queue_groups,
        } = adapter
            .open_with(|family| {
                if family.supports_graphics() && surface.supports_queue_family(family) {
                    Some(1)
                } else {
                    None
                }
            })
            .unwrap();

        let queue_group = hal::QueueGroup::<_, hal::Graphics>::new(queue_groups.remove(0));
        let mut command_pool = device.create_command_pool_typed(
            &queue_group,
            hal::pool::CommandPoolCreateFlags::empty(),
            32,
        );
        command_pool.reset();

        println!("{:?}", surface_format);
        let swap_config = SwapchainConfig::new().with_color(surface_format);
        let (swap_chain, backbuffer) = device.create_swapchain(surface, swap_config);
        println!("backbuffer={:?}", backbuffer);

        let render_pass = {
            let attachment = hal::pass::Attachment {
                format: Some(surface_format),
                ops: hal::pass::AttachmentOps::new(
                    hal::pass::AttachmentLoadOp::Load,
                    hal::pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
                layouts: hal::image::ImageLayout::Undefined .. hal::image::ImageLayout::Present,
            };

            let subpass = hal::pass::SubpassDesc {
                colors: &[(0, hal::image::ImageLayout::ColorAttachmentOptimal)],
                depth_stencil: None,
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

            device.create_render_pass(&[attachment], &[subpass], &[dependency])
        };

        // Framebuffer and render target creation
        let (frame_images, framebuffers) = match backbuffer {
            Backbuffer::Images(images) => {
                let extent = hal::device::Extent {
                    width: pixel_width as _,
                    height: pixel_height as _,
                    depth: 1,
                };
                let pairs = images
                    .into_iter()
                    .map(|image| {
                        let rtv = device
                            .create_image_view(
                                &image,
                                surface_format,
                                Swizzle::NO,
                                COLOR_RANGE.clone(),
                            )
                            .unwrap();
                        (image, rtv)
                    })
                    .collect::<Vec<_>>();
                let fbos = pairs
                    .iter()
                    .map(|&(_, ref rtv)| {
                        device
                            .create_framebuffer(&render_pass, &[rtv], extent)
                            .unwrap()
                    })
                    .collect();
                (pairs, fbos)
            }
            Backbuffer::Framebuffer(fbo) => (Vec::new(), vec![fbo]),
        };

        // Rendering setup
        let viewport = hal::command::Viewport {
            rect: hal::command::Rect {
                x: 0,
                y: 0,
                w: pixel_width,
                h: pixel_height,
            },
            depth: 0.0 .. 1.0,
        };

        // Samplers

        let sampler_linear = device.create_sampler(hal::image::SamplerInfo::new(
            hal::image::FilterMethod::Bilinear,
            hal::image::WrapMode::Tile,
        ));

        let sampler_nearest = device.create_sampler(hal::image::SamplerInfo::new(
            hal::image::FilterMethod::Scale,
            hal::image::WrapMode::Tile,
        ));

        let resource_cache = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 4]>(),
            max_texture_size as u32,
            max_texture_size as u32,
        );

        let render_tasks = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 12]>(),
            RENDER_TASK_TEXTURE_WIDTH as u32,
            TEXTURE_HEIGHT as u32,
        );

        let local_clip_rects = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 4]>(),
            CLIP_RECTS_TEXTURE_WIDTH as u32,
            TEXTURE_HEIGHT as u32,
        );

        let node_data = VertexDataImage::create(
            &device,
            &memory_types,
            mem::size_of::<[f32; 20]>(),
            NODE_TEXTURE_WIDTH as u32,
            TEXTURE_HEIGHT as u32,
        );

        Device {
            device,
            limits,
            memory_types,
            upload_memory_type,
            download_memory_type,
            queue_group,
            command_pool,
            swap_chain: Box::new(swap_chain),
            render_pass,
            framebuffers,
            frame_images,
            viewport,
            sampler_linear,
            sampler_nearest,
            resource_cache,
            render_tasks,
            local_clip_rects,
            node_data,
            upload_queue: Vec::new(),
            current_frame_id: 0,
            resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            upload_method,
            inside_frame: false,

            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },

            bound_textures: [0; 16],
            bound_program: 0,
            //bound_vao: 0,
            bound_read_fbo: FBOId(0),
            bound_draw_fbo: FBOId(0),
            default_read_fbo: 0,
            default_draw_fbo: 0,

            max_texture_size,
            renderer_name,
            //cached_programs,
            frame_id: FrameId(0),
            extensions,
        }
    }

    pub fn update_resource_cache(&mut self, rect: DeviceUintRect, gpu_data: &[[f32; 4]]) {
        debug_assert_eq!(gpu_data.len(), 1024);
        self.upload_queue
            .push(self.resource_cache.update_buffer_and_submit_upload(
                &mut self.device,
                &mut self.command_pool,
                rect.origin,
                gpu_data,
            ));
    }

    pub fn update_render_tasks(&mut self, task_data: &[[f32; 12]]) {
        self.upload_queue
            .push(self.render_tasks.update_buffer_and_submit_upload(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                task_data,
            ));
    }

    pub fn update_local_rects(&mut self, local_data: &[[f32; 4]]) {
        self.upload_queue
            .push(self.local_clip_rects.update_buffer_and_submit_upload(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                local_data,
            ));
    }

    pub fn update_node_data(&mut self, node_data: &[[f32; 20]]) {
        self.upload_queue
            .push(self.node_data.update_buffer_and_submit_upload(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                node_data,
            ));
    }

    pub fn create_program(
        &mut self,
        pipeline_requirements: &mut HashMap<String, PipelineRequirements>,
        shader_name: &str
    ) -> Program<B> {
        let pipeline_requirement = pipeline_requirements.remove(shader_name).expect("Shader name not found");
        let mut program = Program::create(
            pipeline_requirement,
            &self.device,
            &self.memory_types,
            shader_name,
            &self.render_pass,
        );
        program.init_vertex_data(
            &self.device,
            hal::pso::DescriptorWrite::SampledImage(vec![
                (
                    &self.resource_cache.image_srv,
                    hal::image::ImageLayout::Undefined,
                ),
            ]),
            hal::pso::DescriptorWrite::Sampler(vec![&self.sampler_nearest]),
            hal::pso::DescriptorWrite::SampledImage(vec![
                (
                    &self.node_data.image_srv,
                    hal::image::ImageLayout::Undefined,
                ),
            ]),
            hal::pso::DescriptorWrite::Sampler(vec![&self.sampler_nearest]),
            hal::pso::DescriptorWrite::SampledImage(vec![
                (
                    &self.render_tasks.image_srv,
                    hal::image::ImageLayout::Undefined,
                ),
            ]),
            hal::pso::DescriptorWrite::Sampler(vec![&self.sampler_nearest]),
            hal::pso::DescriptorWrite::SampledImage(vec![
                (
                    &self.local_clip_rects.image_srv,
                    hal::image::ImageLayout::Undefined,
                ),
            ]),
            hal::pso::DescriptorWrite::Sampler(vec![&self.sampler_nearest]),
        );
        program
    }

    pub fn draw(
        &mut self,
        program: &mut Program<B>,
        //blend_mode: &BlendMode,
        //enable_depth_write: bool
    ) {
        let submit = program.submit(
            &mut self.command_pool,
            self.viewport.clone(),
            &self.render_pass,
            &self.framebuffers[self.current_frame_id],
            &vec![],
        );

        self.upload_queue.push(submit);
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
        //self.bound_vao = 0;
        self.bound_read_fbo = FBOId(0);
        self.bound_draw_fbo = FBOId(0);
    }

    pub fn begin_frame(&mut self) -> FrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;

        // Retrive the currently set FBO.
        let default_read_fbo = 0;//self.gl.get_integer_v(gl::READ_FRAMEBUFFER_BINDING);
        self.default_read_fbo = default_read_fbo as u32;
        let default_draw_fbo = 1;//self.gl.get_integer_v(gl::DRAW_FRAMEBUFFER_BINDING);
        self.default_draw_fbo = default_draw_fbo as u32;

        // Texture state
        for i in 0 .. self.bound_textures.len() {
            self.bound_textures[i] = 0;
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
        self.bound_read_fbo = FBOId(self.default_read_fbo);
        self.bound_draw_fbo = FBOId(self.default_draw_fbo);

        // Pixel op state
        //self.gl.pixel_store_i(gl::UNPACK_ALIGNMENT, 1);
        //self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);

        // Default is sampler 0, always
        //self.gl.active_texture(gl::TEXTURE0);

        self.frame_id
    }

    fn bind_texture_impl(&mut self, slot: TextureSlot, id: u32, target: TextureTarget) {
        debug_assert!(self.inside_frame);

        if self.bound_textures[slot.0] != id {
            self.bound_textures[slot.0] = id;
            //self.gl.active_texture(gl::TEXTURE0 + slot.0 as u32);
            //self.gl.bind_texture(target, id);
            //self.gl.active_texture(gl::TEXTURE0);
        }
    }

    pub fn bind_texture<S>(&mut self, sampler: S, texture: &Texture)
    where
        S: Into<TextureSlot>,
    {
        self.bind_texture_impl(sampler.into(), texture.id, texture.target);
    }

    pub fn bind_external_texture<S>(&mut self, sampler: S, external_texture: &ExternalTexture)
    where
        S: Into<TextureSlot>,
    {
        self.bind_texture_impl(sampler.into(), external_texture.id, external_texture.target);
    }

    pub fn bind_read_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_read_fbo != fbo_id {
            self.bound_read_fbo = fbo_id;
            //fbo_id.bind(FBOTarget::Read);
        }
    }

    pub fn bind_read_target(&mut self, texture_and_layer: Option<(&Texture, i32)>) {
        let fbo_id = texture_and_layer.map_or(FBOId(self.default_read_fbo), |texture_and_layer| {
            texture_and_layer.0.fbo_ids[texture_and_layer.1 as usize]
        });

        self.bind_read_target_impl(fbo_id)
    }

    fn bind_draw_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
            //fbo_id.bind(FBOTarget::Draw);
        }
    }

    pub fn bind_draw_target(
        &mut self,
        texture_and_layer: Option<(&Texture, i32)>,
        dimensions: Option<DeviceUintSize>,
    ) {
        let fbo_id = texture_and_layer.map_or(FBOId(self.default_draw_fbo), |texture_and_layer| {
            texture_and_layer.0.fbo_ids[texture_and_layer.1 as usize]
        });

        self.bind_draw_target_impl(fbo_id);

        if let Some(dimensions) = dimensions {
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

    pub fn create_texture(
        &mut self, target: TextureTarget, format: ImageFormat,
    ) -> Texture {
        Texture {
            id: 0,
            target,
            width: 0,
            height: 0,
            layer_count: 0,
            format,
            filter: TextureFilter::Nearest,
            render_target: None,
            fbo_ids: vec![FBOId(0)],
            depth_rb: None,
        }
    }

    fn set_texture_parameters(&mut self, target: TextureTarget, filter: TextureFilter) {
        /*let filter = match filter {
            TextureFilter::Nearest => gl::NEAREST,
            TextureFilter::Linear => gl::LINEAR,
        };

        self.gl
            .tex_parameter_i(target, gl::TEXTURE_MAG_FILTER, filter as gl::GLint);
        self.gl
            .tex_parameter_i(target, gl::TEXTURE_MIN_FILTER, filter as gl::GLint);

        self.gl
            .tex_parameter_i(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as gl::GLint);
        self.gl
            .tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);*/
    }

    /// Resizes a texture with enabled render target views,
    /// preserves the data by blitting the old texture contents over.
    pub fn resize_renderable_texture(
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
        self.set_texture_parameters(texture.target, texture.filter);
        self.update_target_storage(texture, &rt_info, true, None);

        let rect = DeviceIntRect::new(DeviceIntPoint::zero(), old_size.to_i32());
        for (read_fbo, &draw_fbo) in old_fbos.into_iter().zip(&texture.fbo_ids) {
            self.bind_read_target_impl(read_fbo);
            self.bind_draw_target_impl(draw_fbo);
            self.blit_render_target(rect, rect);
            self.delete_fbo(read_fbo);
        }
        //self.gl.delete_textures(&[old_texture_id]);
        self.bind_read_target(None);
    }

    pub fn init_texture(
        &mut self,
        texture: &mut Texture,
        width: u32,
        height: u32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
        pixels: Option<&[u8]>,
    ) {
        debug_assert!(self.inside_frame);

        let is_resized = texture.width != width || texture.height != height;

        texture.width = width;
        texture.height = height;
        texture.filter = filter;
        texture.layer_count = layer_count;
        texture.render_target = render_target;

        self.bind_texture(DEFAULT_TEXTURE, texture);
        self.set_texture_parameters(texture.target, filter);

        match render_target {
            Some(info) => {
                self.update_target_storage(texture, &info, is_resized, pixels);
            }
            None => {
                self.update_texture_storage(texture, pixels);
            }
        }
    }

    /// Updates the render target storage for the texture, creating FBOs as required.
    fn update_target_storage(
        &mut self,
        texture: &mut Texture,
        rt_info: &RenderTargetInfo,
        is_resized: bool,
        pixels: Option<&[u8]>,
    ) {
        /*assert!(texture.layer_count > 0 || texture.width + texture.height == 0);

        let needed_layer_count = texture.layer_count - texture.fbo_ids.len() as i32;
        let allocate_color = needed_layer_count != 0 || is_resized || pixels.is_some();

        if allocate_color {
            let desc = gl_describe_format(self.gl(), texture.format);
            match texture.target {
                gl::TEXTURE_2D_ARRAY => {
                    self.gl.tex_image_3d(
                        texture.target,
                        0,
                        desc.internal,
                        texture.width as _,
                        texture.height as _,
                        texture.layer_count,
                        0,
                        desc.external,
                        desc.pixel_type,
                        pixels,
                    )
                }
                _ => {
                    assert_eq!(texture.layer_count, 1);
                    self.gl.tex_image_2d(
                        texture.target,
                        0,
                        desc.internal,
                        texture.width as _,
                        texture.height as _,
                        0,
                        desc.external,
                        desc.pixel_type,
                        pixels,
                    )
                }
            }
        }

        if needed_layer_count > 0 {
            // Create more framebuffers to fill the gap
            let new_fbos = self.gl.gen_framebuffers(needed_layer_count);
            texture
                .fbo_ids
                .extend(new_fbos.into_iter().map(FBOId));
        } else if needed_layer_count < 0 {
            // Remove extra framebuffers
            for old in texture.fbo_ids.drain(texture.layer_count as usize ..) {
                self.gl.delete_framebuffers(&[old.0]);
            }
        }

        let (mut depth_rb, allocate_depth) = match texture.depth_rb {
            Some(rbo) => (rbo.0, is_resized || !rt_info.has_depth),
            None if rt_info.has_depth => {
                let renderbuffer_ids = self.gl.gen_renderbuffers(1);
                let depth_rb = renderbuffer_ids[0];
                texture.depth_rb = Some(RBOId(depth_rb));
                (depth_rb, true)
            },
            None => (0, false),
        };

        if allocate_depth {
            if rt_info.has_depth {
                self.gl.bind_renderbuffer(gl::RENDERBUFFER, depth_rb);
                self.gl.renderbuffer_storage(
                    gl::RENDERBUFFER,
                    gl::DEPTH_COMPONENT24,
                    texture.width as _,
                    texture.height as _,
                );
            } else {
                self.gl.delete_renderbuffers(&[depth_rb]);
                depth_rb = 0;
                texture.depth_rb = None;
            }
        }

        if allocate_color || allocate_depth {
            let original_bound_fbo = self.bound_draw_fbo;
            for (fbo_index, &fbo_id) in texture.fbo_ids.iter().enumerate() {
                self.bind_external_draw_target(fbo_id);
                match texture.target {
                    gl::TEXTURE_2D_ARRAY => {
                        self.gl.framebuffer_texture_layer(
                            gl::DRAW_FRAMEBUFFER,
                            gl::COLOR_ATTACHMENT0,
                            texture.id,
                            0,
                            fbo_index as _,
                        )
                    }
                    _ => {
                        assert_eq!(fbo_index, 0);
                        self.gl.framebuffer_texture_2d(
                            gl::DRAW_FRAMEBUFFER,
                            gl::COLOR_ATTACHMENT0,
                            texture.target,
                            texture.id,
                            0,
                        )
                    }
                }

                self.gl.framebuffer_renderbuffer(
                    gl::DRAW_FRAMEBUFFER,
                    gl::DEPTH_ATTACHMENT,
                    gl::RENDERBUFFER,
                    depth_rb,
                );
            }
            self.bind_external_draw_target(original_bound_fbo);
        }*/
    }

    fn update_texture_storage(&mut self, texture: &Texture, pixels: Option<&[u8]>) {
        /*let desc = gl_describe_format(self.gl(), texture.format);
        match texture.target {
            gl::TEXTURE_2D_ARRAY => {
                self.gl.tex_image_3d(
                    gl::TEXTURE_2D_ARRAY,
                    0,
                    desc.internal,
                    texture.width as _,
                    texture.height as _,
                    texture.layer_count,
                    0,
                    desc.external,
                    desc.pixel_type,
                    pixels,
                );
            }
            gl::TEXTURE_2D | gl::TEXTURE_RECTANGLE | gl::TEXTURE_EXTERNAL_OES => {
                self.gl.tex_image_2d(
                    texture.target,
                    0,
                    desc.internal,
                    texture.width as _,
                    texture.height as _,
                    0,
                    desc.external,
                    desc.pixel_type,
                    pixels,
                );
            }
            _ => panic!("BUG: Unexpected texture target!"),
        }*/
    }

    pub fn blit_render_target(&mut self, src_rect: DeviceIntRect, dest_rect: DeviceIntRect) {
        debug_assert!(self.inside_frame);

        /*self.gl.blit_framebuffer(
            src_rect.origin.x,
            src_rect.origin.y,
            src_rect.origin.x + src_rect.size.width,
            src_rect.origin.y + src_rect.size.height,
            dest_rect.origin.x,
            dest_rect.origin.y,
            dest_rect.origin.x + dest_rect.size.width,
            dest_rect.origin.y + dest_rect.size.height,
            gl::COLOR_BUFFER_BIT,
            gl::LINEAR,
        );*/
    }

    /*fn free_texture_storage_impl(&mut self, target: gl::GLenum, desc: FormatDesc) {
        match target {
            gl::TEXTURE_2D_ARRAY => {
                self.gl.tex_image_3d(
                    gl::TEXTURE_2D_ARRAY,
                    0,
                    desc.internal,
                    0,
                    0,
                    0,
                    0,
                    desc.external,
                    desc.pixel_type,
                    None,
                );
            }
            _ => {
                self.gl.tex_image_2d(
                    target,
                    0,
                    desc.internal,
                    0,
                    0,
                    0,
                    desc.external,
                    desc.pixel_type,
                    None,
                );
            }
        }
    }*/

    pub fn free_texture_storage(&mut self, texture: &mut Texture) {
        /*debug_assert!(self.inside_frame);

        if texture.width + texture.height == 0 {
            return;
        }

        self.bind_texture(DEFAULT_TEXTURE, texture);
        let desc = gl_describe_format(self.gl(), texture.format);

        self.free_texture_storage_impl(texture.target, desc);

        if let Some(RBOId(depth_rb)) = texture.depth_rb.take() {
            self.gl.delete_renderbuffers(&[depth_rb]);
        }

        if !texture.fbo_ids.is_empty() {
            let fbo_ids: Vec<_> = texture
                .fbo_ids
                .drain(..)
                .map(|FBOId(fbo_id)| fbo_id)
                .collect();
            self.gl.delete_framebuffers(&fbo_ids[..]);
        }

        texture.width = 0;
        texture.height = 0;
        texture.layer_count = 0;*/
    }

    pub fn delete_texture(&mut self, mut texture: Texture) {
        self.free_texture_storage(&mut texture);
        //self.gl.delete_textures(&[texture.id]);
        texture.id = 0;
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

    pub fn upload_texture<'a, T>(
        &'a mut self,
        texture: &'a Texture,
        pbo: &PBO,
        upload_count: usize,
    ) -> TextureUploader<'a, T> {
        debug_assert!(self.inside_frame);
        self.bind_texture(DEFAULT_TEXTURE, texture);

        let buffer = match self.upload_method {
            UploadMethod::Immediate => None,
            UploadMethod::PixelBuffer(hint) => {
                let upload_size = upload_count * mem::size_of::<T>();
                /*self.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, pbo.id);
                if upload_size != 0 {
                    self.gl.buffer_data_untyped(
                        gl::PIXEL_UNPACK_BUFFER,
                        upload_size as _,
                        ptr::null(),
                        hint.to_gl(),
                    );
                }*/
                Some(PixelBuffer::new(hint, upload_size))
            },
        };

        TextureUploader {
            target: UploadTarget {
                texture,
            },
            buffer,
            marker: PhantomData,
        }
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
        let image = &self.frame_images[(self.current_frame_id + 1) % self.framebuffers.len()].0;
        let download_buffer: Buffer<B> = Buffer::create(
            &self.device,
            &self.memory_types,
            hal::buffer::Usage::TRANSFER_DST,
            bytes_per_pixel as usize,
            (rect.size.width * rect.size.height) as usize,
        );

        let copy_submit = {
            let mut cmd_buffer = self.command_pool.acquire_command_buffer();
            let image_barrier = hal::memory::Barrier::Image {
                states: (hal::image::Access::COLOR_ATTACHMENT_WRITE, hal::image::ImageLayout::ColorAttachmentOptimal) ..
                    (hal::image::Access::TRANSFER_READ, hal::image::ImageLayout::TransferSrcOptimal),
                target: image,
                range: COLOR_RANGE.clone(),
            };
            cmd_buffer.pipeline_barrier(PipelineStage::TOP_OF_PIPE .. PipelineStage::TRANSFER, &[image_barrier]);

            let buffer_width = rect.size.width * bytes_per_pixel as u32;
            cmd_buffer.copy_image_to_buffer(
                &image,
                hal::image::ImageLayout::TransferSrcOptimal,
                &download_buffer.buffer,
                &[hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: rect.size.width,
                    buffer_height: rect.size.height,
                    image_layers: hal::image::SubresourceLayers {
                        aspects: hal::format::AspectFlags::COLOR,
                        level: 0,
                        layers: 0 .. 1,
                    },
                    image_offset: hal::command::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    image_extent: hal::device::Extent {
                        width: rect.size.width as _,
                        height: rect.size.height as _,
                        depth: 1 as _,
                    },
                }]);
            let image_barrier = hal::memory::Barrier::Image {
                states: (hal::image::Access::TRANSFER_READ, hal::image::ImageLayout::TransferSrcOptimal) ..
                    (hal::image::Access::COLOR_ATTACHMENT_WRITE, hal::image::ImageLayout::ColorAttachmentOptimal),
                target: image,
                range: COLOR_RANGE.clone(),
            };
            cmd_buffer.pipeline_barrier(PipelineStage::TRANSFER .. PipelineStage::BOTTOM_OF_PIPE, &[image_barrier]);
            cmd_buffer.finish()
        };

        let copy_fence = self.device.create_fence(false);
        let submission = hal::queue::Submission::new()
            .submit(&[copy_submit]);
        self.queue_group.queues[0].submit(submission, Some(&copy_fence));
        self.device.wait_for_fences(&[&copy_fence], hal::device::WaitFor::Any, !0);
        self.device.destroy_fence(copy_fence);

        let mut reader = self.device
            .acquire_mapping_reader::<u8>(
                &download_buffer.buffer,
                0 .. (rect.size.width * rect.size.height * bytes_per_pixel as u32) as u64,
            )
            .unwrap();
        assert_eq!(reader.len(), output.len());
        for (i, d) in reader.iter().enumerate() {
            output[i] = *d;
        }
        self.device.release_mapping_reader(reader);
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
        &mut self, texture_id: u32, target: TextureTarget, layer_id: i32
    ) {
        /*match target {
            gl::TEXTURE_2D_ARRAY => {
                self.gl.framebuffer_texture_layer(
                    gl::READ_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    texture_id,
                    0,
                    layer_id,
                )
            }
            _ => {
                assert_eq!(layer_id, 0);
                self.gl.framebuffer_texture_2d(
                    gl::READ_FRAMEBUFFER,
                    gl::COLOR_ATTACHMENT0,
                    target,
                    texture_id,
                    0,
                )
            }
        }*/
    }

    pub fn attach_read_texture_external(
        &mut self, texture_id: u32, target: TextureTarget, layer_id: i32
    ) {
        self.attach_read_texture_raw(texture_id, target, layer_id)
    }

    pub fn attach_read_texture(&mut self, texture: &Texture, layer_id: i32) {
        self.attach_read_texture_raw(texture.id, texture.target, layer_id)
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
        let mut cmd_buffer = self.command_pool.acquire_command_buffer();

        if let Some(rect) = rect {
            cmd_buffer.set_scissors(&[
                hal::command::Rect {
                    x: rect.origin.x as u16,
                    y: rect.origin.y as u16,
                    w: rect.size.width as u16,
                    h: rect.size.height as u16,
                },
            ]);
        }

        if let Some(color) = color {
            cmd_buffer.clear_color_image(
                &self.frame_images[self.current_frame_id].0,
                hal::image::ImageLayout::ColorAttachmentOptimal,
                hal::image::SubresourceRange {
                    aspects: hal::format::AspectFlags::COLOR,
                    levels: 0 .. 1,
                    layers: 0 .. 1,
                },
                hal::command::ClearColor::Float([color[0], color[1], color[2], color[3]]),
            );
        }

        // TODO enable it when the crash is resolved
        /*if let Some(depth) = depth {
            cmd_buffer.clear_depth_stencil_image(
                &self.frame_images[self.current_frame_id].0,
                hal::image::ImageLayout::DepthStencilAttachmentOptimal,
                hal::image::SubresourceRange {
                            aspects: hal::format::AspectFlags::DEPTH,
                            levels: 0 .. 1,
                            layers: 0 .. 1,
                        },
                hal::command::ClearDepthStencil(depth, 0)
            );
        }*/
        self.upload_queue.push(cmd_buffer.finish());
    }

    pub fn enable_depth(&self) {
        //self.gl.enable(gl::DEPTH_TEST);
    }

    pub fn disable_depth(&self) {
        //self.gl.disable(gl::DEPTH_TEST);
    }

    pub fn set_depth_func(&self, depth_func: DepthFunction) {
        //self.gl.depth_func(depth_func as u32);
    }

    pub fn enable_depth_write(&self) {
        //self.gl.depth_mask(true);
    }

    pub fn disable_depth_write(&self) {
        //self.gl.depth_mask(false);
    }

    pub fn disable_stencil(&self) {
        //self.gl.disable(gl::STENCIL_TEST);
    }

    pub fn disable_scissor(&self) {
        //self.gl.disable(gl::SCISSOR_TEST);
    }

    pub fn set_blend(&self, enable: bool) {
        if enable {
            //self.gl.enable(gl::BLEND);
        } else {
            //self.gl.disable(gl::BLEND);
        }
    }

    pub fn set_blend_mode_alpha(&self) {
        //self.gl.blend_func_separate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA,
        //                            gl::ONE, gl::ONE);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }

    pub fn set_blend_mode_premultiplied_alpha(&self) {
        //self.gl.blend_func(gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }

    pub fn set_blend_mode_premultiplied_dest_out(&self) {
        //self.gl.blend_func(gl::ZERO, gl::ONE_MINUS_SRC_ALPHA);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }

    pub fn set_blend_mode_multiply(&self) {
        //self.gl
        //    .blend_func_separate(gl::ZERO, gl::SRC_COLOR, gl::ZERO, gl::SRC_ALPHA);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }
    pub fn set_blend_mode_max(&self) {
        //self.gl
        //    .blend_func_separate(gl::ONE, gl::ONE, gl::ONE, gl::ONE);
        //self.gl.blend_equation_separate(gl::MAX, gl::FUNC_ADD);
    }
    pub fn set_blend_mode_min(&self) {
        //self.gl
        //    .blend_func_separate(gl::ONE, gl::ONE, gl::ONE, gl::ONE);
        //self.gl.blend_equation_separate(gl::MIN, gl::FUNC_ADD);
    }
    pub fn set_blend_mode_subpixel_pass0(&self) {
        //self.gl.blend_func(gl::ZERO, gl::ONE_MINUS_SRC_COLOR);
    }
    pub fn set_blend_mode_subpixel_pass1(&self) {
        //self.gl.blend_func(gl::ONE, gl::ONE);
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass0(&self) {
        //self.gl.blend_func_separate(gl::ZERO, gl::ONE_MINUS_SRC_COLOR, gl::ZERO, gl::ONE);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass1(&self) {
        //self.gl.blend_func_separate(gl::ONE_MINUS_DST_ALPHA, gl::ONE, gl::ZERO, gl::ONE);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }
    pub fn set_blend_mode_subpixel_with_bg_color_pass2(&self) {
        //self.gl.blend_func_separate(gl::ONE, gl::ONE, gl::ONE, gl::ONE_MINUS_SRC_ALPHA);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }
    pub fn set_blend_mode_subpixel_constant_text_color(&self, color: ColorF) {
        // color is an unpremultiplied color.
        //self.gl.blend_color(color.r, color.g, color.b, 1.0);
        //self.gl
        //    .blend_func(gl::CONSTANT_COLOR, gl::ONE_MINUS_SRC_COLOR);
        //self.gl.blend_equation(gl::FUNC_ADD);
    }
    pub fn set_blend_mode_subpixel_dual_source(&self) {
        //self.gl.blend_func(gl::ONE, gl::ONE_MINUS_SRC1_COLOR);
    }

    pub fn supports_extension(&self, extension: &str) -> bool {
        self.extensions.iter().any(|s| s == extension)
    }

    pub fn swap_buffers(&mut self) {
        let mut frame_semaphore = self.device.create_semaphore();
        let mut frame_fence = self.device.create_fence(false); // TODO: remove
        {
            self.device.reset_fences(&[&frame_fence]);

            let frame = self.swap_chain
                .acquire_frame(FrameSync::Semaphore(&mut frame_semaphore));
            assert_eq!(frame.id(), self.current_frame_id);

            let submission = Submission::new()
                .wait_on(&[(&mut frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
                .submit(&self.upload_queue);
            self.queue_group.queues[0].submit(submission, Some(&mut frame_fence));

            // TODO: replace with semaphore
            self.device
                .wait_for_fences(&[&frame_fence], hal::device::WaitFor::All, !0);

            // present frame
            self.swap_chain
                .present(&mut self.queue_group.queues[0], &[]);
            self.current_frame_id = (self.current_frame_id + 1) % self.framebuffers.len();
        }
        self.upload_queue.clear();
        self.device.destroy_fence(frame_fence);
        self.device.destroy_semaphore(frame_semaphore);
    }

    pub fn deinit(self) {
        self.device
            .destroy_command_pool(self.command_pool.downgrade());
        self.device.destroy_renderpass(self.render_pass);
        for framebuffer in self.framebuffers {
            self.device.destroy_framebuffer(framebuffer);
        }
        for (image, rtv) in self.frame_images {
            self.device.destroy_image_view(rtv);
            self.device.destroy_image(image);
        }
        self.resource_cache.deinit(&self.device);
        self.render_tasks.deinit(&self.device);
        self.local_clip_rects.deinit(&self.device);
        self.node_data.deinit(&self.device);
    }
}

/*struct FormatDesc {
    internal: gl::GLint,
    external: u32,
    pixel_type: u32,
}*/

/*fn gl_describe_format(gl: &gl::Gl, format: ImageFormat) -> FormatDesc {
    match format {
        ImageFormat::R8 => FormatDesc {
            internal: gl::RED as _,
            external: gl::RED,
            pixel_type: gl::UNSIGNED_BYTE,
        },
        ImageFormat::BGRA8 => {
            let external = get_gl_format_bgra(gl);
            FormatDesc {
                internal: match gl.get_type() {
                    gl::GlType::Gl => gl::RGBA as _,
                    gl::GlType::Gles => external as _,
                },
                external,
                pixel_type: gl::UNSIGNED_BYTE,
            }
        },
        ImageFormat::RGBAF32 => FormatDesc {
            internal: gl::RGBA32F as _,
            external: gl::RGBA,
            pixel_type: gl::FLOAT,
        },
        ImageFormat::RG8 => FormatDesc {
            internal: gl::RG8 as _,
            external: gl::RG,
            pixel_type: gl::UNSIGNED_BYTE,
        },
    }
}*/

struct UploadChunk {
    rect: DeviceUintRect,
    layer_index: i32,
    stride: Option<u32>,
    offset: usize,
}

struct PixelBuffer {
    usage: VertexUsageHint,
    size_allocated: usize,
    size_used: usize,
    // small vector avoids heap allocation for a single chunk
    chunks: SmallVec<[UploadChunk; 1]>,
}

impl PixelBuffer {
    fn new(
        usage: VertexUsageHint,
        size_allocated: usize,
    ) -> Self {
        PixelBuffer {
            usage,
            size_allocated,
            size_used: 0,
            chunks: SmallVec::new(),
        }
    }
}

struct UploadTarget<'a> {
    //gl: &'a gl::Gl,
    texture: &'a Texture,
}

pub struct TextureUploader<'a, T> {
    target: UploadTarget<'a>,
    buffer: Option<PixelBuffer>,
    marker: PhantomData<T>,
}

impl<'a, T> Drop for TextureUploader<'a, T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            for chunk in buffer.chunks {
                self.target.update_impl(chunk);
            }
            //self.target.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
        }
    }
}

impl<'a, T> TextureUploader<'a, T> {
    pub fn upload(
        &mut self,
        rect: DeviceUintRect,
        layer_index: i32,
        stride: Option<u32>,
        data: &[T],
    ) {
        match self.buffer {
            Some(ref mut buffer) => {
                let upload_size = mem::size_of::<T>() * data.len();
                if buffer.size_used + upload_size > buffer.size_allocated {
                    // flush
                    for chunk in buffer.chunks.drain() {
                        self.target.update_impl(chunk);
                    }
                    buffer.size_used = 0;
                }

                if upload_size > buffer.size_allocated {
                    /*gl::buffer_data(
                        self.target.gl,
                        gl::PIXEL_UNPACK_BUFFER,
                        data,
                        buffer.usage,
                    );*/
                    buffer.size_allocated = upload_size;
                } else {
                    /*gl::buffer_sub_data(
                        self.target.gl,
                        gl::PIXEL_UNPACK_BUFFER,
                        buffer.size_used as _,
                        data,
                    );*/
                }

                buffer.chunks.push(UploadChunk {
                    rect, layer_index, stride,
                    offset: buffer.size_used,
                });
                buffer.size_used += upload_size;
            }
            None => {
                self.target.update_impl(UploadChunk {
                    rect, layer_index, stride,
                    offset: data.as_ptr() as _,
                });
            }
        }
    }
}

impl<'a> UploadTarget<'a> {
    fn update_impl(&mut self, chunk: UploadChunk) {
        /*let (gl_format, bpp, data_type) = match self.texture.format {
            ImageFormat::R8 => (gl::RED, 1, gl::UNSIGNED_BYTE),
            ImageFormat::BGRA8 => (get_gl_format_bgra(self.gl), 4, gl::UNSIGNED_BYTE),
            ImageFormat::RG8 => (gl::RG, 2, gl::UNSIGNED_BYTE),
            ImageFormat::RGBAF32 => (gl::RGBA, 16, gl::FLOAT),
        };

        let row_length = match chunk.stride {
            Some(value) => value / bpp,
            None => self.texture.width,
        };

        if chunk.stride.is_some() {
            self.gl.pixel_store_i(
                gl::UNPACK_ROW_LENGTH,
                row_length as _,
            );
        }

        let pos = chunk.rect.origin;
        let size = chunk.rect.size;

        match self.texture.target {
            gl::TEXTURE_2D_ARRAY => {
                self.gl.tex_sub_image_3d_pbo(
                    self.texture.target,
                    0,
                    pos.x as _,
                    pos.y as _,
                    chunk.layer_index,
                    size.width as _,
                    size.height as _,
                    1,
                    gl_format,
                    data_type,
                    chunk.offset,
                );
            }
            gl::TEXTURE_2D | gl::TEXTURE_RECTANGLE | gl::TEXTURE_EXTERNAL_OES => {
                self.gl.tex_sub_image_2d_pbo(
                    self.texture.target,
                    0,
                    pos.x as _,
                    pos.y as _,
                    size.width as _,
                    size.height as _,
                    gl_format,
                    data_type,
                    chunk.offset,
                );
            }
            _ => panic!("BUG: Unexpected texture target!"),
        }

        // Reset row length to 0, otherwise the stride would apply to all texture uploads.
        if chunk.stride.is_some() {
            self.gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, 0 as _);
        }*/
    }
}
