/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source;
use api::{ColorF, ImageFormat};
use api::{DeviceUintPoint, DeviceIntRect, DeviceUintRect, DeviceUintSize};
use euclid::Transform3D;
//use gleam::gl;
use internal_types::{FastHashMap, RenderTargetInfo};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::fs::File;
use std::io::Read;
use std::iter::repeat;
use std::marker::PhantomData;
use std::mem;
use std::ops::Add;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;
use std::thread;
use std::cmp;

use hal;
use winit;
use back;

// gfx-hal
use hal::{Device as BackendDevice, Instance, PhysicalDevice, QueueFamily, Surface, Swapchain};
use hal::{DescriptorPool, Gpu, FrameSync, Primitive, Backbuffer, SwapchainConfig};
use hal::format::{ChannelType, Swizzle};
use hal::pass::Subpass;
use hal::pso::{PipelineStage, ShaderStageFlags};
use hal::queue::Submission;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureTarget {
    Default,
    Array,
    Rect,
    External,
}

#[derive(Debug)]
pub struct TextureSlot(pub usize);

// In some places we need to temporarily bind a texture to any slot.
const DEFAULT_TEXTURE: TextureSlot = TextureSlot(0);

pub const NODE_TEXTURE_WIDTH: usize = 1022; // 146 * 7
pub const RENDER_TASK_TEXTURE_WIDTH: usize = 1023; // 341 * 3
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
struct Vertex {
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
struct PrimitiveInstance {
    aData0: [i32; 4],
    aData1: [i32; 4],
}

const QUAD: [Vertex; 6] = [
    Vertex { aPosition: [  0.0, 0.0, 0.0  ] },
    Vertex { aPosition: [  1.0, 0.0, 0.0  ] },
    Vertex { aPosition: [  0.0, 1.0, 0.0  ] },
    Vertex { aPosition: [  0.0, 1.0, 0.0  ] },
    Vertex { aPosition: [  1.0, 0.0, 0.0  ] },
    Vertex { aPosition: [  1.0, 1.0, 0.0  ] },
];
// VECS_PER_LAYER = 7 ( 28 / 4 )
struct NodeData { // 28 <- 16 + 4 + 2 + 2 + 1 + 3
    transform: [[f32; 4]; 4],
    local_clip_rect: [f32; 4],
    reference_frame_relative_scroll_offset: [f32; 2],
    scroll_offset: [f32; 2],
    transform_kind: f32,
    padding: [f32; 3],
}

// VECS_PER_RENDER_TASK = 3 ( 12 / 4 )
struct RenderTaskData {
    data: [f32; 12]
}

#[derive(Debug, Copy, Clone, PartialEq, Ord, Eq, PartialOrd)]
pub struct FrameId(usize);

impl FrameId {
    pub fn new(value: usize) -> FrameId {
        FrameId(value)
    }
}

impl Add<usize> for FrameId {
    type Output = FrameId;

    fn add(self, other: usize) -> FrameId {
        FrameId(self.0 + other)
    }
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

fn get_shader_source(filename: &str, extension: &str) -> Vec<u8> {
    use std::io::Read;
    let path_str = format!("{}/{}{}", env!("OUT_DIR"), filename, extension);
    let mut file = File::open(path_str).unwrap();
    let mut shader = Vec::new();
    file.read_to_end(&mut shader);
    shader
}

pub struct ExternalTexture {
    id: u32,
    target: TextureTarget,
}

impl ExternalTexture {
    pub fn new(id: u32, target: TextureTarget) -> ExternalTexture {
        ExternalTexture {
            id,
            target,
        }
    }
}

pub struct Texture {
    target: TextureTarget,
    width: u32,
    height: u32,
    layer_count: i32,
    format: ImageFormat,
}

impl Texture {
    pub fn get_dimensions(&self) -> DeviceUintSize {
        DeviceUintSize::new(self.width, self.height)
    }

    pub fn has_depth(&self) -> bool {
        false
    }

    pub fn get_render_target_layer_count(&self) -> usize {
        0 //fbo num
    }

    pub fn get_layer_count(&self) -> i32 {
        self.layer_count
    }

    pub fn get_format(&self) -> ImageFormat {
        self.format
    }
}

pub struct VertexDataImage<B: hal::Backend> {
    pub image_upload_buffer: Buffer<B>,
    pub image: B::Image,
    pub image_memory: B::Memory,
    pub image_srv: B::ImageView,
    pub image_stride: usize,
    pub mem_stride: usize,
    pub image_width: u32,
    pub image_height: u32
}

impl<B: hal::Backend> VertexDataImage<B> {
    pub fn create(device: &B::Device, memory_types: &Vec<hal::MemoryType>, data_stride: usize, image_width: u32, image_height: u32) -> VertexDataImage<B> {
        let image_upload_buffer =
            Buffer::create(
                device,
                memory_types,
                hal::buffer::Usage::TRANSFER_SRC,
                data_stride,
                (image_width * image_height) as usize
            );
        let kind = hal::image::Kind::D2(image_width as hal::image::Size, image_height as hal::image::Size, hal::image::AaMode::Single);
        let image_unbound = device.create_image(kind, 1, hal::format::Format::Rgba32Float, hal::image::Usage::TRANSFER_DST | hal::image::Usage::SAMPLED).unwrap(); // TODO: usage
        println!("{:?}", image_unbound);
        let image_req = device.get_image_requirements(&image_unbound);

        let device_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                image_req.type_mask & (1 << id) != 0 &&
                    mem_type.properties.contains(hal::memory::Properties::DEVICE_LOCAL)
            })
            .unwrap()
            .into();

        let image_memory = device.allocate_memory(device_type, image_req.size).unwrap();

        let image = device.bind_image_memory(&image_memory, 0, image_unbound).unwrap();
        let image_srv = device.create_image_view(&image, hal::format::Format::Rgba32Float, Swizzle::NO, COLOR_RANGE.clone()).unwrap();

        VertexDataImage {
            image_upload_buffer,
            image,
            image_memory,
            image_srv,
            image_stride: 4usize, // Rgba
            mem_stride: mem::size_of::<f32>(), // Float
            image_width,
            image_height
        }
    }

    pub fn update_buffer_and_submit_upload<T>(
        &self,
        device: &mut B::Device,
        cmd_pool: &mut hal::CommandPool<B, hal::queue::Graphics>,
        image_offset: DeviceUintPoint,
        image_data: &Vec<T>,
    ) -> hal::command::Submit<B, hal::queue::Graphics>
    where
        T: Copy
    {
        let needed_height = (image_data.len() / (self.image_width as  usize * self.image_stride / self.image_upload_buffer.data_stride)) as u32;
        if needed_height > self.image_height {
            unimplemented!("TODO: implement resize");
        }
        let buffer_height = cmp::max(1, needed_height) as u64;
        let buffer_width = (image_data.len() * self.image_upload_buffer.data_stride * self.mem_stride) as u64;
        let buffer_offset = (image_offset.y * buffer_width as u32 * self.image_stride as u32 * self.mem_stride as u32) as u64;
        self.image_upload_buffer.update(device, buffer_offset, buffer_width, image_data);

        let mut cmd_buffer = cmd_pool.acquire_command_buffer();

        let image_barrier = hal::memory::Barrier::Image {
            states: (hal::image::Access::TRANSFER_WRITE, hal::image::ImageLayout::TransferDstOptimal) ..
                (hal::image::Access::TRANSFER_WRITE, hal::image::ImageLayout::TransferDstOptimal),
            target: &self.image,
            range: COLOR_RANGE.clone(),
        };
        cmd_buffer.pipeline_barrier(hal::pso::PipelineStage::TOP_OF_PIPE .. hal::pso::PipelineStage::TRANSFER, &[image_barrier]);

        cmd_buffer.copy_buffer_to_image(
            &self.image_upload_buffer.buffer,
            &self.image,
            hal::image::ImageLayout::TransferDstOptimal,
            &[hal::command::BufferImageCopy {
                buffer_offset,
                buffer_width: buffer_width as u32,
                buffer_height: buffer_height as u32,
                image_layers: hal::image::SubresourceLayers {
                    aspects: hal::format::AspectFlags::COLOR,
                    level: 0,
                    layers: 0 .. 1,
                },
                image_offset: hal::command::Offset { x: image_offset.x as i32, y: image_offset.y as i32, z: 0 },
                image_extent: hal::device::Extent { width: buffer_width as u32, height: buffer_height as u32, depth: 1 },
            }]);

        let image_barrier = hal::memory::Barrier::Image {
            states: (hal::image::Access::TRANSFER_WRITE, hal::image::ImageLayout::TransferDstOptimal) ..
                (hal::image::Access::SHADER_READ, hal::image::ImageLayout::ShaderReadOnlyOptimal),
            target: &self.image,
            range: COLOR_RANGE.clone(),
        };
        cmd_buffer.pipeline_barrier(hal::pso::PipelineStage::TRANSFER .. hal::pso::PipelineStage::VERTEX_SHADER, &[image_barrier]);
        cmd_buffer.finish()
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
        memory_types: &Vec<hal::MemoryType>,
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
            let buffer_memory = device.allocate_memory(buffer_type, buffer_req.size).unwrap();
            let buffer = device.bind_buffer_memory(&buffer_memory, 0, unbound_buffer).unwrap();
            (buffer_memory, buffer)
        };
        Buffer {
            memory,
            buffer,
            data_stride,
        }
    }

    pub fn update<T>(
        &self,
        device: &B::Device,
        buffer_offset: u64,
        buffer_width: u64,
        update_data: &Vec<T>
    )
    where T: Copy
    {
        let mut data = device
            .acquire_mapping_writer::<T>(&self.buffer, buffer_offset..(buffer_offset + buffer_width))
            .unwrap();
        for (i, d) in update_data.iter().enumerate() {
            data[i] = *d;
        }
        device.release_mapping_writer(data);
    }
}

pub struct Device<B: hal::Backend> {
    pub device: B::Device,
    pub memory_types: Vec<hal::MemoryType>,
    pub queue_group: hal::QueueGroup<B, hal::queue::Graphics>,
    pub command_pool: hal::CommandPool<B, hal::queue::Graphics>,
    pub swap_chain: Box<B::Swapchain>,
    pub render_pass: B::RenderPass,
    pub framebuffers: Vec<B::Framebuffer>,
    pub frame_images: Vec<(B::Image, B::ImageView)>,
    pub viewport: hal::command::Viewport,
    pub vs_module: B::ShaderModule,
    pub fs_module: B::ShaderModule,
    pub set_layout: B::DescriptorSetLayout,
    pub pipeline_layout: B::PipelineLayout,
    pub pipelines: Vec<Result<B::GraphicsPipeline, hal::pso::CreationError>>,
    pub desc_pool: B::DescriptorPool,
    pub desc_sets: Vec<B::DescriptorSet>,
    pub vertex_buffer: Buffer<B>,
    pub instance_buffer: Buffer<B>,
    pub locals_buffer: Buffer<B>,
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    pub resource_cache: VertexDataImage<B>,
    pub render_tasks: VertexDataImage<B>,
    pub node_data: VertexDataImage<B>,
    pub image_uploads: Vec<hal::command::Submit<B, hal::queue::Graphics>>,
}

impl<B: hal::Backend> Device<B> {
    pub fn new(
        window: &winit::Window,
        instance: &back::Instance,
        surface: &mut <back::Backend as hal::Backend>::Surface
    ) -> Device<back::Backend> {
        let max_texture_size = 1024;

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
                }
            );

        let memory_types = adapter
            .physical_device
            .memory_properties()
            .memory_types;
        let limits = adapter
            .physical_device
            .get_limits();

        let Gpu { device, mut queue_groups } =
            adapter.open_with(|family| {
                if family.supports_graphics() && surface.supports_queue_family(family) {
                    Some(1)
                } else { None }
            }).unwrap();

        let queue_group = hal::QueueGroup::<_, hal::Graphics>::new(queue_groups.remove(0));
        let mut command_pool = device.create_command_pool_typed(&queue_group, hal::pool::CommandPoolCreateFlags::empty(), 32);
        command_pool.reset();

        println!("{:?}", surface_format);
        let swap_config = SwapchainConfig::new()
            .with_color(surface_format);
        let (mut swap_chain, backbuffer) = device.create_swapchain(surface, swap_config);
        println!("backbuffer={:?}", backbuffer);

        let render_pass = {
            let attachment = hal::pass::Attachment {
                format: Some(surface_format),
                ops: hal::pass::AttachmentOps::new(hal::pass::AttachmentLoadOp::Clear, hal::pass::AttachmentStoreOp::Store),
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
                stages: PipelineStage::COLOR_ATTACHMENT_OUTPUT .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: hal::image::Access::empty() .. (hal::image::Access::COLOR_ATTACHMENT_READ | hal::image::Access::COLOR_ATTACHMENT_WRITE),
            };

            device.create_render_pass(&[attachment], &[subpass], &[dependency])
        };

        // Framebuffer and render target creation
        let (frame_images, framebuffers) = match backbuffer {
            Backbuffer::Images(images) => {
                let extent = hal::device::Extent { width: pixel_width as _, height: pixel_height as _, depth: 1 };
                let pairs = images
                    .into_iter()
                    .map(|image| {
                        let rtv = device.create_image_view(&image, surface_format, Swizzle::NO, COLOR_RANGE.clone()).unwrap();
                        (image, rtv)
                    })
                    .collect::<Vec<_>>();
                let fbos = pairs
                    .iter()
                    .map(|&(_, ref rtv)| {
                        device.create_framebuffer(&render_pass, &[rtv], extent).unwrap()
                    })
                    .collect();
                (pairs, fbos)
            }
            Backbuffer::Framebuffer(fbo) => {
                (Vec::new(), vec![fbo])
            }
        };

        // Rendering setup
        let viewport = hal::command::Viewport {
            rect: hal::command::Rect {
                x: 0, y: 0,
                w: pixel_width, h: pixel_height,
            },
            depth: 0.0 .. 1.0,
        };

        // Setup renderpass and pipeline
        #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
        let vs_module = device
            .create_shader_module(
                get_shader_source("ps_line", ".vert.spv").as_slice()
            )
            .unwrap();
        #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
        let fs_module = device
            .create_shader_module(
                get_shader_source("ps_line", ".frag.spv").as_slice()
            )
            .unwrap();

        let set_layout = device.create_descriptor_set_layout(&[
            hal::pso::DescriptorSetLayoutBinding { // Locals
                binding: 0,
                ty: hal::pso::DescriptorType::UniformBuffer,
                count: 1,
                stage_flags: ShaderStageFlags::VERTEX,
            },
            hal::pso::DescriptorSetLayoutBinding { // tColor0
                binding: 1,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sColor0
                binding: 2,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // tColor1
                binding: 3,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sColor1
                binding: 4,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // tColor2
                binding: 5,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sColor2
                binding: 6,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },

            //binding 7-8 <- dither

            hal::pso::DescriptorSetLayoutBinding { // tCacheA8
                binding: 9,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sCacheA8
                binding: 10,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // tCacheRGBA8
                binding: 11,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sCacheRGBA8
                binding: 12,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // tSharedCacheA8
                binding: 13,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sSharedCacheA8
                binding: 14,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },

            // binding 15-16 <- {t|s}Gradients // Why this even exist?

            hal::pso::DescriptorSetLayoutBinding { // tResourceCache
                binding: 17,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sResourceCache
                binding: 18,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // tClipScrollNodes
                binding: 19,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sClipScrollNodes
                binding: 20,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // tRenderTasks
                binding: 21,
                ty: hal::pso::DescriptorType::SampledImage,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
            hal::pso::DescriptorSetLayoutBinding { // sRenderTasks
                binding: 22,
                ty: hal::pso::DescriptorType::Sampler,
                count: 1,
                stage_flags: ShaderStageFlags::ALL,
            },
        ],
        );

        let pipeline_layout = device.create_pipeline_layout(&[&set_layout], &[]);

        let pipelines = {
            let (vs_entry, fs_entry) = (
                hal::pso::EntryPoint::<back::Backend> { entry: ENTRY_NAME, module: &vs_module, specialization: &[] },
                hal::pso::EntryPoint::<back::Backend> { entry: ENTRY_NAME, module: &fs_module, specialization: &[] },
            );

            let shader_entries = hal::pso::GraphicsShaderSet {
                vertex: vs_entry,
                hull: None,
                domain: None,
                geometry: None,
                fragment: Some(fs_entry),
            };

            let subpass = Subpass { index: 0, main_pass: &render_pass };

            let mut pipeline_desc = hal::pso::GraphicsPipelineDesc::new(
                shader_entries,
                Primitive::TriangleList,
                hal::pso::Rasterizer::FILL,
                &pipeline_layout,
                subpass,
            );
            pipeline_desc.blender.targets.push(hal::pso::ColorBlendDesc(
                hal::pso::ColorMask::ALL,
                hal::pso::BlendState::ALPHA,
            ));
            pipeline_desc.vertex_buffers.push(hal::pso::VertexBufferDesc {
                stride: mem::size_of::<Vertex>() as u32,
                rate: 0, // VertexBuffer
            });
            pipeline_desc.vertex_buffers.push(hal::pso::VertexBufferDesc {
                stride: mem::size_of::<PrimitiveInstance>() as u32,
                rate: 1, // InstanceBuffer
            });

            pipeline_desc.attributes.push(hal::pso::AttributeDesc { // aPosition
                location: 0,
                binding: 0,
                element: hal::pso::Element {
                    format: hal::format::Format::Rgb32Float,
                    offset: 0,
                },
            });
            pipeline_desc.attributes.push(hal::pso::AttributeDesc { // aData0
                location: 4,
                binding: 1,
                element: hal::pso::Element {
                    format: hal::format::Format::Rgba32Int,
                    offset: 0,
                },
            });
            pipeline_desc.attributes.push(hal::pso::AttributeDesc { // aData1
                location: 5,
                binding: 1,
                element: hal::pso::Element {
                    format: hal::format::Format::Rgba32Int,
                    offset: 16,
                },
            });

            device.create_graphics_pipelines(&[pipeline_desc])
        };

        println!("pipelines: {:?}", pipelines);

        let mut desc_pool = device.create_descriptor_pool(
            1, // sets
            &[
                hal::pso::DescriptorRangeDesc {
                    ty: hal::pso::DescriptorType::UniformBuffer,
                    count: 1,
                },
                hal::pso::DescriptorRangeDesc {
                    ty: hal::pso::DescriptorType::SampledImage,
                    count: 11,
                },
                hal::pso::DescriptorRangeDesc {
                    ty: hal::pso::DescriptorType::Sampler,
                    count: 11,
                },
            ],
        );

        let desc_sets = desc_pool.allocate_sets(&[&set_layout]);

        // Buffer allocations
        println!("Memory types: {:?}", memory_types);

        let vertex_buffer_stride = mem::size_of::<Vertex>();
        let vertex_buffer_len = QUAD.len() * vertex_buffer_stride;

        let vertex_buffer =
            Buffer::create(
                &device,
                &memory_types,
                hal::buffer::Usage::VERTEX,
                vertex_buffer_stride,
                vertex_buffer_len
            );

        vertex_buffer.update(&device, 0, vertex_buffer_len as u64, &vec![QUAD]);

        let instance_buffer_stride = mem::size_of::<PrimitiveInstance>();
        let instance_buffer_len = MAX_INSTANCE_COUNT * instance_buffer_stride;

        let instance_buffer =
            Buffer::create(
                &device,
                &memory_types,
                hal::buffer::Usage::VERTEX,
                instance_buffer_stride,
                instance_buffer_len
            );

        let instance_data =
            vec![
                PrimitiveInstance {
                    aData0: [2044, 0, 2147483647, 131074],
                    aData1: [3, 0, 0, 0],
                }
            ];
        instance_buffer.update(
            &device,
            0,
            (instance_data.len() * instance_buffer_stride) as u64,
            &instance_data
        );

        let locals_buffer_stride = mem::size_of::<Locals>();
        let locals_buffer_len = locals_buffer_stride;

        let locals_buffer =
            Buffer::create(
                &device,
                &memory_types,
                hal::buffer::Usage::UNIFORM,
                locals_buffer_stride,
                locals_buffer_len
            );
        let projection = Transform3D::row_major(0.001953125, 0.0, 0.0, 0.0,
                                                0.0,-0.0026041667, 0.0, 0.0,
                                                0.0, 0.0, 0.000001, 0.0,
                                                -1.0, 1.0, 0.0, 1.0);
        let locals_data =
            vec![
                Locals {
                    uTransform: projection.to_row_arrays(),
                    uDevicePixelRatio: 1.0,
                    uMode: 0i32,
                }
            ];
        locals_buffer.update(
            &device,
            0,
            (locals_data.len() * locals_buffer_stride) as u64,
            &locals_data
        );

        // Samplers

        let sampler_linear = device.create_sampler(
            hal::image::SamplerInfo::new(
                hal::image::FilterMethod::Bilinear,
                hal::image::WrapMode::Tile,
            )
        );

        let sampler_nearest = device.create_sampler(
            hal::image::SamplerInfo::new(
                hal::image::FilterMethod::Scale,
                hal::image::WrapMode::Tile,
            )
        );

        // Textures

        let resource_cache =
            VertexDataImage::create(
                &device,
                &memory_types,
                mem::size_of::<[f32; 4]>(),
                max_texture_size as u32,
                max_texture_size as u32
            );

        let render_tasks =
            VertexDataImage::create(
                &device,
                &memory_types,
                mem::size_of::<[f32; 12]>(),
                RENDER_TASK_TEXTURE_WIDTH as u32,
                TEXTURE_HEIGHT as u32
            );

        let node_data =
            VertexDataImage::create(
                &device,
                &memory_types,
                mem::size_of::<[f32; 28]>(),
                NODE_TEXTURE_WIDTH as u32,
                TEXTURE_HEIGHT as u32
            );

        device.update_descriptor_sets(&[
            hal::pso::DescriptorSetWrite {
                set: &desc_sets[0],
                binding: 0,
                array_offset: 0,
                write: hal::pso::DescriptorWrite::UniformBuffer(vec![
                    (&locals_buffer.buffer, 0..mem::size_of::<Locals>() as u64),
                ]),
            },
            hal::pso::DescriptorSetWrite {
                set: &desc_sets[0],
                binding: 17,
                array_offset: 0,
                write: hal::pso::DescriptorWrite::SampledImage(vec![(&resource_cache.image_srv, hal::image::ImageLayout::Undefined)]),
            },
            hal::pso::DescriptorSetWrite {
                set: &desc_sets[0],
                binding: 18,
                array_offset: 0,
                write: hal::pso::DescriptorWrite::Sampler(vec![&sampler_nearest]),
            },
            hal::pso::DescriptorSetWrite {
                set: &desc_sets[0],
                binding: 19,
                array_offset: 0,
                write: hal::pso::DescriptorWrite::SampledImage(vec![(&node_data.image_srv, hal::image::ImageLayout::Undefined)]),
            },
            hal::pso::DescriptorSetWrite {
                set: &desc_sets[0],
                binding: 20,
                array_offset: 0,
                write: hal::pso::DescriptorWrite::Sampler(vec![&sampler_nearest]),
            },
            hal::pso::DescriptorSetWrite {
                set: &desc_sets[0],
                binding: 21,
                array_offset: 0,
                write: hal::pso::DescriptorWrite::SampledImage(vec![(&render_tasks.image_srv, hal::image::ImageLayout::Undefined)]),
            },
            hal::pso::DescriptorSetWrite {
                set: &desc_sets[0],
                binding: 22,
                array_offset: 0,
                write: hal::pso::DescriptorWrite::Sampler(vec![&sampler_nearest]),
            },
        ]);

        let image_uploads = Vec::new();

        Device {
            device,
            memory_types,
            queue_group,
            command_pool,
            swap_chain: Box::new(swap_chain),
            render_pass,
            framebuffers,
            frame_images,
            viewport,
            vs_module,
            fs_module,
            set_layout,
            pipeline_layout,
            pipelines,
            desc_pool,
            desc_sets,
            vertex_buffer,
            instance_buffer,
            locals_buffer,
            sampler_linear,
            sampler_nearest,
            resource_cache,
            render_tasks,
            node_data,
            image_uploads,
        }
    }

    pub fn create_texture(&mut self, target: TextureTarget) -> Texture {
        Texture { target, width: 0, height: 0,  layer_count: 0, format: ImageFormat::Invalid }
    }

    pub fn update_resource_cache(&mut self, rect: DeviceUintRect, gpu_data: &Vec<[f32; 4]>) {
        debug_assert_eq!(gpu_data.len(), 1024);
        self.image_uploads.push(
            self.resource_cache.update_buffer_and_submit_upload(
                &mut self.device,
                &mut self.command_pool,
                rect.origin,
                gpu_data,
            )
        );
    }

    pub fn update_render_tasks(&mut self, task_data: &Vec<[f32; 12]>) {
        self.image_uploads.push(
            self.render_tasks.update_buffer_and_submit_upload(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                task_data,
            )
        );
    }

    pub fn update_node_data(&mut self, node_data: &Vec<[f32; 28]>) {
        self.image_uploads.push(
            self.node_data.update_buffer_and_submit_upload(
                &mut self.device,
                &mut self.command_pool,
                DeviceUintPoint::zero(),
                node_data,
            )
        );
    }

    pub fn max_texture_size(&self) -> u32 {
        1024u32
    }

    pub fn cleanup(mut self) {
        self.device.destroy_command_pool(self.command_pool.downgrade());
        self.device.destroy_descriptor_pool(self.desc_pool);
        self.device.destroy_descriptor_set_layout(self.set_layout);

        {
            self.device.destroy_shader_module(self.vs_module);
            self.device.destroy_shader_module(self.fs_module);
        }

        self.device.destroy_buffer(self.vertex_buffer.buffer);
        self.device.destroy_buffer(self.instance_buffer.buffer);
        self.device.destroy_buffer(self.locals_buffer.buffer);
        self.device.destroy_pipeline_layout(self.pipeline_layout);

        self.device.free_memory(self.vertex_buffer.memory);
        self.device.free_memory(self.instance_buffer.memory);
        self.device.free_memory(self.locals_buffer.memory);

        self.device.destroy_renderpass(self.render_pass);
        for pipeline in self.pipelines {
            if let Ok(pipeline) = pipeline {
                self.device.destroy_graphics_pipeline(pipeline);
            }
        }

        for framebuffer in self.framebuffers {
            self.device.destroy_framebuffer(framebuffer);
        }

        for (image, rtv) in self.frame_images {
            self.device.destroy_image_view(rtv);
            self.device.destroy_image(image);
        }
    }

    pub fn swap_buffers(&mut self) {
        let mut frame_semaphore = self.device.create_semaphore();
        let mut frame_fence = self.device.create_fence(false); // TODO: remove
        {
            self.device.reset_fences(&[&frame_fence]);
            //self.command_pool.reset();

            let frame = self.swap_chain.acquire_frame(FrameSync::Semaphore(&mut frame_semaphore));

            // Rendering
            let submit = {
                let mut cmd_buffer = self.command_pool.acquire_command_buffer();

                cmd_buffer.set_viewports(&[self.viewport.clone()]);
                cmd_buffer.set_scissors(&[self.viewport.rect]);
                cmd_buffer.bind_graphics_pipeline(&self.pipelines[0].as_ref().unwrap());
                cmd_buffer.bind_vertex_buffers(hal::pso::VertexBufferSet(vec![(&self.vertex_buffer.buffer, 0), (&self.instance_buffer.buffer, 0)]));
                cmd_buffer.bind_graphics_descriptor_sets(&self.pipeline_layout, 0, &self.desc_sets[0..1]);

                {
                    let mut encoder = cmd_buffer.begin_renderpass_inline(
                        &self.render_pass,
                        &self.framebuffers[frame.id()],
                        self.viewport.rect,
                        &[hal::command::ClearValue::Color(hal::command::ClearColor::Float([0.3, 0.0, 0.0, 1.0]))],
                    );
                    encoder.draw(0..6, 0..1);
                }

                cmd_buffer.finish()
            };

            self.image_uploads.push(submit);

            let submission = Submission::new()
                .wait_on(&[(&mut frame_semaphore, PipelineStage::BOTTOM_OF_PIPE)])
                .submit(&self.image_uploads);
            self.queue_group.queues[0].submit(submission, Some(&mut frame_fence));


            // TODO: replace with semaphore
            self.device.wait_for_fences(&[&frame_fence], hal::device::WaitFor::All, !0);

            // present frame
            self.swap_chain.present(&mut self.queue_group.queues[0], &[]);
        }
        self.image_uploads.clear();
        self.device.destroy_fence(frame_fence);
        self.device.destroy_semaphore(frame_semaphore);
    }
}