/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorF, ImageFormat, MemoryReport};
use api::round_to_int;
use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize};
use api::TextureTarget;
#[cfg(feature = "capture")]
use api::ImageDescriptor;
use arrayvec::ArrayVec;
use euclid::Transform3D;
use internal_types::{FastHashMap, RenderTargetInfo};
use rand::{self, Rng};
use rendy_memory::{Block, Heaps, HeapsConfig, MemoryUsageValue};
use rendy_descriptor::{DescriptorAllocator, DescriptorRanges, DescriptorSet};
use ron::de::from_str;
use smallvec::SmallVec;
use std::cell::Cell;
use std::convert::Into;
use std::collections::hash_map::Entry;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::mem;
use std::path::PathBuf;
use std::rc::Rc;
use std::slice;

use super::blend_state::*;
use super::buffer::*;
use super::command::*;
use super::descriptor::*;
use super::image::*;
use super::program::{Program, PUSH_CONSTANT_BLOCK_SIZE};
use super::render_pass::*;
use super::{PipelineRequirements, PrimitiveType, TextureId};
use super::{LESS_EQUAL_TEST, LESS_EQUAL_WRITE};
use super::vertex_types;

use super::super::Capabilities;
use super::super::{ShaderKind, ExternalTexture, GpuFrameId, TextureSlot, TextureFilter};
use super::super::{VertexDescriptor, UploadMethod, Texel, ReadPixelsFormat, TextureFlags};
use super::super::{Texture, DrawTarget, ReadTarget, FBOId, RBOId, VertexUsageHint, ShaderError, ShaderPrecacheFlags, SharedDepthTarget, ProgramCache};
use super::super::{depth_target_size_in_bytes, record_gpu_alloc, record_gpu_free};
use super::super::super::shader_source;

use hal;
use hal::pso::{BlendState, DepthTest};
use hal::{Device as BackendDevice, PhysicalDevice, Surface, Swapchain};
use hal::{SwapchainConfig, AcquireError};
use hal::pso::PipelineStage;
use hal::queue::RawCommandQueue;
use hal::window::PresentError;
use hal::command::{CommandBufferFlags, CommandBufferInheritanceInfo, RawCommandBuffer, RawLevel};
use hal::pool::{RawCommandPool};
use hal::queue::{QueueFamilyId};

pub const INVALID_TEXTURE_ID: TextureId = 0;
pub const INVALID_PROGRAM_ID: ProgramId = ProgramId(0);
pub const DEFAULT_READ_FBO: FBOId = FBOId(0);
pub const DEFAULT_DRAW_FBO: FBOId = FBOId(1);
pub const DEBUG_READ_FBO: FBOId = FBOId(2);

const COLOR_RANGE: hal::image::SubresourceRange = hal::image::SubresourceRange {
    aspects: hal::format::Aspects::COLOR,
    levels: 0 .. 1,
    layers: 0 .. 1,
};

#[derive(PartialEq)]
pub enum BackendApiType {
    Vulkan,
    Dx12,
    Metal,
}

const QUAD: [vertex_types::Vertex; 6] = [
    vertex_types::Vertex {
        aPosition: [0.0, 0.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [0.0, 1.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [1.0, 0.0, 0.0],
    },
    vertex_types::Vertex {
        aPosition: [1.0, 1.0, 0.0],
    },
];

pub struct DeviceInit<B: hal::Backend> {
    pub instance: Box<dyn hal::Instance<Backend = B>>,
    pub adapter: hal::Adapter<B>,
    pub surface: Option<B::Surface>,
    pub window_size: (i32, i32),
    pub descriptor_count: Option<u32>,
    pub cache_path: Option<PathBuf>,
    pub save_cache: bool,
    pub backend_api: BackendApiType,
}

const NON_SPECIALIZATION_FEATURES: &'static [&'static str] =
    &["TEXTURE_RECT", "TEXTURE_2D", "DUAL_SOURCE_BLENDING"];

#[repr(u32)]
pub enum DepthFunction {
    Less,
    LessEqual,
}

impl ShaderKind {
    pub(super) fn is_debug(&self) -> bool {
        match *self {
            ShaderKind::DebugFont | ShaderKind::DebugColor => true,
            _ => false,
        }
    }
}

impl Texture {
    pub fn still_in_flight(&self, frame_id: GpuFrameId, frame_count: usize) -> bool {
        for i in 0 .. frame_count {
            if self.bound_in_frame.get() == GpuFrameId(frame_id.0 - i) {
                return true;
            }
        }
        false
    }
}

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct ProgramId(u32);

pub struct PBO;
pub struct VAO;

struct Fence<B: hal::Backend> {
    inner: B::Fence,
    is_submitted: bool,
}

pub struct Device<B: hal::Backend> {
    pub device: B::Device,
    heaps: Heaps<B>,
    pub limits: hal::Limits,
    adapter: hal::Adapter<B>,
    surface: Option<B::Surface>,
    _instance: Box<dyn hal::Instance<Backend = B>>,
    pub surface_format: ImageFormat,
    pub depth_format: hal::format::Format,
    pub queue_group_family: QueueFamilyId,
    pub queue_group_queues: Vec<B::CommandQueue>,
    pub command_pool: SmallVec<[CommandPool<B>; 1]>,
    staging_buffer_pool: SmallVec<[BufferPool<B>; 1]>,
    pub swap_chain: Option<B::Swapchain>,
    render_pass: Option<RenderPass<B>>,
    pub framebuffers: Vec<B::Framebuffer>,
    pub framebuffers_depth: Vec<B::Framebuffer>,
    frame_images: Vec<ImageCore<B>>,
    frame_depths: Vec<DepthBuffer<B>>,
    pub frame_count: usize,
    pub viewport: hal::pso::Viewport,
    pub sampler_linear: B::Sampler,
    pub sampler_nearest: B::Sampler,
    pub current_frame_id: usize,
    current_blend_state: Cell<BlendState>,
    blend_color: Cell<ColorF>,
    current_depth_test: DepthTest,
    // device state
    programs: FastHashMap<ProgramId, Program<B>>,
    shader_modules: FastHashMap<String, (B::ShaderModule, B::ShaderModule)>,
    images: FastHashMap<TextureId, Image<B>>,
    bound_gpu_cache: TextureId,
    gpu_cache_buffers: FastHashMap<TextureId, PMBuffer<B>>,
    retained_textures: Vec<Texture>,
    fbos: FastHashMap<FBOId, Framebuffer<B>>,
    rbos: FastHashMap<RBOId, DepthBuffer<B>>,

    desc_allocator: DescriptorAllocator<B>,

    per_draw_descriptors: DescriptorSetHandler<PerDrawBindings, B, Vec<DescriptorSet<B>>>,
    bound_per_draw_bindings: PerDrawBindings,

    per_pass_descriptors: DescriptorSetHandler<PerPassBindings, B, Vec<DescriptorSet<B>>>,
    bound_per_pass_textures: PerPassBindings,

    per_group_descriptors: DescriptorSetHandler<(DescriptorGroup, PerGroupBindings), B, FastHashMap<DescriptorGroup, Vec<DescriptorSet<B>>>>,
    bound_per_group_textures: PerGroupBindings,

    // Locals related things
    locals_descriptors: DescriptorSetHandler<Locals, B, Vec<DescriptorSet<B>>>,
    bound_locals: Locals,

    descriptor_data: DescriptorData<B>,
    bound_textures: [u32; RENDERER_TEXTURE_COUNT],
    bound_program: ProgramId,
    bound_sampler: [TextureFilter; RENDERER_TEXTURE_COUNT],
    bound_read_texture: (TextureId, i32),
    bound_read_fbo: FBOId,
    bound_draw_fbo: FBOId,
    scissor_rect: Option<DeviceIntRect>,
    //default_read_fbo: FBOId,
    //default_draw_fbo: FBOId,
    device_pixel_ratio: f32,
    depth_available: bool,
    upload_method: UploadMethod,
    locals_buffer: UniformBufferHandler<B>,
    quad_buffer: VertexBufferHandler<B>,
    instance_buffers: SmallVec<[InstanceBufferHandler<B>; 1]>,
    free_instance_buffers: SmallVec<[InstancePoolBuffer<B>; 16]>,
    download_buffer: Option<Buffer<B>>,
    instance_range: std::ops::Range<usize>,

    // HW or API capabilities
    capabilities: Capabilities,

    /// Map from texture dimensions to shared depth buffers for render targets.
    ///
    /// Render targets often have the same width/height, so we can save memory
    /// by sharing these across targets.
    depth_targets: FastHashMap<DeviceIntSize, SharedDepthTarget>,

    // debug
    inside_frame: bool,

    // resources
    _resource_override_path: Option<PathBuf>,

    max_texture_size: i32,
    _renderer_name: String,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: GpuFrameId,

    // Supported features
    features: hal::Features,

    next_id: usize,
    frame_fence: SmallVec<[Fence<B>; 1]>,
    image_available_semaphore: B::Semaphore,
    render_finished_semaphore: B::Semaphore,
    pipeline_requirements: FastHashMap<String, PipelineRequirements>,
    pipeline_cache: Option<B::PipelineCache>,
    cache_path: Option<PathBuf>,
    save_cache: bool,
    wait_for_resize: bool,

    // The device supports push constants
    pub use_push_consts: bool,
}

impl<B: hal::Backend> Device<B> {
    pub fn new(
        init: DeviceInit<B>,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _cached_programs: Option<Rc<ProgramCache>>,
        heaps_config: HeapsConfig,
        instance_buffer_size: usize,
        texture_cache_size: usize,
    ) -> Self {
        let DeviceInit {
            instance,
            adapter,
            mut surface,
            window_size,
            descriptor_count,
            cache_path,
            save_cache,
            backend_api,
        } = init;
        let renderer_name = "TODO renderer name".to_owned();
        let features = adapter.physical_device.features();

        let memory_properties = adapter.physical_device.memory_properties();
        let mut heaps = {
            let types = memory_properties.memory_types.iter().map(|ref mt| {
                let mut config = heaps_config;
                if let Some(mut linear) = config.linear {
                    if !mt
                        .properties
                        .contains(gfx_hal::memory::Properties::CPU_VISIBLE)
                    {
                        config.linear = None;
                    } else {
                        linear.linear_size = linear.linear_size.min(
                            (memory_properties.memory_heaps[mt.heap_index as usize] / 8 - 1)
                                .next_power_of_two(),
                        );
                    }
                }
                if let Some(mut dynamic) = config.dynamic {
                    dynamic.min_device_allocation = dynamic.min_device_allocation.min(
                        (memory_properties.memory_heaps[mt.heap_index as usize] / 1048)
                            .next_power_of_two(),
                    );
                    dynamic.block_size_granularity = dynamic.block_size_granularity.min(
                        (memory_properties.memory_heaps[mt.heap_index as usize] / 1024 - 1)
                            .next_power_of_two(),
                    );
                    dynamic.max_chunk_size = dynamic.max_chunk_size.min(
                        (memory_properties.memory_heaps[mt.heap_index as usize] / 8 - 1)
                            .next_power_of_two(),
                    );
                }
                (mt.properties, mt.heap_index as u32, config)
            });

            let heaps = memory_properties.memory_heaps.iter().cloned();
            unsafe { Heaps::new(types, heaps) }
        };

        let limits = adapter.physical_device.limits();
        let max_texture_size = 4400i32; // TODO use limits after it points to the correct texture size

        let (device, queue_group_family, queue_group_queues) = {
            use hal::Capability;
            use hal::queue::QueueFamily;

            let family = adapter
                .queue_families
                .iter()
                .find(|family| {
                    hal::Graphics::supported_by(family.queue_type())
                        && match &surface {
                            Some(surface) => surface.supports_queue_family(family),
                            None => true,
                        }
                })
                .unwrap();

            let priorities = vec![1.0];
            let (id, families) = (family.id(), [(family, priorities.as_slice())]);
            let hal::Gpu { device, mut queues } = unsafe {
                adapter
                    .physical_device
                    .open(&families, hal::Features::DUAL_SRC_BLENDING)
                    .unwrap_or_else(|_| {
                        adapter
                            .physical_device
                            .open(&families, hal::Features::empty())
                            .unwrap()
                    })
            };
            (device, id, queues.take_raw(id).unwrap())
        };

        let surface_format = hal::format::Format::Bgra8Unorm;
        let depth_format = hal::format::Format::D32Sfloat;
        let render_pass = Device::create_render_passes(&device, surface_format, depth_format);

        // Disable push constants for Intel's Vulkan driver on Windows
        let has_broken_push_const_support = cfg!(target_os = "windows")
            && backend_api == BackendApiType::Vulkan
            && adapter.info.vendor == 0x8086;
        let use_push_consts = !has_broken_push_const_support;

        let (
            swap_chain,
            surface_format,
            depth_format,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport,
            frame_count,
        ) = match surface.as_mut() {
            Some(surface) => {
                let (
                    swap_chain,
                    surface_format,
                    depth_format,
                    framebuffers,
                    framebuffers_depth,
                    frame_depths,
                    frame_images,
                    viewport,
                    frame_count,
                ) = Device::init_swapchain_resources(
                    &device,
                    &mut heaps,
                    &adapter,
                    surface,
                    Some(window_size),
                    None,
                    &render_pass,
                );
                (
                    Some(swap_chain),
                    surface_format,
                    depth_format,
                    framebuffers,
                    framebuffers_depth,
                    frame_depths,
                    frame_images,
                    viewport,
                    frame_count,
                )
            }
            None => {
                let (
                    surface_format,
                    depth_format,
                    framebuffers,
                    framebuffers_depth,
                    frame_depths,
                    frame_images,
                    viewport,
                    frame_count,
                ) = Device::init_resources_without_surface(
                    &device,
                    &mut heaps,
                    window_size,
                    &render_pass
                );
                (
                    None,
                    surface_format,
                    depth_format,
                    framebuffers,
                    framebuffers_depth,
                    frame_depths,
                    frame_images,
                    viewport,
                    frame_count,
                )
            }
        };

        // Samplers
        let sampler_linear = unsafe {
            device.create_sampler(hal::image::SamplerInfo::new(
                hal::image::Filter::Linear,
                hal::image::WrapMode::Clamp,
            ))
        }
        .expect("sampler_linear failed");

        let sampler_nearest = unsafe {
            device.create_sampler(hal::image::SamplerInfo::new(
                hal::image::Filter::Nearest,
                hal::image::WrapMode::Clamp,
            ))
        }
        .expect("sampler_linear failed");

        let pipeline_requirements: FastHashMap<String, PipelineRequirements> =
            from_str(&shader_source::PIPELINES).expect("Failed to load pipeline requirements");

        let mut desc_allocator = DescriptorAllocator::new();

        let mut frame_fence = SmallVec::new();
        let mut command_pool = SmallVec::new();
        let mut staging_buffer_pool = SmallVec::new();
        let mut instance_buffers = SmallVec::new();
        for i in 0 .. frame_count {
            let fence = device.create_fence(false).expect("create_fence failed");
            frame_fence.push(Fence {
                inner: fence,
                is_submitted: false,
            });

            let mut hal_cp = unsafe {
                device.create_command_pool(
                    queue_group_family,
                    hal::pool::CommandPoolCreateFlags::empty(),
                )
            }
            .expect("create_command_pool failed");
            unsafe { hal_cp.reset() };
            let mut cp = CommandPool::new(hal_cp);
            cp.create_command_buffer();
            // Start recording for the 1st frame
            if i == 0 {
                unsafe { Self::begin_cmd_buffer(cp.command_buffer_mut()) };
            }
            command_pool.push(cp);
            staging_buffer_pool.push(BufferPool::new(
                &device,
                &mut heaps,
                hal::buffer::Usage::TRANSFER_SRC,
                1,
                (limits.non_coherent_atom_size - 1) as usize,
                (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                (limits.optimal_buffer_copy_offset_alignment - 1) as usize,
                texture_cache_size,
            ));
            instance_buffers.push(InstanceBufferHandler::new(
                (limits.non_coherent_atom_size - 1) as usize,
                (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                instance_buffer_size,
            ));
        }

        let locals_buffer = UniformBufferHandler::new(
            hal::buffer::Usage::UNIFORM,
            mem::size_of::<Locals>(),
            (limits.min_uniform_buffer_offset_alignment - 1) as usize,
        );

        let quad_buffer = VertexBufferHandler::new(
            &device,
            &mut heaps,
            hal::buffer::Usage::VERTEX,
            &QUAD,
            (limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
            (limits.non_coherent_atom_size - 1) as usize,
        );

        let mut per_group_descriptor_sets = FastHashMap::default();
        let descriptor_data:
            FastHashMap<DescriptorGroup, DescriptorGroupData<B>>
             = [DescriptorGroup::Default, DescriptorGroup::Clip, DescriptorGroup::Primitive]
            .iter()
            .map(|g| {
                let layouts_and_samplers = match g {
                    DescriptorGroup::Default => [
                        (EMPTY_SET_0, vec![]),
                        (DEFAULT_SET_1, vec![&sampler_nearest]),
                        (COMMON_SET_2, vec![]),
                        (COMMON_SET_3, vec![]),
                    ],
                    DescriptorGroup::Clip => [
                        (EMPTY_SET_0, vec![]),
                        (CLIP_SET_1, vec![&sampler_nearest, &sampler_nearest, &sampler_nearest, &sampler_nearest]),
                        (COMMON_SET_2, vec![]),
                        (COMMON_SET_3, vec![]),
                    ],
                    DescriptorGroup::Primitive => [
                        (PRIMITIVE_SET_0, vec![&sampler_linear, &sampler_linear]),
                        (PRIMITIVE_SET_1, vec![&sampler_nearest, &sampler_nearest, &sampler_nearest, &sampler_nearest, &sampler_nearest, &sampler_nearest]),
                        (COMMON_SET_2, vec![]),
                        (COMMON_SET_3, vec![]),
                    ],
                };

                let ranges = layouts_and_samplers
                    .iter()
                    .map(|(bindings, _)| {
                        DescriptorRanges::from_bindings(bindings)
                    }).collect::<ArrayVec<_>>();

                let set_layouts: ArrayVec<[B::DescriptorSetLayout; MAX_DESCRIPTOR_SET_COUNT]> = layouts_and_samplers
                    .iter()
                    .enumerate()
                    .map(|(index, (bindings, immutable_samplers))| {
                        let layout = unsafe { device.create_descriptor_set_layout(*bindings, immutable_samplers.iter().map(|s| *s)) }
                            .expect("create_descriptor_set_layout failed");
                        if index == DESCRIPTOR_SET_PER_GROUP {
                            let mut descriptor_sets = Vec::new();
                            unsafe {
                                desc_allocator.allocate(
                                    &device,
                                    &layout,
                                    ranges[index],
                                    descriptor_count.unwrap_or(DESCRIPTOR_COUNT),
                                    &mut descriptor_sets,
                                )
                            }.expect("Allocate descriptor sets failed");
                            per_group_descriptor_sets.insert(*g, descriptor_sets);
                        }
                        layout
                    }).collect();

                let pipeline_layout = unsafe {
                    device.create_pipeline_layout(
                        &set_layouts,
                        Some((hal::pso::ShaderStageFlags::VERTEX, 0..PUSH_CONSTANT_BLOCK_SIZE as u32)),
                    )
                }
                .expect("create_pipeline_layout failed");
                (*g, DescriptorGroupData {
                    set_layouts,
                    ranges,
                    pipeline_layout,
                })
            }).collect();

        let descriptor_data = DescriptorData(descriptor_data);

        let per_pass_descriptors = DescriptorSetHandler::new(
            &device,
            &mut desc_allocator,
            &descriptor_data,
            &DescriptorGroup::Primitive,
            DESCRIPTOR_SET_PER_PASS,
            descriptor_count.unwrap_or(DESCRIPTOR_COUNT),
            Vec::new(),
        );

        let per_draw_descriptors = DescriptorSetHandler::new(
            &device,
            &mut desc_allocator,
            &descriptor_data,
            &DescriptorGroup::Default,
            DESCRIPTOR_SET_PER_DRAW,
            descriptor_count.unwrap_or(DESCRIPTOR_COUNT),
            Vec::new(),
        );

        let locals_descriptors = DescriptorSetHandler::new(
            &device,
            &mut desc_allocator,
            &descriptor_data,
            &DescriptorGroup::Default,
            DESCRIPTOR_SET_LOCALS,
            if use_push_consts { 1 } else { descriptor_count.unwrap_or(DESCRIPTOR_COUNT) },
            Vec::new(),
        );

        let image_available_semaphore = device.create_semaphore().expect("create_semaphore failed");
        let render_finished_semaphore = device.create_semaphore().expect("create_semaphore failed");

        let pipeline_cache = if let Some(ref path) = cache_path {
            Self::load_pipeline_cache(&device, &path, &adapter.physical_device)
        } else {
            None
        };

        Device {
            device,
            heaps,
            limits,
            surface_format,
            adapter,
            surface,
            _instance: instance,
            depth_format,
            queue_group_family,
            queue_group_queues,
            command_pool,
            staging_buffer_pool,
            swap_chain: swap_chain,
            render_pass: Some(render_pass),
            framebuffers,
            framebuffers_depth,
            frame_images,
            frame_depths,
            frame_count,
            viewport,
            sampler_linear,
            sampler_nearest,
            current_frame_id: 0,
            current_blend_state: Cell::new(BlendState::Off),
            current_depth_test: DepthTest::Off,
            blend_color: Cell::new(ColorF::new(0.0, 0.0, 0.0, 0.0)),
            _resource_override_path: resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            depth_available: true,
            upload_method,
            inside_frame: false,

            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },
            depth_targets: FastHashMap::default(),

            programs: FastHashMap::default(),
            shader_modules: FastHashMap::default(),
            images: FastHashMap::default(),
            bound_gpu_cache: INVALID_TEXTURE_ID,
            gpu_cache_buffers: FastHashMap::default(),
            retained_textures: Vec::new(),
            fbos: FastHashMap::default(),
            rbos: FastHashMap::default(),
            desc_allocator,

            per_draw_descriptors,
            bound_per_draw_bindings: PerDrawBindings::default(),

            per_pass_descriptors,
            bound_per_pass_textures: PerPassBindings::default(),

            per_group_descriptors: DescriptorSetHandler::from_existing(per_group_descriptor_sets),
            bound_per_group_textures: PerGroupBindings::default(),

            locals_descriptors,
            bound_locals: Locals::default(),
            descriptor_data,

            bound_textures: [0; RENDERER_TEXTURE_COUNT],
            bound_program: INVALID_PROGRAM_ID,
            bound_sampler: [TextureFilter::Linear; RENDERER_TEXTURE_COUNT],
            bound_read_fbo: DEFAULT_READ_FBO,
            bound_read_texture: (INVALID_TEXTURE_ID, 0),
            bound_draw_fbo: DEFAULT_DRAW_FBO,
            scissor_rect: None,

            max_texture_size,
            _renderer_name: renderer_name,
            frame_id: GpuFrameId(0),
            features,

            next_id: 0,
            frame_fence,
            image_available_semaphore,
            render_finished_semaphore,
            pipeline_requirements,
            pipeline_cache,
            cache_path,
            save_cache,

            locals_buffer,
            quad_buffer,
            instance_buffers,
            free_instance_buffers: SmallVec::new(),
            download_buffer: None,
            instance_range: 0..0,
            wait_for_resize: false,

            use_push_consts,
        }
    }

    fn load_pipeline_cache(
        device: &B::Device,
        path: &PathBuf,
        physical_device: &B::PhysicalDevice,
    ) -> Option<B::PipelineCache> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(_) => {
                warn!("File not found: {:?}", path);
                return None;
            }
        };
        let mut bytes = Vec::new();
        match file.read_to_end(&mut bytes) {
            Err(_) => {
                warn!("Failed to read file: {:?}", path);
                return None;
            }
            _ => {}
        };

        if physical_device.is_valid_cache(&bytes) {
            let cache = unsafe { device.create_pipeline_cache(Some(&bytes)) }.expect(&format!(
                "Failed to create pipeline cache from file: {:?}",
                path
            ));
            Some(cache)
        } else {
            None
        }
    }

    pub(crate) fn recreate_swapchain(&mut self, window_size: Option<(i32, i32)>) -> DeviceIntSize {
        self.device.wait_idle().unwrap();

        for image in self.frame_images.drain(..) {
            image.deinit(&self.device, &mut self.heaps);
        }

        for depth in self.frame_depths.drain(..) {
            depth.deinit(&self.device, &mut self.heaps);
        }

        self.bound_per_draw_bindings = PerDrawBindings::default();
        self.per_draw_descriptors.reset();

        self.bound_per_pass_textures = PerPassBindings::default();
        self.per_pass_descriptors.reset();

        self.bound_per_group_textures = PerGroupBindings::default();
        self.per_group_descriptors.reset();

        if !self.use_push_consts {
            self.bound_locals = Locals::default();
            self.locals_descriptors.reset();
        }

        self.locals_buffer.reset();

        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer);
            }
            for framebuffer_depth in self.framebuffers_depth.drain(..) {
                self.device.destroy_framebuffer(framebuffer_depth);
            }
        }

        let (
            swap_chain,
            surface_format,
            depth_format,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport,
            _frame_count,
        ) = if let Some (ref mut surface) = self.surface {
            let (
                swap_chain,
                surface_format,
                depth_format,
                framebuffers,
                framebuffers_depth,
                frame_depths,
                frame_images,
                viewport,
                frame_count,
            ) = Device::init_swapchain_resources(
                &self.device,
                &mut self.heaps,
                &self.adapter,
                surface,
                window_size,
                self.swap_chain.take(),
                self.render_pass.as_ref().unwrap()
            );
            (
                Some(swap_chain),
                surface_format,
                depth_format,
                framebuffers,
                framebuffers_depth,
                frame_depths,
                frame_images,
                viewport,
                frame_count,
            )
        } else {
            let (
                surface_format,
                depth_format,
                framebuffers,
                framebuffers_depth,
                frame_depths,
                frame_images,
                viewport,
                frame_count,
            ) = Device::init_resources_without_surface(
                &self.device,
                &mut self.heaps,
                window_size.unwrap_or((0,0)),
                self.render_pass.as_ref().unwrap()
            );
            (
                None,
                surface_format,
                depth_format,
                framebuffers,
                framebuffers_depth,
                frame_depths,
                frame_images,
                viewport,
                frame_count,
            )
        };

        self.swap_chain = swap_chain;
        self.framebuffers = framebuffers;
        self.framebuffers_depth = framebuffers_depth;
        self.frame_depths = frame_depths;
        self.frame_images = frame_images;
        self.viewport = viewport;
        self.surface_format = surface_format;
        self.depth_format = depth_format;
        self.wait_for_resize = false;

        let pipeline_cache = unsafe { self.device.create_pipeline_cache(None) }
            .expect("Failed to create pipeline cache");
        if let Some(ref cache) = self.pipeline_cache {
            unsafe {
                self.device
                    .merge_pipeline_caches(&cache, Some(&pipeline_cache))
                    .expect("merge_pipeline_caches failed");;
                self.device.destroy_pipeline_cache(pipeline_cache);
            }
        } else {
            self.pipeline_cache = Some(pipeline_cache);
        }
        DeviceIntSize::new(self.viewport.rect.w.into(), self.viewport.rect.h.into())
    }

    fn init_swapchain_resources(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        adapter: &hal::Adapter<B>,
        surface: &mut B::Surface,
        window_size: Option<(i32, i32)>,
        old_swap_chain: Option<B::Swapchain>,
        render_pass: &RenderPass<B>,
    ) -> (
        B::Swapchain,
        ImageFormat,
        hal::format::Format,
        Vec<B::Framebuffer>,
        Vec<B::Framebuffer>,
        Vec<DepthBuffer<B>>,
        Vec<ImageCore<B>>,
        hal::pso::Viewport,
        usize,
    ) {
        let (caps, formats, present_modes) = surface.compatibility(&adapter.physical_device);
        let present_mode = {
            use hal::window::PresentMode::*;
            [Mailbox, Fifo, Relaxed, Immediate]
                .iter()
                .cloned()
                .find(|pm| present_modes.contains(pm))
                .expect("No PresentMode values specified!")
        };
        let surface_format = formats.map_or(hal::format::Format::Bgra8Unorm, |formats| {
            formats
                .into_iter()
                .find(|format| format == &hal::format::Format::Bgra8Unorm)
                .expect("Bgra8Unorm surface is not supported!")
        });
        let ext = caps.current_extent.expect("Can't acquire current extent!");
        let ext = (ext.width as i32, ext.height as i32);
        let mut extent =
            hal::window::Extent2D {
                width: (window_size.unwrap_or(ext).0 as u32)
                        .min(caps.extents.end.width)
                        .max(caps.extents.start.width),
                height: (window_size.unwrap_or(ext).1 as u32)
                        .min(caps.extents.end.height)
                        .max(caps.extents.start.height),
            };

        if extent.width == 0 {
            extent.width = 1;
        }
        if extent.height == 0 {
            extent.height = 1;
        }
        let mut swap_config = SwapchainConfig::new(
            extent.width,
            extent.height,
            surface_format,
            caps.image_count.start,
        )
        .with_image_usage(
            hal::image::Usage::TRANSFER_SRC
                | hal::image::Usage::TRANSFER_DST
                | hal::image::Usage::COLOR_ATTACHMENT,
        )
        .with_mode(present_mode);
        if caps.composite_alpha.contains(hal::CompositeAlpha::INHERIT) {
            swap_config.composite_alpha =  hal::CompositeAlpha::INHERIT;
        } else if caps.composite_alpha.contains(hal::CompositeAlpha::OPAQUE) {
            swap_config.composite_alpha = hal::CompositeAlpha::OPAQUE;
        }

        let (swap_chain, images) =
            unsafe { device.create_swapchain(surface, swap_config, old_swap_chain) }
                .expect("create_swapchain failed");
        let depth_format = hal::format::Format::D32Sfloat; //maybe d24s8?

        let image_format = match surface_format {
            hal::format::Format::Bgra8Unorm => ImageFormat::BGRA8,
            f => unimplemented!("Unsupported surface format: {:?}", f),
        };
        let mut frame_depths = Vec::new();
        // Framebuffer and render target creation
        let (frame_images, framebuffers, framebuffers_depth) = {
                let extent = hal::image::Extent {
                    width: extent.width as _,
                    height: extent.height as _,
                    depth: 1,
                };
                let cores = images
                    .into_iter()
                    .map(|image| {
                        frame_depths.push(DepthBuffer::new(
                            device,
                            heaps,
                            extent.width,
                            extent.height,
                            depth_format,
                        ));
                        ImageCore::from_image(
                            device,
                            image,
                            hal::image::ViewKind::D2Array,
                            surface_format,
                            COLOR_RANGE.clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                let fbos = cores
                    .iter()
                    .map(|core| {
                        unsafe {
                            device.create_framebuffer(
                                render_pass.get_render_pass(image_format, false),
                                Some(&core.view),
                                extent,
                            )
                        }
                        .expect("create_framebuffer failed")
                    })
                    .collect();
                let fbos_depth = cores
                    .iter()
                    .zip(frame_depths.iter())
                    .map(|(core, depth)| {
                        unsafe {
                            device.create_framebuffer(
                                render_pass.get_render_pass(image_format, true),
                                Some(&core.view).into_iter().chain(Some(&depth.core.view)),
                                extent,
                            )
                        }
                        .expect("create_framebuffer failed")
                    })
                    .collect();
                (cores, fbos, fbos_depth)
        };

        info!("Frame images: {:?}\nFrame depths: {:?}", frame_images, frame_depths);

        // Rendering setup
        let viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0 .. 1.0,
        };
        (
            swap_chain,
            image_format,
            depth_format,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport,
            if present_mode == hal::window::PresentMode::Mailbox {
                (caps.image_count.end - 1).min(3) as usize
            } else {
                (caps.image_count.end - 1).min(2) as usize
            },
        )
    }

    fn init_resources_without_surface(
        device: &B::Device,
        heaps: &mut Heaps<B>,
        window_size: (i32, i32),
        render_pass: &RenderPass<B>,
    ) -> (
        ImageFormat,
        hal::format::Format,
        Vec<B::Framebuffer>,
        Vec<B::Framebuffer>,
        Vec<DepthBuffer<B>>,
        Vec<ImageCore<B>>,
        hal::pso::Viewport,
        usize,
    ) {
        let surface_format = ImageFormat::BGRA8;
        let depth_format = hal::format::Format::D32Sfloat;

        let extent = hal::image::Extent {
            width: window_size.0 as _,
            height: window_size.1 as _,
            depth: 1,
        };
        let frame_count = 2;
        let (frame_images, frame_depths, framebuffers, framebuffers_depth) = {
            let mut cores = Vec::new();
            let mut frame_depths = Vec::new();
            let mip_levels = 1;
            let kind = hal::image::Kind::D2(
                extent.width as _,
                extent.height as _,
                extent.depth as _,
                1,
            );
            for _ in 0 .. frame_count {
                cores.push(ImageCore::create(
                    device,
                    heaps,
                    kind,
                    hal::image::ViewKind::D2,
                    mip_levels,
                    hal::format::Format::Bgra8Unorm,
                    hal::image::Usage::TRANSFER_SRC
                        | hal::image::Usage::TRANSFER_DST
                        | hal::image::Usage::COLOR_ATTACHMENT,
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: 0 .. 1,
                        layers: 0 .. 1,
                    },
                ));
                frame_depths.push(DepthBuffer::new(
                    device,
                    heaps,
                    extent.width,
                    extent.height,
                    depth_format,
                ));
            }
            let fbos = cores
                .iter()
                .map(|core| {
                    unsafe {
                        device.create_framebuffer(
                            &render_pass.bgra8,
                            Some(&core.view),
                            extent,
                        )
                    }
                    .expect("create_framebuffer failed")
                })
                .collect();
            let fbos_depth = cores
                .iter()
                .zip(frame_depths.iter())
                .map(|(core, depth)| {
                    unsafe {
                        device.create_framebuffer(
                            &render_pass.bgra8_depth,
                            vec![&core.view, &depth.core.view],
                            extent,
                        )
                    }
                    .expect("create_framebuffer failed")
                })
                .collect();
            (cores, frame_depths, fbos, fbos_depth)
        };
        let viewport = hal::pso::Viewport {
            rect: hal::pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0 .. 1.0,
        };

        (
            surface_format,
            depth_format,
            framebuffers,
            framebuffers_depth,
            frame_depths,
            frame_images,
            viewport,
            frame_count,
        )
    }

    fn create_render_passes(
        device: &<B as hal::Backend>::Device,
        surface_format: hal::format::Format,
        depth_format: hal::format::Format,
    ) -> RenderPass<B> {
        let attachment_r8 = hal::pass::Attachment {
            format: Some(hal::format::Format::R8Unorm),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::DontCare,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::ColorAttachmentOptimal
                .. hal::image::Layout::ColorAttachmentOptimal,
        };

        let attachment_bgra8 = hal::pass::Attachment {
            format: Some(surface_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::DontCare,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::ColorAttachmentOptimal
                .. hal::image::Layout::ColorAttachmentOptimal,
        };

        let attachment_depth = hal::pass::Attachment {
            format: Some(depth_format),
            samples: 1,
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::DontCare,
                hal::pass::AttachmentStoreOp::Store,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: hal::image::Layout::DepthStencilAttachmentOptimal
                .. hal::image::Layout::DepthStencilAttachmentOptimal,
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

        let depth_dependency = hal::pass::SubpassDependency {
            passes: hal::pass::SubpassRef::External .. hal::pass::SubpassRef::Pass(0),
            stages: PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::LATE_FRAGMENT_TESTS,
            accesses: hal::image::Access::empty()
                .. (hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
        };

        use std::iter;
        RenderPass {
            r8: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_r8),
                    &[subpass_r8],
                    iter::once(&dependency),
                )
            }
            .expect("create_render_pass failed"),
            r8_depth: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_r8).chain(iter::once(&attachment_depth)),
                    &[subpass_depth_r8],
                    iter::once(&dependency).chain(iter::once(&depth_dependency)),
                )
            }
            .expect("create_render_pass failed"),
            bgra8: unsafe {
                device.create_render_pass(
                    iter::once(&attachment_bgra8),
                    &[subpass_bgra8],
                    iter::once(&dependency),
                )
            }
            .expect("create_render_pass failed"),
            bgra8_depth: unsafe {
                device.create_render_pass(
                    &[attachment_bgra8, attachment_depth],
                    &[subpass_depth_bgra8],
                    &[dependency, depth_dependency],
                )
            }
            .expect("create_render_pass failed"),
        }
    }

    pub fn set_device_pixel_ratio(&mut self, ratio: f32) {
        self.device_pixel_ratio = ratio;
    }

    pub fn update_program_cache(&mut self, _cached_programs: Rc<ProgramCache>) {
        warn!("Program cache is not supported!");
    }

    /// Ensures that the maximum texture size is less than or equal to the
    /// provided value. If the provided value is less than the value supported
    /// by the driver, the latter is used.
    pub fn clamp_max_texture_size(&mut self, size: i32) {
        self.max_texture_size = self.max_texture_size.min(size);
    }

    /// Returns the limit on texture dimensions (width or height).
    pub fn max_texture_size(&self) -> i32 {
        self.max_texture_size
    }

    /// Returns the limit on texture array layers.
    pub fn max_texture_layers(&self) -> usize {
        self.limits.max_image_array_layers as usize
    }

    pub fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn reset_next_frame_resources(&mut self) {
        self.next_id = (self.next_id + 1) % self.frame_count;
        self.reset_state();
        if self.frame_fence[self.next_id].is_submitted {
            unsafe {
                self.device
                    .wait_for_fence(&self.frame_fence[self.next_id].inner, !0)
            }
            .expect("wait_for_fence failed");
            unsafe {
                self.device
                    .reset_fence(&self.frame_fence[self.next_id].inner)
            }
            .expect("reset_fence failed");
            self.frame_fence[self.next_id].is_submitted = false;
        }
        unsafe {
            self.command_pool[self.next_id].reset();
            Self::begin_cmd_buffer(self.command_pool[self.next_id].command_buffer_mut());
        }
        self.staging_buffer_pool[self.next_id].reset();
        self.reset_program_buffer_offsets();
        self.instance_buffers[self.next_id].reset(&mut self.free_instance_buffers);
        self.delete_retained_textures();
    }

    pub fn reset_state(&mut self) {
        self.bound_textures = [INVALID_TEXTURE_ID; RENDERER_TEXTURE_COUNT];
        self.bound_program = INVALID_PROGRAM_ID;
        self.bound_sampler = [TextureFilter::Linear; RENDERER_TEXTURE_COUNT];
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
    }

    fn reset_program_buffer_offsets(&mut self) {
        for program in self.programs.values_mut() {
            if let Some(ref mut index_buffer) = program.index_buffer {
                index_buffer[self.next_id].reset();
                program.vertex_buffer.as_mut().unwrap()[self.next_id].reset();
            }
        }
    }

    pub fn delete_program(&mut self, mut _program: ProgramId) {
        // TODO delete program
        _program = INVALID_PROGRAM_ID;
    }

    pub fn create_program_linked(
        &mut self,
        base_filename: &str,
        features: &str,
        descriptor: &VertexDescriptor,
        shader_kind: &ShaderKind,
    ) -> Result<ProgramId, ShaderError> {
        let program = self.create_program(base_filename, shader_kind, &[features])?;
        self.link_program(program, descriptor)?;
        Ok(program)
    }

    pub fn link_program(
        &mut self,
        _program: ProgramId,
        _descriptor: &VertexDescriptor,
    ) -> Result<(), ShaderError> {
        warn!("link_program is not implemented with gfx backend");
        Ok(())
    }
    pub fn create_program(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
    ) -> Result<ProgramId, ShaderError> {
        let mut name = String::from(shader_name);
        for feature_names in features {
            for feature in feature_names.split(',') {
                if NON_SPECIALIZATION_FEATURES.iter().any(|f| *f == feature) {
                    name.push_str(&format!("_{}", feature.to_lowercase()));
                }
            }
        }

        let desc_group = DescriptorGroup::from(*shader_kind);
        let program = Program::create(
            self.pipeline_requirements
                .get(&name)
                .expect(&format!("Can't load pipeline data for: {}!", name))
                .clone(),
            &self.device,
            self.descriptor_data.pipeline_layout(&desc_group),
            &mut self.heaps,
            &self.limits,
            &name,
            features,
            shader_kind.clone(),
            self.render_pass.as_ref().unwrap(),
            self.frame_count,
            &mut self.shader_modules,
            self.pipeline_cache.as_ref(),
            self.surface_format,
            self.use_push_consts,
        );

        let id = self.generate_program_id();
        self.programs.insert(id, program);
        Ok(id)
    }

    pub fn create_program_with_kind(
        &mut self,
        shader_name: &str,
        shader_kind: &ShaderKind,
        features: &[&str],
        _precache_flags: ShaderPrecacheFlags,
    ) -> Result<ProgramId, ShaderError> {
        self.create_program(shader_name, shader_kind, features)
    }

    pub fn bind_program(&mut self, program_id: &ProgramId) {
        debug_assert!(self.inside_frame);

        if self.bound_program != *program_id {
            self.bound_program = *program_id;
        }
    }

    fn bind_uniforms(&mut self) {
        if self.use_push_consts {
            self.programs
            .get_mut(&self.bound_program)
            .expect("Invalid bound program")
            .constants[..].copy_from_slice(
                unsafe {
                    std::mem::transmute::<_, &[u32; 17]>(&self.bound_locals)
                }
            );
        }

        self.locals_descriptors.bind_locals(
            if self.use_push_consts { Locals::default() } else { self.bound_locals },
            &self.device,
            &mut self.desc_allocator,
            &self.descriptor_data,
            &mut self.locals_buffer,
            &mut self.heaps,
        );
    }

    pub fn set_uniforms(&mut self, _program_id: &ProgramId, projection: &Transform3D<f32>) {
        self.bound_locals.uTransform = projection.to_row_arrays();
    }

    unsafe fn begin_cmd_buffer(cmd_buffer: &mut B::CommandBuffer) {
        let flags = CommandBufferFlags::ONE_TIME_SUBMIT;
        cmd_buffer.begin(flags, CommandBufferInheritanceInfo::default());
    }

    fn bind_textures(&mut self) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");
        let mut cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
        let descriptor_group = program.shader_kind.into();
        // Per draw textures and samplers
        let per_draw_bindings = PerDrawBindings(
            [
                self.bound_textures[0],
                self.bound_textures[1],
                self.bound_textures[2],
            ],
            [
                self.bound_sampler[0],
                self.bound_sampler[1],
                self.bound_sampler[2],
            ],
        );

        self.per_draw_descriptors.bind_textures(
            &self.bound_textures,
            &self.bound_sampler,
            cmd_buffer,
            per_draw_bindings,
            &self.images,
            None,
            &mut self.desc_allocator,
            &self.device,
            &self.descriptor_data,
            &descriptor_group,
            DESCRIPTOR_SET_PER_DRAW,
            0..PER_DRAW_TEXTURE_COUNT,
            &self.sampler_linear,
            &self.sampler_nearest,
        );
        self.bound_per_draw_bindings = per_draw_bindings;

        // Per pass textures
        if descriptor_group == DescriptorGroup::Primitive {
            let per_pass_bindings = PerPassBindings(
                [
                    self.bound_textures[3],
                    self.bound_textures[4],
                ],
            );

            self.per_pass_descriptors.bind_textures(
                &self.bound_textures,
                &self.bound_sampler,
                &mut cmd_buffer,
                per_pass_bindings,
                &self.images,
                None,
                &mut self.desc_allocator,
                &self.device,
                &self.descriptor_data,
                &descriptor_group,
                DESCRIPTOR_SET_PER_PASS,
                PER_DRAW_TEXTURE_COUNT..PER_DRAW_TEXTURE_COUNT + PER_PASS_TEXTURE_COUNT,
                &self.sampler_linear,
                &self.sampler_nearest,
            );
            self.bound_per_pass_textures = per_pass_bindings;
        }

        // Per frame textures
        let per_group_bindings = PerGroupBindings(
            [
                self.bound_gpu_cache,
                self.bound_textures[6],
                self.bound_textures[7],
                self.bound_textures[8],
                self.bound_textures[9],
                self.bound_textures[10],
            ],
        );

        self.per_group_descriptors.bind_textures(
            &self.bound_textures,
            &self.bound_sampler,
            &mut cmd_buffer,
            (descriptor_group, per_group_bindings),
            &self.images,
            match descriptor_group {
                DescriptorGroup::Default => None,
                _ => Some(&self.gpu_cache_buffers[&self.bound_gpu_cache]),
            },
            &mut self.desc_allocator,
            &self.device,
            &self.descriptor_data,
            &descriptor_group,
            DESCRIPTOR_SET_PER_GROUP,
            match descriptor_group {
                DescriptorGroup::Default => PER_GROUP_RANGE_DEFAULT,
                DescriptorGroup::Clip => PER_GROUP_RANGE_CLIP,
                DescriptorGroup::Primitive => PER_GROUP_RANGE_PRIMITIVE,
            },
            &self.sampler_linear,
            &self.sampler_nearest,
        );
        self.bound_per_group_textures = per_group_bindings;
    }

    pub fn update_indices<I: Copy>(&mut self, indices: &[I]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");

        if let Some(ref mut index_buffer) = program.index_buffer {
            index_buffer[self.next_id].update(&self.device, indices, &mut self.heaps);
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    pub fn update_vertices<T: Copy>(&mut self, vertices: &[T]) {
        debug_assert!(self.inside_frame);
        assert_ne!(self.bound_program, INVALID_PROGRAM_ID);
        let program = self
            .programs
            .get_mut(&self.bound_program)
            .expect("Program not found.");

        if program.shader_kind.is_debug() {
            program.vertex_buffer.as_mut().unwrap()[self.next_id].update(&self.device, vertices, &mut self.heaps);
        } else {
            warn!("This function is for debug shaders only!");
        }
    }

    fn update_instances<T: Copy>(&mut self, instances: &[T]) {
        self.instance_range = self.instance_buffers[self.next_id].add(&self.device, instances, &mut self.heaps, &mut self.free_instance_buffers);
    }

    fn draw(&mut self) {
        self.bind_textures();
        self.bind_uniforms();
        let (img, frame_buffer, format, (depth_img, depth_test_changed)) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let texture_id = self.fbos[&self.bound_draw_fbo].texture_id;
            let rbo_id = self.fbos[&self.bound_draw_fbo].rbo;
            (
                &self.images[&texture_id].core,
                &self.fbos[&self.bound_draw_fbo].fbo,
                self.fbos[&self.bound_draw_fbo].format,
                if rbo_id == RBOId(0) {
                    (None, false)
                } else {
                    let mut depth_test_changed =  false;
                    // This is needed to avoid validation layer errors for different attachments between frambuffer renderpass and pipelinelayout
                    if self.current_depth_test == DepthTest::Off {
                        self.current_depth_test = LESS_EQUAL_TEST;
                        depth_test_changed = true;
                    }
                    (Some(&self.rbos[&rbo_id].core), depth_test_changed)
                },
            )
        } else {
            let (frame_buffer, depth_image) = match self.current_depth_test {
                DepthTest::Off => (&self.framebuffers[self.current_frame_id], None),
                _ => (&self.framebuffers_depth[self.current_frame_id], Some(&self.frame_depths[self.current_frame_id].core)),
            };
            (
                &self.frame_images[self.current_frame_id],
                frame_buffer,
                self.surface_format,
                (depth_image, false)
            )
        };
        let rp = self
            .render_pass
            .as_ref()
            .unwrap()
            .get_render_pass(format, depth_img.is_some());

        let before_state = img.state.get();
        let mut before_depth_state = None;
        let mut pre_stage = Some(PipelineStage::empty());
        let mut pre_depth_stage = Some(PipelineStage::empty());
        let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
        unsafe {
            if let Some(barrier) = img.transit(
                hal::image::Access::empty(),
                hal::image::Layout::ColorAttachmentOptimal,
                img.subresource_range.clone(),
                pre_stage.as_mut(),
            ) {
                cmd_buffer.pipeline_barrier(
                    pre_stage.unwrap().. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            if let Some(depth_img) = depth_img {
                before_depth_state = Some(depth_img.state.get());
                if let Some(barrier) = depth_img.transit(
                    hal::image::Access::empty(),
                    hal::image::Layout::DepthStencilAttachmentOptimal,
                    depth_img.subresource_range.clone(),
                    pre_depth_stage.as_mut(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        pre_depth_stage.unwrap() .. PipelineStage::EARLY_FRAGMENT_TESTS,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }
        }

        let descriptor_group = self.programs
            .get(&self.bound_program)
            .expect("Program not found")
            .shader_kind.into();

        let ref desc_set_per_group = self.per_group_descriptors.descriptor_set(&(descriptor_group, self.bound_per_group_textures));
        let desc_set_per_pass = match descriptor_group {
            DescriptorGroup::Primitive => Some(self.per_pass_descriptors.descriptor_set(&self.bound_per_pass_textures)),
            _ => None,
        };
        let ref desc_set_per_draw = self.per_draw_descriptors.descriptor_set(&self.bound_per_draw_bindings);
        let locals = if self.use_push_consts { Locals::default() } else { self.bound_locals };
        let ref desc_set_locals = self.locals_descriptors.descriptor_set(&locals);

        self.programs
            .get_mut(&self.bound_program)
            .expect("Program not found")
            .submit(
                cmd_buffer,
                self.viewport.clone(),
                rp,
                &frame_buffer,
                desc_set_per_draw,
                desc_set_per_pass,
                desc_set_per_group,
                Some(*desc_set_locals),
                &vec![],
                self.current_blend_state.get(),
                self.blend_color.get(),
                self.current_depth_test,
                self.scissor_rect,
                self.next_id,
                self.descriptor_data.pipeline_layout(&descriptor_group),
                self.use_push_consts,
                &self.quad_buffer,
                &self.instance_buffers[self.next_id],
                self.instance_range.clone(),
            );

        if depth_test_changed {
            self.current_depth_test = DepthTest::Off;
        }

        unsafe {
            if let Some(barrier) = img.transit(
                before_state.0,
                before_state.1,
                img.subresource_range.clone(),
                None,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. pre_stage.unwrap(),
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            if let Some(depth_img) = depth_img {
                if let Some(barrier) = depth_img.transit(
                    before_depth_state.unwrap().0,
                    hal::image::Layout::DepthStencilAttachmentOptimal,
                    depth_img.subresource_range.clone(),
                    None,
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::EARLY_FRAGMENT_TESTS .. pre_depth_stage.unwrap(),
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }
        }
    }

    pub fn begin_frame(&mut self) -> GpuFrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;
        self.bound_textures = [INVALID_TEXTURE_ID; RENDERER_TEXTURE_COUNT];
        self.bound_sampler = [TextureFilter::Linear; RENDERER_TEXTURE_COUNT];
        self.bound_read_fbo = DEFAULT_READ_FBO;
        self.bound_draw_fbo = DEFAULT_DRAW_FBO;
        self.bound_locals.uMode = 0;

        self.frame_id
    }

    fn bind_texture_impl(&mut self, slot: TextureSlot, id: TextureId, sampler: TextureFilter) {
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
        texture.bound_in_frame.set(self.frame_id);
    }

    pub fn bind_cache_buffer(&mut self, texture: &Texture) {
        use hal::pso::PipelineStage as PS;
        self.bound_gpu_cache = texture.id;
        let ref buffer = self.gpu_cache_buffers[&self.bound_gpu_cache];
        let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
        if let Some(barrier) = buffer.transit(hal::buffer::Access::HOST_WRITE | hal::buffer::Access::HOST_READ) {
            unsafe {
                cmd_buffer.pipeline_barrier(
                    PS::VERTEX_SHADER | PS::FRAGMENT_SHADER .. PS::HOST,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
        if let Some(barrier) = buffer.transit(hal::buffer::Access::SHADER_READ) {
            unsafe {
                cmd_buffer.pipeline_barrier(
                    PS::HOST .. PS::VERTEX_SHADER | PS::FRAGMENT_SHADER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
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

    pub fn bind_read_target(&mut self, read_target: ReadTarget) {
        let fbo_id = match read_target {
            ReadTarget::Default => DEFAULT_READ_FBO,
            ReadTarget::Texture { texture, layer } => texture.fbos[layer],
        };
        self.bind_read_target_impl(fbo_id)
    }

    #[cfg(feature = "capture")]
    fn bind_read_texture(&mut self, texture_id: TextureId, layer_id: i32) {
        self.bound_read_texture = (texture_id, layer_id);
    }

    fn bind_draw_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
        }
    }

    pub fn reset_read_target(&mut self) {
        self.bind_read_target_impl(DEFAULT_READ_FBO);
    }

    pub fn reset_draw_target(&mut self) {
        self.bind_draw_target_impl(DEFAULT_DRAW_FBO);
        self.depth_available = true;
    }

    pub fn bind_draw_target(&mut self, texture_target: DrawTarget) {
        let (fbo_id, dimensions, depth_available) = match texture_target {
            DrawTarget::Default(dim) => (DEFAULT_DRAW_FBO, dim, true),
            DrawTarget::Texture {
                texture,
                layer,
                with_depth,
            } => {
                texture.bound_in_frame.set(self.frame_id);
                let fbo_id = if with_depth {
                    texture.fbos_with_depth[layer]
                } else {
                    texture.fbos[layer]
                };

                let fbo = self.fbos.get_mut(&fbo_id).unwrap();
                fbo.layer_index = layer as u16;

                let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
                unsafe {
                    let mut src_stage = Some(PipelineStage::empty());
                    if let Some(barrier) = self.images[&texture.id].core.transit(
                        hal::image::Access::COLOR_ATTACHMENT_READ
                            | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                        hal::image::Layout::ColorAttachmentOptimal,
                        self.images[&texture.id].core.subresource_range.clone(),
                        src_stage.as_mut(),
                    ) {
                        cmd_buffer.pipeline_barrier(
                            src_stage.unwrap() .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                            hal::memory::Dependencies::empty(),
                            &[barrier],
                        );
                    }
                }

                (fbo_id, texture.get_dimensions(), with_depth)
            }
        };

        self.depth_available = depth_available;
        self.bind_draw_target_impl(fbo_id);
        self.viewport.rect = hal::pso::Rect {
            x: 0,
            y: 0,
            w: dimensions.width as _,
            h: dimensions.height as _,
        };
    }

    pub fn create_fbo_for_external_texture(&mut self, _texture_id: u32) -> FBOId {
        warn!("External texture creation is missing");
        FBOId(0)
    }

    pub fn create_fbo(&mut self) -> FBOId {
        DEBUG_READ_FBO
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

    fn generate_texture_id(&self) -> TextureId {
        let mut rng = rand::thread_rng();
        let mut texture_id = INVALID_TEXTURE_ID + 1;
        while self.images.contains_key(&texture_id) {
            texture_id = rng.gen_range::<u32>(INVALID_TEXTURE_ID + 1, u32::max_value());
        }
        texture_id
    }

    fn generate_program_id(&self) -> ProgramId {
        let mut rng = rand::thread_rng();
        let mut program_id = ProgramId(INVALID_PROGRAM_ID.0 + 1);
        while self.programs.contains_key(&program_id) {
            program_id =
                ProgramId(rng.gen_range::<u32>(INVALID_PROGRAM_ID.0 + 1, u32::max_value()));
        }
        program_id
    }

    fn generate_fbo_ids(&mut self, count: i32) -> Vec<FBOId> {
        let mut rng = rand::thread_rng();
        let mut fboids = vec![];
        let mut fbo_id = FBOId(DEFAULT_DRAW_FBO.0 + 1);
        for _ in 0 .. count {
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

    fn create_cache_buffer(
        &mut self,
        width: u64,
        height: u64,
    ) -> PMBuffer<B> {
        let buffer_stride = std::mem::size_of::<[f32; 4]>() as u64;
        let buffer_len = height * width * buffer_stride;

        assert_ne!(buffer_len, 0);
        let mut storage_buffer =
            unsafe { self.device.create_buffer(buffer_len, hal::buffer::Usage::STORAGE) }.unwrap();

        let buffer_req = unsafe { self.device.get_buffer_requirements(&storage_buffer) };
        let mut coherent = false;

        use hal::memory::Properties;
        let memory_types = self.adapter.physical_device.memory_properties().memory_types;
        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                buffer_req.type_mask & (1 << id) != 0
                    && mem_type.properties.contains(Properties::CPU_VISIBLE | Properties::COHERENT | Properties::CPU_CACHED)
            });

        let upload_type = match upload_type {
            Some(ty) => {
                info!("A proper memory type found for gpu cache buffer");
                coherent = true;
                ty
            },
            None => {
                warn!("Using non coherent memory for gpu cache buffer!");
                memory_types
                    .iter()
                    .enumerate()
                    .position(|(id, mem_type)| {
                        buffer_req.type_mask & (1 << id) != 0
                            && mem_type.properties.contains(Properties::CPU_VISIBLE | Properties::CPU_CACHED)
                    })
                    .unwrap()
            }
        };

        let buffer_memory = unsafe { self.device.allocate_memory(upload_type.into(), buffer_req.size) }.unwrap();

        unsafe { self.device.bind_buffer_memory(&buffer_memory, 0, &mut storage_buffer) }.unwrap();
        let ptr = unsafe { self.device.map_memory(&buffer_memory, 0..buffer_req.size) }.unwrap();
        PMBuffer {
            buffer: storage_buffer,
            memory: buffer_memory,
            ptr,
            coherent,
            height,
            size: buffer_req.size,
            state: Cell::new(hal::buffer::Access::empty()),
        }
    }

    pub fn create_writer(
        &mut self
    ) -> &mut [[f32; 4]] {
        unsafe {
            self.gpu_cache_buffers
                .get_mut(&self.bound_gpu_cache)
                .expect("Gpu cache not found!")
                .acquire_host_visible_slice(None)
        }
    }

    pub fn flush_mapped_ranges(
        &self,
        ranges: impl Iterator<Item = std::ops::Range<u64>>,
    ) {
        unsafe { self.gpu_cache_buffers[&self.bound_gpu_cache].flush_mapped_ranges(&self.device, ranges) };
    }

    pub fn create_gpu_cache_texture(
        &mut self,
        width: i32,
        height: i32,
    ) -> Texture {
        debug_assert!(self.inside_frame);
        assert!(!(width == 0 || height == 0));

        // Set up the texture book-keeping.
        let texture = Texture {
            id: self.generate_texture_id(),
            target: TextureTarget::Default as _,
            size: DeviceIntSize::new(width, height),
            layer_count: 1,
            format: ImageFormat::RGBAF32,
            filter: Default::default(),
            fbos: vec![],
            fbos_with_depth: vec![],
            last_frame_used: self.frame_id,
            bound_in_frame: Cell::new(GpuFrameId(0)),
            flags: TextureFlags::default(),
            is_buffer: true,
        };

        let buffer = self.create_cache_buffer(width as u64, height as u64);
        if let Some(barrier) = buffer.transit(hal::buffer::Access::HOST_WRITE | hal::buffer::Access::HOST_READ) {
            use hal::pso::PipelineStage as PS;
            let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
            unsafe {
                cmd_buffer.pipeline_barrier(
                    PS::VERTEX_SHADER | PS::FRAGMENT_SHADER .. PS::HOST,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
        self.gpu_cache_buffers.insert(texture.id, buffer);
        record_gpu_alloc(texture.size_in_bytes());

        texture
    }

    pub fn create_texture(
        &mut self,
        target: TextureTarget,
        format: ImageFormat,
        mut width: i32,
        mut height: i32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
    ) -> Texture {
        debug_assert!(self.inside_frame);
        assert!(!(width == 0 || height == 0 || layer_count == 0));

        if width > self.max_texture_size || height > self.max_texture_size {
            error!(
                "Attempting to allocate a texture of size {}x{} above the limit, trimming",
                width, height
            );
            width = width.min(self.max_texture_size);
            height = height.min(self.max_texture_size);
        }

        // Set up the texture book-keeping.
        let mut texture = Texture {
            id: self.generate_texture_id(),
            target: target as _,
            size: DeviceIntSize::new(width, height),
            layer_count,
            format,
            filter,
            fbos: vec![],
            fbos_with_depth: vec![],
            last_frame_used: self.frame_id,
            bound_in_frame: Cell::new(GpuFrameId(0)),
            flags: TextureFlags::default(),
            is_buffer: false,
        };

        assert!(!self.images.contains_key(&texture.id));
        let usage_base = hal::image::Usage::TRANSFER_SRC
            | hal::image::Usage::TRANSFER_DST
            | hal::image::Usage::SAMPLED;

        let view_kind = match target {
            TextureTarget::Array => hal::image::ViewKind::D2Array,
            _ => hal::image::ViewKind::D2,
        };

        let (mip_levels, usage) = match texture.filter {
            TextureFilter::Nearest => (
                1,
                usage_base | hal::image::Usage::COLOR_ATTACHMENT,
            ),
            TextureFilter::Linear => (
                1,
                usage_base | hal::image::Usage::COLOR_ATTACHMENT,
            ),
            TextureFilter::Trilinear => (
                (width as f32).max(height as f32).log2().floor() as u8 + 1,
                usage_base | hal::image::Usage::COLOR_ATTACHMENT,
            ),
        };
        let img = Image::new(
            &self.device,
            &mut self.heaps,
            texture.format,
            texture.size.width,
            texture.size.height,
            texture.layer_count,
            view_kind,
            mip_levels,
            usage,
        );

        unsafe {
            let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();

            if let Some(barrier) = img.core.transit(
                hal::image::Access::COLOR_ATTACHMENT_READ
                    | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                img.core.subresource_range.clone(),
                None,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT
                        .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }

        self.images.insert(texture.id, img);

        // Set up FBOs, if required.
        if let Some(rt_info) = render_target {
            self.init_fbos(&mut texture, false);
            if rt_info.has_depth {
                self.init_fbos(&mut texture, true);
            }
        }

        record_gpu_alloc(texture.size_in_bytes());

        texture
    }

    fn init_fbos(&mut self, texture: &mut Texture, with_depth: bool) {
        let new_fbos = self.generate_fbo_ids(texture.layer_count);
        let (rbo_id, depth) = if with_depth {
            let rbo_id = self.acquire_depth_target(texture.get_dimensions());
            (rbo_id, Some(&self.rbos[&rbo_id].core.view))
        } else {
            (RBOId(0), None)
        };

        for i in 0 .. texture.layer_count as u16 {
            let fbo = Framebuffer::new(
                &self.device,
                &texture,
                &self.images.get(&texture.id).unwrap(),
                i,
                self.render_pass.as_ref().unwrap(),
                rbo_id.clone(),
                depth,
            );
            self.fbos.insert(new_fbos[i as usize], fbo);

            if with_depth {
                texture.fbos_with_depth.push(new_fbos[i as usize])
            } else {
                texture.fbos.push(new_fbos[i as usize])
            }
        }
    }

    pub fn copy_cache_buffer(&mut self, dst: &Texture, src: &Texture) {
        let mut src_buffer = self.gpu_cache_buffers.remove(&src.id).unwrap();
        {
            let size = src_buffer.size;
            let dst_buffer = self.gpu_cache_buffers.get_mut(&dst.id).unwrap();
            unsafe {
                let src_slice = src_buffer.acquire_host_visible_slice(None);
                let dst_slice = dst_buffer.acquire_host_visible_slice(Some(size));
                dst_slice.copy_from_slice(src_slice);
            }
        }
        self.gpu_cache_buffers.insert(src.id, src_buffer);
    }

    /// Copies the contents from one renderable texture to another.
    pub fn blit_renderable_texture(&mut self, dst: &mut Texture, src: &Texture) {
        dst.bound_in_frame.set(self.frame_id);
        src.bound_in_frame.set(self.frame_id);
        debug_assert!(self.inside_frame);
        debug_assert!(dst.size.width >= src.size.width);
        debug_assert!(dst.size.height >= src.size.height);
        assert!(dst.layer_count >= src.layer_count);

        let rect = DeviceIntRect::new(DeviceIntPoint::zero(), src.get_dimensions().to_i32());
        let (src_img, layers) = (&self.images[&src.id].core, src.layer_count);
        let dst_img = &self.images[&dst.id].core;
        let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();

        let range = hal::image::SubresourceRange {
            aspects: hal::format::Aspects::COLOR,
            levels: 0 .. 1,
            layers: 0 .. layers as _,
        };

        unsafe {
            let mut src_stage = Some(PipelineStage::empty());
            if let Some(barrier) = src_img.transit(
                hal::image::Access::TRANSFER_READ,
                hal::image::Layout::TransferSrcOptimal,
                range.clone(),
                src_stage.as_mut(),
            ) {
                cmd_buffer.pipeline_barrier(
                    src_stage.unwrap() .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            if let Some(barrier) = dst_img.transit(
                hal::image::Access::TRANSFER_WRITE,
                hal::image::Layout::TransferDstOptimal,
                range.clone(),
                src_stage.as_mut(),
            ) {
                cmd_buffer.pipeline_barrier(
                    src_stage.unwrap() .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            cmd_buffer.copy_image(
                &src_img.image,
                hal::image::Layout::TransferSrcOptimal,
                &dst_img.image,
                hal::image::Layout::TransferDstOptimal,
                &[hal::command::ImageCopy {
                    src_subresource: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0 .. layers as _,
                    },
                    src_offset: hal::image::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    dst_subresource: hal::image::SubresourceLayers {
                        aspects: hal::format::Aspects::COLOR,
                        level: 0,
                        layers: 0 .. layers as _,
                    },
                    dst_offset: hal::image::Offset {
                        x: rect.origin.x as i32,
                        y: rect.origin.y as i32,
                        z: 0,
                    },
                    extent: hal::image::Extent {
                        width: rect.size.width as u32,
                        height: rect.size.height as u32,
                        depth: 1,
                    },
                }],
            );

            // the blit caller code expects to be able to render to the target
            let barriers = src_img
                .transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::ColorAttachmentOptimal,
                    range.clone(),
                    None,
                )
                .into_iter()
                .chain(
                    dst_img.transit(
                        hal::image::Access::COLOR_ATTACHMENT_READ
                            | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                        hal::image::Layout::ColorAttachmentOptimal,
                        range.clone(),
                        None,
                    )
                );

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                hal::memory::Dependencies::empty(),
                barriers,
            );

        }
    }

    fn generate_mipmaps(&mut self, texture: &Texture) {
        texture.bound_in_frame.set(self.frame_id);
        let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();

        let image = self
            .images
            .get_mut(&texture.id)
            .expect("Texture not found.");

        let mut mip_width = texture.size.width;
        let mut mip_height = texture.size.height;

        let mut half_mip_width = mip_width / 2;
        let mut half_mip_height = mip_height / 2;

        unsafe {
            let mut src_stage = Some(PipelineStage::empty());
            if let Some(barrier) = image.core.transit(
                hal::image::Access::TRANSFER_WRITE,
                hal::image::Layout::TransferDstOptimal,
                image.core.subresource_range.clone(),
                src_stage.as_mut(),
            ) {
                cmd_buffer.pipeline_barrier(
                    src_stage.unwrap() .. PipelineStage::TRANSFER,
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
                    None,
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
                    &[hal::command::ImageBlit {
                        src_subresource: hal::image::SubresourceLayers {
                            aspects: hal::format::Aspects::COLOR,
                            level: index - 1,
                            layers: 0 .. 1,
                        },
                        src_bounds: hal::image::Offset { x: 0, y: 0, z: 0 } .. hal::image::Offset {
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
                    }],
                );
                if let Some(barrier) = image.core.transit(
                    hal::image::Access::TRANSFER_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: index - 1 .. index,
                        layers: 0 .. 1,
                    },
                    None,
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
                hal::image::Access::COLOR_ATTACHMENT_READ
                    | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                hal::image::Layout::ColorAttachmentOptimal,
                image.core.subresource_range.clone(),
                None,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
    }

    fn acquire_depth_target(&mut self, dimensions: DeviceIntSize) -> RBOId {
        if let Entry::Occupied(mut o) = self.depth_targets.entry(dimensions) {
            o.get_mut().refcount += 1;
            return o.get().rbo_id
        }

        let rbo_id = self.generate_rbo_id();
        let rbo = DepthBuffer::new(
            &self.device,
            &mut self.heaps,
            dimensions.width as _,
            dimensions.height as _,
            self.depth_format,
        );
        self.rbos.insert(rbo_id, rbo);
        let target = SharedDepthTarget {
            rbo_id,
            refcount: 1,
        };
        record_gpu_alloc(depth_target_size_in_bytes(&dimensions));
        self.depth_targets.insert(dimensions, target);
        rbo_id
    }

    fn release_depth_target(&mut self, dimensions: DeviceIntSize) {
        let mut entry = match self.depth_targets.entry(dimensions) {
            Entry::Occupied(x) => x,
            Entry::Vacant(..) => panic!("Releasing unknown depth target"),
        };
        debug_assert!(entry.get().refcount != 0);
        entry.get_mut().refcount -= 1;
        if entry.get().refcount == 0 {
            let t = entry.remove();
            let old_rbo = self.rbos.remove(&t.rbo_id).unwrap();
            old_rbo.deinit(&self.device, &mut self.heaps);
            record_gpu_free(depth_target_size_in_bytes(&dimensions));
        }
    }

    pub fn blit_render_target(&mut self, src_rect: DeviceIntRect, dest_rect: DeviceIntRect) {
        debug_assert!(self.inside_frame);

        let (src_format, src_img, src_layer) = if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture_id];
            let layer = fbo.layer_index;
            (img.format, &img.core, layer)
        } else {
            (
                self.surface_format,
                &self.frame_images[self.current_frame_id],
                0,
            )
        };

        let (dest_format, dest_img, dest_layer) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture_id];
            let layer = fbo.layer_index;
            (img.format, &img.core, layer)
        } else {
            (
                self.surface_format,
                &self.frame_images[self.current_frame_id],
                0,
            )
        };

        let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();

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

        unsafe {
            let src_begin_state = src_img.state.get();
            let mut pre_src_stage = Some(PipelineStage::empty());
            if let Some(barrier) = src_img.transit(
                hal::image::Access::TRANSFER_READ,
                hal::image::Layout::TransferSrcOptimal,
                src_range.clone(),
                pre_src_stage.as_mut(),
            ) {
                cmd_buffer.pipeline_barrier(
                    pre_src_stage.unwrap() .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            let dest_begin_state = dest_img.state.get();
            let mut pre_dest_stage = Some(PipelineStage::empty());
            if let Some(barrier) = dest_img.transit(
                hal::image::Access::TRANSFER_WRITE,
                hal::image::Layout::TransferDstOptimal,
                dest_range.clone(),
                pre_dest_stage.as_mut(),
            ) {
                cmd_buffer.pipeline_barrier(
                    pre_dest_stage.unwrap() .. PipelineStage::TRANSFER,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            if src_rect.size != dest_rect.size || src_format != dest_format {
                cmd_buffer.blit_image(
                    &src_img.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &dest_img.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::image::Filter::Linear,
                    &[hal::command::ImageBlit {
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
                    }],
                );
            } else {
                cmd_buffer.copy_image(
                    &src_img.image,
                    hal::image::Layout::TransferSrcOptimal,
                    &dest_img.image,
                    hal::image::Layout::TransferDstOptimal,
                    &[hal::command::ImageCopy {
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
                    }],
                );
            }

            // the blit caller code expects to be able to render to the target
            if let Some(barrier) = src_img.transit(
                src_begin_state.0,
                src_begin_state.1,
                src_range,
                None,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. pre_src_stage.unwrap(),
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }

            if let Some(barrier) = dest_img.transit(
                dest_begin_state.0,
                dest_begin_state.1,
                dest_range,
                None,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. pre_dest_stage.unwrap(),
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
        }
    }

    /// Performs a blit while flipping vertically. Useful for blitting textures
    /// (which use origin-bottom-left) to the main framebuffer (which uses
    /// origin-top-left).
    pub fn blit_render_target_invert_y(
        &mut self,
        src_rect: DeviceIntRect,
        dest_rect: DeviceIntRect,
    ) {
        debug_assert!(self.inside_frame);
        self.blit_render_target(src_rect, dest_rect);
    }

    /// Notifies the device that the contents of a render target are no longer
    /// needed.
    ///
    /// FIXME(bholley): We could/should invalidate the depth targets earlier
    /// than the color targets, i.e. immediately after each pass.
    pub fn invalidate_render_target(&mut self, _texture: &Texture) {
        warn!("invalidate_render_target not implemented!");
    }

    /// Notifies the device that a render target is about to be reused.
    ///
    /// This method adds or removes a depth target as necessary.
    pub fn reuse_render_target<T: Texel>(
        &mut self,
        texture: &mut Texture,
        rt_info: RenderTargetInfo,
    ) {
        texture.last_frame_used = self.frame_id;

        // Add depth support if needed.
        if rt_info.has_depth && !texture.supports_depth() {
            self.init_fbos(texture, true);
        }
    }

    fn free_texture(&mut self, mut texture: Texture) {
        if texture.bound_in_frame.get() == self.frame_id {
            self.retained_textures.push(texture);
            return;
        }

        if texture.still_in_flight(self.frame_id, self.frame_count) {
            self.wait_for_resources();
        }

        if texture.supports_depth() {
            self.release_depth_target(texture.get_dimensions());
        }

        if !texture.fbos_with_depth.is_empty() {
            for old in texture.fbos_with_depth.drain(..) {
                debug_assert!(self.bound_draw_fbo != old || self.bound_read_fbo != old);
                let old_fbo = self.fbos.remove(&old).unwrap();
                old_fbo.deinit(&self.device);
            }
        }

        if !texture.fbos.is_empty() {
            for old in texture.fbos.drain(..) {
                debug_assert!(self.bound_draw_fbo != old || self.bound_read_fbo != old);
                let old_fbo = self.fbos.remove(&old).unwrap();
                old_fbo.deinit(&self.device);
            }
        }

        self.per_draw_descriptors.retain(&texture.id);
        self.per_pass_descriptors.retain(&texture.id);
        self.per_group_descriptors.retain(&texture.id);


        if texture.is_buffer {
            let buffer = self.gpu_cache_buffers.remove(&texture.id).expect("Texture not found.");
            buffer.deinit(&self.device);
        } else {
            let image = self.images.remove(&texture.id).expect("Texture not found.");
            image.deinit(&self.device, &mut self.heaps);
        }
        record_gpu_free(texture.size_in_bytes());
        texture.id = 0;
    }

    fn delete_retained_textures(&mut self) {
        let textures: Vec<_> = self.retained_textures.drain(..).collect();
        for mut texture in textures {
            self.free_texture(texture);
        }
    }

    pub fn delete_texture(&mut self, texture: Texture) {
        //debug_assert!(self.inside_frame);
        if texture.size.width + texture.size.height == 0 {
            return;
        }

        self.free_texture(texture);
    }

    pub fn retain_cache_buffer(&mut self, texture: Texture) {
        self.retained_textures.push(texture);
    }

    #[cfg(feature = "replay")]
    pub fn delete_external_texture(&mut self, mut external: ExternalTexture) {
        warn!("delete external texture is missing");
        external.id = 0;
    }

    pub fn switch_mode(&mut self, mode: i32) {
        debug_assert!(self.inside_frame);
        self.bound_locals.uMode = mode;
    }

    pub fn create_pbo(&mut self) -> PBO {
        PBO {}
    }

    pub fn delete_pbo(&mut self, _pbo: PBO) {}

    pub fn upload_texture<'a>(
        &'a mut self,
        texture: &'a Texture,
        _pbo: &PBO,
        _upload_count: usize,
    ) -> TextureUploader<'a, B> {
        debug_assert!(self.inside_frame);

        match self.upload_method {
            UploadMethod::Immediate => unimplemented!(),
            UploadMethod::PixelBuffer(..) => TextureUploader {
                device: self,
                texture,
            },
        }
    }

    pub fn upload_texture_immediate<T: Texel>(&mut self, texture: &Texture, pixels: &[T]) {
        texture.bound_in_frame.set(self.frame_id);
        let len = pixels.len() / texture.layer_count as usize;
        for i in 0 .. texture.layer_count {
            let start = len * i as usize;

            self.images
                .get_mut(&texture.id)
                .expect("Texture not found.")
                .update(
                    &self.device,
                    self.command_pool[self.next_id].command_buffer_mut(),
                    &mut self.staging_buffer_pool[self.next_id],
                    DeviceIntRect::new(DeviceIntPoint::new(0, 0), texture.size),
                    i,
                    texels_to_u8_slice(&pixels[start .. (start + len)]),
                );
        }
        if texture.filter == TextureFilter::Trilinear {
            self.generate_mipmaps(texture);
        }
    }

    #[cfg(feature = "capture")]
    pub fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8> {
        let mut pixels = vec![0; (img_desc.size.width * img_desc.size.height * 4) as usize];
        self.read_pixels_into(
            DeviceIntRect::new(
                DeviceIntPoint::zero(),
                DeviceIntSize::new(img_desc.size.width, img_desc.size.height),
            ),
            ReadPixelsFormat::Rgba8,
            &mut pixels,
        );
        pixels
    }

    /// Read rectangle of pixels into the specified output slice.
    pub fn read_pixels_into(
        &mut self,
        rect: DeviceIntRect,
        read_format: ReadPixelsFormat,
        output: &mut [u8],
    ) {
        self.wait_for_resources();

        let bytes_per_pixel = match read_format {
            ReadPixelsFormat::Standard(imf) => imf.bytes_per_pixel(),
            ReadPixelsFormat::Rgba8 => 4,
        };
        let size_in_bytes = (bytes_per_pixel * rect.size.width * rect.size.height) as usize;
        assert_eq!(output.len(), size_in_bytes);
        let capture_read =
            cfg!(feature = "capture") && self.bound_read_texture.0 != INVALID_TEXTURE_ID;

        let (image, image_format, layer) = if capture_read {
            let img = &self.images[&self.bound_read_texture.0];
            (&img.core, img.format, self.bound_read_texture.1 as u16)
        } else if self.bound_read_fbo != DEFAULT_READ_FBO {
            let fbo = &self.fbos[&self.bound_read_fbo];
            let img = &self.images[&fbo.texture_id];
            let layer = fbo.layer_index;
            (&img.core, img.format, layer)
        } else {
            (
                &self.frame_images[self.current_frame_id],
                self.surface_format,
                0,
            )
        };

        let (fmt_mismatch, stride) = if bytes_per_pixel < image_format.bytes_per_pixel() {
            // Special case which can occur during png save, because we force to read Rgba8 values from an Rgbaf32 texture.
            (
                true,
                (image_format.bytes_per_pixel() / bytes_per_pixel) as usize,
            )
        } else {
            assert_eq!(bytes_per_pixel, image_format.bytes_per_pixel());
            (false, 1)
        };

        assert!(output.len() <= DOWNLOAD_BUFFER_SIZE, "output len {:?} buffer size {:?}", output.len(), DOWNLOAD_BUFFER_SIZE);
        if self.download_buffer.is_none() {
            self.download_buffer = Some(Buffer::new(
                &self.device,
                &mut self.heaps,
                MemoryUsageValue::Download,
                hal::buffer::Usage::TRANSFER_DST,
                (self.limits.optimal_buffer_copy_pitch_alignment - 1) as usize,
                DOWNLOAD_BUFFER_SIZE / stride,
                stride,
            ));
        }
        let download_buffer = self.download_buffer.as_mut().unwrap();
        let mut command_pool = unsafe {
            self.device.create_command_pool(
                self.queue_group_family,
                hal::pool::CommandPoolCreateFlags::empty(),
            )
        }
        .expect("create_command_pool failed");
        unsafe { command_pool.reset() };

        let mut cmd_buffer = command_pool.allocate_one(RawLevel::Primary);
        unsafe {
            Self::begin_cmd_buffer(&mut cmd_buffer);

            let range = hal::image::SubresourceRange {
                aspects: hal::format::Aspects::COLOR,
                levels: 0 .. 1,
                layers: layer .. layer + 1,
            };

            let barriers = download_buffer
                .transit(hal::buffer::Access::TRANSFER_WRITE)
                .into_iter()
                .chain(image.transit(
                    hal::image::Access::TRANSFER_READ,
                    hal::image::Layout::TransferSrcOptimal,
                    range.clone(),
                    None,
                ));

            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER .. PipelineStage::TRANSFER,
                hal::memory::Dependencies::empty(),
                barriers,
            );

            cmd_buffer.copy_image_to_buffer(
                &image.image,
                hal::image::Layout::TransferSrcOptimal,
                &download_buffer.buffer,
                &[hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: rect.size.width as u32,
                    buffer_height: rect.size.height as u32,
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
                }],
            );
            if let Some(barrier) = image.transit(
                hal::image::Access::empty(),
                hal::image::Layout::ColorAttachmentOptimal,
                range,
                None,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::TRANSFER .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            cmd_buffer.finish();
        }

        let mut copy_fence = self
            .device
            .create_fence(false)
            .expect("create_fence failed");

        unsafe {
            self.device
                .reset_fence(&copy_fence)
                .expect("reset_fence failed");
            let submission = hal::queue::Submission {
                command_buffers: Some(&cmd_buffer),
                wait_semaphores: None,
                signal_semaphores: None,
            };
            self.queue_group_queues[0].submit::<_, _, B::Semaphore, _, _>(
                submission,
                Some(&mut copy_fence),
            );
            self.device
                .wait_for_fence(&copy_fence, !0)
                .expect("wait_for_fence failed");
            self.device.destroy_fence(copy_fence);
        }

        let mut data = vec![0; download_buffer.buffer_size];
        let range = 0 .. download_buffer.buffer_size as u64;
        if fmt_mismatch {
            let mut f32_data = vec![0f32; download_buffer.buffer_size];
            unsafe {
                let mut mapped = download_buffer
                    .memory_block
                    .map(&self.device, range.clone())
                    .expect("Mapping memory block failed");
                let slice = mapped.read(&self.device, range).expect("Read failed");
                f32_data[0 .. slice.len()].copy_from_slice(&slice);
            }
            download_buffer.memory_block.unmap(&self.device);
            for i in 0 .. f32_data.len() {
                data[i] = round_to_int(f32_data[i].min(0f32).max(1f32));
            }
        } else {
            unsafe {
                let mut mapped = download_buffer
                    .memory_block
                    .map(&self.device, range.clone())
                    .expect("Mapping memory block failed");
                let slice = mapped.read(&self.device, range).expect("Read failed");
                data[0 .. slice.len()].copy_from_slice(&slice);
            }
            download_buffer.memory_block.unmap(&self.device);
        }
        data.truncate(output.len());
        if !capture_read && self.surface_format == ImageFormat::BGRA8 {
            let width = rect.size.width as usize;
            let height = rect.size.height as usize;
            let row_pitch: usize = bytes_per_pixel as usize * width;
            // Vertical flip the result and convert to RGBA
            for y in 0 .. height as usize {
                for x in 0 .. width as usize {
                    let offset: usize = y * row_pitch + x * 4;
                    let rev_offset: usize = (height - 1 - y) * row_pitch + x * 4;
                    output[offset + 0] = data[rev_offset + 2];
                    output[offset + 1] = data[rev_offset + 1];
                    output[offset + 2] = data[rev_offset + 0];
                    output[offset + 3] = data[rev_offset + 3];
                }
            }
        } else {
            output.swap_with_slice(&mut data);
        }

        unsafe {
            command_pool.reset();
            self.device.destroy_command_pool(command_pool);
        }
    }

    /// Get texels of a texture into the specified output slice.
    pub fn get_tex_image_into(
        &mut self,
        _texture: &Texture,
        _format: ImageFormat,
        _output: &mut [u8],
    ) {
        unimplemented!();
    }

    /// Attaches the provided texture to the current Read FBO binding.
    #[cfg(feature = "capture")]
    fn attach_read_texture_raw(&mut self, texture_id: u32, _target: TextureTarget, layer_id: i32) {
        self.bind_read_texture(texture_id, layer_id);
    }

    #[cfg(feature = "capture")]
    pub fn attach_read_texture_external(
        &mut self,
        _texture_id: u32,
        _target: TextureTarget,
        _layer_id: i32,
    ) {
        unimplemented!();
    }

    #[cfg(feature = "capture")]
    pub fn attach_read_texture(&mut self, texture: &Texture, layer_id: i32) {
        self.attach_read_texture_raw(texture.id, texture.target.into(), layer_id)
    }

    pub fn invalidate_read_texture(&mut self) {
        self.bound_read_texture = (INVALID_TEXTURE_ID, 0);
    }

    pub fn bind_vao(&mut self, _vao: &VAO) {}

    pub fn create_vao(&mut self, _descriptor: &VertexDescriptor) -> VAO {
        VAO {}
    }

    pub fn delete_vao(&mut self, _vao: VAO) {}

    pub fn create_vao_with_new_instances(
        &mut self,
        _descriptor: &VertexDescriptor,
        _base_vao: &VAO,
    ) -> VAO {
        VAO {}
    }

    pub fn update_vao_main_vertices<V: Copy + PrimitiveType>(
        &mut self,
        _vao: &VAO,
        _vertices: &[V],
        _usage_hint: VertexUsageHint,
    ) {
        if self.bound_program != INVALID_PROGRAM_ID {
            self.update_vertices(
                &_vertices
                    .iter()
                    .map(|v| v.to_primitive_type())
                    .collect::<Vec<_>>(),
            );
        }
    }

    pub fn update_vao_instances<V: PrimitiveType>(
        &mut self,
        _vao: &VAO,
        instances: &[V],
        _usage_hint: VertexUsageHint,
    ) {
        let data = instances
            .iter()
            .map(|pi| pi.to_primitive_type())
            .collect::<Vec<V::Primitive>>();
        self.update_instances(&data);
    }

    pub fn update_vao_indices<I: Copy>(
        &mut self,
        _vao: &VAO,
        _indices: &[I],
        _usage_hint: VertexUsageHint,
    ) {
        if self.bound_program != INVALID_PROGRAM_ID {
            self.update_indices(_indices);
        }
    }

    pub fn draw_triangles_u16(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn draw_triangles_u32(&mut self, _first_vertex: i32, _index_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn draw_nonindexed_lines(&mut self, _first_vertex: i32, _vertex_count: i32) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn draw_indexed_triangles_instanced_u16(
        &mut self,
        _index_count: i32,
        _instance_count: i32,
    ) {
        debug_assert!(self.inside_frame);
        self.draw();
    }

    pub fn end_frame(&mut self) {
        self.reset_draw_target();
        self.reset_read_target();

        debug_assert!(self.inside_frame);
        self.inside_frame = false;

        self.frame_id.0 += 1;
    }

    fn clear_target_rect(
        &mut self,
        rect: DeviceIntRect,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
    ) {
        if color.is_none() && depth.is_none() {
            return;
        }

        let rect = hal::pso::ClearRect {
            rect: hal::pso::Rect {
                x: rect.origin.x as i16,
                y: rect.origin.y as i16,
                w: rect.size.width as i16,
                h: rect.size.height as i16,
            },
            layers: 0 .. 1,
        };

        let color_clear = color.map(|c| hal::command::AttachmentClear::Color {
            index: 0,
            value: hal::command::ClearColor::Float(c),
        });

        let depth_clear = depth.map(|d| hal::command::AttachmentClear::DepthStencil {
            depth: Some(d),
            stencil: None,
        });

        let (img, frame_buffer, format, depth_img) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let texture_id = self.fbos[&self.bound_draw_fbo].texture_id;
            let rbo_id = self.fbos[&self.bound_draw_fbo].rbo;
            (
                &self.images[&texture_id].core,
                &self.fbos[&self.bound_draw_fbo].fbo,
                self.fbos[&self.bound_draw_fbo].format,
                if rbo_id == RBOId(0) {
                    None
                } else {
                    Some(&self.rbos[&rbo_id].core)
                },
            )
        } else {
            let (frame_buffer, depth_image) = match self.current_depth_test {
                DepthTest::Off => (&self.framebuffers[self.current_frame_id], None),
                _ => (&self.framebuffers_depth[self.current_frame_id], Some(&self.frame_depths[self.current_frame_id].core)),
            };
            (
                &self.frame_images[self.current_frame_id],
                frame_buffer,
                self.surface_format,
                depth_image,
            )
        };

        let render_pass = self
            .render_pass
            .as_ref()
            .unwrap()
            .get_render_pass(format, depth_img.is_some());

        let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
        unsafe {
            let before_state = img.state.get();
            let mut before_depth_state = None;
            let mut pre_stage = Some(PipelineStage::empty());
            let mut pre_depth_stage = Some(PipelineStage::empty());
            if let Some(barrier) = img.transit(
                hal::image::Access::empty(),
                hal::image::Layout::ColorAttachmentOptimal,
                img.subresource_range.clone(),
                pre_stage.as_mut(),
            ) {
                cmd_buffer.pipeline_barrier(
                    pre_stage.unwrap() .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            if let Some(depth_img) = depth_img {
                before_depth_state = Some(depth_img.state.get());
                if let Some(barrier) = depth_img.transit(
                    hal::image::Access::empty(),
                    hal::image::Layout::DepthStencilAttachmentOptimal,
                    depth_img.subresource_range.clone(),
                    pre_depth_stage.as_mut(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        pre_depth_stage.unwrap() .. PipelineStage::EARLY_FRAGMENT_TESTS,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }

            cmd_buffer.begin_render_pass(
                render_pass,
                frame_buffer,
                self.viewport.rect,
                &[],
                hal::command::SubpassContents::Inline,
            );
            cmd_buffer.clear_attachments(color_clear.into_iter().chain(depth_clear), Some(rect));
            cmd_buffer.end_render_pass();

            if let Some(barrier) = img.transit(
                before_state.0,
                before_state.1,
                img.subresource_range.clone(),
                None,
            ) {
                cmd_buffer.pipeline_barrier(
                    PipelineStage::COLOR_ATTACHMENT_OUTPUT .. pre_stage.unwrap(),
                    hal::memory::Dependencies::empty(),
                    &[barrier],
                );
            }
            if let Some(depth_img) = depth_img {
                if let Some(barrier) = depth_img.transit(
                    before_depth_state.unwrap().0,
                    hal::image::Layout::DepthStencilAttachmentOptimal,
                    depth_img.subresource_range.clone(),
                    None,
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::EARLY_FRAGMENT_TESTS .. pre_depth_stage.unwrap(),
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
            }
        }
    }

    fn clear_target_image(&mut self, color: Option<[f32; 4]>, depth: Option<f32>) {
        let (img, layer, dimg) = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
            let fbo = &self.fbos[&self.bound_draw_fbo];
            let img = &self.images[&fbo.texture_id];
            let dimg = if depth.is_some() {
                Some(&self.rbos[&fbo.rbo].core)
            } else {
                None
            };
            (&img.core, fbo.layer_index, dimg)
        } else {
            (
                &self.frame_images[self.current_frame_id],
                0,
                Some(&self.frame_depths[self.current_frame_id].core),
            )
        };

        //Note: this function is assumed to be called within an active FBO
        // thus, we bring back the targets into renderable state
        let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
        unsafe {
            if let Some(color) = color {
                let mut src_stage = Some(PipelineStage::empty());
                if let Some(barrier) = img.transit(
                    hal::image::Access::COLOR_ATTACHMENT_READ
                        | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    img.subresource_range.clone(),
                    src_stage.as_mut(),
                ) {
                    cmd_buffer.pipeline_barrier(
                        src_stage.unwrap() .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                cmd_buffer.clear_image(
                    &img.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::command::ClearColorRaw { float32: [color[0], color[1], color[2], color[3]] },
                    hal::command::ClearDepthStencilRaw {
                        depth: 0.0,
                        stencil: 0,
                    },
                    Some(hal::image::SubresourceRange {
                        aspects: hal::format::Aspects::COLOR,
                        levels: 0 .. 1,
                        layers: layer .. layer + 1,
                    }),
                );
            }

            if let (Some(depth), Some(dimg)) = (depth, dimg) {
                assert_ne!(self.current_depth_test, DepthTest::Off);
                if let Some(barrier) = dimg.transit(
                    hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    hal::image::Layout::TransferDstOptimal,
                    dimg.subresource_range.clone(),
                    None,
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::EARLY_FRAGMENT_TESTS .. PipelineStage::LATE_FRAGMENT_TESTS,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                cmd_buffer.clear_image(
                    &dimg.image,
                    hal::image::Layout::TransferDstOptimal,
                    hal::command::ClearColorRaw { float32: [0.0; 4] },
                    hal::command::ClearDepthStencilRaw { depth, stencil: 0 },
                    Some(dimg.subresource_range.clone()),
                );
            }
        }
    }

    pub fn clear_target(
        &mut self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<DeviceIntRect>,
    ) {
        if let Some(rect) = rect {
            let target_rect = if self.bound_draw_fbo != DEFAULT_DRAW_FBO {
                let extent = &self.images[&self.fbos[&self.bound_draw_fbo].texture_id]
                    .kind
                    .extent();
                DeviceIntRect::new(
                    DeviceIntPoint::zero(),
                    DeviceIntSize::new(extent.width as _, extent.height as _),
                )
            } else {
                DeviceIntRect::new(
                    DeviceIntPoint::zero(),
                    DeviceIntSize::new(self.viewport.rect.w as _, self.viewport.rect.h as _),
                )
            };
            if rect.size.width > target_rect.size.width
                || rect.size.height > target_rect.size.height
            {
                // This can happen, when we resize
                self.clear_target_image(color, depth);
            } else if rect == target_rect {
                self.clear_target_image(color, depth);
            } else {
                self.clear_target_rect(rect, color, depth);
            }
        } else {
            self.clear_target_image(color, depth);
        }
    }

    pub fn enable_depth(&mut self) {
        assert!(
            self.depth_available,
            "Enabling depth test without depth target"
        );
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
        assert!(
            self.depth_available,
            "Enabling depth test without depth target"
        );
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

    pub fn set_blend(&self, enable: bool) {
        if !enable {
            self.current_blend_state.set(BlendState::Off)
        }
    }

    pub fn set_blend_mode_alpha(&self) {
        self.current_blend_state.set(ALPHA);
    }

    pub fn set_blend_mode_premultiplied_alpha(&self) {
        self.current_blend_state
            .set(BlendState::PREMULTIPLIED_ALPHA);
    }

    pub fn set_blend_mode_premultiplied_dest_out(&self) {
        self.current_blend_state.set(PREMULTIPLIED_DEST_OUT);
    }

    pub fn set_blend_mode_multiply(&self) {
        self.current_blend_state.set(BlendState::MULTIPLY);
    }

    pub fn set_blend_mode_max(&self) {
        self.current_blend_state.set(MAX);
    }

    pub fn set_blend_mode_min(&self) {
        self.current_blend_state.set(MIN);
    }

    pub fn set_blend_mode_subpixel_pass0(&self) {
        self.current_blend_state.set(SUBPIXEL_PASS0);
    }

    pub fn set_blend_mode_subpixel_pass1(&self) {
        self.current_blend_state.set(SUBPIXEL_PASS1);
    }

    pub fn set_blend_mode_subpixel_with_bg_color_pass0(&self) {
        self.current_blend_state.set(SUBPIXEL_WITH_BG_COLOR_PASS0);
    }

    pub fn set_blend_mode_subpixel_with_bg_color_pass1(&self) {
        self.current_blend_state.set(SUBPIXEL_WITH_BG_COLOR_PASS1);
    }

    pub fn set_blend_mode_subpixel_with_bg_color_pass2(&self) {
        self.current_blend_state.set(SUBPIXEL_WITH_BG_COLOR_PASS2);
    }

    pub fn set_blend_mode_subpixel_constant_text_color(&self, color: ColorF) {
        self.current_blend_state.set(SUBPIXEL_CONSTANT_TEXT_COLOR);
        // color is an unpremultiplied color.
        self.blend_color
            .set(ColorF::new(color.r, color.g, color.b, 1.0));
    }

    pub fn set_blend_mode_subpixel_dual_source(&self) {
        self.current_blend_state.set(SUBPIXEL_DUAL_SOURCE);
    }

    pub fn set_blend_mode_show_overdraw(&self) {
        self.current_blend_state.set(OVERDRAW);
    }

    pub fn supports_features(&self, features: hal::Features) -> bool {
        self.features.contains(features)
    }

    pub fn echo_driver_messages(&self) {
        warn!("echo_driver_messages is unimplemeneted");
    }

    pub fn set_next_frame_id(&mut self) {
        if self.wait_for_resize {
            self.device.wait_idle().unwrap();
            return;
        }
        unsafe {
            match self.swap_chain.as_mut() {
                Some(swap_chain) => {
                    match swap_chain.acquire_image(
                        !0,
                        Some(&mut self.image_available_semaphore),
                        None,
                    ) {
                        Ok((id, _)) => {
                            self.current_frame_id = id as _;
                            let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
                            let image = &self.frame_images[self.current_frame_id];
                            if let Some(barrier) = image.transit(
                                hal::image::Access::COLOR_ATTACHMENT_READ
                                    | hal::image::Access::COLOR_ATTACHMENT_WRITE,
                                hal::image::Layout::ColorAttachmentOptimal,
                                image.subresource_range.clone(),
                                None,
                            ) {
                                cmd_buffer.pipeline_barrier(
                                    PipelineStage::COLOR_ATTACHMENT_OUTPUT
                                        .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                                    hal::memory::Dependencies::empty(),
                                    &[barrier],
                                );
                            }
                            let depth_image = &self.frame_depths[self.current_frame_id].core;
                            if let Some(barrier) = depth_image.transit(
                                hal::image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                                    | hal::image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE,
                                hal::image::Layout::DepthStencilAttachmentOptimal,
                                depth_image.subresource_range.clone(),
                                None,
                            ) {
                                cmd_buffer.pipeline_barrier(
                                    PipelineStage::EARLY_FRAGMENT_TESTS
                                        .. PipelineStage::LATE_FRAGMENT_TESTS,
                                    hal::memory::Dependencies::empty(),
                                    &[barrier],
                                );
                            }
                        }
                        Err(acq_err) => {
                            match acq_err {
                                AcquireError::OutOfDate => warn!("AcquireError : OutOfDate"),
                                AcquireError::SurfaceLost(surf) => warn!("AcquireError : SurfaceLost => {:?}", surf),
                                AcquireError::NotReady => warn!("AcquireError : NotReady"),
                                AcquireError::DeviceLost(dev) => warn!("AcquireError : DeviceLost => {:?}", dev),
                                AcquireError::OutOfMemory(mem) => warn!("AcquireError : OutOfMemory => {:?}", mem),
                            }
                            self.wait_for_resize = true;
                        },
                    }
                }
                None => {
                    self.current_frame_id = (self.current_frame_id + 1) % self.frame_count;
                }
            }
        }
    }

    pub fn submit_to_gpu(&mut self) {
        if self.wait_for_resize {
            self.device.wait_idle().unwrap();
            self.reset_next_frame_resources();
            return;
        }
        {
            let cmd_buffer = self.command_pool[self.next_id].command_buffer_mut();
            let image = &self.frame_images[self.current_frame_id];
            unsafe {
                if let Some(barrier) = image.transit(
                    hal::image::Access::empty(),
                    hal::image::Layout::Present,
                    image.subresource_range.clone(),
                    None,
                ) {
                    cmd_buffer.pipeline_barrier(
                        PipelineStage::COLOR_ATTACHMENT_OUTPUT
                            .. PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                        hal::memory::Dependencies::empty(),
                        &[barrier],
                    );
                }
                cmd_buffer.finish();
            }
        }
        unsafe {
            match self.swap_chain.as_mut() {
                Some(swap_chain) => {
                    let submission = hal::queue::Submission {
                        command_buffers: self.command_pool[self.next_id].command_buffers(),
                        wait_semaphores: Some((
                            &self.image_available_semaphore,
                            PipelineStage::BOTTOM_OF_PIPE,
                        )),
                        signal_semaphores: Some(&self.render_finished_semaphore),
                    };
                    self.queue_group_queues[0]
                        .submit(submission, Some(&mut self.frame_fence[self.next_id].inner));
                    self.frame_fence[self.next_id].is_submitted = true;

                    // present frame
                    match self.queue_group_queues[0]
                        .present(
                            std::iter::once((&swap_chain, self.current_frame_id as _)),
                            Some(&self.render_finished_semaphore),
                        ) {
                            Ok(suboptimal) => {
                                match suboptimal {
                                    Some(_) => {
                                        warn!("Suboptimal: The swapchain no longer matches the surface");
                                        self.wait_for_resize = true;
                                    },
                                    None => {}
                                }
                            }
                            Err(presenterr) => {
                                match presenterr {
                                    PresentError::OutOfDate => warn!("PresentError : OutOfDate"),
                                    PresentError::SurfaceLost(surf) => warn!("PresentError : SurfaceLost => {:?}", surf),
                                    PresentError::DeviceLost(dev) => warn!("PresentError : DeviceLost => {:?}", dev),
                                    PresentError::OutOfMemory(mem) => warn!("PresentError : OutOfMemory => {:?}", mem),
                                }
                                self.wait_for_resize = true;
                            }
                        }
                }
                None => {
                    let submission = hal::queue::Submission {
                        command_buffers: self.command_pool[self.next_id].command_buffers(),
                        wait_semaphores: None,
                        signal_semaphores: None,
                    };
                    self.queue_group_queues[0].submit::<_, _, B::Semaphore, _, _>(
                        submission,
                        Some(&mut self.frame_fence[self.next_id].inner),
                    );
                    self.frame_fence[self.next_id].is_submitted = true;
                }
            }
        };
        self.reset_next_frame_resources();
    }

    pub fn wait_for_resources_and_reset(&mut self) {
        self.wait_for_resources();
        self.reset_command_pools();
    }

    fn wait_for_resources(&mut self) {
        for fence in &mut self.frame_fence {
            if fence.is_submitted {
                unsafe { self.device.wait_for_fence(&fence.inner, !0) }
                    .expect("wait_for_fence failed");
                unsafe { self.device.reset_fence(&fence.inner) }.expect("reset_fence failed");
                fence.is_submitted = false;
            }
        }
    }

    fn reset_command_pools(&mut self) {
        for command_pool in &mut self.command_pool {
            unsafe { command_pool.reset() };
        }
    }

    /// Generates a memory report for the resources managed by the device layer.
    pub fn report_memory(&self) -> MemoryReport {
        let mut report = MemoryReport::default();
        for dim in self.depth_targets.keys() {
            report.depth_target_textures += depth_target_size_in_bytes(dim);
        }
        report
    }

    pub fn deinit(mut self) {
        self.device.wait_idle().unwrap();
        for mut texture in self.retained_textures {
            texture.id = 0;
        }
        unsafe {
            if self.save_cache && self.cache_path.is_some() {
                let pipeline_cache = self
                    .device
                    .create_pipeline_cache(None)
                    .expect("Failed to create pipeline cache");
                let data;
                if let Some(cache) = self.pipeline_cache {
                    self.device
                        .merge_pipeline_caches(&cache, Some(&pipeline_cache))
                        .expect("merge_pipeline_caches failed");
                    data = self
                        .device
                        .get_pipeline_cache_data(&cache)
                        .expect("get_pipeline_cache_data failed");
                    self.device.destroy_pipeline_cache(cache);
                } else {
                    data = self
                        .device
                        .get_pipeline_cache_data(&pipeline_cache)
                        .expect("get_pipeline_cache_data failed");
                };
                self.device.destroy_pipeline_cache(pipeline_cache);

                if data.len() > 0 {
                    let mut file = OpenOptions::new()
                        .write(true)
                        .create(true)
                        .open(&self.cache_path.as_ref().unwrap())
                        .expect("File open/creation failed");

                    use std::io::Write;
                    file.write(&data).expect("File write failed");
                }
            } else {
                if let Some(cache) = self.pipeline_cache {
                    self.device.destroy_pipeline_cache(cache);
                }
            }

            for command_pool in self.command_pool {
                command_pool.destroy(&self.device);
            }
            for staging_buffer_pool in self.staging_buffer_pool {
                staging_buffer_pool.deinit(&self.device, &mut self.heaps);
            }
            self.quad_buffer.deinit(&self.device, &mut self.heaps);
            for instance_buffer in self.instance_buffers {
                instance_buffer.deinit(&self.device, &mut self.heaps);
            }
            for buffer in self.free_instance_buffers {
                buffer.deinit(&self.device, &mut self.heaps);
            }
            if let Some(buffer) = self.download_buffer {
                buffer.deinit(&self.device, &mut self.heaps);
            }
            for image in self.frame_images {
                image.deinit(&self.device, &mut self.heaps);
            }
            for depth in self.frame_depths {
                depth.deinit(&self.device, &mut self.heaps);
            }
            for (_, image) in self.images {
                image.deinit(&self.device, &mut self.heaps);
            }
            for (_, rbo) in self.fbos {
                rbo.deinit(&self.device);
            }
            for (_, rbo) in self.rbos {
                rbo.deinit(&self.device, &mut self.heaps);
            }
            for framebuffer in self.framebuffers {
                self.device.destroy_framebuffer(framebuffer);
            }
            for framebuffer_depth in self.framebuffers_depth {
                self.device.destroy_framebuffer(framebuffer_depth);
            }

            for (_ , buffer) in self.gpu_cache_buffers {
                buffer.deinit(&self.device);
            }
            self.device.destroy_sampler(self.sampler_linear);
            self.device.destroy_sampler(self.sampler_nearest);

            self.per_draw_descriptors.free(&mut self.desc_allocator);
            self.per_group_descriptors.free(&mut self.desc_allocator);
            self.per_pass_descriptors.free(&mut self.desc_allocator);
            self.locals_descriptors.free(&mut self.desc_allocator);

            self.desc_allocator.dispose(&self.device);
            self.locals_buffer.deinit(&self.device, &mut self.heaps);
            for (_, program) in self.programs {
                program.deinit(&self.device, &mut self.heaps)
            }
            self.heaps.dispose(&self.device);
            for (_, (vs_module, fs_module)) in self.shader_modules {
                self.device.destroy_shader_module(vs_module);
                self.device.destroy_shader_module(fs_module);
            }
            self.descriptor_data.deinit(&self.device);
            self.render_pass.unwrap().deinit(&self.device);
            for fence in self.frame_fence {
                self.device.destroy_fence(fence.inner);
            }
            self.device
                .destroy_semaphore(self.image_available_semaphore);
            self.device
                .destroy_semaphore(self.render_finished_semaphore);
            if let Some(swap_chain) = self.swap_chain {
                self.device.destroy_swapchain(swap_chain);
            }
        }
        // We must ensure these are dropped before `self._instance` or we segfault with Vulkan
        mem::drop(self.device);
        mem::drop(self.queue_group_queues);
    }
}

pub struct TextureUploader<'a, B: hal::Backend> {
    device: &'a mut Device<B>,
    texture: &'a Texture,
}

impl<'a, B: hal::Backend> TextureUploader<'a, B> {
    pub fn upload<T>(
        &mut self,
        rect: DeviceIntRect,
        layer_index: i32,
        stride: Option<i32>,
        data: &[T],
    ) -> usize {
        let data = unsafe {
            slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * mem::size_of::<T>())
        };
        let data_stride: usize = self.texture.format.bytes_per_pixel() as usize;
        let width = rect.size.width as usize;
        let height = rect.size.height as usize;
        let size = width * height * data_stride;
        let mut new_data = vec![0u8; size];
        let data = if stride.is_some() {
            let row_length = (stride.unwrap()) as usize;

            if data_stride == 4 {
                for j in 0 .. height {
                    for i in 0 .. width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride ..];
                        assert!(
                            offset + 3 < new_data.len(),
                            "offset={:?}, data len={:?}, data stride={:?}",
                            offset,
                            new_data.len(),
                            data_stride,
                        ); // optimization
                           // convert from BGRA
                        new_data[offset + 0] = src[0];
                        new_data[offset + 1] = src[1];
                        new_data[offset + 2] = src[2];
                        new_data[offset + 3] = src[3];
                    }
                }
            } else {
                for j in 0 .. height {
                    for i in 0 .. width {
                        let offset = i * data_stride + j * data_stride * width;
                        let src = &data[j * row_length + i * data_stride ..];
                        for i in 0 .. data_stride {
                            new_data[offset + i] = src[i];
                        }
                    }
                }
            }

            new_data.as_slice()
        } else {
            data
        };
        assert_eq!(
            data.len(),
            width * height * data_stride,
            "data len = {}, width = {}, height = {}, data stride = {}",
            data.len(),
            width,
            height,
            data_stride
        );

        self.texture.bound_in_frame.set(self.device.frame_id);
        self.device
            .images
            .get_mut(&self.texture.id)
            .expect("Texture not found.")
            .update(
                &self.device.device,
                self.device.command_pool[self.device.next_id].command_buffer_mut(),
                &mut self.device.staging_buffer_pool[self.device.next_id],
                rect,
                layer_index,
                data,
            );

        if self.texture.filter == TextureFilter::Trilinear {
            self.device.generate_mipmaps(self.texture);
        }
        size
    }
}

fn texels_to_u8_slice<T: Texel>(texels: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(
            texels.as_ptr() as *const u8,
            texels.len() * mem::size_of::<T>(),
        )
    }
}
