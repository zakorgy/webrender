/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Matrix4D;
use fnv::FnvHasher;
use internal_types::{PackedVertex, RenderTargetMode, TextureSampler, DEFAULT_TEXTURE};
use internal_types::{BlurAttribute, ClearAttribute, ClipAttribute, VertexAttribute};
use internal_types::{BatchTextures, DebugFontVertex, DebugColorVertex};
//use notify::{self, Watcher};
use super::shader_source;
use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::Read;
use std::iter::repeat;
use std::mem;
use std::path::PathBuf;
use std::rc::Rc;
//use std::sync::mpsc::{channel, Sender};
//use std::thread;
use webrender_traits::{ColorF, ImageFormat};
use webrender_traits::{DeviceIntPoint, DeviceIntRect, DeviceIntSize, DeviceUintSize};

use rand::Rng;
use std;
use std::env;
use glutin;
use gfx;
use gfx::CommandBuffer;
use gfx::pso::PipelineData;
use gfx::state::{Blend, BlendChannel, BlendValue, Equation, Factor, RefValues};
use gfx_core;
use gfx_core::memory::Typed;
use gfx::Factory;
use gfx::texture;
use gfx::traits::FactoryExt;
use gfx::format::{DepthStencil as DepthFormat, Rgba32F as ColorFormat};
use gfx_device_gl as device_gl;
use gfx_device_gl::{Resources as R, CommandBuffer as CB};
use gfx_window_glutin;
use gfx::CombinedError;
use gfx::format::{Format, R8, Unorm, R8_G8_B8_A8, Rgba8, R32_G32_B32_A32, Rgba32F};
use gfx::memory::{Usage, SHADER_RESOURCE};
//use gfx::format::ChannelType::Unorm;
use gfx::format::TextureSurface;
use tiling::{Frame, PackedLayer, PrimitiveInstance};
use render_task::RenderTaskData;
use prim_store::{GpuBlock16, GpuBlock32, GpuBlock64, GpuBlock128, GradientData, PrimitiveGeometry, TexelRect};
use renderer::{BlendMode, DUMMY_A8_ID, DUMMY_RGBA8_ID};

pub type A8 = (R8, Unorm);
pub const VECS_PER_LAYER: u32 = 13;
pub const VECS_PER_RENDER_TASK: u32 = 3;
pub const VECS_PER_PRIM_GEOM: u32 = 2;
pub const MAX_INSTANCE_COUNT: usize = 2000;
pub const VECS_PER_DATA_16: u32 = 1;
pub const VECS_PER_DATA_32: u32 = 2;
pub const VECS_PER_DATA_64: u32 = 4;
pub const VECS_PER_DATA_128: u32 = 8;
pub const VECS_PER_RESOURCE_RECTS: u32 = 1;
pub const VECS_PER_GRADIENT_DATA: u32 = 4;
pub const FLOAT_SIZE: u32 = 4;
pub const TEXTURE_HEIGTH: u32 = 8;
pub const DEVICE_PIXEL_RATIO: f32 = 1.0;

pub const A8_STRIDE: u32 = 1;
pub const RGBA8_STRIDE: u32 = 4;
pub const FIRST_UNRESERVED_ID: u32 = DUMMY_A8_ID + 1;

pub const ALPHA: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::ZeroPlus(BlendValue::SourceAlpha),
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
};

pub const PREM_ALPHA: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::One,
        destination: Factor::OneMinus(BlendValue::SourceAlpha),
    },
};

pub const SUBPIXEL: Blend = Blend {
    color: BlendChannel {
        equation: Equation::Add,
        source: Factor::ZeroPlus(BlendValue::ConstColor),
        destination: Factor::OneMinus(BlendValue::SourceColor),
    },
    alpha: BlendChannel {
        equation: Equation::Add,
        source: Factor::ZeroPlus(BlendValue::ConstColor),
        destination: Factor::OneMinus(BlendValue::SourceColor),
    },
};

type PSPrimitive = gfx::PipelineState<R, primitive::Meta>;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ProgramId {
    CS_BLUR,
    CS_BOX_SHADOW,
    CS_CLIP_IMAGE,
    CS_CLIP_RECTANGLE,
    CS_TEXT_RUN,
    PS_ANGLE_GRADIENT,
    PS_ANGLE_GRADIENT_TRANSFORM,
    PS_BLEND,
    PS_BORDER,
    PS_BORDER_TRANSFORM,
    PS_BORDER_CORNER,
    PS_BORDER_CORNER_TRANSFORM,
    PS_BORDER_EDGE,
    PS_BORDER_EDGE_TRANSFORM,
    PS_BOX_SHADOW,
    PS_BOX_SHADOW_TRANSFORM,
    PS_CACHE_IMAGE,
    PS_CACHE_IMAGE_TRANSFORM,
    PS_CLEAR,
    PS_CLEAR_TRANSFORM,
    PS_COMPOSITE,
    PS_GRADIENT,
    PS_GRADIENT_TRANSFORM,
    PS_HARDWARE_COMPOSITE,
    PS_IMAGE,
    PS_IMAGE_TRANSFORM,
    PS_RADIAL_GRADIENT,
    PS_RADIAL_GRADIENT_TRANSFORM,
    PS_RECTANGLE,
    PS_RECTANGLE_TRANSFORM,
    PS_RECTANGLE_CLIP,
    PS_RECTANGLE_CLIP_TRANSFORM,
    PS_TEXT_RUN,
    PS_TEXT_RUN_TRANSFORM,
    PS_TEXT_RUN_SUBPIXEL,
    PS_TEXT_RUN_SUBPIXEL_TRANSFORM,
    PS_YUV_IMAGE,
    PS_YUV_IMAGE_TRANSFORM,
    PS_SPLIT_COMPOSITE,
}

gfx_defines! {
    vertex Position {
        pos: [f32; 3] = "aPosition",
    }

    vertex Instances {
        glob_prim_id: i32 = "aGlobalPrimId",
        primitive_address: i32 = "aPrimitiveAddress",
        task_index: i32 = "aTaskIndex",
        clip_task_index: i32 = "aClipTaskIndex",
        layer_index: i32 = "aLayerIndex",
        element_index: i32 = "aElementIndex",
        user_data: [i32; 2] = "aUserData",
        z_index: i32 = "aZIndex",
        // Only ps_clear and ps_clear_transform use this
        //clear_rectangle: [i32; 4] = "aClearRectangle",
    }

    pipeline primitive {
        transform: gfx::Global<[[f32; 4]; 4]> = "uTransform",
        device_pixel_ratio: gfx::Global<f32> = "uDevicePixelRatio",
        vbuf: gfx::VertexBuffer<Position> = (),
        ibuf: gfx::InstanceBuffer<Instances> = (),

        // FIXME: Find the correct data type for these color samplers
        color0: gfx::TextureSampler<[f32; 4]> = "sColor0",
        color1: gfx::TextureSampler<[f32; 4]> = "sColor1",
        color2: gfx::TextureSampler<[f32; 4]> = "sColor2",
        dither: gfx::TextureSampler<[f32; 4]> = "sDither",
        cache_a8: gfx::TextureSampler<f32> = "sCacheA8",
        cache_rgba8: gfx::TextureSampler<[f32; 4]> = "sCacheRGBA8",

        layers: gfx::TextureSampler<[f32; 4]> = "sLayers",
        render_tasks: gfx::TextureSampler<[f32; 4]> = "sRenderTasks",
        prim_geometry: gfx::TextureSampler<[f32; 4]> = "sPrimGeometry",
        data16: gfx::TextureSampler<[f32; 4]> = "sData16",

        data32: gfx::TextureSampler<[f32; 4]> = "sData32",
        data64: gfx::TextureSampler<[f32; 4]> = "sData64",
        data128: gfx::TextureSampler<[f32; 4]> = "sData128",
        resource_rects: gfx::TextureSampler<[f32; 4]> = "sResourceRects",
        gradients : gfx::TextureSampler<[f32; 4]> = "sGradients",

        out_color: gfx::RawRenderTarget = ("oFragColor", Format(gfx::format::SurfaceType::R32_G32_B32_A32, gfx::format::ChannelType::Float), gfx::state::MASK_ALL, None),
        out_depth: gfx::DepthTarget<DepthFormat> = gfx::preset::depth::LESS_EQUAL_WRITE,
        blend_value: gfx::BlendRef = (),
    }
}

impl Position {
    fn new(p: [f32; 2]) -> Position {
        Position {
            pos: [p[0], p[1], 0.0],
        }
    }
}

impl Instances {
    fn new() -> Instances {
        Instances {
            glob_prim_id: 0,
            primitive_address: 0,
            task_index: 0,
            clip_task_index: 0,
            layer_index: 0,
            element_index: 0,
            user_data: [0, 0],
            z_index: 0,
            //clear_rectangle: [0, 0, 0, 0],
        }
    }

    fn update(&mut self, instance: &PrimitiveInstance) {
        self.glob_prim_id = instance.global_prim_id;
        self.primitive_address = instance.prim_address.0;
        self.task_index = instance.task_index;
        self.clip_task_index = instance.clip_task_index;
        self.layer_index = instance.layer_index;
        self.element_index = instance.sub_index;
        self.user_data = instance.user_data;
        self.z_index = instance.z_sort_index;
        // FIXME: Find the value which is used to update self.clear_rectangle.
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Texture<R, T> where R: gfx::Resources,
                               T: gfx::format::TextureFormat {
    // Pixel storage for texture.
    pub surface: gfx::handle::Texture<R, T::Surface>,
    // Sampler for texture.
    pub sampler: gfx::handle::Sampler<R>,
    // View used by shader.
    pub view: gfx::handle::ShaderResourceView<R, T::View>,
    // Filtering mode
    pub filter: TextureFilter,
    // ImageFormat
    pub format: ImageFormat,
    // Render Target mode
    pub mode: RenderTargetMode,
}

impl<R, T> Texture<R, T> where R: gfx::Resources, T: gfx::format::TextureFormat {

    pub fn empty<F>(factory: &mut F, size: [u32; 2]) -> Result<Texture<R, T>, CombinedError>
        where F: gfx::Factory<R>
    {
        Texture::create(factory, None, size, TextureFilter::Nearest)
    }

    pub fn create<F>(factory: &mut F,
                     data: Option<&[&[u8]]>,
                     size: [u32; 2],
                     filter: TextureFilter
    ) -> Result<Texture<R, T>, CombinedError>
        where F: gfx::Factory<R>
    {
        let (width, height) = (size[0] as u16, size[1] as u16);
        let tex_kind = gfx::texture::Kind::D2(width, height,
            gfx::texture::AaMode::Single);
        let filter_method = match filter {
            TextureFilter::Nearest => gfx::texture::FilterMethod::Scale,
            TextureFilter::Linear => gfx::texture::FilterMethod::Bilinear,
        };
        let sampler_info = gfx::texture::SamplerInfo::new(
            filter_method,
            gfx::texture::WrapMode::Clamp
        );

        let (surface, view, format) = {
            use gfx::{format, texture};
            use gfx::memory::{Usage, SHADER_RESOURCE};

            let surface = <T::Surface as format::SurfaceTyped>::get_surface_type();
            let desc = texture::Info {
                kind: tex_kind,
                levels: 1,
                format: surface,
                bind: SHADER_RESOURCE,
                usage: Usage::Dynamic,
            };
            let cty = <T::Channel as format::ChannelTyped>::get_channel_type();
            let raw = try!(factory.create_texture_raw(desc, Some(cty), data));
            let levels = (0, raw.get_info().levels - 1);
            let tex = Typed::new(raw);
            let view = try!(factory.view_texture_as_shader_resource::<T>(
                &tex, levels, format::Swizzle::new()
            ));
            let format = match surface {
                R8 => ImageFormat::A8,
                R8_G8_B8_A8 => ImageFormat::RGBA8,
                R32_G32_B32_A32 => ImageFormat::RGBAF32,
            };
            (tex, view, format)
        };

        let sampler = factory.create_sampler(sampler_info);

        Ok(Texture {
            surface: surface,
            sampler: sampler,
            view: view,
            filter: filter,
            format: format,
            mode: RenderTargetMode::None,
        })
    }

    #[inline(always)]
    pub fn get_size(&self) -> (u32, u32) {
        let (w, h, _, _) = self.surface.get_info().kind.get_dimensions();
        (w as u32, h as u32)
    }

    #[inline(always)]
    fn get_width(&self) -> u32 {
        let (w, _) = self.get_size();
        w
    }

    #[inline(always)]
    fn get_height(&self) -> u32 {
        let (_, h) = self.get_size();
        h
    }
}

struct Program {
    pub data: primitive::Data<R>,
    pub pso: PSPrimitive,
    pub pso_alpha: PSPrimitive,
    pub pso_prem_alpha: PSPrimitive,
    pub pso_subpixel: PSPrimitive,
    pub slice: gfx::Slice<R>,
    pub upload: gfx::handle::Buffer<R, Instances>,
}

impl Program {
    fn new(data: primitive::Data<R>, pso: (PSPrimitive, PSPrimitive, PSPrimitive, PSPrimitive), slice: gfx::Slice<R>, upload: gfx::handle::Buffer<R, Instances>) -> Program {
        Program {
            data: data,
            pso: pso.0,
            pso_alpha: pso.1,
            pso_prem_alpha: pso.2,
            pso_subpixel: pso.3,
            slice: slice,
            upload: upload,
        }
    }

    fn get_pso(&self, blend: &BlendMode) -> &PSPrimitive {
        match *blend {
            BlendMode::None => &self.pso,
            BlendMode::Alpha => &self.pso_alpha,
            BlendMode::PremultipliedAlpha => &self.pso_prem_alpha,
            BlendMode::Subpixel(..) => &self.pso_subpixel,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct FrameId(usize);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureTarget {
    Default,
    Array,
    External,
    Rect,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

impl TextureId {
    pub fn new(name: u32, _: TextureTarget) -> TextureId {
        TextureId {
            name: name,
            //target: gfx::texture::Kind::D2(1,1,gfx::texture::AaMode::Single),
        }
    }

    pub fn invalid() -> TextureId {
        TextureId {
            name: 0,
            //target: gfx::texture::Kind::D2(1,1,gfx::texture::AaMode::Single),
        }
    }

    pub fn invalid_a8() -> TextureId {
        TextureId {
            name: 1,
            //target: gfx::texture::Kind::D2(1,1,gfx::texture::AaMode::Single),
        }
    }

    pub fn is_valid(&self) -> bool { *self != TextureId::invalid() && *self != TextureId::invalid_a8() }
}

#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Debug, Copy, Clone)]
pub struct TextureId {
    name: u32,
    //target: gfx::texture::Kind,
}

#[derive(Debug)]
pub struct TextureData {
    id: TextureId,
    data: Vec<u8>,
    stride: u32,
}

#[derive(Clone, Debug)]
pub enum ShaderError {
    Compilation(String, String), // name, error mssage
    Link(String), // error message
}

pub struct Device {
    device: device_gl::Device,
    factory: device_gl::Factory,
    encoder: gfx::Encoder<R,CB>,
    //textures: HashMap<TextureId, gfx::handle::Texture<R, Rgba8>>,
    textures: HashMap<TextureId, TextureData>,
    programs: HashMap<ProgramId, Program>,
    color0: Texture<R, Rgba8>,
    color1: Texture<R, Rgba8>,
    color2: Texture<R, Rgba8>,
    dither: Texture<R, Rgba8>,
    cache_a8: Texture<R, A8>,
    cache_rgba8: Texture<R, Rgba8>,
    layers: Texture<R, Rgba32F>,
    render_tasks: Texture<R, Rgba32F>,
    prim_geo: Texture<R, Rgba32F>,
    data16: Texture<R, Rgba32F>,
    data32: Texture<R, Rgba32F>,
    data64: Texture<R, Rgba32F>,
    data128: Texture<R, Rgba32F>,
    resource_rects: Texture<R, Rgba32F>,
    gradient_data: Texture<R, Rgba8>,
    max_texture_size: u32,
    main_color: gfx_core::handle::RenderTargetView<R, ColorFormat>,
    main_depth: gfx_core::handle::DepthStencilView<R, DepthFormat>,
}

impl Device {
    pub fn new(window: &glutin::Window) -> Device {
        let (mut device, mut factory, main_color, main_depth) =
            gfx_window_glutin::init_existing::<ColorFormat, DepthFormat>(window);
        println!("Vendor: {:?}", device.get_info().platform_name.vendor);
        println!("Renderer: {:?}", device.get_info().platform_name.renderer);
        println!("Version: {:?}", device.get_info().version);
        println!("Shading Language: {:?}", device.get_info().shading_language);
        let mut encoder: gfx::Encoder<_,_> = factory.create_command_buffer().into();
        let max_texture_size = factory.get_capabilities().max_texture_size as u32;

        let x0 = 0.0;
        let y0 = 0.0;
        let x1 = 1.0;
        let y1 = 1.0;

        let quad_indices: &[u16] = &[ 0, 1, 2, 2, 1, 3 ];
        let quad_vertices = [
            Position::new([x0, y0]),
            Position::new([x1, y0]),
            Position::new([x0, y1]),
            Position::new([x1, y1]),
        ];

        let (vertex_buffer, mut slice) = factory.create_vertex_buffer_with_slice(&quad_vertices, quad_indices);
        slice.instances = Some((MAX_INSTANCE_COUNT as u32, 0));

        let (h, w, _, _) = main_color.get_dimensions();
        let texture_size = [std::cmp::max(1024, h as u32), std::cmp::max(1024, w as u32)];
        let color0 = Texture::empty(&mut factory, texture_size).unwrap();
        let color1 = Texture::empty(&mut factory, texture_size).unwrap();
        let color2 = Texture::empty(&mut factory, texture_size).unwrap();
        let dither = Texture::empty(&mut factory, texture_size).unwrap();
        let cache_a8 = Texture::empty(&mut factory, texture_size).unwrap();
        let cache_rgba8 = Texture::empty(&mut factory, texture_size).unwrap();

        let gradient_data = Texture::empty(&mut factory, [1024 / VECS_PER_GRADIENT_DATA as u32 , TEXTURE_HEIGTH * 10]).unwrap();
        let layers_tex = Texture::empty(&mut factory, [1024 / VECS_PER_LAYER as u32, 64]).unwrap();
        let render_tasks_tex = Texture::empty(&mut factory, [1024 / VECS_PER_RENDER_TASK as u32, TEXTURE_HEIGTH]).unwrap();
        let prim_geo_tex = Texture::empty(&mut factory, [1024 / VECS_PER_PRIM_GEOM as u32, TEXTURE_HEIGTH]).unwrap();
        let data16_tex = Texture::empty(&mut factory, [1024 / VECS_PER_DATA_16 as u32, TEXTURE_HEIGTH * 4]).unwrap();
        let data32_tex = Texture::empty(&mut factory, [1024 / VECS_PER_DATA_32 as u32, TEXTURE_HEIGTH]).unwrap();
        let data64_tex = Texture::empty(&mut factory, [1024 / VECS_PER_DATA_64 as u32, TEXTURE_HEIGTH]).unwrap();
        let data128_tex = Texture::empty(&mut factory, [1024 / VECS_PER_DATA_128 as u32, TEXTURE_HEIGTH * 4]).unwrap();
        let resource_rects = Texture::empty(&mut factory, [1024 / VECS_PER_RESOURCE_RECTS as u32, TEXTURE_HEIGTH * 2]).unwrap();

        let mut programs = HashMap::new();

        let mut textures = HashMap::new();
        let (w, h) = color0.get_size();
        let invalid_id = TextureId::invalid();
        textures.insert(invalid_id, TextureData { id: invalid_id, data: vec![0u8; (w*h*RGBA8_STRIDE) as usize], stride: RGBA8_STRIDE });
        let invalid_a8_id = TextureId::invalid_a8();
        textures.insert(invalid_a8_id, TextureData { id: invalid_a8_id, data: vec![0u8; (w*h*A8_STRIDE) as usize], stride: A8_STRIDE });
        let dummy_rgba8_id = TextureId { name: DUMMY_RGBA8_ID };
        textures.insert(dummy_rgba8_id, TextureData { id: dummy_rgba8_id, data: vec![0u8; (w*h*RGBA8_STRIDE) as usize], stride: RGBA8_STRIDE });
        let dummy_a8_id = TextureId { name: DUMMY_A8_ID };
        textures.insert(dummy_a8_id, TextureData { id: dummy_a8_id, data: vec![0u8; (w*h*A8_STRIDE) as usize], stride: A8_STRIDE });

        let mut device = Device {
            device: device,
            factory: factory,
            encoder: encoder,
            textures: textures,
            programs: programs,
            color0: color0,
            color1: color1,
            color2: color2,
            dither: dither,
            cache_a8: cache_a8,
            cache_rgba8: cache_rgba8,
            layers: layers_tex,
            render_tasks: render_tasks_tex,
            prim_geo: prim_geo_tex,
            data16: data16_tex,
            data32: data32_tex,
            data64: data64_tex,
            data128: data128_tex,
            resource_rects: resource_rects,
            gradient_data: gradient_data,
            max_texture_size: max_texture_size,
            main_color: main_color,
            main_depth: main_depth,
        };
        /*device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/cs_blur.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/cs_blur.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::CS_BLUR);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/cs_box_shadow.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/cs_box_shadow.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::CS_BOX_SHADOW);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/cs_clip_image.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/cs_clip_image.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::CS_CLIP_IMAGE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/cs_clip_rectangle.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/cs_clip_rectangle.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::CS_CLIP_RECTANGLE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/cs_text_run.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/cs_text_run.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::CS_TEXT_RUN);*/
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_RECTANGLE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_RECTANGLE_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle_clip.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle_clip.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_RECTANGLE_CLIP);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle_clip_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_rectangle_clip_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_RECTANGLE_CLIP_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_corner.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_corner.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_BORDER_CORNER);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_corner_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_corner_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_BORDER_CORNER_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_edge.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_edge.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_BORDER_EDGE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_edge_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_border_edge_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_BORDER_EDGE_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_angle_gradient.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_angle_gradient.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_ANGLE_GRADIENT);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_angle_gradient_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_angle_gradient_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_ANGLE_GRADIENT_TRANSFORM);
        /*device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_blend.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_blend.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_BLEND);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_box_shadow.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_box_shadow.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_BOX_SHADOW);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_box_shadow_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_box_shadow_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_BOX_SHADOW_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_cache_image.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_cache_image.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_CACHE_IMAGE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_cache_image_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_cache_image_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_CACHE_IMAGE_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_clear.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_clear.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_CLEAR);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_clear_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_clear_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_CLEAR_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_composite.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_composite.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_COMPOSITE);*/
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_gradient.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_gradient.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_GRADIENT);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_gradient_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_gradient_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_GRADIENT_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_hardware_composite.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_hardware_composite.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_HARDWARE_COMPOSITE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_image.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_image.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_IMAGE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_image_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_image_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_IMAGE_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_radial_gradient.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_radial_gradient.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_RADIAL_GRADIENT);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_radial_gradient_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_radial_gradient_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_RADIAL_GRADIENT_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_TEXT_RUN);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_TEXT_RUN_TRANSFORM);
        /*device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run_subpixel.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run_subpixel.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_TEXT_RUN_SUBPIXEL);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run_subpixel_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_text_run_subpixel_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_TEXT_RUN_SUBPIXEL_TRANSFORM);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_yuv_image.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_yuv_image.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_YUV_IMAGE);
        device.add_program(include_bytes!(concat!(env!("OUT_DIR"), "/ps_yuv_image_transform.vert")),
                           include_bytes!(concat!(env!("OUT_DIR"), "/ps_yuv_image_transform.frag")),
                           vertex_buffer.clone(), slice.clone(), ProgramId::PS_YUV_IMAGE_TRANSFORM);*/
        device
    }

    fn create_psos(&mut self, vert_src: &[u8],frag_src: &[u8]) -> (PSPrimitive, PSPrimitive, PSPrimitive, PSPrimitive) {
        let pso = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::new()
        ).unwrap();

        let pso_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("oFragColor", Format(gfx::format::SurfaceType::R32_G32_B32_A32, gfx::format::ChannelType::Float), gfx::state::MASK_ALL, Some(ALPHA)),
                .. primitive::new()
            }
        ).unwrap();

        let pso_prem_alpha = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("oFragColor", Format(gfx::format::SurfaceType::R32_G32_B32_A32, gfx::format::ChannelType::Float), gfx::state::MASK_ALL, Some(PREM_ALPHA)),
                .. primitive::new()
            }
        ).unwrap();

        let pso_subpixel = self.factory.create_pipeline_simple(
            vert_src,
            frag_src,
            primitive::Init {
                out_color: ("oFragColor", Format(gfx::format::SurfaceType::R32_G32_B32_A32, gfx::format::ChannelType::Float), gfx::state::MASK_ALL, Some(SUBPIXEL)),
                .. primitive::new()
            }
        ).unwrap();

        (pso, pso_alpha, pso_prem_alpha, pso_subpixel)
    }

    fn add_program(&mut self,
                   vert_src: &[u8],
                   frag_src: &[u8],
                   vertex_buffer: gfx::handle::Buffer<R, Position>,
                   slice: gfx::Slice<R>,
                   program_id: ProgramId) {
        let upload = self.factory.create_upload_buffer(MAX_INSTANCE_COUNT).unwrap();
        {
            let mut writer = self.factory.write_mapping(&upload).unwrap();
            for i in 0..MAX_INSTANCE_COUNT {
                writer[i] = Instances::new();
            }
        }

        let instances = self.factory.create_buffer(MAX_INSTANCE_COUNT,
                                                   gfx::buffer::Role::Vertex,
                                                   gfx::memory::Usage::Data,
                                                   gfx::TRANSFER_DST).unwrap();

        let data = primitive::Data {
            transform: [[0f32;4];4],
            device_pixel_ratio: DEVICE_PIXEL_RATIO,
            vbuf: vertex_buffer,
            ibuf: instances,
            color0: (self.color0.clone().view, self.color0.clone().sampler),
            color1: (self.color1.clone().view, self.color1.clone().sampler),
            color2: (self.color2.clone().view, self.color2.clone().sampler),
            dither: (self.dither.clone().view, self.dither.clone().sampler),
            cache_a8: (self.cache_a8.clone().view, self.cache_a8.clone().sampler),
            cache_rgba8: (self.cache_rgba8.clone().view, self.cache_rgba8.clone().sampler),
            layers: (self.layers.clone().view, self.layers.clone().sampler),
            render_tasks: (self.render_tasks.clone().view, self.render_tasks.clone().sampler),
            prim_geometry: (self.prim_geo.clone().view, self.prim_geo.clone().sampler),
            data16: (self.data16.clone().view, self.data16.clone().sampler),
            data32: (self.data32.clone().view, self.data32.clone().sampler),
            data64: (self.data64.clone().view, self.data64.clone().sampler),
            data128: (self.data128.clone().view, self.data128.clone().sampler),
            resource_rects: (self.resource_rects.clone().view, self.resource_rects.clone().sampler),
            gradients: (self.gradient_data.clone().view, self.gradient_data.clone().sampler),
            out_color: self.main_color.raw().clone(),
            out_depth: self.main_depth.clone(),
            blend_value: [0.0, 0.0, 0.0, 0.0]
        };
        let psos = self.create_psos(vert_src, frag_src);
        let program = Program::new(data, psos, slice, upload);
        self.programs.insert(program_id, program);
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    fn generate_texture_id(&mut self) -> TextureId {
        //let mut rng = rand::thread_rng();
        use rand::OsRng;
        let mut rng = OsRng::new().unwrap();

        let mut texture_id = TextureId::invalid();
        loop {
            texture_id.name = rng.gen_range(FIRST_UNRESERVED_ID, u32::max_value());
            if !self.textures.contains_key(&texture_id) {
                break;
            }
        }
        texture_id
    }

    pub fn create_texture_ids(&mut self,
                              count: i32,
                              _target: TextureTarget,
                              format: ImageFormat) -> Vec<TextureId> {
        let mut texture_ids = Vec::new();

        let (w, h) = self.color0.get_size();
        for _ in 0..count {
            let texture_id = self.generate_texture_id();

            //let texture = self.create_texture::<Rgba8>(gfx::texture::Kind::D2(w as u16, h as u16, gfx::texture::AaMode::Single)).unwrap();
            let stride = match format {
                ImageFormat::A8 => A8_STRIDE,
                ImageFormat::RGBA8 => RGBA8_STRIDE,
                _ => unimplemented!(),
            };
            let texture_data = vec![0u8; (w*h*stride) as usize];

            debug_assert!(self.textures.contains_key(&texture_id) == false);
            self.textures.insert(texture_id, TextureData {id: texture_id, data: texture_data, stride: stride });
            //println!("after instert {:?} ts:{:?} s:{:?} {:?}", self.textures[&texture_id].id, self.textures[&texture_id].stride, stride, format);

            texture_ids.push(texture_id);
        }

        texture_ids
    }

    pub fn create_texture_id(&mut self,
                             _target: TextureTarget,
                             format: ImageFormat) -> TextureId {
        let mut texture_ids = Vec::new();

        let (w, h) = self.color0.get_size();
        let texture_id = self.generate_texture_id();

        //let texture = self.create_texture::<Rgba8>(gfx::texture::Kind::D2(w as u16, h as u16, gfx::texture::AaMode::Single)).unwrap();
        let stride = match format {
            ImageFormat::A8 => A8_STRIDE,
            ImageFormat::RGBA8 => RGBA8_STRIDE,
            _ => unimplemented!(),
        };
        let texture_data = vec![0u8; (w*h*stride) as usize];

        debug_assert!(self.textures.contains_key(&texture_id) == false);
        self.textures.insert(texture_id, TextureData {id: texture_id, data: texture_data, stride: stride });
        //println!("after instert {:?} ts:{:?} s:{:?} {:?}", self.textures[&texture_id].id, self.textures[&texture_id].stride, stride, format);

        texture_ids.push(texture_id);

        texture_id
    }

    pub fn init_texture(&mut self,
                        texture_id: TextureId,
                        _width: u32,
                        _height: u32,
                        format: ImageFormat,
                        _filter: TextureFilter,
                        _mode: RenderTargetMode,
                        pixels: Option<&[u8]>) {
        let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
        let stride = match format {
            ImageFormat::A8 => A8_STRIDE,
            ImageFormat::RGBA8 => RGBA8_STRIDE,
            _ => unimplemented!(),
        };
        if stride != texture.stride {
            texture.stride = stride;
            texture.data.clear();
        }
        let actual_pixels = match pixels {
            Some(data) => data.to_vec(),
            None => {
                let (w, h) = self.color0.get_size();
                let data = vec![0u8; (w*h*texture.stride) as usize];
                data
            }
        };
        //println!("init_texture id:{:?} {} {} {}", texture_id, texture.data.len(), actual_pixels.len(), texture.stride);
        //debug_assert!(texture.len() == actual_pixels.len());
        //texture = &mut actual_pixels.to_vec();
        mem::replace(&mut texture.data, actual_pixels);
    }

    pub fn update_texture(&mut self,
                          texture_id: TextureId,
                          x0: u32,
                          y0: u32,
                          width: u32,
                          height: u32,
                          stride: Option<u32>,
                          data: &[u8]) {
        let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
        //println!("update_texture id:{:?} {} {} ts:{:?} s:{:?}", texture_id, texture.data.len(), data.len(), texture.stride, stride);
        //debug_assert!(texture.len() == data.len());
        //texture = &mut data.to_vec();
        let (w, h) = self.color0.get_size();
        /*let row_length = match stride {
            Some(value) => value / bpp,
            None => width,
        };*/
        /*let converted_data = */Device::update_texture_data(&mut texture.data, x0, y0, width, height, w, h, data, texture.stride);
        //println!("update_texture id:{:?} {} {}", texture_id, texture.data.len(), converted_data.len());
        
        //mem::replace(&mut texture.data, converted_data);
    }

    pub fn resize_texture(&mut self,
                          texture_id: TextureId,
                          new_width: u32,
                          new_height: u32,
                          format: ImageFormat,
                          filter: TextureFilter,
                          mode: RenderTargetMode) {
          println!("Unimplemented! resize_texture");
    }

    pub fn deinit_texture(&mut self, texture_id: TextureId) {
        let texture = self.textures.get_mut(&texture_id).expect("Didn't find texture!");
        let (w, h) = self.color0.get_size();
        let data = vec![0u8; (w*h*4) as usize];
        //println!("deinit_texture id:{:?} {} {}", texture_id, texture.data.len(), data.len());
        //debug_assert!(texture.len() == data.len());
        //texture = &mut data.to_vec();
        mem::replace(&mut texture.data, data.to_vec());
    }

    fn update_texture_data(data: &mut [u8], x_offset: u32, y_offset: u32, width: u32, height: u32, max_width: u32, max_height: u32, new_data: &[u8], stride: u32)/* -> Vec<u8>*/ {
        //let mut data = vec![0u8; (max_width*max_height*stride) as usize];
        assert_eq!(width * height * stride, new_data.len() as u32);
        for j in 0..height {
            for i in 0..width*stride {
                let k = {
                    if stride == 1 {
                        i
                    } else if i % 4 == 0 {
                        i + 2
                    } else if i % 4 == 2 {
                        i - 2
                    } else {
                        i
                    }
                };
                data[((i+x_offset*stride)+(j+y_offset)*max_width*stride) as usize] = new_data[(k+j*width*stride) as usize];
            }
        }
        /*data.to_vec()*/
    }

    pub fn bind_texture(&mut self,
                        sampler: TextureSampler,
                        texture_id: TextureId) {
        //println!("bind_texture {:?} {:?}", texture_id, sampler);
        let texture = match self.textures.get(&texture_id) {
            Some(data) => data,
            None => {
                println!("Didn't find texture! {}", texture_id.name);
                return;
            }
        };
        match sampler {
            TextureSampler::Color0 => Device::update_rgba_texture_u8(&mut self.encoder, &self.color0, texture.data.as_slice()),
            TextureSampler::Color1 => Device::update_rgba_texture_u8(&mut self.encoder, &self.color1, texture.data.as_slice()),
            TextureSampler::Color2 => Device::update_rgba_texture_u8(&mut self.encoder, &self.color2, texture.data.as_slice()),
            TextureSampler::CacheA8 => Device::update_a_texture_u8(&mut self.encoder, &self.cache_a8, texture.data.as_slice()),
            TextureSampler::CacheRGBA8 => Device::update_rgba_texture_u8(&mut self.encoder, &self.cache_rgba8, texture.data.as_slice()),
            _ => {
                println!("There are only 5 samplers supported. {:?}", sampler);
            }
        }
    }

    pub fn clear_target(&mut self, color: Option<[f32; 4]>, depth: Option<f32>) {
        if let Some(color) = color {
            println!("clear:{:?}", color);
            self.encoder.clear(&self.main_color,
                               [color[0].powf(2.2),
                                color[1].powf(2.2),
                                color[2].powf(2.2),
                                color[3].powf(2.2)]);
        }

        if let Some(depth) = depth {
            self.encoder.clear_depth(&self.main_depth, depth);
        }
    }

    pub fn update(&mut self, frame: &mut Frame) {
        /*println!("update!");
        println!("gpu_data16.len {}", frame.gpu_data16.len());
        println!("gpu_data32.len {}", frame.gpu_data32.len());
        println!("gpu_data64.len {}", frame.gpu_data64.len());
        println!("gpu_data128.len {}", frame.gpu_data128.len());
        println!("gpu_geometry.len {}", frame.gpu_geometry.len());
        println!("gpu_resource_rects.len {}", frame.gpu_resource_rects.len());
        println!("layer_texture_data.len {}", frame.layer_texture_data.len());
        println!("render_task_data.len {}", frame.render_task_data.len());
        println!("gpu_gradient_data.len {}", frame.gpu_gradient_data.len());
        println!("device_pixel_ratio: {}", frame.device_pixel_ratio);*/
        Device::update_texture_f32(&mut self.encoder, &self.layers, Device::convert_layer(frame.layer_texture_data.clone()).as_slice());
        Device::update_texture_f32(&mut self.encoder, &self.render_tasks, Device::convert_render_task(frame.render_task_data.clone()).as_slice());
        Device::update_texture_f32(&mut self.encoder, &self.prim_geo, Device::convert_prim_geo(frame.gpu_geometry.clone()).as_slice());
        Device::update_texture_f32(&mut self.encoder, &self.data16, Device::convert_data16(frame.gpu_data16.clone()).as_slice());
        Device::update_texture_f32(&mut self.encoder, &self.data32, Device::convert_data32(frame.gpu_data32.clone()).as_slice());
        Device::update_texture_f32(&mut self.encoder, &self.data64, Device::convert_data64(frame.gpu_data64.clone()).as_slice());
        Device::update_texture_f32(&mut self.encoder, &self.data128, Device::convert_data128(frame.gpu_data128.clone()).as_slice());
        Device::update_texture_f32(&mut self.encoder, &self.resource_rects, Device::convert_resource_rects(frame.gpu_resource_rects.clone()).as_slice());
        Device::update_rgba_texture_u8(&mut self.encoder, &self.gradient_data, Device::convert_gradient_data(frame.gpu_gradient_data.clone()).as_slice());
    }

    pub fn flush(&mut self) {
        println!("flush");
        self.encoder.flush(&mut self.device);
    }

    pub fn draw(&mut self, program_id: &ProgramId, proj: &Matrix4D<f32>, instances: &[PrimitiveInstance], textures: &BatchTextures, blendmode: &BlendMode) {
        /*println!("draw!");
        println!("proj: {:?}", proj);
        println!("data: {:?}", instances);*/
        if let Some(program) = self.programs.get_mut(program_id) {
            program.data.transform = proj.to_row_arrays();
            {
                let mut writer = self.factory.write_mapping(&program.upload).unwrap();
                //println!("writer: {} instances: {}", writer.len(), instances.len());
                for (i, inst) in instances.iter().enumerate() {
                    //println!("instance[{}]: {:?}", i, inst);
                    writer[i].update(inst);
                    //println!("instance[{}]: {:?}", i, writer[i]);
                }
            }
            {
                //writer[0].update(&instances[0]);
                program.slice.instances = Some((instances.len() as u32, 0));
            }
            //println!("upload {:?}", &self.upload);
            //println!("copy");
            if let &BlendMode::Subpixel(ref color) = blendmode {
                program.data.blend_value = [color.r, color.g, color.b, color.a];
            }

            self.encoder.copy_buffer(&program.upload, &program.data.ibuf,
                                     0, 0, program.upload.len()).unwrap();
            /*println!("vbuf {:?}", self.data.vbuf.get_info());
            println!("ibuf {:?}", self.data.ibuf);
            println!("layers {:?}", self.layers);
            println!("render_tasks {:?}", self.render_tasks);
            println!("prim_geo {:?}", self.prim_geo);
            println!("data16 {:?}", self.data16);*/
            self.encoder.draw(&program.slice, &program.get_pso(blendmode), &program.data);
        } else {
            println!("Shader not yet implemented {:?}",  program_id);
        }
    }

    pub fn update_rgba_texture_u8(encoder: &mut gfx::Encoder<R,CB>, texture: &Texture<R, Rgba8>, memory: &[u8]) {
        let tex = &texture.surface;
        let (width, height) = texture.get_size();
        let img_info = gfx::texture::ImageInfoCommon {
            xoffset: 0,
            yoffset: 0,
            zoffset: 0,
            width: width as u16,
            height: height as u16,
            depth: 0,
            format: (),
            mipmap: 0,
        };

        let data = gfx::memory::cast_slice(memory);
        encoder.update_texture::<_, Rgba8>(tex, None, img_info, data).unwrap();
    }

    pub fn update_a_texture_u8(encoder: &mut gfx::Encoder<R,CB>, texture: &Texture<R, A8>, memory: &[u8]) {
        let tex = &texture.surface;
        let (width, height) = texture.get_size();
        let img_info = gfx::texture::ImageInfoCommon {
            xoffset: 0,
            yoffset: 0,
            zoffset: 0,
            width: width as u16,
            height: height as u16,
            depth: 0,
            format: (),
            mipmap: 0,
        };

        let data = gfx::memory::cast_slice(memory);
        encoder.update_texture::<_, A8>(tex, None, img_info, data).unwrap();
    }

    pub fn update_texture_f32(encoder: &mut gfx::Encoder<R,CB>, texture: &Texture<R, Rgba32F>, memory: &[f32]) {
        let tex = &texture.surface;
        let (width, height) = texture.get_size();
        let img_info = gfx::texture::ImageInfoCommon {
            xoffset: 0,
            yoffset: 0,
            zoffset: 0,
            width: width as u16,
            height: height as u16,
            depth: 0,
            format: (),
            mipmap: 0,
        };

        let data = gfx::memory::cast_slice(memory);
        encoder.update_texture::<_, Rgba32F>(tex, None, img_info, data).unwrap();
    }

    fn convert_data16(data16: Vec<GpuBlock16>) -> Vec<f32> {
        let mut data: Vec<f32> = vec!();
        for d in data16 {
            data.append(&mut d.data.to_vec());
        }
        let max_size = ((1024 / VECS_PER_DATA_16) * FLOAT_SIZE * TEXTURE_HEIGTH * 4) as usize;
        println!("convert_data16 len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_data32(data32: Vec<GpuBlock32>) -> Vec<f32> {
        let mut data: Vec<f32> = vec!();
        for d in data32 {
            data.append(&mut d.data.to_vec());
        }
        let max_size = ((1024 / VECS_PER_DATA_32) * FLOAT_SIZE * TEXTURE_HEIGTH) as usize;
        println!("convert_data32 len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_data64(data64: Vec<GpuBlock64>) -> Vec<f32> {
        let mut data: Vec<f32> = vec!();
        for d in data64 {
            data.append(&mut d.data.to_vec());
        }
        let max_size = ((1024 / VECS_PER_DATA_64) * FLOAT_SIZE * TEXTURE_HEIGTH) as usize;
        println!("convert_data64 len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; (max_size - data.len())];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_data128(data128: Vec<GpuBlock128>) -> Vec<f32> {
        let mut data: Vec<f32> = vec!();
        for d in data128 {
            data.append(&mut d.data.to_vec());
        }
        let max_size = ((1024 / VECS_PER_DATA_128) * FLOAT_SIZE * TEXTURE_HEIGTH * 4) as usize;
        println!("convert_data128 len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_layer(layers: Vec<PackedLayer>) -> Vec<f32> {
        let mut data: Vec<f32> = vec!();
        for l in layers {
            //println!("{:?}", l);
            data.append(&mut l.transform.to_row_major_array().to_vec());
            data.append(&mut l.inv_transform.to_row_major_array().to_vec());
            data.append(&mut l.local_clip_rect.origin.to_array().to_vec());
            data.append(&mut l.local_clip_rect.size.to_array().to_vec());
            data.append(&mut l.screen_vertices[0].to_array().to_vec());
            data.append(&mut l.screen_vertices[1].to_array().to_vec());
            data.append(&mut l.screen_vertices[2].to_array().to_vec());
            data.append(&mut l.screen_vertices[3].to_array().to_vec());
        }
        let max_size = ((1024 / VECS_PER_LAYER) * FLOAT_SIZE * 64) as usize;
        println!("convert_layer len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_render_task(render_tasks: Vec<RenderTaskData>) -> Vec<f32> {
        let mut data: Vec<f32> = vec!();
        for rt in render_tasks {
            data.append(&mut rt.data.to_vec());
        }
        let max_size = ((1024 / VECS_PER_RENDER_TASK) * FLOAT_SIZE * TEXTURE_HEIGTH) as usize;
        println!("convert_render_task len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_prim_geo(prim_geo: Vec<PrimitiveGeometry>) -> Vec<f32> {
        println!("PrimitiveGeometry Vec length: {:?}", prim_geo.len());
        let mut data: Vec<f32> = vec!();
        for pg in prim_geo {
            if data.len() < 30 {
                println!("PrimitiveGeometry : {:?}", pg);
            }
            data.append(&mut pg.local_rect.origin.to_array().to_vec());
            data.append(&mut pg.local_rect.size.to_array().to_vec());
            data.append(&mut pg.local_clip_rect.origin.to_array().to_vec());
            data.append(&mut pg.local_clip_rect.size.to_array().to_vec());
        }
        let max_size = ((1024 / VECS_PER_PRIM_GEOM) * FLOAT_SIZE * TEXTURE_HEIGTH) as usize;
        println!("convert_prim_geo len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_resource_rects(resource_rects: Vec<TexelRect>) -> Vec<f32> {
        let mut data: Vec<f32> = vec!();
        for r in resource_rects {
            data.append(&mut r.uv0.to_array().to_vec());
            data.append(&mut r.uv1.to_array().to_vec());
        }
        let max_size = ((1024 / VECS_PER_RESOURCE_RECTS) * FLOAT_SIZE * TEXTURE_HEIGTH * 2) as usize;
        println!("convert_resource_rects len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0f32; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }

    fn convert_gradient_data(gradient_data_vec: Vec<GradientData>) -> Vec<u8> {
        let mut data: Vec<u8> = vec!();
        for gradient_data in gradient_data_vec {
            for entry in gradient_data.colors_high.iter() {
                data.push(entry.start_color.r);
                data.push(entry.start_color.g);
                data.push(entry.start_color.b);
                data.push(entry.start_color.a);
                data.push(entry.end_color.r);
                data.push(entry.end_color.g);
                data.push(entry.end_color.b);
                data.push(entry.end_color.a);
            }
            for entry in gradient_data.colors_low.iter() {
                data.push(entry.start_color.r);
                data.push(entry.start_color.g);
                data.push(entry.start_color.b);
                data.push(entry.start_color.a);
                data.push(entry.end_color.r);
                data.push(entry.end_color.g);
                data.push(entry.end_color.b);
                data.push(entry.end_color.a);
            }
        }
        let max_size = ((1024 / VECS_PER_GRADIENT_DATA) * 4 * TEXTURE_HEIGTH * 10) as usize;
        println!("convert_gradient_data len {:?} max_size: {}", data.len(), max_size);
        if max_size > data.len() {
            let mut zeros = vec![0u8; max_size - data.len()];
            data.append(&mut zeros);
        }
        assert!(data.len() == max_size);
        data
    }
}
