/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source;
use api::{ColorF, ImageFormat};
use api::{DeviceIntPoint, DeviceIntRect, DeviceUintRect, DeviceUintSize};
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

use hal;
use winit;
use back;

// gfx-hal
use hal::{Device as BackendDevice, Instance, PhysicalDevice, QueueFamily, Surface, Swapchain};
use hal::{
    DescriptorPool, Gpu, FrameSync, Primitive,
    Backbuffer, SwapchainConfig,
};
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

pub struct TextureSlot(pub usize);

// In some places we need to temporarily bind a texture to any slot.
const DEFAULT_TEXTURE: TextureSlot = TextureSlot(0);

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

pub struct Device<B: hal::Backend> {
    pub device: B::Device
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
                hal::format::Format::Rgba8Srgb,
                |formats| {
                    formats
                        .into_iter()
                        .find(|format| {
                            format.base_format().1 == ChannelType::Srgb
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
        Device {
            device,
        }
    }

    pub fn create_texture(&mut self, target: TextureTarget) -> Texture {
        Texture { target, width: 0, height: 0,  layer_count: 0, format: ImageFormat::Invalid }
    }

    pub fn max_texture_size(&self) -> u32 {
        1024u32
    }

    pub fn swap_buffers(&mut self) {
        //println!("swap_buffers");
    }
}