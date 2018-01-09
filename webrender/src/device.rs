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

pub struct Device {

}

impl Device {
    pub fn new() -> Device {
        Device { }
    }

    pub fn create_texture(&mut self, target: TextureTarget) -> Texture {
        Texture { target, width: 0, height: 0,  layer_count: 0, format: ImageFormat::Invalid }
    }

    pub fn max_texture_size(&self) -> u32 {
        1024u32
    }

    pub fn swap_buffers(&mut self) {
        println!("swap_buffers");
    }
}