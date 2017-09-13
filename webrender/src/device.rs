/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Transform3D;
use internal_types::RenderTargetMode;
use super::shader_source;
use std::fs::File;
use std::io::Read;
use std::iter::repeat;
use std::mem;
use std::ops::Add;
use std::path::PathBuf;
use std::ptr;
use std::rc::Rc;
use std::thread;
use api::{ColorF, ImageFormat};
use api::{DeviceIntPoint, DeviceIntRect, DeviceIntSize, DeviceUintSize};

use InitWindow;
use ResultWindow;

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

pub struct TextureSlot(pub usize);

// In some places we need to temporarily bind a texture to any slot.
const DEFAULT_TEXTURE: TextureSlot = TextureSlot(0);

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureTarget {
    Default,
    Array,
    Rect,
    External,
}


#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Debug)]
pub enum VertexAttributeKind {
    F32,
    U8Norm,
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

impl VertexAttributeKind {
    fn size_in_bytes(&self) -> u32 {
        match *self {
            VertexAttributeKind::F32 => 4,
            VertexAttributeKind::U8Norm => 1,
            VertexAttributeKind::I32 => 4,
            VertexAttributeKind::U16 => 2,
        }
    }
}

impl VertexAttribute {
    fn size_in_bytes(&self) -> u32 {
        self.count * self.kind.size_in_bytes()
    }
}

impl VertexDescriptor {
    fn instance_stride(&self) -> u32 {
        self.instance_attributes
            .iter()
            .map(|attr| attr.size_in_bytes()).sum()
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
    mode: RenderTargetMode,
}

impl Texture {
    pub fn get_dimensions(&self) -> DeviceUintSize {
        DeviceUintSize::new(self.width, self.height)
    }

    pub fn get_layer_count(&self) -> i32 {
        self.layer_count
    }

    pub fn get_bpp(&self) -> u32 {
        match self.format {
            ImageFormat::A8 => 1,
            ImageFormat::RGB8 => 3,
            ImageFormat::BGRA8 => 4,
            ImageFormat::RG8 => 2,
            ImageFormat::RGBAF32 => 16,
            ImageFormat::Invalid => unreachable!(),
        }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        debug_assert!(thread::panicking() || self.id == 0);
    }
}

const MAX_TIMERS_PER_FRAME: usize = 256;
const MAX_SAMPLERS_PER_FRAME: usize = 16;
const MAX_PROFILE_FRAMES: usize = 4;

pub trait NamedTag {
    fn get_label(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct GpuTimer<T> {
    pub tag: T,
    pub time_ns: u64,
}

#[derive(Debug, Clone)]
pub struct GpuSampler<T> {
    pub tag: T,
    pub count: u64,
}

pub struct QuerySet<T> {
    data: Vec<T>,
}

impl<T> QuerySet<T> {
    fn new() -> Self {
        QuerySet {
            data: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.data.clear();
    }

    fn add(&mut self, value: T) {
        self.data.push(value);
    }

    fn take(&mut self) -> Vec<T> {
        let mut data = mem::replace(&mut self.data, Vec::new());
        data
    }
}

pub struct GpuFrameProfile<T> {
    timers: QuerySet<GpuTimer<T>>,
    samplers: QuerySet<GpuSampler<T>>,
    frame_id: FrameId,
    inside_frame: bool,
}

impl<T> GpuFrameProfile<T> {
    fn new() -> Self {
        GpuFrameProfile {
            timers: QuerySet::new(),
            samplers: QuerySet::new(),
            frame_id: FrameId(0),
            inside_frame: false,
        }
    }

    fn begin_frame(&mut self, frame_id: FrameId) {
        self.frame_id = frame_id;
        self.timers.reset();
        self.samplers.reset();
        self.inside_frame = true;
    }

    fn end_frame(&mut self) {
        self.done_marker();
        self.done_sampler();
        self.inside_frame = false;
    }

    fn done_marker(&mut self) {
        debug_assert!(self.inside_frame);
    }

    fn add_marker(&mut self, tag: T) -> GpuMarker where T: NamedTag {
        self.done_marker();

        let marker = GpuMarker::new(tag.get_label());

        self.timers.add(GpuTimer { tag, time_ns: 0 });

        marker
    }

    fn done_sampler(&mut self) {
        /* FIXME: samplers crash on MacOS
        debug_assert!(self.inside_frame);
        if self.samplers.pending != 0 {
            self.gl.end_query(gl::SAMPLES_PASSED);
            self.samplers.pending = 0;
        }
        */
    }

    fn add_sampler(&mut self, _tag: T) where T: NamedTag {
        /* FIXME: samplers crash on MacOS
        self.done_sampler();

        if let Some(query) = self.samplers.add(GpuSampler { tag, count: 0 }) {
            self.gl.begin_query(gl::SAMPLES_PASSED, query);
        }
        */
    }

    fn is_valid(&self) -> bool {
        //!self.timers.set.is_empty() || !self.samplers.set.is_empty()
        true
    }

    fn build_samples(&mut self) -> (Vec<GpuTimer<T>>, Vec<GpuSampler<T>>) {
        debug_assert!(!self.inside_frame);
        (self.timers.take(), self.samplers.take())
    }
}

impl<T> Drop for GpuFrameProfile<T> {
    fn drop(&mut self) {
    }
}

pub struct GpuProfiler<T> {
    frames: [GpuFrameProfile<T>; MAX_PROFILE_FRAMES],
    next_frame: usize,
}

impl<T> GpuProfiler<T> {
    pub fn new() -> GpuProfiler<T> {
        GpuProfiler {
            next_frame: 0,
            frames: [
                GpuFrameProfile::new(),
                GpuFrameProfile::new(),
                GpuFrameProfile::new(),
                GpuFrameProfile::new(),
            ],
        }
    }

    pub fn build_samples(&mut self) -> Option<(FrameId, Vec<GpuTimer<T>>, Vec<GpuSampler<T>>)> {
        let frame = &mut self.frames[self.next_frame];
        if frame.is_valid() {
            let (timers, samplers) = frame.build_samples();
            Some((frame.frame_id, timers, samplers))
        } else {
            None
        }
    }

    pub fn begin_frame(&mut self, frame_id: FrameId) {
        let frame = &mut self.frames[self.next_frame];
        frame.begin_frame(frame_id);
    }

    pub fn end_frame(&mut self) {
        let frame = &mut self.frames[self.next_frame];
        frame.end_frame();
        self.next_frame = (self.next_frame + 1) % MAX_PROFILE_FRAMES;
    }

    pub fn add_marker(&mut self, tag: T) -> GpuMarker
    where T: NamedTag {
        self.frames[self.next_frame].add_marker(tag)
    }

    pub fn add_sampler(&mut self, tag: T)
    where T: NamedTag {
        self.frames[self.next_frame].add_sampler(tag)
    }

    pub fn done_sampler(&mut self) {
        self.frames[self.next_frame].done_sampler()
    }
}

#[must_use]
pub struct GpuMarker{
}

impl GpuMarker {
    pub fn new(message: &str) -> GpuMarker {
        GpuMarker{
        }
    }

    pub fn fire(message: &str) {
    }
}

#[cfg(not(any(target_arch="arm", target_arch="aarch64")))]
impl Drop for GpuMarker {
    fn drop(&mut self) {
    }
}

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
    Link(String, String), // name, error message
}

pub struct Device {
    // device state
    device_pixel_ratio: f32,

    // HW or API capabilties
    capabilities: Capabilities,

    // debug
    inside_frame: bool,

    // resources
    resource_override_path: Option<PathBuf>,

    max_texture_size: u32,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: FrameId,
}

impl Device {
    pub fn new(window: Rc<InitWindow>, resource_override_path: Option<PathBuf>) -> (Device, ResultWindow) {
        let max_texture_size = 1024;

        let device = Device {
            resource_override_path,
            // This is initialized to 1 by default, but it is set
            // every frame by the call to begin_frame().
            device_pixel_ratio: 1.0,
            inside_frame: false,

            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },

            max_texture_size,
            frame_id: FrameId(0),
        };
        (device, None)
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn get_capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    pub fn reset_state(&mut self) {
    }

    pub fn begin_frame(&mut self, device_pixel_ratio: f32) -> FrameId {
        debug_assert!(!self.inside_frame);
        self.inside_frame = true;
        self.device_pixel_ratio = device_pixel_ratio;
        self.frame_id
    }

    pub fn bind_texture<S>(&mut self,
                           sampler: S,
                           texture: &Texture) where S: Into<TextureSlot> {
        debug_assert!(self.inside_frame);

        let sampler_index = sampler.into().0;
        /*if self.bound_textures[sampler_index] != texture.id {
            self.bound_textures[sampler_index] = texture.id;
            self.gl.active_texture(gl::TEXTURE0 + sampler_index as gl::GLuint);
            self.gl.bind_texture(texture.target, texture.id);
            self.gl.active_texture(gl::TEXTURE0);
        }*/
    }


    pub fn create_texture(&mut self, target: TextureTarget) -> Texture {
        Texture {
            id: 0,
            target: target,
            width: 0,
            height: 0,
            layer_count: 0,
            format: ImageFormat::Invalid,
            filter: TextureFilter::Nearest,
            mode: RenderTargetMode::None,
        }
    }

    pub fn init_texture(&mut self,
                        texture: &mut Texture,
                        width: u32,
                        height: u32,
                        format: ImageFormat,
                        filter: TextureFilter,
                        mode: RenderTargetMode,
                        layer_count: i32,
                        pixels: Option<&[u8]>) {
        debug_assert!(self.inside_frame);

        let resized = texture.width != width || texture.height != height;

        texture.format = format;
        texture.width = width;
        texture.height = height;
        texture.filter = filter;
        texture.layer_count = layer_count;
        texture.mode = mode;

        match mode {
            RenderTargetMode::RenderTarget => {
            }
            RenderTargetMode::None => {
            }
        }
    }

    pub fn free_texture_storage(&mut self, texture: &mut Texture) {
        debug_assert!(self.inside_frame);

        if texture.format == ImageFormat::Invalid {
            return;
        }

        texture.format = ImageFormat::Invalid;
        texture.width = 0;
        texture.height = 0;
        texture.layer_count = 0;
    }

    pub fn delete_texture(&mut self, mut texture: Texture) {
        self.free_texture_storage(&mut texture);
        texture.id = 0;
    }

    pub fn update_pbo_data<T>(&mut self, data: &[T]) {
        debug_assert!(self.inside_frame);
        //debug_assert_ne!(self.bound_pbo, 0);

        /*gl::buffer_data(&*self.gl,
                        gl::PIXEL_UNPACK_BUFFER,
                        data,
                        gl::STREAM_DRAW);*/
    }

    pub fn update_texture_from_pbo(&mut self,
                                   texture: &Texture,
                                   x0: u32,
                                   y0: u32,
                                   width: u32,
                                   height: u32,
                                   layer_index: i32,
                                   stride: Option<u32>,
                                   offset: usize) {
        debug_assert!(self.inside_frame);

        /*let (gl_format, bpp, data_type) = match texture.format {
            ImageFormat::A8 => (GL_FORMAT_A, 1, gl::UNSIGNED_BYTE),
            ImageFormat::RGB8 => (gl::RGB, 3, gl::UNSIGNED_BYTE),
            ImageFormat::BGRA8 => (get_gl_format_bgra(self.gl()), 4, gl::UNSIGNED_BYTE),
            ImageFormat::RG8 => (gl::RG, 2, gl::UNSIGNED_BYTE),
            ImageFormat::RGBAF32 => (gl::RGBA, 16, gl::FLOAT),
            ImageFormat::Invalid => unreachable!(),
        };

        let row_length = match stride {
            Some(value) => value / bpp,
            None => width,
        };

        if let Some(..) = stride {
            self.gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, row_length as gl::GLint);
        }

        self.bind_texture(DEFAULT_TEXTURE, texture);

        match texture.target {
            gl::TEXTURE_2D_ARRAY => {
                self.gl.tex_sub_image_3d_pbo(texture.target,
                                             0,
                                             x0 as gl::GLint,
                                             y0 as gl::GLint,
                                             layer_index,
                                             width as gl::GLint,
                                             height as gl::GLint,
                                             1,
                                             gl_format,
                                             data_type,
                                             offset);
            }
            gl::TEXTURE_2D |
            gl::TEXTURE_RECTANGLE |
            gl::TEXTURE_EXTERNAL_OES => {
                self.gl.tex_sub_image_2d_pbo(texture.target,
                                             0,
                                             x0 as gl::GLint,
                                             y0 as gl::GLint,
                                             width as gl::GLint,
                                             height as gl::GLint,
                                             gl_format,
                                             data_type,
                                             offset);
            }
            _ => panic!("BUG: Unexpected texture target!"),
        }

        // Reset row length to 0, otherwise the stride would apply to all texture uploads.
        if let Some(..) = stride {
            self.gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, 0 as gl::GLint);
        }*/
    }

    pub fn end_frame(&mut self) {
        debug_assert!(self.inside_frame);
        self.inside_frame = false;
        self.frame_id.0 += 1;
    }

    pub fn clear_target(&self,
                        color: Option<[f32; 4]>,
                        depth: Option<f32>) {
        let mut clear_bits = 0;

       /* if let Some(color) = color {
            self.gl.clear_color(color[0], color[1], color[2], color[3]);
            clear_bits |= gl::COLOR_BUFFER_BIT;
        }

        if let Some(depth) = depth {
            self.gl.clear_depth(depth as f64);
            clear_bits |= gl::DEPTH_BUFFER_BIT;
        }

        if clear_bits != 0 {
            self.gl.clear(clear_bits);
        }*/
    }
}