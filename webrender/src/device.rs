/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::shader_source;
use api::{ColorF, ImageDescriptor, ImageFormat};
use api::{DeviceIntPoint, DeviceIntRect, DeviceUintRect, DeviceUintSize};
use api::TextureTarget;
use euclid::Transform3D;
//use gleam::gl;
use internal_types::{FastHashMap, RenderTargetInfo};
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

pub struct Device {
    //gl: Rc<gl::Gl>,
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

impl Device {
    pub fn new(
        //gl: Rc<gl::Gl>,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        //cached_programs: Option<Rc<ProgramCache>>,
    ) -> Device {
        let max_texture_size = 2048u32;
        let renderer_name = "WIP".to_owned();

        let mut extensions = Vec::new();
        /*let extension_count = gl.get_integer_v(gl::NUM_EXTENSIONS) as u32;
        for i in 0 .. extension_count {
            extensions.push(gl.get_string_i(gl::EXTENSIONS, i));
        }*/

        Device {
            //gl,
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
        /*let (bytes_per_pixel, desc) = match format {
            ReadPixelsFormat::Standard(imf) => {
                (imf.bytes_per_pixel(), gl_describe_format(self.gl(), imf))
            }
            ReadPixelsFormat::Rgba8 => {
                (4, FormatDesc {
                    external: gl::RGBA,
                    internal: gl::RGBA8 as _,
                    pixel_type: gl::UNSIGNED_BYTE,
                })
            }
        };
        let size_in_bytes = (bytes_per_pixel * rect.size.width * rect.size.height) as usize;
        assert_eq!(output.len(), size_in_bytes);

        self.gl.flush();
        self.gl.read_pixels_into_buffer(
            rect.origin.x as _,
            rect.origin.y as _,
            rect.size.width as _,
            rect.size.height as _,
            desc.external,
            desc.pixel_type,
            output,
        );*/
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
        &self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<DeviceIntRect>,
    ) {
        let mut clear_bits = 0;

        if let Some(color) = color {
            //self.gl.clear_color(color[0], color[1], color[2], color[3]);
            //clear_bits |= gl::COLOR_BUFFER_BIT;
        }

        if let Some(depth) = depth {
            //debug_assert_ne!(self.gl.get_boolean_v(gl::DEPTH_WRITEMASK), 0);
            //self.gl.clear_depth(depth as f64);
            //clear_bits |= gl::DEPTH_BUFFER_BIT;
        }

        if clear_bits != 0 {
            match rect {
                Some(rect) => {
                    /*self.gl.enable(gl::SCISSOR_TEST);
                    self.gl.scissor(
                        rect.origin.x,
                        rect.origin.y,
                        rect.size.width,
                        rect.size.height,
                    );
                    self.gl.clear(clear_bits);
                    self.gl.disable(gl::SCISSOR_TEST);*/
                }
                None => {
                    //self.gl.clear(clear_bits);
                }
            }
        }
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
