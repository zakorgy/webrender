/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use self::device_api::Device as DeviceTrait;
use gleam::gl;
use smallvec::SmallVec;
use std::marker::PhantomData;
use super::*;

const GL_FORMAT_RGBA: gl::GLuint = gl::RGBA;

const GL_FORMAT_BGRA_GL: gl::GLuint = gl::BGRA;

const GL_FORMAT_BGRA_GLES: gl::GLuint = gl::BGRA_EXT;

const SHADER_VERSION_GL: &str = "#version 150\n";
const SHADER_VERSION_GLES: &str = "#version 300 es\n";

// In some places we need to temporarily bind a texture to any slot.
const DEFAULT_TEXTURE: TextureSlot = TextureSlot(0);

#[repr(u32)]
pub enum DepthFunction {
    #[cfg(feature = "debug_renderer")]
    Less = gl::LESS,
    LessEqual = gl::LEQUAL,
}


enum FBOTarget {
    Read,
    Draw,
}

pub fn get_gl_target(target: TextureTarget) -> gl::GLuint {
    match target {
        TextureTarget::Default => gl::TEXTURE_2D,
        TextureTarget::Array => gl::TEXTURE_2D_ARRAY,
        TextureTarget::Rect => gl::TEXTURE_RECTANGLE,
        TextureTarget::External => gl::TEXTURE_EXTERNAL_OES,
    }
}

fn supports_extension(extensions: &[String], extension: &str) -> bool {
    extensions.iter().any(|s| s == extension)
}

fn get_shader_version(gl: &gl::Gl) -> &'static str {
    match gl.get_type() {
        gl::GlType::Gl => SHADER_VERSION_GL,
        gl::GlType::Gles => SHADER_VERSION_GLES,
    }
}

impl VertexAttributeKind {
    fn size_in_bytes(&self) -> u32 {
        match *self {
            VertexAttributeKind::F32 => 4,
            #[cfg(feature = "debug_renderer")]
            VertexAttributeKind::U8Norm => 1,
            VertexAttributeKind::U16Norm => 2,
            VertexAttributeKind::I32 => 4,
            VertexAttributeKind::U16 => 2,
        }
    }
}

impl VertexAttribute {
    fn size_in_bytes(&self) -> u32 {
        self.count * self.kind.size_in_bytes()
    }

    fn bind_to_vao(
        &self,
        attr_index: gl::GLuint,
        divisor: gl::GLuint,
        stride: gl::GLint,
        offset: gl::GLuint,
        gl: &gl::Gl,
    ) {
        gl.enable_vertex_attrib_array(attr_index);
        gl.vertex_attrib_divisor(attr_index, divisor);

        match self.kind {
            VertexAttributeKind::F32 => {
                gl.vertex_attrib_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::FLOAT,
                    false,
                    stride,
                    offset,
                );
            }
            #[cfg(feature = "debug_renderer")]
            VertexAttributeKind::U8Norm => {
                gl.vertex_attrib_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::UNSIGNED_BYTE,
                    true,
                    stride,
                    offset,
                );
            }
            VertexAttributeKind::U16Norm => {
                gl.vertex_attrib_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::UNSIGNED_SHORT,
                    true,
                    stride,
                    offset,
                );
            }
            VertexAttributeKind::I32 => {
                gl.vertex_attrib_i_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::INT,
                    stride,
                    offset,
                );
            }
            VertexAttributeKind::U16 => {
                gl.vertex_attrib_i_pointer(
                    attr_index,
                    self.count as gl::GLint,
                    gl::UNSIGNED_SHORT,
                    stride,
                    offset,
                );
            }
        }
    }
}

impl VertexDescriptor {
    fn instance_stride(&self) -> u32 {
        self.instance_attributes
            .iter()
            .map(|attr| attr.size_in_bytes())
            .sum()
    }

    fn bind_attributes(
        attributes: &[VertexAttribute],
        start_index: usize,
        divisor: u32,
        gl: &gl::Gl,
        vbo: VBOId,
    ) {
        vbo.bind(gl);

        let stride: u32 = attributes
            .iter()
            .map(|attr| attr.size_in_bytes())
            .sum();

        let mut offset = 0;
        for (i, attr) in attributes.iter().enumerate() {
            let attr_index = (start_index + i) as gl::GLuint;
            attr.bind_to_vao(attr_index, divisor, stride as _, offset, gl);
            offset += attr.size_in_bytes();
        }
    }

    fn bind(&self, gl: &gl::Gl, main: VBOId, instance: VBOId) {
        Self::bind_attributes(self.vertex_attributes, 0, 0, gl, main);

        if !self.instance_attributes.is_empty() {
            Self::bind_attributes(
                self.instance_attributes,
                self.vertex_attributes.len(),
                1, gl, instance,
            );
        }
    }
}

impl VBOId {
    fn bind(&self, gl: &gl::Gl) {
        gl.bind_buffer(gl::ARRAY_BUFFER, self.0);
    }
}

impl IBOId {
    fn bind(&self, gl: &gl::Gl) {
        gl.bind_buffer(gl::ELEMENT_ARRAY_BUFFER, self.0);
    }
}

impl FBOId {
    fn bind(&self, gl: &gl::Gl, target: FBOTarget) {
        let target = match target {
            FBOTarget::Read => gl::READ_FRAMEBUFFER,
            FBOTarget::Draw => gl::DRAW_FRAMEBUFFER,
        };
        gl.bind_framebuffer(target, self.0);
    }
}

pub struct Stream<'a> {
    attributes: &'a [VertexAttribute],
    vbo: VBOId,
}

pub struct VBO<V> {
    id: gl::GLuint,
    target: gl::GLenum,
    allocated_count: usize,
    marker: PhantomData<V>,
}

impl<V> VBO<V> {
    pub fn allocated_count(&self) -> usize {
        self.allocated_count
    }

    pub fn stream_with<'a>(&self, attributes: &'a [VertexAttribute]) -> Stream<'a> {
        debug_assert_eq!(
            mem::size_of::<V>(),
            attributes.iter().map(|a| a.size_in_bytes() as usize).sum::<usize>()
        );
        Stream {
            attributes,
            vbo: VBOId(self.id),
        }
    }
}

impl<T> Drop for VBO<T> {
    fn drop(&mut self) {
        debug_assert!(thread::panicking() || self.id == 0);
    }
}

#[cfg_attr(feature = "replay", derive(Clone))]
pub struct ExternalTexture {
    id: gl::GLuint,
    target: gl::GLuint,
}

impl ExternalTexture {
    pub fn new(id: u32, target: TextureTarget) -> Self {
        ExternalTexture {
            id,
            target: get_gl_target(target),
        }
    }

    #[cfg(feature = "replay")]
    pub fn internal_id(&self) -> gl::GLuint {
        self.id
    }
}

pub struct Texture {
    id: gl::GLuint,
    target: gl::GLuint,
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

pub struct Program {
    id: gl::GLuint,
    u_transform: gl::GLint,
    u_device_pixel_ratio: gl::GLint,
    u_mode: gl::GLint,
}

impl Drop for Program {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called"
        );
    }
}

pub struct CustomVAO {
    id: gl::GLuint,
}

impl Drop for CustomVAO {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called"
        );
    }
}

pub struct VAO {
    id: gl::GLuint,
    ibo_id: IBOId,
    main_vbo_id: VBOId,
    instance_vbo_id: VBOId,
    instance_stride: usize,
    owns_vertices_and_indices: bool,
}

impl Drop for VAO {
    fn drop(&mut self) {
        debug_assert!(
            thread::panicking() || self.id == 0,
            "renderer::deinit not called"
        );
    }
}

pub struct PBO {
    id: gl::GLuint,
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
pub struct FBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct RBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
pub struct VBOId(gl::GLuint);

#[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
struct IBOId(gl::GLuint);

impl VertexUsageHint {
    fn to_gl(&self) -> gl::GLuint {
        match *self {
            VertexUsageHint::Static => gl::STATIC_DRAW,
            VertexUsageHint::Dynamic => gl::DYNAMIC_DRAW,
            VertexUsageHint::Stream => gl::STREAM_DRAW,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct UniformLocation(gl::GLint);

impl UniformLocation {
    pub const INVALID: Self = UniformLocation(-1);
}

pub struct Device {
    gl: Rc<gl::Gl>,
    // device state
    bound_textures: [gl::GLuint; 16],
    bound_program: gl::GLuint,
    bound_vao: gl::GLuint,
    bound_read_fbo: FBOId,
    bound_draw_fbo: FBOId,
    program_mode_id: UniformLocation,
    default_read_fbo: gl::GLuint,
    default_draw_fbo: gl::GLuint,

    device_pixel_ratio: f32,
    upload_method: UploadMethod,

    // HW or API capabilities
    #[cfg(feature = "debug_renderer")]
    capabilities: Capabilities,

    bgra_format: gl::GLuint,

    // debug
    inside_frame: bool,

    // resources
    resource_override_path: Option<PathBuf>,

    max_texture_size: u32,
    renderer_name: String,
    cached_programs: Option<Rc<ProgramCache>>,

    // Frame counter. This is used to map between CPU
    // frames and GPU frames.
    frame_id: FrameId,

    // GL extensions
    extensions: Vec<String>,
}

impl Device {
    pub fn new(
        gl: Rc<gl::Gl>,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        cached_programs: Option<Rc<ProgramCache>>,
    ) -> Device {
        let mut max_texture_size = [0];
        unsafe {
            gl.get_integer_v(gl::MAX_TEXTURE_SIZE, &mut max_texture_size);
        }
        let max_texture_size = max_texture_size[0] as u32;
        let renderer_name = gl.get_string(gl::RENDERER);

        let mut extension_count = [0];
        unsafe {
            gl.get_integer_v(gl::NUM_EXTENSIONS, &mut extension_count);
        }
        let extension_count = extension_count[0] as gl::GLuint;
        let mut extensions = Vec::new();
        for i in 0 .. extension_count {
            extensions.push(gl.get_string_i(gl::EXTENSIONS, i));
        }

        let supports_bgra = supports_extension(&extensions, "GL_EXT_texture_format_BGRA8888");
        let bgra_format = match gl.get_type() {
            gl::GlType::Gl => GL_FORMAT_BGRA_GL,
            gl::GlType::Gles => if supports_bgra {
                GL_FORMAT_BGRA_GLES
            } else {
                GL_FORMAT_RGBA
            }
        };

        Device {
            gl,
            resource_override_path,
            // This is initialized to 1 by default, but it is reset
            // at the beginning of each frame in `Renderer::bind_frame_data`.
            device_pixel_ratio: 1.0,
            upload_method,
            inside_frame: false,

            #[cfg(feature = "debug_renderer")]
            capabilities: Capabilities {
                supports_multisampling: false, //TODO
            },

            bgra_format,

            bound_textures: [0; 16],
            bound_program: 0,
            bound_vao: 0,
            bound_read_fbo: FBOId(0),
            bound_draw_fbo: FBOId(0),
            program_mode_id: UniformLocation::INVALID,
            default_read_fbo: 0,
            default_draw_fbo: 0,

            max_texture_size,
            renderer_name,
            cached_programs,
            frame_id: FrameId(0),
            extensions,
        }
    }

    fn gl(&self) -> &gl::Gl {
        &*self.gl
    }

    fn rc_gl(&self) -> &Rc<gl::Gl> {
        &self.gl
    }

    #[cfg(debug_assertions)]
    fn print_shader_errors(source: &str, log: &str) {
        // hacky way to extract the offending lines
        if !log.starts_with("0:") {
            return;
        }
        let end_pos = match log[2..].chars().position(|c| !c.is_digit(10)) {
            Some(pos) => 2 + pos,
            None => return,
        };
        let base_line_number = match log[2 .. end_pos].parse::<usize>() {
            Ok(number) if number >= 2 => number - 2,
            _ => return,
        };
        for (line, prefix) in source.lines().skip(base_line_number).zip(&["|",">","|"]) {
            error!("{}\t{}", prefix, line);
        }
    }

    fn compile_shader(
        gl: &gl::Gl,
        name: &str,
        shader_type: gl::GLenum,
        source: &String,
    ) -> Result<gl::GLuint, ShaderError> {
        debug!("compile {}", name);
        let id = gl.create_shader(shader_type);
        gl.shader_source(id, &[source.as_bytes()]);
        gl.compile_shader(id);
        let log = gl.get_shader_info_log(id);
        let mut status = [0];
        unsafe {
            gl.get_shader_iv(id, gl::COMPILE_STATUS, &mut status);
        }
        if status[0] == 0 {
            error!("Failed to compile shader: {}\n{}", name, log);
            #[cfg(debug_assertions)]
                Self::print_shader_errors(source, &log);
            Err(ShaderError::Compilation(name.to_string(), log))
        } else {
            if !log.is_empty() {
                warn!("Warnings detected on shader: {}\n{}", name, log);
            }
            Ok(id)
        }
    }

    fn bind_texture_impl(&mut self, slot: TextureSlot, id: gl::GLuint, target: gl::GLenum) {
        debug_assert!(self.inside_frame);

        if self.bound_textures[slot.0] != id {
            self.bound_textures[slot.0] = id;
            self.gl.active_texture(gl::TEXTURE0 + slot.0 as gl::GLuint);
            self.gl.bind_texture(target, id);
            self.gl.active_texture(gl::TEXTURE0);
        }
    }

    fn bind_draw_target_impl(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
            fbo_id.bind(self.gl(), FBOTarget::Draw);
        }
    }

    pub fn bind_external_draw_target(&mut self, fbo_id: FBOId) {
        debug_assert!(self.inside_frame);

        if self.bound_draw_fbo != fbo_id {
            self.bound_draw_fbo = fbo_id;
            fbo_id.bind(self.gl(), FBOTarget::Draw);
        }
    }

    fn set_texture_parameters(&mut self, target: gl::GLuint, filter: TextureFilter) {
        let mag_filter = match filter {
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
            .tex_parameter_i(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as gl::GLint);
    }

    /// Updates the render target storage for the texture, creating FBOs as required.
    fn update_target_storage<T: Texel>(
        &mut self,
        texture: &mut Texture,
        rt_info: &RenderTargetInfo,
        is_resized: bool,
        pixels: Option<&[T]>,
    ) {
        assert!(texture.layer_count > 0 || texture.width + texture.height == 0);

        let needed_layer_count = texture.layer_count - texture.fbo_ids.len() as i32;
        let allocate_color = needed_layer_count != 0 || is_resized || pixels.is_some();

        if allocate_color {
            let desc = self.gl_describe_format(texture.format);
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
                        pixels.map(texels_to_u8_slice),
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
                        pixels.map(texels_to_u8_slice),
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
        }
    }

    fn update_texture_storage<T: Texel>(&mut self, texture: &Texture, pixels: Option<&[T]>) {
        let desc = self.gl_describe_format(texture.format);
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
                    pixels.map(texels_to_u8_slice),
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
                    pixels.map(texels_to_u8_slice),
                );
            }
            _ => panic!("BUG: Unexpected texture target!"),
        }
    }

    fn free_texture_storage_impl(&mut self, target: gl::GLenum, desc: FormatDesc) {
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
    }

    #[cfg(feature = "debug_renderer")]
    pub fn get_uniform_location(&self, program: &Program, name: &str) -> UniformLocation {
        UniformLocation(self.gl.get_uniform_location(program.id, name))
    }

    /// Get texels of a texture into the specified output slice.
    #[cfg(feature = "debug_renderer")]
    pub fn get_tex_image_into(
        &mut self,
        texture: &Texture,
        format: ImageFormat,
        output: &mut [u8],
    ) {
        self.bind_texture(DEFAULT_TEXTURE, texture);
        let desc = self.gl_describe_format(format);
        self.gl.get_tex_image_into_buffer(
            texture.target,
            0,
            desc.external,
            desc.pixel_type,
            output,
        );
    }

    /// Attaches the provided texture to the current Read FBO binding.
    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    fn attach_read_texture_raw(
        &mut self, texture_id: gl::GLuint, target: gl::GLuint, layer_id: i32
    ) {
        match target {
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
        }
    }

    fn bind_vao_impl(&mut self, id: gl::GLuint) {
        debug_assert!(self.inside_frame);

        if self.bound_vao != id {
            self.bound_vao = id;
            self.gl.bind_vertex_array(id);
        }
    }

    fn create_vao_with_vbos(
        &mut self,
        descriptor: &VertexDescriptor,
        main_vbo_id: VBOId,
        instance_vbo_id: VBOId,
        ibo_id: IBOId,
        owns_vertices_and_indices: bool,
    ) -> VAO {
        debug_assert!(self.inside_frame);

        let instance_stride = descriptor.instance_stride() as usize;
        let vao_id = self.gl.gen_vertex_arrays(1)[0];

        self.gl.bind_vertex_array(vao_id);

        descriptor.bind(self.gl(), main_vbo_id, instance_vbo_id);
        ibo_id.bind(self.gl()); // force it to be a part of VAO

        self.gl.bind_vertex_array(0);

        VAO {
            id: vao_id,
            ibo_id,
            main_vbo_id,
            instance_vbo_id,
            instance_stride,
            owns_vertices_and_indices,
        }
    }

    fn update_vbo_data<V>(
        &mut self,
        vbo: VBOId,
        vertices: &[V],
        usage_hint: VertexUsageHint,
    ) {
        debug_assert!(self.inside_frame);

        vbo.bind(self.gl());
        gl::buffer_data(self.gl(), gl::ARRAY_BUFFER, vertices, usage_hint.to_gl());
    }

    fn gl_describe_format(&self, format: ImageFormat) -> FormatDesc {
        match format {
            ImageFormat::R8 => FormatDesc {
                internal: gl::RED as _,
                external: gl::RED,
                pixel_type: gl::UNSIGNED_BYTE,
            },
            ImageFormat::BGRA8 => {
                let external = self.bgra_format;
                FormatDesc {
                    internal: match self.gl.get_type() {
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
    }
}

struct FormatDesc {
    internal: gl::GLint,
    external: gl::GLuint,
    pixel_type: gl::GLuint,
}

struct UploadChunk {
    rect: DeviceUintRect,
    layer_index: i32,
    stride: Option<u32>,
    offset: usize,
}

struct PixelBuffer {
    usage: gl::GLenum,
    size_allocated: usize,
    size_used: usize,
    // small vector avoids heap allocation for a single chunk
    chunks: SmallVec<[UploadChunk; 1]>,
}

impl PixelBuffer {
    fn new(
        usage: gl::GLenum,
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
    gl: &'a gl::Gl,
    bgra_format: gl::GLuint,
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
            self.target.gl.bind_buffer(gl::PIXEL_UNPACK_BUFFER, 0);
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
                    gl::buffer_data(
                        self.target.gl,
                        gl::PIXEL_UNPACK_BUFFER,
                        data,
                        buffer.usage,
                    );
                    buffer.size_allocated = upload_size;
                } else {
                    gl::buffer_sub_data(
                        self.target.gl,
                        gl::PIXEL_UNPACK_BUFFER,
                        buffer.size_used as _,
                        data,
                    );
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
        let (gl_format, bpp, data_type) = match self.texture.format {
            ImageFormat::R8 => (gl::RED, 1, gl::UNSIGNED_BYTE),
            ImageFormat::BGRA8 => (self.bgra_format, 4, gl::UNSIGNED_BYTE),
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

        // If using tri-linear filtering, build the mip-map chain for this texture.
        if self.texture.filter == TextureFilter::Trilinear {
            self.gl.generate_mipmap(self.texture.target);
        }

        // Reset row length to 0, otherwise the stride would apply to all texture uploads.
        if chunk.stride.is_some() {
            self.gl.pixel_store_i(gl::UNPACK_ROW_LENGTH, 0 as _);
        }
    }
}

impl<'a, 'b, T, V> DeviceTrait<'a, 'b, T, V> for Device {
    type DeviceInit = Rc<gl::Gl>;
    type ExternalTexture = ExternalTexture;
    type FBOId = FBOId;
    type RBOId = RBOId;
    type Program = Program;
    type PBO = PBO;
    type TextureUploader = TextureUploader<'a, T>;
    type VAO = VAO;
    type CustomVAO = CustomVAO;
    type Stream = Stream<'b>;
    type VBO = VBO<V>;
    type DepthFunction = DepthFunction;
    type Texture = Texture;

    fn new(
        init: Self::DeviceInit,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        cached_programs: Option<Rc<ProgramCache>>,
    ) -> Self {
        Device::new(
            init,
            resource_override_path,
            upload_method,
            _file_changed_handler,
            cached_programs
        )
    }
}