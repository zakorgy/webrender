/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use euclid::Transform3D;
//use self::gl::{DeviceInit, DepthFunction, Stream, TextureUploader, VBO, Texture, ExternalTexture};
use std::convert::From;
use std::marker::Sized;
use super::*;

pub trait DeviceApi
    where Self: Sized,
{
    type CustomVAO;
    //type DepthFunction;
    //type ExternalTexture;
    type FBOId: PartialEq + Eq + Copy + Clone;
    type PBO;
    type Program;
    type RBOId: PartialEq + Eq + Copy + Clone;
    //type Texture;
    type TextureId: PartialEq + Eq + Copy + Clone + From<u32>;
    type VAO;

    fn new(
        init: DeviceInit,
        resource_override_path: Option<PathBuf>,
        upload_method: UploadMethod,
        _file_changed_handler: Box<FileWatcherHandler>,
        cached_programs: Option<Rc<ProgramCache>>,
    ) -> Self;

    fn renderer(&self) -> String;

    fn version(&self) -> String;

    fn get_context(&self) -> &DeviceInit;

    fn set_device_pixel_ratio(&mut self, ratio: f32);

    fn update_program_cache(&mut self, cached_programs: Rc<ProgramCache>);

    fn max_texture_size(&self) -> u32;

    #[cfg(feature = "debug_renderer")]
    fn get_capabilities(&self) -> &Capabilities;

    fn reset_state(&mut self);

    fn begin_frame(&mut self) -> FrameId;

    fn bind_texture<S>(&mut self, sampler: S, texture: &/*Self::*/Texture)
        where
            S: Into<TextureSlot>;

    fn bind_external_texture<S>(&mut self, sampler: S, external_texture: &/*Self::*/ExternalTexture)
        where
            S: Into<TextureSlot>;

    fn bind_read_target_impl(&mut self, fbo_id: Self::FBOId);

    fn bind_read_target(&mut self, texture_and_layer: Option<(&/*Self::*/Texture, i32)>);

    fn bind_draw_target(
        &mut self,
        texture_and_layer: Option<(&/*Self::*/Texture, i32)>,
        dimensions: Option<DeviceUintSize>,
    );

    fn create_fbo_for_external_texture(&mut self, texture_id: u32) -> Self::FBOId;

    fn delete_fbo(&mut self, fbo: Self::FBOId);

    fn bind_external_draw_target(&mut self, fbo_id: Self::FBOId);

    fn bind_program(&mut self, program: &Self::Program);

    fn create_texture(
        &mut self,
        target: TextureTarget,
        format: ImageFormat,
    ) -> /*Self::*/Texture;

    /// Resizes a texture with enabled render target views,
    /// preserves the data by blitting the old texture contents over.
    fn resize_renderable_texture(
        &mut self,
        texture: &mut /*Self::*/Texture,
        new_size: DeviceUintSize,
    );

    fn init_texture<T: Texel>(
        &mut self,
        texture: &mut /*Self::*/Texture,
        width: u32,
        height: u32,
        filter: TextureFilter,
        render_target: Option<RenderTargetInfo>,
        layer_count: i32,
        pixels: Option<&[T]>,
    );

    fn blit_render_target(&mut self, src_rect: DeviceIntRect, dest_rect: DeviceIntRect);

    fn free_texture_storage(&mut self, texture: &mut /*Self::*/Texture);

    fn delete_texture(&mut self, texture: /*Self::*/Texture);

    #[cfg(feature = "replay")]
    fn delete_external_texture(&mut self, mut external: /*Self::*/ExternalTexture);

    fn delete_program(&mut self, program: Self::Program);

    fn create_program(
        &mut self,
        base_filename: &str,
        features: &str,
        descriptor: &VertexDescriptor,
    ) -> Result<Self::Program, ShaderError>;

    fn bind_shader_samplers<S>(&mut self, program: &Self::Program, bindings: &[(&'static str, S)])
        where
            S: Into<TextureSlot> + Copy;

    fn set_uniforms(
        &self,
        program: &Self::Program,
        transform: &Transform3D<f32>,
    );

    fn switch_mode(&self, mode: i32);

    fn create_pbo(&mut self) -> Self::PBO;

    fn delete_pbo(&mut self, pbo: Self::PBO);

    fn upload_texture<'a, T>(
        &'a mut self,
        texture: &'a /*Self::*/Texture,
        pbo: &Self::PBO,
        upload_count: usize,
    ) -> TextureUploader<'a, T>;

    #[cfg(any(feature = "debug_renderer", feature = "capture"))]
    fn read_pixels(&mut self, img_desc: &ImageDescriptor) -> Vec<u8>;

    /// Read rectangle of pixels into the specified output slice.
    fn read_pixels_into(
        &mut self,
        rect: DeviceUintRect,
        format: ReadPixelsFormat,
        output: &mut [u8],
    );

    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    fn attach_read_texture_external(
        &mut self,
        texture_id: Self::TextureId,
        target: TextureTarget,
        layer_id: i32,
    );

    #[cfg(any(feature = "debug_renderer", feature="capture"))]
    fn attach_read_texture(
        &mut self,
        texture: &/*Self::*/Texture,
        layer_id: i32,
    );

    fn bind_vao(&mut self, vao: &Self::VAO);

    fn bind_custom_vao(&mut self, vao: &Self::CustomVAO);

    fn create_custom_vao(
        &mut self,
        streams: &[Stream],
    ) -> Self::CustomVAO;

    fn delete_custom_vao(&mut self, vao: Self::CustomVAO);

    fn create_vbo<V>(&mut self) -> VBO<V>;

    fn delete_vbo<V>(&mut self, vbo: VBO<V>);

    fn create_vao(&mut self, descriptor: &VertexDescriptor) -> Self::VAO;

    fn delete_vao(&mut self, vao: Self::VAO);

    fn allocate_vbo<V>(
        &mut self,
        vbo: &mut VBO<V>,
        count: usize,
        usage_hint: VertexUsageHint,
    );

    fn fill_vbo<V>(
        &mut self,
        vbo: &VBO<V>,
        data: &[V],
        offset: usize,
    );

    fn create_vao_with_new_instances(
        &mut self,
        descriptor: &VertexDescriptor,
        base_vao: &Self::VAO,
    ) -> Self::VAO;

    fn update_vao_main_vertices<V>(
        &mut self,
        vao: &Self::VAO,
        vertices: &[V],
        usage_hint: VertexUsageHint,
    );

    fn update_vao_instances<V>(
        &mut self,
        vao: &Self::VAO,
        instances: &[V],
        usage_hint: VertexUsageHint,
    );

    fn update_vao_indices<I>(
        &mut self,
        vao: &Self::VAO,
        indices: &[I],
        usage_hint: VertexUsageHint,
    );

    fn draw_triangles_u16(&mut self, first_vertex: i32, index_count: i32);

    #[cfg(feature = "debug_renderer")]
    fn draw_triangles_u32(&mut self, first_vertex: i32, index_count: i32);

    fn draw_nonindexed_points(&mut self, first_vertex: i32, vertex_count: i32);

    #[cfg(feature = "debug_renderer")]
    fn draw_nonindexed_lines(&mut self, first_vertex: i32, vertex_count: i32);

    fn draw_indexed_triangles_instanced_u16(
        &mut self,
        index_count: i32,
        instance_count: i32,
    );

    fn end_frame(&mut self);

    fn clear_target(
        &self,
        color: Option<[f32; 4]>,
        depth: Option<f32>,
        rect: Option<DeviceIntRect>,
    );

    fn enable_depth(&self);

    fn disable_depth(&self);

    fn set_depth_func(&self, depth_func: /*Self::*/DepthFunction);

    fn enable_depth_write(&self);

    fn disable_depth_write(&self);

    fn disable_stencil(&self);

    fn set_scissor_rect(&self, rect: DeviceIntRect);

    fn enable_scissor(&self);

    fn disable_scissor(&self);

    fn set_blend(&self, enable: bool);

    fn set_blend_mode_alpha(&self);

    fn set_blend_mode_premultiplied_alpha(&self);

    fn set_blend_mode_premultiplied_dest_out(&self);

    fn set_blend_mode_multiply(&self);

    fn set_blend_mode_max(&self);

    #[cfg(feature = "debug_renderer")]
    fn set_blend_mode_min(&self);

    fn set_blend_mode_subpixel_pass0(&self);

    fn set_blend_mode_subpixel_pass1(&self);

    fn set_blend_mode_subpixel_with_bg_color_pass0(&self);

    fn set_blend_mode_subpixel_with_bg_color_pass1(&self);

    fn set_blend_mode_subpixel_with_bg_color_pass2(&self);

    fn set_blend_mode_subpixel_constant_text_color(&self, color: ColorF);

    fn set_blend_mode_subpixel_dual_source(&self);

    fn supports_extension(&self, extension: &str) -> bool;

    fn echo_driver_messages(&self);
}