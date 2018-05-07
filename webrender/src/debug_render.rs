/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{ColorU, DeviceIntRect, DeviceUintSize, ImageFormat, TextureTarget};
use debug_font_data;
use device::{Device, ProgramId, Texture, TextureSlot};
use device::{PipelineRequirements, ShaderKind, TextureFilter};
use euclid::{Point2D, Rect, Size2D, Transform3D};
use hal;
use internal_types::{ORTHO_FAR_PLANE, ORTHO_NEAR_PLANE};
use ron::de::from_reader;
use std::collections::HashMap;
use std::f32;
use std::fs::File;

#[derive(Debug, Copy, Clone)]
enum DebugSampler {
    Font,
}

impl Into<TextureSlot> for DebugSampler {
    fn into(self) -> TextureSlot {
        match self {
            DebugSampler::Font => TextureSlot(0),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DebugFontVertex {
    pub x: f32,
    pub y: f32,
    z: f32,
    pub color: ColorU,
    pub u: f32,
    pub v: f32,
}

impl DebugFontVertex {
    pub fn new(x: f32, y: f32, u: f32, v: f32, color: ColorU) -> DebugFontVertex {
        DebugFontVertex { x, y, z: 0.0, color, u, v }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct DebugColorVertex {
    pub x: f32,
    pub y: f32,
    z: f32,
    pub color: ColorU,
}

impl DebugColorVertex {
    pub fn new(x: f32, y: f32, color: ColorU) -> DebugColorVertex {
        DebugColorVertex { x, y, z: 0.0, color }
    }
}

pub struct DebugRenderer {
    font_vertices: Vec<DebugFontVertex>,
    font_indices: Vec<u32>,
    font_program: ProgramId,
    font_texture: Texture,

    tri_vertices: Vec<DebugColorVertex>,
    tri_indices: Vec<u32>,
    line_vertices: Vec<DebugColorVertex>,
    color_program: ProgramId,
}

impl DebugRenderer {
    pub fn new<B: hal::Backend>(device: &mut Device<B>) -> Self {
        let file =
            File::open(concat!(env!("OUT_DIR"), "/shader_bindings.ron")).expect("Unable to open the file");
        let mut pipeline_requirements: HashMap<String, PipelineRequirements> =
            from_reader(file).expect("Failed to load shader_bindings.ron");

        let pipeline_requirement =
            pipeline_requirements
                .remove("debug_font")
                .expect("Pipeline requirements not found for debug_font");

        let font_program =
            device.create_program(
                pipeline_requirement,
                "debug_font",
                &ShaderKind::DebugFont,
            );

        let pipeline_requirement_color =
            pipeline_requirements
                .remove("debug_color")
                .expect("Pipeline requirements not found for debug_color");

        let color_program = device
            .create_program(
                pipeline_requirement_color,
                "debug_color",
                &ShaderKind::DebugColor
            );

        let mut font_texture = device.create_texture(TextureTarget::Array, ImageFormat::R8);
        device.init_texture(
            &mut font_texture,
            debug_font_data::BMP_WIDTH,
            debug_font_data::BMP_HEIGHT,
            TextureFilter::Linear,
            None,
            1,
            Some(&debug_font_data::FONT_BITMAP),
        );

        DebugRenderer {
            font_vertices: Vec::new(),
            font_indices: Vec::new(),
            line_vertices: Vec::new(),
            tri_vertices: Vec::new(),
            tri_indices: Vec::new(),
            font_program,
            color_program,
            font_texture,
        }
    }

    pub fn deinit<B: hal::Backend>(self, device: &mut Device<B>) {
        device.delete_texture(self.font_texture);
        device.delete_program(self.font_program);
        device.delete_program(self.color_program);
    }

    pub fn line_height(&self) -> f32 {
        debug_font_data::FONT_SIZE as f32 * 1.1
    }

    pub fn add_text(&mut self, x: f32, y: f32, text: &str, color: ColorU) -> Rect<f32> {
        let mut x_start = x;
        let ipw = 1.0 / debug_font_data::BMP_WIDTH as f32;
        let iph = 1.0 / debug_font_data::BMP_HEIGHT as f32;

        let mut min_x = f32::MAX;
        let mut max_x = -f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_y = -f32::MAX;

        for c in text.chars() {
            let c = c as usize - debug_font_data::FIRST_GLYPH_INDEX as usize;
            if c < debug_font_data::GLYPHS.len() {
                let glyph = &debug_font_data::GLYPHS[c];

                let x0 = (x_start + glyph.xo + 0.5).floor();
                let y0 = (y + glyph.yo + 0.5).floor();

                let x1 = x0 + glyph.x1 as f32 - glyph.x0 as f32;
                let y1 = y0 + glyph.y1 as f32 - glyph.y0 as f32;

                let s0 = glyph.x0 as f32 * ipw;
                let t0 = glyph.y0 as f32 * iph;
                let s1 = glyph.x1 as f32 * ipw;
                let t1 = glyph.y1 as f32 * iph;

                x_start += glyph.xa;

                let vertex_count = self.font_vertices.len() as u32;

                self.font_vertices
                    .push(DebugFontVertex::new(x0, y0, s0, t0, color));
                self.font_vertices
                    .push(DebugFontVertex::new(x1, y0, s1, t0, color));
                self.font_vertices
                    .push(DebugFontVertex::new(x0, y1, s0, t1, color));
                self.font_vertices
                    .push(DebugFontVertex::new(x1, y1, s1, t1, color));

                self.font_indices.push(vertex_count + 0);
                self.font_indices.push(vertex_count + 1);
                self.font_indices.push(vertex_count + 2);
                self.font_indices.push(vertex_count + 2);
                self.font_indices.push(vertex_count + 1);
                self.font_indices.push(vertex_count + 3);

                min_x = min_x.min(x0);
                max_x = max_x.max(x1);
                min_y = min_y.min(y0);
                max_y = max_y.max(y1);
            }
        }

        Rect::new(
            Point2D::new(min_x, min_y),
            Size2D::new(max_x - min_x, max_y - min_y),
        )
    }

    pub fn add_quad(
        &mut self,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        color_top: ColorU,
        color_bottom: ColorU,
    ) {
        let vertex_count = self.tri_vertices.len() as u32;

        self.tri_vertices
            .push(DebugColorVertex::new(x0, y0, color_top));
        self.tri_vertices
            .push(DebugColorVertex::new(x1, y0, color_top));
        self.tri_vertices
            .push(DebugColorVertex::new(x0, y1, color_bottom));
        self.tri_vertices
            .push(DebugColorVertex::new(x1, y1, color_bottom));

        self.tri_indices.push(vertex_count + 0);
        self.tri_indices.push(vertex_count + 1);
        self.tri_indices.push(vertex_count + 2);
        self.tri_indices.push(vertex_count + 2);
        self.tri_indices.push(vertex_count + 1);
        self.tri_indices.push(vertex_count + 3);
    }

    #[allow(dead_code)]
    pub fn add_line(&mut self, x0: i32, y0: i32, color0: ColorU, x1: i32, y1: i32, color1: ColorU) {
        self.line_vertices
            .push(DebugColorVertex::new(x0 as f32, y0 as f32, color0));
        self.line_vertices
            .push(DebugColorVertex::new(x1 as f32, y1 as f32, color1));
    }


    pub fn add_rect(&mut self, rect: &DeviceIntRect, color: ColorU) {
        let p0 = rect.origin;
        let p1 = p0 + rect.size;
        self.add_line(p0.x, p0.y, color, p1.x, p0.y, color);
        self.add_line(p1.x, p0.y, color, p1.x, p1.y, color);
        self.add_line(p1.x, p1.y, color, p0.x, p1.y, color);
        self.add_line(p0.x, p1.y, color, p0.x, p0.y, color);
    }

    pub fn render<B: hal::Backend>(
        &mut self,
        device: &mut Device<B>,
        viewport_size: Option<DeviceUintSize>,
    ) {
        if let Some(viewport_size) = viewport_size {
            device.disable_depth();
            device.set_blend(true);
            device.set_blend_mode_premultiplied_alpha();

            let projection = Transform3D::ortho(
                0.0,
                viewport_size.width as f32,
                0.0,
                viewport_size.height as f32,
                ORTHO_NEAR_PLANE,
                ORTHO_FAR_PLANE,
            );

            // Triangles
            if !self.tri_vertices.is_empty() {
                device.bind_program(self.color_program);
                device.set_uniforms(&projection);
                device.update_indices(self.tri_indices.as_slice());
                device.update_vertices(&self.tri_vertices);
                device.draw();
            }

            // Lines
            if !self.line_vertices.is_empty() {
                device.bind_program(self.color_program);
                device.set_uniforms(&projection);
                device.update_vertices(&self.line_vertices);
                device.draw();
            }

            // Glyph
            if !self.font_indices.is_empty() {
                device.bind_program(self.font_program);
                device.set_uniforms(&projection);
                device.bind_texture(DebugSampler::Font, &self.font_texture);
                device.update_indices(self.font_indices.as_slice());
                device.update_vertices(&self.font_vertices);
                device.bind_textures();
                device.draw();
            }
        }

        self.font_indices.clear();
        self.font_vertices.clear();
        self.line_vertices.clear();
        self.tri_vertices.clear();
        self.tri_indices.clear();
    }
}
