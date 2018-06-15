/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use api::{
    YUV_COLOR_SPACES, YUV_FORMATS,
    YuvColorSpace, YuvFormat,
};
use batch::{BatchKey, BatchKind, BrushBatchKind, TransformBatchKind};
use device::{Device, ProgramId, ShaderError, ShaderKind};
use device::{PipelineRequirements, VertexArrayKind};
use euclid::{Transform3D};
use glyph_rasterizer::GlyphFormat;
use hal;
use renderer::{
    //desc,
    //MAX_VERTEX_TEXTURE_WIDTH,
    BlendMode, ImageBufferKind, RendererError, RendererOptions,
    //TextureSampler,
};
use ron::de::from_reader;
use std::collections::HashMap;
use std::fs::File;
use util::TransformedRectKind;

//use gleam::gl::GlType;
//use time::precise_time_ns;


impl ImageBufferKind {
    pub(crate) fn _get_feature_string(&self) -> &'static str {
        match *self {
            ImageBufferKind::Texture2D => "TEXTURE_2D",
            ImageBufferKind::Texture2DArray => "",
            ImageBufferKind::TextureRect => "TEXTURE_RECT",
            ImageBufferKind::TextureExternal => "TEXTURE_EXTERNAL",
        }
    }

    /*fn has_platform_support(&self, gl_type: &GlType) -> bool {
        match (*self, gl_type) {
            (ImageBufferKind::Texture2D, _) => true,
            (ImageBufferKind::Texture2DArray, _) => true,
            (ImageBufferKind::TextureRect, _) => true,
            (ImageBufferKind::TextureExternal, &GlType::Gles) => true,
            (ImageBufferKind::TextureExternal, &GlType::Gl) => false,
        }
    }*/
}

pub const _IMAGE_BUFFER_KINDS: [ImageBufferKind; 4] = [
    ImageBufferKind::Texture2D,
    ImageBufferKind::TextureRect,
    ImageBufferKind::TextureExternal,
    ImageBufferKind::Texture2DArray,
];

const _ALPHA_FEATURE: &str = "ALPHA_PASS";
const _DITHERING_FEATURE: &str = "DITHERING";
const _DUAL_SOURCE_FEATURE: &str = "DUAL_SOURCE_BLENDING";

pub struct LazilyCompiledShader {
    program: Option<ProgramId>,
    name: &'static str,
    kind: ShaderKind,
    pipeline_requirements: PipelineRequirements,
}

impl LazilyCompiledShader {
    pub(crate) fn new<B: hal::Backend>(
        kind: ShaderKind,
        name: &'static str,
        pipeline_requirements: &mut HashMap<String, PipelineRequirements>,
        _device: &mut Device<B>,
        precache: bool,
    ) -> Result<Self, ShaderError> {
        let pipeline_requirements =
            pipeline_requirements.remove(name).expect(&format!("Pipeline requirements not found for: {}", name));
        let shader = LazilyCompiledShader {
            program: None,
            name,
            kind,
            pipeline_requirements,
        };

        if precache {
            // TODO
            /*let t0 = precise_time_ns();
            let program = shader.get(device)?;
            let t1 = precise_time_ns();
            device.bind_program(program);
            device.draw_triangles_u16(0, 3);
            let t2 = precise_time_ns();
            debug!("[C: {:.1} ms D: {:.1} ms] Precache {} {:?}",
                (t1 - t0) as f64 / 1000000.0,
                (t2 - t1) as f64 / 1000000.0,
                name,
                features
            );*/
        }

        Ok(shader)
    }

    pub fn bind<B: hal::Backend>(
        &mut self,
        device: &mut Device<B>,
        projection: &Transform3D<f32>,
        renderer_errors: &mut Vec<RendererError>,
    ) {
        let program = match self.get(device) {
            Ok(program) => program,
            Err(e) => {
                renderer_errors.push(RendererError::from(e));
                return;
            }
        };
        device.bind_program(program);
        device.set_uniforms(&program, projection);
    }

    fn get<B: hal::Backend>(&mut self, device: &mut Device<B>) -> Result<ProgramId, ShaderError> {
        if self.program.is_none() {
            let program = device.create_program(
                self.pipeline_requirements.clone(),
                self.name,
                &self.kind,
            );
            self.program = Some(program);
        }

        Ok(self.program.unwrap())
    }

    fn deinit<B: hal::Backend>(self, device: &mut Device<B>) {
        if let Some(program) = self.program {
            device.delete_program(program);
        }
    }
}

// A brush shader supports two modes:
// opaque:
//   Used for completely opaque primitives,
//   or inside segments of partially
//   opaque primitives. Assumes no need
//   for clip masks, AA etc.
// alpha:
//   Used for brush primitives in the alpha
//   pass. Assumes that AA should be applied
//   along the primitive edge, and also that
//   clip mask is present.
struct BrushShader {
    opaque: LazilyCompiledShader,
    alpha: LazilyCompiledShader,
    dual_source: Option<LazilyCompiledShader>,
}

impl BrushShader {
    fn new<B: hal::Backend>(
        name: &'static str,
        alpha_name: &'static str,
        dual_source_name: &'static str,
        pipeline_requirements: &mut HashMap<String, PipelineRequirements>,
        device: &mut Device<B>,
        precache: bool,
        dual_source: bool,
    ) -> Result<Self, ShaderError> {
        let opaque = LazilyCompiledShader::new(
            ShaderKind::Brush,
            name,
            pipeline_requirements,
            device,
            precache,
        )?;

        let alpha = LazilyCompiledShader::new(
            ShaderKind::Brush,
            alpha_name,
            pipeline_requirements,
            device,
            precache,
        )?;

        let dual_source = if dual_source {
            let shader = LazilyCompiledShader::new(
                ShaderKind::Brush,
                dual_source_name,
                pipeline_requirements,
                device,
                precache,
            )?;

            Some(shader)
        } else {
            None
        };

        Ok(BrushShader {
            opaque,
            alpha,
            dual_source,
        })
    }

    fn get(&mut self, blend_mode: BlendMode) -> &mut LazilyCompiledShader {
        match blend_mode {
            BlendMode::None => &mut self.opaque,
            BlendMode::Alpha |
            BlendMode::PremultipliedAlpha |
            BlendMode::PremultipliedDestOut |
            BlendMode::SubpixelConstantTextColor(..) |
            BlendMode::SubpixelWithBgColor => &mut self.alpha,
            BlendMode::SubpixelDualSource => {
                self.dual_source
                    .as_mut()
                    .expect("bug: no dual source shader loaded")
            }
        }
    }

    fn deinit<B: hal::Backend>(self, device: &mut Device<B>) {
        self.opaque.deinit(device);
        self.alpha.deinit(device);
        if let Some(dual_source) = self.dual_source {
            dual_source.deinit(device);
        }
    }
}

pub struct TextShader {
    simple: LazilyCompiledShader,
    transform: LazilyCompiledShader,
    glyph_transform: LazilyCompiledShader,
}

impl TextShader {
    fn new<B: hal::Backend>(
        name: &'static str,
        transform_name: &'static str,
        glyph_transform_name: &'static str,
        pipeline_requirements: &mut HashMap<String, PipelineRequirements>,
        device: &mut Device<B>,
        precache: bool,
    ) -> Result<Self, ShaderError> {
        let simple = LazilyCompiledShader::new(
            ShaderKind::Text,
            name,
            pipeline_requirements,
            device,
            precache,
        )?;

        let transform = LazilyCompiledShader::new(
            ShaderKind::Text,
            transform_name,
            pipeline_requirements,
            device,
            precache,
        )?;

        let glyph_transform = LazilyCompiledShader::new(
            ShaderKind::Text,
            glyph_transform_name,
            pipeline_requirements,
            device,
            precache,
        )?;

        Ok(TextShader { simple, transform, glyph_transform })
    }

    pub fn get(
        &mut self,
        glyph_format: GlyphFormat,
        transform_kind: TransformedRectKind,
    ) -> &mut LazilyCompiledShader {
        match glyph_format {
            GlyphFormat::Alpha |
            GlyphFormat::Subpixel |
            GlyphFormat::Bitmap |
            GlyphFormat::ColorBitmap => match transform_kind {
                TransformedRectKind::AxisAligned => &mut self.simple,
                TransformedRectKind::Complex => &mut self.transform,
            }
            GlyphFormat::TransformedAlpha |
            GlyphFormat::TransformedSubpixel => &mut self.glyph_transform,
        }
    }

    fn deinit<B: hal::Backend>(self, device: &mut Device<B>) {
        self.simple.deinit(device);
        self.transform.deinit(device);
        self.glyph_transform.deinit(device);
    }
}

pub struct Shaders {
    // These are "cache shaders". These shaders are used to
    // draw intermediate results to cache targets. The results
    // of these shaders are then used by the primitive shaders.
    pub cs_blur_a8: LazilyCompiledShader,
    pub cs_blur_rgba8: LazilyCompiledShader,
    pub cs_border_segment: LazilyCompiledShader,

    // Brush shaders
    brush_solid: BrushShader,
    //brush_image: Vec<Option<BrushShader<B>>>,
    brush_image: BrushShader,
    brush_blend: BrushShader,
    brush_mix_blend: BrushShader,
    //brush_yuv_image: Vec<Option<BrushShader<B>>>,
    brush_yuv_image: Vec<BrushShader>,
    brush_radial_gradient: BrushShader,
    brush_linear_gradient: BrushShader,

    /// These are "cache clip shaders". These shaders are used to
    /// draw clip instances into the cached clip mask. The results
    /// of these shaders are also used by the primitive shaders.
    pub cs_clip_rectangle: LazilyCompiledShader,
    pub cs_clip_box_shadow: LazilyCompiledShader,
    pub cs_clip_image: LazilyCompiledShader,
    pub cs_clip_line: LazilyCompiledShader,

    // The are "primitive shaders". These shaders draw and blend
    // final results on screen. They are aware of tile boundaries.
    // Most draw directly to the framebuffer, but some use inputs
    // from the cache shaders to draw. Specifically, the box
    // shadow primitive shader stretches the box shadow cache
    // output, and the cache_image shader blits the results of
    // a cache shader (e.g. blur) to the screen.
    pub ps_text_run: TextShader,
    pub ps_text_run_dual_source: TextShader,

    ps_split_composite: LazilyCompiledShader,
}

impl Shaders {
    pub fn new<B: hal::Backend>(
        device: &mut Device<B>,
        options: &RendererOptions,
    ) -> Result<Self, ShaderError> {
        let file =
            File::open(concat!(env!("OUT_DIR"), "/shader_bindings.ron")).expect("Unable to open the file");
        let mut pipeline_requirements: HashMap<String, PipelineRequirements> =
            from_reader(file).expect("Failed to load shader_bindings.ron");

        let brush_solid = BrushShader::new(
            "brush_solid",
            "brush_solid_alpha_pass",
            "brush_solid_alpha_pass_dual_source_blending",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
            false,
        )?;

        // We only support one type of image shaders for now.
        let brush_image = BrushShader::new(
            "brush_image",
            "brush_image_alpha_pass",
            "brush_image_alpha_pass_dual_source_blending",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
            true,
        )?;

        let brush_blend = BrushShader::new(
            "brush_blend",
            "brush_blend_alpha_pass",
            "brush_blend_alpha_pass_dual_source_blending",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
            false,
        )?;

        let brush_mix_blend = BrushShader::new(
            "brush_mix_blend",
            "brush_mix_blend_alpha_pass",
            "brush_mix_blend_alpha_pass_dual_source_blending",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
            false,
        )?;

        let brush_radial_gradient = BrushShader::new(
            if options.enable_dithering {
                "brush_radial_gradient_dithering"
            } else {
                "brush_radial_gradient"
            },
            if options.enable_dithering {
                "brush_radial_gradient_dithering_alpha_pass"
            } else {
                "brush_radial_gradient_alpha_pass"
            },
            if options.enable_dithering {
                "brush_radial_gradient_dual_source_dithering"
            } else {
                "brush_radial_gradient_alpha_pass_dual_source_blending"
            },
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
            false,
        )?;

        let brush_linear_gradient = BrushShader::new(
            if options.enable_dithering {
                "brush_linear_gradient_dithering"
            } else {
                "brush_linear_gradient"
            },
            if options.enable_dithering {
                "brush_linear_gradient_dithering_alpha_pass"
            } else {
                "brush_linear_gradient_alpha_pass"
            },
            if options.enable_dithering {
                "brush_linear_gradient_dual_source_dithering"
            } else {
                "brush_linear_gradient_alpha_pass_dual_source_blending"
            },
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
            false,
        )?;

        let cs_blur_a8 = LazilyCompiledShader::new(
            ShaderKind::Cache(VertexArrayKind::Blur),
            "cs_blur_alpha_target",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        let cs_blur_rgba8 = LazilyCompiledShader::new(
            ShaderKind::Cache(VertexArrayKind::Blur),
            "cs_blur_color_target",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        let cs_clip_rectangle = LazilyCompiledShader::new(
            ShaderKind::ClipCache,
            "cs_clip_rectangle_transform",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        let cs_clip_box_shadow = LazilyCompiledShader::new(
            ShaderKind::ClipCache,
            "cs_clip_box_shadow_transform",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        let cs_clip_line = LazilyCompiledShader::new(
            ShaderKind::ClipCache,
            "cs_clip_line_transform",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        let cs_clip_image = LazilyCompiledShader::new(
            ShaderKind::ClipCache,
            "cs_clip_image_transform",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        let ps_text_run = TextShader::new(
            "ps_text_run",
            "ps_text_run_transform",
            "ps_text_run_glyph_transform",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        let ps_text_run_dual_source = TextShader::new(
            "ps_text_run_dual_source_blending",
            "ps_text_run_dual_source_blending_transform",
            "ps_text_run_dual_source_blending_glyph_transform",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        // All yuv_image configuration.
        let brush_yuv_image = vec![
            BrushShader::new(
                "brush_yuv_image_yuv_nv12_yuv_rec601",
                "brush_yuv_image_yuv_nv12_yuv_rec601_alpha_pass",
                "brush_yuv_image_yuv_nv12_yuv_rec601_alpha_pass_dual_source_blending",
                &mut pipeline_requirements,
                device,
                options.precache_shaders,
                false,
            )?,
            BrushShader::new(
                "brush_yuv_image_yuv_nv12_yuv_rec709",
                "brush_yuv_image_yuv_nv12_yuv_rec709_alpha_pass",
                "brush_yuv_image_yuv_nv12_yuv_rec709_alpha_pass_dual_source_blending",
                &mut pipeline_requirements,
                device,
                options.precache_shaders,
                false,
            )?,
            BrushShader::new(
                "brush_yuv_image_yuv_planar_yuv_rec601",
                "brush_yuv_image_yuv_planar_yuv_rec601_alpha_pass",
                "brush_yuv_image_yuv_planar_yuv_rec601_alpha_pass_dual_source_blending",
                &mut pipeline_requirements,
                device,
                options.precache_shaders,
                false,
            )?,
            BrushShader::new(
                "brush_yuv_image_yuv_planar_yuv_rec709",
                "brush_yuv_image_yuv_planar_yuv_rec709_alpha_pass",
                "brush_yuv_image_yuv_planar_yuv_rec709_alpha_pass_dual_source_blending",
                &mut pipeline_requirements,
                device,
                options.precache_shaders,
                false,
            )?,
            BrushShader::new(
                "brush_yuv_image_yuv_interleaved_yuv_rec601",
                "brush_yuv_image_yuv_interleaved_yuv_rec601_alpha_pass",
                "brush_yuv_image_yuv_interleaved_yuv_rec601_alpha_pass_dual_source_blending",
                &mut pipeline_requirements,
                device,
                options.precache_shaders,
                false,
            )?,
            BrushShader::new(
                "brush_yuv_image_yuv_interleaved_yuv_rec709",
                "brush_yuv_image_yuv_interleaved_yuv_rec709_alpha_pass",
                "brush_yuv_image_yuv_interleaved_yuv_rec709_alpha_pass_dual_source_blending",
                &mut pipeline_requirements,
                device,
                options.precache_shaders,
                false,
            )?,
        ];

        let cs_border_segment = LazilyCompiledShader::new(
            ShaderKind::Cache(VertexArrayKind::Border),
            "cs_border_segment",
            &mut pipeline_requirements,
             device,
             options.precache_shaders,
        )?;

        let ps_split_composite = LazilyCompiledShader::new(
            ShaderKind::Primitive,
            "ps_split_composite",
            &mut pipeline_requirements,
            device,
            options.precache_shaders,
        )?;

        Ok(Shaders {
            cs_blur_a8,
            cs_blur_rgba8,
            cs_border_segment,
            brush_solid,
            brush_image,
            brush_blend,
            brush_mix_blend,
            brush_yuv_image,
            brush_radial_gradient,
            brush_linear_gradient,
            cs_clip_rectangle,
            cs_clip_box_shadow,
            cs_clip_image,
            cs_clip_line,
            ps_text_run,
            ps_text_run_dual_source,
            ps_split_composite,
        })
    }

    fn get_yuv_shader_index(
        buffer_kind: ImageBufferKind,
        format: YuvFormat,
        color_space: YuvColorSpace,
    ) -> usize {
        ((buffer_kind as usize) * YUV_FORMATS.len() + (format as usize)) * YUV_COLOR_SPACES.len() +
            (color_space as usize)
    }

    pub fn get(&mut self, key: &BatchKey) -> &mut LazilyCompiledShader {
        match key.kind {
            BatchKind::SplitComposite => {
                &mut self.ps_split_composite
            }
            BatchKind::Brush(brush_kind) => {
                let brush_shader = match brush_kind {
                    BrushBatchKind::Solid => {
                        &mut self.brush_solid
                    }
                    BrushBatchKind::Image(_image_buffer_kind) => {
                        &mut self.brush_image/*[image_buffer_kind as usize]
                            .as_mut()
                            .expect("Unsupported image shader kind")*/
                    }
                    BrushBatchKind::Blend => {
                        &mut self.brush_blend
                    }
                    BrushBatchKind::MixBlend { .. } => {
                        &mut self.brush_mix_blend
                    }
                    BrushBatchKind::RadialGradient => {
                        &mut self.brush_radial_gradient
                    }
                    BrushBatchKind::LinearGradient => {
                        &mut self.brush_linear_gradient
                    }
                    BrushBatchKind::YuvImage(image_buffer_kind, format, color_space) => {
                        let shader_index =
                            Self::get_yuv_shader_index(image_buffer_kind, format, color_space)
                                % self.brush_yuv_image.len();
                        &mut self.brush_yuv_image[shader_index]
                            /*.as_mut()
                            .expect("Unsupported YUV shader kind")*/
                    }
                };
                brush_shader.get(key.blend_mode)
            }
            BatchKind::Transformable(transform_kind, batch_kind) => {
                match batch_kind {
                    TransformBatchKind::TextRun(glyph_format) => {
                        let text_shader = match key.blend_mode {
                            BlendMode::SubpixelDualSource => {
                                &mut self.ps_text_run_dual_source
                            }
                            _ => {
                                &mut self.ps_text_run
                            }
                        };
                        return text_shader.get(glyph_format, transform_kind);
                    }
                }
            }
        }
    }

    pub fn deinit<B: hal::Backend>(self, device: &mut Device<B>) {
        self.cs_blur_a8.deinit(device);
        self.cs_blur_rgba8.deinit(device);
        self.brush_solid.deinit(device);
        self.brush_blend.deinit(device);
        self.brush_mix_blend.deinit(device);
        self.brush_radial_gradient.deinit(device);
        self.brush_linear_gradient.deinit(device);
        self.cs_clip_rectangle.deinit(device);
        self.cs_clip_box_shadow.deinit(device);
        self.cs_clip_image.deinit(device);
        self.cs_clip_line.deinit(device);
        self.ps_text_run.deinit(device);
        self.ps_text_run_dual_source.deinit(device);
        /*for shader in self.brush_image {
            if let Some(shader) = shader {
                shader.deinit(device);
            }
        }*/
        self.brush_image.deinit(device);
        for shader in self.brush_yuv_image {
            //if let Some(shader) = shader {
                shader.deinit(device);
            //}
        }
        self.cs_border_segment.deinit(device);
        self.ps_split_composite.deinit(device);
    }
}
