/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::pso::{DescriptorSetLayoutBinding, DescriptorType as DT, ShaderStageFlags as SSF};
use super::super::{ShaderKind, VertexArrayKind};

const fn descriptor_set_layout_binding(
    binding: u32,
    ty: DT,
    stage_flags: SSF,
    immutable_samplers: bool,
) -> DescriptorSetLayoutBinding {
    DescriptorSetLayoutBinding {
        binding,
        ty,
        count: 1,
        stage_flags,
        immutable_samplers,
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub(super) enum DescriptorGroup {
    Default,
    Clip,
    Primitive,
}

impl From<ShaderKind> for DescriptorGroup {
    fn from(kind: ShaderKind) -> Self {
        match kind {
            ShaderKind::Cache(VertexArrayKind::Border) | ShaderKind::Cache(VertexArrayKind::LineDecoration)
                | ShaderKind::DebugColor | ShaderKind::DebugFont
                | ShaderKind::VectorStencil | ShaderKind::VectorCover => DescriptorGroup::Default,
            ShaderKind::ClipCache => DescriptorGroup::Clip,
            _ => DescriptorGroup::Primitive,
        }
    }
}

pub(super) const DEFAULT_SET_0: &'static [DescriptorSetLayoutBinding] = &[
    // Dither
    descriptor_set_layout_binding(0, DT::CombinedImageSampler, SSF::ALL, true),
];

pub(super) const COMMON_SET_1: &'static [DescriptorSetLayoutBinding] = &[
    // Color0 sampler
    descriptor_set_layout_binding(12, DT::Sampler, SSF::ALL, false),
    // Color1 sampler
    descriptor_set_layout_binding(13, DT::Sampler, SSF::ALL, false),
    // Color2 sampler
    descriptor_set_layout_binding(14, DT::Sampler, SSF::ALL, false),
];

pub(super) const DEFAULT_SET_2: &'static [DescriptorSetLayoutBinding] = &[
    // Color0
    descriptor_set_layout_binding(0, DT::SampledImage, SSF::ALL, false),
    // Color1
    descriptor_set_layout_binding(1, DT::SampledImage, SSF::ALL, false),
    // Color2
    descriptor_set_layout_binding(2, DT::SampledImage, SSF::ALL, false),
];

pub(super) const COMMON_SET_3: &'static [DescriptorSetLayoutBinding] = &[
    // Locals
    descriptor_set_layout_binding(0, DT::UniformBuffer, SSF::VERTEX, false),
];

pub(super) const CLIP_SET_0: &'static [DescriptorSetLayoutBinding] = &[
    // Dither
    descriptor_set_layout_binding(0, DT::CombinedImageSampler, SSF::ALL, true),
    // RenderTasks
    descriptor_set_layout_binding(1, DT::CombinedImageSampler, SSF::VERTEX, true),
    // GpuCache
    descriptor_set_layout_binding(2, DT::CombinedImageSampler, SSF::ALL, true),
    // TransformPalette
    descriptor_set_layout_binding(3, DT::CombinedImageSampler, SSF::VERTEX, true),
];

pub(super) const CLIP_SET_2: &'static [DescriptorSetLayoutBinding] = DEFAULT_SET_2;

pub(super) const PRIMITIVE_SET_0: &'static [DescriptorSetLayoutBinding] = &[
    // Dither
    descriptor_set_layout_binding(0, DT::CombinedImageSampler, SSF::ALL, true),
    // RenderTasks
    descriptor_set_layout_binding(1, DT::CombinedImageSampler, SSF::VERTEX, true),
    // GpuCache
    descriptor_set_layout_binding(2, DT::CombinedImageSampler, SSF::ALL, true),
    // TransformPalette
    descriptor_set_layout_binding(3, DT::CombinedImageSampler, SSF::VERTEX, true),
    // PrimitiveHeadersF
    descriptor_set_layout_binding(4, DT::CombinedImageSampler, SSF::VERTEX, true),
    // PrimitiveHeadersI
    descriptor_set_layout_binding(5, DT::CombinedImageSampler, SSF::VERTEX, true),
];

pub(super) const PRIMITIVE_SET_2: &'static [DescriptorSetLayoutBinding] = &[
    // Color0
    descriptor_set_layout_binding(0, DT::SampledImage, SSF::ALL, false),
    // Color1
    descriptor_set_layout_binding(1, DT::SampledImage, SSF::ALL, false),
    // Color2
    descriptor_set_layout_binding(2, DT::SampledImage, SSF::ALL, false),
    // PrevPassAlpha
    descriptor_set_layout_binding(3, DT::CombinedImageSampler, SSF::ALL, true),
    // PrevPassColor
    descriptor_set_layout_binding(4, DT::CombinedImageSampler, SSF::ALL, true),
];
