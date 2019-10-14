/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::Device;
use hal::format::Format;
use hal::image::Layout;
use hal::pass::{Attachment, SubpassDesc};
use internal_types::FastHashMap;

pub(super) const DEPTH_FORMAT: Format = hal::format::Format::D32Sfloat;

const SUBPASS_R8: SubpassDesc = SubpassDesc {
    colors: &[(0, Layout::ColorAttachmentOptimal)],
    depth_stencil: None,
    inputs: &[],
    resolves: &[],
    preserves: &[],
};

const SUBPASS_DEPTH_R8: SubpassDesc = SubpassDesc {
    colors: &[(0, Layout::ColorAttachmentOptimal)],
    depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
    inputs: &[],
    resolves: &[],
    preserves: &[],
};

const SUBPASS_BGRA8: SubpassDesc = SubpassDesc {
    colors: &[(0, Layout::ColorAttachmentOptimal)],
    depth_stencil: None,
    inputs: &[],
    resolves: &[],
    preserves: &[],
};

const SUBPASS_DEPTH_BGRA8: SubpassDesc = SubpassDesc {
    colors: &[(0, Layout::ColorAttachmentOptimal)],
    depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
    inputs: &[],
    resolves: &[],
    preserves: &[],
};

pub(super) const DEPTH_ATTACHMENT_STATE: AttachmentState = AttachmentState {
    format: DEPTH_FORMAT,
    src_layout: Layout::DepthStencilAttachmentOptimal,
    dst_layout: Layout::DepthStencilAttachmentOptimal,
};

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub(super) struct AttachmentState {
    pub format: Format,
    pub src_layout: Layout,
    pub dst_layout: Layout,
}

impl<T: Into<Format>> From<T> for AttachmentState {
    fn from(format: T) -> Self {
        AttachmentState {
            format: format.into(),
            src_layout: Layout::ColorAttachmentOptimal,
            dst_layout: Layout::ColorAttachmentOptimal,
        }
    }
}

pub(super) type RenderPassSKey = (AttachmentState, Option<AttachmentState>);

pub(super) struct RenderPassManager<B: hal::Backend> {
    attachments: FastHashMap<AttachmentState, Attachment>,
    render_passes: FastHashMap<RenderPassSKey, B::RenderPass>,
}

impl<B: hal::Backend> RenderPassManager<B> {
    pub(super) fn new() -> RenderPassManager<B> {
        RenderPassManager {
            attachments: FastHashMap::default(),
            render_passes: FastHashMap::default(),
        }
    }

    fn create_attachment(att_state: &AttachmentState) -> Attachment {
        hal::pass::Attachment {
            format: Some(att_state.format),
            samples: 1,
            // TODO(zakorgy): add AttachmentOps to AttachmentState
            ops: hal::pass::AttachmentOps::new(
                hal::pass::AttachmentLoadOp::Load,
                hal::pass::AttachmentStoreOp::DontCare,
            ),
            stencil_ops: hal::pass::AttachmentOps::DONT_CARE,
            layouts: att_state.src_layout .. att_state.dst_layout,
        }
    }

    fn get_subpass_for_attachments((color_att, depth_att): &RenderPassSKey) -> SubpassDesc {
        match color_att.format {
            Format::Bgra8Unorm => {
                if depth_att.is_some() {
                    SUBPASS_DEPTH_BGRA8
                } else {
                    SUBPASS_BGRA8
                }
            }
            Format::R8Unorm => {
                if depth_att.is_some() {
                    SUBPASS_DEPTH_R8
                } else {
                    SUBPASS_R8
                }
            }
            _ => unimplemented!("Format not supported {:?}", color_att.format),
        }
    }

    pub(super) fn get_render_pass(
        &mut self,
        device: &B::Device,
        key: RenderPassSKey,
    ) -> &B::RenderPass {
        if !self.render_passes.contains_key(&key) {
            let ref mut attachment1 = self.attachments.entry(key.0).or_insert(Self::create_attachment(&key.0)).clone();
            let attachment2 = key.1.map(|k| self.attachments.entry(k).or_insert(Self::create_attachment(&k)));
            let rp = unsafe {
                device.create_render_pass(
                    Some(attachment1).into_iter().chain(attachment2),
                    &[Self::get_subpass_for_attachments(&key)],
                    &[],
                )
                .expect("create_render_pass failed")
            };
            self.render_passes.insert(key, rp);
        }
        self.render_passes.get(&key).unwrap()
    }

    pub(super) fn deinit(self, device: &B::Device) {
        for (_, rp) in self.render_passes {
            unsafe { device.destroy_render_pass(rp); }
        }
    }
}
