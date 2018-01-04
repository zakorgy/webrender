/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate webrender;
extern crate winit;

#[path = "common/boilerplate_hal.rs"]
mod boilerplate;

use app_units::Au;
use boilerplate::{Example, HandyDandyRectBuilder};
use euclid::vec2;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use webrender::api::*;

fn load_file(name: &str) -> Vec<u8> {
    let mut file = File::open(name).unwrap();
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).unwrap();
    buffer
}

fn main() {
    let mut app = App {};
    boilerplate::main_wrapper(&mut app, None);
}

struct App {}

impl Example for App {
    fn render(
        &mut self,
        api: &RenderApi,
        builder: &mut DisplayListBuilder,
        resources: &mut ResourceUpdates,
        _: DeviceUintSize,
        _pipeline_id: PipelineId,
        _document_id: DocumentId,
    ) {
        /*let bounds = LayoutRect::new(LayoutPoint::zero(), builder.content_size());
        let info = LayoutPrimitiveInfo::new(bounds);
        builder.push_stacking_context(
            &info,
            ScrollPolicy::Scrollable,
            None,
            TransformStyle::Flat,
            None,
            MixBlendMode::Normal,
            Vec::new(),
        );

        let image_mask_key = api.generate_image_key();
        resources.add_image(
            image_mask_key,
            ImageDescriptor::new(2, 2, ImageFormat::A8, true),
            ImageData::new(vec![0, 80, 180, 255]),
            None,
        );
        let mask = ImageMask {
            image: image_mask_key,
            rect: (75, 75).by(100, 100),
            repeat: false,
        };
        let complex = ComplexClipRegion::new(
            (50, 50).to(150, 150),
            BorderRadius::uniform(20.0),
            ClipMode::Clip
        );
        let id = builder.define_clip(None, bounds, vec![complex], Some(mask));
        builder.push_clip_id(id);

        let info = LayoutPrimitiveInfo::new((100, 100).to(200, 200));
        builder.push_rect(&info, ColorF::new(0.0, 1.0, 0.0, 1.0));

        let info = LayoutPrimitiveInfo::new((250, 100).to(350, 200));
        builder.push_rect(&info, ColorF::new(0.0, 1.0, 0.0, 1.0));
        let border_side = BorderSide {
            color: ColorF::new(0.0, 0.0, 1.0, 1.0),
            style: BorderStyle::Groove,
        };
        let border_widths = BorderWidths {
            top: 10.0,
            left: 10.0,
            bottom: 10.0,
            right: 10.0,
        };
        let border_details = BorderDetails::Normal(NormalBorder {
            top: border_side,
            right: border_side,
            bottom: border_side,
            left: border_side,
            radius: BorderRadius::uniform(20.0),
        });

        let info = LayoutPrimitiveInfo::new((100, 100).to(200, 200));
        builder.push_border(&info, border_widths, border_details);
        builder.pop_clip_id();

        builder.pop_stacking_context();*/
    }

    fn on_event(&mut self, event: winit::Event, api: &RenderApi, document_id: DocumentId) -> bool {
        false
    }
}