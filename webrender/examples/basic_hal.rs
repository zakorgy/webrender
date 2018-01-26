/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate webrender;
extern crate winit;

#[path = "common/boilerplate_hal.rs"]
mod boilerplate;

use boilerplate::{Example, HandyDandyRectBuilder};
use webrender::api::*;

fn main() {
    let mut app = App {};
    boilerplate::main_wrapper(&mut app, None);
}

struct App {}

impl Example for App {
    fn render(
        &mut self,
        _api: &RenderApi,
        builder: &mut DisplayListBuilder,
        _resources: &mut ResourceUpdates,
        _: DeviceUintSize,
        _pipeline_id: PipelineId,
        _document_id: DocumentId,
    ) {
        let bounds = LayoutRect::new(LayoutPoint::zero(), builder.content_size());
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
        let id = builder.define_clip(None, bounds, vec![], None);
        builder.push_clip_id(id);

        let info = LayoutPrimitiveInfo::new((100, 100).to(200, 200));
        builder.push_line(
            &info,
            0.0,
            LineOrientation::Horizontal,
            &ColorF::new(0.0, 1.0, 0.0, 1.0),
            LineStyle::Solid,
        );

        let info = LayoutPrimitiveInfo::new((300, 100).to(400, 200));
        builder.push_line(
            &info,
            0.0,
            LineOrientation::Horizontal,
            &ColorF::new(1.0, 1.0, 0.0, 1.0),
            LineStyle::Solid,
        );

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

        let info = LayoutPrimitiveInfo::new((100, 400).to(200, 600));
        builder.push_border(&info, border_widths, border_details);

        builder.pop_clip_id();
        builder.pop_stacking_context();
    }

    fn on_event(&mut self, _event: winit::Event, _api: &RenderApi, _document_id: DocumentId) -> bool {
        false
    }
}
