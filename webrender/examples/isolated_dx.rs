/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate webrender;
extern crate winit;
extern crate image;

#[path="common/boilerplate_dx.rs"]
mod boilerplate;

use app_units::Au;
use boilerplate::{Example, HandyDandyRectBuilder};
use euclid::vec2;
use winit::TouchPhase;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use webrender::api::*;

fn main() {
    let mut app = App {};
    boilerplate::main_wrapper(&mut app, None);
}

struct App { }

impl Example for App {
    fn render(&mut self,
              api: &RenderApi,
              builder: &mut DisplayListBuilder,
              resources: &mut ResourceUpdates,
              layout_size: LayoutSize,
              _pipeline_id: PipelineId,
              _document_id: DocumentId) {
        let bounds = LayoutRect::new(LayoutPoint::zero(), layout_size);
        builder.push_stacking_context(ScrollPolicy::Scrollable,
                                      bounds,
                                      None,
                                      TransformStyle::Flat,
                                      None,
                                      MixBlendMode::Normal,
                                      Vec::new());

        let bounds = (0, 0).to(100, 100);
        builder.push_rect(bounds, None, ColorF::new(1.0, 0.0, 0.0, 1.0));

        builder.pop_stacking_context();
    }

    fn on_event(&mut self,
                event: winit::Event,
                api: &RenderApi,
                document_id: DocumentId) -> bool {
        false
    }
}
