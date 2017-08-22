/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate winit;
extern crate webrender;
extern crate webrender_traits;

#[macro_use]
extern crate lazy_static;

use std::sync::Mutex;
use webrender_traits::*;

use app_units::Au;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::rc::Rc;
use webrender_traits::{ClipRegionToken, ColorF, DisplayListBuilder, Epoch, GlyphInstance};
use webrender_traits::{DeviceIntPoint, DeviceUintSize, LayoutPoint, LayoutRect, LayoutSize};
use webrender_traits::{ImageData, ImageDescriptor, ImageFormat};
use webrender_traits::{PipelineId, RenderApi, TransformStyle, BoxShadowClipMode};
use euclid::vec2;

lazy_static! {
    static ref CURSOR_POSITION: Mutex<WorldPoint> = Mutex::new(WorldPoint::zero());
}

pub trait HandyDandyRectBuilder {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect;
    fn by(&self, w: i32, h: i32) -> LayoutRect;
}
// Allows doing `(x, y).to(x2, y2)` or `(x, y).by(width, height)` with i32
// values to build a f32 LayoutRect
impl HandyDandyRectBuilder for (i32, i32) {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect {
        LayoutRect::new(LayoutPoint::new(self.0 as f32, self.1 as f32),
                        LayoutSize::new((x2 - self.0) as f32, (y2 - self.1) as f32))
    }

    fn by(&self, w: i32, h: i32) -> LayoutRect {
        LayoutRect::new(LayoutPoint::new(self.0 as f32, self.1 as f32),
                        LayoutSize::new(w as f32, h as f32))
    }
}

struct Notifier {
    proxy: winit::EventsLoopProxy,
}

impl Notifier {
    fn new(proxy: winit::EventsLoopProxy) -> Notifier {
        Notifier {
            proxy: proxy,
        }
    }
}

impl webrender_traits::RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        #[cfg(not(target_os = "android"))]
        self.proxy.wakeup();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        #[cfg(not(target_os = "android"))]
        self.proxy.wakeup();
    }
}

fn push_sub_clip(api: &RenderApi, builder: &mut DisplayListBuilder, bounds: &LayoutRect)
                 -> ClipRegionToken {
    let mask_image = api.generate_image_key();
    api.add_image(mask_image,
                  ImageDescriptor::new(2, 2, ImageFormat::A8, true),
                  ImageData::new(vec![0, 80, 180, 255]),
                  None);
    let mask = webrender_traits::ImageMask {
        image: mask_image,
        rect: LayoutRect::new(LayoutPoint::new(75.0, 75.0), LayoutSize::new(100.0, 100.0)),
        repeat: false,
    };
    let complex = webrender_traits::ComplexClipRegion::new(
        LayoutRect::new(LayoutPoint::new(50.0, 50.0), LayoutSize::new(100.0, 100.0)),
        webrender_traits::BorderRadius::uniform(20.0));

    builder.push_clip_region(bounds, vec![/*complex*/], None/*Some(mask)*/)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let mut events_loop = winit::EventsLoop::new();
    let window = Rc::new(winit::WindowBuilder::new()
                         .with_title("WebRender Sample dx11")
                         .build(&events_loop)
                         .unwrap());

    let (width, height) = window.get_inner_size_pixels().unwrap();

    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        debug: true,
        precache_shaders: true,
        device_pixel_ratio: window.hidpi_factor(),
        .. Default::default()
    };

    let size = DeviceUintSize::new(width, height);
    let (mut renderer, sender, mut gfx_window) = webrender::renderer::Renderer::new(window.clone(), opts, size).unwrap();
    let api = sender.create_api();

    let notifier = Box::new(Notifier::new(events_loop.create_proxy()));
    renderer.set_render_notifier(notifier);

    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

    let pipeline_id = PipelineId(0, 0);
    let layout_size = LayoutSize::new(width as f32, height as f32);
    let mut builder = webrender_traits::DisplayListBuilder::new(pipeline_id, layout_size);

    let bounds = LayoutRect::new(LayoutPoint::zero(), layout_size);
    builder.push_stacking_context(webrender_traits::ScrollPolicy::Scrollable,
                                  bounds,
                                  None,
                                  TransformStyle::Flat,
                                  None,
                                  webrender_traits::MixBlendMode::Normal,
                                  Vec::new());

    if true {   // scrolling and clips stuff
        // let's make a scrollbox
        let scrollbox = (0, 0).to(300, 400);
        builder.push_stacking_context(webrender_traits::ScrollPolicy::Scrollable,
                                      LayoutRect::new(LayoutPoint::new(10.0, 10.0),
                                                      LayoutSize::zero()),
                                      None,
                                      TransformStyle::Flat,
                                      None,
                                      webrender_traits::MixBlendMode::Normal,
                                      Vec::new());
        // set the scrolling clip
        let clip = builder.push_clip_region(&scrollbox, vec![], None);
        let clip_id = builder.define_clip((0, 0).to(1000, 1000),
                                          clip,
                                          Some(ClipId::new(42, pipeline_id)));
        builder.push_clip_id(clip_id);
        // now put some content into it.
        // start with a white background
        let clip = builder.push_clip_region(&(0, 0).to(1000, 1000), vec![], None);
        builder.push_rect((0, 0).to(500, 500),
                          clip,
                          ColorF::new(1.0, 1.0, 1.0, 1.0));
        // let's make a 50x50 blue square as a visual reference
        let clip = builder.push_clip_region(&(0, 0).to(50, 50), vec![], None);
        builder.push_rect((0, 0).to(50, 50),
                          clip,
                          ColorF::new(0.0, 0.0, 1.0, 1.0));
        // and a 50x50 green square next to it with an offset clip
        // to see what that looks like
        let clip = builder.push_clip_region(&(60, 10).to(110, 60), vec![], None);
        builder.push_rect((50, 0).to(100, 50),
                          clip,
                          ColorF::new(0.0, 1.0, 0.0, 1.0));

        // Below the above rectangles, set up a nested scrollbox. It's still in
        // the same stacking context, so note that the rects passed in need to
        // be relative to the stacking context.
        let clip = builder.push_clip_region(&(0, 100).to(200, 300), vec![], None);
        let nested_clip_id = builder.define_clip((0, 100).to(300, 400),
                                                 clip,
                                                 Some(ClipId::new(43, pipeline_id)));
        builder.push_clip_id(nested_clip_id);
        // give it a giant gray background just to distinguish it and to easily
        // visually identify the nested scrollbox
        let clip = builder.push_clip_region(&(-1000, -1000).to(5000, 5000), vec![], None);
        builder.push_rect((-1000, -1000).to(5000, 5000),
                          clip,
                          ColorF::new(0.5, 0.5, 0.5, 1.0));
        // add a teal square to visualize the scrolling/clipping behaviour
        // as you scroll the nested scrollbox with WASD keys
        let clip = builder.push_clip_region(&(0, 100).to(50, 150), vec![], None);
        builder.push_rect((0, 100).to(50, 150),
                          clip,
                          ColorF::new(0.0, 1.0, 1.0, 1.0));
        // just for good measure add another teal square in the bottom-right
        // corner of the nested scrollframe content, which can be scrolled into
        // view by the user
        let clip = builder.push_clip_region(&(250, 350).to(300, 400), vec![], None);
        builder.push_rect((250, 350).to(300, 400),
                          clip,
                          ColorF::new(0.0, 1.0, 1.0, 1.0));
        builder.pop_clip_id(); // nested_clip_id

        builder.pop_clip_id(); // clip_id
        builder.pop_stacking_context();
    }

    builder.pop_stacking_context();

    api.set_display_list(
        Some(root_background_color),
        epoch,
        LayoutSize::new(width as f32, height as f32),
        builder.finalize(),
        true);
    api.set_root_pipeline(pipeline_id);
    api.generate_frame(None);

    events_loop.run_forever(|event| {
        match event {
            winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => {
                winit::ControlFlow::Break
            },

            winit::Event::WindowEvent { event: winit::WindowEvent::MouseMoved{device_id: _, position}, .. }=> {
                *CURSOR_POSITION.lock().unwrap() = WorldPoint::new(position.0 as f32, position.1 as f32);
                winit::ControlFlow::Continue
            }
             winit::Event::WindowEvent { event: winit::WindowEvent::MouseWheel {device_id: _, delta, phase}, .. } => {
                const LINE_HEIGHT: f32 = 38.0;
                let (dx, dy) = match delta {
                    winit::MouseScrollDelta::LineDelta(dx, dy) => (dx, dy * LINE_HEIGHT),
                    winit::MouseScrollDelta::PixelDelta(dx, dy) => (dx, dy),
                };

                api.scroll(ScrollLocation::Delta(LayoutVector2D::new(dx, dy)),
                           *CURSOR_POSITION.lock().unwrap(),
                           ScrollEventPhase::Start);
                winit::ControlFlow::Continue
            },
            _ => {
                renderer.update();
                renderer.render(DeviceUintSize::new(width, height));
                gfx_window.swap_buffers(1);
                winit::ControlFlow::Continue
            },
        }
    });
}
