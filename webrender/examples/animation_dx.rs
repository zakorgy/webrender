/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate winit;
extern crate webrender;
extern crate webrender_traits;
extern crate rayon;

#[macro_use]
extern crate lazy_static;

use rayon::ThreadPool;
use rayon::Configuration as ThreadPoolConfig;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::rc::Rc;
use std::sync::mpsc::{channel, Sender, Receiver};
use std::sync::Mutex;
use webrender_traits::*;

lazy_static! {
    static ref TRANSFORM: Mutex<LayoutTransform> = Mutex::new(LayoutTransform::identity());
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

fn main() {
    let mut events_loop = winit::EventsLoop::new();
    let window = Rc::new(winit::WindowBuilder::new()
                         .with_title("WebRender Sample dx11")
                         .build(&events_loop)
                         .unwrap());

    let (width, height) = window.get_inner_size_pixels().unwrap();

    let opts = webrender::RendererOptions {
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

    let bounds = (0,0).to(100, 100);
    builder.push_stacking_context(webrender_traits::ScrollPolicy::Scrollable,
                                  bounds,
                                  Some(PropertyBinding::Binding(PropertyBindingKey::new(42))),
                                  TransformStyle::Flat,
                                  None,
                                  webrender_traits::MixBlendMode::Normal,
                                  Vec::new());

    // Fill it with a white rect
    let clip = builder.push_clip_region(&bounds, vec![], None);
    builder.push_rect(bounds,
                      clip,
                      ColorF::new(1.0, 1.0, 1.0, 1.0));

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
            winit::Event::WindowEvent { event, .. } => {
                match event {
                    winit::WindowEvent::Closed => winit::ControlFlow::Break,
                    winit::WindowEvent::KeyboardInput { input: winit::KeyboardInput { state: winit::ElementState::Pressed, virtual_keycode: Some(key), .. }, .. } => {
                        let offset = match key {
                             winit::VirtualKeyCode::Down => (0.0, 10.0),
                             winit::VirtualKeyCode::Up => (0.0, -10.0),
                             winit::VirtualKeyCode::Right => (10.0, 0.0),
                             winit::VirtualKeyCode::Left => (-10.0, 0.0),
                             _ => return winit::ControlFlow::Continue,
                        };
                        // Update the transform based on the keyboard input and push it to
                        // webrender using the generate_frame API. This will recomposite with
                        // the updated transform.
                        let new_transform = TRANSFORM.lock().unwrap().post_translate(LayoutVector3D::new(offset.0, offset.1, 0.0));
                        api.generate_frame(Some(DynamicProperties {
                            transforms: vec![
                              PropertyValue {
                                key: PropertyBindingKey::new(42),
                                value: new_transform,
                              },
                            ],
                            floats: vec![],
                        }));
                        *TRANSFORM.lock().unwrap() = new_transform;
                        winit::ControlFlow::Continue
                    },
                    _ => winit::ControlFlow::Continue,
                }
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
