/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate winit;
extern crate webrender;
extern crate webrender_traits;

use std::env;
use std::path::PathBuf;
use std::rc::Rc;
use webrender_traits::{ClipRegionToken, ColorF, DisplayListBuilder, Epoch};
use webrender_traits::{DeviceUintSize, LayoutPoint, LayoutRect, LayoutSize};
use webrender_traits::{ImageData, ImageDescriptor, ImageFormat};
use webrender_traits::{PipelineId, RenderApi, TransformStyle};

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
        self.proxy.wakeup().unwrap();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        #[cfg(not(target_os = "android"))]
        self.proxy.wakeup().unwrap();
    }
}

fn push_sub_clip(api: &RenderApi, builder: &mut DisplayListBuilder, bounds: &LayoutRect)
                 -> ClipRegionToken {
    let mask_image = api.generate_image_key();
    api.add_image(mask_image,
                  ImageDescriptor::new(2, 2, ImageFormat::A8, true),
                  ImageData::new(vec![0, 80, 180, 255]),
                  None);

    builder.push_clip_region(bounds, vec![], None)
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
    let (mut renderer, sender, gfx_window) = webrender::renderer::Renderer::new(window.clone(), opts, size).unwrap();
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

    let border_widths = webrender_traits::BorderWidths {
        top: 3.0,
        left: 3.0,
        bottom: 3.0,
        right: 3.0,
    };

    let border_styles = vec![webrender_traits::BorderStyle::None,
                             webrender_traits::BorderStyle::Solid,
                             webrender_traits::BorderStyle::Double,
                             webrender_traits::BorderStyle::Dotted,
                             webrender_traits::BorderStyle::Dashed,
                             webrender_traits::BorderStyle::Hidden,
                             webrender_traits::BorderStyle::Groove,
                             webrender_traits::BorderStyle::Ridge,
                             webrender_traits::BorderStyle::Inset,
                             webrender_traits::BorderStyle::Outset
                             ];
    for (i, style) in border_styles.into_iter().enumerate() {

        let border_side = webrender_traits::BorderSide {
            color: ColorF::new(0.0, 1.0, 0.0, 1.0),
            style: style,
        };

        let border_details = webrender_traits::BorderDetails::Normal(webrender_traits::NormalBorder {
            top: border_side,
            right: border_side,
            bottom: border_side,
            left: border_side,
            radius: webrender_traits::BorderRadius::uniform(0.0),
        });

        let clip = push_sub_clip(&api, &mut builder, &bounds);
        builder.push_border(LayoutRect::new(LayoutPoint::new(10.0, 10.0 + i as f32 * 60.0), LayoutSize::new(400.0, 50.0)),
                            clip,
                            border_widths.clone(),
                            border_details);
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
            _ => {
                renderer.update();
                renderer.render(DeviceUintSize::new(width, height));
                gfx_window.swap_buffers(1);
                winit::ControlFlow::Continue
            },
        }
    });
}