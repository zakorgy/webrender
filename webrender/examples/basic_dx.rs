/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate winit;
extern crate webrender;
extern crate webrender_traits;

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

fn load_file(name: &str) -> Vec<u8> {
    let mut file = File::open(name).unwrap();
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).unwrap();
    buffer
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

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_rect(LayoutRect::new(LayoutPoint::new(100.0, 100.0), LayoutSize::new(100.0, 200.0)),
                      clip,
                      ColorF::new(0.0, 1.0, 0.0, 1.0));

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_rect(LayoutRect::new(LayoutPoint::new(400.0, 100.0), LayoutSize::new(200.0, 200.0)),
                      clip,
                      ColorF::new(1.0, 0.0, 1.0, 1.0));

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_rect(LayoutRect::new(LayoutPoint::new(100.0, 250.0), LayoutSize::new(300.0, 200.0)),
                      clip,
                      ColorF::new(0.0, 1.0, 1.0, 1.0));

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_rect(LayoutRect::new(LayoutPoint::new(400.0, 400.0), LayoutSize::new(100.0, 200.0)),
                      clip,
                      ColorF::new(1.0, 1.0, 0.0, 1.0));

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_rect(LayoutRect::new(LayoutPoint::new(100.0, 500.0), LayoutSize::new(100.0, 200.0)),
                      clip,
                      ColorF::new(1.0, 1.0, 1.0, 1.0));
    let border_side = webrender_traits::BorderSide {
        color: ColorF::new(0.0, 0.0, 1.0, 1.0),
        style: webrender_traits::BorderStyle::Groove,
    };
    let border_widths = webrender_traits::BorderWidths {
        top: 10.0,
        left: 10.0,
        bottom: 10.0,
        right: 10.0,
    };
    let border_details = webrender_traits::BorderDetails::Normal(webrender_traits::NormalBorder {
        top: border_side,
        right: border_side,
        bottom: border_side,
        left: border_side,
        radius: webrender_traits::BorderRadius::uniform(20.0),
    });

    let clip = push_sub_clip(&api, &mut builder, &bounds);
    builder.push_border(LayoutRect::new(LayoutPoint::new(100.0, 100.0), LayoutSize::new(100.0, 100.0)),
                        clip,
                        border_widths,
                        border_details);

    if true { // draw text?
        let font_key = api.generate_font_key();
        let font_bytes = load_file("res/FreeSans.ttf");
        api.add_raw_font(font_key, font_bytes, 0);

        let text_bounds = LayoutRect::new(LayoutPoint::new(100.0, 200.0), LayoutSize::new(700.0, 300.0));

        let glyphs = vec![
            GlyphInstance {
                index: 48,
                point: LayoutPoint::new(100.0, 100.0),
            },
            GlyphInstance {
                index: 68,
                point: LayoutPoint::new(150.0, 100.0),
            },
            GlyphInstance {
                index: 80,
                point: LayoutPoint::new(200.0, 100.0),
            },
            GlyphInstance {
                index: 82,
                point: LayoutPoint::new(250.0, 100.0),
            },
            GlyphInstance {
                index: 81,
                point: LayoutPoint::new(300.0, 100.0),
            },
            GlyphInstance {
                index: 3,
                point: LayoutPoint::new(350.0, 100.0),
            },
            GlyphInstance {
                index: 86,
                point: LayoutPoint::new(400.0, 100.0),
            },
            GlyphInstance {
                index: 79,
                point: LayoutPoint::new(450.0, 100.0),
            },
            GlyphInstance {
                index: 72,
                point: LayoutPoint::new(500.0, 100.0),
            },
            GlyphInstance {
                index: 83,
                point: LayoutPoint::new(550.0, 100.0),
            },
            GlyphInstance {
                index: 87,
                point: LayoutPoint::new(600.0, 100.0),
            },
            GlyphInstance {
                index: 17,
                point: LayoutPoint::new(650.0, 100.0),
            },
        ];

        let clip = builder.push_clip_region(&bounds, Vec::new(), None);
        builder.push_text(text_bounds,
                          clip,
                          &glyphs,
                          font_key,
                          ColorF::new(1.0, 1.0, 0.0, 1.0),
                          Au::from_px(32),
                          0.0,
                          None);
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
