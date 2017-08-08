/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate winit;
extern crate webrender;
extern crate webrender_traits;
extern crate rayon;

use rayon::ThreadPool;
use rayon::Configuration as ThreadPoolConfig;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::rc::Rc;
use std::sync::mpsc::{channel, Sender, Receiver};
use webrender_traits::{BlobImageData, BlobImageDescriptor, BlobImageError, BlobImageRenderer, BlobImageRequest};
use webrender_traits::{BlobImageResult, TileOffset, ColorF, ColorU, Epoch};
use webrender_traits::{DeviceUintSize, DeviceUintRect, LayoutPoint, LayoutRect, LayoutSize};
use webrender_traits::{ImageData, ImageDescriptor, ImageFormat, ImageRendering, ImageKey, TileSize};
use webrender_traits::{PipelineId, RasterizedBlobImage, TransformStyle};
use webrender_traits::{ExtendMode, GradientStop};

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
    let root_background_color = ColorF::new(0.2, 0.2, 0.2, 1.0);

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

    let stops = vec![
        GradientStop {
            offset: 0.0,
            color: ColorU::new(255u8, 255u8, 0u8, 0u8).into(),
        },
        GradientStop {
            offset: 1.0,
            color: ColorU::new(255u8, 0u8, 0u8, 255u8).into(),
        },
    ];
    let gradient = builder.create_gradient(LayoutPoint::new(0.0, 0.0),
                                           LayoutPoint::new(0.0, 300.0),
                                           stops,
                                           ExtendMode::Clamp);

    let clip = builder.push_clip_region(&bounds, vec![], None);
    builder.push_gradient(LayoutRect::new(LayoutPoint::new(30.0, 100.0), LayoutSize::new(300.0, 300.0)),
                          clip,
                          gradient,
                          layout_size,
                          webrender_traits::LayoutSize::zero());
    let stops = vec![
        GradientStop {
            offset: 0.0,
            color: ColorU::new(255u8, 255u8, 0u8, 255u8).into(),
        },
        GradientStop {
            offset: 1.0,
            color: ColorU::new(255, 0u8, 0u8, 255u8).into(),
        },
    ];

    let gradient = builder.create_gradient(LayoutPoint::new(0.0, 0.0),
                                           LayoutPoint::new(0.0, 300.0),
                                           stops,
                                           ExtendMode::Clamp);
    let clip = builder.push_clip_region(&bounds, vec![], None);
    builder.push_gradient(LayoutRect::new(LayoutPoint::new(400.0, 100.0), LayoutSize::new(300.0, 300.0)),
                          clip,
                          gradient,
                          layout_size,
                          webrender_traits::LayoutSize::zero());

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
