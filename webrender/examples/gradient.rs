/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate gleam;
extern crate glutin;
extern crate webrender;
extern crate webrender_traits;
extern crate rayon;

use gleam::gl;
use rayon::ThreadPool;
use rayon::Configuration as ThreadPoolConfig;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender, Receiver};
use webrender_traits::{BlobImageData, BlobImageDescriptor, BlobImageError, BlobImageRenderer, BlobImageRequest};
use webrender_traits::{BlobImageResult, TileOffset, ColorF, ColorU, Epoch};
use webrender_traits::{DeviceUintSize, DeviceUintRect, LayoutPoint, LayoutRect, LayoutSize};
use webrender_traits::{ImageData, ImageDescriptor, ImageFormat, ImageRendering, ImageKey, TileSize};
use webrender_traits::{PipelineId, RasterizedBlobImage, TransformStyle};
use webrender_traits::{ExtendMode, GradientStop};

fn main() {
    let window = glutin::WindowBuilder::new()
                .with_title("WebRender Sample (Gradient)")
                .with_multitouch()
                .with_gl(glutin::GlRequest::GlThenGles {
                    opengl_version: (3, 2),
                    opengles_version: (3, 0)
                })
                .build()
                .unwrap();

    unsafe {
        window.make_current().ok();
    }

    let gl = match gl::GlType::default() {
        gl::GlType::Gl => unsafe { gl::GlFns::load_with(|symbol| window.get_proc_address(symbol) as *const _) },
        gl::GlType::Gles => unsafe { gl::GlesFns::load_with(|symbol| window.get_proc_address(symbol) as *const _) },
    };

    println!("OpenGL version {}", gl.get_string(gl::VERSION));

    let (width, height) = window.get_inner_size_pixels().unwrap();

    let worker_config = ThreadPoolConfig::new().thread_name(|idx|{
        format!("WebRender:Worker#{}", idx)
    });

    let workers = Arc::new(ThreadPool::new(worker_config).unwrap());

    let opts = webrender::RendererOptions {
        debug: true,
        workers: Some(Arc::clone(&workers)),
        // Register our blob renderer, so that WebRender integrates it in the resource cache..
        // Share the same pool of worker threads between WebRender and our blob renderer.
        device_pixel_ratio: window.hidpi_factor(),
        .. Default::default()
    };

    let size = DeviceUintSize::new(width, height);
    // Original
    // let (mut renderer, sender) = webrender::renderer::Renderer::new(gl, opts, size).unwrap();
    // Gfx
    let (mut renderer, sender) = webrender::renderer::Renderer::new(&window, opts, size).unwrap();
    let api = sender.create_api();

    let notifier = Box::new(Notifier::new(window.create_window_proxy()));
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

    'outer: for event in window.wait_events() {
        let mut events = Vec::new();
        events.push(event);

        for event in window.poll_events() {
            events.push(event);
        }

        for event in events {
            match event {
                glutin::Event::Closed |
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
                glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Q)) => break 'outer,
                glutin::Event::KeyboardInput(glutin::ElementState::Pressed,
                                             _, Some(glutin::VirtualKeyCode::P)) => {
                    //let enable_profiler = !renderer.get_profiler_enabled();
                    //renderer.set_profiler_enabled(enable_profiler);
                    api.generate_frame(None);
                }
                _ => ()
            }
        }

        renderer.update();
        renderer.render(DeviceUintSize::new(width, height));
        window.swap_buffers().ok();
    }
}

struct Notifier {
    window_proxy: glutin::WindowProxy,
}

impl Notifier {
    fn new(window_proxy: glutin::WindowProxy) -> Notifier {
        Notifier {
            window_proxy: window_proxy,
        }
    }
}

impl webrender_traits::RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        #[cfg(not(target_os = "android"))]
        self.window_proxy.wakeup_event_loop();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        #[cfg(not(target_os = "android"))]
        self.window_proxy.wakeup_event_loop();
    }
}
