/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![cfg_attr(
    not(any(feature = "gfx-hal", feature = "gl")),
    allow(dead_code, unused_imports)
)]

extern crate app_units;
extern crate euclid;
extern crate webrender;
extern crate winit;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;
#[cfg(not(feature = "gfx-hal"))]
extern crate gfx_backend_empty as back;
#[cfg(feature = "gl")]
extern crate gleam;
#[cfg(feature = "gl")]
extern crate glutin;

use app_units::Au;
#[cfg(feature = "gl")]
use gleam::gl;
#[cfg(feature = "gl")]
use glutin::GlContext;
#[cfg(feature = "gfx-hal")]
use std::boxed::Box;
use std::fs::File;
use std::io::Read;
#[cfg(feature = "gl")]
use std::marker::PhantomData;
use webrender::api::*;
#[cfg(feature = "gfx-hal")]
use webrender::hal::{self, Instance};
use winit::dpi::LogicalSize;

struct Notifier {
    events_proxy: winit::EventsLoopProxy,
}

impl Notifier {
    fn new(events_proxy: winit::EventsLoopProxy) -> Notifier {
        Notifier { events_proxy }
    }
}

impl RenderNotifier for Notifier {
    fn clone(&self) -> Box<RenderNotifier> {
        Box::new(Notifier {
            events_proxy: self.events_proxy.clone(),
        })
    }

    fn wake_up(&self) {
        #[cfg(not(target_os = "android"))]
        let _ = self.events_proxy.wakeup();
    }

    fn new_frame_ready(&self,
                       _: DocumentId,
                       _scrolled: bool,
                       _composite_needed: bool,
                       _render_time: Option<u64>) {
        self.wake_up();
    }
}


struct Window {
    events_loop: winit::EventsLoop, //TODO: share events loop?
    #[cfg(feature = "gl")]
    window: glutin::GlWindow,
    #[cfg(feature = "gfx-hal")]
    window: winit::Window,
    renderer: webrender::Renderer<back::Backend>,
    name: &'static str,
    pipeline_id: PipelineId,
    document_id: DocumentId,
    epoch: Epoch,
    api: RenderApi,
    font_instance_key: FontInstanceKey,
    #[cfg(feature = "gfx-hal")]
    surface: Box<hal::Surface<back::Backend>>,
}

#[cfg(any(feature = "gfx-hal", feature = "gl"))]
impl Window {
    fn new(name: &'static str, clear_color: ColorF) -> Self {
        let events_loop = winit::EventsLoop::new();
        let window_builder = winit::WindowBuilder::new()
            .with_title(name)
            .with_multitouch()
            .with_dimensions(LogicalSize::new(800., 600.));

        #[cfg(feature = "gl")]
        let (init, window) = {
            let context_builder = glutin::ContextBuilder::new()
                .with_gl(glutin::GlRequest::GlThenGles {
                    opengl_version: (3, 2),
                    opengles_version: (3, 0),
                });
            let window = glutin::GlWindow::new(window_builder, context_builder, &events_loop)
                .unwrap();

            unsafe {
                window.make_current().ok();
            }

            let gl = match window.get_api() {
                glutin::Api::OpenGl => unsafe {
                    gl::GlFns::load_with(|symbol| window.get_proc_address(symbol) as *const _)
                },
                glutin::Api::OpenGlEs => unsafe {
                    gl::GlesFns::load_with(|symbol| window.get_proc_address(symbol) as *const _)
                },
                glutin::Api::WebGl => unimplemented!(),
            };

            let init = webrender::RendererInit {
                gl,
                phantom_data: PhantomData,
            };
            (init, window)
        };

        #[cfg(feature = "gfx-hal")]
        let (window, adapter, mut surface) = {
            let window = window_builder.build(&events_loop).unwrap();
            let instance = back::Instance::create("gfx-rs instance", 1);
            let mut adapters = instance.enumerate_adapters();
            let adapter = adapters.remove(0);
            let mut surface = instance.create_surface(&window);
            (window, adapter, surface)
        };

        let LogicalSize { width, height } = window.get_inner_size().unwrap();

        let framebuffer_size = DeviceUintSize::new(width as u32, height as u32);
        let notifier = Box::new(Notifier::new(events_loop.create_proxy()));
        let (renderer, sender) = {
            #[cfg(feature = "gfx-hal")]
            let init = webrender::RendererInit {
                adapter: &adapter,
                surface: &mut surface,
                window_size: (width as u32, height as u32),
            };
            let device_pixel_ratio = window.get_hidpi_factor() as f32;
            let opts = webrender::RendererOptions {
                device_pixel_ratio,
                clear_color: Some(clear_color),
                ..webrender::RendererOptions::default()
            };
            webrender::Renderer::new(init, notifier, opts).unwrap()
        };
        let api = sender.create_api();
        let document_id = api.add_document(framebuffer_size, 0);

        let epoch = Epoch(0);
        let pipeline_id = PipelineId(0, 0);
        let mut txn = Transaction::new();

        let font_key = api.generate_font_key();
        let font_bytes = load_file("../wrench/reftests/text/FreeSans.ttf");
        txn.add_raw_font(font_key, font_bytes, 0);

        let font_instance_key = api.generate_font_instance_key();
        txn.add_font_instance(font_instance_key, font_key, Au::from_px(32), None, None, Vec::new());

        api.send_transaction(document_id, txn);

        Window {
            events_loop,
            window,
            renderer,
            name,
            epoch,
            pipeline_id,
            document_id,
            api,
            font_instance_key,
            #[cfg(feature = "gfx-hal")]
            surface: Box::new(surface),
        }
    }

    fn tick(&mut self) -> bool {
        #[cfg(not(any(feature = "vulkan", feature = "dx12", feature = "metal")))]
        unsafe {
            self.window.make_current().ok();
        }
        let mut do_exit = false;
        let my_name = &self.name;
        let renderer = &mut self.renderer;

        self.events_loop.poll_events(|global_event| match global_event {
            winit::Event::WindowEvent { event, .. } => match event {
                winit::WindowEvent::CloseRequested |
                winit::WindowEvent::KeyboardInput {
                    input: winit::KeyboardInput {
                        virtual_keycode: Some(winit::VirtualKeyCode::Escape),
                        ..
                    },
                    ..
                } => {
                    do_exit = true
                }
                winit::WindowEvent::KeyboardInput {
                    input: winit::KeyboardInput {
                        state: winit::ElementState::Pressed,
                        virtual_keycode: Some(winit::VirtualKeyCode::P),
                        ..
                    },
                    ..
                } => {
                    println!("toggle flags {}", my_name);
                    renderer.toggle_debug_flags(webrender::DebugFlags::PROFILER_DBG);
                }
                _ => {}
            }
            _ => {}
        });
        if do_exit {
            return true
        }

        let framebuffer_size = {
            let LogicalSize { width, height } = self.window.get_inner_size().unwrap();
            DeviceUintSize::new(width as u32, height as u32)
        };
        let device_pixel_ratio = self.window.get_hidpi_factor() as f32;
        let layout_size = framebuffer_size.to_f32() / euclid::TypedScale::new(device_pixel_ratio);
        let mut txn = Transaction::new();
        let mut builder = DisplayListBuilder::new(self.pipeline_id, layout_size);

        let bounds = LayoutRect::new(LayoutPoint::zero(), builder.content_size());
        let info = LayoutPrimitiveInfo::new(bounds);
        builder.push_stacking_context(
            &info,
            None,
            TransformStyle::Flat,
            MixBlendMode::Normal,
            Vec::new(),
            GlyphRasterSpace::Screen,
        );

        let info = LayoutPrimitiveInfo::new(LayoutRect::new(
            LayoutPoint::new(100.0, 100.0),
            LayoutSize::new(100.0, 200.0)
        ));
        builder.push_rect(&info, ColorF::new(0.0, 1.0, 0.0, 1.0));

        let text_bounds = LayoutRect::new(
            LayoutPoint::new(100.0, 50.0),
            LayoutSize::new(700.0, 200.0)
        );
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

        let info = LayoutPrimitiveInfo::new(text_bounds);
        builder.push_text(
            &info,
            &glyphs,
            self.font_instance_key,
            ColorF::new(1.0, 1.0, 0.0, 1.0),
            None,
        );

        builder.pop_stacking_context();

        txn.set_display_list(
            self.epoch,
            None,
            layout_size,
            builder.finalize(),
            true,
        );
        txn.set_root_pipeline(self.pipeline_id);
        txn.generate_frame();
        self.api.send_transaction(self.document_id, txn);

        renderer.update();
        renderer.render(framebuffer_size).unwrap();
        #[cfg(feature = "gl")]
        self.window.swap_buffers().ok();

        false
    }

    fn deinit(self) {
        self.renderer.deinit();
    }
}

#[cfg(any(feature = "gfx-hal", feature = "gl"))]
fn main() {
    let mut win1 = Window::new("window1", ColorF::new(0.3, 0.0, 0.0, 1.0));
    let mut win2 = Window::new("window2", ColorF::new(0.0, 0.3, 0.0, 1.0));

    loop {
        if win1.tick() {
            break;
        }
        if win2.tick() {
            break;
        }
    }

    win1.deinit();
    win2.deinit();
}

#[cfg(not(any(feature = "gfx-hal", feature = "gl")))]
fn main() {
    println!("You need to enable one of the native API features (dx12/gl/metal/vulkan) in order to run this example.");
}

fn load_file(name: &str) -> Vec<u8> {
    let mut file = File::open(name).unwrap();
    let mut buffer = vec![];
    file.read_to_end(&mut buffer).unwrap();
    buffer
}
