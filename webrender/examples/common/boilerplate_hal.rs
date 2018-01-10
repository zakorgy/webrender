/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate env_logger;
extern crate euclid;
extern crate gfx_hal;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use std::env;
use std::path::PathBuf;
use webrender;
use webrender::api::*;
use winit;

struct Notifier {
    proxy: winit::EventsLoopProxy,
}

impl Notifier {
    fn new(proxy: winit::EventsLoopProxy) -> Notifier {
        Notifier { proxy }
    }
}

impl RenderNotifier for Notifier {
    fn clone(&self) -> Box<RenderNotifier> {
        Box::new(Notifier {
            proxy: self.proxy.clone(),
        })
    }


    fn wake_up(&self) {
        #[cfg(not(target_os = "android"))]
            self.proxy.wakeup().unwrap();
    }

    fn new_document_ready(&self, _: DocumentId, _scrolled: bool, _composite_needed: bool) {
        self.wake_up();
    }
}

pub trait HandyDandyRectBuilder {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect;
    fn by(&self, w: i32, h: i32) -> LayoutRect;
}
// Allows doing `(x, y).to(x2, y2)` or `(x, y).by(width, height)` with i32
// values to build a f32 LayoutRect
impl HandyDandyRectBuilder for (i32, i32) {
    fn to(&self, x2: i32, y2: i32) -> LayoutRect {
        LayoutRect::new(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new((x2 - self.0) as f32, (y2 - self.1) as f32),
        )
    }

    fn by(&self, w: i32, h: i32) -> LayoutRect {
        LayoutRect::new(
            LayoutPoint::new(self.0 as f32, self.1 as f32),
            LayoutSize::new(w as f32, h as f32),
        )
    }
}

pub trait Example {
    const TITLE: &'static str = "WebRender Sample App";
    const PRECACHE_SHADERS: bool = false;
    fn render(
        &mut self,
        api: &RenderApi,
        builder: &mut DisplayListBuilder,
        resources: &mut ResourceUpdates,
        framebuffer_size: DeviceUintSize,
        pipeline_id: PipelineId,
        document_id: DocumentId,
    );
    fn on_event(&mut self, winit::Event, &RenderApi, DocumentId) -> bool {
        false
    }
    /*fn get_image_handlers(
        &mut self,
        _gl: &gl::Gl,
    ) -> (Option<Box<webrender::ExternalImageHandler>>,
          Option<Box<webrender::OutputImageHandler>>) {
        (None, None)
    }
    fn draw_custom(&self, _gl: &gl::Gl) {
    }*/
}

pub fn main_wrapper<E: Example>(
    example: &mut E,
    options: Option<webrender::RendererOptions>,
) {
    env_logger::init().unwrap();
    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let mut events_loop = winit::EventsLoop::new();

    let wb = winit::WindowBuilder::new()
        .with_dimensions(1024, 768)
        .with_title(E::TITLE);

    let window = wb
        .build(&events_loop)
        .unwrap();

    let device_pixel_ratio = window.hidpi_factor();
    println!("Loading shaders...");
    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        debug: true,
        precache_shaders: E::PRECACHE_SHADERS,
        device_pixel_ratio,
        clear_color: Some(ColorF::new(0.3, 0.0, 0.0, 1.0)),
        //scatter_gpu_cache_updates: false,
        ..options.unwrap_or(webrender::RendererOptions::default())
    };

    let framebuffer_size = {
        let (width, height) = window.get_inner_size().unwrap();
        DeviceUintSize::new(width, height)
    };
    let notifier = Box::new(Notifier::new(events_loop.create_proxy()));
    let instance = back::Instance::create("gfx-rs instance", 1);
    let mut surface = instance.create_surface(&window);
    let (mut renderer, sender) = webrender::Renderer::new(notifier, opts, &window, &instance, &mut surface).unwrap();
    let api = sender.create_api();
    let document_id = api.add_document(framebuffer_size, 0);

    //let (external, output) = example.get_image_handlers(&*gl);

    /*if let Some(output_image_handler) = output {
        renderer.set_output_image_handler(output_image_handler);
    }

    if let Some(external_image_handler) = external {
        renderer.set_external_image_handler(external_image_handler);
    }*/

    let epoch = Epoch(0);
    let pipeline_id = PipelineId(0, 0);
    let layout_size = framebuffer_size.to_f32() / euclid::TypedScale::new(device_pixel_ratio);
    let mut builder = DisplayListBuilder::new(pipeline_id, layout_size);
    let mut resources = ResourceUpdates::new();

    example.render(
        &api,
        &mut builder,
        &mut resources,
        framebuffer_size,
        pipeline_id,
        document_id,
    );
    api.set_display_list(
        document_id,
        epoch,
        None,
        layout_size,
        builder.finalize(),
        true,
        resources,
    );
    api.set_root_pipeline(document_id, pipeline_id);
    api.generate_frame(document_id, None);

    println!("Entering event loop");
    events_loop.run_forever(|event| {
        match event {
            winit::Event::WindowEvent { event: winit::WindowEvent::Closed, .. } => {
                winit::ControlFlow::Break
            },

            /*glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Escape)) |
            glutin::Event::KeyboardInput(_, _, Some(glutin::VirtualKeyCode::Q)) => break 'outer,*/

            winit::Event::WindowEvent {
                window_id,
                event: winit::WindowEvent::KeyboardInput {
                    device_id,
                    input: winit::KeyboardInput {
                        scancode,
                        state: winit::ElementState::Pressed,
                        virtual_keycode: Some(winit::VirtualKeyCode::P),
                        modifiers
                    }
                },
            } => {
                let mut flags = renderer.get_debug_flags();
                flags.toggle(webrender::DebugFlags::PROFILER_DBG);
                renderer.set_debug_flags(flags);
                winit::ControlFlow::Continue
            },

            /*glutin::Event::KeyboardInput(glutin::ElementState::Pressed,pp
                                         _, Some(glutin::VirtualKeyCode::P)) => {
                let mut flags = renderer.get_debug_flags();
                flags.toggle(webrender::PROFILER_DBG);
                renderer.set_debug_flags(flags);
            }
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed,
                                         _, Some(glutin::VirtualKeyCode::O)) => {
                let mut flags = renderer.get_debug_flags();
                flags.toggle(webrender::RENDER_TARGET_DBG);
                renderer.set_debug_flags(flags);
            }
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed,
                                         _, Some(glutin::VirtualKeyCode::I)) => {
                let mut flags = renderer.get_debug_flags();
                flags.toggle(webrender::TEXTURE_CACHE_DBG);
                renderer.set_debug_flags(flags);
            }
            glutin::Event::KeyboardInput(glutin::ElementState::Pressed,
                                         _, Some(glutin::VirtualKeyCode::M)) => {
                api.notify_memory_pressure();
            }*/
            _ => {
                if example.on_event(event, &api, document_id) {
                    let mut builder = DisplayListBuilder::new(pipeline_id, layout_size);
                    let mut resources = ResourceUpdates::new();

                    example.render(
                        &api,
                        &mut builder,
                        &mut resources,
                        framebuffer_size,
                        pipeline_id,
                        document_id,
                    );
                    api.set_display_list(
                        document_id,
                        epoch,
                        None,
                        layout_size,
                        builder.finalize(),
                        true,
                        resources,
                    );
                    api.generate_frame(document_id, None);
                }
                renderer.update();
                renderer.render(framebuffer_size).unwrap();
                renderer.swap_buffers();
                winit::ControlFlow::Continue
            },
        }
    });

    renderer.deinit();
}