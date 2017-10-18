/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use std::env;
use std::path::PathBuf;
use std::rc::Rc;
use webrender;
use webrender::api::*;
use winit;

struct Notifier {
    proxy: winit::EventsLoopProxy,
}

impl Notifier {
    fn new(proxy: winit::EventsLoopProxy) -> Notifier {
        Notifier {
            proxy,
        }
    }
}

impl RenderNotifier for Notifier {
    fn new_frame_ready(&mut self) {
        #[cfg(not(target_os = "android"))]
        self.proxy.wakeup().unwrap();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        #[cfg(not(target_os = "android"))]
        self.proxy.wakeup().unwrap();
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
        LayoutRect::new(LayoutPoint::new(self.0 as f32, self.1 as f32),
                        LayoutSize::new((x2 - self.0) as f32, (y2 - self.1) as f32))
    }

    fn by(&self, w: i32, h: i32) -> LayoutRect {
        LayoutRect::new(LayoutPoint::new(self.0 as f32, self.1 as f32),
                        LayoutSize::new(w as f32, h as f32))
    }
}

pub trait Example {
    fn render(&mut self,
              api: &RenderApi,
              builder: &mut DisplayListBuilder,
              resources: &mut ResourceUpdates,
              layout_size: LayoutSize,
              pipeline_id: PipelineId,
              document_id: DocumentId);
    fn on_event(&mut self,
                event: winit::Event,
                api: &RenderApi,
                document_id: DocumentId) -> bool;
    fn get_external_image_handler(&self) -> Option<Box<webrender::ExternalImageHandler>> {
        None
    }
}

pub fn main_wrapper(example: &mut Example,
                    options: Option<webrender::RendererOptions>)
{
    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let mut events_loop = winit::EventsLoop::new();
    let winit_window = winit::WindowBuilder::new()
                       .with_title("WebRender Sample D3D11")
                       .build(&events_loop)
                       .unwrap();
    println!("Shader resource path: {:?}", res_path);

    let (width, height) = winit_window.get_inner_size_pixels().unwrap();

    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        debug: true,
        precache_shaders: true,
        device_pixel_ratio: winit_window.hidpi_factor(),
        enable_dithering: true,
        .. options.unwrap_or(webrender::RendererOptions::default())
    };

    let (window, mut device_init_params) = webrender::create_rgba8_window(winit_window);
    let size = DeviceUintSize::new(width, height);
    let (mut renderer, sender) = webrender::Renderer::new(opts, device_init_params).unwrap();
    let api = sender.create_api();
    let document_id = api.add_document(size);

    let notifier = Box::new(Notifier::new(events_loop.create_proxy()));
    renderer.set_render_notifier(notifier);

    if let Some(external_image_handler) = example.get_external_image_handler() {
        renderer.set_external_image_handler(external_image_handler);
    }

    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

    let pipeline_id = PipelineId(0, 0);
    let layout_size = LayoutSize::new(width as f32, height as f32);
    let mut builder = DisplayListBuilder::new(pipeline_id, layout_size);
    let mut resources = ResourceUpdates::new();

    example.render(&api, &mut builder, &mut resources, layout_size, pipeline_id, document_id);
    api.set_display_list(
        document_id,
        epoch,
        Some(root_background_color),
        LayoutSize::new(width as f32, height as f32),
        builder.finalize(),
        true,
        resources
    );
    api.set_root_pipeline(document_id, pipeline_id);
    api.generate_frame(document_id, None);

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
                flags.toggle(webrender::PROFILER_DBG);
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

                    example.render(&api, &mut builder, &mut resources, layout_size, pipeline_id, document_id);
                    api.set_display_list(
                        document_id,
                        epoch,
                        Some(root_background_color),
                        LayoutSize::new(width as f32, height as f32),
                        builder.finalize(),
                        true,
                        resources
                    );
                    api.generate_frame(document_id, None);
                }
                renderer.update();
                renderer.render(DeviceUintSize::new(width, height)).unwrap();
                window.swap_buffers(1);
                winit::ControlFlow::Continue
            },
        }
    });

    renderer.deinit();
}
