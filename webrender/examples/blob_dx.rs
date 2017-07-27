/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

extern crate app_units;
extern crate euclid;
extern crate winit;
extern crate webrender;
extern crate webrender_traits;
extern crate rayon;

use app_units::Au;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use webrender_traits::{ClipRegionToken, ColorF, DisplayListBuilder, Epoch, GlyphInstance};
use webrender_traits::{DeviceIntPoint, DeviceUintSize, LayoutPoint, LayoutRect, LayoutSize};
use webrender_traits::{ImageData, ImageDescriptor, ImageFormat};
use webrender_traits::{PipelineId, RenderApi, TransformStyle, BoxShadowClipMode};
use euclid::vec2;

use rayon::ThreadPool;
use rayon::Configuration as ThreadPoolConfig;
use std::collections::hash_map::Entry;
use std::sync::Arc;
use std::sync::mpsc::{channel, Sender, Receiver};
use webrender_traits as wt;

// This example shows how to implement a very basic BlobImageRenderer that can only render
// a checkerboard pattern.

// The deserialized command list internally used by this example is just a color.
type ImageRenderingCommands = wt::ColorU;

// Serialize/deserialze the blob.
// Ror real usecases you should probably use serde rather than doing it by hand.

fn serialize_blob(color: wt::ColorU) -> Vec<u8> {
    vec![color.r, color.g, color.b, color.a]
}

fn deserialize_blob(blob: &[u8]) -> Result<ImageRenderingCommands, ()> {
    let mut iter = blob.iter();
    return match (iter.next(), iter.next(), iter.next(), iter.next()) {
        (Some(&r), Some(&g), Some(&b), Some(&a)) => Ok(wt::ColorU::new(r, g, b, a)),
        (Some(&a), None, None, None) => Ok(wt::ColorU::new(a, a, a, a)),
        _ => Err(()),
    }
}

// This is the function that applies the deserialized drawing commands and generates
// actual image data.
fn render_blob(
    commands: Arc<ImageRenderingCommands>,
    descriptor: &wt::BlobImageDescriptor,
    tile: Option<wt::TileOffset>,
) -> wt::BlobImageResult {
    let color = *commands;

    // Allocate storage for the result. Right now the resource cache expects the
    // tiles to have have no stride or offset.
    let mut texels = Vec::with_capacity((descriptor.width * descriptor.height * 4) as usize);

    // Generate a per-tile pattern to see it in the demo. For a real use case it would not
    // make sense for the rendered content to depend on its tile.
    let tile_checker = match tile {
        Some(tile) => (tile.x % 2 == 0) != (tile.y % 2 == 0),
        None => true,
    };

    for y in 0..descriptor.height {
        for x in 0..descriptor.width {
            // Apply the tile's offset. This is important: all drawing commands should be
            // translated by this offset to give correct results with tiled blob images.
            let x2 = x + descriptor.offset.x as u32;
            let y2 = y + descriptor.offset.y as u32;

            // Render a simple checkerboard pattern
            let checker = if (x2 % 20 >= 10) != (y2 % 20 >= 10) { 1 } else { 0 };
            // ..nested in the per-tile cherkerboard pattern
            let tc = if tile_checker { 0 } else { (1 - checker) * 40 };

            match descriptor.format {
                wt::ImageFormat::BGRA8 => {
                    texels.push(color.b * checker + tc);
                    texels.push(color.g * checker + tc);
                    texels.push(color.r * checker + tc);
                    texels.push(color.a * checker + tc);
                }
                wt::ImageFormat::A8 => {
                    texels.push(color.a * checker + tc);
                }
                _ => {
                    return Err(wt::BlobImageError::Other(format!(
                        "Usupported image format {:?}",
                        descriptor.format
                    )));
                }
            }
        }
    }

    Ok(wt::RasterizedBlobImage {
        data: texels,
        width: descriptor.width,
        height: descriptor.height,
    })
}

struct CheckerboardRenderer {
    // We are going to defer the rendering work to worker threads.
    // Using a pre-built Arc<ThreadPool> rather than creating our own threads
    // makes it possible to share the same thread pool as the glyph renderer (if we
    // want to).
    workers: Arc<ThreadPool>,

    // the workers will use an mpsc channel to communicate the result.
    tx: Sender<(wt::BlobImageRequest, wt::BlobImageResult)>,
    rx: Receiver<(wt::BlobImageRequest, wt::BlobImageResult)>,

    // The deserialized drawing commands.
    // In this example we store them in Arcs. This isn't necessary since in this simplified
    // case the command list is a simple 32 bits value and would be cheap to clone before sending
    // to the workers. But in a more realistic scenario the commands would typically be bigger
    // and more expensive to clone, so let's pretend it is also the case here.
    image_cmds: HashMap<wt::ImageKey, Arc<ImageRenderingCommands>>,

    // The images rendered in the current frame (not kept here between frames).
    rendered_images: HashMap<wt::BlobImageRequest, Option<wt::BlobImageResult>>,
}

impl CheckerboardRenderer {
    fn new(workers: Arc<ThreadPool>) -> Self {
        let (tx, rx) = channel();
        CheckerboardRenderer {
            image_cmds: HashMap::new(),
            rendered_images: HashMap::new(),
            workers: workers,
            tx: tx,
            rx: rx,
        }
    }
}

impl wt::BlobImageRenderer for CheckerboardRenderer {
    fn add(&mut self, key: wt::ImageKey, cmds: wt::BlobImageData, _: Option<wt::TileSize>) {
        self.image_cmds.insert(key, Arc::new(deserialize_blob(&cmds[..]).unwrap()));
    }

    fn update(&mut self, key: wt::ImageKey, cmds: wt::BlobImageData) {
        // Here, updating is just replacing the current version of the commands with
        // the new one (no incremental updates).
        self.image_cmds.insert(key, Arc::new(deserialize_blob(&cmds[..]).unwrap()));
    }

    fn delete(&mut self, key: wt::ImageKey) {
        self.image_cmds.remove(&key);
    }

    fn request(&mut self,
               resources: &wt::BlobImageResources,
               request: wt::BlobImageRequest,
               descriptor: &wt::BlobImageDescriptor,
               _dirty_rect: Option<wt::DeviceUintRect>) {
        // This method is where we kick off our rendering jobs.
        // It should avoid doing work on the calling thread as much as possible.
        // In this example we will use the thread pool to render individual tiles.

        // Gather the input data to send to a worker thread.
        let cmds = Arc::clone(&self.image_cmds.get(&request.key).unwrap());
        let tx = self.tx.clone();
        let descriptor = descriptor.clone();

        self.workers.spawn(move || {
            let result = render_blob(cmds, &descriptor, request.tile);
            tx.send((request, result)).unwrap();
        });

        // Add None in the map of rendered images. This makes it possible to differentiate
        // between commands that aren't finished yet (entry in the map is equal to None) and
        // keys that have never been requested (entry not in the map), which would cause deadlocks
        // if we were to block upon receing their result in resolve!
        self.rendered_images.insert(request, None);
    }

    fn resolve(&mut self, request: wt::BlobImageRequest) -> wt::BlobImageResult {
        // In this method we wait until the work is complete on the worker threads and
        // gather the results.

        // First look at whether we have already received the rendered image
        // that we are looking for.
        match self.rendered_images.entry(request) {
            Entry::Vacant(_) => {
                return Err(wt::BlobImageError::InvalidKey);
            }
            Entry::Occupied(entry) => {
                // None means we haven't yet received the result.
                if entry.get().is_some() {
                    let result = entry.remove();
                    return result.unwrap();
                }
            }
        }

        // We haven't received it yet, pull from the channel until we receive it.
        while let Ok((req, result)) = self.rx.recv() {
            if req == request {
                // There it is!
                return result
            }
            self.rendered_images.insert(req, Some(result));
        }

        // If we break out of the loop above it means the channel closed unexpectedly.
        Err(wt::BlobImageError::Other("Channel closed".into()))
    }
    fn delete_font(&mut self, font: wt::FontKey) {}
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
        self.proxy.wakeup().unwrap();
    }

    fn new_scroll_frame_ready(&mut self, _composite_needed: bool) {
        #[cfg(not(target_os = "android"))]
        self.proxy.wakeup().unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let res_path = if args.len() > 1 {
        Some(PathBuf::from(&args[1]))
    } else {
        None
    };

    let mut events_loop = winit::EventsLoop::new();
    let window = winit::WindowBuilder::new()
                .with_title("WebRender Sample")
                .build(&events_loop)
                .unwrap();

    let (width, height) = window.get_inner_size_pixels().unwrap();

    let worker_config = ThreadPoolConfig::new().thread_name(|idx|{
        format!("WebRender:Worker#{}", idx)
    });

    let workers = Arc::new(ThreadPool::new(worker_config).unwrap());

    let opts = webrender::RendererOptions {
        resource_override_path: res_path,
        debug: true,
        precache_shaders: true,
        device_pixel_ratio: window.hidpi_factor(),
        blob_image_renderer: Some(Box::new(CheckerboardRenderer::new(Arc::clone(&workers)))),
        .. Default::default()
    };

    let size = DeviceUintSize::new(width, height);
    let (mut renderer, sender, mut window) = webrender::renderer::Renderer::new(window, opts, size).unwrap();
    let api = sender.create_api();

    let notifier = Box::new(Notifier::new(events_loop.create_proxy()));
    renderer.set_render_notifier(notifier);

    let epoch = Epoch(0);
    let root_background_color = ColorF::new(0.3, 0.0, 0.0, 1.0);

    let pipeline_id = PipelineId(0, 0);
    let layout_size = LayoutSize::new(width as f32, height as f32);
    let mut builder = wt::DisplayListBuilder::new(pipeline_id, layout_size);

    let blob_img1 = api.generate_image_key();
    api.add_image(
        blob_img1,
        wt::ImageDescriptor::new(500, 500, wt::ImageFormat::BGRA8, true),
        wt::ImageData::new_blob_image(serialize_blob(wt::ColorU::new(50, 50, 150, 255))),
        Some(128),
    );

    let blob_img2 = api.generate_image_key();
    api.add_image(
        blob_img2,
        wt::ImageDescriptor::new(200, 200, wt::ImageFormat::BGRA8, true),
        wt::ImageData::new_blob_image(serialize_blob(wt::ColorU::new(50, 150, 50, 255))),
        None,
    );

    let bounds = wt::LayoutRect::new(wt::LayoutPoint::zero(), layout_size);
    builder.push_stacking_context(wt::ScrollPolicy::Scrollable,
                                  bounds,
                                  None,
                                  wt::TransformStyle::Flat,
                                  None,
                                  wt::MixBlendMode::Normal,
                                  Vec::new());

    let clip = builder.push_clip_region(&bounds, vec![], None);
    builder.push_image(
        (30, 30).by(500, 500),
        clip,
        wt::LayoutSize::new(500.0, 500.0),
        wt::LayoutSize::new(0.0, 0.0),
        wt::ImageRendering::Auto,
        blob_img1,
    );

    let clip = builder.push_clip_region(&bounds, vec![], None);
    builder.push_image(
        (600, 600).by(200, 200),
        clip,
        wt::LayoutSize::new(200.0, 200.0),
        wt::LayoutSize::new(0.0, 0.0),
        wt::ImageRendering::Auto,
        blob_img2,
    );    

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
                window.swap_buffers(1);
                winit::ControlFlow::Continue
            },
        }
    });
}
