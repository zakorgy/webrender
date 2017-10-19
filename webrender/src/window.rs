/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use backend::{self, Resources as R};
use backend_window::{self, Window};
use device::{BackendDevice, DeviceInitParams};
use gfx::{self, Factory};
use gfx::format::{Depth, DepthStencil, Formatted, Rgba8};
use gfx::memory::Typed;
use std;
use std::os::raw::c_void;
#[cfg(all(target_os = "windows", feature="dx11"))]
use winit;
#[cfg(not(feature = "dx11"))]
pub type ExistingWindow = backend_window::Window;
#[cfg(all(target_os = "windows", feature="dx11"))]
pub type ExistingWindow = winit::Window;

// The value of the type GL_FRAMEBUFFER_SRGB from https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_framebuffer_sRGB.txt
const GL_FRAMEBUFFER_SRGB: u32 = 0x8DB9;

#[cfg(not(feature = "dx11"))]
fn init_existing<Cf, Df>(glutin_window: ExistingWindow) ->
                            (Window, BackendDevice, backend::Factory,
                             gfx::handle::RenderTargetView<R, Cf>, gfx::handle::DepthStencilView<R, Df>)
where Cf: gfx::format::RenderFormat, Df: gfx::format::DepthFormat,
{
    unsafe { glutin_window.make_current().unwrap() };
    let (mut device, factory) = backend::create(|s|
        glutin_window.get_proc_address(s) as *const std::os::raw::c_void);

    unsafe { device.with_gl(|ref gl| gl.Disable(GL_FRAMEBUFFER_SRGB)); }

    let (width, height) = glutin_window.get_inner_size().unwrap();
    let aa = glutin_window.get_pixel_format().multisampling.unwrap_or(0) as gfx::texture::NumSamples;
    let dim = ((width as f32 * glutin_window.hidpi_factor()) as gfx::texture::Size,
               (height as f32 * glutin_window.hidpi_factor()) as gfx::texture::Size,
               1,
               aa.into());
    let (color_view, ds_view) = backend::create_main_targets_raw(dim, Cf::get_format().0, Df::get_format().0);
    (glutin_window, device, factory, Typed::new(color_view), Typed::new(ds_view))
}

#[cfg(all(target_os = "windows", feature="dx11"))]
fn init_existing<Cf, Df>(winit_window: ExistingWindow)
    -> (Window, BackendDevice, backend::Factory,
        gfx::handle::RenderTargetView<R, Cf>,
        gfx::handle::DepthStencilView<R, Df>)

where Cf: gfx::format::RenderFormat,
      Df: gfx::format::DepthFormat,
      <Df as gfx::format::Formatted>::Surface: gfx::format::TextureSurface,
      <Df as gfx::format::Formatted>::Channel: gfx::format::TextureChannel
{
    let (mut win, device, mut factory, main_color) = backend_window::init_existing_raw(winit_window, Cf::get_format()).unwrap();
    let main_depth = factory.create_depth_stencil_view_only(win.size.0, win.size.1).unwrap();
    let mut device = backend::Deferred::from(device);
    (win, device, factory, gfx::memory::Typed::new(main_color), main_depth)
}

pub fn create_rgba8_window(window: ExistingWindow)
    -> (Window, DeviceInitParams)
{
    let (window, device, factory, main_color, main_depth) = init_existing::<Rgba8, DepthStencil>(window);
    let params = DeviceInitParams {device, factory, main_color, main_depth};
    (window, params)
}

#[cfg(target_os = "linux")]
pub fn create_rgba8_headless<F>(get_proc_address: F, width: u32, height: u32)
    -> DeviceInitParams
    where F: Fn(&str) -> *const c_void
{
    let dimension = (width as u16, height as u16, 1, gfx::texture::AaMode::Single);
    let (device, factory) = backend::create(|s| get_proc_address(s));

    let (main_color, main_depth) = backend::create_main_targets_raw(dimension, Rgba8::get_format().0, Depth::get_format().0);
    DeviceInitParams {device, factory,
                      main_color: gfx::memory::Typed::new(main_color),
                      main_depth: gfx::memory::Typed::new(main_depth)}
}
