/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::device::Device as _;
use api::channel::MsgReceiver;
use rendy_memory::{Heaps, MemoryBlock};
use smallvec::SmallVec;
use std::thread;
use std::sync::{Arc, Mutex};

pub enum DeviceMessage<B: hal::Backend> {
    // TODO: maybe add a vector of disposable
    Dispose(Disposable<B>),
    DisposeMultiple(SmallVec<[Disposable<B>; 32]>),
    Free,
    Exit,
}

pub enum Disposable<B: hal::Backend> {
    Image {
        view: B::ImageView,
        image: B::Image,
        memory: MemoryBlock<B>,
    },
    Buffer {
        buffer: B::Buffer,
        memory: MemoryBlock<B>,
    },
    Framebuffer {
        frame_buffer: B::Framebuffer,
        image_view: Option<B::ImageView>,
    },
}

impl<B: hal::Backend> Disposable<B> {
    unsafe fn dispose(self, device: &B::Device, heaps: &mut Heaps<B>) {
        let memory = match self {
            Disposable::Image { view, image, memory } => {
                device.destroy_image_view(view);
                device.destroy_image(image);
                memory
            },
            Disposable::Buffer { buffer, memory } => {
                device.destroy_buffer(buffer);
                memory
            },
            Disposable::Framebuffer {frame_buffer, image_view} => {
                device.destroy_framebuffer(frame_buffer);
                if let Some(view) = image_view {
                    device.destroy_image_view(view);
                }
                return
            },
        };
        heaps.free(device, memory);
    }
}

pub struct Destroyer<B: hal::Backend> {
    device: Arc<B::Device>,
    heaps: Arc<Mutex<Heaps<B>>>,
    disposable: Vec<Disposable<B>>,
    receiver: MsgReceiver<DeviceMessage<B>>,
}

impl<B: hal::Backend> Destroyer<B> {
    fn new(
        device: Arc<B::Device>,
        heaps: Arc<Mutex<Heaps<B>>>,
        receiver: MsgReceiver<DeviceMessage<B>>
    ) -> Self {
        Destroyer {
            device,
            heaps,
            disposable: Vec::new(),
            receiver,
        }
    }

    unsafe fn run(&mut self) {
        loop {
            match self.receiver.recv() {
                Ok(DeviceMessage::Dispose(d)) => self.disposable.push(d),
                Ok(DeviceMessage::DisposeMultiple(d)) => self.disposable.extend(d),
                Ok(DeviceMessage::Free) => self.dispose(),
                Err(_) | Ok(DeviceMessage::Exit) => {
                    self.dispose();
                    break;
                }
            }
        }
    }

    unsafe fn dispose(&mut self) {
        let heaps_locked = &mut *self.heaps.lock().unwrap();
        for d in self.disposable.drain(..) {
            d.dispose(&self.device.as_ref(), heaps_locked)
        }
    }

    pub unsafe fn start(
        device: Arc<B::Device>,
        heaps: Arc<Mutex<Heaps<B>>>,
        receiver: MsgReceiver<DeviceMessage<B>>
    ) -> thread::JoinHandle<()> {
        thread::Builder::new().name("WR resource desctroy thread".to_owned()).spawn(|| {
            let mut destroyer = Destroyer::new(
                device,
                heaps,
                receiver
            );
            destroyer.run();
        }).unwrap()
    }
}