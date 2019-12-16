/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal::device::Device as BackendDevice;
use hal::pool::CommandPool as HalCommandPool;

pub struct CommandPool<B: hal::Backend> {
    command_pool: B::CommandPool,
    command_buffers: Vec<B::CommandBuffer>,
    next_id: usize,
}

impl<B: hal::Backend> CommandPool<B> {
    pub(super) fn new(mut command_pool: B::CommandPool) -> Self {
        let command_buffer =
            unsafe { command_pool.allocate_one(hal::command::Level::Primary) };
        CommandPool {
            command_pool,
            command_buffers: vec![command_buffer],
            next_id: 0,
        }
    }

    /*pub fn step(&mut self) {
        if self.next_id >= self.command_buffers.len() {
            let command_buffer =
                unsafe { self.command_pool.allocate_one(hal::command::Level::Primary) };
            self.command_buffers.push(command_buffer);
        }
        self.next_id += 1;
    }*/

    pub fn remove_cmd_buffer(&mut self) -> B::CommandBuffer {
        if self.next_id >= self.command_buffers.len() {
            unsafe { self.command_pool.allocate_one(hal::command::Level::Primary) }
        } else {
            self.command_buffers.pop().unwrap()
        }
    }

    pub fn return_cmd_buffer(&mut self, cmd_buffer: B::CommandBuffer) {
        self.command_buffers.insert(self.next_id, cmd_buffer);
        self.next_id += 1;
    }

    pub(super) unsafe fn reset(&mut self) {
        self.command_pool.reset(false);
        self.next_id = 0;
    }

    pub(super) unsafe fn destroy(self, device: &B::Device) {
        device.destroy_command_pool(self.command_pool);
    }
}
