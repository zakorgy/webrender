/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use hal;
use hal::Device as BackendDevice;
use hal::DescriptorPool;
use hal::pso::{DescriptorRangeDesc, DescriptorSetLayoutBinding};
use internal_types::FastHashMap;

use super::PipelineRequirements;
use super::super::ShaderKind;

const DEBUG_DESCRIPTOR_COUNT: usize = 5;

struct DescPool<B: hal::Backend> {
    descriptor_pool: B::DescriptorPool,
    descriptor_set: Vec<B::DescriptorSet>,
    descriptor_set_layout: B::DescriptorSetLayout,
    current_descriptor_set_id: usize,
    max_descriptor_set_size: usize,
}

impl<B: hal::Backend> DescPool<B> {
    fn new(
        device: &B::Device,
        max_size: usize,
        descriptor_range_descriptors: Vec<DescriptorRangeDesc>,
        descriptor_set_layout: Vec<DescriptorSetLayoutBinding>,
    ) -> Self {
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                max_size,
                descriptor_range_descriptors.as_slice(),
                hal::pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("create_descriptor_pool failed");
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_set_layout, &[]) }
                .expect("create_descriptor_set_layout failed");
        let mut dp = DescPool {
            descriptor_pool,
            descriptor_set: vec![],
            descriptor_set_layout,
            current_descriptor_set_id: 0,
            max_descriptor_set_size: max_size,
        };
        dp.allocate();
        dp
    }

    fn descriptor_set(&self) -> &B::DescriptorSet {
        &self.descriptor_set[self.current_descriptor_set_id]
    }

    fn descriptor_set_layout(&self) -> &B::DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    fn next(&mut self) {
        self.current_descriptor_set_id += 1;
        assert!(
            self.current_descriptor_set_id < self.max_descriptor_set_size,
            "Maximum descriptor set size({}) exceeded!",
            self.max_descriptor_set_size
        );
        if self.current_descriptor_set_id == self.descriptor_set.len() {
            self.allocate();
        }
    }

    fn allocate(&mut self) {
        let desc_set = unsafe {
            self.descriptor_pool
                .allocate_set(&self.descriptor_set_layout)
        }
        .expect(&format!(
            "Failed to allocate set with layout: {:?}",
            self.descriptor_set_layout
        ));
        self.descriptor_set.push(desc_set);
    }

    fn reset(&mut self) {
        self.current_descriptor_set_id = 0;
    }

    fn deinit(self, device: &B::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout);
            device.destroy_descriptor_pool(self.descriptor_pool);
        }
    }
}

pub(super) struct DescriptorPools<B: hal::Backend> {
    debug_pool: DescPool<B>,
    cache_clip_pool: DescPool<B>,
    default_pool: DescPool<B>,
}

impl<B: hal::Backend> DescriptorPools<B> {
    pub(super) fn new(
        device: &B::Device,
        descriptor_count: usize,
        pipeline_requirements: &FastHashMap<String, PipelineRequirements>,
        set: usize,
    ) -> Self {
        fn increase_range_count(range: &mut Vec<DescriptorRangeDesc>, count: usize) {
            for r in range {
                r.count *= count;
            }
        }
        fn get_layout_and_range(
            pipeline: &PipelineRequirements,
            set: usize,
        ) -> (Vec<DescriptorSetLayoutBinding>, Vec<DescriptorRangeDesc>) {
            (
                pipeline.descriptor_set_layout_bindings[set].clone(),
                pipeline.descriptor_range_descriptors[set].clone(),
            )
        }

        let (debug_layout, mut debug_layout_range) = get_layout_and_range(
            pipeline_requirements
                .get("debug_color")
                .expect("debug_color missing"),
            set,
        );
        increase_range_count(&mut debug_layout_range, DEBUG_DESCRIPTOR_COUNT);

        let (cache_clip_layout, mut cache_clip_layout_range) = get_layout_and_range(
            pipeline_requirements
                .get("cs_clip_rectangle")
                .expect("cs_clip_rectangle missing"),
            set,
        );
        increase_range_count(&mut cache_clip_layout_range, descriptor_count);

        let (default_layout, mut default_layout_range) = get_layout_and_range(
            pipeline_requirements
                .get("brush_solid")
                .expect("brush_solid missing"),
            set,
        );
        increase_range_count(&mut default_layout_range, descriptor_count);

        DescriptorPools {
            debug_pool: DescPool::new(
                device,
                DEBUG_DESCRIPTOR_COUNT,
                debug_layout_range,
                debug_layout,
            ),
            cache_clip_pool: DescPool::new(
                device,
                descriptor_count,
                cache_clip_layout_range,
                cache_clip_layout,
            ),
            default_pool: DescPool::new(
                device,
                descriptor_count,
                default_layout_range,
                default_layout,
            ),
        }
    }

    fn get_pool(&self, shader_kind: &ShaderKind) -> &DescPool<B> {
        match *shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => &self.debug_pool,
            ShaderKind::ClipCache => &self.cache_clip_pool,
            _ => &self.default_pool,
        }
    }

    fn get_pool_mut(&mut self, shader_kind: &ShaderKind) -> &mut DescPool<B> {
        match *shader_kind {
            ShaderKind::DebugColor | ShaderKind::DebugFont => &mut self.debug_pool,
            ShaderKind::ClipCache => &mut self.cache_clip_pool,
            _ => &mut self.default_pool,
        }
    }

    pub(super) fn get(&self, shader_kind: &ShaderKind) -> &B::DescriptorSet {
        self.get_pool(shader_kind).descriptor_set()
    }

    pub(super) fn get_layout(&self, shader_kind: &ShaderKind) -> &B::DescriptorSetLayout {
        self.get_pool(shader_kind).descriptor_set_layout()
    }

    pub(super) fn next(&mut self, shader_kind: &ShaderKind) {
        self.get_pool_mut(shader_kind).next()
    }

    pub(super) fn reset(&mut self) {
        self.debug_pool.reset();
        self.cache_clip_pool.reset();
        self.default_pool.reset();
    }

    pub(super) fn deinit(self, device: &B::Device) {
        self.debug_pool.deinit(device);
        self.cache_clip_pool.deinit(device);
        self.default_pool.deinit(device);
    }
}

