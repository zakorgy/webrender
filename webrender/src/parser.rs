/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use device::{PrimitiveInstance, Vertex};
use hal::pso::{AttributeDesc, DescriptorRangeDesc, DescriptorSetLayoutBinding, VertexBufferDesc};
use serde_json::{self, Error, Value};
use std::collections::HashMap;
use std::io::prelude::*;
use std::fs::File;
use std::mem;

const ATTRIBUTE_DESCRIPTORS: &'static str = "attributeDescriptors";
const DESCRIPTORS: &'static str = "descriptors";
const DESCRIPTOR_POOL: &'static str = "descriptorPool";
const DESCRIPTOR_SET_LAYOUTS: &'static str = "descriptorSetLayouts";
const BINDING: &'static str = "binding";
const NAME: &'static str = "name";
const RATE: &'static str = "rate";
const SETS: &'static str = "sets";
const STRIDE: &'static str = "stride";
const VERTEX_BUFFER_DESCRIPTORS: &'static str = "vertexBufferDescriptors";

pub fn read_json() -> Value {
    //include_bytes!(concat!(env!("OUT_DIR"), "/pipelines.json"))
    let mut file =
        File::open(concat!(env!("OUT_DIR"), "/pipelines.json")).expect("Unable to open the file");
    let mut content = String::new();
    file.read_to_string(&mut content)
        .expect("Unable to read the file");
    serde_json::from_str(&content).expect("Unable to create json")
}

pub fn create_attribute_descriptors(json: &Value, shader_name: &str) -> Vec<AttributeDesc> {
    let descriptors = &json[shader_name][ATTRIBUTE_DESCRIPTORS]
        .as_array()
        .expect("Unable to create array");
    descriptors
        .iter()
        .map(|d| {
            serde_json::from_str(&d.to_string()).expect("Unable to create attribute descriptor")
        })
        .collect::<Vec<AttributeDesc>>()
}

pub fn create_range_descriptors_and_set_count(
    json: &Value,
    shader_name: &str,
) -> (Vec<DescriptorRangeDesc>, usize) {
    let range_descriptors = &json[shader_name][DESCRIPTOR_POOL][DESCRIPTORS]
        .as_array()
        .expect("Unable to create array");
    let range_descs = range_descriptors
        .iter()
        .map(|d| serde_json::from_str(&d.to_string()).expect("Unable to create range descriptor"))
        .collect::<Vec<DescriptorRangeDesc>>();
    let set_count = *&json[shader_name][DESCRIPTOR_POOL][SETS]
        .as_i64()
        .expect("Unable to get number of sets") as usize;
    (range_descs, set_count)
}

pub fn create_descriptor_set_layout_bindings(
    json: &Value,
    shader_name: &str,
) -> (Vec<DescriptorSetLayoutBinding>, HashMap<String, usize>) {
    let descriptor_set_layouts = &json[shader_name][DESCRIPTOR_SET_LAYOUTS]
        .as_array()
        .expect("Unable to create array");
    let mut descriptors = HashMap::new();
    let ds_layouts = descriptor_set_layouts
        .iter()
        .map(|d| {
            descriptors.insert(d[NAME].as_str().unwrap().to_owned(), d[BINDING].as_i64().unwrap() as usize);
            serde_json::from_str(&d.to_string())
                .expect("Unable to create descriptor set layout bindings")
        })
        .collect::<Vec<DescriptorSetLayoutBinding>>();
    (ds_layouts, descriptors)
}

pub fn create_vertex_buffer_descriptors(json: &Value, shader_name: &str) -> Vec<VertexBufferDesc> {
    let vertex_buffer_descriptors_json = &json[shader_name][VERTEX_BUFFER_DESCRIPTORS]
        .as_array()
        .expect("Unable to create array");
    let mut vertex_buffer_descriptors = Vec::with_capacity(vertex_buffer_descriptors_json.len());
    for desc_json in vertex_buffer_descriptors_json.iter() {
        let stride = match desc_json[STRIDE].as_str().unwrap() {
            "Vertex" => mem::size_of::<Vertex>() as u32,
            "PrimitiveInstance" => mem::size_of::<PrimitiveInstance>() as u32,
            //TODO
            _ => 0,
        };
        let desc = VertexBufferDesc {
            stride: stride,
            rate: desc_json[RATE].as_u64().unwrap() as u8,
        };
        vertex_buffer_descriptors.push(desc);
    }
    vertex_buffer_descriptors
}
