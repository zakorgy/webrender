/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use internal_types::FastHashMap;

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use ron::de::from_reader;
use ron::ser::{to_string_pretty, PrettyConfig};

/// Helper struct to measure pipeline load times with gfx backend
pub struct PipelineProfiler {
    load_times: FastHashMap<String, (Vec<f64>, f64)>,
    current: String,
    start: Instant,
    pretty: PrettyConfig,
    ron_file_path: PathBuf,
}

impl PipelineProfiler {
    pub fn new(ron_file_path: PathBuf) -> Self {
        let mut profiler = PipelineProfiler {
            load_times: FastHashMap::default(),
            current: String::new(),
            start: Instant::now(),
            pretty: PrettyConfig {
                enumerate_arrays: true,
                .. ron::ser::PrettyConfig::default()
            },
            ron_file_path,
        };
        let file = match File::open(&profiler.ron_file_path) {
            Ok(f) => f,
            Err(_) => return profiler,
        };
        let load_times: FastHashMap<String, (Vec<f64>, f64)> = match from_reader(file) {
            Ok(lt) => lt,
            Err(e) => {
                println!("Failed to load data from file: {}", e);
                return profiler;
            }
        };
        profiler.load_times = load_times;
        profiler
    }

    pub fn start(&mut self, name: String) {
        self.current = name;
        self.start = Instant::now();
    }

    pub fn end(&mut self) {
        let duration = self.start.elapsed().as_float_secs();
        self.load_times.entry(self.current.clone())
            .and_modify(|e|  {
                e.0.push(duration);
                e.1 = e.0.iter().sum::<f64>() / e.0.len() as f64;
            })
            .or_insert((vec![duration], duration));
    }

    pub fn write_out(&self) {
        let mut file = OpenOptions::new().write(true).create(true).open(&self.ron_file_path).expect("File not found/created");
        let s = to_string_pretty(&self.load_times, self.pretty.clone()).expect("Serialization failed");
        file.write(s.as_bytes()).expect("File write failed");
    }
}
