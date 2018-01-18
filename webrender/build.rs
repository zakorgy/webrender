/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[macro_use]
extern crate serde_json;

use serde_json::value::Value as JsonValue;
use std::cmp::max;
use std::env;
use std::fs::{canonicalize, read_dir, File};
use std::io::BufReader;
use std::io::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{self, Command, Stdio};

const SHADER_IMPORT: &str = "#include ";
const SHADER_KIND_FRAGMENT: &str = "#define WR_FRAGMENT_SHADER\n";
const SHADER_KIND_VERTEX: &str = "#define WR_VERTEX_SHADER\n";
const SHADER_PREFIX: &str = "#define WR_MAX_VERTEX_TEXTURE_WIDTH 1024\n";
const SHADER_VERSION_VK: &'static str = "#version 450\n";
const VK_EXTENSIONS: &'static str = "#extension GL_ARB_shading_language_420pack : enable\n\
                                     #extension GL_ARB_explicit_attrib_location : enable\n\
                                     #extension GL_ARB_separate_shader_objects : enable\n";

struct Shader {
    name: &'static str,
    source_name: &'static str,
    features: &'static [&'static str],
}

const SHADERS: &[Shader] = &[
    // Cache shaders
    Shader {
        name: "cs_text_run",
        source_name: "cs_text_run",
        features: &[""],
    },
    Shader {
        name: "cs_line",
        source_name: "ps_line",
        features: &["CACHE"],
    },
    Shader {
        name: "cs_blur_a8",
        source_name: "cs_blur",
        features: &["ALPHA_TARGET"],
    },
    Shader {
        name: "cs_blur_rgba8",
        source_name: "cs_blur",
        features: &["COLOR_TARGET"],
    },
    // Brush shaders
    Shader {
        name: "brush_mask_corner",
        source_name: "brush_mask_corner",
        features: &[""],
    },
    Shader {
        name: "brush_mask_rounded_rect",
        source_name: "brush_mask_rounded_rect",
        features: &[""],
    },
    Shader {
        name: "brush_image",
        source_name: "brush_image",
        features: &[
            "ALPHA_TARGET",
            "ALPHA_TARGET, ALPHA_PASS",
            "COLOR_TARGET",
            "COLOR_TARGET, ALPHA_PASS",
            "COLOR_TARGET_ALPHA_MASK",
            "COLOR_TARGET_ALPHA_MASK, ALPHA_PASS",
        ],
    },
    Shader {
        name: "brush_solid",
        source_name: "brush_solid",
        features: &["", "ALPHA_PASS"],
    },
    // Cache clip shaders
    Shader {
        name: "cs_clip_rectangle",
        source_name: "cs_clip_rectangle",
        features: &["TRANSFORM"],
    },
    Shader {
        name: "cs_clip_image",
        source_name: "cs_clip_image",
        features: &["TRANSFORM"],
    },
    Shader {
        name: "cs_clip_border",
        source_name: "cs_clip_border",
        features: &["TRANSFORM"],
    },
    // Prim shaders
    Shader {
        name: "ps_line",
        source_name: "ps_line",
        features: &["", "TRANSFORM"],
    },
    Shader {
        name: "ps_border_corner",
        source_name: "ps_border_corner",
        features: &["", "TRANSFORM"],
    },
    Shader {
        name: "ps_border_edge",
        source_name: "ps_border_edge",
        features: &["", "TRANSFORM"],
    },
    Shader {
        name: "ps_gradient",
        source_name: "ps_gradient",
        features: &["", "TRANSFORM", "DITHERING", "DITHERING,TRANSFORM"],
    },
    Shader {
        name: "ps_angle_gradient",
        source_name: "ps_angle_gradient",
        features: &["", "TRANSFORM", "DITHERING", "DITHERING,TRANSFORM"],
    },
    Shader {
        name: "ps_radial_gradient",
        source_name: "ps_radial_gradient",
        features: &["", "TRANSFORM", "DITHERING", "DITHERING,TRANSFORM"],
    },
    Shader {
        name: "ps_blend",
        source_name: "ps_blend",
        features: &[""],
    },
    Shader {
        name: "ps_composite",
        source_name: "ps_composite",
        features: &[""],
    },
    Shader {
        name: "ps_hardware_composite",
        source_name: "ps_hardware_composite",
        features: &[""],
    },
    Shader {
        name: "ps_split_composite",
        source_name: "ps_split_composite",
        features: &[""],
    },
    Shader {
        name: "ps_image",
        source_name: "ps_image",
        features: &["", "TRANSFORM"],
    },
    Shader {
        name: "ps_yuv_image",
        source_name: "ps_yuv_image",
        features: &[
            "NV12",
            "",
            "INTERLEAVED_Y_CB_CR",
            "NV12,YUV_REC709",
            "YUV_REC709",
            "INTERLEAVED_Y_CB_CR,YUV_REC709",
            "NV12,TRANSFORM",
            "TRANSFORM",
            "INTERLEAVED_Y_CB_CR,TRANSFORM",
            "NV12,YUV_REC709,TRANSFORM",
            "YUV_REC709,TRANSFORM",
            "INTERLEAVED_Y_CB_CR,YUV_REC709,TRANSFORM",
        ],
    },
    Shader {
        name: "ps_text_run",
        source_name: "ps_text_run",
        features: &["", "TRANSFORM", "GLYPH_TRANSFORM"],
    },
    // Debug shaders
    Shader {
        name: "debug_color",
        source_name: "debug_color",
        features: &[""],
    },
    Shader {
        name: "debug_font",
        source_name: "debug_font",
        features: &[""],
    },
];

fn create_shaders(out_dir: &str, shaders: &HashMap<String, String>) -> Vec<String> {
    fn get_shader_source(shader_name: &str, shaders: &HashMap<String, String>) -> Option<String> {
        if let Some(shader_file) = shaders.get(shader_name) {
            let shader_file_path = Path::new(shader_file);
            if let Ok(mut shader_source_file) = File::open(shader_file_path) {
                let mut source = String::new();
                shader_source_file.read_to_string(&mut source).unwrap();
                Some(source)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn parse_shader_source(source: &str, shaders: &HashMap<String, String>, output: &mut String) {
        for line in source.lines() {
            if line.starts_with(SHADER_IMPORT) {
                let imports = line[SHADER_IMPORT.len() ..].split(",");
                // For each import, get the source, and recurse.
                for import in imports {
                    if let Some(include) = get_shader_source(import, shaders) {
                        parse_shader_source(&include, shaders, output);
                    }
                }
            } else {
                output.push_str(line);
                output.push_str("\n");
            }
        }
    }

    fn build_shader_strings(
        base_filename: &str,
        features: &str,
        shaders: &HashMap<String, String>,
    ) -> (String, String) {
        // Construct a list of strings to be passed to the shader compiler.
        let mut vs_source = String::new();
        let mut fs_source = String::new();

        vs_source.push_str(SHADER_VERSION_VK);
        fs_source.push_str(SHADER_VERSION_VK);

        // Define a constant depending on whether we are compiling VS or FS.
        vs_source.push_str(SHADER_KIND_VERTEX);
        fs_source.push_str(SHADER_KIND_FRAGMENT);

        // Add any defines that were passed by the caller.
        vs_source.push_str(features);
        fs_source.push_str(features);

        // Parse the main .glsl file, including any imports
        // and append them to the list of sources.
        let mut shared_result = String::new();
        if let Some(shared_source) = get_shader_source(base_filename, shaders) {
            parse_shader_source(&shared_source, shaders, &mut shared_result);
        }

        //vs_source.push_str(SHADER_LINE_MARKER);
        vs_source.push_str(&shared_result);
        //fs_source.push_str(SHADER_LINE_MARKER);
        fs_source.push_str(&shared_result);

        (vs_source, fs_source)
    }

    let mut file_names = Vec::new();
    for shader in SHADERS {
        for config in shader.features {
            let mut features = String::new();

            features.push_str(SHADER_PREFIX);
            features.push_str(format!("//Source: {}.glsl\n", shader.source_name).as_str());

            let mut file_name_postfix = String::new();
            for feature in config.split(",") {
                if !feature.is_empty() {
                    features.push_str(&format!("#define WR_FEATURE_{}\n", feature));
                    if shader.name == shader.source_name {
                        file_name_postfix
                            .push_str(&format!("_{}", feature.to_lowercase().as_str()));
                    }
                }
            }

            features.push_str(VK_EXTENSIONS);

            let (mut vs_source, mut fs_source) =
                build_shader_strings(shader.source_name, &features, shaders);

            let mut filename = String::from(shader.name);
            filename.push_str(file_name_postfix.as_str());
            let (mut vs_name, mut fs_name) = (filename.clone(), filename);
            vs_name.push_str(".vert");
            fs_name.push_str(".frag");
            println!("vs_name = {}, shader.name = {}", vs_name, shader.name);
            let (vs_file_path, fs_file_path) = (
                Path::new(out_dir).join(vs_name.clone()),
                Path::new(out_dir).join(fs_name.clone()),
            );
            let (mut vs_file, mut fs_file) = (
                File::create(vs_file_path).unwrap(),
                File::create(fs_file_path).unwrap(),
            );
            write!(vs_file, "{}", vs_source).unwrap();
            write!(fs_file, "{}", fs_source).unwrap();
            file_names.push(vs_name);
            file_names.push(fs_name);
        }
    }
    file_names
}

fn write_shaders(glsl_files: Vec<PathBuf>, shader_file_path: &Path) -> HashMap<String, String> {
    let mut shader_file = File::create(shader_file_path).unwrap();
    let mut shader_map: HashMap<String, String> = HashMap::with_capacity(glsl_files.len());

    write!(shader_file, "/// AUTO GENERATED BY build.rs\n\n").unwrap();
    write!(shader_file, "use std::collections::HashMap;\n").unwrap();
    write!(shader_file, "lazy_static! {{\n").unwrap();
    write!(
        shader_file,
        "  pub static ref SHADERS: HashMap<&'static str, &'static str> = {{\n"
    ).unwrap();
    write!(shader_file, "    let mut h = HashMap::new();\n").unwrap();
    for glsl in glsl_files {
        let shader_name = glsl.file_name().unwrap().to_str().unwrap();
        // strip .glsl
        let shader_name = shader_name.replace(".glsl", "");
        let full_path = canonicalize(&glsl).unwrap();
        let full_name = full_path.as_os_str().to_str().unwrap();
        // if someone is building on a network share, I'm sorry.
        let full_name = full_name.replace("\\\\?\\", "");
        let full_name = full_name.replace("\\", "/");
        shader_map.insert(shader_name.clone(), full_name.clone());
        write!(
            shader_file,
            "    h.insert(\"{}\", include_str!(\"{}\"));\n",
            shader_name,
            full_name
        ).unwrap();
    }
    write!(shader_file, "    h\n").unwrap();
    write!(shader_file, "  }};\n").unwrap();
    write!(shader_file, "}}\n").unwrap();
    shader_map
}

fn process_glsl_for_spirv(file_path: &Path, file_name: &str) -> Option<JsonValue> {
    let mut new_data = String::new();
    let mut binding = 0;
    let mut in_location = 0;
    let mut out_location = 0;
    let mut attribute_descriptors: Vec<JsonValue> = Vec::new();
    let mut descriptor_set_layouts = Vec::new();
    let mut vertex_offset = 0;
    let mut instance_offset = 0;
    let mut pipeline_requirmenets = HashMap::new();
    // Since the .vert and .frag files for the same shader use the same layout qualifiers
    // we extract layout datas from .vert files only.
    let write_json = file_name.ends_with(".vert");

    // Mapping from glsl sampler variable name to a tuple,
    // in which the first item is the corresponding expression used in vulkan glsl files,
    // the second is the layout binding index.
    // e.g.: sColor0 -> ("sampler2DArray(tColor0, sColor0)", 0)
    let mut sampler_mapping: HashMap<String, (String, i8)> = HashMap::new();

    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let mut l = line.unwrap();
        let trimmed = l.trim_left();

        // Replace uniforms in shader:
        //      Sampler uniforms are splitted to texture + sampler.
        //      Other types occur in a group. These are replaced with a static string for now.
        if trimmed.starts_with("uniform") {
            if trimmed.contains("sampler") {
                let code = split_code(trimmed);
                replace_sampler_definition_with_texture_and_sampler(
                    &mut binding,
                    code,
                    &mut descriptor_set_layouts,
                    &mut new_data,
                    &mut sampler_mapping,
                    write_json,
                );

            // Replace non-sampler uniforms with a structure.
            // We just place a predifened structure to the position of the last non-uniform
            // variable (uDevicePixelRatio), since all shader uses the same variables.
            } else if trimmed.starts_with("uniform float uDevicePixelRatio") {
                replace_non_sampler_uniforms(&mut new_data)
            }

        // Adding location info for non-uniform variables.
        } else if trimmed.contains(';') && // If the line contains a semicolon we assume it is a variable declaration.
            (trimmed.starts_with("varying ") || trimmed.starts_with("flat varying ")
                || trimmed.starts_with("in ") || trimmed.starts_with("out "))
        {
            extend_non_uniform_variables_with_location_info(
                &mut attribute_descriptors,
                file_name,
                &mut in_location,
                &mut instance_offset,
                trimmed,
                &mut new_data,
                &mut out_location,
                &mut vertex_offset,
                write_json,
            );
        // Replacing sampler variables with the corresponding expression from sampler_mapping.
        } else if l.contains("TEX_SAMPLE(") || l.contains("TEXEL_FETCH(")
            || l.contains("texelFetch(") || l.contains("texture(")
            || l.contains("textureLod(") || l.contains("textureSize(")
        {
            let mut line = l.clone();
            for (k, v) in sampler_mapping.iter() {
                if line.contains(k) {
                    line = line.replace(k, &v.0);
                }
            }
            new_data.push_str(&line);
            new_data.push('\n');
        } else {
            new_data.push_str(&l);
            new_data.push('\n');
        }
    }
    let mut file = File::create(file_path).unwrap();
    file.write(new_data.as_bytes()).unwrap();
    if write_json {
        let descriptor_pool = create_desciptor_pool_json(sampler_mapping.len() as i8);
        let vertex_buffer_descriptors = create_vertex_buffer_descriptors_json(file_name);

        pipeline_requirmenets.insert("attributeDescriptors", json!(attribute_descriptors));
        pipeline_requirmenets.insert("descriptorPool", descriptor_pool);
        pipeline_requirmenets.insert("descriptorSetLayouts", json!(descriptor_set_layouts));
        pipeline_requirmenets.insert("vertexBufferDescriptors", json!(vertex_buffer_descriptors));
        return Some(json!({
            file_name.trim_right_matches(".vert"): pipeline_requirmenets
        }));
    }
    None
}

fn split_code(line: &str) -> Vec<&str> {
    line.split(';').collect::<Vec<&str>>()[0]
        .split(' ')
        .collect::<Vec<&str>>()
}

fn replace_sampler_definition_with_texture_and_sampler(
    binding: &mut i8,
    code: Vec<&str>,
    descriptor_set_layouts: &mut Vec<JsonValue>,
    new_data: &mut String,
    sampler_mapping: &mut HashMap<String, (String, i8)>,
    write_json: bool,
) {
    // Get the name of the sampler.
    let (sampler_name, code) = code.split_last().unwrap();

    // Get the exact type of the sampler.
    let (sampler_type, code) = code.split_last().unwrap();
    let mut code_str = String::new();
    for i in 0 .. code.len() {
        code_str.push_str(code[i]);
        code_str.push(' ');
    }

    let texture_type = sampler_type.replace("sampler", "texture");
    let texture_name = sampler_name.replacen('s', "t", 1);

    // If the sampler is redefined we use the same binding index, but the sampler type gets updated.
    // Note: This should be handled by parsing the defines and use them to determine which definition is correct.
    //       Since only sColor samplers are involved in this case, the last definition for these samplers
    //       which uses sampler/texture2DArray is good for us to go with.
    if let Some(&(_, binding)) = sampler_mapping.get(*sampler_name) {
        let mut layout_str = format!(
            "layout(set = 0, binding = {}) {}{} {};\n",
            binding, code_str, texture_type, texture_name
        );
        new_data.push_str(&layout_str);
        sampler_mapping.insert(
            String::from(*sampler_name),
            (
                format!("{}({}, {})", sampler_type, texture_name, sampler_name),
                binding,
            ),
        );

        layout_str = format!(
            "layout(set = 0, binding = {}) {}sampler {};\n",
            binding + 1,
            code_str,
            sampler_name
        );
        new_data.push_str(&layout_str);

    // Replace sampler definition with a texture and a sampler.
    } else {
        let mut layout_str = format!(
            "layout(set = 0, binding = {}) {}{} {};\n",
            binding, code_str, texture_type, texture_name
        );
        if write_json {
            descriptor_set_layouts.push(json!({
                        "name": texture_name,
                        "binding": binding,
                        "ty": "SampledImage",
                        "count": 1,
                        "stage_flags": "All",
                    }));
        }
        new_data.push_str(&layout_str);
        sampler_mapping.insert(
            String::from(*sampler_name),
            (
                format!("{}({}, {})", sampler_type, texture_name, sampler_name),
                *binding,
            ),
        );
        *binding += 1;

        layout_str = format!(
            "layout(set = 0, binding = {}) {}sampler {};\n",
            binding, code_str, sampler_name
        );
        if write_json {
            descriptor_set_layouts.push(json!({
                        "name": sampler_name,
                        "binding": binding,
                        "ty": "Sampler",
                        "count": 1,
                        "stage_flags": "All",
                    }));
        }
        new_data.push_str(&layout_str);
        *binding += 1;
    }
}

fn replace_non_sampler_uniforms(new_data: &mut String) {
    new_data.push_str(
        "\tlayout(set = 0, binding = 0) uniform Locals {\n\
         \t\tmat4 uTransform;       // Orthographic projection\n\
         \t\tuniform float uDevicePixelRatio;\n\
         \t\t// A generic uniform that shaders can optionally use to configure\n\
         \t\t// an operation mode for this batch.\n\
         \t\tuniform int uMode;\n\
         \t};\n",
    );
}

fn extend_non_uniform_variables_with_location_info(
    attribute_descriptors: &mut Vec<JsonValue>,
    file_name: &str,
    in_location: &mut i8,
    instance_offset: &mut i8,
    line: &str,
    new_data: &mut String,
    out_location: &mut i8,
    vertex_offset: &mut i8,
    write_json: bool,
) {
    let layout_str;
    if line.starts_with("in") {
        layout_str = format!("layout(location = {}) {}\n", in_location, line);
        if write_json {
            add_attribute_descriptors_json(
                attribute_descriptors,
                file_name,
                in_location,
                instance_offset,
                line,
                vertex_offset,
            );
        }
        *in_location += 1;
    } else if line.starts_with("out") {
        layout_str = format!("layout(location = {}) {}\n", out_location, line);
        *out_location += 1;
    } else {
        let location = max(*in_location, *out_location);
        layout_str = format!("layout(location = {}) {}\n", location, line);
        *in_location = location + 1;
        *out_location = location + 1;
    }
    new_data.push_str(&layout_str)
}

fn add_attribute_descriptors_json(
    attribute_descriptors: &mut Vec<JsonValue>,
    file_name: &str,
    in_location: &mut i8,
    instance_offset: &mut i8,
    line: &str,
    vertex_offset: &mut i8,
) {
    let def = split_code(line);
    let (format, offset) = match def[1] {
        "int" => ("R8Int", 4),
        "ivec4" => ("Rgba32Int", 16),
        "vec2" => ("Rg32Float", 8),
        "vec3" => ("Rgb32Float", 12),
        "vec4" => ("Rgba32Float", 16),
        _ => unimplemented!(),
    };
    let var_name = def[2].trim_right_matches(';');
    match var_name {
        "aColor" | "aColorTexCoord" | "aPosition" => {
            attribute_descriptors.push(json!({
                                    "name": var_name,
                                    "location": in_location,
                                    "binding": 0,
                                    "element": {
                                        "format": format,
                                        "offset": vertex_offset,
                                    }
                                }));
            *vertex_offset += offset;
        }
        // All shader files contain aData0 and aData1 variables.
        // We must remove them from the attribute descriptors of blur and clip shaders,
        // because they are not used in these and we don't want them when calculating offsets.
        "aData0" | "aData1" => {
            if !(file_name.starts_with("cs_blur") || file_name.starts_with("cs_clip")) {
                attribute_descriptors.push(json!({
                                        "name": var_name,
                                        "location": in_location,
                                        "binding": 1,
                                        "element": {
                                            "format": format,
                                            "offset": instance_offset,
                                        }
                                    }));
                *instance_offset += offset;
            }
        }
        _ => {
            attribute_descriptors.push(json!({
                                    "name": var_name,
                                    "location": in_location,
                                    "binding": 1,
                                    "element": {
                                        "format": format,
                                        "offset": instance_offset,
                                    }
                                }));
            *instance_offset += offset;
        }
    };
}

fn create_desciptor_pool_json(count: i8) -> JsonValue {
    json!({
        "sets": 1,
        "descriptors": [
            {
                "ty": "SampledImage",
                "count": count,
            },
            {
                "ty": "Sampler",
                "count": count,
            },
            {
                "ty": "UniformBuffer",
                "count": 1,
            },
        ],
    })
}

fn create_vertex_buffer_descriptors_json(file_name: &str) -> Vec<JsonValue> {
    let mut descriptors = Vec::new();
    descriptors.push(json!({
                "stride": "Vertex",
                "rate": 0,
            }));
    if file_name.starts_with("cs_blur") {
        descriptors.push(json!({
                    "stride": "BlurInstance",
                    "rate": 1,
                }));
    } else if file_name.starts_with("cs_clip") {
        descriptors.push(json!({
                    "stride": "ClipInstance",
                    "rate": 1,
                }));
    } else if file_name.starts_with("debug_color") {
        descriptors = vec![
            json!({
                    "stride": "DebugColorVertex",
                    "rate": 0,
                }),
        ];
    } else if file_name.starts_with("debug_font") {
        descriptors = vec![
            json!({
                    "stride": "DebugFontVertex",
                    "rate": 0,
                }),
        ];
    // Primitive and brush shaders
    } else {
        descriptors.push(json!({
                    "stride": "PrimitiveInstance",
                    "rate": 1,
                }));
    }
    descriptors
}

fn compile_glsl_to_spirv(file_name_vector: Vec<String>, out_dir: &str) -> JsonValue {
    let mut requirements = Vec::new();
    for mut file_name in file_name_vector {
        let file_path = Path::new(&out_dir).join(&file_name);
        if let Some(req) = process_glsl_for_spirv(&file_path, &file_name) {
            requirements.push(req);
        }
        file_name.push_str(".spv");
        let spirv_file_path = Path::new(&out_dir).join(&file_name);
        #[cfg(any(target_os = "android", target_os = "linux"))]
        let mut glslang_cmd = Command::new(Path::new("./tools/glslangValidator"));
        #[cfg(target_os = "windows")]
        let mut glslang_cmd = Command::new(Path::new("./tools/glslangValidator.exe"));
        glslang_cmd
            .arg("-V")
            .arg("-o")
            .arg(&spirv_file_path)
            .arg(&file_path);
        if glslang_cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .unwrap()
            .code()
            .unwrap() != 0
        {
            println!("Error while compiling spirv: {:?}", file_name);
            process::exit(1)
        };
        #[cfg(any(target_os = "android", target_os = "linux"))]
        let mut spirv_val_cmd = Command::new(Path::new("./tools/spirv-val"));
        #[cfg(target_os = "windows")]
        let mut spirv_val_cmd = Command::new(Path::new("./tools/spirv-val.exe"));
        spirv_val_cmd.arg(&spirv_file_path);
        if spirv_val_cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .unwrap()
            .code()
            .unwrap() != 0
        {
            println!("Error while validating spirv shader: {:?}", file_name);
            process::exit(1)
        }
    }
    json!(requirements)
}

fn write_json_to_file(requriements: JsonValue, out_dir: &str) {
    let json_file_path = Path::new(&out_dir).join("pipelines.json");
    let mut json_file = File::create(&json_file_path).unwrap();
    json_file
        .write(
            serde_json::to_string_pretty(&requriements)
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
}

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap_or("out".to_owned());

    let shaders_file = Path::new(&out_dir).join("shaders.rs");
    let mut glsl_files = vec![];

    println!("cargo:rerun-if-changed=res");
    let res_dir = Path::new("res");
    for entry in read_dir(res_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if entry.file_name().to_str().unwrap().ends_with(".glsl") {
            println!("cargo:rerun-if-changed={}", path.display());
            glsl_files.push(path.to_owned());
        }
    }

    // Sort the file list so that the shaders.rs file is filled
    // deterministically.
    glsl_files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    let shader_map = write_shaders(glsl_files, &shaders_file);
    let shaders = create_shaders(&out_dir, &shader_map);
    let requirements = compile_glsl_to_spirv(shaders, &out_dir);
    write_json_to_file(requirements, &out_dir);
}
