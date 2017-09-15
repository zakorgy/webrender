/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Matches BorderCorner enum in border.rs
#define CORNER_TOP_LEFT     0
#define CORNER_TOP_RIGHT    1
#define CORNER_BOTTOM_LEFT  2
#define CORNER_BOTTOM_RIGHT 3

// Matches BorderCornerClipKind enum in border.rs
#define CLIP_MODE_DASH      0
#define CLIP_MODE_DOT       1

// Header for a border corner clip.
struct BorderCorner {
    RectWithSize rect;
    vec2 clip_center;
    int corner;
    int clip_mode;
};

BorderCorner fetch_border_corner(ivec2 address) {
    ResourceCacheData2 data = fetch_from_resource_cache_2_direct(address);
    RectWithSize rect;
    rect.p0 = data.data0.xy;
    rect.size = data.data0.zw;
    BorderCorner border_corner;
    border_corner.rect = rect;
    border_corner.clip_center = data.data1.xy;
    border_corner.corner = int(data.data1.z);
    border_corner.clip_mode = int(data.data1.w);
    return border_corner;
}

// Per-dash clip information.
struct BorderClipDash {
    vec4 point_tangent_0;
    vec4 point_tangent_1;
};

BorderClipDash fetch_border_clip_dash(ivec2 address, int segment) {
    ResourceCacheData2 data = fetch_from_resource_cache_2_direct(address + ivec2(2 + 2 * (segment - 1), 0));
    BorderClipDash border_clip_dash;
    border_clip_dash.point_tangent_0 = data.data0;
    border_clip_dash.point_tangent_1 = data.data1;
    return border_clip_dash;
}

// Per-dot clip information.
struct BorderClipDot {
    vec3 center_radius;
};

BorderClipDot fetch_border_clip_dot(ivec2 address, int segment) {
    vec4 data = fetch_from_resource_cache_1_direct(address + ivec2(2 + (segment - 1), 0));
    BorderClipDot border_clip_dot;
    border_clip_dot.center_radius = data.xyz;
    return border_clip_dot;
}


#ifndef WR_DX11
void main(void) {
#else
void main(in a2v_clip IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    int aClipRenderTaskIndex = IN.aClipRenderTaskIndex;
    int aClipLayerIndex = IN.aClipLayerIndex;
    int aClipSegment = IN.aClipSegment;
    ivec4 aClipDataResourceAddress = IN.aClipDataResourceAddress;
#endif //WR_DX11
    CacheClipInstance cci = fetch_clip_item(aClipRenderTaskIndex,
                                            aClipLayerIndex,
                                            aClipSegment,
                                            aClipDataResourceAddress);
    ClipArea area = fetch_clip_area(cci.render_task_index);
    Layer layer = fetch_layer(cci.layer_index);

    // Fetch the header information for this corner clip.
    BorderCorner corner = fetch_border_corner(cci.clip_data_address);
    SHADER_OUT(vClipCenter, corner.clip_center);

    if (cci.segment == 0) {
        // The first segment is used to zero out the border corner.
        SHADER_OUT(vAlphaMask, vec2(0.0, 0.0));
        SHADER_OUT(vDotParams, vec3(0.0, 0.0, 0.0));
        SHADER_OUT(vPoint_Tangent0, vec4(1.0, 1.0, 1.0, 1.0));
        SHADER_OUT(vPoint_Tangent1, vec4(1.0, 1.0, 1.0, 1.0));
    } else {
        vec2 sign_modifier;
        switch (corner.corner) {
            case CORNER_TOP_LEFT:
                sign_modifier = vec2(-1.0, -1.0);
                break;
            case CORNER_TOP_RIGHT:
                sign_modifier = vec2(1.0, -1.0);
                break;
            case CORNER_BOTTOM_RIGHT:
                sign_modifier = vec2(1.0, 1.0);
                break;
            case CORNER_BOTTOM_LEFT:
                sign_modifier = vec2(-1.0, 1.0);
                break;
            default:
                sign_modifier = vec2(-1.0, -1.0);
                break;
        };

        switch (corner.clip_mode) {
            case CLIP_MODE_DASH: {
                // Fetch the information about this particular dash.
                BorderClipDash dash = fetch_border_clip_dash(cci.clip_data_address, cci.segment);
                SHADER_OUT(vPoint_Tangent0, dash.point_tangent_0 * sign_modifier.xyxy);
                SHADER_OUT(vPoint_Tangent1, dash.point_tangent_1 * sign_modifier.xyxy);
                SHADER_OUT(vDotParams, vec3(0.0, 0.0, 0.0));
                SHADER_OUT(vAlphaMask, vec2(0.0, 1.0));
                break;
            }
            case CLIP_MODE_DOT: {
                BorderClipDot cdot = fetch_border_clip_dot(cci.clip_data_address, cci.segment);
                SHADER_OUT(vPoint_Tangent0, vec4(1.0, 1.0, 1.0, 1.0));
                SHADER_OUT(vPoint_Tangent1, vec4(1.0, 1.0, 1.0, 1.0));
                SHADER_OUT(vDotParams, vec3(cdot.center_radius.xy * sign_modifier, cdot.center_radius.z));
                SHADER_OUT(vAlphaMask, vec2(1.0, 1.0));
                break;
            }
            default: {
                // Fetch the information about this particular dash.
                BorderClipDash dash = fetch_border_clip_dash(cci.clip_data_address, cci.segment);
                SHADER_OUT(vPoint_Tangent0, dash.point_tangent_0 * sign_modifier.xyxy);
                SHADER_OUT(vPoint_Tangent1, dash.point_tangent_1 * sign_modifier.xyxy);
                SHADER_OUT(vDotParams, vec3(0.0, 0.0, 0.0));
                SHADER_OUT(vAlphaMask, vec2(0.0, 1.0));
                break;
            }
        }
    }

    // Get local vertex position for the corner rect.
    // TODO(gw): We could reduce the number of pixels written here
    // by calculating a tight fitting bounding box of the dash itself.
    vec2 pos = corner.rect.p0 + aPosition.xy * corner.rect.size;

    // Transform to world pos
#ifdef WR_DX11
    vec4 world_pos = mul(vec4(pos, 0.0, 1.0), layer.transform);
#else
    vec4 world_pos = layer.transform * vec4(pos, 0.0, 1.0);
#endif //WR_DX11
    world_pos.xyz /= world_pos.w;

    // Scale into device pixels.
    vec2 device_pos = world_pos.xy * uDevicePixelRatio;

    // Position vertex within the render task area.
    vec2 final_pos = device_pos -
                     area.screen_origin_target_index.xy +
                     area.task_bounds.xy;

    // Calculate the local space position for this vertex.
    vec4 layer_pos = get_layer_pos(world_pos.xy, layer);
    SHADER_OUT(vPos, layer_pos.xyw);
    SHADER_OUT(gl_Position, mul(vec4(final_pos, 0.0, 1.0), uTransform));
}
