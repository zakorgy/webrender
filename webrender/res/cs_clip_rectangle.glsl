/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared,clip_shared,ellipse

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        vec3 vPos : vPos;
        flat vec4 vLocalBounds : vLocalBounds;
        flat float vClipMode : vClipMode;
        flat vec4 vClipCenter_Radius_TL : vClipCenter_Radius_TL;
        flat vec4 vClipCenter_Radius_TR : vClipCenter_Radius_TR;
        flat vec4 vClipCenter_Radius_BL : vClipCenter_Radius_BL;
        flat vec4 vClipCenter_Radius_BR : vClipCenter_Radius_BR;
    };
#else
varying vec3 vPos;
flat varying float vClipMode;
flat varying vec4 vClipCenter_Radius_TL;
flat varying vec4 vClipCenter_Radius_TR;
flat varying vec4 vClipCenter_Radius_BL;
flat varying vec4 vClipCenter_Radius_BR;
#endif //WR_DX11

#ifdef WR_VERTEX_SHADER
struct ClipRect {
    RectWithSize rect;
    vec4 mode;
};

ClipRect fetch_clip_rect(ivec2 address) {
    ResourceCacheData2 data = fetch_from_resource_cache_2_direct(address);
    RectWithSize rect;
    rect.p0 = data.data0.xy;
    rect.size = data.data0.zw;
    ClipRect clip_rect;
    clip_rect.rect = rect;
    clip_rect.mode = data.data1;
    return clip_rect;
}

struct ClipCorner {
    RectWithSize rect;
    vec4 outer_inner_radius;
};

ClipCorner fetch_clip_corner(ivec2 address, int index) {
    address += ivec2(2 + 2 * index, 0);
    ResourceCacheData2 data = fetch_from_resource_cache_2_direct(address);
    RectWithSize rect;
    rect.p0 = data.data0.xy;
    rect.size = data.data0.zw;
    ClipCorner clip_corner;
    clip_corner.rect = rect;
    clip_corner.outer_inner_radius = data.data1;
    return clip_corner;
}

struct ClipData {
    ClipRect rect;
    ClipCorner top_left;
    ClipCorner top_right;
    ClipCorner bottom_left;
    ClipCorner bottom_right;
};

ClipData fetch_clip(ivec2 address) {
    ClipData clip;

    clip.rect = fetch_clip_rect(address);
    clip.top_left = fetch_clip_corner(address, 0);
    clip.top_right = fetch_clip_corner(address, 1);
    clip.bottom_left = fetch_clip_corner(address, 2);
    clip.bottom_right = fetch_clip_corner(address, 3);

    return clip;
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
    ClipData clip = fetch_clip(cci.clip_data_address);
    RectWithSize local_rect = clip.rect.rect;

    ClipVertexInfo vi = write_clip_tile_vertex(aPosition,
                                               local_rect,
                                               layer,
                                               area,
                                               cci.segment
#ifdef WR_DX11
                                               , OUT.Position
                                               , OUT.vLocalBounds
#endif //WR_DX11
                                               );
    SHADER_OUT(vPos, vi.local_pos);

    SHADER_OUT(vClipMode, clip.rect.mode.x);

    RectWithEndpoint clip_rect = to_rect_with_endpoint(local_rect);

    SHADER_OUT(vClipCenter_Radius_TL, vec4(clip_rect.p0 + clip.top_left.outer_inner_radius.xy,
                                 clip.top_left.outer_inner_radius.xy));

    SHADER_OUT(vClipCenter_Radius_TR, vec4(clip_rect.p1.x - clip.top_right.outer_inner_radius.x,
                                 clip_rect.p0.y + clip.top_right.outer_inner_radius.y,
                                 clip.top_right.outer_inner_radius.xy));

    SHADER_OUT(vClipCenter_Radius_BR, vec4(clip_rect.p1 - clip.bottom_right.outer_inner_radius.xy,
                                 clip.bottom_right.outer_inner_radius.xy));

    SHADER_OUT(vClipCenter_Radius_BL, vec4(clip_rect.p0.x + clip.bottom_left.outer_inner_radius.x,
                                 clip_rect.p1.y - clip.bottom_left.outer_inner_radius.y,
                                 clip.bottom_left.outer_inner_radius.xy));
}
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
float clip_against_ellipse_if_needed(vec2 pos,
                                     float current_distance,
                                     vec4 ellipse_center_radius,
                                     vec2 sign_modifier,
                                     float afwidth) {
    float ellipse_distance = distance_to_ellipse(pos - ellipse_center_radius.xy,
                                                 ellipse_center_radius.zw);

    return mix(current_distance,
               ellipse_distance + afwidth,
               all(lessThan(sign_modifier * pos, sign_modifier * ellipse_center_radius.xy)));
}

float rounded_rect(vec2 pos,
                   vec4 vClipCenter_Radius_TL,
                   vec4 vClipCenter_Radius_TR,
                   vec4 vClipCenter_Radius_BL,
                   vec4 vClipCenter_Radius_BR) {
    float current_distance = 0.0;

    // Apply AA
    float afwidth = 0.5 * length(fwidth(pos));

    // Clip against each ellipse.
    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_TL,
                                                      vec2(1.0, 1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_TR,
                                                      vec2(-1.0, 1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_BR,
                                                      vec2(-1.0, 1.0),
                                                      afwidth);

    current_distance = clip_against_ellipse_if_needed(pos,
                                                      current_distance,
                                                      vClipCenter_Radius_BL,
                                                      vec2(1.0, -1.0),
                                                      afwidth);

    return smoothstep(0.0, afwidth, 1.0 - current_distance);
}


#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec3 vPos = IN.vPos;
    vec4 vLocalBounds = IN.vLocalBounds;
    float vClipMode = IN.vClipMode;
    vec4 vClipCenter_Radius_TL = IN.vClipCenter_Radius_TL;
    vec4 vClipCenter_Radius_TR = IN.vClipCenter_Radius_TR;
    vec4 vClipCenter_Radius_BL = IN.vClipCenter_Radius_BL;
    vec4 vClipCenter_Radius_BR = IN.vClipCenter_Radius_BR;
#endif //WR_DX11
    float alpha = 1.f;
    vec2 local_pos = init_transform_fs(vPos, vLocalBounds, alpha);

    float clip_alpha = rounded_rect(local_pos,
                                    vClipCenter_Radius_TL,
                                    vClipCenter_Radius_TR,
                                    vClipCenter_Radius_BL,
                                    vClipCenter_Radius_BR);

    float combined_alpha = min(alpha, clip_alpha);

    // Select alpha or inverse alpha depending on clip in/out.
    float final_alpha = mix(combined_alpha, 1.0 - combined_alpha, vClipMode);

    SHADER_OUT(Target0, vec4(final_alpha, 0.0, 0.0, 1.0));
}
#endif //WR_FRAGMENT_SHADER
