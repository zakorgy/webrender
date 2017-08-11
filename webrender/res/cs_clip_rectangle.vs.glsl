//#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

struct ClipRect {
    RectWithSize rect;
    vec4 mode;
};

ClipRect fetch_clip_rect(int index) {
    ResourceCacheData2 data = fetch_from_resource_cache_2(index);
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

ClipCorner fetch_clip_corner(int index) {
    ResourceCacheData2 data = fetch_from_resource_cache_2(index);
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

ClipData fetch_clip(int index) {
    ClipData clip;

    clip.rect = fetch_clip_rect(index + 0);
    clip.top_left = fetch_clip_corner(index + 2);
    clip.top_right = fetch_clip_corner(index + 4);
    clip.bottom_left = fetch_clip_corner(index + 6);
    clip.bottom_right = fetch_clip_corner(index + 8);

    return clip;
}

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    int aClipRenderTaskIndex = IN.aClipRenderTaskIndex;
    int aClipLayerIndex = IN.aClipLayerIndex;
    int aClipDataIndex = IN.aClipDataIndex;
    int aClipSegmentIndex = IN.aClipSegmentIndex;
    int aClipResourceAddress = IN.aClipResourceAddress;
#endif //WR_DX11
    CacheClipInstance cci = fetch_clip_item(aClipRenderTaskIndex,
                                            aClipLayerIndex,
                                            aClipDataIndex,
                                            aClipSegmentIndex,
                                            aClipResourceAddress);
    ClipArea area = fetch_clip_area(cci.render_task_index);
    Layer layer = fetch_layer(cci.layer_index);
    ClipData clip = fetch_clip(cci.data_index);
    RectWithSize local_rect = clip.rect.rect;

    ClipVertexInfo vi = write_clip_tile_vertex(aPosition,
                                               local_rect,
                                               layer,
                                               area,
                                               cci.segment_index
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
