//#line 1
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

vec2 write_edge_distance(float p0,
                         float original_width,
                         float adjusted_width,
                         float style,
                         //float axis_select,
                         float sign_adjust) {
    switch (int(style)) {
        case BORDER_STYLE_DOUBLE:
            return vec2(p0 + adjusted_width, p0 + original_width - adjusted_width);
        case BORDER_STYLE_GROOVE:
        case BORDER_STYLE_RIDGE:
            return vec2(p0 + adjusted_width, sign_adjust);
        default:
            return vec2(0.0, 0.0);
    }

    //vAxisSelect = axis_select;
}

float write_alpha_select(float style) {
    switch (int(style)) {
        case BORDER_STYLE_DOUBLE:
            return 0.0;
        default:
            return 1.0;
    }
}

struct WriteColorResult {
    vec4 color0;
    vec4 color1;
};

WriteColorResult write_color(vec4 color, float style, bool flip) {
    vec2 modulate;

    WriteColorResult wcr;
    switch (int(style)) {
        case BORDER_STYLE_GROOVE:
        {
            modulate = flip ? vec2(1.3, 0.7) : vec2(0.7, 1.3);
            break;
        }
        case BORDER_STYLE_RIDGE:
        {
            modulate = flip ? vec2(0.7, 1.3) : vec2(1.3, 0.7);
            break;
        }
        default:
            modulate = vec2(1.0, 1.0);
            break;
    }

    /*#ifdef WR_DX11
    wcr.color0 = vec4(mul(color.rgb, modulate.x), color.a);
    wcr.color1 = vec4(mul(color.rgb, modulate.y), color.a);
    #else*/
    wcr.color0 = vec4(color.rgb * modulate.x, color.a);
    wcr.color1 = vec4(color.rgb * modulate.y, color.a);
    //#endif
    return wcr;
}

struct WriteClipParamsResult {
    vec4 clip_params;
    float clip_select;
};

WriteClipParamsResult write_clip_params(float style,
                                  float border_width,
                                  float edge_length,
                                  float edge_offset,
                                  float center_line) {
    // x = offset
    // y = dash on + off length
    // z = dash length
    // w = center line of edge cross-axis (for dots only)
    WriteClipParamsResult wcpr;
    switch (int(style)) {
        case BORDER_STYLE_DASHED: {
            float desired_dash_length = border_width * 3.0;
            // Consider half total length since there is an equal on/off for each dash.
            float dash_count = ceil(0.5 * edge_length / desired_dash_length);
            float dash_length = 0.5 * edge_length / dash_count;
            wcpr.clip_params = vec4(edge_offset - 0.5 * dash_length,
                                    2.0 * dash_length,
                                    dash_length,
                                    0.0);
            wcpr.clip_select = 0.0;
            break;
        }
        case BORDER_STYLE_DOTTED: {
            float diameter = border_width;
            float radius = 0.5 * diameter;
            float dot_count = ceil(0.5 * edge_length / diameter);
            float empty_space = edge_length - dot_count * diameter;
            float distance_between_centers = diameter + empty_space / dot_count;
            wcpr.clip_params = vec4(edge_offset - radius,
                                    distance_between_centers,
                                    radius,
                                    center_line);
            wcpr.clip_select = 1.0;
            break;
        }
        default:
            wcpr.clip_params = vec4(1.0, 1.0, 1.0, 1.0);
            wcpr.clip_select = 0.0;
            break;
    }
    return wcpr;
}

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    ivec4 aDataA = IN.data0;
    ivec4 aDataB = IN.data1;
#endif
    Primitive prim = load_primitive(aDataA, aDataB);
    Border border = fetch_border(prim.specific_prim_address);
    int sub_part = prim.user_data0;
    BorderCorners corners = get_border_corners(border, prim.local_rect);
    vec4 color = border.colors[sub_part];

    // TODO(gw): Now that all border styles are supported, the switch
    //           statement below can be tidied up quite a bit.

    float style;
    bool color_flip;

    RectWithSize segment_rect;
    switch (sub_part) {
        case 0: {
            segment_rect.p0 = vec2(corners.tl_outer.x, corners.tl_inner.y);
            segment_rect.size = vec2(border.widths.x, corners.bl_inner.y - corners.tl_inner.y);
            vec4 adjusted_widths = get_effective_border_widths(border, int(border.style.x));
            SHADER_OUT(vEdgeDistance, write_edge_distance(segment_rect.p0.x, border.widths.x, adjusted_widths.x, border.style.x, 1.0));
            SHADER_OUT(vAxisSelect, 0.0);
            style = border.style.x;
            color_flip = false;
            WriteClipParamsResult wcpr = write_clip_params(border.style.x,
                                                           border.widths.x,
                                                           segment_rect.size.y,
                                                           segment_rect.p0.y,
                                                           segment_rect.p0.x + 0.5 * segment_rect.size.x);

            SHADER_OUT(vClipParams, wcpr.clip_params);
            SHADER_OUT(vClipSelect, wcpr.clip_select);
            break;
        }
        case 1: {
            segment_rect.p0 = vec2(corners.tl_inner.x, corners.tl_outer.y);
            segment_rect.size = vec2(corners.tr_inner.x - corners.tl_inner.x, border.widths.y);
            vec4 adjusted_widths = get_effective_border_widths(border, int(border.style.y));
            SHADER_OUT(vEdgeDistance, write_edge_distance(segment_rect.p0.y, border.widths.y, adjusted_widths.y, border.style.y, 1.0));
            SHADER_OUT(vAxisSelect, 1.0);
            style = border.style.y;
            color_flip = false;
            WriteClipParamsResult wcpr = write_clip_params(border.style.y,
                                                           border.widths.y,
                                                           segment_rect.size.x,
                                                           segment_rect.p0.x,
                                                           segment_rect.p0.y + 0.5 * segment_rect.size.y);
            SHADER_OUT(vClipParams, wcpr.clip_params);
            SHADER_OUT(vClipSelect, wcpr.clip_select);
            break;
        }
        case 2: {
            segment_rect.p0 = vec2(corners.tr_outer.x - border.widths.z, corners.tr_inner.y);
            segment_rect.size = vec2(border.widths.z, corners.br_inner.y - corners.tr_inner.y);
            vec4 adjusted_widths = get_effective_border_widths(border, int(border.style.z));
            SHADER_OUT(vEdgeDistance, write_edge_distance(segment_rect.p0.x, border.widths.z, adjusted_widths.z, border.style.z, -1.0));
            SHADER_OUT(vAxisSelect, 0.0);
            style = border.style.z;
            color_flip = true;
            WriteClipParamsResult wcpr = write_clip_params(border.style.z,
                                                           border.widths.z,
                                                           segment_rect.size.y,
                                                           segment_rect.p0.y,
                                                           segment_rect.p0.x + 0.5 * segment_rect.size.x);
            SHADER_OUT(vClipParams, wcpr.clip_params);
            SHADER_OUT(vClipSelect, wcpr.clip_select);
            break;
        }
        default: {
            segment_rect.p0 = vec2(corners.bl_inner.x, corners.bl_outer.y - border.widths.w);
            segment_rect.size = vec2(corners.br_inner.x - corners.bl_inner.x, border.widths.w);
            vec4 adjusted_widths = get_effective_border_widths(border, int(border.style.w));
            write_edge_distance(segment_rect.p0.y, border.widths.w, adjusted_widths.w, border.style.w, -1.0);
            SHADER_OUT(vEdgeDistance, write_edge_distance(segment_rect.p0.y, border.widths.w, adjusted_widths.w, border.style.w, -1.0));
            SHADER_OUT(vAxisSelect, 1.0);
            style = border.style.w;
            color_flip = true;
            WriteClipParamsResult wcpr = write_clip_params(border.style.w,
                                                           border.widths.w,
                                                           segment_rect.size.x,
                                                           segment_rect.p0.x,
                                                           segment_rect.p0.y + 0.5 * segment_rect.size.y);
            SHADER_OUT(vClipParams, wcpr.clip_params);
            SHADER_OUT(vClipSelect, wcpr.clip_select);
            break;
        }
    }

    SHADER_OUT(vAlphaSelect, write_alpha_select(style));

    WriteColorResult wcr = write_color(color, style, color_flip);

    SHADER_OUT(vColor0, wcr.color0);
    SHADER_OUT(vColor1, wcr.color1);

#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(IN.vertexId,
                                                    segment_rect,
                                                    prim.local_clip_rect,
                                                    prim.z,
                                                    prim.layer,
                                                    prim.task,
                                                    prim.local_rect);
#else
    VertexInfo vi = write_vertex(aPosition,
                                 segment_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect);
#endif

    WriteClipResult write_clip_res = write_clip(vi.screen_pos, prim.clip_area);

    SHADER_OUT(vClipMaskUvBounds, write_clip_res.clip_mask_uv_bounds);
    SHADER_OUT(vClipMaskUv, write_clip_res.clip_mask_uv);
    SHADER_OUT(vLocalPos, vi.local_pos);

#ifdef WR_DX11
    OUT.Position = vi.out_pos;
#endif
}
