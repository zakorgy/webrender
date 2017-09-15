/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared,shared_border

#ifdef WR_DX11
    struct v2p {
        vec4 Position : SV_Position;
        flat vec4 vClipMaskUvBounds : vClipMaskUvBounds;
        vec3 vClipMaskUv : vClipMaskUv;
        flat vec4 vColor0 : vColor0;
        flat vec4 vColor1 : vColor1;
        flat vec2 vEdgeDistance : vEdgeDistance;
        flat float vAxisSelect : vAxisSelect;
        flat float vAlphaSelect : vAlphaSelect;
        flat vec4 vClipParams : vClipParams;
        flat float vClipSelect : vClipSelect;
#ifdef WR_FEATURE_TRANSFORM
        vec3 vLocalPos : vLocalPos;
        flat vec4 vLocalBounds : vLocalBounds;
#else
        vec2 vLocalPos : vLocalPos;
#endif //WR_FEATURE_TRANSFORM
    };
#else

flat varying vec4 vColor0;
flat varying vec4 vColor1;
flat varying vec2 vEdgeDistance;
flat varying float vAxisSelect;
flat varying float vAlphaSelect;
flat varying vec4 vClipParams;
flat varying float vClipSelect;

#ifdef WR_FEATURE_TRANSFORM
varying vec3 vLocalPos;
#else
varying vec2 vLocalPos;
#endif //WR_FEATURE_TRANSFORM
#endif //WR_DX11

#ifdef WR_VERTEX_SHADER
void write_edge_distance(float p0,
                         float original_width,
                         float adjusted_width,
                         float style,
                         float axis_select,
                         float sign_adjust
#ifdef WR_DX11
                         , out vec2 vEdgeDistance
                         , out float vAxisSelect
#endif //WR_DX11
                         ) {
    switch (int(style)) {
        case BORDER_STYLE_DOUBLE:
            vEdgeDistance = vec2(p0 + adjusted_width,
                                 p0 + original_width - adjusted_width);
            break;
        case BORDER_STYLE_GROOVE:
        case BORDER_STYLE_RIDGE:
            vEdgeDistance = vec2(p0 + adjusted_width, sign_adjust);
            break;
        default:
            vEdgeDistance = vec2(0.0, 0.0);
            break;
    }

    vAxisSelect = axis_select;
}

void write_alpha_select(float style
#ifdef WR_DX11
                        , out float vAlphaSelect
#endif //WR_DX11
                        ) {
    switch (int(style)) {
        case BORDER_STYLE_DOUBLE:
            vAlphaSelect = 0.0;
            break;
        default:
            vAlphaSelect = 1.0;
            break;
    }
}

void write_color(vec4 color,
                 float style,
                 bool flip
#ifdef WR_DX11
                 , out vec4 vColor0
                 , out vec4 vColor1
#endif //WR_DX11
                 ) {
    vec2 modulate;

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

    vColor0 = vec4(color.rgb * modulate.x, color.a);
    vColor1 = vec4(color.rgb * modulate.y, color.a);
}

void write_clip_params(float style,
                       float border_width,
                       float edge_length,
                       float edge_offset,
                       float center_line
#ifdef WR_DX11
                       , out vec4 vClipParams
                       , out float vClipSelect
#endif //WR_DX11
                       ) {
    // x = offset
    // y = dash on + off length
    // z = dash length
    // w = center line of edge cross-axis (for dots only)
    switch (int(style)) {
        case BORDER_STYLE_DASHED: {
            float desired_dash_length = border_width * 3.0;
            // Consider half total length since there is an equal on/off for each dash.
            float dash_count = ceil(0.5 * edge_length / desired_dash_length);
            float dash_length = 0.5 * edge_length / dash_count;
            vClipParams = vec4(edge_offset - 0.5 * dash_length,
                               2.0 * dash_length,
                               dash_length,
                               0.0);
            vClipSelect = 0.0;
            break;
        }
        case BORDER_STYLE_DOTTED: {
            float diameter = border_width;
            float radius = 0.5 * diameter;
            float dot_count = ceil(0.5 * edge_length / diameter);
            float empty_space = edge_length - dot_count * diameter;
            float distance_between_centers = diameter + empty_space / dot_count;
            vClipParams = vec4(edge_offset - radius,
                               distance_between_centers,
                               radius,
                               center_line);
            vClipSelect = 1.0;
            break;
        }
        default:
            vClipParams = vec4(1.0, 1.0, 1.0, 1.0);
            vClipSelect = 0.0;
            break;
    }
}

#ifndef WR_DX11
void main(void) {
#else
void main(in a2v IN, out v2p OUT) {
    vec3 aPosition = IN.pos;
    ivec4 aDataA = IN.data0;
    ivec4 aDataB = IN.data1;
    int gl_VertexID = IN.vertexId;
#endif //WR_DX11
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
            write_edge_distance(segment_rect.p0.x,
                                border.widths.x,
                                adjusted_widths.x,
                                border.style.x,
                                0.0,
                                1.0
#ifdef WR_DX11
                                , OUT.vEdgeDistance
                                , OUT.vAxisSelect
#endif //WR_DX11
                                );
            style = border.style.x;
            color_flip = false;
            write_clip_params(border.style.x,
                              border.widths.x,
                              segment_rect.size.y,
                              segment_rect.p0.y,
                              segment_rect.p0.x + 0.5 * segment_rect.size.x
#ifdef WR_DX11
                              , OUT.vClipParams
                              , OUT.vClipSelect
#endif //WR_DX11
                              );
            break;
        }
        case 1: {
            segment_rect.p0 = vec2(corners.tl_inner.x, corners.tl_outer.y);
            segment_rect.size = vec2(corners.tr_inner.x - corners.tl_inner.x, border.widths.y);
            vec4 adjusted_widths = get_effective_border_widths(border, int(border.style.y));
            write_edge_distance(segment_rect.p0.y,
                                border.widths.y,
                                adjusted_widths.y,
                                border.style.y,
                                1.0,
                                1.0
#ifdef WR_DX11
                                , OUT.vEdgeDistance
                                , OUT.vAxisSelect
#endif //WR_DX11
                                );
            style = border.style.y;
            color_flip = false;
            write_clip_params(border.style.y,
                              border.widths.y,
                              segment_rect.size.x,
                              segment_rect.p0.x,
                              segment_rect.p0.y + 0.5 * segment_rect.size.y
#ifdef WR_DX11
                              , OUT.vClipParams
                              , OUT.vClipSelect
#endif //WR_DX11
                              );
            break;
        }
        case 2: {
            segment_rect.p0 = vec2(corners.tr_outer.x - border.widths.z, corners.tr_inner.y);
            segment_rect.size = vec2(border.widths.z, corners.br_inner.y - corners.tr_inner.y);
            vec4 adjusted_widths = get_effective_border_widths(border, int(border.style.z));
            write_edge_distance(segment_rect.p0.x,
                                border.widths.z,
                                adjusted_widths.z,
                                border.style.z,
                                0.0,
                                -1.0
#ifdef WR_DX11
                                , OUT.vEdgeDistance
                                , OUT.vAxisSelect
#endif //WR_DX11
                                );
            style = border.style.z;
            color_flip = true;
            write_clip_params(border.style.z,
                              border.widths.z,
                              segment_rect.size.y,
                              segment_rect.p0.y,
                              segment_rect.p0.x + 0.5 * segment_rect.size.x
#ifdef WR_DX11
                              , OUT.vClipParams
                              , OUT.vClipSelect
#endif //WR_DX11
                              );
            break;
        }
        default: {
            segment_rect.p0 = vec2(corners.bl_inner.x, corners.bl_outer.y - border.widths.w);
            segment_rect.size = vec2(corners.br_inner.x - corners.bl_inner.x, border.widths.w);
            vec4 adjusted_widths = get_effective_border_widths(border, int(border.style.w));
            write_edge_distance(segment_rect.p0.y,
                                border.widths.w,
                                adjusted_widths.w,
                                border.style.w,
                                1.0,
                                -1.0
#ifdef WR_DX11
                                , OUT.vEdgeDistance
                                , OUT.vAxisSelect
#endif //WR_DX11
                                );
            style = border.style.w;
            color_flip = true;
            write_clip_params(border.style.w,
                              border.widths.w,
                              segment_rect.size.x,
                              segment_rect.p0.x,
                              segment_rect.p0.y + 0.5 * segment_rect.size.y
#ifdef WR_DX11
                              , OUT.vClipParams
                              , OUT.vClipSelect
#endif //WR_DX11
                              );
            break;
        }
    }

    write_alpha_select(style
#ifdef WR_DX11
                       , OUT.vAlphaSelect
#endif //WR_DX11
                        );

    write_color(color,
                style,
                color_flip
#ifdef WR_DX11
                , OUT.vColor0
                , OUT.vColor1
#endif //WR_DX11
                );

#ifdef WR_FEATURE_TRANSFORM
    TransformVertexInfo vi = write_transform_vertex(gl_VertexID,
                                                    segment_rect,
                                                    prim.local_clip_rect,
                                                    prim.z,
                                                    prim.layer,
                                                    prim.task,
                                                    prim.local_rect
#ifdef WR_DX11
                                                    , OUT.Position
                                                    , OUT.vLocalBounds
#endif //WR_DX11
                                                    );
#else
    VertexInfo vi = write_vertex(aPosition,
                                 segment_rect,
                                 prim.local_clip_rect,
                                 prim.z,
                                 prim.layer,
                                 prim.task,
                                 prim.local_rect
#ifdef WR_DX11
                                 , OUT.Position
#endif //WR_DX11
                                 );
#endif //WR_FEATURE_TRANSFORM

    SHADER_OUT(vLocalPos, vi.local_pos);
    write_clip(vi.screen_pos,
               prim.clip_area
#ifdef WR_DX11
               , OUT.vClipMaskUvBounds
               , OUT.vClipMaskUv
#endif //WR_DX11
               );
}
#endif //WR_VERTEX_SHADER

#ifdef WR_FRAGMENT_SHADER
#ifndef WR_DX11
void main(void) {
#else
void main(in v2p IN, out p2f OUT) {
    vec4 vClipMaskUvBounds = IN.vClipMaskUvBounds;
    vec3 vClipMaskUv = IN.vClipMaskUv;
    float vAxisSelect = IN.vAxisSelect;
    vec2 vEdgeDistance = IN.vEdgeDistance;
    float vAlphaSelect = IN.vAlphaSelect;
    vec4 vColor0 = IN.vColor0;
    vec4 vColor1 = IN.vColor1;
    vec4 vClipParams = IN.vClipParams;
    float vClipSelect = IN.vClipSelect;
#endif //WR_DX11
    float alpha = 1.0;
#ifdef WR_FEATURE_TRANSFORM
    alpha = 0.0;
#ifdef WR_DX11
    vec3 vLocalPos = IN.vLocalPos;
    vec4 vLocalBounds = IN.vLocalBounds;
#endif //WR_DX11
    vec2 local_pos = init_transform_fs(vLocalPos, vLocalBounds, alpha);
#else
#ifdef WR_DX11
        vec2 vLocalPos = IN.vLocalPos;
#endif //WR_DX11
    vec2 local_pos = vLocalPos;
#endif //WR_FEATURE_TRANSFORM

    alpha = min(alpha, do_clip(vClipMaskUvBounds, vClipMaskUv));

    // Find the appropriate distance to apply the step over.
    vec2 fw = fwidth(local_pos);
    float afwidth = length(fw);

    // Applies the math necessary to draw a style: double
    // border. In the case of a solid border, the vertex
    // shader sets interpolator values that make this have
    // no effect.

    // Select the x/y coord, depending on which axis this edge is.
    vec2 pos = mix(local_pos.xy, local_pos.yx, vAxisSelect);

    // Get signed distance from each of the inner edges.
    float d0 = pos.x - vEdgeDistance.x;
    float d1 = vEdgeDistance.y - pos.x;

    // SDF union to select both outer edges.
    float d = min(d0, d1);

    // Select fragment on/off based on signed distance.
    // No AA here, since we know we're on a straight edge
    // and the width is rounded to a whole CSS pixel.
    alpha = min(alpha, mix(vAlphaSelect, 1.0, d < 0.0));

    // Mix color based on first distance.
    // TODO(gw): Support AA for groove/ridge border edge with transforms.
    bool b = d0 * vEdgeDistance.y > 0.0;
    vec4 color = mix(vColor0, vColor1, bvec4(b, b, b, b));

    // Apply dashing / dotting parameters.

    // Get the main-axis position relative to closest dot or dash.
    float x = mod(pos.y - vClipParams.x, vClipParams.y);

    // Calculate dash alpha (on/off) based on dash length
    float dash_alpha = step(x, vClipParams.z);

    // Get the dot alpha
    vec2 dot_relative_pos = vec2(x, pos.x) - vClipParams.zw;
    float dot_distance = length(dot_relative_pos) - vClipParams.z;
    float dot_alpha = 1.0 - smoothstep(-0.5 * afwidth,
                                        0.5 * afwidth,
                                        dot_distance);
    // Select between dot/dash alpha based on clip mode.
    alpha = min(alpha, mix(dash_alpha, dot_alpha, vClipSelect));
    SHADER_OUT(Target0, color * vec4(1.0, 1.0, 1.0, alpha));
}
#endif //WR_FRAGMENT_SHADER
