/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_DX11
    #define vec2 float2
    #define vec3 float3
    #define vec4 float4
    #define ivec2 int2
    #define ivec3 int3
    #define ivec4 int4
    #define bvec2 bool2
    #define bvec3 bool3
    #define bvec4 bool4
    #define mat2 float2x2
    #define mat3 float3x3
    #define mat4 float4x4
    #define mix lerp
    #define fract frac
    #define uniform
    #define sampler2D Texture2D
    #define sampler2DArray Texture2DArray
    #define point p
    #define flat nointerpolation
    #define texelFetchOffset(sampler, loc, index, offset) sampler.Load(vec3(loc, 0.0), offset)
    #define texelFetch(sampler, loc, index) sampler.Load(vec3(loc, 0.0))
    #define texture(sampler, loc) sampler.Sample(sampler##_, loc)
    #define textureLod(sampler, loc, level) sampler.SampleLevel(sampler##_, loc, level)

    vec2 textureSize(sampler2D s, int lod) {
        uint width;
        uint height;
        s.GetDimensions(width, height);
        return vec2(width, height);
    }

    vec2 textureSize(sampler2DArray s, int lod) {
        uint width;
        uint height;
        uint elements;
        s.GetDimensions(width, height, elements);
        return vec2(width, height);
    }
    #define SHADER_OUT(value, expr) OUT.##value = expr
#else
    #define SHADER_OUT(value, expr) value = expr
    #define static
    #define mul(vector, matrix) matrix * vector
#endif

#ifdef WR_FEATURE_TEXTURE_EXTERNAL
// Please check https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
// for this extension.
#extension GL_OES_EGL_image_external_essl3 : require
#endif

// The textureLod() doesn't support samplerExternalOES for WR_FEATURE_TEXTURE_EXTERNAL.
// https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
//
// The textureLod() doesn't support sampler2DRect for WR_FEATURE_TEXTURE_RECT, too.
//
// Use texture() instead.
#if defined(WR_FEATURE_TEXTURE_EXTERNAL) || defined(WR_FEATURE_TEXTURE_RECT) || defined(WR_FEATURE_TEXTURE_2D)
#define TEX_SAMPLE(sampler, tex_coord) texture(sampler, tex_coord.xy)
#else
// In normal case, we use textureLod(). We haven't used the lod yet. So, we always pass 0.0 now.
#define TEX_SAMPLE(sampler, tex_coord) textureLod(sampler, tex_coord, 0.0)
#endif

#ifdef WR_DX11
bool2 lessThan(float2 value, float2 comparison) {
    return bool2(value.x < comparison.x,
                 value.y < comparison.y);
}

bool2 greaterThan(float2 value, float2 comparison) {
    return bool2(value.x > comparison.x,
                 value.y > comparison.y);
}

bool4 lessThanEqual(float4 value, float4 comparison) {
    return bool4(value.x <= comparison.x,
                 value.y <= comparison.y,
                 value.z <= comparison.z,
                 value.w <= comparison.w);
}

float mod(float x, float y) {
    return x - y * floor(x/y);
}

float2 mod(float2 x, float2 y) {
    return x - y * floor(x/y);
}
#endif //WR_DX11

//======================================================================================
// Vertex shader attributes and uniforms
//======================================================================================
#ifdef WR_VERTEX_SHADER

#ifndef WR_DX11
    #define varying out

    // Uniform inputs
    uniform mat4 uTransform;       // Orthographic projection
    uniform float uDevicePixelRatio;
#else
    cbuffer Locals {
        mat4 uTransform;       // Orthographic projection
        float uDevicePixelRatio;
    }
#endif //WR_DX11

    // Attribute inputs
#ifdef WR_DX11
    // The hlsl input struct in prim and clip shaders
#else
    in vec3 aPosition;
#endif //WR_DX11

#endif

//======================================================================================
// Fragment shader attributes and uniforms
//======================================================================================
#ifdef WR_FRAGMENT_SHADER

#ifndef WR_DX11
    precision highp float;

    #define varying in
#endif
    // Uniform inputs

    // Fragment shader outputs
#ifdef WR_DX11
    struct p2f {
        vec4 Target0 : SV_Target;
    };
#else
    out vec4 Target0;
#endif

#endif

//======================================================================================
// Shared shader uniforms
//======================================================================================
/*#if defined(GL_ES)
    #if GL_ES == 1
        #ifdef GL_FRAGMENT_PRECISION_HIGH
        precision highp sampler2DArray;
        #else
        precision mediump sampler2DArray;
        #endif

        // Sampler default precision is lowp on mobile GPUs.
        // This causes RGBA32F texture data to be clamped to 16 bit floats on some GPUs (e.g. Mali-T880).
        // Define highp precision macro to allow lossless FLOAT texture sampling.
        #define HIGHP_SAMPLER_FLOAT highp
    #else
        #define HIGHP_SAMPLER_FLOAT
    #endif
#else
    #define HIGHP_SAMPLER_FLOAT
#endif*/
#define HIGHP_SAMPLER_FLOAT

#ifdef WR_FEATURE_TEXTURE_2D
uniform sampler2D sColor0;
uniform sampler2D sColor1;
uniform sampler2D sColor2;
#elif defined WR_FEATURE_TEXTURE_RECT
uniform sampler2DRect sColor0;
uniform sampler2DRect sColor1;
uniform sampler2DRect sColor2;
#elif defined WR_FEATURE_TEXTURE_EXTERNAL
uniform samplerExternalOES sColor0;
uniform samplerExternalOES sColor1;
uniform samplerExternalOES sColor2;
#else
uniform sampler2DArray sColor0;
uniform sampler2DArray sColor1;
uniform sampler2DArray sColor2;
#endif

#ifdef WR_DX11
SamplerState sColor0_;
SamplerState sColor1_;
SamplerState sColor2_;
#endif //WR_DX11

//======================================================================================
// Interpolator definitions
//======================================================================================

//======================================================================================
// VS only types and UBOs
//======================================================================================

//======================================================================================
// VS only functions
//======================================================================================
