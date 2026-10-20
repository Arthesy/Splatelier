// LOCAL PATCH (see VENDORED.md): stochastic transparency render path.
// The coverage thresholds and the quad resolve are adapted from SuperSplat
// (https://github.com/playcanvas/supersplat), Copyright (c) 2011-2026 PlayCanvas Ltd., MIT License.
//
// Instead of blending back-to-front sorted splats, each fragment is kept with probability equal
// to its alpha and written opaque with a depth test, so draw order does not matter and no sort
// is needed. In expectation this is the sorted "over" composite. Following SuperSplat
// (PlayCanvas, MIT; "StochasticSplats" 1 spp stochastic transparency):
// - Coverage thresholds are stratified over each screen-space 2x2 pixel quad: the quad's four
//   pixels take the four strata of [0, 1) in a per-(quad, splat) scrambled order, so a splat with
//   alpha a covers 4a +/- 1 of the quad's pixels.
// - An optional resolve pass averages each quad and bilinearly interpolates between quad
//   centres, replacing most of the sampling noise with quantization error.
// - An optional temporal resolve reprojects the previous output with this frame's depth and blends
//   it in, after SteadySplats (Windisch et al. 2026, arXiv:2610.05576, Sec. "temporal importance
//   resampling"): an exponential moving average whose history weight is lowered softly where the
//   history falls outside the current 3x3 neighbourhood's spread.
// Without the temporal resolve the pattern depends only on the pixel and the splat, so a still frame
// does not flicker.

#include "SplatProcessing.h"

// Unlike FragmentIn, no half-typed varyings: mixing them with a 32-bit [[flat]] varying made every
// varying but the position arrive as zero on an M-series Mac (fine under Metal shader validation,
// which changes the varying layout). The color is constant over the splat, so it travels flat,
// packed as half4 bits.
typedef struct
{
    float4 position [[position]];
    float2 relativePosition;   // Ranges from -kBoundsRadius to +kBoundsRadius
    uint2 packedColor [[flat]]; // half4 color, as_type-packed
    uint splatID [[flat]];      // Stable per-splat hash input for the coverage thresholds
} StochasticFragmentIn;

vertex StochasticFragmentIn stochasticSplatVertexShader(uint vertexID [[vertex_id]],
                                                        uint instanceID [[instance_id]],
                                                        ushort amplificationID [[amplification_id]],
                                                        device const ChunkInfo* chunks [[ buffer(BufferIndexChunks) ]],
                                                        constant ChunkedSplatIndex* splatIndexArray [[ buffer(BufferIndexSplatIndex) ]],
                                                        constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    Uniforms uniforms = uniformsArray.uniforms[min(int(amplificationID), kMaxViewCount)];

    StochasticFragmentIn out;

    uint drawID = instanceID * uniforms.indexedSplatCount + (vertexID / 4);
    if (drawID >= uniforms.splatCount) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    // The index buffer is sorted back to front for some recent camera pose. Reading it in reverse
    // draws roughly front to back, so the depth test rejects most fragments behind the first
    // opaque coverage before they are shaded.
    ChunkedSplatIndex idx = splatIndexArray[uniforms.splatCount - 1 - drawID];

    if (idx.chunkIndex >= uniforms.chunkCount) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    ChunkInfo chunk = chunks[idx.chunkIndex];

    if (idx.splatIndex >= chunk.splatCount || !chunk.enabled) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    Splat splat = chunk.splats[idx.splatIndex];

    FragmentIn splatOut = splatVertex(splat, uniforms, vertexID % 4,
                                      chunk.shCoefficients, chunk.shDegree,
                                      idx.splatIndex);
    out.position = splatOut.position;
    out.relativePosition = float2(splatOut.relativePosition);
    out.packedColor = as_type<uint2>(half4(colorForBlending(splatOut.color.rgb), splatOut.color.a));
    out.splatID = idx.splatIndex ^ (uint(idx.chunkIndex) * 0x9E3779B9u);
    return out;
}

/// Integer hash (Chris Wellons' "prospector" mix) -> uniform uint.
static inline uint hashU32(uint x) {
    uint v = x;
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

/// `thresholdOffset` rotates every coverage threshold by the same amount modulo 1 (a Cranley-Patterson
/// rotation), which keeps the quad stratification. The temporal resolve advances it every frame so
/// that successive frames take different samples; 0 gives the fixed pattern.
fragment half4 stochasticSplatFragmentShader(StochasticFragmentIn in [[stage_in]],
                                             constant float& thresholdOffset [[buffer(0)]]) {
    half4 color = as_type<half4>(in.packedColor);
    half alpha = splatFragmentAlpha(half2(in.relativePosition), color.a);

    uint2 pixel = uint2(in.position.xy);
    uint2 quad = pixel >> 1;
    uint h = hashU32((quad.x * 1973u) ^ (quad.y * 9277u) ^ ((in.splatID + 1u) * 26699u));
    uint stratum = ((pixel.y & 1u) * 2u + (pixel.x & 1u)) ^ (h & 3u);
    float threshold = fract((float(stratum) + float(h >> 8) * (1.0f / 16777216.0f)) * 0.25f
                            + thresholdOffset);
    if (threshold >= float(alpha)) {
        discard_fragment();
    }
    // Premultiplied like the sorted path: opaque coverage, empty pixels keep the clear color.
    return half4(color.rgb, 1);
}

// MARK: - 2x2 quad resolve

typedef struct
{
    float4 position [[position]];
} StochasticResolveIn;

/// A single triangle covering the entire screen.
vertex StochasticResolveIn stochasticResolveVertexShader(uint vertexID [[vertex_id]]) {
    StochasticResolveIn out;
    out.position = float4((vertexID == 2) ? 3.0 : -1.0,
                          (vertexID == 0) ? -3.0 : 1.0,
                          0.0,
                          1.0);
    return out;
}

fragment half4 stochasticResolveFragmentShader(StochasticResolveIn in [[stage_in]],
                                               texture2d<half> source [[texture(0)]]) {
    constexpr sampler bilinear(coord::pixel, address::clamp_to_edge, filter::linear);
    // Quad space: integer coordinates sit on the centres of the 2x2 quads the coverage thresholds
    // are stratified over (pixel centres are at +0.5, so quad q's centre is at 2q + 1).
    float2 u = (in.position.xy - 1.0f) * 0.5f;
    float2 q0 = floor(u);
    float2 f = u - q0;
    // A bilinear tap at a quad's centre is the mean of its four pixels.
    float2 c = q0 * 2.0f + 1.0f;
    half4 a = source.sample(bilinear, c);
    half4 b = source.sample(bilinear, c + float2(2, 0));
    half4 d = source.sample(bilinear, c + float2(0, 2));
    half4 e = source.sample(bilinear, c + float2(2, 2));
    return mix(mix(a, b, half(f.x)), mix(d, e, half(f.x)), half(f.y));
}

// MARK: - Temporal resolve

// Keep in sync with SplatRenderer.StochasticTemporalParams.
typedef struct
{
    /// Maps this frame's clip coordinates (NDC x, y, depth, 1) to the history frame's clip
    /// coordinates.
    float4x4 reprojection;
    float2 viewportOrigin;
    float2 viewportSize;
    /// History weight before rejection (SteadySplats' delta-bar); 0 outputs the current frame only.
    float historyWeight;
    /// Non-zero for reverse-Z depth (nearer is larger).
    uint reverseZ;
    /// Non-zero: `current` is the raw coverage, and the pass applies the 2x2 quad resolve itself.
    uint quadResolve;
} StochasticTemporalParams;

typedef struct
{
    half4 color [[color(0)]];
    half4 history [[color(1)]];
} StochasticTemporalOut;

static inline float3 rgbToYCoCg(float3 c) {
    return float3( 0.25f * c.r + 0.5f * c.g + 0.25f * c.b,
                   0.5f  * c.r              - 0.5f  * c.b,
                  -0.25f * c.r + 0.5f * c.g - 0.25f * c.b);
}

/// Catmull-Rom filtered history at `p` (pixel coordinates, centres at +0.5) with 5 bilinear taps:
/// the separable 9-tap decomposition (the two middle weights share one tap per axis) without its
/// four corner taps, renormalized (their weights are small).
static inline float3 sampleCatmullRom(texture2d<half> tex, float2 p) {
    constexpr sampler bilinear(coord::pixel, address::clamp_to_edge, filter::linear);
    float2 t1 = floor(p - 0.5f) + 0.5f;
    float2 f = p - t1;
    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);
    float2 w12 = w1 + w2;
    float2 t0 = t1 - 1.0f;
    float2 t3 = t1 + 2.0f;
    float2 t12 = t1 + w2 / w12;
    float wTop = w12.x * w0.y, wLeft = w0.x * w12.y, wCentre = w12.x * w12.y;
    float wRight = w3.x * w12.y, wBottom = w12.x * w3.y;
    float3 r = float3(tex.sample(bilinear, float2(t12.x, t0.y)).rgb) * wTop
             + float3(tex.sample(bilinear, float2(t0.x, t12.y)).rgb) * wLeft
             + float3(tex.sample(bilinear, float2(t12.x, t12.y)).rgb) * wCentre
             + float3(tex.sample(bilinear, float2(t3.x, t12.y)).rgb) * wRight
             + float3(tex.sample(bilinear, float2(t12.x, t3.y)).rgb) * wBottom;
    return max(r / (wTop + wLeft + wCentre + wRight + wBottom), 0.0f);
}

/// Blends the reprojected history into this frame's stochastic image and writes the result both to
/// the output and to the next history. With `quadResolve` it reads the raw coverage and applies the
/// quad resolve to the 3x3 neighbourhood itself (3x3 quad means cover it), instead of a separate pass.
fragment StochasticTemporalOut stochasticTemporalFragmentShader(
    StochasticResolveIn in [[stage_in]],
    texture2d<half> current [[texture(0)]],
    texture2d<half> history [[texture(1)]],
    depth2d<float> depth [[texture(2)]],
    constant StochasticTemporalParams& params [[buffer(0)]]) {
    int2 size = int2(current.get_width(), current.get_height());
    int2 p = int2(in.position.xy);

    // 3x3 neighbourhood statistics in YCoCg, and the nearest depth of the pixel and its four
    // neighbours (so edges move with the foreground).
    float4 quadMeans[3][3];
    int2 quadBase = 0;
    if (params.quadResolve) {
        // Pixel x interpolates between quads floor((x - 0.5) / 2) and the next one (see
        // stochasticResolveFragmentShader); over x = p - 1 ... p + 1 that is three quads per axis.
        constexpr sampler bilinear(coord::pixel, address::clamp_to_edge, filter::linear);
        quadBase = int2(floor((float2(p) - 1.5f) * 0.5f));
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                quadMeans[i][j] = float4(current.sample(bilinear, float2(quadBase + int2(i, j)) * 2.0f + 1.0f));
            }
        }
    }

    float3 mean = 0, meanSquare = 0;
    float4 centre = 0;
    float nearest = params.reverseZ ? 0.0f : 1.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            uint2 q = uint2(clamp(p + int2(dx, dy), int2(0), size - 1));
            float4 c;
            if (params.quadResolve) {
                float2 u = (float2(p + int2(dx, dy)) - 0.5f) * 0.5f;
                float2 q0 = floor(u);
                float2 f = u - q0;
                int2 k = int2(q0) - quadBase;
                c = mix(mix(quadMeans[k.x][k.y], quadMeans[k.x + 1][k.y], f.x),
                        mix(quadMeans[k.x][k.y + 1], quadMeans[k.x + 1][k.y + 1], f.x), f.y);
            } else {
                c = float4(current.read(q));
            }
            if (dx == 0 && dy == 0) centre = c;
            float3 y = rgbToYCoCg(c.rgb);
            mean += y;
            meanSquare += y * y;
            if (dx == 0 || dy == 0) {
                float z = depth.read(q);
                nearest = params.reverseZ ? max(nearest, z) : min(nearest, z);
            }
        }
    }
    mean *= (1.0f / 9.0f);
    float3 sigma = sqrt(max(meanSquare * (1.0f / 9.0f) - mean * mean, 0.0f));

    StochasticTemporalOut out;
    out.color = half4(centre);
    out.history = half4(centre);
    if (params.historyWeight <= 0.0f) return out;

    // Reproject the pixel centre at the nearest depth into the history frame.
    float2 pixel = in.position.xy;
    float2 ndc = (pixel - params.viewportOrigin) / params.viewportSize * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    float4 previous = params.reprojection * float4(ndc, nearest, 1.0f);
    if (previous.w <= 0.0f) return out;
    float2 previousNDC = previous.xy / previous.w;
    float2 previousPixel = params.viewportOrigin
        + float2(previousNDC.x + 1.0f, 1.0f - previousNDC.y) * 0.5f * params.viewportSize;
    // Off screen or within two pixels of the border: no usable history.
    if (any(previousPixel < 2.0f) || any(previousPixel > float2(size) - 2.0f)) return out;

    float3 historyColor = sampleCatmullRom(history, previousPixel);

    // Soft rejection: the confidence box narrows with screen-space motion, and the weight falls as
    // 1/m once the history leaves it by a factor m.
    float velocity = length(pixel - previousPixel);
    float gamma = mix(4.0f, 1.0f, min(1.0f, velocity / 4.0f));
    float3 extent = max(gamma * sigma, float3(0.01f, 0.005f, 0.005f));
    float3 deviation = abs(rgbToYCoCg(historyColor) - mean) / extent;
    float m = max(deviation.x, max(deviation.y, deviation.z));
    float weight = m <= 1.0f ? params.historyWeight : params.historyWeight / m;

    float3 blended = mix(centre.rgb, historyColor, weight);
    out.color = half4(half3(blended), half(centre.a));
    out.history = out.color;
    return out;
}

/// Copies an exact (sorted) frame into the history, so the first stochastic frames after the camera
/// starts moving blend with it.
fragment half4 stochasticHistorySeedFragmentShader(StochasticResolveIn in [[stage_in]],
                                                   texture2d<half> source [[texture(0)]]) {
    return source.read(uint2(in.position.xy));
}
