// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Per-observation BA linearization (1 thread / obs, plain writes only). The projection includes
// radial distortion shared by all cameras (COLMAP RADIAL in normalized coordinates):
//   Pc = R·P + t,  n = (x/z, y/z),  r² = |n|²,  f = 1 + k1 r² + k2 r⁴,
//   π = (fx·f·n.x + cx, fy·f·n.y + cy),  r = π − pixel.
// Writes the whitened Jacobian rows √w·J and residual √w·r (w = Huber IRLS weight) so the point
// stage can form normal-equation blocks as J̃ᵀJ̃ without separate weights:
//   lin[i*24 + 0..11]  J̃c (2×6, row-major; columns ω then v of the left perturbation)
//   lin[i*24 + 12..17] J̃p (2×3)
//   lin[i*24 + 18..21] J̃k (2×2, columns k1, k2)
//   lin[i*24 + 22..23] r̃
// Observations with z ≤ 1e-3 are written as zeros and so contribute nothing.

StructuredBuffer<float> viewmats;        // [num_cameras * 16] row-major world→camera
StructuredBuffer<float> intrinsics_buf;  // [num_cameras * 4] (fx, fy, cx, cy)
StructuredBuffer<float> points;          // [num_points * 3]
StructuredBuffer<int>   obs_cam_idx;     // [num_obs]
StructuredBuffer<int>   obs_pt_idx;      // [num_obs]
StructuredBuffer<float> obs_pixels;      // [num_obs * 2]
StructuredBuffer<float> lin_params;      // [3] : huber_threshold (<=0 disables), k1, k2
StructuredBuffer<int>   lin_dims;        // [1] : num_obs

RWStructuredBuffer<float> lin;           // [num_obs * 24]

[shader("compute")]
[numthreads(128, 1, 1)]
void ba_linearize(uint3 tid : SV_DispatchThreadID)
{
    int i = int(tid.x);
    int N = lin_dims[0];
    if (i >= N) { return; }
    int base = i * 24;

    int cam = obs_cam_idx[i];
    int pt = obs_pt_idx[i];
    int vm = cam * 16;
    float fx = intrinsics_buf[cam * 4 + 0];
    float fy = intrinsics_buf[cam * 4 + 1];
    float cx = intrinsics_buf[cam * 4 + 2];
    float cy = intrinsics_buf[cam * 4 + 3];
    float k1 = lin_params[1];
    float k2 = lin_params[2];

    float Px = points[pt * 3 + 0];
    float Py = points[pt * 3 + 1];
    float Pz = points[pt * 3 + 2];
    float R00 = viewmats[vm + 0], R01 = viewmats[vm + 1], R02 = viewmats[vm + 2];
    float R10 = viewmats[vm + 4], R11 = viewmats[vm + 5], R12 = viewmats[vm + 6];
    float R20 = viewmats[vm + 8], R21 = viewmats[vm + 9], R22 = viewmats[vm + 10];
    float pcx = R00 * Px + R01 * Py + R02 * Pz + viewmats[vm + 3];
    float pcy = R10 * Px + R11 * Py + R12 * Pz + viewmats[vm + 7];
    float pcz = R20 * Px + R21 * Py + R22 * Pz + viewmats[vm + 11];

    if (pcz <= 1e-3) {
        [unroll]
        for (int k = 0; k < 24; k++) { lin[base + k] = 0.0; }
        return;
    }

    float invZ = 1.0 / pcz;
    float nx = pcx * invZ;
    float ny = pcy * invZ;
    float r2 = nx * nx + ny * ny;
    float f = 1.0 + k1 * r2 + k2 * r2 * r2;
    float df = k1 + 2.0 * k2 * r2;          // ∂f/∂r²
    float rU = fx * f * nx + cx - obs_pixels[i * 2 + 0];
    float rV = fy * f * ny + cy - obs_pixels[i * 2 + 1];

    // ∂π/∂n (2×2) = diag(fx, fy) · [[f + 2nx²f', 2nx·ny·f'], [2nx·ny·f', f + 2ny²f']]
    float a00 = fx * (f + 2.0 * nx * nx * df);
    float a01 = fx * (2.0 * nx * ny * df);
    float a10 = fy * (2.0 * nx * ny * df);
    float a11 = fy * (f + 2.0 * ny * ny * df);
    // ∂π/∂Pc (2×3) = ∂π/∂n · [[1/z, 0, −x/z²], [0, 1/z, −y/z²]]
    float d00 = a00 * invZ, d01 = a01 * invZ, d02 = -(a00 * nx + a01 * ny) * invZ;
    float d10 = a10 * invZ, d11 = a11 * invZ, d12 = -(a10 * nx + a11 * ny) * invZ;

    float w = 1.0;
    float huberK = lin_params[0];
    if (huberK > 0.0) {
        float absR = sqrt(rU * rU + rV * rV);
        if (absR > huberK) { w = huberK / max(absR, 1e-9); }
    }
    float sw = sqrt(w);

    // J̃c = √w · ∂π/∂Pc · [−[Pc]× | I]; −[Pc]× = [[0, z, −y], [−z, 0, x], [y, −x, 0]]
    lin[base + 0] = sw * (-d01 * pcz + d02 * pcy);
    lin[base + 1] = sw * ( d00 * pcz - d02 * pcx);
    lin[base + 2] = sw * (-d00 * pcy + d01 * pcx);
    lin[base + 3] = sw * d00;
    lin[base + 4] = sw * d01;
    lin[base + 5] = sw * d02;
    lin[base + 6] = sw * (-d11 * pcz + d12 * pcy);
    lin[base + 7] = sw * ( d10 * pcz - d12 * pcx);
    lin[base + 8] = sw * (-d10 * pcy + d11 * pcx);
    lin[base + 9] = sw * d10;
    lin[base + 10] = sw * d11;
    lin[base + 11] = sw * d12;
    // J̃p = √w · ∂π/∂Pc · R
    lin[base + 12] = sw * (d00 * R00 + d01 * R10 + d02 * R20);
    lin[base + 13] = sw * (d00 * R01 + d01 * R11 + d02 * R21);
    lin[base + 14] = sw * (d00 * R02 + d01 * R12 + d02 * R22);
    lin[base + 15] = sw * (d10 * R00 + d11 * R10 + d12 * R20);
    lin[base + 16] = sw * (d10 * R01 + d11 * R11 + d12 * R21);
    lin[base + 17] = sw * (d10 * R02 + d11 * R12 + d12 * R22);
    // J̃k = √w · [[fx·nx·r², fx·nx·r⁴], [fy·ny·r², fy·ny·r⁴]]
    lin[base + 18] = sw * fx * nx * r2;
    lin[base + 19] = sw * fx * nx * r2 * r2;
    lin[base + 20] = sw * fy * ny * r2;
    lin[base + 21] = sw * fy * ny * r2 * r2;
    lin[base + 22] = sw * rU;
    lin[base + 23] = sw * rV;
}
