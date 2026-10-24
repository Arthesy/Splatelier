// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Stage (2) of the point elimination (see ba_point_qr.slang for the derivation): 1 thread / point
// adds the point's contribution to the reduced system over cameras (and the shared distortion
// k1, k2 when k_offset >= 0):
//   S_il  += δ_il·J̃c_iᵀJ̃c_i − C_iᵀ·W·C_l          rhs_i += −J̃c_iᵀ·r̃_i + C_iᵀ·W·q
//   S_kk  += Σ J̃k_iᵀJ̃k_i − C_kᵀ·W·C_k            rhs_k += −Σ J̃k_iᵀ·r̃_i + C_kᵀ·W·q
//   S_k,i += J̃k_iᵀJ̃c_i − C_kᵀ·W·C_i   (and its transpose)
// with C_i = (S·R⁻¹)ᵀ·J̃p_iᵀ·J̃c_i. Also accumulates diag(Σ J̃cᵀJ̃c) per camera and diag(Σ J̃kᵀJ̃k)
// (the undamped diagonals that Marquardt damping scales). Cameras below first_free are fixed and
// get no rows. All outputs are atomic adds into zero-initialized buffers; the k cells, which every
// point hits, are summed over the SIMD group first.

static const int LIN = 24;
static const int PD = 24;

StructuredBuffer<float> lin;              // [nObs * 24] (ba_linearize)
StructuredBuffer<int>   sorted_obs;       // [nObs] observation indices sorted by point
StructuredBuffer<int>   point_obs_starts; // [nPt]
StructuredBuffer<int>   point_obs_counts; // [nPt]
StructuredBuffer<int>   obs_cam_idx;      // [nObs]
StructuredBuffer<float> pdata;            // [nPt * 24] (ba_point_qr)
StructuredBuffer<int>   red_dims;         // [4] : nPt, D, first_free, k_offset (−1: k not estimated)

RWStructuredBuffer<float> redLHS;         // [D*D] row-major
RWStructuredBuffer<float> redRHS;         // [D]
RWStructuredBuffer<float> cam_diag;       // [nCam * 6]
RWStructuredBuffer<float> k_diag;         // [2]

// C = (S·R⁻¹)ᵀ·J̃pᵀ·J̃c (3×6) from one observation's J̃c (2×6) and J̃p (2×3). (Helpers cannot read
// the buffers, so callers load the rows.)
void projectCam(float jc[12], float jp[6], float sm[9], out float c[18])
{
    // h = J̃pᵀ·J̃c (3×6)
    float h[18];
    [unroll]
    for (int x = 0; x < 3; x++) {
        [unroll]
        for (int y = 0; y < 6; y++) {
            h[x * 6 + y] = jp[x] * jc[y] + jp[3 + x] * jc[6 + y];
        }
    }
    [unroll]
    for (int x = 0; x < 3; x++) {
        [unroll]
        for (int y = 0; y < 6; y++) {
            c[x * 6 + y] = sm[0 * 3 + x] * h[0 * 6 + y] + sm[1 * 3 + x] * h[1 * 6 + y] + sm[2 * 3 + x] * h[2 * 6 + y];
        }
    }
}

// v ← W·v for the symmetric W (00 01 02 11 12 22) applied to each column of a 3×n block.
float3 applyW(float w[6], float3 v)
{
    return float3(w[0] * v.x + w[1] * v.y + w[2] * v.z,
                  w[1] * v.x + w[3] * v.y + w[4] * v.z,
                  w[2] * v.x + w[4] * v.y + w[5] * v.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ba_point_reduce(uint3 tid : SV_DispatchThreadID)
{
    int j = int(tid.x);
    int nPt = red_dims[0];
    int D = red_dims[1];
    int firstFree = red_dims[2];
    int kOff = red_dims[3];

    // Per-lane contributions to the k cells (summed over the SIMD group at the end)
    float kk00 = 0, kk01 = 0, kk11 = 0, rk0 = 0, rk1 = 0, kd0 = 0, kd1 = 0;

    if (j < nPt) {
        int start = point_obs_starts[j];
        int count = point_obs_counts[j];
        int o = j * PD;
        float sm[9];
        [unroll]
        for (int k = 0; k < 9; k++) { sm[k] = pdata[o + k]; }
        float3 q = float3(pdata[o + 9], pdata[o + 10], pdata[o + 11]);
        float w[6];
        [unroll]
        for (int k = 0; k < 6; k++) { w[k] = pdata[o + 18 + k]; }
        float3 wq = applyW(w, q);
        // W·C_k (3×2) as two columns
        float3 ck0 = float3(pdata[o + 12], pdata[o + 14], pdata[o + 16]);
        float3 ck1 = float3(pdata[o + 13], pdata[o + 15], pdata[o + 17]);
        float3 wck0 = applyW(w, ck0);
        float3 wck1 = applyW(w, ck1);
        if (kOff >= 0) {
            kk00 -= dot(ck0, wck0);
            kk01 -= dot(ck0, wck1);
            kk11 -= dot(ck1, wck1);
            rk0 += dot(ck0, wq);
            rk1 += dot(ck1, wq);
        }

        for (int a = 0; a < count; a++) {
            int obsI = sorted_obs[start + a];
            int camI = obs_cam_idx[obsI];
            int bI = obsI * LIN;
            float jk00 = lin[bI + 18], jk01 = lin[bI + 19], jk10 = lin[bI + 20], jk11 = lin[bI + 21];
            float r0 = lin[bI + 22], r1 = lin[bI + 23];

            [unroll]
            for (int y = 0; y < 6; y++) {
                InterlockedAdd(cam_diag[camI * 6 + y], lin[bI + y] * lin[bI + y] + lin[bI + 6 + y] * lin[bI + 6 + y]);
            }
            kd0 += jk00 * jk00 + jk10 * jk10;
            kd1 += jk01 * jk01 + jk11 * jk11;
            if (kOff >= 0) {
                kk00 += jk00 * jk00 + jk10 * jk10;
                kk01 += jk00 * jk01 + jk10 * jk11;
                kk11 += jk01 * jk01 + jk11 * jk11;
                rk0 -= jk00 * r0 + jk10 * r1;
                rk1 -= jk01 * r0 + jk11 * r1;
            }
            if (camI < firstFree) { continue; }
            int fI = camI - firstFree;

            float jcI[12];
            float jpI[6];
            [unroll]
            for (int y = 0; y < 12; y++) { jcI[y] = lin[bI + y]; }
            [unroll]
            for (int y = 0; y < 6; y++) { jpI[y] = lin[bI + 12 + y]; }
            float cI[18];
            projectCam(jcI, jpI, sm, cI);

            // rhs_i += −J̃c_iᵀ·r̃_i + C_iᵀ·W·q
            [unroll]
            for (int y = 0; y < 6; y++) {
                float v = -(lin[bI + y] * r0 + lin[bI + 6 + y] * r1)
                        + cI[0 * 6 + y] * wq.x + cI[1 * 6 + y] * wq.y + cI[2 * 6 + y] * wq.z;
                InterlockedAdd(redRHS[fI * 6 + y], v);
            }

            // S_k,i += J̃k_iᵀJ̃c_i − C_kᵀ·W·C_i (2×6), both triangles
            if (kOff >= 0) {
                [unroll]
                for (int y = 0; y < 6; y++) {
                    float c0 = lin[bI + y], c1 = lin[bI + 6 + y];
                    float3 ciy = float3(cI[0 * 6 + y], cI[1 * 6 + y], cI[2 * 6 + y]);
                    float s0 = jk00 * c0 + jk10 * c1 - dot(wck0, ciy);
                    float s1 = jk01 * c0 + jk11 * c1 - dot(wck1, ciy);
                    InterlockedAdd(redLHS[(kOff + 0) * D + fI * 6 + y], s0);
                    InterlockedAdd(redLHS[(kOff + 1) * D + fI * 6 + y], s1);
                    InterlockedAdd(redLHS[(fI * 6 + y) * D + kOff + 0], s0);
                    InterlockedAdd(redLHS[(fI * 6 + y) * D + kOff + 1], s1);
                }
            }

            // W·C_i (3×6)
            float wcI[18];
            [unroll]
            for (int y = 0; y < 6; y++) {
                float3 v = applyW(w, float3(cI[0 * 6 + y], cI[1 * 6 + y], cI[2 * 6 + y]));
                wcI[0 * 6 + y] = v.x; wcI[1 * 6 + y] = v.y; wcI[2 * 6 + y] = v.z;
            }

            // S_il += δ_il·J̃c_iᵀJ̃c_i − (W·C_i)ᵀ·C_l over this point's free cameras l
            for (int b2 = 0; b2 < count; b2++) {
                int obsL = sorted_obs[start + b2];
                int camL = obs_cam_idx[obsL];
                if (camL < firstFree) { continue; }
                int fL = camL - firstFree;
                int bL = obsL * LIN;
                float jcL[12];
                float jpL[6];
                [unroll]
                for (int y = 0; y < 12; y++) { jcL[y] = lin[bL + y]; }
                [unroll]
                for (int y = 0; y < 6; y++) { jpL[y] = lin[bL + 12 + y]; }
                float cL[18];
                projectCam(jcL, jpL, sm, cL);
                bool same = (b2 == a);
                [unroll]
                for (int x = 0; x < 6; x++) {
                    int rowBase = (fI * 6 + x) * D + fL * 6;
                    [unroll]
                    for (int y = 0; y < 6; y++) {
                        float v = -(wcI[0 * 6 + x] * cL[0 * 6 + y] + wcI[1 * 6 + x] * cL[1 * 6 + y]
                                    + wcI[2 * 6 + x] * cL[2 * 6 + y]);
                        if (same) { v += lin[bI + x] * lin[bI + y] + lin[bI + 6 + x] * lin[bI + 6 + y]; }
                        InterlockedAdd(redLHS[rowBase + y], v);
                    }
                }
            }
        }
    }

    // k cells: SIMD-group sum, then one atomic per group
    float s_kd0 = WaveActiveSum(kd0);
    float s_kd1 = WaveActiveSum(kd1);
    float s_kk00 = WaveActiveSum(kk00);
    float s_kk01 = WaveActiveSum(kk01);
    float s_kk11 = WaveActiveSum(kk11);
    float s_rk0 = WaveActiveSum(rk0);
    float s_rk1 = WaveActiveSum(rk1);
    if (WaveIsFirstLane()) {
        InterlockedAdd(k_diag[0], s_kd0);
        InterlockedAdd(k_diag[1], s_kd1);
        if (kOff >= 0) {
            InterlockedAdd(redLHS[kOff * D + kOff], s_kk00);
            InterlockedAdd(redLHS[kOff * D + kOff + 1], s_kk01);
            InterlockedAdd(redLHS[(kOff + 1) * D + kOff], s_kk01);
            InterlockedAdd(redLHS[(kOff + 1) * D + kOff + 1], s_kk11);
            InterlockedAdd(redRHS[kOff], s_rk0);
            InterlockedAdd(redRHS[kOff + 1], s_rk1);
        }
    }
}
