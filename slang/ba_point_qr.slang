// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Point elimination of the BA normal equations by a per-point QR of the point Jacobian, in the
// manner of square-root BA (Demmel et al., CVPR 2021 "Square Root Bundle Adjustment for
// Large-Scale Reconstruction"). Forming H_tt − H_tp·H_pp⁻¹·H_tpᵀ with an explicit 3×3 inverse in
// float32 amplifies rounding by cond(H_pp) and left the reduced camera system indefinite on real
// captures; with an orthonormal Q no inverse appears and each point's contribution is a Gram matrix
// ZᵀZ, so the sum stays positive semi-definite.
//
// For point j with observations i (whitened rows from ba_linearize), the stacked point Jacobian
// with Marquardt damping rows, column-scaled by S = diag(1/√diag(H_pp)),
//   A = [J̃p·S ; √λ·I]  (2n+3 rows × 3)
// is factored A = Q·R by shifted CholeskyQR3 (Fukaya et al. 2020): three Gram passes over the rows,
// each a 3×3 Cholesky, so Q is never stored; Q's rows are J̃p_i·(S·R⁻¹). With C_i = Q_iᵀ·J̃c_i,
// q = Σ Q_iᵀ·r̃_i, C_k = Σ Q_iᵀ·J̃k_i and W = 2I − QᵀQ (= I for an exactly orthonormal Q; keeping
// it makes every block below the Gram matrix of the projected rows for the Q actually computed):
//   S_il  += δ_il·J̃c_iᵀJ̃c_i − C_iᵀ·W·C_l          rhs_i += −J̃c_iᵀ·r̃_i + C_iᵀ·W·q
//   S_kk  += Σ J̃k_iᵀJ̃k_i − C_kᵀ·W·C_k            rhs_k += −Σ J̃k_iᵀ·r̃_i + C_kᵀ·W·q
//   S_k,i += J̃k_iᵀJ̃c_i − C_kᵀ·W·C_i
// and after solving for the cameras and k: δp = −S·R⁻¹·(q + Σ C_i·δc_i + C_k·δk).
// A point whose factorization breaks down (no usable observation, or rank deficient without
// damping) gets S·R⁻¹ = 0: it constrains the cameras as a fixed point and does not move.
//
// This file is stage (1), the QR; ba_point_reduce.slang and ba_point_update.slang use its output.
// Per-point observation lists (sorted_obs / starts / counts) are built by the host.
// pdata layout per point: S·R⁻¹ (3×3 row-major) | q (3) | C_k (3×2 row-major) | W (00 01 02 11 12 22)

static const int LIN = 24;
static const int PD = 24;

StructuredBuffer<float> lin;              // [nObs * 24] (ba_linearize)
StructuredBuffer<int>   sorted_obs;       // [nObs] observation indices sorted by point
StructuredBuffer<int>   point_obs_starts; // [nPt]
StructuredBuffer<int>   point_obs_counts; // [nPt]
StructuredBuffer<float> qr_params;        // [1] : Marquardt λ for the point blocks (0 = none)
StructuredBuffer<int>   qr_dims;          // [1] : nPt

RWStructuredBuffer<float> pdata;          // [nPt * 24]


// G = RᵀR for symmetric G (g00 g01 g02 g11 g12 g22) and upper-triangular R (r00 r01 r02 r11 r12 r22).
// Returns false if a pivot is not above minPivot.
bool chol3(float g[6], float minPivot, out float r[6])
{
    r = { 0, 0, 0, 0, 0, 0 };
    float p0 = g[0];
    if (!(p0 > minPivot)) { return false; }
    r[0] = sqrt(p0);
    r[1] = g[1] / r[0];
    r[2] = g[2] / r[0];
    float p1 = g[3] - r[1] * r[1];
    if (!(p1 > minPivot)) { return false; }
    r[3] = sqrt(p1);
    r[4] = (g[4] - r[1] * r[2]) / r[3];
    float p2 = g[5] - r[2] * r[2] - r[4] * r[4];
    if (!(p2 > minPivot)) { return false; }
    r[5] = sqrt(p2);
    return true;
}

// m ← m · R⁻¹ for 3×3 row-major m and upper-triangular R.
void mulUpperInv(inout float m[9], float r[6])
{
    [unroll]
    for (int row = 0; row < 3; row++) {
        float x0 = m[row * 3 + 0] / r[0];
        float x1 = (m[row * 3 + 1] - x0 * r[1]) / r[3];
        float x2 = (m[row * 3 + 2] - x0 * r[2] - x1 * r[4]) / r[5];
        m[row * 3 + 0] = x0;
        m[row * 3 + 1] = x1;
        m[row * 3 + 2] = x2;
    }
}

// g += qᵀq for row vector q.
void addOuter(inout float g[6], float q0, float q1, float q2)
{
    g[0] += q0 * q0; g[1] += q0 * q1; g[2] += q0 * q2;
    g[3] += q1 * q1; g[4] += q1 * q2; g[5] += q2 * q2;
}

// g += λ·mᵀm, the Gram of the damping rows √λ·e_c·m.
void addDamping(inout float g[6], float lambda, float m[9])
{
    g[0] += lambda * (m[0] * m[0] + m[3] * m[3] + m[6] * m[6]);
    g[1] += lambda * (m[0] * m[1] + m[3] * m[4] + m[6] * m[7]);
    g[2] += lambda * (m[0] * m[2] + m[3] * m[5] + m[6] * m[8]);
    g[3] += lambda * (m[1] * m[1] + m[4] * m[4] + m[7] * m[7]);
    g[4] += lambda * (m[1] * m[2] + m[4] * m[5] + m[7] * m[8]);
    g[5] += lambda * (m[2] * m[2] + m[5] * m[5] + m[8] * m[8]);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void ba_point_qr(uint3 tid : SV_DispatchThreadID)
{
    int j = int(tid.x);
    if (j >= qr_dims[0]) { return; }
    int start = point_obs_starts[j];
    int count = point_obs_counts[j];
    float lambda = qr_params[0];

    [unroll]
    for (int k = 0; k < PD; k++) { pdata[j * PD + k] = 0.0; }
    // W = 2I − 0 for a frozen point (its C_i are 0, so W is unused, but keep it well defined)
    pdata[j * PD + 18] = 2.0; pdata[j * PD + 21] = 2.0; pdata[j * PD + 23] = 2.0;

    // Column scaling from diag(J̃pᵀJ̃p)
    float d[3] = { 0, 0, 0 };
    for (int k = 0; k < count; k++) {
        int b = sorted_obs[start + k] * LIN + 12;
        [unroll]
        for (int c = 0; c < 3; c++) {
            d[c] += lin[b + c] * lin[b + c] + lin[b + 3 + c] * lin[b + 3 + c];
        }
    }
    if (!(d[0] > 1e-20 && d[1] > 1e-20 && d[2] > 1e-20)) { return; }
    float s[3] = { 1.0 / sqrt(d[0]), 1.0 / sqrt(d[1]), 1.0 / sqrt(d[2]) };

    // Gram passes over the rows of [J̃p·S ; √λ·I]·m (helpers cannot read the buffers, so the row
    // loop is inline). Pass 0 is shifted: s = 11·(m·n + n(n+1))·u·‖A‖², ‖A‖² ≤ trace(G) (u = 2⁻²⁴,
    // n = 3, m = 2·count + 3). Passes 1 and 2 are CholeskyQR2 on its result; a Gram pivot this small
    // there means the point is unconstrained in that direction. Pass 3 only measures QᵀQ.
    float m[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    float g[6];
    float r[6];
    for (int pass = 0; pass < 4; pass++) {
        g = { 0, 0, 0, 0, 0, 0 };
        for (int k = 0; k < count; k++) {
            int b = sorted_obs[start + k] * LIN + 12;
            [unroll]
            for (int row = 0; row < 2; row++) {
                float a0 = lin[b + row * 3 + 0] * s[0];
                float a1 = lin[b + row * 3 + 1] * s[1];
                float a2 = lin[b + row * 3 + 2] * s[2];
                addOuter(g,
                         a0 * m[0] + a1 * m[3] + a2 * m[6],
                         a0 * m[1] + a1 * m[4] + a2 * m[7],
                         a0 * m[2] + a1 * m[5] + a2 * m[8]);
            }
        }
        addDamping(g, lambda, m);
        if (pass == 3) { break; }
        if (pass == 0) {
            float shift = 11.0 * (3.0 * float(2 * count + 3) + 12.0) * 5.96e-8 * (g[0] + g[3] + g[5]);
            float gs[6] = { g[0] + shift, g[1], g[2], g[3] + shift, g[4], g[5] + shift };
            if (!chol3(gs, 0.0, r)) { return; }
        } else {
            if (!chol3(g, 1e-7, r)) { return; }
        }
        mulUpperInv(m, r);
    }

    // S·R⁻¹ = S·m (row c scaled by s_c)
    float sm[9];
    [unroll]
    for (int row = 0; row < 3; row++) {
        [unroll]
        for (int c = 0; c < 3; c++) { sm[row * 3 + c] = s[row] * m[row * 3 + c]; }
    }

    // q and C_k (QᵀQ is the last Gram pass)
    float q[3] = { 0, 0, 0 };
    float ck[6] = { 0, 0, 0, 0, 0, 0 };
    float gq[6] = { g[0], g[1], g[2], g[3], g[4], g[5] };
    for (int k = 0; k < count; k++) {
        int b = sorted_obs[start + k] * LIN;
        [unroll]
        for (int row = 0; row < 2; row++) {
            float p0 = lin[b + 12 + row * 3 + 0];
            float p1 = lin[b + 12 + row * 3 + 1];
            float p2 = lin[b + 12 + row * 3 + 2];
            // Q row = J̃p row · S·R⁻¹
            float q0 = p0 * sm[0] + p1 * sm[3] + p2 * sm[6];
            float q1 = p0 * sm[1] + p1 * sm[4] + p2 * sm[7];
            float q2 = p0 * sm[2] + p1 * sm[5] + p2 * sm[8];
            float rr = lin[b + 22 + row];
            float jk0 = lin[b + 18 + row * 2 + 0];
            float jk1 = lin[b + 18 + row * 2 + 1];
            q[0] += q0 * rr; q[1] += q1 * rr; q[2] += q2 * rr;
            ck[0] += q0 * jk0; ck[1] += q0 * jk1;
            ck[2] += q1 * jk0; ck[3] += q1 * jk1;
            ck[4] += q2 * jk0; ck[5] += q2 * jk1;
        }
    }

    int o = j * PD;
    [unroll]
    for (int k = 0; k < 9; k++) { pdata[o + k] = sm[k]; }
    pdata[o + 9] = q[0]; pdata[o + 10] = q[1]; pdata[o + 11] = q[2];
    [unroll]
    for (int k = 0; k < 6; k++) { pdata[o + 12 + k] = ck[k]; }
    pdata[o + 18] = 2.0 - gq[0];
    pdata[o + 19] = -gq[1];
    pdata[o + 20] = -gq[2];
    pdata[o + 21] = 2.0 - gq[3];
    pdata[o + 22] = -gq[4];
    pdata[o + 23] = 2.0 - gq[5];
}
