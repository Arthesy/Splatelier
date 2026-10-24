// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Stage (3) of the point elimination (see ba_point_qr.slang): 1 thread / point back-substitutes
// the point step from the camera and distortion steps,
//   δp = −S·R⁻¹·(q + (S·R⁻¹)ᵀ·Σ J̃p_iᵀ·J̃c_i·δc_i + C_k·δk).
// δc of fixed cameras is passed as 0; δk is 0 when k is not estimated.

static const int LIN = 24;
static const int PD = 24;

StructuredBuffer<float> lin;              // [nObs * 24] (ba_linearize)
StructuredBuffer<int>   sorted_obs;       // [nObs]
StructuredBuffer<int>   point_obs_starts; // [nPt]
StructuredBuffer<int>   point_obs_counts; // [nPt]
StructuredBuffer<int>   obs_cam_idx;      // [nObs]
StructuredBuffer<float> pdata;            // [nPt * 24] (ba_point_qr)
StructuredBuffer<float> delta_cam;        // [nCam * 6]
StructuredBuffer<float> delta_k;          // [2]
StructuredBuffer<int>   upd_dims;         // [1] : nPt

RWStructuredBuffer<float> delta_pt;       // [nPt * 3]

[shader("compute")]
[numthreads(128, 1, 1)]
void ba_point_update(uint3 tid : SV_DispatchThreadID)
{
    int j = int(tid.x);
    if (j >= upd_dims[0]) { return; }
    int start = point_obs_starts[j];
    int count = point_obs_counts[j];
    int o = j * PD;

    // v = Σ J̃p_iᵀ·(J̃c_i·δc_i)
    float3 v = float3(0, 0, 0);
    for (int a = 0; a < count; a++) {
        int obsI = sorted_obs[start + a];
        int b = obsI * LIN;
        int dc = obs_cam_idx[obsI] * 6;
        float e0 = 0, e1 = 0;
        [unroll]
        for (int y = 0; y < 6; y++) {
            e0 += lin[b + y] * delta_cam[dc + y];
            e1 += lin[b + 6 + y] * delta_cam[dc + y];
        }
        v += float3(lin[b + 12], lin[b + 13], lin[b + 14]) * e0
           + float3(lin[b + 15], lin[b + 16], lin[b + 17]) * e1;
    }
    float sm[9];
    [unroll]
    for (int k = 0; k < 9; k++) { sm[k] = pdata[o + k]; }
    float dk0 = delta_k[0], dk1 = delta_k[1];
    // h = q + smᵀ·v + C_k·δk
    float3 h = float3(pdata[o + 9], pdata[o + 10], pdata[o + 11])
             + float3(sm[0] * v.x + sm[3] * v.y + sm[6] * v.z,
                      sm[1] * v.x + sm[4] * v.y + sm[7] * v.z,
                      sm[2] * v.x + sm[5] * v.y + sm[8] * v.z)
             + float3(pdata[o + 12] * dk0 + pdata[o + 13] * dk1,
                      pdata[o + 14] * dk0 + pdata[o + 15] * dk1,
                      pdata[o + 16] * dk0 + pdata[o + 17] * dk1);
    delta_pt[j * 3 + 0] = -(sm[0] * h.x + sm[1] * h.y + sm[2] * h.z);
    delta_pt[j * 3 + 1] = -(sm[3] * h.x + sm[4] * h.y + sm[5] * h.z);
    delta_pt[j * 3 + 2] = -(sm[6] * h.x + sm[7] * h.y + sm[8] * h.z);
}
