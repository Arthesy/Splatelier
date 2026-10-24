#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* upd_dims_0;
    int device* point_obs_starts_0;
    int device* point_obs_counts_0;
    int device* sorted_obs_0;
    int device* obs_cam_idx_0;
    float device* lin_0;
    float device* delta_cam_0;
    float device* pdata_0;
    float device* delta_k_0;
    float device* delta_pt_0;
};

[[kernel]] void ba_point_update(uint3 tid_0 [[thread_position_in_grid]], int device* upd_dims_1 [[buffer(8)]], int device* point_obs_starts_1 [[buffer(2)]], int device* point_obs_counts_1 [[buffer(3)]], int device* sorted_obs_1 [[buffer(1)]], int device* obs_cam_idx_1 [[buffer(4)]], float device* lin_1 [[buffer(0)]], float device* delta_cam_1 [[buffer(6)]], float device* pdata_1 [[buffer(5)]], float device* delta_k_1 [[buffer(7)]], float device* delta_pt_1 [[buffer(9)]])
{
    int y_0;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->upd_dims_0 = upd_dims_1;
    (&kernelContext_0)->point_obs_starts_0 = point_obs_starts_1;
    (&kernelContext_0)->point_obs_counts_0 = point_obs_counts_1;
    (&kernelContext_0)->sorted_obs_0 = sorted_obs_1;
    (&kernelContext_0)->obs_cam_idx_0 = obs_cam_idx_1;
    (&kernelContext_0)->lin_0 = lin_1;
    (&kernelContext_0)->delta_cam_0 = delta_cam_1;
    (&kernelContext_0)->pdata_0 = pdata_1;
    (&kernelContext_0)->delta_k_0 = delta_k_1;
    (&kernelContext_0)->delta_pt_0 = delta_pt_1;
    int j_0 = int(tid_0.x);
    if(j_0 >= upd_dims_1[int(0)])
    {
        return;
    }
    int _S1 = (&kernelContext_0)->point_obs_starts_0[j_0];
    int _S2 = (&kernelContext_0)->point_obs_counts_0[j_0];
    int o_0 = j_0 * int(24);
    float3 _S3 = float3(0.0, 0.0, 0.0);
    int a_0 = int(0);
    float3 v_0 = _S3;
    for(;;)
    {
        if(a_0 < _S2)
        {
        }
        else
        {
            break;
        }
        int obsI_0 = (&kernelContext_0)->sorted_obs_0[_S1 + a_0];
        int b_0 = obsI_0 * int(24);
        int _S4 = (&kernelContext_0)->obs_cam_idx_0[obsI_0] * int(6);
        y_0 = int(0);
        float e0_0 = 0.0;
        float e1_0 = 0.0;
        for(;;)
        {
            if(y_0 < int(6))
            {
            }
            else
            {
                break;
            }
            int _S5 = _S4 + y_0;
            float e0_1 = e0_0 + (&kernelContext_0)->lin_0[b_0 + y_0] * (&kernelContext_0)->delta_cam_0[_S5];
            float e1_1 = e1_0 + (&kernelContext_0)->lin_0[b_0 + int(6) + y_0] * (&kernelContext_0)->delta_cam_0[_S5];
            y_0 = y_0 + int(1);
            e0_0 = e0_1;
            e1_0 = e1_1;
        }
        float3 v_1 = v_0 + (float3((&kernelContext_0)->lin_0[b_0 + int(12)], (&kernelContext_0)->lin_0[b_0 + int(13)], (&kernelContext_0)->lin_0[b_0 + int(14)]) * float3(e0_0)  + float3((&kernelContext_0)->lin_0[b_0 + int(15)], (&kernelContext_0)->lin_0[b_0 + int(16)], (&kernelContext_0)->lin_0[b_0 + int(17)]) * float3(e1_0) );
        a_0 = a_0 + int(1);
        v_0 = v_1;
    }
    thread array<float, int(9)> sm_0;
    y_0 = int(0);
    for(;;)
    {
        if(y_0 < int(9))
        {
        }
        else
        {
            break;
        }
        sm_0[y_0] = (&kernelContext_0)->pdata_0[o_0 + y_0];
        y_0 = y_0 + int(1);
    }
    float dk0_0 = (&kernelContext_0)->delta_k_0[int(0)];
    float dk1_0 = (&kernelContext_0)->delta_k_0[int(1)];
    float _S6 = v_0.x;
    float _S7 = v_0.y;
    float _S8 = v_0.z;
    float3 h_0 = float3((&kernelContext_0)->pdata_0[o_0 + int(9)], (&kernelContext_0)->pdata_0[o_0 + int(10)], (&kernelContext_0)->pdata_0[o_0 + int(11)]) + float3(sm_0[int(0)] * _S6 + sm_0[int(3)] * _S7 + sm_0[int(6)] * _S8, sm_0[int(1)] * _S6 + sm_0[int(4)] * _S7 + sm_0[int(7)] * _S8, sm_0[int(2)] * _S6 + sm_0[int(5)] * _S7 + sm_0[int(8)] * _S8) + float3((&kernelContext_0)->pdata_0[o_0 + int(12)] * dk0_0 + (&kernelContext_0)->pdata_0[o_0 + int(13)] * dk1_0, (&kernelContext_0)->pdata_0[o_0 + int(14)] * dk0_0 + (&kernelContext_0)->pdata_0[o_0 + int(15)] * dk1_0, (&kernelContext_0)->pdata_0[o_0 + int(16)] * dk0_0 + (&kernelContext_0)->pdata_0[o_0 + int(17)] * dk1_0);
    int _S9 = j_0 * int(3);
    float _S10 = h_0.x;
    float _S11 = h_0.y;
    float _S12 = h_0.z;
    *((&kernelContext_0)->delta_pt_0+_S9) = - (sm_0[int(0)] * _S10 + sm_0[int(1)] * _S11 + sm_0[int(2)] * _S12);
    *((&kernelContext_0)->delta_pt_0+(_S9 + int(1))) = - (sm_0[int(3)] * _S10 + sm_0[int(4)] * _S11 + sm_0[int(5)] * _S12);
    *((&kernelContext_0)->delta_pt_0+(_S9 + int(2))) = - (sm_0[int(6)] * _S10 + sm_0[int(7)] * _S11 + sm_0[int(8)] * _S12);
    return;
}

