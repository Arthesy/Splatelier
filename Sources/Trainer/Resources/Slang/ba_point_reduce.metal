#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
float3 applyW_0(const array<float, int(6)> thread* w_0, float3 v_0)
{
    float _S1 = v_0.x;
    float _S2 = v_0.y;
    float _S3 = v_0.z;
    return float3((*w_0)[int(0)] * _S1 + (*w_0)[int(1)] * _S2 + (*w_0)[int(2)] * _S3, (*w_0)[int(1)] * _S1 + (*w_0)[int(3)] * _S2 + (*w_0)[int(4)] * _S3, (*w_0)[int(2)] * _S1 + (*w_0)[int(4)] * _S2 + (*w_0)[int(5)] * _S3);
}

void projectCam_0(const array<float, int(12)> thread* jc_0, const array<float, int(6)> thread* jp_0, const array<float, int(9)> thread* sm_0, array<float, int(18)> thread* c_0)
{
    int y_0;
    thread array<float, int(18)> h_0;
    int x_0 = int(0);
    for(;;)
    {
        if(x_0 < int(3))
        {
        }
        else
        {
            break;
        }
        y_0 = int(0);
        for(;;)
        {
            if(y_0 < int(6))
            {
            }
            else
            {
                break;
            }
            h_0[x_0 * int(6) + y_0] = (*jp_0)[x_0] * (*jc_0)[y_0] + (*jp_0)[int(3) + x_0] * (*jc_0)[int(6) + y_0];
            y_0 = y_0 + int(1);
        }
        x_0 = x_0 + int(1);
    }
    x_0 = int(0);
    for(;;)
    {
        if(x_0 < int(3))
        {
        }
        else
        {
            break;
        }
        y_0 = int(0);
        for(;;)
        {
            if(y_0 < int(6))
            {
            }
            else
            {
                break;
            }
            (*c_0)[x_0 * int(6) + y_0] = (*sm_0)[x_0] * h_0[y_0] + (*sm_0)[int(3) + x_0] * h_0[int(6) + y_0] + (*sm_0)[int(6) + x_0] * h_0[int(12) + y_0];
            y_0 = y_0 + int(1);
        }
        x_0 = x_0 + int(1);
    }
    return;
}

struct KernelContext_0
{
    int device* red_dims_0;
    int device* point_obs_starts_0;
    int device* point_obs_counts_0;
    float device* pdata_0;
    int device* sorted_obs_0;
    int device* obs_cam_idx_0;
    float device* lin_0;
    float device* cam_diag_0;
    float device* redRHS_0;
    float device* redLHS_0;
    float device* k_diag_0;
};

[[kernel]] void ba_point_reduce(uint3 tid_0 [[thread_position_in_grid]], int device* red_dims_1 [[buffer(6)]], int device* point_obs_starts_1 [[buffer(2)]], int device* point_obs_counts_1 [[buffer(3)]], float device* pdata_1 [[buffer(5)]], int device* sorted_obs_1 [[buffer(1)]], int device* obs_cam_idx_1 [[buffer(4)]], float device* lin_1 [[buffer(0)]], float device* cam_diag_1 [[buffer(9)]], float device* redRHS_1 [[buffer(8)]], float device* redLHS_1 [[buffer(7)]], float device* k_diag_1 [[buffer(10)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->red_dims_0 = red_dims_1;
    (&kernelContext_0)->point_obs_starts_0 = point_obs_starts_1;
    (&kernelContext_0)->point_obs_counts_0 = point_obs_counts_1;
    (&kernelContext_0)->pdata_0 = pdata_1;
    (&kernelContext_0)->sorted_obs_0 = sorted_obs_1;
    (&kernelContext_0)->obs_cam_idx_0 = obs_cam_idx_1;
    (&kernelContext_0)->lin_0 = lin_1;
    (&kernelContext_0)->cam_diag_0 = cam_diag_1;
    (&kernelContext_0)->redRHS_0 = redRHS_1;
    (&kernelContext_0)->redLHS_0 = redLHS_1;
    (&kernelContext_0)->k_diag_0 = k_diag_1;
    int j_0 = int(tid_0.x);
    int D_0 = red_dims_1[int(1)];
    int _S4 = red_dims_1[int(2)];
    int kOff_0 = red_dims_1[int(3)];
    float kk00_0;
    float kk01_0;
    float kk11_0;
    float rk0_0;
    float rk1_0;
    float kd0_0;
    float kd1_0;
    if(j_0 < red_dims_1[int(0)])
    {
        int _S5 = (&kernelContext_0)->point_obs_starts_0[j_0];
        int _S6 = (&kernelContext_0)->point_obs_counts_0[j_0];
        int o_0 = j_0 * int(24);
        thread array<float, int(9)> sm_1;
        int k_0 = int(0);
        for(;;)
        {
            if(k_0 < int(9))
            {
            }
            else
            {
                break;
            }
            sm_1[k_0] = (&kernelContext_0)->pdata_0[o_0 + k_0];
            k_0 = k_0 + int(1);
        }
        float3 q_0 = float3((&kernelContext_0)->pdata_0[o_0 + int(9)], (&kernelContext_0)->pdata_0[o_0 + int(10)], (&kernelContext_0)->pdata_0[o_0 + int(11)]);
        thread array<float, int(6)> w_1;
        k_0 = int(0);
        for(;;)
        {
            if(k_0 < int(6))
            {
            }
            else
            {
                break;
            }
            w_1[k_0] = (&kernelContext_0)->pdata_0[o_0 + int(18) + k_0];
            k_0 = k_0 + int(1);
        }
        thread array<float, int(6)> _S7 = w_1;
        float3 _S8 = applyW_0(&_S7, q_0);
        float3 ck0_0 = float3((&kernelContext_0)->pdata_0[o_0 + int(12)], (&kernelContext_0)->pdata_0[o_0 + int(14)], (&kernelContext_0)->pdata_0[o_0 + int(16)]);
        float3 ck1_0 = float3((&kernelContext_0)->pdata_0[o_0 + int(13)], (&kernelContext_0)->pdata_0[o_0 + int(15)], (&kernelContext_0)->pdata_0[o_0 + int(17)]);
        thread array<float, int(6)> _S9 = w_1;
        float3 _S10 = applyW_0(&_S9, ck0_0);
        thread array<float, int(6)> _S11 = w_1;
        float3 _S12 = applyW_0(&_S11, ck1_0);
        bool _S13 = kOff_0 >= int(0);
        if(_S13)
        {
            float kk01_1 = 0.0 - dot(ck0_0, _S12);
            float kk11_1 = 0.0 - dot(ck1_0, _S12);
            float _S14 = dot(ck0_0, _S8);
            float _S15 = dot(ck1_0, _S8);
            kk00_0 = 0.0 - dot(ck0_0, _S10);
            kk01_0 = kk01_1;
            kk11_0 = kk11_1;
            rk0_0 = _S14;
            rk1_0 = _S15;
        }
        else
        {
            kk00_0 = 0.0;
            kk01_0 = 0.0;
            kk11_0 = 0.0;
            rk0_0 = 0.0;
            rk1_0 = 0.0;
        }
        int a_0 = int(0);
        kd0_0 = 0.0;
        kd1_0 = 0.0;
        for(;;)
        {
            if(a_0 < _S6)
            {
            }
            else
            {
                break;
            }
            int obsI_0 = (&kernelContext_0)->sorted_obs_0[_S5 + a_0];
            int camI_0 = (&kernelContext_0)->obs_cam_idx_0[obsI_0];
            int bI_0 = obsI_0 * int(24);
            float jk00_0 = (&kernelContext_0)->lin_0[bI_0 + int(18)];
            float jk01_0 = (&kernelContext_0)->lin_0[bI_0 + int(19)];
            float jk10_0 = (&kernelContext_0)->lin_0[bI_0 + int(20)];
            float jk11_0 = (&kernelContext_0)->lin_0[bI_0 + int(21)];
            float r0_0 = (&kernelContext_0)->lin_0[bI_0 + int(22)];
            float r1_0 = (&kernelContext_0)->lin_0[bI_0 + int(23)];
            int y_1 = int(0);
            for(;;)
            {
                if(y_1 < int(6))
                {
                }
                else
                {
                    break;
                }
                int _S16 = bI_0 + y_1;
                int _S17 = bI_0 + int(6) + y_1;
                float _S18 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->cam_diag_0+(camI_0 * int(6) + y_1))), (&kernelContext_0)->lin_0[_S16] * (&kernelContext_0)->lin_0[_S16] + (&kernelContext_0)->lin_0[_S17] * (&kernelContext_0)->lin_0[_S17], memory_order_relaxed);
                y_1 = y_1 + int(1);
            }
            float _S19 = jk00_0 * jk00_0 + jk10_0 * jk10_0;
            float kd0_1 = kd0_0 + _S19;
            float _S20 = jk01_0 * jk01_0 + jk11_0 * jk11_0;
            float kd1_1 = kd1_0 + _S20;
            if(_S13)
            {
                float kk01_2 = kk01_0 + (jk00_0 * jk01_0 + jk10_0 * jk11_0);
                float kk11_2 = kk11_0 + _S20;
                float rk0_1 = rk0_0 - (jk00_0 * r0_0 + jk10_0 * r1_0);
                float rk1_1 = rk1_0 - (jk01_0 * r0_0 + jk11_0 * r1_0);
                kk00_0 = kk00_0 + _S19;
                kk01_0 = kk01_2;
                kk11_0 = kk11_2;
                rk0_0 = rk0_1;
                rk1_0 = rk1_1;
            }
            if(camI_0 < _S4)
            {
                a_0 = a_0 + int(1);
                kd0_0 = kd0_1;
                kd1_0 = kd1_1;
                continue;
            }
            int _S21 = camI_0 - _S4;
            thread array<float, int(12)> jcI_0;
            thread array<float, int(6)> jpI_0;
            int y_2 = int(0);
            for(;;)
            {
                if(y_2 < int(12))
                {
                }
                else
                {
                    break;
                }
                jcI_0[y_2] = (&kernelContext_0)->lin_0[bI_0 + y_2];
                y_2 = y_2 + int(1);
            }
            int y_3 = int(0);
            for(;;)
            {
                if(y_3 < int(6))
                {
                }
                else
                {
                    break;
                }
                jpI_0[y_3] = (&kernelContext_0)->lin_0[bI_0 + int(12) + y_3];
                y_3 = y_3 + int(1);
            }
            thread array<float, int(12)> _S22 = jcI_0;
            thread array<float, int(6)> _S23 = jpI_0;
            thread array<float, int(9)> _S24 = sm_1;
            thread array<float, int(18)> cI_0;
            projectCam_0(&_S22, &_S23, &_S24, &cI_0);
            int y_4 = int(0);
            for(;;)
            {
                if(y_4 < int(6))
                {
                }
                else
                {
                    break;
                }
                float _S25 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redRHS_0+(_S21 * int(6) + y_4))), - ((&kernelContext_0)->lin_0[bI_0 + y_4] * r0_0 + (&kernelContext_0)->lin_0[bI_0 + int(6) + y_4] * r1_0) + cI_0[y_4] * _S8.x + cI_0[int(6) + y_4] * _S8.y + cI_0[int(12) + y_4] * _S8.z, memory_order_relaxed);
                y_4 = y_4 + int(1);
            }
            int y_5;
            if(_S13)
            {
                y_5 = int(0);
                for(;;)
                {
                    if(y_5 < int(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    float c0_0 = (&kernelContext_0)->lin_0[bI_0 + y_5];
                    float c1_0 = (&kernelContext_0)->lin_0[bI_0 + int(6) + y_5];
                    float3 ciy_0 = float3(cI_0[y_5], cI_0[int(6) + y_5], cI_0[int(12) + y_5]);
                    float s0_0 = jk00_0 * c0_0 + jk10_0 * c1_0 - dot(_S10, ciy_0);
                    float s1_0 = jk01_0 * c0_0 + jk11_0 * c1_0 - dot(_S12, ciy_0);
                    int _S26 = _S21 * int(6);
                    float _S27 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(kOff_0 * D_0 + _S26 + y_5))), s0_0, memory_order_relaxed);
                    float _S28 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+((kOff_0 + int(1)) * D_0 + _S26 + y_5))), s1_0, memory_order_relaxed);
                    int _S29 = (_S26 + y_5) * D_0 + kOff_0;
                    float _S30 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+_S29)), s0_0, memory_order_relaxed);
                    float _S31 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(_S29 + int(1)))), s1_0, memory_order_relaxed);
                    y_5 = y_5 + int(1);
                }
            }
            thread array<float, int(18)> wcI_0;
            y_5 = int(0);
            for(;;)
            {
                if(y_5 < int(6))
                {
                }
                else
                {
                    break;
                }
                int _S32 = int(6) + y_5;
                int _S33 = int(12) + y_5;
                float3 _S34 = float3(cI_0[y_5], cI_0[_S32], cI_0[_S33]);
                thread array<float, int(6)> _S35 = w_1;
                float3 _S36 = applyW_0(&_S35, _S34);
                wcI_0[y_5] = _S36.x;
                wcI_0[_S32] = _S36.y;
                wcI_0[_S33] = _S36.z;
                y_5 = y_5 + int(1);
            }
            int b2_0 = int(0);
            for(;;)
            {
                if(b2_0 < _S6)
                {
                }
                else
                {
                    break;
                }
                int obsL_0 = (&kernelContext_0)->sorted_obs_0[_S5 + b2_0];
                int camL_0 = (&kernelContext_0)->obs_cam_idx_0[obsL_0];
                if(camL_0 < _S4)
                {
                    b2_0 = b2_0 + int(1);
                    continue;
                }
                int _S37 = camL_0 - _S4;
                int _S38 = obsL_0 * int(24);
                thread array<float, int(12)> jcL_0;
                thread array<float, int(6)> jpL_0;
                int y_6 = int(0);
                for(;;)
                {
                    if(y_6 < int(12))
                    {
                    }
                    else
                    {
                        break;
                    }
                    jcL_0[y_6] = (&kernelContext_0)->lin_0[_S38 + y_6];
                    y_6 = y_6 + int(1);
                }
                int y_7 = int(0);
                for(;;)
                {
                    if(y_7 < int(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    jpL_0[y_7] = (&kernelContext_0)->lin_0[_S38 + int(12) + y_7];
                    y_7 = y_7 + int(1);
                }
                thread array<float, int(12)> _S39 = jcL_0;
                thread array<float, int(6)> _S40 = jpL_0;
                thread array<float, int(9)> _S41 = sm_1;
                thread array<float, int(18)> cL_0;
                projectCam_0(&_S39, &_S40, &_S41, &cL_0);
                bool _S42 = b2_0 == a_0;
                int x_1 = int(0);
                for(;;)
                {
                    if(x_1 < int(6))
                    {
                    }
                    else
                    {
                        break;
                    }
                    int _S43 = (_S21 * int(6) + x_1) * D_0 + _S37 * int(6);
                    int y_8 = int(0);
                    for(;;)
                    {
                        if(y_8 < int(6))
                        {
                        }
                        else
                        {
                            break;
                        }
                        float v_1 = - (wcI_0[x_1] * cL_0[y_8] + wcI_0[int(6) + x_1] * cL_0[int(6) + y_8] + wcI_0[int(12) + x_1] * cL_0[int(12) + y_8]);
                        float v_2;
                        if(_S42)
                        {
                            int _S44 = bI_0 + int(6);
                            v_2 = v_1 + ((&kernelContext_0)->lin_0[bI_0 + x_1] * (&kernelContext_0)->lin_0[bI_0 + y_8] + (&kernelContext_0)->lin_0[_S44 + x_1] * (&kernelContext_0)->lin_0[_S44 + y_8]);
                        }
                        else
                        {
                            v_2 = v_1;
                        }
                        float _S45 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(_S43 + y_8))), v_2, memory_order_relaxed);
                        y_8 = y_8 + int(1);
                    }
                    x_1 = x_1 + int(1);
                }
                b2_0 = b2_0 + int(1);
            }
            a_0 = a_0 + int(1);
            kd0_0 = kd0_1;
            kd1_0 = kd1_1;
        }
    }
    else
    {
        kd0_0 = 0.0;
        kd1_0 = 0.0;
        kk00_0 = 0.0;
        kk01_0 = 0.0;
        kk11_0 = 0.0;
        rk0_0 = 0.0;
        rk1_0 = 0.0;
    }
    float s_kd0_0 = simd_sum(kd0_0);
    float s_kd1_0 = simd_sum(kd1_0);
    float s_kk00_0 = simd_sum(kk00_0);
    float s_kk01_0 = simd_sum(kk01_0);
    float s_kk11_0 = simd_sum(kk11_0);
    float s_rk0_0 = simd_sum(rk0_0);
    float s_rk1_0 = simd_sum(rk1_0);
    bool _S46 = simd_is_first();
    if(_S46)
    {
        float _S47 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->k_diag_0+int(0))), s_kd0_0, memory_order_relaxed);
        float _S48 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->k_diag_0+int(1))), s_kd1_0, memory_order_relaxed);
        if(kOff_0 >= int(0))
        {
            int _S49 = kOff_0 * D_0 + kOff_0;
            float _S50 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+_S49)), s_kk00_0, memory_order_relaxed);
            float _S51 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(_S49 + int(1)))), s_kk01_0, memory_order_relaxed);
            int _S52 = kOff_0 + int(1);
            int _S53 = _S52 * D_0 + kOff_0;
            float _S54 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+_S53)), s_kk01_0, memory_order_relaxed);
            float _S55 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redLHS_0+(_S53 + int(1)))), s_kk11_0, memory_order_relaxed);
            float _S56 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redRHS_0+kOff_0)), s_rk0_0, memory_order_relaxed);
            float _S57 = atomic_fetch_add_explicit(((atomic_float device*)((&kernelContext_0)->redRHS_0+_S52)), s_rk1_0, memory_order_relaxed);
        }
    }
    return;
}

