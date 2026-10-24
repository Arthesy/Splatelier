#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
void addOuter_0(array<float, int(6)> thread* g_0, float q0_0, float q1_0, float q2_0)
{
    (*g_0)[int(0)] = (*g_0)[int(0)] + q0_0 * q0_0;
    (*g_0)[int(1)] = (*g_0)[int(1)] + q0_0 * q1_0;
    (*g_0)[int(2)] = (*g_0)[int(2)] + q0_0 * q2_0;
    (*g_0)[int(3)] = (*g_0)[int(3)] + q1_0 * q1_0;
    (*g_0)[int(4)] = (*g_0)[int(4)] + q1_0 * q2_0;
    (*g_0)[int(5)] = (*g_0)[int(5)] + q2_0 * q2_0;
    return;
}

void addDamping_0(array<float, int(6)> thread* g_1, float lambda_0, const array<float, int(9)> thread* m_0)
{
    float _S1 = (*m_0)[int(0)];
    float _S2 = (*m_0)[int(3)];
    float _S3 = (*m_0)[int(6)];
    (*g_1)[int(0)] = (*g_1)[int(0)] + lambda_0 * ((*m_0)[int(0)] * (*m_0)[int(0)] + (*m_0)[int(3)] * (*m_0)[int(3)] + (*m_0)[int(6)] * (*m_0)[int(6)]);
    float _S4 = (*m_0)[int(1)];
    float _S5 = (*m_0)[int(4)];
    float _S6 = (*m_0)[int(7)];
    (*g_1)[int(1)] = (*g_1)[int(1)] + lambda_0 * (_S1 * (*m_0)[int(1)] + _S2 * (*m_0)[int(4)] + _S3 * (*m_0)[int(7)]);
    float _S7 = (*m_0)[int(2)];
    float _S8 = (*m_0)[int(5)];
    float _S9 = (*m_0)[int(8)];
    (*g_1)[int(2)] = (*g_1)[int(2)] + lambda_0 * (_S1 * (*m_0)[int(2)] + _S2 * (*m_0)[int(5)] + _S3 * (*m_0)[int(8)]);
    (*g_1)[int(3)] = (*g_1)[int(3)] + lambda_0 * (_S4 * _S4 + _S5 * _S5 + _S6 * _S6);
    (*g_1)[int(4)] = (*g_1)[int(4)] + lambda_0 * (_S4 * _S7 + _S5 * _S8 + _S6 * _S9);
    (*g_1)[int(5)] = (*g_1)[int(5)] + lambda_0 * (_S7 * _S7 + _S8 * _S8 + _S9 * _S9);
    return;
}

bool chol3_0(const array<float, int(6)> thread* g_2, float minPivot_0, array<float, int(6)> thread* r_0)
{
    (*r_0)[int(0)] = 0.0;
    (*r_0)[int(1)] = 0.0;
    (*r_0)[int(2)] = 0.0;
    (*r_0)[int(3)] = 0.0;
    (*r_0)[int(4)] = 0.0;
    (*r_0)[int(5)] = 0.0;
    float _S10 = (*g_2)[int(0)];
    if(!(((*g_2)[int(0)]) > minPivot_0))
    {
        return false;
    }
    float _S11 = sqrt(_S10);
    (*r_0)[int(0)] = _S11;
    float _S12 = (*g_2)[int(1)] / _S11;
    (*r_0)[int(1)] = _S12;
    (*r_0)[int(2)] = (*g_2)[int(2)] / _S11;
    float p1_0 = (*g_2)[int(3)] - _S12 * _S12;
    if(!(p1_0 > minPivot_0))
    {
        return false;
    }
    float _S13 = sqrt(p1_0);
    (*r_0)[int(3)] = _S13;
    float _S14 = ((*g_2)[int(4)] - (*r_0)[int(1)] * (*r_0)[int(2)]) / _S13;
    (*r_0)[int(4)] = _S14;
    float p2_0 = (*g_2)[int(5)] - (*r_0)[int(2)] * (*r_0)[int(2)] - _S14 * _S14;
    if(!(p2_0 > minPivot_0))
    {
        return false;
    }
    (*r_0)[int(5)] = sqrt(p2_0);
    return true;
}

void mulUpperInv_0(array<float, int(9)> thread* m_1, const array<float, int(6)> thread* r_1)
{
    int row_0 = int(0);
    for(;;)
    {
        if(row_0 < int(3))
        {
        }
        else
        {
            break;
        }
        int _S15 = row_0 * int(3);
        float x0_0 = (*m_1)[_S15] / (*r_1)[int(0)];
        float x1_0 = ((*m_1)[_S15 + int(1)] - x0_0 * (*r_1)[int(1)]) / (*r_1)[int(3)];
        float x2_0 = ((*m_1)[_S15 + int(2)] - x0_0 * (*r_1)[int(2)] - x1_0 * (*r_1)[int(4)]) / (*r_1)[int(5)];
        (*m_1)[_S15] = x0_0;
        (*m_1)[_S15 + int(1)] = x1_0;
        (*m_1)[_S15 + int(2)] = x2_0;
        row_0 = row_0 + int(1);
    }
    return;
}

struct KernelContext_0
{
    int device* qr_dims_0;
    int device* point_obs_starts_0;
    int device* point_obs_counts_0;
    float device* qr_params_0;
    float device* pdata_0;
    int device* sorted_obs_0;
    float device* lin_0;
};

[[kernel]] void ba_point_qr(uint3 tid_0 [[thread_position_in_grid]], int device* qr_dims_1 [[buffer(5)]], int device* point_obs_starts_1 [[buffer(2)]], int device* point_obs_counts_1 [[buffer(3)]], float device* qr_params_1 [[buffer(4)]], float device* pdata_1 [[buffer(6)]], int device* sorted_obs_1 [[buffer(1)]], float device* lin_1 [[buffer(0)]])
{
    int c_0;
    int row_1;
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->qr_dims_0 = qr_dims_1;
    (&kernelContext_0)->point_obs_starts_0 = point_obs_starts_1;
    (&kernelContext_0)->point_obs_counts_0 = point_obs_counts_1;
    (&kernelContext_0)->qr_params_0 = qr_params_1;
    (&kernelContext_0)->pdata_0 = pdata_1;
    (&kernelContext_0)->sorted_obs_0 = sorted_obs_1;
    (&kernelContext_0)->lin_0 = lin_1;
    int j_0 = int(tid_0.x);
    if(j_0 >= qr_dims_1[int(0)])
    {
        return;
    }
    int _S16 = (&kernelContext_0)->point_obs_starts_0[j_0];
    int _S17 = (&kernelContext_0)->point_obs_counts_0[j_0];
    float _S18 = (&kernelContext_0)->qr_params_0[int(0)];
    int k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(24))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_0)->pdata_0+(j_0 * int(24) + k_0)) = 0.0;
        k_0 = k_0 + int(1);
    }
    int _S19 = j_0 * int(24);
    int _S20 = _S19 + int(18);
    *((&kernelContext_0)->pdata_0+_S20) = 2.0;
    int _S21 = _S19 + int(21);
    *((&kernelContext_0)->pdata_0+_S21) = 2.0;
    int _S22 = _S19 + int(23);
    *((&kernelContext_0)->pdata_0+_S22) = 2.0;
    thread array<float, int(3)> d_0;
    d_0[int(0)] = 0.0;
    d_0[int(1)] = 0.0;
    d_0[int(2)] = 0.0;
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < _S17)
        {
        }
        else
        {
            break;
        }
        int _S23 = (&kernelContext_0)->sorted_obs_0[_S16 + k_0] * int(24) + int(12);
        c_0 = int(0);
        for(;;)
        {
            if(c_0 < int(3))
            {
            }
            else
            {
                break;
            }
            int _S24 = _S23 + c_0;
            int _S25 = _S23 + int(3) + c_0;
            d_0[c_0] = d_0[c_0] + ((&kernelContext_0)->lin_0[_S24] * (&kernelContext_0)->lin_0[_S24] + (&kernelContext_0)->lin_0[_S25] * (&kernelContext_0)->lin_0[_S25]);
            c_0 = c_0 + int(1);
        }
        k_0 = k_0 + int(1);
    }
    bool _S26;
    if((d_0[int(0)]) > 9.99999968265522539e-21)
    {
        _S26 = (d_0[int(1)]) > 9.99999968265522539e-21;
    }
    else
    {
        _S26 = false;
    }
    if(_S26)
    {
        _S26 = (d_0[int(2)]) > 9.99999968265522539e-21;
    }
    else
    {
        _S26 = false;
    }
    if(!_S26)
    {
        return;
    }
    float _S27 = 1.0 / sqrt(d_0[int(0)]);
    float _S28 = 1.0 / sqrt(d_0[int(1)]);
    float _S29 = 1.0 / sqrt(d_0[int(2)]);
    array<float, int(3)> _S30 = { _S27, _S28, _S29 };
    thread array<float, int(9)> m_2;
    m_2[int(0)] = 1.0;
    m_2[int(1)] = 0.0;
    m_2[int(2)] = 0.0;
    m_2[int(3)] = 0.0;
    m_2[int(4)] = 1.0;
    m_2[int(5)] = 0.0;
    m_2[int(6)] = 0.0;
    m_2[int(7)] = 0.0;
    m_2[int(8)] = 1.0;
    thread array<float, int(6)> g_3;
    thread array<float, int(6)> r_2;
    int pass_0 = int(0);
    for(;;)
    {
        if(pass_0 < int(4))
        {
        }
        else
        {
            break;
        }
        g_3[int(0)] = 0.0;
        g_3[int(1)] = 0.0;
        g_3[int(2)] = 0.0;
        g_3[int(3)] = 0.0;
        g_3[int(4)] = 0.0;
        g_3[int(5)] = 0.0;
        k_0 = int(0);
        for(;;)
        {
            if(k_0 < _S17)
            {
            }
            else
            {
                break;
            }
            int _S31 = (&kernelContext_0)->sorted_obs_0[_S16 + k_0] * int(24) + int(12);
            row_1 = int(0);
            for(;;)
            {
                if(row_1 < int(2))
                {
                }
                else
                {
                    break;
                }
                int _S32 = _S31 + row_1 * int(3);
                float a0_0 = (&kernelContext_0)->lin_0[_S32] * _S27;
                float a1_0 = (&kernelContext_0)->lin_0[_S32 + int(1)] * _S28;
                float a2_0 = (&kernelContext_0)->lin_0[_S32 + int(2)] * _S29;
                addOuter_0(&g_3, a0_0 * m_2[int(0)] + a1_0 * m_2[int(3)] + a2_0 * m_2[int(6)], a0_0 * m_2[int(1)] + a1_0 * m_2[int(4)] + a2_0 * m_2[int(7)], a0_0 * m_2[int(2)] + a1_0 * m_2[int(5)] + a2_0 * m_2[int(8)]);
                row_1 = row_1 + int(1);
            }
            k_0 = k_0 + int(1);
        }
        thread array<float, int(9)> _S33 = m_2;
        addDamping_0(&g_3, _S18, &_S33);
        if(pass_0 == int(3))
        {
            break;
        }
        if(pass_0 == int(0))
        {
            float shift_0 = 11.0 * (3.0 * float(int(2) * _S17 + int(3)) + 12.0) * 5.96000013786124327e-08 * (g_3[int(0)] + g_3[int(3)] + g_3[int(5)]);
            float _S34 = g_3[int(3)] + shift_0;
            float _S35 = g_3[int(5)] + shift_0;
            thread array<float, int(6)> _S36;
            _S36[int(0)] = g_3[int(0)] + shift_0;
            _S36[int(1)] = g_3[int(1)];
            _S36[int(2)] = g_3[int(2)];
            _S36[int(3)] = _S34;
            _S36[int(4)] = g_3[int(4)];
            _S36[int(5)] = _S35;
            bool _S37 = chol3_0(&_S36, 0.0, &r_2);
            if(!_S37)
            {
                return;
            }
        }
        else
        {
            thread array<float, int(6)> _S38 = g_3;
            bool _S39 = chol3_0(&_S38, 1.00000001168609742e-07, &r_2);
            if(!_S39)
            {
                return;
            }
        }
        thread array<float, int(6)> _S40 = r_2;
        mulUpperInv_0(&m_2, &_S40);
        pass_0 = pass_0 + int(1);
    }
    thread array<float, int(9)> sm_0;
    row_1 = int(0);
    for(;;)
    {
        if(row_1 < int(3))
        {
        }
        else
        {
            break;
        }
        c_0 = int(0);
        for(;;)
        {
            if(c_0 < int(3))
            {
            }
            else
            {
                break;
            }
            int _S41 = row_1 * int(3) + c_0;
            sm_0[_S41] = _S30[row_1] * m_2[_S41];
            c_0 = c_0 + int(1);
        }
        row_1 = row_1 + int(1);
    }
    thread array<float, int(3)> q_0;
    q_0[int(0)] = 0.0;
    q_0[int(1)] = 0.0;
    q_0[int(2)] = 0.0;
    thread array<float, int(6)> ck_0;
    ck_0[int(0)] = 0.0;
    ck_0[int(1)] = 0.0;
    ck_0[int(2)] = 0.0;
    ck_0[int(3)] = 0.0;
    ck_0[int(4)] = 0.0;
    ck_0[int(5)] = 0.0;
    float _S42 = g_3[int(0)];
    float _S43 = g_3[int(1)];
    float _S44 = g_3[int(2)];
    float _S45 = g_3[int(3)];
    float _S46 = g_3[int(4)];
    float _S47 = g_3[int(5)];
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < _S17)
        {
        }
        else
        {
            break;
        }
        int _S48 = (&kernelContext_0)->sorted_obs_0[_S16 + k_0] * int(24);
        row_1 = int(0);
        for(;;)
        {
            if(row_1 < int(2))
            {
            }
            else
            {
                break;
            }
            int _S49 = _S48 + int(12) + row_1 * int(3);
            float p0_0 = (&kernelContext_0)->lin_0[_S49];
            float p1_1 = (&kernelContext_0)->lin_0[_S49 + int(1)];
            float p2_1 = (&kernelContext_0)->lin_0[_S49 + int(2)];
            float q0_1 = p0_0 * sm_0[int(0)] + p1_1 * sm_0[int(3)] + p2_1 * sm_0[int(6)];
            float q1_1 = p0_0 * sm_0[int(1)] + p1_1 * sm_0[int(4)] + p2_1 * sm_0[int(7)];
            float q2_1 = p0_0 * sm_0[int(2)] + p1_1 * sm_0[int(5)] + p2_1 * sm_0[int(8)];
            float rr_0 = (&kernelContext_0)->lin_0[_S48 + int(22) + row_1];
            int _S50 = _S48 + int(18) + row_1 * int(2);
            float jk0_0 = (&kernelContext_0)->lin_0[_S50];
            float jk1_0 = (&kernelContext_0)->lin_0[_S50 + int(1)];
            q_0[int(0)] = q_0[int(0)] + q0_1 * rr_0;
            q_0[int(1)] = q_0[int(1)] + q1_1 * rr_0;
            q_0[int(2)] = q_0[int(2)] + q2_1 * rr_0;
            ck_0[int(0)] = ck_0[int(0)] + q0_1 * jk0_0;
            ck_0[int(1)] = ck_0[int(1)] + q0_1 * jk1_0;
            ck_0[int(2)] = ck_0[int(2)] + q1_1 * jk0_0;
            ck_0[int(3)] = ck_0[int(3)] + q1_1 * jk1_0;
            ck_0[int(4)] = ck_0[int(4)] + q2_1 * jk0_0;
            ck_0[int(5)] = ck_0[int(5)] + q2_1 * jk1_0;
            row_1 = row_1 + int(1);
        }
        k_0 = k_0 + int(1);
    }
    k_0 = int(0);
    for(;;)
    {
        if(k_0 < int(9))
        {
        }
        else
        {
            break;
        }
        *((&kernelContext_0)->pdata_0+(_S19 + k_0)) = sm_0[k_0];
        k_0 = k_0 + int(1);
    }
    *((&kernelContext_0)->pdata_0+(_S19 + int(9))) = q_0[int(0)];
    *((&kernelContext_0)->pdata_0+(_S19 + int(10))) = q_0[int(1)];
    *((&kernelContext_0)->pdata_0+(_S19 + int(11))) = q_0[int(2)];
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
        *((&kernelContext_0)->pdata_0+(_S19 + int(12) + k_0)) = ck_0[k_0];
        k_0 = k_0 + int(1);
    }
    *((&kernelContext_0)->pdata_0+_S20) = 2.0 - _S42;
    *((&kernelContext_0)->pdata_0+(_S19 + int(19))) = - _S43;
    *((&kernelContext_0)->pdata_0+(_S19 + int(20))) = - _S44;
    *((&kernelContext_0)->pdata_0+_S21) = 2.0 - _S45;
    *((&kernelContext_0)->pdata_0+(_S19 + int(22))) = - _S46;
    *((&kernelContext_0)->pdata_0+_S22) = 2.0 - _S47;
    return;
}

