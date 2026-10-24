#include <metal_stdlib>
#include <metal_math>
#include <metal_texture>
using namespace metal;
struct KernelContext_0
{
    int device* lin_dims_0;
    int device* obs_cam_idx_0;
    int device* obs_pt_idx_0;
    float device* intrinsics_buf_0;
    float device* lin_params_0;
    float device* points_0;
    float device* viewmats_0;
    float device* lin_0;
    float device* obs_pixels_0;
};

[[kernel]] void ba_linearize(uint3 tid_0 [[thread_position_in_grid]], int device* lin_dims_1 [[buffer(7)]], int device* obs_cam_idx_1 [[buffer(3)]], int device* obs_pt_idx_1 [[buffer(4)]], float device* intrinsics_buf_1 [[buffer(1)]], float device* lin_params_1 [[buffer(6)]], float device* points_1 [[buffer(2)]], float device* viewmats_1 [[buffer(0)]], float device* lin_1 [[buffer(8)]], float device* obs_pixels_1 [[buffer(5)]])
{
    thread KernelContext_0 kernelContext_0;
    (&kernelContext_0)->lin_dims_0 = lin_dims_1;
    (&kernelContext_0)->obs_cam_idx_0 = obs_cam_idx_1;
    (&kernelContext_0)->obs_pt_idx_0 = obs_pt_idx_1;
    (&kernelContext_0)->intrinsics_buf_0 = intrinsics_buf_1;
    (&kernelContext_0)->lin_params_0 = lin_params_1;
    (&kernelContext_0)->points_0 = points_1;
    (&kernelContext_0)->viewmats_0 = viewmats_1;
    (&kernelContext_0)->lin_0 = lin_1;
    (&kernelContext_0)->obs_pixels_0 = obs_pixels_1;
    int i_0 = int(tid_0.x);
    if(i_0 >= lin_dims_1[int(0)])
    {
        return;
    }
    int base_0 = i_0 * int(24);
    int cam_0 = (&kernelContext_0)->obs_cam_idx_0[i_0];
    int vm_0 = cam_0 * int(16);
    int _S1 = cam_0 * int(4);
    float fx_0 = (&kernelContext_0)->intrinsics_buf_0[_S1];
    float fy_0 = (&kernelContext_0)->intrinsics_buf_0[_S1 + int(1)];
    float cx_0 = (&kernelContext_0)->intrinsics_buf_0[_S1 + int(2)];
    float cy_0 = (&kernelContext_0)->intrinsics_buf_0[_S1 + int(3)];
    float k1_0 = (&kernelContext_0)->lin_params_0[int(1)];
    float k2_0 = (&kernelContext_0)->lin_params_0[int(2)];
    int _S2 = (&kernelContext_0)->obs_pt_idx_0[i_0] * int(3);
    float Px_0 = (&kernelContext_0)->points_0[_S2];
    float Py_0 = (&kernelContext_0)->points_0[_S2 + int(1)];
    float Pz_0 = (&kernelContext_0)->points_0[_S2 + int(2)];
    float R00_0 = (&kernelContext_0)->viewmats_0[vm_0];
    float R01_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(1)];
    float R02_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(2)];
    float R10_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(4)];
    float R11_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(5)];
    float R12_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(6)];
    float R20_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(8)];
    float R21_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(9)];
    float R22_0 = (&kernelContext_0)->viewmats_0[vm_0 + int(10)];
    float pcx_0 = R00_0 * Px_0 + R01_0 * Py_0 + R02_0 * Pz_0 + (&kernelContext_0)->viewmats_0[vm_0 + int(3)];
    float pcy_0 = R10_0 * Px_0 + R11_0 * Py_0 + R12_0 * Pz_0 + (&kernelContext_0)->viewmats_0[vm_0 + int(7)];
    float pcz_0 = R20_0 * Px_0 + R21_0 * Py_0 + R22_0 * Pz_0 + (&kernelContext_0)->viewmats_0[vm_0 + int(11)];
    if(pcz_0 <= 0.00100000004749745)
    {
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
            *((&kernelContext_0)->lin_0+(base_0 + k_0)) = 0.0;
            k_0 = k_0 + int(1);
        }
        return;
    }
    float invZ_0 = 1.0 / pcz_0;
    float nx_0 = pcx_0 * invZ_0;
    float ny_0 = pcy_0 * invZ_0;
    float r2_0 = nx_0 * nx_0 + ny_0 * ny_0;
    float f_0 = 1.0 + k1_0 * r2_0 + k2_0 * r2_0 * r2_0;
    float df_0 = k1_0 + 2.0 * k2_0 * r2_0;
    int _S3 = i_0 * int(2);
    float rU_0 = fx_0 * f_0 * nx_0 + cx_0 - (&kernelContext_0)->obs_pixels_0[_S3];
    float rV_0 = fy_0 * f_0 * ny_0 + cy_0 - (&kernelContext_0)->obs_pixels_0[_S3 + int(1)];
    float _S4 = 2.0 * nx_0;
    float a00_0 = fx_0 * (f_0 + _S4 * nx_0 * df_0);
    float _S5 = _S4 * ny_0 * df_0;
    float a01_0 = fx_0 * _S5;
    float a10_0 = fy_0 * _S5;
    float a11_0 = fy_0 * (f_0 + 2.0 * ny_0 * ny_0 * df_0);
    float d00_0 = a00_0 * invZ_0;
    float d01_0 = a01_0 * invZ_0;
    float d02_0 = - (a00_0 * nx_0 + a01_0 * ny_0) * invZ_0;
    float d10_0 = a10_0 * invZ_0;
    float d11_0 = a11_0 * invZ_0;
    float d12_0 = - (a10_0 * nx_0 + a11_0 * ny_0) * invZ_0;
    float huberK_0 = (&kernelContext_0)->lin_params_0[int(0)];
    float w_0;
    if(huberK_0 > 0.0)
    {
        float absR_0 = sqrt(rU_0 * rU_0 + rV_0 * rV_0);
        if(absR_0 > huberK_0)
        {
            w_0 = huberK_0 / max(absR_0, 9.99999971718068537e-10);
        }
        else
        {
            w_0 = 1.0;
        }
    }
    else
    {
        w_0 = 1.0;
    }
    float sw_0 = sqrt(w_0);
    *((&kernelContext_0)->lin_0+base_0) = sw_0 * (- d01_0 * pcz_0 + d02_0 * pcy_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(1))) = sw_0 * (d00_0 * pcz_0 - d02_0 * pcx_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(2))) = sw_0 * (- d00_0 * pcy_0 + d01_0 * pcx_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(3))) = sw_0 * d00_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(4))) = sw_0 * d01_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(5))) = sw_0 * d02_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(6))) = sw_0 * (- d11_0 * pcz_0 + d12_0 * pcy_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(7))) = sw_0 * (d10_0 * pcz_0 - d12_0 * pcx_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(8))) = sw_0 * (- d10_0 * pcy_0 + d11_0 * pcx_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(9))) = sw_0 * d10_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(10))) = sw_0 * d11_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(11))) = sw_0 * d12_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(12))) = sw_0 * (d00_0 * R00_0 + d01_0 * R10_0 + d02_0 * R20_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(13))) = sw_0 * (d00_0 * R01_0 + d01_0 * R11_0 + d02_0 * R21_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(14))) = sw_0 * (d00_0 * R02_0 + d01_0 * R12_0 + d02_0 * R22_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(15))) = sw_0 * (d10_0 * R00_0 + d11_0 * R10_0 + d12_0 * R20_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(16))) = sw_0 * (d10_0 * R01_0 + d11_0 * R11_0 + d12_0 * R21_0);
    *((&kernelContext_0)->lin_0+(base_0 + int(17))) = sw_0 * (d10_0 * R02_0 + d11_0 * R12_0 + d12_0 * R22_0);
    float _S6 = sw_0 * fx_0 * nx_0 * r2_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(18))) = _S6;
    *((&kernelContext_0)->lin_0+(base_0 + int(19))) = _S6 * r2_0;
    float _S7 = sw_0 * fy_0 * ny_0 * r2_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(20))) = _S7;
    *((&kernelContext_0)->lin_0+(base_0 + int(21))) = _S7 * r2_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(22))) = sw_0 * rU_0;
    *((&kernelContext_0)->lin_0+(base_0 + int(23))) = sw_0 * rV_0;
    return;
}

