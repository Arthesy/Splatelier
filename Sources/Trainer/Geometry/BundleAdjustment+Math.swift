// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

// SE(3) left perturbation and Rodrigues rotation for BA.
extension BundleAdjustment {

    /// Left perturbation T_new = exp_se3([ω, v]) · T_old (row-major 4×4). Assumes ω, v are small.
    static func applyLeftSE3Perturbation(
        viewmat: [Float], omega: SIMD3<Float>, v: SIMD3<Float>
    ) -> [Float] {
        // Approximate the translational left Jacobian V as V ≈ I, so t_new = R_pert · t_old + v.
        let R_pert = rodriguesRotation(omega: omega)
        // R_new = R_pert · R_old
        let R00 = R_pert[0] * viewmat[0] + R_pert[1] * viewmat[4] + R_pert[2] * viewmat[8]
        let R01 = R_pert[0] * viewmat[1] + R_pert[1] * viewmat[5] + R_pert[2] * viewmat[9]
        let R02 = R_pert[0] * viewmat[2] + R_pert[1] * viewmat[6] + R_pert[2] * viewmat[10]
        let R10 = R_pert[3] * viewmat[0] + R_pert[4] * viewmat[4] + R_pert[5] * viewmat[8]
        let R11 = R_pert[3] * viewmat[1] + R_pert[4] * viewmat[5] + R_pert[5] * viewmat[9]
        let R12 = R_pert[3] * viewmat[2] + R_pert[4] * viewmat[6] + R_pert[5] * viewmat[10]
        let R20 = R_pert[6] * viewmat[0] + R_pert[7] * viewmat[4] + R_pert[8] * viewmat[8]
        let R21 = R_pert[6] * viewmat[1] + R_pert[7] * viewmat[5] + R_pert[8] * viewmat[9]
        let R22 = R_pert[6] * viewmat[2] + R_pert[7] * viewmat[6] + R_pert[8] * viewmat[10]
        let t0 = R_pert[0] * viewmat[3] + R_pert[1] * viewmat[7] + R_pert[2] * viewmat[11] + v.x
        let t1 = R_pert[3] * viewmat[3] + R_pert[4] * viewmat[7] + R_pert[5] * viewmat[11] + v.y
        let t2 = R_pert[6] * viewmat[3] + R_pert[7] * viewmat[7] + R_pert[8] * viewmat[11] + v.z
        return [
            R00, R01, R02, t0,
            R10, R11, R12, t1,
            R20, R21, R22, t2,
            0, 0, 0, 1,
        ]
    }

    /// Rodrigues: axis-angle ω → 3×3 (row-major). First-order approximation for θ < 1e-6.
    static func rodriguesRotation(omega: SIMD3<Float>) -> [Float] {
        let theta2 = omega.x * omega.x + omega.y * omega.y + omega.z * omega.z
        let theta = theta2.squareRoot()
        if theta < 1e-6 {
            // R ≈ I + [ω]_×
            return [
                1, -omega.z, omega.y,
                omega.z, 1, -omega.x,
                -omega.y, omega.x, 1,
            ]
        }
        let kx = omega.x / theta
        let ky = omega.y / theta
        let kz = omega.z / theta
        let s = sin(theta)
        let c = cos(theta)
        let oneMinusC = 1 - c
        // R = I + sin(θ) K + (1 - cos(θ)) K²
        let r00 = c + kx * kx * oneMinusC
        let r01 = kx * ky * oneMinusC - kz * s
        let r02 = kx * kz * oneMinusC + ky * s
        let r10 = ky * kx * oneMinusC + kz * s
        let r11 = c + ky * ky * oneMinusC
        let r12 = ky * kz * oneMinusC - kx * s
        let r20 = kz * kx * oneMinusC - ky * s
        let r21 = kz * ky * oneMinusC + kx * s
        let r22 = c + kz * kz * oneMinusC
        return [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    }
}
