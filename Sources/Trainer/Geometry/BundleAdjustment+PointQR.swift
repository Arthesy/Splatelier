// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// CPU (double precision) version of the GPU point elimination in slang/ba_linearize.slang and
// slang/ba_point_{qr,reduce,update}.slang, with the same data layout and the same steps. BA uses it
// for small problems; tests use it as the reference for the GPU kernels.

import Foundation
import simd

extension BundleAdjustment {

    /// Same as `ba_linearize`: [J̃c (2×6) | J̃p (2×3) | J̃k (2×2) | r̃ (2)] per observation.
    static func linearizeCPU(
        state: State, intrinsics: [Intrinsics], layout: ObservationLayout, huberThreshold: Float?
    ) -> [Double] {
        let k1 = Double(state.distortion.k1), k2 = Double(state.distortion.k2)
        var lin = [Double](repeating: 0, count: layout.nObs * linStride)
        for i in 0..<layout.nObs {
            let b = i * linStride
            let vm = state.viewmats[Int(layout.camIdx[i])].map(Double.init)
            let K = intrinsics[Int(layout.camIdx[i])]
            let fx = Double(K.fx), fy = Double(K.fy), cx = Double(K.cx), cy = Double(K.cy)
            let P = SIMD3<Double>(state.points[Int(layout.ptIdx[i])])
            let x = vm[0] * P.x + vm[1] * P.y + vm[2] * P.z + vm[3]
            let y = vm[4] * P.x + vm[5] * P.y + vm[6] * P.z + vm[7]
            let z = vm[8] * P.x + vm[9] * P.y + vm[10] * P.z + vm[11]
            guard z > 1e-3 else { continue }
            let nx = x / z, ny = y / z
            let r2 = nx * nx + ny * ny
            let f = 1 + k1 * r2 + k2 * r2 * r2
            let df = k1 + 2 * k2 * r2
            let rU = fx * f * nx + cx - Double(layout.pixels[i * 2])
            let rV = fy * f * ny + cy - Double(layout.pixels[i * 2 + 1])
            let a00 = fx * (f + 2 * nx * nx * df), a01 = fx * (2 * nx * ny * df)
            let a10 = fy * (2 * nx * ny * df), a11 = fy * (f + 2 * ny * ny * df)
            let d = [[a00 / z, a01 / z, -(a00 * nx + a01 * ny) / z],
                     [a10 / z, a11 / z, -(a10 * nx + a11 * ny) / z]]
            var w = 1.0
            if let h = huberThreshold, h > 0 {
                let absR = (rU * rU + rV * rV).squareRoot()
                if absR > Double(h) { w = Double(h) / max(absR, 1e-9) }
            }
            let sw = w.squareRoot()
            for row in 0..<2 {
                let (d0, d1, d2) = (d[row][0], d[row][1], d[row][2])
                let c = b + row * 6
                lin[c + 0] = sw * (-d1 * z + d2 * y)
                lin[c + 1] = sw * (d0 * z - d2 * x)
                lin[c + 2] = sw * (-d0 * y + d1 * x)
                lin[c + 3] = sw * d0
                lin[c + 4] = sw * d1
                lin[c + 5] = sw * d2
                for col in 0..<3 {
                    lin[b + 12 + row * 3 + col] = sw * (d0 * vm[col] + d1 * vm[4 + col] + d2 * vm[8 + col])
                }
            }
            lin[b + 18] = sw * fx * nx * r2
            lin[b + 19] = sw * fx * nx * r2 * r2
            lin[b + 20] = sw * fy * ny * r2
            lin[b + 21] = sw * fy * ny * r2 * r2
            lin[b + 22] = sw * rU
            lin[b + 23] = sw * rV
        }
        return lin
    }

    /// Same as `ba_point_qr`: [S·R⁻¹ (3×3) | q (3) | C_k (3×2) | W (6)] per point.
    static func pointQRCPU(lin: [Double], layout: ObservationLayout, lambda: Double) -> [Double] {
        var pdata = [Double](repeating: 0, count: layout.nPt * pointDataStride)
        for j in 0..<layout.nPt {
            let o = j * pointDataStride
            pdata[o + 18] = 2; pdata[o + 21] = 2; pdata[o + 23] = 2
            let obs = (0..<Int(layout.pointCounts[j])).map { Int(layout.sortedObs[Int(layout.pointStarts[j]) + $0]) }
            var d = SIMD3<Double>(repeating: 0)
            for i in obs {
                let b = i * linStride + 12
                for c in 0..<3 { d[c] += lin[b + c] * lin[b + c] + lin[b + 3 + c] * lin[b + 3 + c] }
            }
            guard d.min() > 1e-20 else { continue }
            let s = 1 / d.squareRoot()
            func gram(_ m: simd_double3x3) -> simd_double3x3 {
                var g = simd_double3x3()
                for i in obs {
                    let b = i * linStride + 12
                    for row in 0..<2 {
                        let a = SIMD3(lin[b + row * 3], lin[b + row * 3 + 1], lin[b + row * 3 + 2]) * s
                        let q = m.transpose * a   // row vector a·m
                        g += simd_double3x3(rows: [q * q.x, q * q.y, q * q.z])
                    }
                }
                return g + lambda * (m.transpose * m)
            }
            func chol(_ g: simd_double3x3, _ minPivot: Double) -> simd_double3x3? {
                // Upper-triangular R with g = RᵀR (simd matrices are column-major: m[col][row])
                let p0 = g[0][0]; guard p0 > minPivot else { return nil }
                let r00 = p0.squareRoot(), r01 = g[1][0] / r00, r02 = g[2][0] / r00
                let p1 = g[1][1] - r01 * r01; guard p1 > minPivot else { return nil }
                let r11 = p1.squareRoot(), r12 = (g[2][1] - r01 * r02) / r11
                let p2 = g[2][2] - r02 * r02 - r12 * r12; guard p2 > minPivot else { return nil }
                return simd_double3x3(rows: [SIMD3(r00, r01, r02), SIMD3(0, r11, r12), SIMD3(0, 0, p2.squareRoot())])
            }
            var m = matrix_identity_double3x3
            let g0 = gram(m)
            let shift = 11 * (3 * Double(2 * obs.count + 3) + 12) * 1.11e-16 * (g0[0][0] + g0[1][1] + g0[2][2])
            guard let r1 = chol(g0 + shift * matrix_identity_double3x3, 0) else { continue }
            m = m * r1.inverse
            var ok = true
            for _ in 0..<2 {
                guard let r = chol(gram(m), 1e-12) else { ok = false; break }
                m = m * r.inverse
            }
            guard ok else { continue }
            let sm = simd_double3x3(diagonal: s) * m
            let gq = gram(m)
            var q = SIMD3<Double>(repeating: 0)
            var ck0 = SIMD3<Double>(repeating: 0), ck1 = SIMD3<Double>(repeating: 0)
            for i in obs {
                let b = i * linStride
                for row in 0..<2 {
                    let p = SIMD3(lin[b + 12 + row * 3], lin[b + 13 + row * 3], lin[b + 14 + row * 3])
                    let qr = sm.transpose * p
                    q += qr * lin[b + 22 + row]
                    ck0 += qr * lin[b + 18 + row * 2]
                    ck1 += qr * lin[b + 19 + row * 2]
                }
            }
            for row in 0..<3 { for c in 0..<3 { pdata[o + row * 3 + c] = sm[c][row] } }
            for c in 0..<3 {
                pdata[o + 9 + c] = q[c]
                pdata[o + 12 + c * 2] = ck0[c]
                pdata[o + 13 + c * 2] = ck1[c]
            }
            let w = 2 * matrix_identity_double3x3 - gq
            pdata[o + 18] = w[0][0]; pdata[o + 19] = w[1][0]; pdata[o + 20] = w[2][0]
            pdata[o + 21] = w[1][1]; pdata[o + 22] = w[2][1]; pdata[o + 23] = w[2][2]
        }
        return pdata
    }

    /// Same as `ba_point_reduce`.
    static func pointReduceCPU(
        lin: [Double], pdata: [Double], layout: ObservationLayout, firstFree: Int, estimateDistortion: Bool
    ) -> ReducedSystem {
        let nFree = layout.nCam - firstFree
        let D = 6 * nFree + (estimateDistortion ? 2 : 0)
        let kOff = 6 * nFree
        var lhs = [Double](repeating: 0, count: D * D)
        var rhs = [Double](repeating: 0, count: D)
        var camDiag = [Double](repeating: 0, count: layout.nCam * 6)
        var kDiag = [Double](repeating: 0, count: 2)
        for j in 0..<layout.nPt {
            let o = j * pointDataStride
            let sm = (0..<9).map { pdata[o + $0] }
            let q = SIMD3(pdata[o + 9], pdata[o + 10], pdata[o + 11])
            let w = simd_double3x3(rows: [SIMD3(pdata[o + 18], pdata[o + 19], pdata[o + 20]),
                                          SIMD3(pdata[o + 19], pdata[o + 21], pdata[o + 22]),
                                          SIMD3(pdata[o + 20], pdata[o + 22], pdata[o + 23])])
            let ck = [SIMD3(pdata[o + 12], pdata[o + 14], pdata[o + 16]), SIMD3(pdata[o + 13], pdata[o + 15], pdata[o + 17])]
            let wq = w * q
            let wck = ck.map { w * $0 }
            let obs = (0..<Int(layout.pointCounts[j])).map { Int(layout.sortedObs[Int(layout.pointStarts[j]) + $0]) }
            // C_i column y as a 3-vector
            func cCols(_ i: Int) -> [SIMD3<Double>] {
                let b = i * linStride
                return (0..<6).map { y in
                    let h = SIMD3((0..<3).map { x in lin[b + 12 + x] * lin[b + y] + lin[b + 15 + x] * lin[b + 6 + y] })
                    return SIMD3((0..<3).map { x in sm[0 * 3 + x] * h[0] + sm[1 * 3 + x] * h[1] + sm[2 * 3 + x] * h[2] })
                }
            }
            if estimateDistortion {
                for a in 0..<2 {
                    for c in 0..<2 { lhs[(kOff + a) * D + kOff + c] -= simd_dot(ck[a], wck[c]) }
                    rhs[kOff + a] += simd_dot(ck[a], wq)
                }
            }
            let cs = obs.map(cCols)
            for (ai, i) in obs.enumerated() {
                let b = i * linStride
                let cam = Int(layout.camIdx[i])
                let jk = [[lin[b + 18], lin[b + 19]], [lin[b + 20], lin[b + 21]]]
                let r = [lin[b + 22], lin[b + 23]]
                for y in 0..<6 { camDiag[cam * 6 + y] += lin[b + y] * lin[b + y] + lin[b + 6 + y] * lin[b + 6 + y] }
                for a in 0..<2 {
                    kDiag[a] += jk[0][a] * jk[0][a] + jk[1][a] * jk[1][a]
                    if estimateDistortion {
                        for c in 0..<2 { lhs[(kOff + a) * D + kOff + c] += jk[0][a] * jk[0][c] + jk[1][a] * jk[1][c] }
                        rhs[kOff + a] -= jk[0][a] * r[0] + jk[1][a] * r[1]
                    }
                }
                guard cam >= firstFree else { continue }
                let fI = cam - firstFree
                let cI = cs[ai]
                for y in 0..<6 {
                    rhs[fI * 6 + y] += -(lin[b + y] * r[0] + lin[b + 6 + y] * r[1]) + simd_dot(cI[y], wq)
                    if estimateDistortion {
                        for a in 0..<2 {
                            let v = jk[0][a] * lin[b + y] + jk[1][a] * lin[b + 6 + y] - simd_dot(wck[a], cI[y])
                            lhs[(kOff + a) * D + fI * 6 + y] += v
                            lhs[(fI * 6 + y) * D + kOff + a] += v
                        }
                    }
                }
                for (al, l) in obs.enumerated() {
                    let camL = Int(layout.camIdx[l])
                    guard camL >= firstFree else { continue }
                    let fL = camL - firstFree
                    let cL = cs[al]
                    for x in 0..<6 {
                        let wcx = w * cI[x]
                        for y in 0..<6 {
                            var v = -simd_dot(wcx, cL[y])
                            if al == ai { v += lin[b + x] * lin[b + y] + lin[b + 6 + x] * lin[b + 6 + y] }
                            lhs[(fI * 6 + x) * D + fL * 6 + y] += v
                        }
                    }
                }
            }
        }
        return ReducedSystem(lhs: lhs.map(Float.init), rhs: rhs.map(Float.init),
                             camDiag: camDiag.map(Float.init), kDiag: kDiag.map(Float.init))
    }

    /// Same as `ba_point_update`.
    static func backSubstituteCPU(
        lin: [Double], pdata: [Double], layout: ObservationLayout, deltaCam: [Float], deltaK: SIMD2<Float>
    ) -> [Float] {
        var out = [Float](repeating: 0, count: layout.nPt * 3)
        for j in 0..<layout.nPt {
            let o = j * pointDataStride
            var v = SIMD3<Double>(repeating: 0)
            for a in 0..<Int(layout.pointCounts[j]) {
                let i = Int(layout.sortedObs[Int(layout.pointStarts[j]) + a])
                let b = i * linStride
                let dc = Int(layout.camIdx[i]) * 6
                var e0 = 0.0, e1 = 0.0
                for y in 0..<6 {
                    e0 += lin[b + y] * Double(deltaCam[dc + y])
                    e1 += lin[b + 6 + y] * Double(deltaCam[dc + y])
                }
                let p0 = SIMD3<Double>(lin[b + 12], lin[b + 13], lin[b + 14])
                let p1 = SIMD3<Double>(lin[b + 15], lin[b + 16], lin[b + 17])
                v += p0 * e0
                v += p1 * e1
            }
            let sm = simd_double3x3(rows: [SIMD3(pdata[o], pdata[o + 1], pdata[o + 2]),
                                           SIMD3(pdata[o + 3], pdata[o + 4], pdata[o + 5]),
                                           SIMD3(pdata[o + 6], pdata[o + 7], pdata[o + 8])])
            let dk = SIMD2<Double>(deltaK)
            let q = SIMD3<Double>(pdata[o + 9], pdata[o + 10], pdata[o + 11])
            let ck0 = SIMD3<Double>(pdata[o + 12], pdata[o + 14], pdata[o + 16])
            let ck1 = SIMD3<Double>(pdata[o + 13], pdata[o + 15], pdata[o + 17])
            let h = q + sm.transpose * v + ck0 * dk.x + ck1 * dk.y
            let dp = -(sm * h)
            out[j * 3] = Float(dp.x); out[j * 3 + 1] = Float(dp.y); out[j * 3 + 2] = Float(dp.z)
        }
        return out
    }
}
