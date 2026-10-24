// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Point elimination by per-point QR: the GPU kernels against the double-precision CPU version, and
// positive definiteness of the reduced system with low-parallax points.

import Accelerate
import Foundation
import simd
import Testing
@testable import Trainer

@Suite struct BundleAdjustmentPointQRTests {

    struct LCG {
        var state: UInt64
        mutating func next() -> Float {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(state >> 40) / Float(1 << 24)
        }
        mutating func uniform(_ lo: Float, _ hi: Float) -> Float { lo + (hi - lo) * next() }
    }

    static let k: Intrinsics = (fx: 900, fy: 900, cx: 540, cy: 405)

    /// 12 cameras on an arc looking at a box of points, with `farFraction` of the points placed
    /// 60 m away (almost no parallax), perturbed poses and points, and distorted observations.
    static func scene(points nPt: Int, farFraction: Float, distortion: RadialDistortion)
        -> (state: BundleAdjustment.State, observations: [BundleAdjustment.Observation]) {
        var rng = LCG(state: 3)
        let viewmats: [[Float]] = (0..<12).map { i in
            let a = Float(i - 6) * 0.04
            let R = simd_float3x3(simd_quatf(angle: -a, axis: SIMD3(0, 1, 0)))
            let center = SIMD3<Float>(sin(a) * 2, 0.02 * Float(i % 3), 2 - cos(a) * 2)
            let t = -(R * center)
            return [R[0][0], R[1][0], R[2][0], t.x, R[0][1], R[1][1], R[2][1], t.y,
                    R[0][2], R[1][2], R[2][2], t.z, 0, 0, 0, 1]
        }
        let truthPoints: [SIMD3<Float>] = (0..<nPt).map { j in
            if Float(j) < farFraction * Float(nPt) {
                return SIMD3(rng.uniform(-20, 20), rng.uniform(-15, 15), 60)
            }
            return SIMD3(rng.uniform(-0.8, 0.8), rng.uniform(-0.6, 0.6), rng.uniform(1.5, 2.5))
        }
        var obs: [BundleAdjustment.Observation] = []
        for (c, vm) in viewmats.enumerated() {
            for (j, p) in truthPoints.enumerated() where rng.next() < 0.5 {
                let x = vm[0] * p.x + vm[1] * p.y + vm[2] * p.z + vm[3]
                let y = vm[4] * p.x + vm[5] * p.y + vm[6] * p.z + vm[7]
                let z = vm[8] * p.x + vm[9] * p.y + vm[10] * p.z + vm[11]
                guard z > 0.1 else { continue }
                let pd = distortion.distort(SIMD2(x / z, y / z))
                let uv = SIMD2(k.fx * pd.x + k.cx + rng.uniform(-0.5, 0.5), k.fy * pd.y + k.cy + rng.uniform(-0.5, 0.5))
                guard uv.x >= 0, uv.y >= 0, uv.x < 1080, uv.y < 810 else { continue }
                obs.append(BundleAdjustment.Observation(cameraIndex: c, pointIndex: j, pixel: uv))
            }
        }
        var vms = viewmats
        for i in 1..<vms.count {
            vms[i] = BundleAdjustment.applyLeftSE3Perturbation(
                viewmat: vms[i], omega: SIMD3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) * 0.002,
                v: SIMD3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) * 0.003)
        }
        let pts = truthPoints.map { $0 + SIMD3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)) * 0.005 }
        return (BundleAdjustment.State(viewmats: vms, points: pts), obs)
    }

    /// The GPU reduction and back-substitution match the double-precision CPU version.
    @Test func gpuMatchesCPU() {
        let s = Self.scene(points: 3000, farFraction: 0.1, distortion: RadialDistortion(k1: 0.02, k2: -0.03))
        let intr = Array(repeating: Self.k, count: s.state.viewmats.count)
        let layout = BundleAdjustment.ObservationLayout(intrinsics: intr, observations: s.observations, nPt: s.state.points.count)
        let gpu = BundleAdjustment.reduceGPU(state: s.state, layout: layout, huberThreshold: 2, pointDamping: 1e-3,
                                             firstFree: 1, estimateDistortion: true)
        let lin = BundleAdjustment.linearizeCPU(state: s.state, intrinsics: intr, layout: layout, huberThreshold: 2)
        let pdata = BundleAdjustment.pointQRCPU(lin: lin, layout: layout, lambda: 1e-3)
        let cpu = BundleAdjustment.pointReduceCPU(lin: lin, pdata: pdata, layout: layout, firstFree: 1, estimateDistortion: true)

        func relErr(_ a: [Float], _ b: [Float]) -> Float {
            var num: Float = 0, den: Float = 0
            for i in a.indices { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i] }
            return (num / max(den, 1e-30)).squareRoot()
        }
        #expect(relErr(gpu.system.lhs, cpu.lhs) < 1e-4)
        #expect(relErr(gpu.system.rhs, cpu.rhs) < 1e-3)
        #expect(relErr(gpu.system.camDiag, cpu.camDiag) < 1e-5)
        #expect(relErr(gpu.system.kDiag, cpu.kDiag) < 1e-4)

        let D = cpu.rhs.count
        let delta = BundleAdjustment.solveReducedSystem(lhs: cpu.lhs, rhs: cpu.rhs, dimension: D)
        var deltaCam = [Float](repeating: 0, count: intr.count * 6)
        for i in 0..<(D - 2) { deltaCam[6 + i] = delta[i] }
        let dk = SIMD2(delta[D - 2], delta[D - 1])
        let dpGPU = BundleAdjustment.backSubstituteGPU(lin: gpu.lin, pdata: gpu.pdata, layout: layout, deltaCam: deltaCam, deltaK: dk)
        let dpCPU = BundleAdjustment.backSubstituteCPU(lin: lin, pdata: pdata, layout: layout, deltaCam: deltaCam, deltaK: dk)
        #expect(relErr(dpGPU, dpCPU) < 1e-3)
    }

    /// With a third of the points at almost no parallax and nearly no damping, the float32 GPU
    /// reduction stays positive definite (Cholesky succeeds).
    @Test func reducedSystemStaysPositiveDefinite() {
        let s = Self.scene(points: 4000, farFraction: 0.33, distortion: RadialDistortion(k1: 0.02, k2: -0.03))
        let intr = Array(repeating: Self.k, count: s.state.viewmats.count)
        let layout = BundleAdjustment.ObservationLayout(intrinsics: intr, observations: s.observations, nPt: s.state.points.count)
        let gpu = BundleAdjustment.reduceGPU(state: s.state, layout: layout, huberThreshold: 2, pointDamping: 1e-9,
                                             firstFree: 1, estimateDistortion: true)
        var a = gpu.system.lhs
        let D = gpu.system.rhs.count
        // Prior-strength floor on the camera diagonal, as BA's weak pose prior adds
        for i in 0..<(D - 2) { a[i * (D + 1)] += 1e2 }
        var n = __LAPACK_int(D), lda = n, info = __LAPACK_int(0)
        var uplo = Int8(UInt8(ascii: "U"))
        spotrf_(&uplo, &n, &a, &lda, &info)
        #expect(info == 0)
    }

    /// Full BA with distortion on the GPU path recovers k and drives the residual to the noise.
    @Test func gpuBARecoversDistortion() {
        let truth = RadialDistortion(k1: 0.02, k2: -0.03)
        let s = Self.scene(points: 3000, farFraction: 0.05, distortion: truth)
        var p = BundleAdjustment.Parameters(
            iterations: 15, freezeFirstCamera: true, scaleAwareGauge: true, huberThreshold: 2,
            lmInitialDamping: 1e-3, adaptiveLM: true, dampingMode: .marquardt)
        p.estimateDistortion = true
        let result = BundleAdjustment.refine(
            state: s.state, intrinsics: Array(repeating: Self.k, count: s.state.viewmats.count),
            observations: s.observations, parameters: p)
        #expect(abs(result.state.distortion.k1 - truth.k1) < 0.003, "k1 \(result.state.distortion.k1)")
        #expect(abs(result.state.distortion.k2 - truth.k2) < 0.01, "k2 \(result.state.distortion.k2)")
        #expect(result.residualNorms.last! < 0.5)
    }
}
