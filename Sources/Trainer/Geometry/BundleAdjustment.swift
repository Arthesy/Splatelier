// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Bundle Adjustment (PocketGS Operator G 3c, Sec. III-A 2). GN / LM + Huber + scale-aware gauge.
// Reference: Triggs et al. 2000, "Bundle Adjustment — A Modern Synthesis".
//
// The points are eliminated (Schur complement, Triggs sec. 4.3 / PocketGS eq. 3) by a per-point QR
// of the point Jacobian as in square-root BA (Demmel et al. 2021), which keeps the reduced camera
// system positive definite in float32; see slang/ba_point_qr.slang. The projection includes radial
// distortion (k1, k2) shared by all cameras, which BA can estimate with the poses and points
// (self-calibration, as COLMAP refines a camera's extra parameters). Pose update is a left
// perturbation T ← exp([δ_rot, δ_trans]) · T (small-step approximation, Sola 2018 eq. 95-102).

import Foundation
import simd
import Accelerate
import MLX
import MLXFast

public enum BundleAdjustment {

    /// Camera i observed point j at `pixel`. `pixel` is continuous, with pixel centers at integer +
    /// 0.5.
    public struct Observation: Sendable, Equatable {
        public let cameraIndex: Int
        public let pointIndex: Int
        public let pixel: SIMD2<Float>
        public init(cameraIndex: Int, pointIndex: Int, pixel: SIMD2<Float>) {
            self.cameraIndex = cameraIndex
            self.pointIndex = pointIndex
            self.pixel = pixel
        }
    }

    /// viewmats are row-major 4×4 world→camera; points are in world coordinates. `distortion` is
    /// the radial distortion of all cameras (observations are distorted pixels; `.none` is
    /// pinhole).
    public struct State: Sendable {
        public var viewmats: [[Float]]
        public var points: [SIMD3<Float>]
        public var distortion: RadialDistortion
        public init(viewmats: [[Float]], points: [SIMD3<Float>], distortion: RadialDistortion = .none) {
            self.viewmats = viewmats
            self.points = points
            self.distortion = distortion
        }
    }

    public struct Parameters: Sendable {
        public var iterations: Int
        public var freezeFirstCamera: Bool   // fix camera 0 (gauge fix)
        /// Scale-Aware Gauge Fixing (PocketGS Sec. III-A 2). Fixes the global-scale freedom left
        /// after fixing camera 0 by removing the baseline-axis component from camera 1's
        /// translation update (the 2 orthogonal components stay free).
        public var scaleAwareGauge: Bool
        /// Huber threshold (px). nil means L2. The paper does not disclose the value.
        public var huberThreshold: Float?
        /// Initial LM damping λ (added to the reduced system's diagonal; 0 is pure GN). When not
        /// adaptive, multiplied by `lmDecay` each iteration.
        public var lmInitialDamping: Float
        public var lmDecay: Float
        /// If the residual drops, accept and λ /= 3; if it rises, reject and λ *= 2 (simplified
        /// Triggs sec. 4.6 / Madsen-Nielsen eq. 3.36). Fixed decay tends to diverge on small
        /// systems with outliers.
        public var adaptiveLM: Bool
        /// How damping is applied. `.absolute` is the old scheme that adds λ directly to the
        /// diagonal of the reduced camera system (the right λ depends on the problem's scale).
        /// `.marquardt` scales the diagonals of H_tt and H_pp by (1+λ), making λ a dimensionless
        /// relative value. With `.marquardt` and `adaptiveLM`, steps are accepted or rejected on
        /// the actual objective (Huber cost, with a penalty for observations that end up behind the
        /// camera).
        public var dampingMode: DampingMode
        /// Weak constraint toward prior poses (`refine(priorViewmats:)`). Given an orientation σ
        /// (degrees) and position σ (m), adds 0.5·(σpx/σθ)²·|Log(R_priorᵀR)|² + 0.5·(σpx/σp)²·|O −
        /// O_prior|² to the cost (σpx = `priorPixelSigma`, same scale as the pixel residuals).
        /// Cameras with many observations follow the data; cameras with few stay near their prior
        /// pose. nil disables.
        public var priorRotationSigmaDeg: Float? = nil
        public var priorPositionSigma: Float? = nil
        public var priorPixelSigma: Float = 1.0
        /// Estimate `State.distortion` (k1, k2) together with the poses and points. Otherwise it
        /// stays fixed (still used in the projection).
        public var estimateDistortion: Bool = false

        public enum DampingMode: Sendable { case absolute, marquardt }

        public init(
            iterations: Int = 5,
            freezeFirstCamera: Bool = true,
            scaleAwareGauge: Bool = false,
            huberThreshold: Float? = nil,
            lmInitialDamping: Float = 0,
            lmDecay: Float = 0.5,
            adaptiveLM: Bool = false,
            dampingMode: DampingMode = .absolute,
            priorRotationSigmaDeg: Float? = nil,
            priorPositionSigma: Float? = nil,
            estimateDistortion: Bool = false
        ) {
            precondition(iterations > 0)
            precondition(lmInitialDamping >= 0)
            precondition(lmDecay > 0)
            self.iterations = iterations
            self.freezeFirstCamera = freezeFirstCamera
            self.scaleAwareGauge = scaleAwareGauge
            self.huberThreshold = huberThreshold
            self.lmInitialDamping = lmInitialDamping
            self.lmDecay = lmDecay
            self.adaptiveLM = adaptiveLM
            self.dampingMode = dampingMode
            self.priorRotationSigmaDeg = priorRotationSigmaDeg
            self.priorPositionSigma = priorPositionSigma
            self.estimateDistortion = estimateDistortion
        }
    }

    public struct Result: Sendable {
        public let state: State
        public let residualNorms: [Float]   // RMS residual (px): initial + after each iteration
    }

    /// Takes `parameters.iterations` GN / LM steps. Intrinsics are fixed; the distortion is
    /// estimated if `parameters.estimateDistortion`.
    public static func refine(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation],
        parameters: Parameters = Parameters(),
        priorViewmats: [[Float]]? = nil
    ) -> Result {
        precondition(state.viewmats.count == intrinsics.count,
                     "viewmats and intrinsics counts must match")
        precondition(!observations.isEmpty)

        var current = state
        var residualNorms: [Float] = []
        var lambda = parameters.lmInitialDamping
        residualNorms.append(residualNorm(state: current, intrinsics: intrinsics, observations: observations))
        let marquardt = parameters.dampingMode == .marquardt
        let prior = PosePrior(parameters: parameters, viewmats: priorViewmats)
        let layout = ObservationLayout(intrinsics: intrinsics, observations: observations, nPt: state.points.count)
        var currentCost = marquardt
            ? robustCost(state: current, intrinsics: intrinsics, observations: observations,
                         huberThreshold: parameters.huberThreshold) + (prior?.cost(current) ?? 0)
            : 0
        for _ in 0..<parameters.iterations {
            let candidate = singleGaussNewtonStep(
                state: current, intrinsics: intrinsics,
                observations: observations,
                layout: layout,
                freezeFirstCamera: parameters.freezeFirstCamera,
                scaleAwareGauge: parameters.scaleAwareGauge,
                huberThreshold: parameters.huberThreshold,
                lmDamping: lambda,
                dampingMode: parameters.dampingMode,
                estimateDistortion: parameters.estimateDistortion,
                prior: prior
            )
            if marquardt && parameters.adaptiveLM {
                // Accept/reject on the objective (Huber cost). Accept: λ/3; reject: λ×4.
                let candCost = robustCost(
                    state: candidate, intrinsics: intrinsics, observations: observations,
                    huberThreshold: parameters.huberThreshold) + (prior?.cost(candidate) ?? 0)
                if candCost < currentCost {
                    current = candidate
                    currentCost = candCost
                    residualNorms.append(residualNorm(state: current, intrinsics: intrinsics, observations: observations))
                    lambda = max(lambda / 3, 1e-9)
                } else {
                    residualNorms.append(residualNorms.last ?? 0)
                    lambda *= 4
                    if lambda > 1e6 { break }   // more damping won't lower it = converged
                }
            } else if parameters.adaptiveLM {
                // Accept/reject on RMS residual (simple Madsen-Nielsen). Accept: λ/3; reject: λ×2.
                let beforeRes = residualNorms.last ?? Float.infinity
                let afterRes = residualNorm(state: candidate, intrinsics: intrinsics, observations: observations)
                if afterRes < beforeRes {
                    current = candidate
                    residualNorms.append(afterRes)
                    lambda = max(lambda * (1.0 / 3.0), 1e-8)
                } else {
                    // Reject and leave the state unchanged
                    residualNorms.append(beforeRes)
                    lambda = lambda > 0 ? lambda * 2.0 : 1e-4
                    if lambda > 1e10 { break }
                }
            } else {
                current = candidate
                residualNorms.append(residualNorm(state: current, intrinsics: intrinsics, observations: observations))
                lambda *= parameters.lmDecay
            }
        }
        return Result(state: current, residualNorms: residualNorms)
    }

    // MARK: - Internals

    /// Below this many points the elimination runs on the CPU in double precision (same algorithm
    /// as the GPU kernels); the GPU launch overhead does not pay off for small problems.
    static let gpuMinPoints = 1000

    /// One GN / LM step. Eliminates the points per point by QR (slang/ba_point_qr.slang), solves
    /// the reduced system over the free cameras (and k1, k2 when `estimateDistortion`), and
    /// back-substitutes the points.
    static func singleGaussNewtonStep(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation],
        layout: ObservationLayout? = nil,
        freezeFirstCamera: Bool,
        scaleAwareGauge: Bool = false,
        huberThreshold: Float? = nil,
        lmDamping: Float = 0,
        dampingMode: Parameters.DampingMode = .absolute,
        estimateDistortion: Bool = false,
        prior: PosePrior? = nil
    ) -> State {
        let nCam = state.viewmats.count
        let nPt = state.points.count
        let firstFree = freezeFirstCamera ? 1 : 0
        let nFree = nCam - firstFree
        guard nFree > 0 || estimateDistortion else { return state }
        let layout = layout ?? ObservationLayout(intrinsics: intrinsics, observations: observations, nPt: nPt)
        let marquardt = dampingMode == .marquardt && lmDamping > 0
        let D = 6 * nFree + (estimateDistortion ? 2 : 0)
        let kOff = 6 * nFree

        // --- 1. Linearize and eliminate the points (Marquardt damps the point blocks too, which
        // keeps low-parallax points from blowing up) ---
        let pointDamping = marquardt ? lmDamping : 0
        var system: ReducedSystem
        var gpu: (lin: MLXArray, pdata: MLXArray)?
        var cpu: (lin: [Double], pdata: [Double])?
        if nPt >= gpuMinPoints {
            let r = reduceGPU(state: state, layout: layout, huberThreshold: huberThreshold,
                              pointDamping: pointDamping, firstFree: firstFree,
                              estimateDistortion: estimateDistortion)
            system = r.system
            gpu = (r.lin, r.pdata)
        } else {
            let lin = linearizeCPU(state: state, intrinsics: intrinsics, layout: layout, huberThreshold: huberThreshold)
            let pdata = pointQRCPU(lin: lin, layout: layout, lambda: Double(pointDamping))
            system = pointReduceCPU(lin: lin, pdata: pdata, layout: layout, firstFree: firstFree,
                                    estimateDistortion: estimateDistortion)
            cpu = (lin, pdata)
        }

        // --- 2. Prior-pose constraint and damping on the reduced diagonal ---
        var priorDiag = [Float](repeating: 0, count: nCam * 6)
        prior?.accumulate(state: state, lhs: &system.lhs, rhs: &system.rhs, dimension: D,
                          firstFree: firstFree, diagonal: &priorDiag)
        for camI in firstFree..<nCam {
            let fI = camI - firstFree
            for r in 0..<6 {
                let i = (fI * 6 + r) * (D + 1)
                if marquardt {
                    // Marquardt: scale the undamped diagonal by (1+λ); a small floor keeps
                    // unobserved components non-singular.
                    system.lhs[i] += lmDamping * (system.camDiag[camI * 6 + r] + priorDiag[camI * 6 + r]) + 1e-9
                } else if lmDamping > 0 {
                    system.lhs[i] += lmDamping
                }
            }
        }
        if estimateDistortion {
            for a in 0..<2 {
                let i = (kOff + a) * (D + 1)
                if marquardt {
                    system.lhs[i] += lmDamping * system.kDiag[a] + 1e-9
                } else if lmDamping > 0 {
                    system.lhs[i] += lmDamping
                }
            }
        }

        // --- 3. Solve the reduced system (D = 6 × free cameras (+2), ~3600 for 600 frames) ---
        let solution = D > 0 ? solveReducedSystem(lhs: system.lhs, rhs: system.rhs, dimension: D) : []

        var deltaCam = [Float](repeating: 0, count: nCam * 6)
        for camI in firstFree..<nCam {
            let fI = camI - firstFree
            for r in 0..<6 { deltaCam[camI * 6 + r] = solution[fI * 6 + r] }
        }
        let deltaK = estimateDistortion ? SIMD2(solution[kOff], solution[kOff + 1]) : SIMD2<Float>(0, 0)

        // --- 4. Scale-aware gauge: remove the baseline-axis component from camera 1's translation
        // update --- The baseline is in world coordinates but δ_cam's translation (v of the left
        // perturbation) is in camera 1 coordinates, so rotate the baseline by camera 1's rotation
        // before the dot product. (The points are back-substituted with the projected update.)
        if scaleAwareGauge && freezeFirstCamera && nCam >= 2 {
            let c0 = FrameSubsampler.cameraCenter(viewmat: state.viewmats[0])
            let c1 = FrameSubsampler.cameraCenter(viewmat: state.viewmats[1])
            let bvec = SIMD3<Float>(c1.x - c0.x, c1.y - c0.y, c1.z - c0.z)
            let blen = (bvec.x * bvec.x + bvec.y * bvec.y + bvec.z * bvec.z).squareRoot()
            if blen > 1e-6 {
                let bhatWorld = SIMD3<Float>(bvec.x / blen, bvec.y / blen, bvec.z / blen)
                let vm1 = state.viewmats[1]
                // R1 · bhatWorld (a rotation, so still unit length)
                let bhat = SIMD3<Float>(
                    vm1[0] * bhatWorld.x + vm1[1] * bhatWorld.y + vm1[2] * bhatWorld.z,
                    vm1[4] * bhatWorld.x + vm1[5] * bhatWorld.y + vm1[6] * bhatWorld.z,
                    vm1[8] * bhatWorld.x + vm1[9] * bhatWorld.y + vm1[10] * bhatWorld.z
                )
                let dx = deltaCam[1 * 6 + 3]
                let dy = deltaCam[1 * 6 + 4]
                let dz = deltaCam[1 * 6 + 5]
                let proj = dx * bhat.x + dy * bhat.y + dz * bhat.z
                deltaCam[1 * 6 + 3] = dx - proj * bhat.x
                deltaCam[1 * 6 + 4] = dy - proj * bhat.y
                deltaCam[1 * 6 + 5] = dz - proj * bhat.z
            }
        }

        // --- 5. Back-substitute the points ---
        let deltaPt: [Float]
        if let gpu {
            deltaPt = backSubstituteGPU(lin: gpu.lin, pdata: gpu.pdata, layout: layout,
                                        deltaCam: deltaCam, deltaK: deltaK)
        } else if let cpu {
            deltaPt = backSubstituteCPU(lin: cpu.lin, pdata: cpu.pdata, layout: layout,
                                        deltaCam: deltaCam, deltaK: deltaK)
        } else {
            deltaPt = [Float](repeating: 0, count: nPt * 3)
        }

        // --- 6. Update state ---
        var newViewmats = state.viewmats
        for camI in 0..<nCam {
            let omega = SIMD3<Float>(
                deltaCam[camI * 6 + 0], deltaCam[camI * 6 + 1], deltaCam[camI * 6 + 2]
            )
            let v = SIMD3<Float>(
                deltaCam[camI * 6 + 3], deltaCam[camI * 6 + 4], deltaCam[camI * 6 + 5]
            )
            newViewmats[camI] = applyLeftSE3Perturbation(
                viewmat: state.viewmats[camI], omega: omega, v: v
            )
        }
        var newPoints = state.points
        for j in 0..<nPt {
            newPoints[j].x += deltaPt[j * 3 + 0]
            newPoints[j].y += deltaPt[j * 3 + 1]
            newPoints[j].z += deltaPt[j * 3 + 2]
        }
        var distortion = state.distortion
        distortion.k1 += deltaK.x
        distortion.k2 += deltaK.y
        return State(viewmats: newViewmats, points: newPoints, distortion: distortion)
    }

    /// Solves the reduced camera system `lhs · x = rhs` (`lhs` row-major `dimension²`) with
    /// Accelerate's LAPACK. The Schur complement of the damped H is symmetric positive definite, so
    /// Cholesky (sposv) solves it, ~10× faster than MLX's CPU LU at 3600 unknowns. If float32
    /// accumulation made it lose definiteness, falls back to LU (sgesv). The matrix is symmetric,
    /// so LAPACK's column-major view of the row-major buffer is the same matrix.
    static func solveReducedSystem(lhs: [Float], rhs: [Float], dimension: Int) -> [Float] {
        var n = __LAPACK_int(dimension), lda = n, ldb = n, nrhs = __LAPACK_int(1)
        var info = __LAPACK_int(0)
        var uplo = Int8(UInt8(ascii: "U"))
        var a = lhs, x = rhs
        sposv_(&uplo, &n, &nrhs, &a, &lda, &x, &ldb, &info)
        if info == 0 { return x }
        a = lhs
        x = rhs
        var ipiv = [__LAPACK_int](repeating: 0, count: dimension)
        sgesv_(&n, &nrhs, &a, &lda, &ipiv, &x, &ldb, &info)
        return x
    }

    // MARK: - Cost

    /// Objective cost: sum of per-observation Huber ρ(|r|) (0.5·|r|² if `huberThreshold` is nil).
    /// Observations that end up behind the camera (z ≤ 1e-3) are charged the cost of a
    /// `behindPenaltyPx` residual (excluding them would make steps that push points behind the
    /// camera look like cost drops).
    static func robustCost(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation],
        huberThreshold: Float?,
        behindPenaltyPx: Float = 100
    ) -> Float {
        guard !observations.isEmpty else { return 0 }
        let (residualsFlat, zCam) = computeResidualsGPU(
            state: state, intrinsics: intrinsics, observations: observations
        )
        let n = observations.count
        let res2 = residualsFlat.reshaped([n, 2])
        let mag = MLX.sqrt((res2 * res2).sum(axis: 1))
        func rho(_ r: MLXArray) -> MLXArray {
            guard let k = huberThreshold else { return 0.5 * r * r }
            return MLX.which(r .<= k, 0.5 * r * r, k * (r - 0.5 * k))
        }
        let valid = zCam .> Float(1e-3)
        let penalty = rho(MLXArray(behindPenaltyPx))
        let per = MLX.which(valid, rho(mag), penalty)
        let total = per.sum()
        eval(total)
        return total.item(Float.self)
    }

    /// Prior-pose constraint term. With the left perturbation (R' = exp(ω)R, t' = exp(ω)t + v):
    ///   orientation: e_R = Log(R_pᵀ R), ∂e_R/∂ω ≈ Rᵀ  →  H_ωω += w_R·I,  b_ω += −w_R·R·e_R
    ///   position:    e_O = O − O_p,     ∂e_O/∂v = −Rᵀ  →  H_vv += w_O·I,  b_v += w_O·R·e_O
    /// (b follows the existing kernels' convention b = −Σ w Jᵀ r).
    struct PosePrior {
        let wRot: Double
        let wPos: Double
        let priorR: [simd_double3x3]
        let priorO: [SIMD3<Double>]

        init?(parameters p: Parameters, viewmats: [[Float]]?) {
            guard let vms = viewmats, p.priorRotationSigmaDeg != nil || p.priorPositionSigma != nil else { return nil }
            let spx = Double(p.priorPixelSigma)
            wRot = p.priorRotationSigmaDeg.map { let s = Double($0) * .pi / 180; return (spx / s) * (spx / s) } ?? 0
            wPos = p.priorPositionSigma.map { let s = Double($0); return (spx / s) * (spx / s) } ?? 0
            priorR = vms.map { PosePrior.rot($0) }
            priorO = vms.map { PosePrior.center($0) }
        }

        static func rot(_ v: [Float]) -> simd_double3x3 {
            simd_double3x3(rows: [
                SIMD3(Double(v[0]), Double(v[1]), Double(v[2])),
                SIMD3(Double(v[4]), Double(v[5]), Double(v[6])),
                SIMD3(Double(v[8]), Double(v[9]), Double(v[10])),
            ])
        }
        static func center(_ v: [Float]) -> SIMD3<Double> {
            -(rot(v).transpose * SIMD3(Double(v[3]), Double(v[7]), Double(v[11])))
        }
        static func logSO3(_ R: simd_double3x3) -> SIMD3<Double> {
            let c = max(-1, min(1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2))
            let th = acos(c)
            let v = SIMD3(R[1][2] - R[2][1], R[2][0] - R[0][2], R[0][1] - R[1][0])
            return th < 1e-9 ? 0.5 * v : v * (th / (2 * sin(th)))
        }

        func cost(_ state: State) -> Float {
            var c = 0.0
            for (i, v) in state.viewmats.enumerated() {
                let R = PosePrior.rot(v)
                c += 0.5 * wRot * simd_length_squared(PosePrior.logSO3(priorR[i].transpose * R))
                c += 0.5 * wPos * simd_length_squared(PosePrior.center(v) - priorO[i])
            }
            return Float(c)
        }

        /// Adds the prior to the reduced system's free-camera blocks (`firstFree` fixed cameras get
        /// no rows) and its diagonal to `diagonal` (nCam × 6, for Marquardt damping).
        func accumulate(state: State, lhs: inout [Float], rhs: inout [Float], dimension D: Int,
                        firstFree: Int, diagonal: inout [Float]) {
            for (i, v) in state.viewmats.enumerated() where i >= firstFree {
                let f = (i - firstFree) * 6
                let R = PosePrior.rot(v)
                let eR = PosePrior.logSO3(priorR[i].transpose * R)
                let eO = PosePrior.center(v) - priorO[i]
                let bw = -(R * eR) * wRot
                let bv = (R * eO) * wPos
                for k in 0..<3 {
                    lhs[(f + k) * (D + 1)] += Float(wRot)
                    lhs[(f + 3 + k) * (D + 1)] += Float(wPos)
                    diagonal[i * 6 + k] += Float(wRot)
                    diagonal[i * 6 + 3 + k] += Float(wPos)
                    rhs[f + k] += Float(bw[k])
                    rhs[f + 3 + k] += Float(bv[k])
                }
            }
        }
    }

    /// RMS residual (px) over observations with z_cam > 1e-3.
    static func residualNorm(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation]
    ) -> Float {
        guard !observations.isEmpty else { return 0 }
        let (residualsFlat, zCam) = computeResidualsGPU(
            state: state, intrinsics: intrinsics, observations: observations
        )
        let valid = (zCam .> Float(1e-3)).asType(.float32)
        let n = observations.count
        let res2 = residualsFlat.reshaped([n, 2])
        let sqMag = (res2 * res2).sum(axis: 1)   // [N]
        let sumSq = (sqMag * valid).sum()
        let count = valid.sum()
        eval(sumSq, count)
        let cs = count.item(Float.self)
        if cs <= 0 { return 0 }
        return (sumSq.item(Float.self) / cs).squareRoot()
    }
}
