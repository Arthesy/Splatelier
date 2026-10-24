// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

// GPU kernel wrappers for BA: per-observation linearization, point elimination by per-point QR
// (reduction into the camera / distortion system and back-substitution), and residuals. See
// slang/ba_point_qr.slang for the math.
extension BundleAdjustment {

    /// Per-observation residual kernel (shared by `residualNorm` and `filterReprojection`).
    static let residualKernel = SlangCompiledKernel.required("ba_residual")
    static let linearizeKernel = SlangCompiledKernel.required("ba_linearize")
    static let pointQRKernel = SlangCompiledKernel.required("ba_point_qr")
    /// Every point adds into the shared reduced system, so the outputs must be real atomics.
    static let pointReduceKernel = SlangCompiledKernel.required("ba_point_reduce", atomicOutputs: true)
    static let pointUpdateKernel = SlangCompiledKernel.required("ba_point_update")

    /// Floats per observation in the linearization buffer and per point in the QR buffer.
    static let linStride = 24
    static let pointDataStride = 24

    /// Observation arrays and per-point observation lists, fixed for the observations of one
    /// `refine` call (built once rather than every iteration).
    struct ObservationLayout {
        let nCam: Int
        let nPt: Int
        let nObs: Int
        let camIdx: [Int32]
        let ptIdx: [Int32]
        /// Observation indices sorted by point, with each point's (start, count) into it.
        let sortedObs: [Int32]
        let pointStarts: [Int32]
        let pointCounts: [Int32]
        let pixels: [Float]
        // GPU copies of the above
        let camIdxArr: MLXArray
        let ptIdxArr: MLXArray
        let pixelsArr: MLXArray
        let sortedObsArr: MLXArray
        let pointStartsArr: MLXArray
        let pointCountsArr: MLXArray
        let intrinsicsArr: MLXArray

        init(intrinsics: [Intrinsics], observations: [Observation], nPt: Int) {
            nCam = intrinsics.count
            self.nPt = nPt
            nObs = observations.count
            var cam = [Int32](repeating: 0, count: nObs)
            var pt = [Int32](repeating: 0, count: nObs)
            var px = [Float](repeating: 0, count: nObs * 2)
            var counts = [Int32](repeating: 0, count: nPt)
            for (i, o) in observations.enumerated() {
                cam[i] = Int32(o.cameraIndex)
                pt[i] = Int32(o.pointIndex)
                px[i * 2] = o.pixel.x
                px[i * 2 + 1] = o.pixel.y
                counts[o.pointIndex] += 1
            }
            // Counting sort by point (stable: each point's observations stay in input order)
            var starts = [Int32](repeating: 0, count: nPt)
            var running: Int32 = 0
            for j in 0..<nPt {
                starts[j] = running
                running += counts[j]
            }
            var fill = starts
            var sorted = [Int32](repeating: 0, count: nObs)
            for i in 0..<nObs {
                let j = Int(pt[i])
                sorted[Int(fill[j])] = Int32(i)
                fill[j] += 1
            }
            var intr = [Float](repeating: 0, count: nCam * 4)
            for (i, k) in intrinsics.enumerated() {
                intr[i * 4 + 0] = k.fx
                intr[i * 4 + 1] = k.fy
                intr[i * 4 + 2] = k.cx
                intr[i * 4 + 3] = k.cy
            }
            camIdx = cam
            ptIdx = pt
            sortedObs = sorted
            pointStarts = starts
            pointCounts = counts
            pixels = px
            camIdxArr = MLXArray(cam, [max(nObs, 1)])
            ptIdxArr = MLXArray(pt, [max(nObs, 1)])
            pixelsArr = MLXArray(px, [max(nObs * 2, 1)])
            sortedObsArr = MLXArray(sorted, [max(nObs, 1)])
            pointStartsArr = MLXArray(starts, [max(nPt, 1)])
            pointCountsArr = MLXArray(counts, [max(nPt, 1)])
            intrinsicsArr = MLXArray(intr, [nCam * 4])
        }
    }

    /// The reduced system over free cameras (and k), with the undamped diagonals that Marquardt
    /// damping scales.
    struct ReducedSystem {
        var lhs: [Float]        // D×D row-major
        var rhs: [Float]        // D
        var camDiag: [Float]    // nCam × 6, diag(Σ J̃cᵀJ̃c)
        var kDiag: [Float]      // 2, diag(Σ J̃kᵀJ̃k)
    }

    static func flatViewmats(_ state: State) -> [Float] {
        var out = [Float](repeating: 0, count: state.viewmats.count * 16)
        for (i, vm) in state.viewmats.enumerated() {
            precondition(vm.count == 16, "viewmat must be 16 floats")
            for k in 0..<16 { out[i * 16 + k] = vm[k] }
        }
        return out
    }

    static func flatPoints(_ state: State) -> [Float] {
        var out = [Float](repeating: 0, count: state.points.count * 3)
        for (i, p) in state.points.enumerated() {
            out[i * 3 + 0] = p.x
            out[i * 3 + 1] = p.y
            out[i * 3 + 2] = p.z
        }
        return out
    }

    /// Linearizes all observations and eliminates the points on the GPU. Returns the reduced
    /// system on the host and keeps the linearization and per-point QR data on the GPU for
    /// `backSubstituteGPU`.
    static func reduceGPU(
        state: State, layout: ObservationLayout, huberThreshold: Float?,
        pointDamping: Float, firstFree: Int, estimateDistortion: Bool
    ) -> (system: ReducedSystem, lin: MLXArray, pdata: MLXArray) {
        let nCam = layout.nCam, nPt = layout.nPt, nObs = layout.nObs
        let nFree = nCam - firstFree
        let D = 6 * nFree + (estimateDistortion ? 2 : 0)

        let lin = linearizeKernel(
            [MLXArray(flatViewmats(state), [nCam * 16]), layout.intrinsicsArr,
             MLXArray(flatPoints(state), [nPt * 3]), layout.camIdxArr, layout.ptIdxArr, layout.pixelsArr,
             MLXArray([huberThreshold ?? -1, state.distortion.k1, state.distortion.k2], [3]),
             MLXArray([Int32(nObs)], [1])],
            grid: (nObs, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[nObs * linStride]], outputDTypes: [.float32]
        )[0]
        let pdata = pointQRKernel(
            [lin, layout.sortedObsArr, layout.pointStartsArr, layout.pointCountsArr,
             MLXArray([pointDamping], [1]), MLXArray([Int32(nPt)], [1])],
            grid: (nPt, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[nPt * pointDataStride]], outputDTypes: [.float32]
        )[0]
        let reduced = pointReduceKernel(
            [lin, layout.sortedObsArr, layout.pointStartsArr, layout.pointCountsArr, layout.camIdxArr, pdata,
             MLXArray([Int32(nPt), Int32(D), Int32(firstFree), Int32(estimateDistortion ? 6 * nFree : -1)], [4])],
            grid: (nPt, 1, 1), threadGroup: (64, 1, 1),
            outputShapes: [[D * D], [D], [nCam * 6], [2]],
            outputDTypes: [.float32, .float32, .float32, .float32],
            initValue: 0.0
        )
        eval(reduced)
        let system = ReducedSystem(
            lhs: reduced[0].asArray(Float.self), rhs: reduced[1].asArray(Float.self),
            camDiag: reduced[2].asArray(Float.self), kDiag: reduced[3].asArray(Float.self))
        return (system, lin, pdata)
    }

    /// δp = −S·R⁻¹·(q + Σ C_i·δc_i + C_k·δk) on the GPU (`deltaCam` has 0 for fixed cameras).
    static func backSubstituteGPU(
        lin: MLXArray, pdata: MLXArray, layout: ObservationLayout,
        deltaCam: [Float], deltaK: SIMD2<Float>
    ) -> [Float] {
        let nPt = layout.nPt
        let out = pointUpdateKernel(
            [lin, layout.sortedObsArr, layout.pointStartsArr, layout.pointCountsArr, layout.camIdxArr, pdata,
             MLXArray(deltaCam, [layout.nCam * 6]), MLXArray([deltaK.x, deltaK.y], [2]),
             MLXArray([Int32(nPt)], [1])],
            grid: (nPt, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[nPt * 3]], outputDTypes: [.float32]
        )[0]
        eval(out)
        return out.asArray(Float.self)
    }

    /// Computes per-observation residuals (du, dv) and z_cam on the GPU and returns them as
    /// MLXArrays. The projection includes `state.distortion`.
    static func computeResidualsGPU(
        state: State,
        intrinsics: [Intrinsics],
        observations: [Observation]
    ) -> (residuals: MLXArray, zCam: MLXArray) {
        let nCam = state.viewmats.count
        let nPt = state.points.count
        let nObs = observations.count
        precondition(nObs > 0, "computeResidualsGPU: observations must be non-empty")
        precondition(intrinsics.count == nCam, "intrinsics / viewmats count mismatch")

        var intrFlat = [Float](repeating: 0, count: nCam * 4)
        for (i, intr) in intrinsics.enumerated() {
            intrFlat[i * 4 + 0] = intr.fx
            intrFlat[i * 4 + 1] = intr.fy
            intrFlat[i * 4 + 2] = intr.cx
            intrFlat[i * 4 + 3] = intr.cy
        }
        var camIdx = [Int32](repeating: 0, count: nObs)
        var ptIdx = [Int32](repeating: 0, count: nObs)
        var pixels = [Float](repeating: 0, count: nObs * 2)
        for (i, obs) in observations.enumerated() {
            camIdx[i] = Int32(obs.cameraIndex)
            ptIdx[i] = Int32(obs.pointIndex)
            pixels[i * 2 + 0] = obs.pixel.x
            pixels[i * 2 + 1] = obs.pixel.y
        }

        let outputs = residualKernel(
            [MLXArray(flatViewmats(state), [nCam * 16]), MLXArray(intrFlat, [nCam * 4]),
             MLXArray(flatPoints(state), [nPt * 3]), MLXArray(camIdx, [nObs]), MLXArray(ptIdx, [nObs]),
             MLXArray(pixels, [nObs * 2]), MLXArray([state.distortion.k1, state.distortion.k2], [2]),
             MLXArray([Int32(nObs)], [1])],
            grid: (nObs, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [[nObs * 2], [nObs]],
            outputDTypes: [.float32, .float32]
        )
        return (residuals: outputs[0], zCam: outputs[1])
    }
}
