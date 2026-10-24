// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Unprojects ARFrame.sceneDepth (LiDAR) into camera-space samples. depthMap is DepthFloat32 with
// distance along the optical axis (m, not ray length). confidence is ARConfidenceLevel raw. Samples
// are in OpenCV camera coordinates (+X right, +Y down, +Z forward), the same convention as the
// viewmats, so they can be moved to world with the post-BA pose (ARKit camera = (x, −y, −z) of
// these, following the WWDC20 ARKit 4 sample).

#if canImport(ARKit) && os(iOS)

import Foundation
import CoreVideo
import simd
import ARKit

public enum DepthUnprojection {

    public struct Params: Sendable {
        public let stride: Int
        public let minConfidenceRaw: UInt8        // 0=low / 1=medium / 2=high
        public let minDepth: Float                // m. Lower trust bound for close-range LiDAR
        public let maxDepth: Float                // m. Nominal LiDAR range

        public init(
            stride: Int = 8,
            minConfidenceRaw: UInt8 = 1,
            minDepth: Float = 0.2,
            maxDepth: Float = 5.0
        ) {
            precondition(stride > 0)
            precondition(minDepth > 0)
            precondition(maxDepth > minDepth)
            self.stride = stride
            self.minConfidenceRaw = minConfidenceRaw
            self.minDepth = minDepth
            self.maxDepth = maxDepth
        }
    }

    /// One depth sample.
    public struct Sample: Sendable {
        /// OpenCV camera coordinates (m).
        public let cameraPoint: SIMD3<Float>
        /// Unit normal in OpenCV camera coordinates, facing the camera. From the neighboring samples
        /// (`stride` px away); the view direction where a neighbor is missing or across a depth
        /// edge.
        public let normal: SIMD3<Float>
        /// Pixel center in normalized image coordinates (0..1). The depth map has the field of view
        /// of `capturedImage`, so this also indexes the color image.
        public let uv: SIMD2<Float>
        /// Size of one depth pixel on the surface facing the camera (depth / fx, m).
        public let pixelSize: Float
        /// ARConfidenceLevel raw value of the pixel (0=low / 1=medium / 2=high).
        public let confidenceLevel: UInt8
        /// Largest depth difference to the 8 adjacent depth pixels (m). Large at depth edges, where
        /// a ToF pixel mixes the near and far surface (mixed pixel).
        public let depthSpread: Float
        /// Column and row in the depth map.
        public let x: Int
        public let y: Int
    }

    /// Turns `ARFrame.sceneDepth` into samples every `params.stride` px.
    public static func unproject(
        depthMap: CVPixelBuffer,
        confidenceMap: CVPixelBuffer?,
        cameraIntrinsics: matrix_float3x3,
        imageResolution: CGSize,
        params: Params = .init()
    ) -> [Sample] {
        precondition(
            CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32,
            "ARKit depthMap format must be kCVPixelFormatType_DepthFloat32"
        )
        let depthW = CVPixelBufferGetWidth(depthMap)
        let depthH = CVPixelBufferGetHeight(depthMap)

        // Intrinsics are relative to the capturedImage resolution, so rescale them to the depthMap
        // resolution.
        let K = PoseConversion.scaledIntrinsics(
            fx: cameraIntrinsics[0, 0], fy: cameraIntrinsics[1, 1],
            cx: cameraIntrinsics[2, 0], cy: cameraIntrinsics[2, 1],
            originalWidth: Int(imageResolution.width),
            originalHeight: Int(imageResolution.height),
            scaledWidth: depthW, scaledHeight: depthH
        )

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        guard let depthBaseRaw = CVPixelBufferGetBaseAddress(depthMap) else { return [] }
        let depthBase = depthBaseRaw.assumingMemoryBound(to: Float32.self)
        let depthRowFloats = CVPixelBufferGetBytesPerRow(depthMap) / MemoryLayout<Float32>.stride

        var confBase: UnsafePointer<UInt8>? = nil
        var confRowBytes: Int = 0
        if let conf = confidenceMap {
            CVPixelBufferLockBaseAddress(conf, .readOnly)
            if let confBaseRaw = CVPixelBufferGetBaseAddress(conf) {
                confBase = UnsafePointer(confBaseRaw.assumingMemoryBound(to: UInt8.self))
                confRowBytes = CVPixelBufferGetBytesPerRow(conf)
            }
        }
        defer {
            if let conf = confidenceMap {
                CVPixelBufferUnlockBaseAddress(conf, .readOnly)
            }
        }

        func level(_ x: Int, _ y: Int) -> UInt8 { confBase.map { $0[y * confRowBytes + x] } ?? 2 }
        /// Camera point of a valid pixel, nil if out of range or below the confidence threshold.
        func point(_ x: Int, _ y: Int) -> SIMD3<Float>? {
            guard x < depthW, y < depthH else { return nil }
            let z = depthBase[y * depthRowFloats + x]
            guard z >= params.minDepth, z <= params.maxDepth else { return nil }
            if level(x, y) < params.minConfidenceRaw { return nil }
            // pixel center +0.5
            return SIMD3<Float>((Float(x) + 0.5 - K.cx) / K.fx * z, (Float(y) + 0.5 - K.cy) / K.fy * z, z)
        }

        let stride = Swift.max(1, params.stride)
        var out: [Sample] = []
        out.reserveCapacity(((depthW + stride - 1) / stride) * ((depthH + stride - 1) / stride))
        for y in Swift.stride(from: 0, to: depthH, by: stride) {
            for x in Swift.stride(from: 0, to: depthW, by: stride) {
                guard let p = point(x, y) else { continue }
                let toCamera = -simd_normalize(p)
                var normal = toCamera
                // Neighbors more than 10% off in depth lie across an edge, not on this surface.
                if let px = point(x + stride, y), let py = point(x, y + stride),
                   abs(px.z - p.z) < 0.1 * p.z, abs(py.z - p.z) < 0.1 * p.z {
                    let n = simd_cross(px - p, py - p)
                    if simd_length(n) > 1e-12 {
                        normal = simd_normalize(n)
                        if simd_dot(normal, toCamera) < 0 { normal = -normal }
                    }
                }
                var spread: Float = 0
                for ny in Swift.max(0, y - 1)...Swift.min(depthH - 1, y + 1) {
                    for nx in Swift.max(0, x - 1)...Swift.min(depthW - 1, x + 1) {
                        let nz = depthBase[ny * depthRowFloats + nx]
                        if nz > 0 { spread = Swift.max(spread, abs(nz - p.z)) }
                    }
                }
                out.append(Sample(
                    cameraPoint: p, normal: normal,
                    uv: SIMD2((Float(x) + 0.5) / Float(depthW), (Float(y) + 0.5) / Float(depthH)),
                    pixelSize: p.z / K.fx, confidenceLevel: level(x, y), depthSpread: spread, x: x, y: y))
            }
        }
        return out
    }
}

#endif
