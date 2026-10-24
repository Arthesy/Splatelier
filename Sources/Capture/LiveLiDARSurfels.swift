// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// LiDAR layer of the capture display. Each kept frame's depth samples go into their own
// `CaptureSurfelMap` with the capture-time ARKit pose (no BA during capture), colored from the
// frame's image. The capture view draws these surfels together with the MVS ones; the two layers
// are not merged.
//
// Each sample becomes a splat one depth pixel across on screen (radius = depth / focal length),
// the initialization SplaTAM (Keetha et al., CVPR 2024) uses for new Gaussians from a depth image.
//
// Shown only where it helps the capture guide: cells that at least `minViews` keyframes saw (LiDAR
// fills a whole surface from a single glance, which would hide where photos are still missing),
// and only where MVS shows nothing yet (untextured surfaces such as plain walls). Their color is
// partly desaturated so they read as a stand-in next to the MVS surfels.
//
// The snapshot of the whole map grows with the scanned area, so frames that arrive while one is
// being built are inserted together and followed by a single snapshot instead of queuing one
// snapshot per frame. The layer is display only: `cancel` drops the queued frames rather than
// finishing them.

import Foundation
import simd

public final class LiveLiDARSurfels: @unchecked Sendable {
    /// One sample in world coordinates.
    public struct Point: Sendable {
        public let position: SIMD3<Float>
        public let normal: SIMD3<Float>
        /// 0..1 RGB.
        public let color: SIMD3<Float>
        /// Surface footprint of the sample (m): its depth pixel size × the sampling stride.
        public let footprint: Float

        public init(position: SIMD3<Float>, normal: SIMD3<Float>, color: SIMD3<Float>, footprint: Float) {
            self.position = position
            self.normal = normal
            self.color = color
            self.footprint = footprint
        }
    }

    private struct Frame {
        let points: [Point]
        let cameraCenter: SIMD3<Float>
        let index: Int
    }

    private let queue = DispatchQueue(label: "live-lidar-surfels", qos: .userInitiated)
    private let lock = NSLock()
    // Guarded by `lock`
    private var queued: [Frame] = []
    private var drainScheduled = false
    private var cancelled = false
    // Only on `queue`
    private var surfelMap: CaptureSurfelMap
    private let minViews: Int
    private let mvsCoverCell: Float
    private let saturation: Float
    private let mvsSurfels: () -> [CaptureSurfelMap.Surfel]
    private let onSurfels: ([CaptureSurfelMap.Surfel]) -> Void

    /// - Parameters:
    ///   - minViews: show a cell only once this many keyframes contributed to it.
    ///   - mvsCoverCell: hide a LiDAR surfel when an MVS surfel lies in the same cell of this edge
    ///     (m).
    ///   - saturation: share of the image color's saturation kept (1: as captured, 0: grey).
    ///   - mvsSurfels: the MVS surfels currently shown.
    ///   - onSurfels: receives the whole layer after each batch of frames (on an internal queue).
    public init(surfelMap: CaptureSurfelMap = CaptureSurfelMap(), minViews: Int = 3, mvsCoverCell: Float = 0.03,
                saturation: Float = 0.5,
                mvsSurfels: @escaping () -> [CaptureSurfelMap.Surfel] = { [] },
                onSurfels: @escaping ([CaptureSurfelMap.Surfel]) -> Void) {
        self.surfelMap = surfelMap
        self.minViews = minViews
        self.mvsCoverCell = mvsCoverCell
        self.saturation = saturation
        self.mvsSurfels = mvsSurfels
        self.onSurfels = onSurfels
    }

    /// Adds one frame's samples. `frame` is the kept-frame index.
    public func submit(_ points: [Point], cameraCenter: SIMD3<Float>, frame: Int) {
        guard !points.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        guard !cancelled else { return }
        queued.append(Frame(points: points, cameraCenter: cameraCenter, index: frame))
        guard !drainScheduled else { return }
        drainScheduled = true
        queue.async { [self] in drain() }
    }

    /// Drops the queued frames and stops updating.
    public func cancel() {
        lock.lock()
        cancelled = true
        queued.removeAll()
        lock.unlock()
    }

    private func drain() {
        lock.lock()
        let frames = queued
        queued.removeAll()
        drainScheduled = false
        let stop = cancelled
        lock.unlock()
        guard !stop, !frames.isEmpty else { return }
        for f in frames {
            for p in f.points {
                surfelMap.insert(position: p.position, normal: p.normal, color: p.color,
                                 pixelSize: p.footprint, cameraCenter: f.cameraCenter, frame: f.index)
            }
        }
        let surfels = Self.shown(surfelMap.snapshot(), minViews: minViews,
                                 mvs: mvsSurfels(), coverCell: mvsCoverCell)
            .map { Self.desaturated($0, keep: saturation) }
        lock.lock()
        let stopNow = cancelled
        lock.unlock()
        if !stopNow { onSurfels(surfels) }
    }

    /// `s` with its color moved toward its luma, keeping `keep` of the saturation.
    public static func desaturated(_ s: CaptureSurfelMap.Surfel, keep: Float) -> CaptureSurfelMap.Surfel {
        let luma = simd_dot(s.color, SIMD3<Float>(0.299, 0.587, 0.114))
        return CaptureSurfelMap.Surfel(
            position: s.position, normal: s.normal, color: SIMD3(repeating: luma) + keep * (s.color - luma),
            cellMeters: s.cellMeters, level: s.level, pointCount: s.pointCount, viewCount: s.viewCount)
    }

    /// The LiDAR surfels seen by at least `minViews` keyframes and with no MVS surfel in their
    /// `coverCell` cell.
    public static func shown(
        _ lidar: [CaptureSurfelMap.Surfel], minViews: Int, mvs: [CaptureSurfelMap.Surfel], coverCell: Float
    ) -> [CaptureSurfelMap.Surfel] {
        func key(_ p: SIMD3<Float>) -> SIMD3<Int32> { SIMD3<Int32>((p / coverCell).rounded(.down)) }
        var covered = Set<SIMD3<Int32>>()
        covered.reserveCapacity(mvs.count)
        for s in mvs { covered.insert(key(s.position)) }
        return lidar.filter { $0.viewCount >= minViews && !covered.contains(key($0.position)) }
    }
}
