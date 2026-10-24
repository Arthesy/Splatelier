// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// How the seed's sources (MVS dense points and LiDAR samples) share the initial Gaussians. With
// `.confidence`, each cell keeps the source with the higher mean confidence and drops the other,
// the region-wise choice of ToF / stereo fusion (Marin et al., ECCV 2016: ToF is reliable on flat,
// untextured surfaces and fails at depth edges; stereo the other way round). The cap on the initial
// count then thins the cells holding the most points.
//
// Cells are sized to the pixel footprint at which a point was seen (depth / focal length, times
// `cellPixels`), in power-of-two steps as in CaptureSurfelMap. Both sources' point spacing grows
// with depth, so a cell holds about the same number of samples at any distance, and the per-cell
// cap leaves about the same number of points per image pixel instead of per square metre (which
// would thin the close-up surfaces, where the photos have the most detail, first).

import Foundation
import simd

extension CaptureTraining {

    public enum SeedAllocation: Sendable, Equatable {
        /// Fuse every source and thin evenly to the cap.
        case even
        /// Per cell (`cellPixels` image pixels across at the depth the point was seen) keep the
        /// source with the higher mean confidence (LiDAR's scaled by `lidarWeight`), then thin the
        /// fullest cells to the cap. The default cell is half the spacing of the recorded LiDAR
        /// samples (every 8th of 256 depth pixels across an image about 1080 wide), the best of 16 /
        /// 32 px on three captures.
        case confidence(cellPixels: Float = 16, lidarWeight: Float = 1)
    }

    /// Points of one source on the host.
    struct HostCloud {
        var positions: [SIMD3<Float>] = []
        var colors: [SIMD3<Float>] = []
        var confidences: [Float] = []
        /// Depth (m) at which the frames saw each point.
        var depths: [Float] = []

        var count: Int { positions.count }

        func subset(_ keep: [Bool]) -> HostCloud {
            var out = HostCloud()
            for i in 0..<count where keep[i] {
                out.positions.append(positions[i])
                out.colors.append(colors[i])
                out.confidences.append(confidences[i])
                out.depths.append(depths[i])
            }
            return out
        }

        func appending(_ other: HostCloud) -> HostCloud {
            HostCloud(positions: positions + other.positions, colors: colors + other.colors,
                      confidences: confidences + other.confidences, depths: depths + other.depths)
        }
    }

    /// Footprint-sized cells: a point seen at depth d goes into the power-of-two cell nearest
    /// above d / fx × cellPixels.
    struct AdaptiveCells {
        /// 1 / fx (median over the frames).
        let pixelAngle: Float
        let cellPixels: Float
        /// Edge of the finest cell (m).
        static let finest: Float = 0.002
        static let levelCount = 10
        /// Xor-ed into the keys so a table filled in another table's order does not share its hash
        /// order (see SeedFusion).
        let salt = Int64.random(in: .min ... .max)

        init(frames: [CapturedFrame], cellPixels: Float) {
            let fx = frames.map(\.intrinsics.fx).sorted()
            pixelAngle = fx.isEmpty ? 1.0 / 800 : 1 / fx[fx.count / 2]
            self.cellPixels = cellPixels
        }

        func key(_ p: SIMD3<Float>, depth: Float) -> Int64 {
            let footprint = Swift.max(depth, 1e-3) * pixelAngle * cellPixels
            let level = Swift.min(Self.levelCount - 1, Swift.max(0, Int(ceil(log2(footprint / Self.finest)))))
            let c = SIMD3<Int64>((p / (Self.finest * Float(1 << level))).rounded(.down))
            let bias: Int64 = 1 << 18
            return ((Int64(level) << 57) | ((c.x + bias) << 38) | ((c.y + bias) << 19) | (c.z + bias)) ^ salt
        }
    }

    /// Per cell, which source's points to keep. A cell with one source keeps it; a cell with both
    /// keeps the one with the higher mean confidence.
    static func selectSources(
        mvs: HostCloud, lidar: HostCloud, cells: AdaptiveCells, lidarWeight: Float, log: (String) -> Void
    ) -> (mvsKeep: [Bool], lidarKeep: [Bool]) {
        let mvsKeys = (0..<mvs.count).map { cells.key(mvs.positions[$0], depth: mvs.depths[$0]) }
        let lidarKeys = (0..<lidar.count).map { cells.key(lidar.positions[$0], depth: lidar.depths[$0]) }
        var table: [Int64: (mvsSum: Float, mvsCount: Int, lidarSum: Float, lidarCount: Int)] = [:]
        table.reserveCapacity((mvs.count + lidar.count) / 8)
        for (i, k) in mvsKeys.enumerated() {
            var c = table[k] ?? (0, 0, 0, 0)
            c.mvsSum += mvs.confidences[i]
            c.mvsCount += 1
            table[k] = c
        }
        for (i, k) in lidarKeys.enumerated() {
            var c = table[k] ?? (0, 0, 0, 0)
            c.lidarSum += lidarWeight * lidar.confidences[i]
            c.lidarCount += 1
            table[k] = c
        }
        var lidarWins = Set<Int64>()
        var mvsOnly = 0, lidarOnly = 0, bothMVS = 0, bothLiDAR = 0
        for (k, c) in table {
            if c.lidarCount == 0 { mvsOnly += 1; continue }
            if c.mvsCount == 0 { lidarOnly += 1; lidarWins.insert(k); continue }
            if c.lidarSum / Float(c.lidarCount) > c.mvsSum / Float(c.mvsCount) {
                bothLiDAR += 1
                lidarWins.insert(k)
            } else {
                bothMVS += 1
            }
        }
        let mvsKeep = mvsKeys.map { !lidarWins.contains($0) }
        let lidarKeep = lidarKeys.map { lidarWins.contains($0) }
        log(String(format:
            "[seed alloc] cells of %.0f px: MVS only %d, LiDAR only %d, both → MVS %d, both → LiDAR %d; "
            + "kept MVS %d/%d, LiDAR %d/%d (LiDAR weight %.2f)",
            cells.cellPixels, mvsOnly, lidarOnly, bothMVS, bothLiDAR,
            mvsKeep.lazy.filter { $0 }.count, mvs.count, lidarKeep.lazy.filter { $0 }.count, lidar.count,
            lidarWeight))
        return (mvsKeep, lidarKeep)
    }

    /// Keep mask of at most `budget` points, taking at most K points from every cell (the largest
    /// K that fits), evenly spread within a cell. The fullest cells are thinned first; cells with
    /// fewer than K points keep all of them.
    static func capPerCell(_ cloud: HostCloud, cells: AdaptiveCells, budget: Int) -> [Bool] {
        guard cloud.count > budget, budget > 0 else { return Array(repeating: true, count: cloud.count) }
        var members: [Int64: [Int]] = [:]
        for i in 0..<cloud.count {
            members[cells.key(cloud.positions[i], depth: cloud.depths[i]), default: []].append(i)
        }
        let counts = members.values.map(\.count)
        func kept(_ k: Int) -> Int { counts.reduce(0) { $0 + Swift.min($1, k) } }
        var lo = 1, hi = counts.max() ?? 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if kept(mid) <= budget { lo = mid } else { hi = mid - 1 }
        }
        var keep = [Bool](repeating: false, count: cloud.count)
        for list in members.values {
            if list.count <= lo {
                for i in list { keep[i] = true }
            } else {
                for j in 0..<lo { keep[list[j * list.count / lo]] = true }
            }
        }
        return keep
    }

    /// Mean of the samples in each voxel of edge `voxelSize`, weighted by confidence (position and
    /// color); the voxel's confidence and depth are the means of its samples'.
    static func voxelAverage(
        _ samples: [CapturedScene.WorldLiDARSample], voxelSize: Float
    ) -> [CapturedScene.WorldLiDARSample] {
        precondition(voxelSize > 0)
        var cells: [SIMD3<Int32>: (sum: SIMD3<Float>, color: SIMD3<Float>, weight: Float, confidence: Float,
                                   depth: Float, count: Int)] = [:]
        cells.reserveCapacity(samples.count / 4)
        for s in samples {
            let key = SIMD3<Int32>((s.position / voxelSize).rounded(.down))
            let w = Swift.max(s.confidence, 1e-4)
            let c = cells[key] ?? (.zero, .zero, 0, 0, 0, 0)
            cells[key] = (c.sum + s.position * w, c.color + s.color * w, c.weight + w,
                          c.confidence + s.confidence, c.depth + s.depth, c.count + 1)
        }
        return cells.values.map {
            ($0.sum / $0.weight, $0.confidence / Float($0.count), $0.depth / Float($0.count), $0.color / $0.weight)
        }
    }
}
