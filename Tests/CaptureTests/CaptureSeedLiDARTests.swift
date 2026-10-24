// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import MLX
import simd
import Testing
@testable import Capture

@Test
func lidarSamplesAreAveragedPerVoxel() {
    // Samples of one surface scattered within a voxel become their confidence-weighted mean; a
    // sample in another voxel stays separate.
    let a = SIMD3<Float>(0.001, 0.002, 0.003)
    let b = SIMD3<Float>(0.009, 0.008, 0.007)
    let far = SIMD3<Float>(0.5, -0.5, 1.25)
    let grey = SIMD3<Float>(repeating: 0.5)
    let out = CaptureTraining.voxelAverage([(a, 1, 1, grey), (b, 1, 3, grey), (far, 0.5, 2, grey)], voxelSize: 0.01)
    #expect(out.count == 2)
    #expect(out.contains { simd_distance($0.position, (a + b) / 2) < 1e-6 && $0.confidence == 1 && $0.depth == 2 })
    #expect(out.contains { simd_distance($0.position, far) < 1e-6 && $0.confidence == 0.5 })
    #expect(CaptureTraining.voxelAverage([], voxelSize: 0.01).isEmpty)
}

@Test
func lidarConfidenceFallsAtEdgesAndRange() {
    let flat = LiDARSample(cameraPoint: SIMD3(0, 0, 1), confidenceLevel: 2, depthSpread: 0)
    #expect(abs(flat.confidence() - 0.8) < 1e-6)             // range term 1 − 1/5
    let medium = LiDARSample(cameraPoint: SIMD3(0, 0, 1), confidenceLevel: 1, depthSpread: 0)
    #expect(abs(medium.confidence() - 0.4) < 1e-6)
    // Spread of 10% of the depth (a depth edge): mixed pixel, no confidence.
    let edge = LiDARSample(cameraPoint: SIMD3(0, 0, 1), confidenceLevel: 2, depthSpread: 0.1)
    #expect(edge.confidence() == 0)
}

/// Cells of `cellPixels` at fx = 1000 (no frames → the 800 fallback is not used here).
private func cells(_ cellPixels: Float) -> CaptureTraining.AdaptiveCells {
    let frame = CapturedFrame(
        image: MLXArray([UInt8](repeating: 0, count: 12), [2, 2, 3]),
        viewmat: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        intrinsics: (fx: 1000, fy: 1000, cx: 1, cy: 1), width: 2, height: 2, timestamp: 0)
    return CaptureTraining.AdaptiveCells(frames: [frame], cellPixels: cellPixels)
}

private func cloud(_ p: [SIMD3<Float>], conf: [Float], depth: Float) -> CaptureTraining.HostCloud {
    CaptureTraining.HostCloud(positions: p, colors: Array(repeating: .zero, count: p.count),
                              confidences: conf, depths: Array(repeating: depth, count: p.count))
}

@Test
func seedSourcesAreChosenPerCell() {
    // Seen from 1 m, a 20 px cell is about 2 cm (the 32 mm power-of-two step). Cell A has both
    // sources and confident MVS → MVS; B both with confident LiDAR → LiDAR; C only LiDAR; D only
    // MVS.
    let a = SIMD3<Float>(0.005, 0.005, 0.005), b = SIMD3<Float>(0.105, 0.005, 0.005)
    let c = SIMD3<Float>(0.205, 0.005, 0.005), d = SIMD3<Float>(0.305, 0.005, 0.005)
    let mvs = cloud([a, a, b, d], conf: [0.9, 0.9, 0.2, 0.3], depth: 1)
    let lidar = cloud([a, b, c], conf: [0.5, 0.8, 0.1], depth: 1)
    let keep = CaptureTraining.selectSources(mvs: mvs, lidar: lidar, cells: cells(20), lidarWeight: 1, log: { _ in })
    #expect(keep.mvsKeep == [true, true, false, true])
    #expect(keep.lidarKeep == [false, true, true])
}

@Test
func cellsGrowWithTheDepthAPointWasSeenAt() {
    // Two points 1 cm apart share a cell when seen from 4 m (cells of ~13 cm) but not from 0.25 m
    // (cells of ~8 mm).
    let c = cells(32)
    let p = SIMD3<Float>(0.001, 0.001, 0.001), q = SIMD3<Float>(0.011, 0.001, 0.001)
    #expect(c.key(p, depth: 4) == c.key(q, depth: 4))
    #expect(c.key(p, depth: 0.25) != c.key(q, depth: 0.25))
}

@Test
func capPerCellThinsFullCellsFirst() {
    // 10 points in one cell and 1 in another, budget 4: the full cell gives up its points and the
    // single point stays.
    let dense = (0..<10).map { SIMD3<Float>(0.001 * Float($0), 0, 0) }
    let lone = SIMD3<Float>(1, 1, 1)
    let keep = CaptureTraining.capPerCell(cloud(dense + [lone], conf: Array(repeating: 1, count: 11), depth: 1),
                                          cells: cells(20), budget: 4)
    #expect(keep.filter { $0 }.count == 4)
    #expect(keep[10])
}

@Test
func lidarSamplesTakeTheColorOfTheirPixel() {
    // A 2×2 image with a distinct color per pixel; a sample straight ahead of pixel (1, 0) takes
    // that pixel's color.
    let pixels: [UInt8] = [255, 0, 0,  0, 255, 0,  0, 0, 255,  255, 255, 255]
    let sample = LiDARSample(cameraPoint: SIMD3(0.5, -0.5, 1), confidenceLevel: 2, depthSpread: 0)
    let frame = CapturedFrame(
        image: MLXArray(pixels, [2, 2, 3]),
        viewmat: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        intrinsics: (fx: 1, fy: 1, cx: 1, cy: 1), width: 2, height: 2, timestamp: 0,
        lidarSamples: [sample])
    let scene = CapturedScene(frames: [frame], sparsePoints: [])
    #expect(scene.lidarWorldSamples(colors: true)[0].color == SIMD3(0, 1, 0))
    #expect(scene.lidarWorldSamples(colors: false)[0].color == SIMD3(repeating: 0.5))
}

@Test
func liveLiDARShowsOnlyWellSeenCellsWithoutMVS() {
    func surfel(_ p: SIMD3<Float>, views: Int) -> CaptureSurfelMap.Surfel {
        CaptureSurfelMap.Surfel(position: p, normal: SIMD3(0, 0, 1), color: .zero, cellMeters: 0.01,
                                level: 1, pointCount: 4, viewCount: views)
    }
    let lidar = [surfel(SIMD3(0.005, 0, 0), views: 5),   // MVS shows this spot → hidden
                 surfel(SIMD3(0.5, 0, 0), views: 1),     // one glance → hidden
                 surfel(SIMD3(1.0, 0, 0), views: 3)]     // shown
    let mvs = [surfel(SIMD3(0.01, 0.01, 0.01), views: 2)]
    let shown = LiveLiDARSurfels.shown(lidar, minViews: 3, mvs: mvs, coverCell: 0.03)
    #expect(shown.map(\.position) == [SIMD3(1.0, 0, 0)])
}

@Test
func liveLiDARColorsAreHalfDesaturated() {
    let red = CaptureSurfelMap.Surfel(position: .zero, normal: SIMD3(0, 0, 1), color: SIMD3(1, 0, 0),
                                      cellMeters: 0.01, level: 1, pointCount: 4, viewCount: 3)
    let faded = LiveLiDARSurfels.desaturated(red, keep: 0.5).color
    let luma: Float = 0.299
    #expect(simd_distance(faded, SIMD3(luma + 0.5 * (1 - luma), 0.5 * luma, 0.5 * luma)) < 1e-6)
    #expect(LiveLiDARSurfels.desaturated(red, keep: 1).color == red.color)
}
