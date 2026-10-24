// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import Testing
@testable import Trainer

private func previewTestCamera(width: Int, height: Int) -> Camera {
    let viewmat = MLXArray(
        [Float(1), 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1],
        [16]
    )
    return Camera(
        viewmat: viewmat,
        intrinsics: MLXArray([Float(width), Float(width), Float(width) / 2, Float(height) / 2], [4]),
        width: width, height: height, tileSize: 16,
        background: MLXArray([Float(0), 0, 0], [3])
    )
}

@Test
func trainingPreviewIsShrunkToAboutTheMaximumWidth() throws {
    let trainer = try GaussianTrainer(
        model: GaussianModel.random(count: 40, seed: 3), renderer: try GaussianRenderer())
    // 1024 / 720 rounds up to a factor of 2.
    let frame = try #require(TrainingPreviewFrame.render(
        trainer: trainer, camera: previewTestCamera(width: 1024, height: 768), shDegree: 0))
    #expect(frame.width == 512 && frame.height == 384)
    #expect(frame.rgba.count == frame.width * frame.height * 4)
    #expect(stride(from: 3, to: frame.rgba.count, by: 4).allSatisfy { frame.rgba[$0] == 255 })
}

@Test
func trainingPreviewKeepsASmallCameraAsItIs() throws {
    let trainer = try GaussianTrainer(
        model: GaussianModel.random(count: 40, seed: 4), renderer: try GaussianRenderer())
    let frame = try #require(TrainingPreviewFrame.render(
        trainer: trainer, camera: previewTestCamera(width: 64, height: 48), shDegree: 0))
    #expect(frame.width == 64 && frame.height == 48)
    // Gaussians were drawn: not every color channel is zero.
    let anyColor = (0..<(frame.width * frame.height)).contains { p in
        frame.rgba[p * 4] != 0 || frame.rgba[p * 4 + 1] != 0 || frame.rgba[p * 4 + 2] != 0
    }
    #expect(anyColor)
}

@Test
func trainingPreviewWalksThroughViewsSpreadOverTheCapture() {
    // 120 views, 6 slots: one per 20 views, taken from the middle of each stretch.
    let picked = (0..<12).map { TrainingPreviewFrame.viewIndex(forPreview: $0, viewCount: 120) }
    #expect(picked == [10, 10, 30, 30, 50, 50, 70, 70, 90, 90, 110, 110])
    // Then it starts over.
    #expect(TrainingPreviewFrame.viewIndex(forPreview: 12, viewCount: 120) == 10)
    // Few views: one slot per view; a single view stays put; every index is in range.
    #expect((0..<8).map { TrainingPreviewFrame.viewIndex(forPreview: $0, viewCount: 2) } == [0, 0, 1, 1, 0, 0, 1, 1])
    #expect(TrainingPreviewFrame.viewIndex(forPreview: 5, viewCount: 1) == 0)
    for count in [2, 3, 7, 50, 301] {
        for n in 0..<40 {
            let i = TrainingPreviewFrame.viewIndex(forPreview: n, viewCount: count)
            #expect(i >= 0 && i < count)
        }
    }
}
