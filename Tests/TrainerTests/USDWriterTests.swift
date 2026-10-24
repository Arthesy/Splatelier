// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import Testing
@testable import Trainer

// `USDWriter` output for one splat at SH degree 1, checked against the conventions it documents:
// the 180° turn about Z (x, y negated) for a Y-up stage, activated scale and opacity, (w, x, y, z)
// quaternions, and the (-1)^m sign on the rest SH coefficients.
@Suite struct USDWriterTests {

    private func oneSplatTable() -> SplatTable {
        // Channel-major rest SH: channel r has coefficients 1, 2, 3; g has 4, 5, 6; b has 7, 8, 9.
        SplatTable(
            positions: [1, 2, 3],
            logScales: [0, 0, 0],
            quats: [0.6, 0, 0.8, 0],
            logitOpacities: [0],
            sh0: [0.1, 0.2, 0.3],
            shRest: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            shDegree: 1
        )
    }

    @Test func writesSchemaAttributesWithDocumentedConventions() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("usdwriter-\(UUID().uuidString).usda")
        defer { try? FileManager.default.removeItem(at: url) }

        try USDWriter.write(oneSplatTable(), to: url)
        let text = try String(contentsOf: url, encoding: .utf8)

        #expect(text.hasPrefix("#usda 1.0\n"))
        #expect(text.contains("upAxis = \"Y\""))
        #expect(text.contains("def ParticleField3DGaussianSplat \"Splats\""))
        #expect(text.contains("point3f[] positions = [(-1.0, -2.0, 3.0)]"))
        #expect(text.contains("quatf[] orientations = [(-0.0, -0.8, 0.0, 0.6)]"))
        #expect(text.contains("float3[] scales = [(1.0, 1.0, 1.0)]"))
        #expect(text.contains("float[] opacities = [0.5]"))
        #expect(text.contains("int radiance:sphericalHarmonicsDegree = 1"))
        #expect(text.contains("float3[] radiance:sphericalHarmonicsCoefficients = ["
                              + "(0.1, 0.2, 0.3), (-1.0, -4.0, -7.0), (2.0, 5.0, 8.0), (-3.0, -6.0, -9.0)]"))
        #expect(text.contains("elementSize = 4"))
        #expect(text.contains("interpolation = \"vertex\""))
        #expect(text.contains("float3[] extent = [(-1.0, -2.0, 3.0), (-1.0, -2.0, 3.0)]"))
    }
}
