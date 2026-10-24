// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Viewer colour grade: a non-destructive tint, temperature, saturation, brightness, levels and
// transparency applied to the foreground splats. The grade is baked into a copy of the splat
// points, so the loaded data stays as trained and any grade can be replaced or reset.
//
// Order: levels (black and white point), then tint and temperature, then saturation, then
// brightness. Levels and brightness are affine, so they only shift the base (SH0) colour; the
// higher SH bands are view-dependent and take the linear part alone.

#if os(iOS) || os(macOS)

import Foundation
import SplatIO
import simd

struct SplatViewerColorGrade: Equatable {
    /// Per-channel multiplier. White leaves the colour alone.
    var tint = SIMD3<Float>(1, 1, 1)
    /// -0.5 cools (blue up, red down), 0.5 warms (red up, blue down).
    var temperature: Float = 0
    /// 0 is grey, 1 is unchanged, 2 doubles the chroma.
    var saturation: Float = 1
    /// Added after the colour is graded, in sRGB. -1 darkens fully, 1 brightens fully.
    var brightness: Float = 0
    /// sRGB value that maps to black. 0 is unchanged.
    var blackPoint: Float = 0
    /// sRGB value that maps to white. 1 is unchanged.
    var whitePoint: Float = 1
    /// Log scale of the opacity factor: 0 is unchanged, positive is more opaque.
    var transparency: Float = 0

    static let identity = SplatViewerColorGrade()

    /// Levels need white above black. A grade that fails this is left unapplied instead of dividing by zero.
    var isValid: Bool { whitePoint > blackPoint + 0.001 }

    var isIdentity: Bool { self == .identity }

    /// Returns the point with this grade baked in. Opacity and scale and rotation are kept.
    func apply(to point: SplatPoint) -> SplatPoint {
        guard !isIdentity, isValid else { return point }
        var graded = point
        graded.color = gradedColor(point.color)
        graded.opacity = gradedOpacity(point.opacity)
        return graded
    }

    private var range: Float { 1 / (whitePoint - blackPoint) }

    // Per-channel gain after the levels stage: the levels range times tint and temperature.
    private var channelGain: SIMD3<Float> {
        range * SIMD3(tint.x * (1 + temperature), tint.y, tint.z * (1 - temperature))
    }

    // Saturation mix about the BT.601 luma weights, which sum to 1. A uniform value stays uniform.
    private static let luma = SIMD3<Float>(0.299, 0.587, 0.114)

    private func saturate(_ c: SIMD3<Float>) -> SIMD3<Float> {
        saturation * c + (1 - saturation) * simd_dot(Self.luma, c) * SIMD3(repeating: 1)
    }

    private func gradedColor(_ color: SplatPoint.Color) -> SplatPoint.Color {
        var coefficients = color.asSphericalHarmonicFloat
        guard !coefficients.isEmpty else { return color }

        // Base colour: the full affine grade on the decoded sRGB value.
        let dc = 0.5 + SplatPoint.Color.SH_C0 * coefficients[0]
        let base = saturate(channelGain * (dc - blackPoint)) + SIMD3(repeating: brightness)
        coefficients[0] = (base - 0.5) * SplatPoint.Color.INV_SH_C0

        // View-dependent bands: the linear part of the same grade, with no offset.
        for i in coefficients.indices.dropFirst() {
            coefficients[i] = saturate(channelGain * coefficients[i])
        }
        return .sphericalHarmonicFloat(coefficients)
    }

    private func gradedOpacity(_ opacity: SplatPoint.Opacity) -> SplatPoint.Opacity {
        let factor = exp(transparency)
        return .linearFloat(min(1, max(0, opacity.asLinearFloat * factor)))
    }
}

#endif
