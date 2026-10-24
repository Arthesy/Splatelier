// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Colour grade maths: identity, levels, tint and temperature, saturation, brightness, the SH split
// between the base colour and the view-dependent bands, and opacity.

import Foundation
import SplatIO
import Testing
import simd
@testable import SplatViewerKit

private func point(sRGB: SIMD3<Float>, opacity: Float = 0.5, extraBands: [SIMD3<Float>] = []) -> SplatPoint {
    let dc = (sRGB - 0.5) * SplatPoint.Color.INV_SH_C0
    return SplatPoint(
        position: .zero, color: .sphericalHarmonicFloat([dc] + extraBands),
        opacity: .linearFloat(opacity), scale: .linearFloat(SIMD3(repeating: 0.01)),
        rotation: simd_quatf(ix: 0, iy: 0, iz: 0, r: 1))
}

private func srgb(_ point: SplatPoint) -> SIMD3<Float> { point.color.asSRGBFloat }

private func near(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tol: Float = 1e-4) -> Bool {
    simd_length(a - b) < tol
}

@Test func identityLeavesPointUnchanged() {
    let p = point(sRGB: SIMD3(0.2, 0.5, 0.8), opacity: 0.3)
    let graded = SplatViewerColorGrade.identity.apply(to: p)
    #expect(near(srgb(graded), srgb(p)))
    #expect(abs(graded.opacity.asLinearFloat - 0.3) < 1e-5)
}

@Test func levelsMapBlackAndWhiteToEnds() {
    var grade = SplatViewerColorGrade()
    grade.blackPoint = 0.2
    grade.whitePoint = 0.8
    #expect(near(srgb(grade.apply(to: point(sRGB: SIMD3(repeating: 0.2)))), SIMD3(repeating: 0)))
    #expect(near(srgb(grade.apply(to: point(sRGB: SIMD3(repeating: 0.8)))), SIMD3(repeating: 1)))
    #expect(near(srgb(grade.apply(to: point(sRGB: SIMD3(repeating: 0.5)))), SIMD3(repeating: 0.5)))
}

@Test func invalidLevelsLeavePointUnchanged() {
    var grade = SplatViewerColorGrade()
    grade.blackPoint = 0.6
    grade.whitePoint = 0.4
    let p = point(sRGB: SIMD3(0.3, 0.5, 0.7))
    #expect(near(srgb(grade.apply(to: p)), srgb(p)))
}

@Test func tintMultipliesChannels() {
    var grade = SplatViewerColorGrade()
    grade.tint = SIMD3(1, 0.5, 1)
    let out = srgb(grade.apply(to: point(sRGB: SIMD3(repeating: 0.6))))
    // Tint scales the decoded value, so green goes from 0.6 to 0.3 while red and blue stay put.
    #expect(abs(out.x - 0.6) < 1e-4)
    #expect(abs(out.y - 0.3) < 1e-4)
    #expect(abs(out.z - 0.6) < 1e-4)
}

@Test func warmTemperatureRaisesRedAndLowersBlue() {
    var grade = SplatViewerColorGrade()
    grade.temperature = 0.5
    let out = srgb(grade.apply(to: point(sRGB: SIMD3(repeating: 0.5))))
    #expect(out.x > 0.5)
    #expect(out.z < 0.5)
    #expect(abs(out.y - 0.5) < 1e-4)
}

@Test func zeroSaturationGivesGrey() {
    var grade = SplatViewerColorGrade()
    grade.saturation = 0
    let out = srgb(grade.apply(to: point(sRGB: SIMD3(0.9, 0.4, 0.1))))
    #expect(abs(out.x - out.y) < 1e-4)
    #expect(abs(out.y - out.z) < 1e-4)
}

@Test func brightnessShiftsBaseColorOnly() {
    var grade = SplatViewerColorGrade()
    grade.brightness = 0.25
    let out = srgb(grade.apply(to: point(sRGB: SIMD3(repeating: 0.25))))
    #expect(near(out, SIMD3(repeating: 0.5)))
}

@Test func viewDependentBandsTakeOnlyTheLinearPart() {
    var grade = SplatViewerColorGrade()
    grade.tint = SIMD3(2, 1, 1)
    grade.brightness = 0.25
    let band: SIMD3<Float> = SIMD3(0.4, 0.4, 0.4)
    let p = point(sRGB: SIMD3(repeating: 0.5), extraBands: [band])
    let graded = grade.apply(to: p)
    guard case let .sphericalHarmonicFloat(coefficients) = graded.color else {
        Issue.record("graded colour is not spherical harmonics")
        return
    }
    // The band is scaled by the tint only: no brightness offset reaches it.
    #expect(abs(coefficients[1].x - 0.8) < 1e-5)
    #expect(abs(coefficients[1].y - 0.4) < 1e-5)
}

@Test func transparencyScalesOpacityInLogSpace() {
    var more = SplatViewerColorGrade()
    more.transparency = log(2)
    #expect(abs(more.apply(to: point(sRGB: SIMD3(repeating: 0.5), opacity: 0.3)).opacity.asLinearFloat - 0.6) < 1e-5)

    var less = SplatViewerColorGrade()
    less.transparency = -log(2)
    #expect(abs(less.apply(to: point(sRGB: SIMD3(repeating: 0.5), opacity: 0.8)).opacity.asLinearFloat - 0.4) < 1e-5)

    var clamped = SplatViewerColorGrade()
    clamped.transparency = log(4)
    #expect(clamped.apply(to: point(sRGB: SIMD3(repeating: 0.5), opacity: 0.5)).opacity.asLinearFloat == 1)
}
