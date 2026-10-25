// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// OpenUSD ASCII (.usda) with a `ParticleField3DGaussianSplat` prim, the 3D Gaussian splat schema
/// that OpenUSD v26.03 added. Attribute names and value conventions follow that schema
/// (`positions`, `orientations`, `scales`, `opacities`, `radiance:sphericalHarmonicsDegree`,
/// `radiance:sphericalHarmonicsCoefficients`) and the reference converter py3dgsPlyToUsd.py:
/// scales and opacities are stored activated (linear), orientations are (w, x, y, z), and the SH
/// coefficients are per splat with the DC term first, one float3 per coefficient.
///
/// As in `GLBWriter`, the PLY-space table is rotated 180° about Z so the stage is Y-up.
public enum USDWriter {

    public static func write(_ table: SplatTable, to url: URL) throws {
        let n = table.count
        let rc = table.restCoeffs
        let perSplat = rc + 1

        // The 180° turn about Z multiplies the real SH basis Y_l^m by (-1)^m; the DC term is l = 0.
        var restSign = [Float]()
        for degree in stride(from: 1, through: table.shDegree, by: 1) {
            for c in 0..<(2 * degree + 1) {
                restSign.append((c - degree) % 2 == 0 ? 1 : -1)
            }
        }

        var lo = [Float](repeating: .infinity, count: 3)
        var hi = [Float](repeating: -.infinity, count: 3)
        for i in 0..<n {
            let p = position(table, i)
            for k in 0..<3 {
                lo[k] = Swift.min(lo[k], p[k]); hi[k] = Swift.max(hi[k], p[k])
            }
        }
        if n == 0 { lo = [0, 0, 0]; hi = [0, 0, 0] }

        let out = try Output(url: url)
        try out.write("""
            #usda 1.0
            (
                defaultPrim = "Splats"
                metersPerUnit = 1
                upAxis = "Y"
                doc = "Converted by Splatelier"
            )

            def ParticleField3DGaussianSplat "Splats"
            {
            """ + "\n")

        try out.array("point3f[] positions", count: n) { i in
            tuple(position(table, i))
        }
        try out.array("quatf[] orientations", count: n) { i in
            // Rotating by (w 0, x 0, y 0, z 1) gives (-qz, -qy, qx, qw) as (w, x, y, z).
            let q = table.quats
            return tuple([-q[i * 4 + 3], -q[i * 4 + 2], q[i * 4 + 1], q[i * 4]])
        }
        try out.array("float3[] scales", count: n) { i in
            tuple((0..<3).map { expf(table.logScales[i * 3 + $0]) })
        }
        try out.array("float[] opacities", count: n) { i in
            "\(table.opacity(i))"
        }
        try out.line("int radiance:sphericalHarmonicsDegree = \(table.shDegree)")
        try out.array("float3[] radiance:sphericalHarmonicsCoefficients", count: n * perSplat,
                      attributes: "(\n        elementSize = \(perSplat)\n        interpolation = \"vertex\"\n    )") { e in
            let i = e / perSplat, j = e % perSplat
            if j == 0 {
                return tuple((0..<3).map { table.sh0[i * 3 + $0] })
            }
            return tuple((0..<3).map { ch in
                restSign[j - 1] * table.shRest[(i * 3 + ch) * rc + (j - 1)]
            })
        }
        try out.line("float3[] extent = [\(tuple(lo)), \(tuple(hi))]")
        try out.write("}\n")
        try out.close()
    }

    /// The position after the 180° turn about Z: x and y negated.
    private static func position(_ table: SplatTable, _ i: Int) -> [Float] {
        [-table.positions[i * 3], -table.positions[i * 3 + 1], table.positions[i * 3 + 2]]
    }

    private static func tuple(_ v: [Float]) -> String {
        "(" + v.map { "\($0)" }.joined(separator: ", ") + ")"
    }

    /// Buffered text output, so a large scene is written in chunks instead of one big string.
    private final class Output {
        private let handle: FileHandle
        private var buffer = Data()

        init(url: URL) throws {
            FileManager.default.createFile(atPath: url.path, contents: nil)
            handle = try FileHandle(forWritingTo: url)
        }

        func write(_ text: String) throws {
            buffer.append(contentsOf: text.utf8)
            if buffer.count >= 1 << 20 { try flush() }
        }

        func line(_ text: String) throws {
            try write("    " + text + "\n")
        }

        /// `    <type> <name> = [ tuple, ... ]`, one entry per element of `count`.
        func array(_ decl: String, count: Int, attributes: String = "",
                   _ element: (Int) -> String) throws {
            try write("    \(decl) = [")
            for e in 0..<count {
                if e > 0 { try write(", ") }
                try write(element(e))
            }
            try write("]\(attributes.isEmpty ? "" : " " + attributes)\n")
        }

        func close() throws {
            try flush()
            try handle.close()
        }

        private func flush() throws {
            try handle.write(contentsOf: buffer)
            buffer.removeAll(keepingCapacity: true)
        }
    }
}
