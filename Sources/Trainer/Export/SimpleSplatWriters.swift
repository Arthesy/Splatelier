// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import spz

/// antimatter15 `.splat`: per splat position float3, linear scale float3, RGBA u8 (DC color and
/// sigmoid opacity), quaternion u8x4 (w, x, y, z mapped by q·128+128). No higher-order SH. Rows
/// are written in table order.
public enum DotSplatWriter {
    public static func write(_ table: SplatTable, to url: URL) throws {
        let n = table.count
        var data = Data(count: n * 32)
        data.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            func u8(_ v: Float) -> UInt8 { UInt8(Swift.max(0, Swift.min(255, v))) }
            for i in 0..<n {
                let rec = base + i * 32
                for k in 0..<3 {
                    rec.storeBytes(of: table.positions[i * 3 + k].bitPattern.littleEndian,
                                   toByteOffset: k * 4, as: UInt32.self)
                    rec.storeBytes(of: expf(table.logScales[i * 3 + k]).bitPattern.littleEndian,
                                   toByteOffset: 12 + k * 4, as: UInt32.self)
                    rec.storeBytes(of: u8((0.5 + table.sh0[i * 3 + k] * GaussianModel.shC0) * 255),
                                   toByteOffset: 24 + k, as: UInt8.self)
                }
                rec.storeBytes(of: u8(table.opacity(i) * 255), toByteOffset: 27, as: UInt8.self)
                for k in 0..<4 {
                    rec.storeBytes(of: u8(table.quats[i * 4 + k] * 128 + 128),
                                   toByteOffset: 28 + k, as: UInt8.self)
                }
            }
        }
        try data.write(to: url)
    }
}

/// Niantic SPZ through spz-swift (version 3, gzip). The table is PLY space (RDF); spz converts it
/// to its own axes. Higher-order SH keeps the table's degree.
public enum SPZWriter {
    public static func write(_ table: SplatTable, to url: URL) throws {
        let n = table.count
        let rc = table.restCoeffs
        var rotations = [Float](repeating: 0, count: n * 4)
        var sh = [Float](repeating: 0, count: n * rc * 3)
        for i in 0..<n {
            // (w, x, y, z) → (x, y, z, w)
            rotations[i * 4 + 0] = table.quats[i * 4 + 1]
            rotations[i * 4 + 1] = table.quats[i * 4 + 2]
            rotations[i * 4 + 2] = table.quats[i * 4 + 3]
            rotations[i * 4 + 3] = table.quats[i * 4 + 0]
            // channel-major → coefficient-major with channels innermost
            for c in 0..<rc {
                for ch in 0..<3 {
                    sh[(i * rc + c) * 3 + ch] = table.shRest[(i * 3 + ch) * rc + c]
                }
            }
        }
        let cloud = GaussianCloud(
            numPoints: Int32(n), shDegree: Int32(table.shDegree),
            positions: table.positions, scales: table.logScales, rotations: rotations,
            alphas: table.logitOpacities, colors: table.sh0, sh: sh
        )
        try saveSpz(cloud, to: url, options: PackOptions(from: .rdf))
    }
}

/// glTF 2.0 binary with KHR_gaussian_splatting, as splat-transform writes it. glTF is Y-up, so the
/// PLY-space table is rotated 180° about Z (x, y negated), with the quaternion and the SH rotated
/// to match. Values are activated (linear scale, sigmoid opacity); SH stays raw.
public enum GLBWriter {

    public static func write(_ table: SplatTable, to url: URL) throws {
        let n = table.count
        let rc = table.restCoeffs

        struct Segment {
            var name: String
            var bytes: Data
            var componentType: Int
            var type: String
            var normalized = false
            var min: [Float]? = nil
            var max: [Float]? = nil
        }
        func floats(_ v: [Float]) -> Data { v.withUnsafeBytes { Data($0) } }

        var pos = [Float](repeating: 0, count: n * 3)
        var lo = [Float](repeating: .infinity, count: 3), hi = [Float](repeating: -.infinity, count: 3)
        for i in 0..<n {
            let p = [-table.positions[i * 3], -table.positions[i * 3 + 1], table.positions[i * 3 + 2]]
            for k in 0..<3 {
                pos[i * 3 + k] = p[k]
                lo[k] = Swift.min(lo[k], p[k]); hi[k] = Swift.max(hi[k], p[k])
            }
        }
        if n == 0 { lo = [0, 0, 0]; hi = [0, 0, 0] }
        var segments = [Segment(name: "POSITION", bytes: floats(pos), componentType: 5126,
                                type: "VEC3", min: lo, max: hi)]

        var color = [UInt8](repeating: 0, count: n * 4)
        for i in 0..<n {
            for k in 0..<3 {
                let v = ((table.sh0[i * 3 + k] * GaussianModel.shC0 + 0.5) * 255).rounded()
                color[i * 4 + k] = UInt8(Swift.max(0, Swift.min(255, v)))
            }
            color[i * 4 + 3] = UInt8(Swift.max(0, Swift.min(255, (table.opacity(i) * 255).rounded())))
        }
        segments.append(Segment(name: "COLOR_0", bytes: Data(color), componentType: 5121,
                                type: "VEC4", normalized: true))

        // Rotating by r = (w 0, x 0, y 0, z 1): r·q = (-qz, -qy, qx, qw) as (w, x, y, z); stored
        // xyzw.
        var rot = [Float](repeating: 0, count: n * 4)
        for i in 0..<n {
            let w = table.quats[i * 4], x = table.quats[i * 4 + 1]
            let y = table.quats[i * 4 + 2], z = table.quats[i * 4 + 3]
            rot[i * 4 + 0] = -y
            rot[i * 4 + 1] = x
            rot[i * 4 + 2] = w
            rot[i * 4 + 3] = -z
        }
        segments.append(Segment(name: "KHR_gaussian_splatting:ROTATION", bytes: floats(rot),
                                componentType: 5126, type: "VEC4"))
        segments.append(Segment(name: "KHR_gaussian_splatting:SCALE",
                                bytes: floats(table.logScales.map { expf($0) }),
                                componentType: 5126, type: "VEC3"))
        segments.append(Segment(name: "KHR_gaussian_splatting:OPACITY",
                                bytes: floats((0..<n).map { table.opacity($0) }),
                                componentType: 5126, type: "SCALAR"))
        segments.append(Segment(name: "KHR_gaussian_splatting:SH_DEGREE_0_COEF_0",
                                bytes: floats(table.sh0), componentType: 5126, type: "VEC3"))

        // A 180° turn about Z multiplies the real SH basis Y_l^m by (-1)^m.
        var coef = 0
        for degree in stride(from: 1, through: table.shDegree, by: 1) {
            for c in 0..<(2 * degree + 1) {
                let m = c - degree
                let sign: Float = m % 2 == 0 ? 1 : -1
                var buf = [Float](repeating: 0, count: n * 3)
                for i in 0..<n {
                    for ch in 0..<3 { buf[i * 3 + ch] = sign * table.shRest[(i * 3 + ch) * rc + coef] }
                }
                segments.append(Segment(name: "KHR_gaussian_splatting:SH_DEGREE_\(degree)_COEF_\(c)",
                                        bytes: floats(buf), componentType: 5126, type: "VEC3"))
                coef += 1
            }
        }

        func align4(_ v: Int) -> Int { (v + 3) & ~3 }
        var bin = Data()
        var bufferViews: [[String: Any]] = []
        var accessors: [[String: Any]] = []
        var attributes: [String: Int] = [:]
        for (i, seg) in segments.enumerated() {
            bufferViews.append(["buffer": 0, "byteOffset": bin.count,
                                "byteLength": seg.bytes.count, "target": 34962])
            bin.append(seg.bytes)
            bin.append(Data(count: align4(seg.bytes.count) - seg.bytes.count))
            var acc: [String: Any] = ["bufferView": i, "byteOffset": 0,
                                      "componentType": seg.componentType, "count": n, "type": seg.type]
            if seg.normalized { acc["normalized"] = true }
            if let mn = seg.min, let mx = seg.max { acc["min"] = mn; acc["max"] = mx }
            accessors.append(acc)
            attributes[seg.name] = i
        }

        let gltf: [String: Any] = [
            "asset": ["version": "2.0", "generator": "Splatelier"],
            "extensionsUsed": ["KHR_gaussian_splatting"],
            "scene": 0,
            "scenes": [["nodes": [0]]],
            "nodes": [["mesh": 0]],
            "buffers": [["byteLength": bin.count]],
            "bufferViews": bufferViews,
            "accessors": accessors,
            "meshes": [[
                "primitives": [[
                    "attributes": attributes,
                    "mode": 0,
                    "extensions": ["KHR_gaussian_splatting": [
                        "kernel": "ellipse",
                        "colorSpace": "srgb_rec709_display",
                        "sortingMethod": "cameraDistance",
                        "projection": "perspective",
                    ]],
                ]],
            ]],
        ]
        var json = try JSONSerialization.data(withJSONObject: gltf, options: [.sortedKeys])
        json.append(Data(repeating: 0x20, count: align4(json.count) - json.count))

        var out = Data()
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { out.append(contentsOf: $0) } }
        u32(0x4654_6C67)                                  // "glTF"
        u32(2)
        u32(12 + 8 + json.count + 8 + bin.count)
        u32(json.count); u32(0x4E4F_534A)                 // "JSON"
        out.append(json)
        u32(bin.count); u32(0x004E_4942)                  // "BIN\0"
        out.append(bin)
        try out.write(to: url)
    }
}

/// PlayCanvas / SuperSplat compressed PLY. Rows are Morton-ordered (foreground and background
/// separately, background last), grouped into chunks of 256 with per-chunk min/max, and packed to
/// four uint32 per splat: position 11-10-11, rotation smallest-three 2-10-10-10, log-scale
/// 11-10-11, color RGBA 8-8-8-8. Higher-order SH goes to a separate `sh` element as uint8.
public enum CompressedPLYWriter {

    static let chunkSize = 256

    public static func write(_ table: SplatTable, to url: URL) throws {
        let order = table.tailPreservingOrder { MortonOrder.sort($0, positions: table.positions) }
        let t = table.permuted(order)
        let n = t.count
        let numChunks = (n + chunkSize - 1) / chunkSize
        let rc = t.restCoeffs

        var header = [
            "ply",
            "format binary_little_endian 1.0",
            "comment Generated by Splatelier",
            "element chunk \(numChunks)",
        ]
        for p in ["min_x", "min_y", "min_z", "max_x", "max_y", "max_z",
                  "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
                  "min_r", "min_g", "min_b", "max_r", "max_g", "max_b"] {
            header.append("property float \(p)")
        }
        header.append("element vertex \(n)")
        for p in ["packed_position", "packed_rotation", "packed_scale", "packed_color"] {
            header.append("property uint \(p)")
        }
        if rc > 0 {
            header.append("element sh \(n)")
            for k in 0..<(rc * 3) { header.append("property uchar f_rest_\(k)") }
        }
        header.append("end_header\n")

        var chunkData = [Float](repeating: 0, count: numChunks * 18)
        var packed = [UInt32](repeating: 0, count: n * 4)
        var shData = [UInt8](repeating: 0, count: n * rc * 3)

        func packUnorm(_ v: Float, _ bits: Int) -> UInt32 {
            let t = Float((1 << bits) - 1)
            return UInt32(Swift.max(0, Swift.min(t, (v * t + 0.5).rounded(.down))))
        }
        func normalize(_ x: Float, _ lo: Float, _ hi: Float) -> Float {
            if x <= lo { return 0 }
            if x >= hi { return 1 }
            return hi - lo < 0.00001 ? 0 : (x - lo) / (hi - lo)
        }
        func pack111011(_ x: Float, _ y: Float, _ z: Float) -> UInt32 {
            (packUnorm(x, 11) << 21) | (packUnorm(y, 10) << 11) | packUnorm(z, 11)
        }

        for c in 0..<numChunks {
            let start = c * chunkSize
            let end = Swift.min(n, start + chunkSize)
            // A short final chunk repeats its last splat, which leaves the min/max unchanged.
            var pMin = [Float](repeating: .infinity, count: 3), pMax = [Float](repeating: -.infinity, count: 3)
            var sMin = pMin, sMax = pMax, cMin = pMin, cMax = pMax
            for i in start..<end {
                for k in 0..<3 {
                    let p = t.positions[i * 3 + k], s = t.logScales[i * 3 + k]
                    let col = t.sh0[i * 3 + k] * GaussianModel.shC0 + 0.5
                    pMin[k] = Swift.min(pMin[k], p); pMax[k] = Swift.max(pMax[k], p)
                    sMin[k] = Swift.min(sMin[k], s); sMax[k] = Swift.max(sMax[k], s)
                    cMin[k] = Swift.min(cMin[k], col); cMax[k] = Swift.max(cMax[k], col)
                }
            }
            // Scales can be ±inf; clamp the range.
            for k in 0..<3 {
                sMin[k] = Swift.max(-20, Swift.min(20, sMin[k]))
                sMax[k] = Swift.max(-20, Swift.min(20, sMax[k]))
            }
            chunkData.replaceSubrange(c * 18..<(c * 18 + 18), with: pMin + pMax + sMin + sMax + cMin + cMax)

            for i in start..<end {
                packed[i * 4 + 0] = pack111011(
                    normalize(t.positions[i * 3], pMin[0], pMax[0]),
                    normalize(t.positions[i * 3 + 1], pMin[1], pMax[1]),
                    normalize(t.positions[i * 3 + 2], pMin[2], pMax[2]))
                packed[i * 4 + 1] = packRotation(t.quats, i, packUnorm)
                packed[i * 4 + 2] = pack111011(
                    normalize(t.logScales[i * 3], sMin[0], sMax[0]),
                    normalize(t.logScales[i * 3 + 1], sMin[1], sMax[1]),
                    normalize(t.logScales[i * 3 + 2], sMin[2], sMax[2]))
                var color: UInt32 = 0
                for k in 0..<3 {
                    let col = t.sh0[i * 3 + k] * GaussianModel.shC0 + 0.5
                    color |= packUnorm(normalize(col, cMin[k], cMax[k]), 8) << UInt32(24 - 8 * k)
                }
                packed[i * 4 + 3] = color | packUnorm(t.opacity(i), 8)

                for k in 0..<(rc * 3) {
                    let v = (t.shRest[i * rc * 3 + k] / 8 + 0.5) * 256
                    shData[i * rc * 3 + k] = UInt8(Swift.max(0, Swift.min(255, v.rounded(.towardZero))))
                }
            }
        }

        var data = Data(header.joined(separator: "\n").utf8)
        chunkData.withUnsafeBytes { data.append(contentsOf: $0) }
        packed.withUnsafeBytes { data.append(contentsOf: $0) }
        data.append(contentsOf: shData)
        try data.write(to: url)
    }

    /// Smallest-three: 2 bits for the index of the largest |component| (in w, x, y, z order),
    /// then the other three, sign-flipped so the largest is positive, scaled by √2/2 into [0, 1].
    private static func packRotation(
        _ quats: [Float], _ i: Int, _ packUnorm: (Float, Int) -> UInt32
    ) -> UInt32 {
        var a = [quats[i * 4], quats[i * 4 + 1], quats[i * 4 + 2], quats[i * 4 + 3]]
        var largest = 0
        for k in 1..<4 where abs(a[k]) > abs(a[largest]) { largest = k }
        if a[largest] < 0 { a = a.map { -$0 } }
        let norm = Float(2).squareRoot() * 0.5
        var result = UInt32(largest)
        for k in 0..<4 where k != largest {
            result = (result << 10) | packUnorm(a[k] * norm + 0.5, 10)
        }
        return result
    }
}
