// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// PlayCanvas SOG (version 2), bundled: a zip holding `meta.json` and lossless WebP textures, one
/// texel per splat in Morton order (foreground and background separately, background last).
///
/// - means_l / means_u: sign(x)·log(|x|+1) quantized to 16 bits over its min/max, low and high
///   bytes.
/// - quats: smallest-three, the dropped component's index in alpha (252 + index).
/// - scales: indices into a 256-entry codebook of log-scales.
/// - sh0: indices into a 256-entry codebook of f_dc, opacity in alpha.
/// - shN_labels / shN_centroids: k-means palette of the higher-order SH (centroids quantized to a
///   256-entry codebook) and each splat's palette index.
public enum SOGWriter {

    public struct Options: Sendable {
        /// k-means iterations for the higher-order SH palette.
        public var shIterations: Int
        /// Receives one line per finished stage with its time (for logs).
        public var log: (@Sendable (String) -> Void)?
        public init(shIterations: Int = 10, log: (@Sendable (String) -> Void)? = nil) {
            self.shIterations = shIterations
            self.log = log
        }
    }

    enum Error: Swift.Error, CustomStringConvertible {
        case empty
        case tooLarge(Int)
        var description: String {
            switch self {
            case .empty: return "SOG needs at least one splat"
            case .tooLarge(let n): return "SOG textures are capped at 16383×16383 texels; got \(n) splats"
            }
        }
    }

    public static func write(_ table: SplatTable, to url: URL, options: Options = Options()) throws {
        let files = try encode(table, options: options)
        var zip = ZipStoreWriter()
        for (name, data) in files { zip.add(name, data) }
        try zip.finish().write(to: url)
    }

    /// The bundle's files in write order (textures, then meta.json).
    static func encode(_ table: SplatTable, options: Options) throws -> [(String, Data)] {
        let n = table.count
        guard n > 0 else { throw Error.empty }
        let width = Int((Double(n).squareRoot() / 4).rounded(.up)) * 4
        let height = Int((Double(n) / Double(width) / 4).rounded(.up)) * 4
        guard width <= 16383, height <= 16383 else { throw Error.tooLarge(n) }

        let order = table.tailPreservingOrder { MortonOrder.sort($0, positions: table.positions) }
        let t = table.permuted(order)
        var files: [(String, Data)] = []
        var stageStart = Date()
        func stage(_ name: String) {
            options.log?(String(format: "SOG %@: %.2fs", name, Date().timeIntervalSince(stageStart)))
            stageStart = Date()
        }
        stage("morton order")
        func texture(_ name: String, _ rgba: [UInt8], _ w: Int = width, _ h: Int = height) {
            files.append((name, WebPLosslessEncoder.encode(rgba: rgba, width: w, height: h)))
        }
        func blank() -> [UInt8] { [UInt8](repeating: 0, count: width * height * 4) }

        // means
        func logTransform(_ v: Float) -> Float { (v < 0 ? -1 : 1) * logf(abs(v) + 1) }
        var mins = [Float](repeating: .infinity, count: 3), maxs = [Float](repeating: -.infinity, count: 3)
        for i in 0..<n {
            for a in 0..<3 {
                let v = logTransform(t.positions[i * 3 + a])
                mins[a] = Swift.min(mins[a], v); maxs[a] = Swift.max(maxs[a], v)
            }
        }
        var meansL = blank(), meansU = blank()
        for i in 0..<n {
            for a in 0..<3 {
                let range = maxs[a] - mins[a]
                let q = range > 0 ? 65535 * (logTransform(t.positions[i * 3 + a]) - mins[a]) / range : 0
                let v = UInt32(clampForPacking(q, 65535))
                meansL[i * 4 + a] = UInt8(v & 0xFF)
                meansU[i * 4 + a] = UInt8(v >> 8)
            }
            meansL[i * 4 + 3] = 255
            meansU[i * 4 + 3] = 255
        }
        texture("means_l.webp", meansL)
        texture("means_u.webp", meansU)
        stage("means")

        // quats
        var quats = blank()
        let others = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        for i in 0..<n {
            var q = (0..<4).map { t.quats[i * 4 + $0] }
            var largest = 0
            for k in 1..<4 where abs(q[k]) > abs(q[largest]) { largest = k }
            let s: Float = (q[largest] < 0 ? -1 : 1) * Float(2).squareRoot()
            q = q.map { $0 * s }
            for (c, k) in others[largest].enumerated() {
                quats[i * 4 + c] = UInt8(clampForPacking(255 * (q[k] * 0.5 + 0.5), 255))
            }
            quats[i * 4 + 3] = UInt8(252 + largest)
        }
        texture("quats.webp", quats)
        stage("quats")

        // scales
        let scaleQ = Quantize1D.quantize((0..<3).map { a in (0..<n).map { t.logScales[$0 * 3 + a] } })
        var scales = blank()
        for i in 0..<n {
            for a in 0..<3 { scales[i * 4 + a] = scaleQ.labels[a][i] }
            scales[i * 4 + 3] = 255
        }
        texture("scales.webp", scales)
        stage("scales")

        // sh0 + opacity
        let colorQ = Quantize1D.quantize((0..<3).map { a in (0..<n).map { t.sh0[$0 * 3 + a] } })
        var sh0 = blank()
        for i in 0..<n {
            for a in 0..<3 { sh0[i * 4 + a] = colorQ.labels[a][i] }
            sh0[i * 4 + 3] = UInt8(clampForPacking(t.opacity(i) * 255, 255))
        }
        texture("sh0.webp", sh0)
        stage("sh0")

        // shN
        var meta: [String: Any] = [
            "version": 2,
            "asset": ["generator": "Splatelier"],
            "count": n,
            "means": ["mins": mins, "maxs": maxs, "files": ["means_l.webp", "means_u.webp"]],
            "scales": ["codebook": scaleQ.centroids, "files": ["scales.webp"]],
            "quats": ["files": ["quats.webp"]],
            "sh0": ["codebook": colorQ.centroids, "files": ["sh0.webp"]],
        ]
        if t.shDegree > 0 {
            let coeffs = t.restCoeffs
            let dims = coeffs * 3
            let paletteSize = Int(Swift.min(64, pow(2, (log2(Double(n) / 1024)).rounded(.down))) * 1024)
            let (centroids, labels) = KMeans.cluster(t.shRest, dims: dims, k: paletteSize,
                                                     iterations: options.shIterations)
            stage("shN k-means (\(paletteSize) clusters, \(dims) dims)")
            let numCentroids = centroids.count / dims
            let cbQ = Quantize1D.quantize((0..<dims).map { j in
                (0..<numCentroids).map { centroids[$0 * dims + j] }
            })
            let rows = (numCentroids + 63) / 64
            var cent = [UInt8](repeating: 0, count: 64 * coeffs * rows * 4)
            for i in 0..<numCentroids {
                for j in 0..<coeffs {
                    let p = (i * coeffs + j) * 4
                    for ch in 0..<3 { cent[p + ch] = cbQ.labels[ch * coeffs + j][i] }
                    cent[p + 3] = 255
                }
            }
            var lab = blank()
            for i in 0..<n {
                lab[i * 4] = UInt8(labels[i] & 0xFF)
                lab[i * 4 + 1] = UInt8((labels[i] >> 8) & 0xFF)
                lab[i * 4 + 3] = 255
            }
            texture("shN_centroids.webp", cent, 64 * coeffs, rows)
            texture("shN_labels.webp", lab)
            stage("shN textures")
            meta["shN"] = [
                "count": paletteSize,
                "bands": t.shDegree,
                "codebook": cbQ.centroids,
                "files": ["shN_centroids.webp", "shN_labels.webp"],
            ] as [String: Any]
        }

        files.append(("meta.json", try JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys])))
        return files
    }
}

/// Zip archive with stored (uncompressed) entries.
struct ZipStoreWriter {
    private var body = Data()
    private var central = Data()
    private var entries = 0

    mutating func add(_ name: String, _ data: Data) {
        let nameBytes = Data(name.utf8)
        let crc = CRC32.checksum(data)
        let offset = body.count
        // Local file header
        Self.append(&body, u32: 0x0403_4B50)
        Self.append(&body, u16: 20)                  // version needed
        Self.append(&body, u16: 0)                   // flags
        Self.append(&body, u16: 0)                   // stored
        Self.append(&body, u16: 0); Self.append(&body, u16: 0x21)   // time, date (1980-01-01)
        Self.append(&body, u32: crc)
        Self.append(&body, u32: UInt32(data.count)); Self.append(&body, u32: UInt32(data.count))
        Self.append(&body, u16: UInt16(nameBytes.count))
        Self.append(&body, u16: 0)                   // extra length
        body.append(nameBytes)
        body.append(data)
        // Central directory record
        Self.append(&central, u32: 0x0201_4B50)
        Self.append(&central, u16: 20)               // version made by
        Self.append(&central, u16: 20)
        Self.append(&central, u16: 0)
        Self.append(&central, u16: 0)
        Self.append(&central, u16: 0); Self.append(&central, u16: 0x21)
        Self.append(&central, u32: crc)
        Self.append(&central, u32: UInt32(data.count)); Self.append(&central, u32: UInt32(data.count))
        Self.append(&central, u16: UInt16(nameBytes.count))
        Self.append(&central, u16: 0)                // extra
        Self.append(&central, u16: 0)                // comment
        Self.append(&central, u16: 0)                // disk
        Self.append(&central, u16: 0)                // internal attrs
        Self.append(&central, u32: 0)                // external attrs
        Self.append(&central, u32: UInt32(offset))
        central.append(nameBytes)
        entries += 1
    }

    func finish() -> Data {
        var out = body
        let cdOffset = out.count
        out.append(central)
        Self.append(&out, u32: 0x0605_4B50)
        Self.append(&out, u16: 0); Self.append(&out, u16: 0)
        Self.append(&out, u16: UInt16(entries)); Self.append(&out, u16: UInt16(entries))
        Self.append(&out, u32: UInt32(central.count))
        Self.append(&out, u32: UInt32(cdOffset))
        Self.append(&out, u16: 0)
        return out
    }

    private static func append(_ d: inout Data, u16 v: UInt16) {
        withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
    }
    private static func append(_ d: inout Data, u32 v: UInt32) {
        withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) }
    }
}

enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { i in
        var c = UInt32(i)
        for _ in 0..<8 { c = c & 1 != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func checksum(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in data { c = table[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }
}
