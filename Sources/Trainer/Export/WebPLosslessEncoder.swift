// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Minimal lossless WebP (VP8L, RFC 9649) encoder. ImageIO can decode WebP but not encode it, and
/// SOG stores its data as lossless WebP textures.
///
/// Pipeline: subtract-green transform, predictor transform (per 16×16 block, the mode with the
/// smallest residual magnitude), then the residuals as literals plus backward references to the
/// pixel to the left or above (runs), entropy-coded with one group of five canonical prefix codes.
/// No color cache or meta prefix codes. Output is exact (lossless) but larger than libwebp's.
enum WebPLosslessEncoder {

    /// Encodes `rgba` (`width*height*4` bytes, non-premultiplied RGBA) as a `.webp` file.
    static func encode(rgba: [UInt8], width: Int, height: Int) -> Data {
        precondition(width >= 1 && height >= 1 && width <= 16384 && height <= 16384,
                     "VP8L dimensions must be 1...16384")
        precondition(rgba.count == width * height * 4, "rgba must be width*height*4 bytes")

        var argb = [UInt32](repeating: 0, count: width * height)
        var hasAlpha = false
        for i in 0..<argb.count {
            let r = UInt32(rgba[i * 4]), g = UInt32(rgba[i * 4 + 1])
            let b = UInt32(rgba[i * 4 + 2]), a = UInt32(rgba[i * 4 + 3])
            if a != 255 { hasAlpha = true }
            argb[i] = (a << 24) | (r << 16) | (g << 8) | b
        }

        var w = BitWriter()
        w.write(0x2F, 8)                       // signature
        w.write(UInt32(width - 1), 14)
        w.write(UInt32(height - 1), 14)
        w.write(hasAlpha ? 1 : 0, 1)
        w.write(0, 3)                          // version

        // Transforms, in the order applied (the decoder undoes them in reverse).
        w.write(1, 1); w.write(2, 2)           // SUBTRACT_GREEN
        subtractGreen(&argb)

        let sizeBits = 4
        w.write(1, 1); w.write(0, 2)           // PREDICTOR
        w.write(UInt32(sizeBits - 2), 3)
        let (residuals, modes, bw, bh) = predict(argb, width: width, height: height, sizeBits: sizeBits)
        // Sub-image: the mode in the green channel.
        writeImage(&w, modes.map { 0xFF00_0000 | (UInt32($0) << 8) }, width: bw, height: bh,
                   isMain: false, useBackRefs: false)

        w.write(0, 1)                          // no more transforms
        writeImage(&w, residuals, width: width, height: height, isMain: true, useBackRefs: true)

        let payload = w.finish()
        var out = Data()
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { out.append(contentsOf: $0) } }
        let pad = payload.count & 1
        out.append(contentsOf: Array("RIFF".utf8))
        u32(4 + 8 + payload.count + pad)
        out.append(contentsOf: Array("WEBPVP8L".utf8))
        u32(payload.count)
        out.append(payload)
        if pad == 1 { out.append(0) }
        return out
    }

    // MARK: - Transforms

    private static func subtractGreen(_ px: inout [UInt32]) {
        for i in 0..<px.count {
            let p = px[i]
            let g = (p >> 8) & 0xFF
            let r = ((p >> 16) &- g) & 0xFF
            let b = (p &- g) & 0xFF
            px[i] = (p & 0xFF00_FF00) | (r << 16) | b
        }
    }

    @inline(__always) private static func average2(_ a: UInt32, _ b: UInt32) -> UInt32 {
        (((a ^ b) & 0xFEFE_FEFE) >> 1) &+ (a & b)
    }

    @inline(__always) private static func channel(_ p: UInt32, _ shift: UInt32) -> Int {
        Int((p >> shift) & 0xFF)
    }

    private static func select(_ l: UInt32, _ t: UInt32, _ tl: UInt32) -> UInt32 {
        var pl = 0, pt = 0
        for s in stride(from: UInt32(0), through: 24, by: 8) {
            let p = channel(l, s) + channel(t, s) - channel(tl, s)
            pl += abs(p - channel(l, s))
            pt += abs(p - channel(t, s))
        }
        return pl < pt ? l : t
    }

    private static func clampAddSubtractFull(_ a: UInt32, _ b: UInt32, _ c: UInt32) -> UInt32 {
        var out: UInt32 = 0
        for s in stride(from: UInt32(0), through: 24, by: 8) {
            let v = Swift.max(0, Swift.min(255, channel(a, s) + channel(b, s) - channel(c, s)))
            out |= UInt32(v) << s
        }
        return out
    }

    private static func clampAddSubtractHalf(_ a: UInt32, _ b: UInt32) -> UInt32 {
        var out: UInt32 = 0
        for s in stride(from: UInt32(0), through: 24, by: 8) {
            let ca = channel(a, s), cb = channel(b, s)
            let v = Swift.max(0, Swift.min(255, ca + (ca - cb) / 2))
            out |= UInt32(v) << s
        }
        return out
    }

    /// Prediction of pixel `i` (not on the top row or left column) with `mode`.
    @inline(__always) private static func prediction(
        _ px: [UInt32], _ i: Int, _ width: Int, _ mode: Int
    ) -> UInt32 {
        let l = px[i - 1], t = px[i - width], tl = px[i - width - 1]
        // For the rightmost column this is the leftmost pixel of the current row, as the spec says.
        let tr = px[i - width + 1]
        switch mode {
        case 0: return 0xFF00_0000
        case 1: return l
        case 2: return t
        case 3: return tr
        case 4: return tl
        case 5: return average2(average2(l, tr), t)
        case 6: return average2(l, tl)
        case 7: return average2(l, t)
        case 8: return average2(tl, t)
        case 9: return average2(t, tr)
        case 10: return average2(average2(l, tl), average2(t, tr))
        case 11: return select(l, t, tl)
        case 12: return clampAddSubtractFull(l, t, tl)
        default: return clampAddSubtractHalf(average2(l, t), tl)
        }
    }

    @inline(__always) private static func sub(_ a: UInt32, _ b: UInt32) -> UInt32 {
        // Per-channel (a - b) mod 256.
        let ag = (a | 0x00FF_00FF) &- (b & 0xFF00_FF00)
        let rb = (a | 0xFF00_FF00) &- (b & 0x00FF_00FF)
        return (ag & 0xFF00_FF00) | (rb & 0x00FF_00FF)
    }

    @inline(__always) private static func cost(_ r: UInt32) -> Int {
        var c = 0
        for s in stride(from: UInt32(0), through: 24, by: 8) {
            let v = Int((r >> s) & 0xFF)
            c += Swift.min(v, 256 - v)
        }
        return c
    }

    /// Residual image, per-block modes, and the mode image size.
    private static func predict(
        _ px: [UInt32], width: Int, height: Int, sizeBits: Int
    ) -> (residuals: [UInt32], modes: [UInt8], bw: Int, bh: Int) {
        let bs = 1 << sizeBits
        let bw = (width + bs - 1) >> sizeBits, bh = (height + bs - 1) >> sizeBits
        var modes = [UInt8](repeating: 1, count: bw * bh)
        for by in 0..<bh {
            for bx in 0..<bw {
                var best = 0, bestCost = Int.max
                for mode in 0..<14 {
                    var c = 0
                    for y in (by * bs)..<Swift.min(height, by * bs + bs) where y > 0 {
                        for x in (bx * bs)..<Swift.min(width, bx * bs + bs) where x > 0 {
                            let i = y * width + x
                            c += cost(sub(px[i], prediction(px, i, width, mode)))
                        }
                    }
                    if c < bestCost { bestCost = c; best = mode }
                }
                modes[by * bw + bx] = UInt8(best)
            }
        }
        var res = [UInt32](repeating: 0, count: px.count)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x
                let pred: UInt32
                if y == 0 {
                    pred = x == 0 ? 0xFF00_0000 : px[i - 1]
                } else if x == 0 {
                    pred = px[i - width]
                } else {
                    pred = prediction(px, i, width, Int(modes[(y >> sizeBits) * bw + (x >> sizeBits)]))
                }
                res[i] = sub(px[i], pred)
            }
        }
        return (res, modes, bw, bh)
    }

    // MARK: - Entropy-coded image

    private enum Token {
        case literal(UInt32)
        case copy(length: Int, distanceCode: Int)
    }

    private static let maxCopy = 4096
    private static let minCopy = 3

    private static func tokenize(_ px: [UInt32], width: Int, useBackRefs: Bool) -> [Token] {
        guard useBackRefs else { return px.map { .literal($0) } }
        var tokens = [Token]()
        tokens.reserveCapacity(px.count)
        var i = 0
        let n = px.count
        while i < n {
            // Candidates: the pixel to the left (distance code 2 → (1, 0)) and above (1 → (0, 1)).
            var bestLen = 0, bestCode = 0
            for (dist, code) in [(1, 2), (width, 1)] where dist <= i {
                var len = 0
                while len < maxCopy, i + len < n, px[i + len] == px[i + len - dist] { len += 1 }
                if len > bestLen { bestLen = len; bestCode = code }
            }
            if bestLen >= minCopy {
                tokens.append(.copy(length: bestLen, distanceCode: bestCode))
                i += bestLen
            } else {
                tokens.append(.literal(px[i]))
                i += 1
            }
        }
        return tokens
    }

    /// (prefix code, extra bit count, extra bits) for a length or distance code value ≥ 1.
    @inline(__always) private static func prefixEncode(_ value: Int) -> (Int, Int, UInt32) {
        let d = value - 1
        if d < 4 { return (d, 0, 0) }
        let h = Int.bitWidth - 1 - d.leadingZeroBitCount
        let second = (d >> (h - 1)) & 1
        let extraBits = h - 1
        return (2 * h + second, extraBits, UInt32(d & ((1 << extraBits) - 1)))
    }

    private static func writeImage(
        _ w: inout BitWriter, _ px: [UInt32], width: Int, height: Int, isMain: Bool, useBackRefs: Bool
    ) {
        w.write(0, 1)                          // no color cache
        if isMain { w.write(0, 1) }            // no meta prefix codes

        let tokens = tokenize(px, width: width, useBackRefs: useBackRefs)
        let alphabet = [256 + 24, 256, 256, 256, 40]
        var hist = alphabet.map { [Int](repeating: 0, count: $0) }
        for t in tokens {
            switch t {
            case .literal(let p):
                hist[0][Int((p >> 8) & 0xFF)] += 1
                hist[1][Int((p >> 16) & 0xFF)] += 1
                hist[2][Int(p & 0xFF)] += 1
                hist[3][Int(p >> 24)] += 1
            case .copy(let len, let dc):
                hist[0][256 + prefixEncode(len).0] += 1
                hist[4][prefixEncode(dc).0] += 1
            }
        }
        let codes = hist.map { PrefixCode(counts: $0, maxLength: 15) }
        for c in codes { c.writeDefinition(&w) }

        for t in tokens {
            switch t {
            case .literal(let p):
                codes[0].write(&w, Int((p >> 8) & 0xFF))
                codes[1].write(&w, Int((p >> 16) & 0xFF))
                codes[2].write(&w, Int(p & 0xFF))
                codes[3].write(&w, Int(p >> 24))
            case .copy(let len, let dc):
                let (lp, lb, le) = prefixEncode(len)
                codes[0].write(&w, 256 + lp)
                w.write(le, lb)
                let (dp, db, de) = prefixEncode(dc)
                codes[4].write(&w, dp)
                w.write(de, db)
            }
        }
    }
}

// MARK: - Prefix codes

/// Canonical prefix code for one alphabet, length-limited, with VP8L's code definition encoding.
private struct PrefixCode {
    var lengths: [Int]
    var codes: [UInt32]      // bit-reversed, ready for LSB-first writing
    /// Symbols with nonzero count (the simple-code form when at most two, all < 256).
    var used: [Int]

    init(counts: [Int], maxLength: Int) {
        used = counts.indices.filter { counts[$0] > 0 }
        if used.count <= 1 {
            // One (or no) symbol costs zero bits per occurrence.
            lengths = [Int](repeating: 0, count: counts.count)
            codes = [UInt32](repeating: 0, count: counts.count)
            if used.isEmpty { used = [0] }
            // A symbol ≥ 256 can't use the simple form; a normal code with one length-1 symbol is
            // also read as zero-bit.
            lengths[used[0]] = 1
            return
        }
        lengths = PrefixCode.buildLengths(counts, maxLength: maxLength)
        codes = PrefixCode.canonicalCodes(lengths)
    }

    var isSimple: Bool { used.count <= 2 && used.allSatisfy { $0 < 256 } }

    func write(_ w: inout BitWriter, _ symbol: Int) {
        if used.count == 1 { return }
        if isSimple {
            // Simple code with two symbols: one bit each, in the order they were listed.
            w.write(symbol == used[0] ? 0 : 1, 1)
            return
        }
        w.write(codes[symbol], lengths[symbol])
    }

    func writeDefinition(_ w: inout BitWriter) {
        if isSimple {
            w.write(1, 1)                                  // simple code
            w.write(UInt32(used.count - 1), 1)
            if used[0] < 2 {
                w.write(0, 1); w.write(UInt32(used[0]), 1)
            } else {
                w.write(1, 1); w.write(UInt32(used[0]), 8)
            }
            if used.count == 2 { w.write(UInt32(used[1]), 8) }
            return
        }
        w.write(0, 1)                                      // normal code

        // Code lengths as a stream of code-length symbols: 0...15 literal, 17/18 runs of zeros.
        var syms: [(sym: Int, extraBits: Int, extra: UInt32)] = []
        var i = 0
        while i < lengths.count {
            if lengths[i] == 0 {
                var run = 0
                while i + run < lengths.count, lengths[i + run] == 0, run < 138 { run += 1 }
                if run >= 11 {
                    syms.append((18, 7, UInt32(run - 11)))
                } else if run >= 3 {
                    syms.append((17, 3, UInt32(run - 3)))
                } else {
                    for _ in 0..<run { syms.append((0, 0, 0)) }
                }
                i += run
            } else {
                syms.append((lengths[i], 0, 0))
                i += 1
            }
        }
        var clCounts = [Int](repeating: 0, count: 19)
        for s in syms { clCounts[s.sym] += 1 }
        // A code with one used symbol would be read as zero-bit; keep at least two.
        if clCounts.filter({ $0 > 0 }).count < 2 {
            clCounts[clCounts[0] > 0 ? 1 : 0] += 1
        }
        let clLengths = PrefixCode.buildLengths(clCounts, maxLength: 7)
        let clCodes = PrefixCode.canonicalCodes(clLengths)

        let order = [17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        var numCodes = 19
        while numCodes > 4, clLengths[order[numCodes - 1]] == 0 { numCodes -= 1 }
        w.write(UInt32(numCodes - 4), 4)
        for k in 0..<numCodes { w.write(UInt32(clLengths[order[k]]), 3) }
        w.write(0, 1)                                      // max_symbol = alphabet size
        for s in syms {
            w.write(clCodes[s.sym], clLengths[s.sym])
            w.write(s.extra, s.extraBits)
        }
    }

    /// Huffman code lengths limited to `maxLength`: when the tree is too deep, small counts are
    /// raised (doubling floor) and the tree rebuilt, which keeps it complete.
    static func buildLengths(_ counts: [Int], maxLength: Int) -> [Int] {
        var floor = 1
        while true {
            let adjusted = counts.map { $0 > 0 ? Swift.max($0, floor) : 0 }
            let lengths = huffmanLengths(adjusted)
            if (lengths.max() ?? 0) <= maxLength { return lengths }
            floor *= 2
        }
    }

    private static func huffmanLengths(_ counts: [Int]) -> [Int] {
        // Nodes: leaves first, then internal nodes; parent links give depths.
        var weight = [Int](), parent = [Int]()
        var leafOf = [Int](repeating: -1, count: counts.count)
        for (s, c) in counts.enumerated() where c > 0 {
            leafOf[s] = weight.count
            weight.append(c); parent.append(-1)
        }
        var lengths = [Int](repeating: 0, count: counts.count)
        if weight.count == 1 {
            lengths[leafOf.firstIndex { $0 >= 0 }!] = 1
            return lengths
        }
        // Two sorted queues (leaves by weight, internal nodes in creation order) build the tree in
        // O(n log n).
        let leaves = weight.indices.sorted { weight[$0] != weight[$1] ? weight[$0] < weight[$1] : $0 < $1 }
        var li = 0
        var merged = [Int](), ii = 0
        func pop() -> Int {
            if li < leaves.count, ii >= merged.count || weight[leaves[li]] <= weight[merged[ii]] {
                li += 1; return leaves[li - 1]
            }
            ii += 1; return merged[ii - 1]
        }
        while (leaves.count - li) + (merged.count - ii) > 1 {
            let a = pop(), b = pop()
            let node = weight.count
            weight.append(weight[a] + weight[b]); parent.append(-1)
            parent[a] = node; parent[b] = node
            merged.append(node)
        }
        var depth = [Int](repeating: 0, count: weight.count)
        for node in stride(from: weight.count - 2, through: 0, by: -1) {
            depth[node] = depth[parent[node]] + 1
        }
        for (s, leaf) in leafOf.enumerated() where leaf >= 0 { lengths[s] = depth[leaf] }
        return lengths
    }

    static func canonicalCodes(_ lengths: [Int]) -> [UInt32] {
        let maxLen = lengths.max() ?? 0
        var blCount = [Int](repeating: 0, count: maxLen + 1)
        for l in lengths where l > 0 { blCount[l] += 1 }
        var next = [Int](repeating: 0, count: maxLen + 2)
        var code = 0
        if maxLen >= 1 {
            for len in 1...maxLen {
                code = (code + blCount[len - 1]) << 1
                next[len] = code
            }
        }
        var codes = [UInt32](repeating: 0, count: lengths.count)
        for (s, l) in lengths.enumerated() where l > 0 {
            var c = next[l], rev: UInt32 = 0
            next[l] += 1
            for _ in 0..<l { rev = (rev << 1) | UInt32(c & 1); c >>= 1 }
            codes[s] = rev
        }
        return codes
    }
}

/// LSB-first bit writer.
private struct BitWriter {
    private var bytes = [UInt8]()
    private var acc: UInt64 = 0
    private var bits = 0

    mutating func write(_ value: UInt32, _ count: Int) {
        guard count > 0 else { return }
        acc |= UInt64(value & UInt32(truncatingIfNeeded: (UInt64(1) << UInt64(count)) - 1)) << UInt64(bits)
        bits += count
        while bits >= 8 {
            bytes.append(UInt8(truncatingIfNeeded: acc))
            acc >>= 8
            bits -= 8
        }
    }

    mutating func finish() -> Data {
        if bits > 0 { bytes.append(UInt8(truncatingIfNeeded: acc)) }
        acc = 0; bits = 0
        return Data(bytes)
    }
}
