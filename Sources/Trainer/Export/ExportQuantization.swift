// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation
import MLX
import MLXFast

/// Clamps to [0, maxValue] with NaN → 0, so integer conversions never trap.
@inline(__always) func clampForPacking(_ v: Float, _ maxValue: Float) -> Float {
    v.isNaN ? 0 : Swift.max(0, Swift.min(maxValue, v))
}

/// Z-order of 3D points, as splat-transform orders splats before compressing: positions are
/// quantized to 10 bits per axis over the extent of the given points, radix-sorted by the 30-bit
/// code (stable), and runs of more than 256 points sharing a code are re-sorted within their own
/// extent.
enum MortonOrder {

    /// `indices` reordered by Morton code of `positions[idx*3 ..< idx*3+3]`.
    static func sort(_ indices: [Int], positions: [Float]) -> [Int] {
        var idx = indices
        idx.withUnsafeMutableBufferPointer { refine($0, positions) }
        return idx
    }

    private static func refine(_ idx: UnsafeMutableBufferPointer<Int>, _ p: [Float]) {
        let n = idx.count
        guard n > 0 else { return }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        for g in idx {
            let v = SIMD3(p[g * 3], p[g * 3 + 1], p[g * 3 + 2])
            lo = pointwiseMin(lo, v); hi = pointwiseMax(hi, v)
        }
        let len = hi - lo
        guard len.x.isFinite, len.y.isFinite, len.z.isFinite, len != .zero else { return }
        let mul = SIMD3<Float>(len.x == 0 ? 0 : 1024 / len.x,
                               len.y == 0 ? 0 : 1024 / len.y,
                               len.z == 0 ? 0 : 1024 / len.z)

        var key = [UInt32](repeating: 0, count: n)
        for (i, g) in idx.enumerated() {
            let q = (SIMD3(p[g * 3], p[g * 3 + 1], p[g * 3 + 2]) - lo) * mul
            key[i] = encode(cell(q.x), cell(q.y), cell(q.z))
        }

        // LSD radix sort, three stable passes of 10 bits.
        var srcIdx = Array(idx), srcKey = key
        var dstIdx = srcIdx, dstKey = srcKey
        var counts = [Int](repeating: 0, count: 1024)
        for shift in stride(from: 0, to: 30, by: 10) {
            for d in 0..<1024 { counts[d] = 0 }
            for k in srcKey { counts[Int((k >> UInt32(shift)) & 1023)] += 1 }
            var sum = 0
            for d in 0..<1024 { let c = counts[d]; counts[d] = sum; sum += c }
            for i in 0..<n {
                let b = Int((srcKey[i] >> UInt32(shift)) & 1023)
                dstIdx[counts[b]] = srcIdx[i]; dstKey[counts[b]] = srcKey[i]
                counts[b] += 1
            }
            swap(&srcIdx, &dstIdx); swap(&srcKey, &dstKey)
        }
        for i in 0..<n { idx[i] = srcIdx[i] }

        var start = 0
        while start < n {
            var end = start + 1
            while end < n, srcKey[end] == srcKey[start] { end += 1 }
            if end - start > 256 { refine(UnsafeMutableBufferPointer(rebasing: idx[start..<end]), p) }
            start = end
        }
    }

    @inline(__always) private static func cell(_ v: Float) -> UInt32 {
        UInt32(clampForPacking(v, 1023))
    }

    @inline(__always) private static func encode(_ x: UInt32, _ y: UInt32, _ z: UInt32) -> UInt32 {
        func part1By2(_ v: UInt32) -> UInt32 {
            var v = v & 0x0000_03FF
            v = (v ^ (v << 16)) & 0xFF00_00FF
            v = (v ^ (v << 8)) & 0x0300_F00F
            v = (v ^ (v << 4)) & 0x030C_30C3
            v = (v ^ (v << 2)) & 0x0924_9249
            return v
        }
        return (part1By2(z) << 2) + (part1By2(y) << 1) + part1By2(x)
    }
}

/// splat-transform's 1D codebook quantizer: pools the columns, bins the sorted finite values into
/// up to 1024 bins placed by an IQR-adaptive blend of quantile and uniform position, then finds the
/// `k` centroids minimizing count^alpha-weighted SSE by dynamic programming. ±inf get their own end
/// centroids 20 units outside the finite range.
enum Quantize1D {

    struct Result {
        /// `k` ascending centroids.
        var centroids: [Float]
        /// Per column, the codebook index of each value.
        var labels: [[UInt8]]
    }

    static func quantize(_ columns: [[Float]], k: Int = 256, alpha: Double = 0.5) -> Result {
        let rows = columns.first?.count ?? 0
        let data = columns.flatMap { $0 }
        let total = data.count
        guard total > 0 else {
            return Result(centroids: [Float](repeating: 0, count: k),
                          labels: columns.map { [UInt8](repeating: 0, count: $0.count) })
        }

        var finite = data.filter { $0.isFinite }
        finite.sort()
        let hasNegInf = data.contains(-.infinity), hasPosInf = data.contains(.infinity)
        let loSlots = hasNegInf ? 1 : 0, hiSlots = hasPosInf ? 1 : 0
        let negInfCentroid = (finite.first ?? 0) - 20
        let posInfCentroid = (finite.last ?? 0) + 20
        let vMin = finite.first ?? 0, vMax = finite.last ?? 0

        if loSlots == 0, hiSlots == 0, vMax - vMin < 1e-20 {
            return Result(centroids: [Float](repeating: vMin, count: k),
                          labels: columns.map { [UInt8](repeating: 0, count: $0.count) })
        }

        let kFinite = Swift.max(0, k - loSlots - hiSlots)
        let values: [Float]
        if finite.isEmpty || kFinite == 0 {
            values = []
        } else if vMax - vMin < 1e-20 {
            values = [vMin]
        } else {
            values = quantizeFinite(finite, kTarget: kFinite, alpha: alpha)
        }

        // [-inf sentinel][finite centroids, padded with the last][+inf sentinel]
        let pad = values.last ?? (hasNegInf ? negInfCentroid : posInfCentroid)
        var centroids = [Float](repeating: pad, count: k)
        for (i, v) in values.enumerated() { centroids[loSlots + i] = v }
        if hasNegInf { centroids[0] = negInfCentroid }
        if hasPosInf { centroids[k - 1] = posInfCentroid }

        var labels = [[UInt8]]()
        for c in 0..<columns.count {
            var out = [UInt8](repeating: 0, count: rows)
            for r in 0..<rows {
                let v = data[c * rows + r]
                var lo = 0, hi = k - 1
                while lo < hi {
                    let mid = (lo + hi) >> 1
                    if v < (centroids[mid] + centroids[mid + 1]) * 0.5 { hi = mid } else { lo = mid + 1 }
                }
                out[r] = UInt8(lo)
            }
            labels.append(out)
        }
        return Result(centroids: centroids, labels: labels)
    }

    private static func quantizeFinite(_ sorted: [Float], kTarget: Int, alpha: Double) -> [Float] {
        let n = sorted.count
        let vMin = Double(sorted[0]), vMax = Double(sorted[n - 1])
        let h = Swift.min(1024, n)
        let range = vMax - vMin
        let iqr = Double(sorted[n * 3 / 4]) - Double(sorted[n / 4])
        let beta = Swift.max(0.5, Swift.min(0.999, 1 - iqr / range))

        var counts = [Double](repeating: 0, count: h), sums = [Double](repeating: 0, count: h)
        for i in 0..<n {
            let v = Double(sorted[i])
            let pos = beta * Double(i) / Double(n) + (1 - beta) * (v - vMin) / range
            let bin = Swift.min(h - 1, Int((Double(h) * pos).rounded(.down)))
            counts[bin] += 1; sums[bin] += v
        }
        var centers = [Double](repeating: 0, count: h)
        var prefW = [Double](repeating: 0, count: h + 1)
        var prefWX = prefW, prefWXX = prefW
        for i in 0..<h {
            centers[i] = counts[i] > 0 ? sums[i] / counts[i] : vMin + (Double(i) + 0.5) / Double(h) * range
            let w = counts[i] > 0 ? pow(counts[i], alpha) : 0
            prefW[i + 1] = prefW[i] + w
            prefWX[i + 1] = prefWX[i] + w * centers[i]
            prefWXX[i + 1] = prefWXX[i] + w * centers[i] * centers[i]
        }
        @inline(__always) func cost(_ a: Int, _ b: Int) -> Double {
            let w = prefW[b + 1] - prefW[a]
            if w <= 0 { return 0 }
            let wx = prefWX[b + 1] - prefWX[a]
            return prefWXX[b + 1] - prefWXX[a] - wx * wx / w
        }
        func mean(_ a: Int, _ b: Int) -> Double {
            let w = prefW[b + 1] - prefW[a]
            if w <= 0 { return (centers[a] + centers[b]) * 0.5 }
            return (prefWX[b + 1] - prefWX[a]) / w
        }

        let effectiveK = Swift.min(kTarget, counts.reduce(0) { $0 + ($1 > 0 ? 1 : 0) })
        var prev = (0..<h).map { cost(0, $0) }
        var curr = [Double](repeating: 0, count: h)
        var split = [[Int32]](repeating: [], count: effectiveK + 1)
        split[1] = [Int32](repeating: -1, count: h)
        if effectiveK >= 2 {
            for m in 2...effectiveK {
                var splitM = [Int32](repeating: 0, count: h)
                for j in 0..<h { curr[j] = 1e30 }
                for j in (m - 1)..<h {
                    var best = 1e30, bestS = m - 2
                    for s in (m - 2)..<j {
                        let c = prev[s] + cost(s + 1, j)
                        if c < best { best = c; bestS = s }
                    }
                    curr[j] = best
                    splitM[j] = Int32(bestS)
                }
                split[m] = splitM
                swap(&prev, &curr)
            }
        }

        var out = [Float](repeating: 0, count: effectiveK)
        var j = h - 1
        for m in stride(from: effectiveK, through: 1, by: -1) {
            let s = m > 1 ? Int(split[m][j]) : -1
            out[m - 1] = Float(mean(s + 1, j))
            j = s
        }
        return out.sorted()
    }
}

/// Lloyd k-means over interleaved points, with the nearest-centroid search on the GPU. Init
/// picks `k` distinct random points; an empty cluster is re-seeded from a random point.
enum KMeans {

    static func cluster(
        _ points: [Float], dims: Int, k: Int, iterations: Int,
        seed: UInt64 = 0x5EED, onIteration: ((Int) -> Void)? = nil
    ) -> (centroids: [Float], labels: [UInt32]) {
        let n = points.count / dims
        if n <= k {
            return (points, (0..<n).map { UInt32($0) })
        }
        var rng = SplitMix64(seed: seed)
        var chosen = Set<Int>()
        // Floyd's sampling of k distinct indices
        for j in (n - k)..<n {
            let t = Int(rng.next() % UInt64(j + 1))
            chosen.insert(chosen.contains(t) ? j : t)
        }
        var centroids = [Float]()
        centroids.reserveCapacity(k * dims)
        for i in chosen.sorted() { centroids += points[(i * dims)..<(i * dims + dims)] }

        let pointsArr = MLXArray(points, [n, dims])
        var labels = [UInt32](repeating: 0, count: n)
        for step in 0..<iterations {
            labels = assign(pointsArr, n: n, centroids: centroids, k: k, dims: dims)
            var sums = [Double](repeating: 0, count: k * dims)
            var counts = [Int](repeating: 0, count: k)
            for r in 0..<n {
                let c = Int(labels[r])
                counts[c] += 1
                for d in 0..<dims { sums[c * dims + d] += Double(points[r * dims + d]) }
            }
            for c in 0..<k {
                if counts[c] == 0 {
                    let src = Int(rng.next() % UInt64(n))
                    for d in 0..<dims { centroids[c * dims + d] = points[src * dims + d] }
                } else {
                    for d in 0..<dims { centroids[c * dims + d] = Float(sums[c * dims + d] / Double(counts[c])) }
                }
            }
            onIteration?(step + 1)
        }
        // Final labels against the final centroids.
        labels = assign(pointsArr, n: n, centroids: centroids, k: k, dims: dims)
        return (centroids, labels)
    }

    /// Nearest centroid per point. One GPU thread per point scans every centroid in exact float32
    /// (a [rows, k] distance matrix from matmul would be memory-bound at these few dims).
    private static let nearestKernel = MLXFast.metalKernel(
        name: "kmeans_nearest",
        inputNames: ["points", "centroids", "params"],
        outputNames: ["labels"],
        source: """
            uint i = thread_position_in_grid.x;
            uint n = params[0], k = params[1];
            if (i >= n) return;
            float x[DIMS];
            for (uint d = 0; d < DIMS; ++d) x[d] = points[i * DIMS + d];
            float best = INFINITY;
            uint bestIdx = 0;
            for (uint c = 0; c < k; ++c) {
                float dist = 0;
                for (uint d = 0; d < DIMS; ++d) {
                    float t = x[d] - centroids[c * DIMS + d];
                    dist += t * t;
                }
                if (dist < best) { best = dist; bestIdx = c; }
            }
            labels[i] = bestIdx;
            """
    )

    /// Labels in row blocks, so no single GPU command runs long.
    private static func assign(
        _ points: MLXArray, n: Int, centroids: [Float], k: Int, dims: Int
    ) -> [UInt32] {
        let c = MLXArray(centroids, [k * dims])
        let block = 1 << 16
        var out = [UInt32]()
        out.reserveCapacity(n)
        var start = 0
        while start < n {
            let rows = Swift.min(block, n - start)
            let x = points[start..<(start + rows)].reshaped([rows * dims])
            let labels = nearestKernel(
                [x, c, MLXArray([UInt32(rows), UInt32(k)])],
                template: [("DIMS", dims)],
                grid: (((rows + 255) / 256) * 256, 1, 1),
                threadGroup: (256, 1, 1),
                outputShapes: [[rows]],
                outputDTypes: [.uint32]
            )[0]
            out += labels.asArray(UInt32.self)
            start += rows
        }
        return out
    }
}
