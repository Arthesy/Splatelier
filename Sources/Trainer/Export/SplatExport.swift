// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors

import Foundation

/// Output formats for trained splats. The file layouts follow PlayCanvas splat-transform (MIT)
/// where it defines them (compressed PLY, SOG, GLB).
public enum SplatExportFormat: String, CaseIterable, Sendable, Identifiable {
    /// Standard 3DGS PLY, uncompressed float32.
    case ply
    /// PlayCanvas / SuperSplat compressed PLY: 256-splat chunks quantized to 16 bytes per splat.
    case compressedPly
    /// Niantic SPZ (gzip-compressed fixed-point).
    case spz
    /// antimatter15 `.splat`: 32 bytes per splat, no higher-order SH.
    case splat
    /// glTF binary with the KHR_gaussian_splatting extension.
    case glb
    /// PlayCanvas SOG: WebP textures plus meta.json, bundled in a zip.
    case sog
    /// OpenUSD ASCII with a ParticleField3DGaussianSplat prim (OpenUSD v26.03).
    case usda

    public var id: String { rawValue }

    /// File extension including any compound part (no leading dot).
    public var fileExtension: String {
        switch self {
        case .ply: return "ply"
        case .compressedPly: return "compressed.ply"
        case .spz: return "spz"
        case .splat: return "splat"
        case .glb: return "glb"
        case .sog: return "sog"
        case .usda: return "usda"
        }
    }

    public var displayName: String {
        switch self {
        case .ply: return "PLY"
        case .compressedPly: return "Compressed PLY"
        case .spz: return "SPZ"
        case .splat: return ".splat"
        case .glb: return "glTF (GLB)"
        case .sog: return "SOG"
        case .usda: return "OpenUSD"
        }
    }

    /// Whether the in-app viewer (MetalSplatter's SplatIO) opens this format.
    public var viewerReadable: Bool {
        switch self {
        case .ply, .spz, .splat: return true
        case .compressedPly, .glb, .sog, .usda: return false
        }
    }

    /// The format implied by `url`'s name (`.compressed.ply` before `.ply`), or nil.
    public init?(url: URL) {
        let name = url.lastPathComponent.lowercased()
        // Longest extension first so "compressed.ply" wins over "ply".
        let byLength = Self.allCases.sorted { $0.fileExtension.count > $1.fileExtension.count }
        guard let f = byLength.first(where: { name.hasSuffix("." + $0.fileExtension) }) else {
            return nil
        }
        self = f
    }

    /// `url` with its (possibly compound) splat extension replaced by this format's.
    public func url(replacingExtensionOf url: URL) -> URL {
        var base = url
        if let current = SplatExportFormat(url: url) {
            for _ in current.fileExtension.split(separator: ".") { base.deletePathExtension() }
        } else {
            base.deletePathExtension()
        }
        return base.appendingPathExtension(fileExtension)
    }
}

public enum SplatExporter {

    /// Writes `table` to `url` in `format`. For the formats our viewer opens (PLY, SPZ, .splat) the
    /// `<base>.background.json` sidecar is written (or removed) next to it. Every writer keeps the
    /// background rows at the tail.
    public static func write(
        _ table: SplatTable, format: SplatExportFormat, to url: URL,
        options: SOGWriter.Options = SOGWriter.Options()
    ) throws {
        switch format {
        case .ply: try PLYWriter.write(table, to: url)
        case .compressedPly: try CompressedPLYWriter.write(table, to: url)
        case .spz: try SPZWriter.write(table, to: url)
        case .splat: try DotSplatWriter.write(table, to: url)
        case .glb: try GLBWriter.write(table, to: url)
        case .sog: try SOGWriter.write(table, to: url, options: options)
        case .usda: try USDWriter.write(table, to: url)
        }
        if format.viewerReadable {
            try PLYWriter.writeBackgroundSidecar(splatCount: table.backgroundCount, siblingOf: url)
        }
    }
}
