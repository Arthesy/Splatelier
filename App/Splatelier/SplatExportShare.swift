// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// The "Gaussian splats" entry of the viewer's Share menu: the trained PLY as it is, or converted
// to a smaller / other format when picked.

import Foundation
import SplatViewerKit
import Trainer

enum SplatExportShare {
    /// Formats offered besides the PLY itself, in menu order.
    private static let convertedFormats: [SplatExportFormat] = [.compressedPly, .spz, .splat, .glb, .sog]

    /// The Share entry for the capture's GS file. Only a PLY can be converted (it is read back in
    /// full); anything else, such as an imported .spz, is shared as it is.
    static func item(for url: URL, title: String = "Gaussian splats") -> SplatViewerShareItem {
        guard url.pathExtension.lowercased() == "ply" else {
            return SplatViewerShareItem(title: title, systemImage: "cube.fill", url: url)
        }
        var options = [SplatViewerShareItem.Option(title: "PLY (as trained)", url: url)]
        for format in convertedFormats {
            options.append(SplatViewerShareItem.Option(title: format.displayName) {
                try await export(url, as: format)
            })
        }
        return SplatViewerShareItem(title: title, systemImage: "cube.fill", options: options)
    }

    /// Converts on a background thread and returns the file in a fresh temporary directory (the
    /// share sheet reads it from there). Earlier exports are deleted first.
    private static func export(_ plyURL: URL, as format: SplatExportFormat) async throws -> URL {
        try await Task.detached(priority: .userInitiated) {
            let fm = FileManager.default
            let root = fm.temporaryDirectory.appendingPathComponent("splat-export", isDirectory: true)
            try? fm.removeItem(at: root)
            let dir = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            let out = format.url(replacingExtensionOf: dir.appendingPathComponent(plyURL.lastPathComponent))

            // readPLY already drops the zero padding above the trained SH degree.
            let table = try SplatTable.readPLY(url: plyURL)
            try SplatExporter.write(table, format: format, to: out)
            return out
        }.value
    }
}
