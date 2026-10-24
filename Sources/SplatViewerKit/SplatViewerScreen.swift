// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Splatelier contributors
//
// Public entry point. SwiftUI screen that shows a trained PLY (any format SplatIO can read) with an
// orbit camera. The seed point cloud, capture cameras and mesh (OBJ) can be overlaid in the same
// scene and toggled from the "Display" menu, where the trained background can be hidden and the mesh
// switched to shaded. The "Measure" button at the bottom enters measure mode to measure the distance
// between 2 tapped points. Double tap (double click on macOS) zooms in, making that point the center
// of rotation.

import SwiftUI

#if os(iOS) || os(macOS)

import MetalKit

#if os(iOS)
import UIKit
#endif

#if os(macOS)
private typealias SplatViewRepresentable = NSViewRepresentable
#else
private typealias SplatViewRepresentable = UIViewRepresentable
#endif

public struct SplatViewerScreen: View {
    private let url: URL?
    private let seedURL: URL?
    private let meshURL: URL?
    private let shareItems: [SplatViewerShareItem]
    /// Handler for the "Train more" button. nil hides the button (the screen has no training
    /// session to continue).
    private let onTrainMore: (() -> Void)?
    /// What the confirmation before "Train more" tells the user.
    private let trainMoreMessage: String
    /// Receiver of rename notifications. nil hides the editing UI (saving the sidecar is the app's
    /// job).
    private let onRename: ((String) -> Void)?
    /// Held in @State so edits show immediately.
    @State private var displayName: String
    @State private var showRename = false
    @State private var showHint = false
    @State private var showTrainMoreConfirm = false
    @State private var renameText = ""
    @StateObject private var model = SplatViewerModel()
    #if os(iOS)
    @StateObject private var sharer = SplatViewerSharePreparer()
    #endif

#if os(macOS)
    // Values at gesture start. drag / pinch are handled as absolute amounts from the start point
    // (macOS only).
    @State private var dragBase: SIMD2<Float>?
    @State private var zoomBase: Float?
    /// Size of the MTKView, for normalizing click positions.
    @State private var metalViewSize: CGSize = .zero
#endif

    /// - Parameters:
    ///   - url: The GS PLY. If nil, only the mesh (and seed) are shown (e.g. checking the mesh
    ///     before training).
    ///   - seedURL: PLY of the seed point cloud + capture-camera markers. If given, merged into
    ///     the same scene and toggleable.
    ///   - meshURL: OBJ of a vertex-colored mesh. If given, overlaid in the same scene and
    ///     toggleable.
    ///   - name: Display name (file name if nil).
    ///   - shareItems: Files offered by the Share button, one entry each. Empty hides the button.
    ///   - onTrainMore: Called when the user confirms "Train more". nil hides the button.
    ///   - trainMoreMessage: Shown in the confirmation (what will happen, how long it takes).
    public init(
        url: URL?,
        seedURL: URL? = nil,
        meshURL: URL? = nil,
        name: String? = nil,
        shareItems: [SplatViewerShareItem] = [],
        onTrainMore: (() -> Void)? = nil,
        trainMoreMessage: String = "Training continues from this result.",
        onRename: ((String) -> Void)? = nil
    ) {
        self.url = url
        self.seedURL = seedURL
        self.meshURL = meshURL
        self.shareItems = shareItems
        self.onTrainMore = onTrainMore
        self.trainMoreMessage = trainMoreMessage
        self.onRename = onRename
        _displayName = State(initialValue: name ?? Self.fileName(url ?? meshURL))
    }

    private static func fileName(_ url: URL?) -> String {
        url?.deletingPathExtension().lastPathComponent ?? "Viewer"
    }

    public var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            SplatMetalView(url: url, seedURL: seedURL, meshURL: meshURL, model: model)
                .ignoresSafeArea()
            #if os(macOS)
                .gesture(orbitGesture)
                .simultaneousGesture(zoomGesture)
                .simultaneousGesture(measureTapGesture)
                .simultaneousGesture(focusTapGesture)
                .onGeometryChange(for: CGSize.self) { $0.size } action: { metalViewSize = $0 }
            #endif

            SplatMeasureOverlay(projection: model.measureProjection)
                .ignoresSafeArea()

            overlay
            #if os(iOS)
            preparingOverlay
            #endif
        }
        #if os(iOS)
        .sheet(item: $sharer.ready) { file in SplatViewerShareSheet(url: file.url) }
        .alert("Couldn't share", isPresented: Binding(
            get: { sharer.errorMessage != nil }, set: { if !$0 { sharer.errorMessage = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(sharer.errorMessage ?? "")
        }
        #endif
        .navigationTitle(displayName)
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(.visible, for: .navigationBar)
        #endif
        .toolbar {
            ToolbarItemGroup(placement: toolbarPlacement) {
                hintButton
                if !shareItems.isEmpty { shareMenu }
                if onTrainMore != nil {
                    // Training is long and replaces this result, so ask before starting it.
                    Button { showTrainMoreConfirm = true } label: {
                        Label("Train more", systemImage: "plus.circle")
                    }
                }
                if onRename != nil {
                    Button {
                        renameText = displayName
                        showRename = true
                    } label: {
                        Label("Rename", systemImage: "pencil")
                    }
                }
            }
        }
        .alert("Train more?", isPresented: $showTrainMoreConfirm) {
            Button("Train more") { onTrainMore?() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(trainMoreMessage)
        }
        .alert("Rename", isPresented: $showRename) {
            TextField("Name", text: $renameText)
            Button("Save") {
                let trimmed = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
                displayName = trimmed.isEmpty ? Self.fileName(url ?? meshURL) : trimmed
                onRename?(trimmed)
            }
            Button("Cancel", role: .cancel) {}
        }
    }

    private var toolbarPlacement: ToolbarItemPlacement {
        #if os(iOS)
        .topBarTrailing
        #else
        .automatic
        #endif
    }

    /// One entry per thing to share (GS, mesh, logs). An entry with several options (the GS in
    /// different formats) is a submenu. Options that are made when picked show a progress overlay
    /// and then the share sheet.
    private var shareMenu: some View {
        Menu {
            ForEach(shareItems) { item in
                if item.options.count == 1, let option = item.options.first {
                    optionControl(option, label: Label(item.title, systemImage: item.systemImage))
                } else {
                    Menu {
                        ForEach(item.options) { option in
                            optionControl(option, label: Text(option.title))
                        }
                    } label: {
                        Label(item.title, systemImage: item.systemImage)
                    }
                }
            }
        } label: {
            Label("Share", systemImage: "square.and.arrow.up")
        }
    }

    @ViewBuilder
    private func optionControl<Content: View>(
        _ option: SplatViewerShareItem.Option, label: Content
    ) -> some View {
        switch option.source {
        case .file(let url):
            ShareLink(item: url) { label }
        case .files(let urls):
            ShareLink(items: urls) { label }
        case .prepared(let prepare):
            #if os(iOS)
            Button { sharer.prepare(title: option.title, prepare) } label: { label }
            #endif
        }
    }

    /// Opens a popover that lists the camera controls.
    private var hintButton: some View {
        Button {
            showHint = true
        } label: {
            Label("Hint", systemImage: "questionmark.circle")
        }
        .popover(isPresented: $showHint) {
            hintContent
                .presentationCompactAdaptation(.popover)
        }
    }

    private var hintContent: some View {
        VStack(alignment: .leading, spacing: 10) {
            #if os(iOS)
            hintRow("hand.point.up.left", "Rotate", "Drag with 1 finger")
            hintRow("hand.draw", "Pan", "Drag with 2 fingers")
            hintRow("arrow.up.left.and.arrow.down.right", "Zoom", "Pinch")
            hintRow("scope", "Focus", "Double-tap a point to zoom in and orbit around it")
            hintRow("ruler", "Measure", "Turn on Measure, then tap 2 points")
            #else
            hintRow("cursorarrow.motionlines", "Rotate", "Drag")
            hintRow("arrow.up.left.and.arrow.down.right", "Zoom", "Pinch or scroll")
            hintRow("scope", "Focus", "Double-click a point to zoom in and orbit around it")
            hintRow("ruler", "Measure", "Turn on Measure, then click 2 points")
            #endif
        }
        .padding(16)
        .frame(minWidth: 260)
    }

    private func hintRow(_ symbol: String, _ title: String, _ detail: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: symbol)
                .frame(width: 22)
                .foregroundStyle(Color.accentColor)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.subheadline.weight(.semibold))
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    #if os(iOS)
    /// Shown while a share option is being prepared (a format conversion can take a while).
    @ViewBuilder
    private var preparingOverlay: some View {
        if let title = sharer.preparing {
            ZStack {
                Color.black.opacity(0.45).ignoresSafeArea()
                VStack(spacing: 14) {
                    ProgressView()
                    Text("Preparing \(title)…").font(.subheadline)
                    Button("Cancel") { sharer.cancel() }
                        .buttonStyle(.bordered)
                }
                .padding(24)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            }
            .transition(.opacity)
        }
    }
    #endif

    // MARK: Overlay

    @ViewBuilder
    private var overlay: some View {
        switch model.loadState {
        case .loading:
            ProgressView("Loading…")
                .padding(16)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        case .empty:
            status(symbol: "cube.transparent", title: "No splats", detail: "This file has no Gaussians to show")
        case .failed(let msg):
            status(symbol: "exclamationmark.triangle", title: "Couldn't load", detail: msg)
        case .loaded(let count):
            // Top to bottom: measurement hint (while measuring) → Display / Measure / Reset →
            // splat / triangle counts.
            VStack {
                Spacer()
                if model.isMeasuring {
                    measureControls
                }
                bottomBar(splats: count)
                    .padding(.horizontal, 12).padding(.vertical)
            }
        }
    }

    /// Display options (layer visibility, mesh appearance) in one menu, so the bottom edge doesn't
    /// fill up with a row of toggles. Background and Shading sit in their parent layer's section and are disabled (not hidden)
    /// while that layer is off, so rows don't jump. The menu stays open after each pick, letting
    /// several layers be switched in one go.
    private var displayMenu: some View {
        Menu {
            if model.hasGS {
                Section("Gaussian splats") {
                    Toggle(isOn: $model.showGS) { Label("Splats", systemImage: "cube.fill") }
                    if model.hasBackground {
                        Toggle(isOn: $model.showBackground) { Label("Background", systemImage: "globe") }
                            .disabled(!model.showGS)
                    }
                }
            }
            if model.hasMesh {
                Section("Mesh") {
                    Toggle(isOn: $model.showMesh) { Label("Mesh", systemImage: "triangle") }
                    Picker(selection: $model.meshShaded) {
                        Label("Vertex colors", systemImage: "paintpalette").tag(false)
                        Label("Shaded", systemImage: "light.max").tag(true)
                    } label: {
                        Text("Appearance")
                    }
                    .pickerStyle(.inline)
                    .disabled(!model.showMesh)
                }
            }
            if model.hasSeed || model.hasCameraMarkers {
                Section("Capture") {
                    if model.hasSeed {
                        Toggle(isOn: $model.showSeed) { Label("Seed points", systemImage: "circle.grid.3x3.fill") }
                    }
                    if model.hasCameraMarkers {
                        Toggle(isOn: $model.showCameraMarkers) { Label("Cameras", systemImage: "camera.fill") }
                    }
                }
            }
        } label: {
            Label("Display", systemImage: "eye")
                .font(.caption)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
        }
        #if os(iOS)
        // Stay open after each pick, so several layers can be switched at once.
        .menuActionDismissBehavior(.disabled)
        // Opening upward from the bottom bar, iOS would otherwise flip the sections bottom-up.
        .menuOrder(.fixed)
        #endif
        #if os(macOS)
        .menuStyle(.button)
        .menuIndicator(.hidden)
        #endif
        .buttonStyle(.plain)
    }

    /// Bottom edge, two rows: Display, Measure and Reset centered, with the splat and triangle
    /// counts right-aligned below them. Counts never wrap into a vertical stack; they stay side by
    /// side.
    private func bottomBar(splats: Int) -> some View {
        VStack(spacing: 8) {
            trailingControls
                .frame(maxWidth: .infinity)
            counts(splats: splats)
                .frame(maxWidth: .infinity, alignment: .trailing)
        }
    }

    /// Puts the splat-count and triangle-count capsules side by side. Only the splat count without
    /// a mesh, only the triangle count without splats.
    private func counts(splats: Int) -> some View {
        HStack(spacing: 6) {
            if splats > 0 || !model.hasMesh {
                countCapsule("\(splats) splats")
            }
            if model.hasMesh {
                countCapsule("\(model.meshTriangleCount) triangles")
            }
        }
    }

    private func countCapsule(_ text: String) -> some View {
        Text(text)
            .font(.caption.monospacedDigit())
            .lineLimit(1)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(.ultraThinMaterial, in: Capsule())
    }

    /// Display menu, measure-mode toggle and Reset.
    private var trailingControls: some View {
        HStack(spacing: 6) {
            if model.hasGS || model.hasSeed || model.hasCameraMarkers || model.hasMesh {
                displayMenu
            }
            toggleCapsule("Measure", systemImage: "ruler", isOn: model.isMeasuring) {
                model.isMeasuring.toggle()
            }
            Button {
                model.resetCamera()
            } label: {
                Label("Reset", systemImage: "arrow.counterclockwise")
                    .font(.caption)
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
            }
            .buttonStyle(.plain)
        }
    }

    /// Measure-mode hint, plus undo and clear-all for measurement points.
    private var measureControls: some View {
        HStack(spacing: 8) {
            Text(model.measureNotice ?? measureHint)
                .font(.caption)
                .foregroundStyle(model.measureNotice == nil ? Color.primary : Color.yellow)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
            if !model.measurePoints.isEmpty {
                Button {
                    model.undoMeasurePoint()
                } label: {
                    Label("Undo", systemImage: "arrow.uturn.backward")
                        .font(.caption)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .buttonStyle(.plain)
                Button {
                    model.clearMeasurements()
                } label: {
                    Label("Clear all", systemImage: "trash")
                        .font(.caption)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.ultraThinMaterial, in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.bottom, 4)
    }

    private var measureHint: String {
        #if os(iOS)
        let action = "Tap"
        #else
        let action = "Click"
        #endif
        return model.measurePoints.count.isMultiple(of: 2)
            ? "\(action) to place the start point"
            : "\(action) to place the end point"
    }

    /// Fill of a button that is on: the kikyo purple, with white text on it.
    private static let onFill = Color(red: 0x7C / 255, green: 0x5C / 255, blue: 0xE6 / 255)

    /// On / off capsule button (filled = on). Used for measure mode.
    private func toggleCapsule(
        _ title: String, systemImage: String, isOn: Bool, action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.caption.weight(isOn ? .semibold : .regular))
                .foregroundStyle(isOn ? Color.white : Color.secondary)
                .padding(.horizontal, 10).padding(.vertical, 6)
                .background(
                    isOn ? AnyShapeStyle(Self.onFill) : AnyShapeStyle(.ultraThinMaterial),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }

    private func status(symbol: String, title: String, detail: String) -> some View {
        VStack(spacing: 8) {
            Image(systemName: symbol).font(.largeTitle).foregroundStyle(.secondary)
            Text(title).font(.headline)
            Text(detail).font(.caption).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(24)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(32)
    }

    // MARK: Gestures (macOS only; iOS uses UIKit recognizers on the MTKView)

    #if os(macOS)
    private var orbitGesture: some Gesture {
        DragGesture(minimumDistance: 0)
            .onChanged { value in
                let base = dragBase ?? SIMD2(model.azimuth, model.elevation)
                if dragBase == nil { dragBase = base }
                model.azimuth = base.x + Float(value.translation.width) * 0.01
                model.elevation = min(1.5, max(-1.5, base.y + Float(value.translation.height) * 0.01))
            }
            .onEnded { _ in dragBase = nil }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let base = zoomBase ?? model.distance
                if zoomBase == nil { zoomBase = base }
                model.distance = max(0.05, base / Float(value.magnification))
            }
            .onEnded { _ in zoomBase = nil }
    }

    /// Click in measure mode (the renderer ignores it outside measure mode).
    private var measureTapGesture: some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                if let p = normalized(value.location) { model.onMeasureTap?(p) }
            }
    }

    /// Double click zooms to that point (the renderer ignores it in measure mode).
    private var focusTapGesture: some Gesture {
        SpatialTapGesture(count: 2)
            .onEnded { value in
                if let p = normalized(value.location) { model.onFocusTap?(p) }
            }
    }

    /// Converts a position on the MTKView to normalized coordinates (top-left 0 / bottom-right 1).
    private func normalized(_ location: CGPoint) -> SIMD2<Float>? {
        guard metalViewSize.width > 0, metalViewSize.height > 0 else { return nil }
        return SIMD2(Float(location.x / metalViewSize.width), Float(location.y / metalViewSize.height))
    }
    #endif
}

// MARK: - MTKView host

private struct SplatMetalView: SplatViewRepresentable {
    let url: URL?
    let seedURL: URL?
    let meshURL: URL?
    let model: SplatViewerModel

    func makeCoordinator() -> Coordinator { Coordinator() }

    private func makeView(_ coordinator: Coordinator) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        let renderer = SplatViewerRenderer(view: view, model: model)
        coordinator.renderer = renderer
        view.delegate = renderer
        #if os(iOS)
        coordinator.installGestures(on: view)
        #endif
        renderer?.load(url, seedURL: seedURL, meshURL: meshURL)
        return view
    }

#if os(macOS)
    func makeNSView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateNSView(_ nsView: MTKView, context: Context) {}
#else
    func makeUIView(context: Context) -> MTKView { makeView(context.coordinator) }
    func updateUIView(_ uiView: MTKView, context: Context) {}
#endif

    @MainActor
    final class Coordinator: NSObject {
        var renderer: SplatViewerRenderer?

        #if os(iOS)
        /// 1 finger = rotate / 2 fingers = pan / pinch = zoom / tap = measurement point / double
        /// tap = zoom to that point. SwiftUI's DragGesture can't tell finger counts apart, so UIKit
        /// recognizers are used. Taps don't wait for the double tap to fail (measurement points are
        /// placed immediately). The renderer ignores one or the other depending on measure mode.
        func installGestures(on view: UIView) {
            let orbit = UIPanGestureRecognizer(target: self, action: #selector(handleOrbit(_:)))
            orbit.maximumNumberOfTouches = 1
            let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            pan.minimumNumberOfTouches = 2
            pan.maximumNumberOfTouches = 2
            let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
            let doubleTap = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
            doubleTap.numberOfTapsRequired = 2
            for recognizer in [orbit, pan, pinch, tap, doubleTap] as [UIGestureRecognizer] {
                recognizer.delegate = self
                view.addGestureRecognizer(recognizer)
            }
        }

        @objc private func handleOrbit(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            g.setTranslation(.zero, in: g.view)
            renderer?.orbit(dxPixels: Float(t.x), dyPixels: Float(t.y))
        }

        @objc private func handlePan(_ g: UIPanGestureRecognizer) {
            let t = g.translation(in: g.view)
            g.setTranslation(.zero, in: g.view)
            renderer?.pan(dxPixels: Float(t.x), dyPixels: Float(t.y))
        }

        @objc private func handlePinch(_ g: UIPinchGestureRecognizer) {
            renderer?.zoom(by: Float(g.scale))
            g.scale = 1
        }

        /// The renderer ignores it outside measure mode.
        @objc private func handleTap(_ g: UITapGestureRecognizer) {
            if let p = normalized(g) { renderer?.measureTap(normalized: p) }
        }

        /// The renderer ignores it in measure mode.
        @objc private func handleDoubleTap(_ g: UITapGestureRecognizer) {
            if let p = normalized(g) { renderer?.focusTap(normalized: p) }
        }

        /// Converts a tap position to normalized coordinates (top-left 0 / bottom-right 1).
        private func normalized(_ g: UITapGestureRecognizer) -> SIMD2<Float>? {
            guard let view = g.view, view.bounds.width > 0, view.bounds.height > 0 else { return nil }
            let p = g.location(in: view)
            return SIMD2(Float(p.x / view.bounds.width), Float(p.y / view.bounds.height))
        }
        #endif
    }
}

#if os(iOS)
extension SplatMetalView.Coordinator: UIGestureRecognizerDelegate {
    // Let pinch (2 fingers) and pan (2 fingers) work simultaneously. Orbit is 1 finger, so no
    // conflict. Simultaneous recognition is limited to our own recognizers. Allowing outside ones
    // (navigation swipe-back etc.) would start popping the whole screen at the same time as a
    // rotate drag.
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool { other.view === gestureRecognizer.view }

    // iOS 26's full-screen swipe-back is the same motion as a 1-finger drag (rotate), so let it run
    // only when our recognizer fails = effectively disabled in the viewer (go back with the back
    // button).
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldBeRequiredToFailBy other: UIGestureRecognizer
    ) -> Bool {
        guard #available(iOS 26.0, *) else { return false }
        return other === navigationController(of: gestureRecognizer)?.interactiveContentPopGestureRecognizer
    }

    // Keep swipe-back from the screen's left edge, and give it priority for touches that start at
    // the left edge (touches elsewhere fail it immediately, so rotation isn't delayed).
    func gestureRecognizer(
        _ gestureRecognizer: UIGestureRecognizer,
        shouldRequireFailureOf other: UIGestureRecognizer
    ) -> Bool {
        other === navigationController(of: gestureRecognizer)?.interactivePopGestureRecognizer
    }

    /// The UINavigationController of the screen hosting the MTKView (what backs SwiftUI's
    /// NavigationStack).
    private func navigationController(of recognizer: UIGestureRecognizer) -> UINavigationController? {
        var responder: UIResponder? = recognizer.view
        while let r = responder {
            if let vc = r as? UIViewController { return vc.navigationController }
            responder = r.next
        }
        return nil
    }
}
#endif

#else // platforms without the MetalKit render path (visionOS etc.)

public struct SplatViewerScreen: View {
    public init(
        url: URL?,
        seedURL: URL? = nil,
        meshURL: URL? = nil,
        name: String? = nil,
        shareItems: [SplatViewerShareItem] = [],
        onTrainMore: (() -> Void)? = nil,
        trainMoreMessage: String = "",
        onRename: ((String) -> Void)? = nil
    ) {}
    public var body: some View {
        ContentUnavailableView(
            "Viewer unavailable",
            systemImage: "eye.slash",
            description: Text("The PLY viewer isn't available on this platform")
        )
    }
}

#endif
