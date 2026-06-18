import SwiftUI
import PhotosUI

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

struct ModelEntry: Identifiable {
    let id: String          // display key
    let ggufName: String    // Bundle resource name (no extension)
    let mmprojName: String
    let chatTemplate: String
    let hfId: String
    let quantization: String
    /// H003: resize input images to this square resolution before inference. 0 = model default.
    var inputResolution: Int = 0

    var modelPath:  String { Bundle.main.path(forResource: ggufName,  ofType: "gguf") ?? "" }
    var mmprojPath: String { Bundle.main.path(forResource: mmprojName, ofType: "gguf") ?? "" }
    var isAvailable: Bool  { !modelPath.isEmpty && !mmprojPath.isEmpty }
}

private let kModels: [ModelEntry] = [
    ModelEntry(
        id:           "LFM2-VL-450M",
        ggufName:     "LFM2-VL-450M-Q4_0",
        mmprojName:   "mmproj-LFM2-VL-450M-Q8_0",
        chatTemplate: "chatml",
        hfId:         "LiquidAI/LFM2-VL-450M",
        quantization: "Q4_0"
    ),
    // Phase 1 Week 2 — H001 pivot: Q4_K_M (bartowski imatrix build)
    // CLIP-score Mac eval: 28.59 ± 3.60 (vs Q4_0 baseline 27.60)
    // experiment_id: 12d065239be5693a9e3aa57bcc6e0a814143c00145de441fc29d17ad5922d580
    ModelEntry(
        id:           "LFM2-Q4KM",
        ggufName:     "LFM2-VL-450M-Q4_K_M",
        mmprojName:   "mmproj-LFM2-VL-450M-Q8_0",
        chatTemplate: "chatml",
        hfId:         "LiquidAI/LFM2-VL-450M",
        quantization: "Q4_K_M"
    ),
    // Phase 1 Week 3 — H003: input resize 336→224px (no model change)
    // CLIP-score Mac eval: 27.88 ± 4.54 (n=50) — quality nearly unchanged ✅
    // experiment_id: a8d879818a188ad8af461a1c32a8e4285ee1beff23d40c8e7cf4ecb0a2ae6ef9
    ModelEntry(
        id:             "LFM2-224px",
        ggufName:       "LFM2-VL-450M-Q4_K_M",
        mmprojName:     "mmproj-LFM2-VL-450M-Q8_0",
        chatTemplate:   "chatml",
        hfId:           "LiquidAI/LFM2-VL-450M",
        quantization:   "Q4_K_M",
        inputResolution: 224
    ),
    ModelEntry(
        id:           "SmolVLM-500M",
        ggufName:     "SmolVLM-500M-Instruct.Q4_K_M",
        mmprojName:   "mmproj-SmolVLM-500M-Instruct-Q8_0",
        chatTemplate: "smolvlm",
        hfId:         "HuggingFaceTB/SmolVLM-500M-Instruct",
        quantization: "Q4_K_M"
    ),
    // Phase 1 Week 2 — H002 pivot: i1-Q4_0 (mradermacher imatrix build)
    // CLIP-score Mac eval: 27.78 ± 4.26 (n=50) — fp16 proxy, perf measured on iPhone
    // experiment_id: ccd9d9bca7d6c15ff1d9fa7196fa9f57d412a437d2a052f5b79c25a6c9d9a30e
    ModelEntry(
        id:           "SmolVLM-Q4K0",
        ggufName:     "SmolVLM-500M-Instruct.i1-Q4_0",
        mmprojName:   "mmproj-SmolVLM-500M-Instruct-Q8_0",
        chatTemplate: "smolvlm",
        hfId:         "HuggingFaceTB/SmolVLM-500M-Instruct",
        quantization: "i1-Q4_0"
    ),
    ModelEntry(
        id:           "MiniCPM-V-4.6",
        ggufName:     "MiniCPM-V-4.6-Q4_K_M",
        mmprojName:   "mmproj-MiniCPM-V-4.6-Q8_0",
        chatTemplate: "chatml",
        hfId:         "openbmb/MiniCPM-V-4.6",
        quantization: "Q4_K_M"
    ),
]

// Sample images bundled in the app for consistent cross-run testing
private var sampleImagePaths: [String] {
    (1...5).compactMap { Bundle.main.path(forResource: "sample\($0)", ofType: "jpg") }
}

// Image source: the fixed bundled set (reproducible cross-run benchmark) or
// user-picked photos from the device library (qualitative / ad-hoc testing).
enum ImageSource: String, CaseIterable, Identifiable {
    case bundled = "Bundled (5)"
    case library = "Photo library"
    var id: String { rawValue }
}

// ---------------------------------------------------------------------------
// ContentView
// ---------------------------------------------------------------------------

struct ContentView: View {
    @StateObject private var session = MeasurementSession()
    @State private var selectedModelIndex = 0
    @State private var reportURL: URL?
    @State private var showShareSheet = false

    // Image source (A): bundled set vs. photo-library picks
    @State private var imageSource: ImageSource = .bundled
    @State private var pickerItems: [PhotosPickerItem] = []
    @State private var libraryImagePaths: [String] = []
    @State private var isLoadingPhotos = false

    private var selectedModel: ModelEntry { kModels[selectedModelIndex] }

    /// Images the next run will use, given the selected source.
    private var activeImagePaths: [String] {
        (imageSource == .library && !libraryImagePaths.isEmpty) ? libraryImagePaths : sampleImagePaths
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                modelPickerSection
                imageSourceSection
                headerSection
                if !session.log.isEmpty { logSection }
                if let stats = session.lastStats { statsSection(stats) }
                if !session.lastOutputSample.isEmpty { outputSection }
                Spacer()
                actionButtons
            }
            .padding()
            .navigationTitle("VLM Harness")
            .navigationBarTitleDisplayMode(.large)
        }
        .sheet(isPresented: $showShareSheet) {
            if let url = reportURL {
                ShareSheet(activityItems: [url])
            }
        }
    }

    // MARK: – Sub-views

    private var modelPickerSection: some View {
        Picker("Model", selection: $selectedModelIndex) {
            ForEach(kModels.indices, id: \.self) { i in
                Text(kModels[i].id).tag(i)
            }
        }
        .pickerStyle(.segmented)
        .onChange(of: selectedModelIndex) { _, _ in
            // Reset results when switching models
            session.lastStats = nil
            session.lastOutputSample = ""
            session.log = []
            reportURL = nil
        }
    }

    private var imageSourceSection: some View {
        VStack(spacing: 8) {
            Picker("Images", selection: $imageSource) {
                ForEach(ImageSource.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            if imageSource == .library {
                PhotosPicker(selection: $pickerItems, maxSelectionCount: 10, matching: .images) {
                    Label(libraryImagePaths.isEmpty
                            ? "Select photos…"
                            : "\(libraryImagePaths.count) photo(s) selected — change",
                          systemImage: "photo.on.rectangle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(session.isRunning)
                .onChange(of: pickerItems) { _, items in
                    Task { await loadPickedPhotos(items) }
                }
                if isLoadingPhotos {
                    Text("Loading photos…").font(.caption2).foregroundStyle(.secondary)
                } else if !libraryImagePaths.isEmpty {
                    Text("Latency/memory are valid on any image; outputs are qualitative (no fixed reference set).")
                        .font(.caption2).foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
        }
    }

    /// Materialize picked PhotosPickerItems to temp JPEGs and collect their paths.
    /// Normalizes via UIImage so HEIC/other formats become runner-readable JPEGs.
    private func loadPickedPhotos(_ items: [PhotosPickerItem]) async {
        isLoadingPhotos = true
        defer { isLoadingPhotos = false }
        var paths: [String] = []
        let tmp = FileManager.default.temporaryDirectory
        for (i, item) in items.enumerated() {
            guard let data = try? await item.loadTransferable(type: Data.self) else { continue }
            let url = tmp.appendingPathComponent("vlmh_lib_\(i).jpg")
            if let img = UIImage(data: data), let jpeg = img.jpegData(compressionQuality: 0.95) {
                try? jpeg.write(to: url)
            } else {
                try? data.write(to: url)   // fallback: write raw bytes
            }
            paths.append(url.path)
        }
        libraryImagePaths = paths
        // switching the image set invalidates prior results
        session.lastStats = nil
        session.lastOutputSample = ""
        reportURL = nil
    }

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("\(selectedModel.id) · \(selectedModel.quantization)",
                  systemImage: "cpu")
                .font(.headline)
            Text("iPhone 16 Pro  ·  Phase 1 Week 2")
                .font(.caption)
                .foregroundStyle(.secondary)
            if !selectedModel.isAvailable {
                Text("⚠️ GGUF files not found in bundle")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var logSection: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(session.log.enumerated()), id: \.offset) { idx, line in
                        Text(line)
                            .font(.system(.caption2, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(idx)
                    }
                }
                .padding(8)
            }
            .frame(maxHeight: 220)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            .onChange(of: session.log.count) { _, _ in
                proxy.scrollTo(session.log.count - 1, anchor: .bottom)
            }
        }
    }

    private func statsSection(_ stats: RunStats) -> some View {
        Grid(alignment: .leading, horizontalSpacing: 20, verticalSpacing: 8) {
            GridRow {
                statCell("TTFT",
                         String(format: "%.0f ms", stats.ttftMsMean),
                         sub: String(format: "±%.0f", stats.ttftMsStddev))
                statCell("TPS",
                         String(format: "%.1f t/s", stats.tpsMean))
                statCell("Peak Mem",
                         String(format: "%.0f MB", stats.peakMemMBMean))
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func statCell(_ title: String, _ value: String, sub: String = "") -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.title3.bold())
            if !sub.isEmpty {
                Text(sub).font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    private var outputSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Last output")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(session.lastOutputSample)
                .font(.caption)
                .lineLimit(4)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private var actionButtons: some View {
        VStack(spacing: 12) {
            Button(action: startMeasurement) {
                Label(session.isRunning ? "Running…" : "Run Measurement",
                      systemImage: session.isRunning ? "hourglass" : "play.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(session.isRunning || !selectedModel.isAvailable
                      || (imageSource == .library && libraryImagePaths.isEmpty))

            if let url = reportURL {
                Button(action: { showShareSheet = true }) {
                    Label("Export MetricsReport JSON", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                Text(url.lastPathComponent)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: – Actions

    private func startMeasurement() {
        let model = selectedModel
        let images = activeImagePaths
        // Bundled: 5 fixed runs. Library: one measured run per picked photo.
        let measureCount = (imageSource == .library && !libraryImagePaths.isEmpty)
            ? libraryImagePaths.count : 5
        let config = RunConfig(
            modelKey:        model.id,
            modelPath:       model.modelPath,
            mmprojPath:      model.mmprojPath,
            chatTemplate:    model.chatTemplate,
            hfId:            model.hfId,
            quantization:    model.quantization,
            imagePaths:      images,
            prompt:          "Describe this image briefly.",
            maxTokens:       64,
            nWarmup:         1,
            nMeasure:        measureCount,
            inputResolution: model.inputResolution
        )
        reportURL = nil
        Task {
            guard let stats = await session.run(config: config) else { return }
            let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let modelInfo = ReportExporter.ModelInfo(
                key:          model.id,
                hfId:         model.hfId,
                quantization: model.quantization,
                onDiskSizeMB: ggufFileSizeMB(model.modelPath) + ggufFileSizeMB(model.mmprojPath)
            )
            reportURL = try? ReportExporter.export(modelInfo: modelInfo,
                                                   stats: stats,
                                                   outputDir: docsDir)
        }
    }

    private func ggufFileSizeMB(_ path: String) -> Double {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int64 else { return 0 }
        return Double(size) / (1024 * 1024)
    }
}

// MARK: – ShareSheet

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    func updateUIViewController(_ uvc: UIActivityViewController, context: Context) {}
}

#Preview {
    ContentView()
}
