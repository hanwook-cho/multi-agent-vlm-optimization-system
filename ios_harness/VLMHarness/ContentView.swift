import SwiftUI

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
    ModelEntry(
        id:           "SmolVLM-500M",
        ggufName:     "SmolVLM-500M-Instruct.Q4_K_M",
        mmprojName:   "mmproj-SmolVLM-500M-Instruct-Q8_0",
        chatTemplate: "smolvlm",
        hfId:         "HuggingFaceTB/SmolVLM-500M-Instruct",
        quantization: "Q4_K_M"
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

// ---------------------------------------------------------------------------
// ContentView
// ---------------------------------------------------------------------------

struct ContentView: View {
    @StateObject private var session = MeasurementSession()
    @State private var selectedModelIndex = 0
    @State private var reportURL: URL?
    @State private var showShareSheet = false

    private var selectedModel: ModelEntry { kModels[selectedModelIndex] }

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                modelPickerSection
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

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("\(selectedModel.id) · \(selectedModel.quantization)",
                  systemImage: "cpu")
                .font(.headline)
            Text("iPhone 16 Pro  ·  Phase 0 Task 3.4")
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
            .disabled(session.isRunning || !selectedModel.isAvailable)

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
        let config = RunConfig(
            modelKey:     model.id,
            modelPath:    model.modelPath,
            mmprojPath:   model.mmprojPath,
            chatTemplate: model.chatTemplate,
            hfId:         model.hfId,
            quantization: model.quantization,
            imagePaths:   sampleImagePaths,
            prompt:       "Describe this image briefly.",
            maxTokens:    64,
            nWarmup:      1,
            nMeasure:     5
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
