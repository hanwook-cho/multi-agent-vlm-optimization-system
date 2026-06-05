import SwiftUI

// ---------------------------------------------------------------------------
// Model registry — update paths after copying files to the device
// ---------------------------------------------------------------------------

private let kModelPath  = Bundle.main.path(forResource: "LFM2-VL-450M-Q4_0", ofType: "gguf") ?? ""
private let kMmprojPath = Bundle.main.path(forResource: "mmproj-LFM2-VL-450M-Q8_0", ofType: "gguf") ?? ""

// Sample images bundled in the app for consistent cross-run testing
private var sampleImagePaths: [String] {
    (1...5).compactMap { Bundle.main.path(forResource: "sample\($0)", ofType: "jpg") }
}

// ---------------------------------------------------------------------------
// ContentView
// ---------------------------------------------------------------------------

struct ContentView: View {
    @StateObject private var session = MeasurementSession()
    @State private var reportURL: URL?
    @State private var showShareSheet = false

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
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

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label("LFM2-VL-450M · Q4_0", systemImage: "cpu")
                .font(.headline)
            Text("iPhone 16 Pro  ·  Phase 0 Task 3.2")
                .font(.caption)
                .foregroundStyle(.secondary)
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
            .disabled(session.isRunning || kModelPath.isEmpty)

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

            if kModelPath.isEmpty {
                Text("⚠️ Model files not found in bundle.\nSee GGUF_SETUP.md.")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .multilineTextAlignment(.center)
            }
        }
    }

    // MARK: – Actions

    private func startMeasurement() {
        let config = RunConfig(
            modelKey:   "LFM2-VL-450M",
            modelPath:  kModelPath,
            mmprojPath: kMmprojPath,
            imagePaths: sampleImagePaths,
            prompt:     "Describe this image briefly.",
            maxTokens:  64,
            nWarmup:    1,
            nMeasure:   5
        )
        Task {
            guard let stats = await session.run(config: config) else { return }
            let docsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let modelInfo = ReportExporter.ModelInfo(
                key:          "LFM2-VL-450M",
                hfId:         "LiquidAI/LFM2-VL-450M",
                quantization: "Q4_0",
                onDiskSizeMB: ggufFileSizeMB(kModelPath)
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
