import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "cpu")
                .imageScale(.large)
                .foregroundStyle(.tint)
                .font(.system(size: 60))
            Text("VLM Harness")
                .font(.largeTitle.bold())
            Text("Phase 0 · Task 3.1 smoke test")
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Text("✅ App deployed successfully")
                .font(.callout)
                .padding(.top, 8)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
