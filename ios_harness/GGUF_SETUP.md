# GGUF Model File Setup for VLMHarness

The app looks for model files in its bundle via `Bundle.main.path(forResource:ofType:)`.

## Files needed

### Phase 0 baseline

| File | Size | Source |
|---|---|---|
| `LFM2-VL-450M-Q4_0.gguf` | 209 MB | HuggingFace `LiquidAI/LFM2-VL-450M-GGUF` |
| `mmproj-LFM2-VL-450M-Q8_0.gguf` | ~90 MB | HuggingFace `LiquidAI/LFM2-VL-450M-GGUF` |
| `SmolVLM-500M-Instruct.Q4_K_M.gguf` | 367 MB | HuggingFace `bartowski/SmolVLM-500M-Instruct-GGUF` |
| `mmproj-SmolVLM-500M-Instruct-Q8_0.gguf` | ~50 MB | HuggingFace community |
| `MiniCPM-V-4.6-Q4_K_M.gguf` | ~900 MB | HuggingFace community |
| `mmproj-MiniCPM-V-4.6-Q8_0.gguf` | ~80 MB | HuggingFace community |
| `sample1.jpg` … `sample5.jpg` | small | copy from `datasets/stage_a_proxy/photos/` |

### Phase 1 Week 2 — H001 experiment (LFM2 Q4_K_M)

| File | Size | Source |
|---|---|---|
| `LFM2-VL-450M-Q4_K_M.gguf` | 218 MB | `artifacts/experiments/lfm2-q4km/` (bartowski imatrix build) |
| `mmproj-LFM2-VL-450M-Q8_0.gguf` | ~90 MB | same as Phase 0 — reuse existing bundle resource |

Mac quality eval: CLIP-score **28.59 ± 3.60** (n=50) vs Phase 0 Q4_0 baseline **27.60** ✅  
experiment_id: `12d065239be5693a9e3aa57bcc6e0a814143c00145de441fc29d17ad5922d580`

## How to add to the Xcode project

1. Open `VLMHarness.xcodeproj` in Xcode
2. Drag each GGUF file into the **VLMHarness** group in the Project Navigator
3. In the "Add files" dialog: ✅ **"Copy items if needed"**, ✅ Target **VLMHarness** checked
4. Do the same for `sample1.jpg`–`sample5.jpg` (rename from `img1.jpg`–`img5.jpg` if needed)
5. Build & run (⌘R) — the "Run Measurement" button activates when files are found in bundle

**For H001 Phase 1 run:** the model picker now has a "LFM2-Q4KM" tab. Select it to run the Q4_K_M variant. The mmproj file is shared with the Q4_0 entry — add it once.

## Phase 1 iPhone measurement protocol

After adding `LFM2-VL-450M-Q4_K_M.gguf` to the bundle:

1. Select **LFM2-Q4KM** in the model picker
2. Tap **Run Measurement** (1 warm-up + 5 measured runs, 5 images each)
3. Tap **Export MetricsReport JSON** and AirDrop / Files app to Mac
4. On Mac, call:
   ```python
   from services.experiment_runner import ExperimentRunner
   runner = ExperimentRunner()
   report = runner.import_iphone_results(
       "12d065239be5693a9e3aa57bcc6e0a814143c00145de441fc29d17ad5922d580",
       "/path/to/exported_report.json"
   )
   print(report)
   ```

## Model file paths (for reference)

```
LFM2-VL-450M-Q4_0.gguf:
  ~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-450M-GGUF/snapshots/64443d474c11969a91bed19226b9cdada82c628e/LFM2-VL-450M-Q4_0.gguf

mmproj-LFM2-VL-450M-Q8_0.gguf:
  ~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-450M-GGUF/snapshots/64443d474c11969a91bed19226b9cdada82c628e/mmproj-LFM2-VL-450M-Q8_0.gguf

LFM2-VL-450M-Q4_K_M.gguf (Phase 1 H001):
  artifacts/experiments/lfm2-q4km/LFM2-VL-450M-Q4_K_M.gguf
```
