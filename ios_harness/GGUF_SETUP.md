# GGUF Model File Setup for VLMHarness

The app looks for model files in its bundle via `Bundle.main.path(forResource:ofType:)`.

## Files needed

| File | Size | Source |
|---|---|---|
| `LFM2-VL-450M-Q4_0.gguf` | 219 MB | HuggingFace LiquidAI/LFM2-VL-450M-GGUF |
| `mmproj-LFM2-VL-450M-Q8_0.gguf` | ~90 MB | HuggingFace LiquidAI/LFM2-VL-450M-GGUF |
| `sample1.jpg` … `sample5.jpg` | small | copy from datasets/stage_a_proxy/photos/ |

Files are already in the HF cache on the Mac mini:
```
~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-450M-GGUF/snapshots/<hash>/
```

## How to add to the Xcode project

1. Open `VLMHarness.xcodeproj` in Xcode
2. Drag the GGUF files into the **VLMHarness** group in the Project Navigator
3. In the "Add files" dialog: ✅ **"Copy items if needed"**, ✅ Target **VLMHarness** checked
4. Do the same for `sample1.jpg`–`sample5.jpg` (rename from `img1.jpg`–`img5.jpg`)
5. Build & run (⌘R) — the "Run Measurement" button will become active

## Model paths (for reference)
```
LFM2-VL-450M-Q4_0.gguf:
  ~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-450M-GGUF/snapshots/64443d474c11969a91bed19226b9cdada82c628e/LFM2-VL-450M-Q4_0.gguf

mmproj-LFM2-VL-450M-Q8_0.gguf:
  ~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-450M-GGUF/snapshots/64443d474c11969a91bed19226b9cdada82c628e/mmproj-LFM2-VL-450M-Q8_0.gguf
```
