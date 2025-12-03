# YuE Pipeline Modes

This backend supports two different pipeline modes for music generation:

## 1. GGUF Pipeline (Current - Fast)

**Pros:**
- Fast inference with llama.cpp
- Lower VRAM usage (~8-12GB)
- Works on consumer GPUs

**Cons:**
- Quantized models (lower quality)
- Currently generates placeholder audio (token decoding not fully implemented)

**Setup:**
```bash
# Already configured - uses existing GGUF models
```

**Configuration:**
In `config.py`:
```python
PIPELINE_MODE = "gguf"
```

## 2. HuggingFace Pipeline (High Quality)

**Pros:**
- Full precision FP16 models (best quality)
- Proper audio token decoding with Stage 2
- Official model architecture

**Cons:**
- Slower inference
- Higher VRAM usage (~20-24GB for Stage 1, ~8GB for Stage 2)
- Requires downloading large models (~14GB for Stage 1)

**Setup:**
```bash
# Install additional dependencies
pip install -r requirements_hq.txt

# Models will auto-download on first run to ./models/huggingface_cache
# Or download manually:
huggingface-cli download m-a-p/YuE-s1-7B-anneal-en-cot --cache-dir ./models/huggingface_cache
huggingface-cli download m-a-p/YuE-s2-1B-general --cache-dir ./models/huggingface_cache
```

**Configuration:**
In `config.py`:
```python
PIPELINE_MODE = "huggingface"
```

## Current Status

### GGUF Pipeline
- ✅ Stage 1 token generation working
- ❌ Stage 2 audio decoding not implemented (uses placeholder)
- ⚠️ Generates 30-second test tone for now

### HuggingFace Pipeline
- ✅ Stage 1 token generation implemented
- ⚠️ Stage 2 audio decoding partially implemented
- ⚠️ May use placeholder if model doesn't have `decode()` method
- 🔨 Needs testing with real models

## Switching Between Modes

1. Edit `backend/config.py`
2. Change `PIPELINE_MODE` to either `"gguf"` or `"huggingface"`
3. Restart uvicorn

```bash
# Restart the server
uvicorn backend.main:app --reload
```

## Recommended Approach

**For testing/development:** Use GGUF mode (current default)
- Fast iteration
- Lower hardware requirements

**For production/quality:** Use HuggingFace mode
- Download the full models first
- Ensure you have 24GB+ VRAM
- Expect slower generation (~2-5 minutes per song)

## Hardware Requirements

### GGUF Mode
- GPU: 8GB+ VRAM (RTX 3060 12GB, RTX 4060 Ti 16GB)
- RAM: 16GB+

### HuggingFace Mode
- GPU: 24GB+ VRAM (RTX 3090, RTX 4090, A5000)
- RAM: 32GB+
- Storage: 20GB+ for models

## Notes on Audio Quality

The YuE model architecture requires:
1. **Stage 1**: Generates semantic audio tokens (structure, melody, rhythm)
2. **xcodec/vocoder**: Decodes tokens to waveform (this is the missing piece!)

The HuggingFace pipeline attempts to use Stage 2 as a vocoder, but the actual xcodec decoder may be separate. If you encounter issues with audio quality, check:

- The model cards on HuggingFace for the correct decoder
- YuE official repo for the xcodec configuration files
- Consider using the ComfyUI_YuE implementation as reference

## Troubleshooting

**"Out of memory" errors with HuggingFace mode:**
- Close other applications
- Use `device_map="auto"` (already configured)
- Consider using `load_in_8bit=True` for Stage 1 (quality trade-off)

**Audio is still placeholder:**
- Stage 2 decoder integration needs the proper vocoder
- Check model documentation for `decode()` or `generate_audio()` methods
- May need to integrate xcodec separately

**Import errors:**
- Make sure you installed `requirements_hq.txt`
- Restart uvicorn after installing dependencies
