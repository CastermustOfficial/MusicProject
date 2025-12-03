"""
Configuration for YuE Pipeline
"""

# Choose which pipeline to use:
# - "gguf": Fast inference with llama.cpp (current implementation)
# - "huggingface": High-quality inference with transformers (slower, needs more VRAM)
PIPELINE_MODE = "huggingface"  # Change to "huggingface" for best quality

# GGUF Configuration (for fast inference)
GGUF_MODEL_STAGE1 = "./models/YuE-s1-7B-anneal-en-cot-Q4_K_S.gguf"
GGUF_MODEL_STAGE2 = "./models/yue-s2-1b-general-q8_0.gguf"

# HuggingFace Configuration (for high quality)
HF_MODEL_STAGE1 = "m-a-p/YuE-s1-7B-anneal-en-cot"
HF_MODEL_STAGE2 = "m-a-p/YuE-s2-1B-general"
HF_CACHE_DIR = "./models/huggingface_cache"

# Output directory
OUTPUT_DIR = "./outputs"
