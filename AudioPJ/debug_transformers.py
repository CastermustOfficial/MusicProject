import transformers
import torch
import sys

print(f"Python executable: {sys.executable}")
print(f"Transformers version: {transformers.__version__}")
print(f"Transformers path: {transformers.__file__}")

try:
    from transformers import LlamaForCausalLM
    print("SUCCESS: LlamaForCausalLM imported from transformers")
except ImportError as e:
    print(f"FAIL: Error importing LlamaForCausalLM from transformers: {e}")
except Exception as e:
    print(f"FAIL: Unexpected error importing LlamaForCausalLM from transformers: {e}")

try:
    from transformers.models.llama import LlamaForCausalLM
    print("SUCCESS: LlamaForCausalLM imported from transformers.models.llama")
except ImportError as e:
    print(f"FAIL: Error importing LlamaForCausalLM from transformers.models.llama: {e}")
except Exception as e:
    print(f"FAIL: Unexpected error importing LlamaForCausalLM from transformers.models.llama: {e}")

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
