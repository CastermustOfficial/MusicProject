import os
import sys

print("Starting verification...")
try:
    import torch
    print(f"Torch found at: {os.path.dirname(torch.__file__)}")
    
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        print(f"Adding torch lib to DLL path: {torch_lib}")
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
    else:
        print("Torch lib dir not found!")
        
except ImportError as e:
    print(f"Could not import torch: {e}")

print("Attempting to import llama_cpp...")
try:
    import llama_cpp
    print("SUCCESS: llama_cpp imported successfully!")
except ImportError as e:
    print(f"ImportError: {e}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
