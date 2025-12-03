import os
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"Torch path: {os.path.dirname(torch.__file__)}")
    
    # Try to add torch lib to DLL path
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        print(f"Adding torch lib to DLL search path: {torch_lib}")
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + os.pathsep + os.environ['PATH']
    else:
        print("Torch lib dir not found")
        
except ImportError as e:
    print(f"Failed to import torch: {e}")

try:
    import llama_cpp
    print(f"Llama-cpp version: {llama_cpp.__version__}")
except ImportError as e:
    print(f"Failed to import llama_cpp: {e}")
except RuntimeError as e:
    print(f"RuntimeError importing llama_cpp: {e}")
except Exception as e:
    print(f"Error importing llama_cpp: {e}")
