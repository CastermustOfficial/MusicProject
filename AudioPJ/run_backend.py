import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure the current directory is in sys.path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    print("Starting Audio Insane Backend...")
    print("Ensure you are running this from the 'PROJECT AUDIO INSANE' directory.")
    
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
