#!/usr/bin/env python3
"""
Test script to verify that all dependencies are installed correctly.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher."""
    print(f"Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python version {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed and in PATH."""
    print(f"Checking FFmpeg installation...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode != 0:
            print("❌ FFmpeg is not installed or not in PATH.")
            return False
        
        # Extract version from output
        version_line = result.stdout.split('\n')[0]
        print(f"✅ {version_line}")
        return True
    except FileNotFoundError:
        print("❌ FFmpeg is not installed or not in PATH.")
        return False

def check_python_dependencies():
    """Check if required Python packages are installed."""
    print(f"Checking Python dependencies...")
    
    required_packages = [
        "faster-whisper",
        "opencv-python",
        "mediapipe",
        "librosa",
        "pyscenedetect",
        "numpy",
        "requests",
        "tqdm",
        "ffmpeg-python"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nMissing packages. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_cuda():
    """Check if CUDA is available (for GPU acceleration)."""
    print(f"Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️ CUDA is not available. The application will run on CPU, which may be slower.")
            return False
    except ImportError:
        print("⚠️ PyTorch is not installed. Cannot check CUDA availability.")
        print("  This is not a critical error, but GPU acceleration may not work.")
        return False

def check_ollama():
    """Check if Ollama is running (optional, for visual analysis)."""
    print(f"Checking Ollama availability (optional)...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            llava_models = [m for m in models if "llava" in m.get("name", "").lower()]
            
            if llava_models:
                print(f"✅ Ollama is running with LLaVA models: {', '.join(m['name'] for m in llava_models)}")
            else:
                print("⚠️ Ollama is running but no LLaVA models found. Run 'ollama pull llava:7b' to download.")
            return True
        else:
            print("⚠️ Ollama API returned an error. Make sure Ollama is running correctly.")
            return False
    except Exception:
        print("⚠️ Ollama is not running or not installed. Visual analysis will not be available.")
        print("  This is not a critical error if you don't plan to use visual analysis.")
        return False

def main():
    """Run all checks."""
    print("Auto-Clipper Installation Test\n")
    
    python_ok = check_python_version()
    ffmpeg_ok = check_ffmpeg()
    deps_ok = check_python_dependencies()
    
    # Optional checks
    cuda_ok = check_cuda()
    ollama_ok = check_ollama()
    
    print("\nSummary:")
    print(f"{'✅' if python_ok else '❌'} Python version")
    print(f"{'✅' if ffmpeg_ok else '❌'} FFmpeg")
    print(f"{'✅' if deps_ok else '❌'} Python dependencies")
    print(f"{'✅' if cuda_ok else '⚠️'} CUDA (optional)")
    print(f"{'✅' if ollama_ok else '⚠️'} Ollama (optional)")
    
    if python_ok and ffmpeg_ok and deps_ok:
        print("\n✅ Basic requirements are met! You can run Auto-Clipper.")
        if not cuda_ok:
            print("⚠️ Note: Running without GPU acceleration will be slower.")
        if not ollama_ok:
            print("⚠️ Note: Visual analysis with LLaVA will not be available.")
    else:
        print("\n❌ Some requirements are not met. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
