#!/usr/bin/env python3
"""
Install monitoring dependencies for RAG Mistral system
"""

import subprocess
import sys

def install_package(package):
    """Install a Python package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🔧 Installing monitoring dependencies for RAG Mistral system...")
    
    # List of monitoring packages
    monitoring_packages = [
        "psutil>=5.9.0",
        "GPUtil>=1.4.0", 
        "nvidia-ml-py3>=7.352.0"
    ]
    
    success_count = 0
    total_packages = len(monitoring_packages)
    
    for package in monitoring_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"  • Successfully installed: {success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("🎉 All monitoring dependencies installed successfully!")
        print("\nYou can now run the RAG system with full monitoring:")
        print("  python cli_chat.py")
        print("  python cli_chat.py --stats")
    else:
        print("⚠️  Some packages failed to install. The system will still work but with limited monitoring.")
        print("\nNote: nvidia-ml-py3 and GPUtil are only needed if you have NVIDIA GPUs.")
        print("The system will automatically fall back to basic monitoring if these are not available.")
    
    # Test imports
    print("\n🧪 Testing imports...")
    
    try:
        import psutil
        print("✅ psutil (System monitoring)")
    except ImportError:
        print("❌ psutil not available")
    
    try:
        import GPUtil
        print("✅ GPUtil (GPU monitoring)")
    except ImportError:
        print("⚠️  GPUtil not available (GPU monitoring limited)")
    
    try:
        import nvidia_ml_py3
        print("✅ nvidia-ml-py3 (NVIDIA GPU monitoring)")
    except ImportError:
        print("⚠️  nvidia-ml-py3 not available (NVIDIA monitoring limited)")

if __name__ == "__main__":
    main()
