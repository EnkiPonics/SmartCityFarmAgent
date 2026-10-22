#!/usr/bin/env python3
"""
Quick start script for EnkiPonics Dashboard
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    print("="*60)
    print("EnkiPonics Dashboard Launcher")
    print("="*60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    required = ["fastapi", "uvicorn"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        print()
        print("Installing missing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    
    # Check dashboard files
    print()
    print("Checking files...")
    required_files = ["dashboard.py", "dashboard.html", "agent.py"]
    for fname in required_files:
        fpath = Path(__file__).parent / fname
        if fpath.exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} (missing)")
    
    # Start server
    print()
    print("Starting server...")
    print("="*60)
    print()
    print("Visit: http://localhost:8000")
    print("Press Ctrl+C to stop")
    print()
    print("="*60)
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "dashboard:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
