#!/usr/bin/env python3
"""
Script to check if required packages are installed
"""
import sys
import subprocess
import json

def check_package(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# List of required packages for training
required_packages = [
    'tensorflow',
    'numpy',
    'scikit-learn',
    'cv2',  # opencv-python
    'tqdm',
    'matplotlib',
    'tf2onnx',
    'onnx'
]

print("Checking required packages...")
results = {}

for package in required_packages:
    if package == 'cv2':
        # Special case for opencv-python
        installed = check_package('cv2')
        display_name = 'opencv-python'
    else:
        installed = check_package(package)
        display_name = package

    results[display_name] = installed
    status = "[PASS] INSTALLED" if installed else "[FAIL] MISSING"
    print(f"{display_name:<20} {status}")

# Summary
missing = [pkg for pkg, installed in results.items() if not installed]
if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("Please install missing packages using:")
    print("pip install " + " ".join(missing))
else:
    print("\nAll required packages are installed!")

# Save results to file
with open('package_check_results.json', 'w') as f:
    json.dump(results, f, indent=2)