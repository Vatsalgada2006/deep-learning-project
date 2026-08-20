#!/usr/bin/env python3
"""
Backend verification script
Tests that the backend can import and initialize correctly
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        import fastapi
        print("[OK] FastAPI imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import FastAPI: {e}")
        return False

    try:
        import uvicorn
        print("[OK] Uvicorn imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import Uvicorn: {e}")
        return False

    try:
        import pymongo
        print("[OK] PyMongo imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import PyMongo: {e}")
        return False

    try:
        import onnxruntime
        print("[OK] ONNX Runtime imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import ONNX Runtime: {e}")
        return False

    try:
        import cv2
        print("[OK] OpenCV imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import OpenCV: {e}")
        return False

    try:
        import numpy
        print("[OK] NumPy imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import NumPy: {e}")
        return False

    try:
        from dotenv import load_dotenv
        print("[OK] Python-dotenv imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import Python-dotenv: {e}")
        return False

    return True

def test_app_creation():
    """Test that the FastAPI app can be created."""
    print("\nTesting FastAPI app creation...")

    try:
        # Change to backend directory to import main
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from main import app
        print("[OK] FastAPI app created successfully")
        print(f"[INFO] App title: {app.title}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create FastAPI app: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("Road Accident Detection System - Backend Verification")
    print("=" * 55)

    tests_passed = 0
    total_tests = 2

    if test_imports():
        tests_passed += 1

    if test_app_creation():
        tests_passed += 1

    print("\n" + "=" * 55)
    print(f"Verification Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("[OK] All tests passed! Backend is ready for model integration.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())