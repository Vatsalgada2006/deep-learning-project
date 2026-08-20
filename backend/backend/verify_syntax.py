#!/usr/bin/env python3
"""
Backend syntax verification script
Tests that the backend files have correct Python syntax
"""

import sys
import os
import ast

def check_python_syntax(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error in {file_path}: line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading {file_path}: {str(e)}"

def main():
    """Check syntax of all Python files in backend."""
    print("Road Accident Detection System - Backend Syntax Verification")
    print("=" * 60)

    backend_dir = os.path.dirname(os.path.abspath(__file__))
    python_files = []

    # Find all Python files
    for root, dirs, files in os.walk(backend_dir):
        # Skip __pycache__ and .pyc files
        dirs[:] = [d for d in dirs if not d.startswith('__pycache__') and d not in ['venv', '__pycache__']]
        for file in files:
            if file.endswith('.py') and not file.endswith('.pyc'):
                python_files.append(os.path.join(root, file))

    if not python_files:
        print("[WARNING] No Python files found!")
        return 1

    print(f"Found {len(python_files)} Python files to check\n")

    all_passed = True
    passed_count = 0

    for file_path in sorted(python_files):
        # Make path relative for cleaner output
        rel_path = os.path.relpath(file_path, backend_dir)
        passed, error_msg = check_python_syntax(file_path)

        if passed:
            print(f"[OK] {rel_path}")
            passed_count += 1
        else:
            print(f"[ERROR] {rel_path}")
            print(f"       {error_msg}")
            all_passed = False

    print("\n" + "=" * 60)
    print(f"Syntax Check Results: {passed_count}/{len(python_files)} files passed")

    if all_passed:
        print("[OK] All Python files have valid syntax!")
        print("[INFO] Backend structure is ready for dependency installation.")
        return 0
    else:
        print("[ERROR] Syntax errors found. Please fix them before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())