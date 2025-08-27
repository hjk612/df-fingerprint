#!/usr/bin/env python3
"""
Development environment setup script.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def main():
    print("🚀 Setting up df-fingerprint development environment")
    
    # Install package in development mode
    run_command("pip install -e '.[dev,test]'", "Installing package in development mode")
    
    # Install pre-commit hooks (if available)
    try:
        run_command("pip install pre-commit", "Installing pre-commit")
        run_command("pre-commit install", "Setting up pre-commit hooks")
    except:
        print("⚠️  Pre-commit setup skipped (optional)")
    
    # Run initial tests
    run_command("pytest tests/ -v", "Running initial test suite")
    
    # Check code formatting
    run_command("black --check df_fingerprint tests", "Checking code formatting")
    
    # Type checking
    run_command("mypy df_fingerprint --ignore-missing-imports", "Running type checks")
    
    print("\n🎉 Development environment setup complete!")
    print("\nUseful commands:")
    print("  pytest tests/                    # Run tests")
    print("  black df_fingerprint tests       # Format code")
    print("  mypy df_fingerprint              # Type check")
    print("  python scripts/bump_version.py   # Bump version")
    print("  python -m build                  # Build package")


if __name__ == "__main__":
    main()