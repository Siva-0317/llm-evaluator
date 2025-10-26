#!/bin/bash
# Build script for LLM Evaluator

echo "Building LLM Evaluator with PyInstaller..."

# Install PyInstaller if not already installed
pip install pyinstaller

# Run PyInstaller with spec file
pyinstaller build.spec

echo "Build complete! Executable is in dist/LLM_Evaluator/"
