@echo off
REM Build script for LLM Evaluator (Windows)

echo Building LLM Evaluator with PyInstaller...

REM Install PyInstaller if not already installed
pip install pyinstaller

REM Run PyInstaller with spec file
pyinstaller build.spec

echo Build complete! Executable is in dist\LLM_Evaluator\
pause
