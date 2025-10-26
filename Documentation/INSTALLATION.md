# Installation Guide

Complete step-by-step installation instructions for the Purpose-Driven Local LLM Evaluator.

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 20.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB+ recommended)
- **Storage**: 2GB+ for app, 2-10GB per model
- **CPU**: Modern x64 processor

### Recommended Requirements
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster inference)
- **Storage**: 20GB+ for multiple models

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer, **check "Add Python to PATH"**
3. Verify: `python --version`

**macOS:**
```bash
# Using Homebrew
brew install python@3.11
```

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### Step 2: Download the Application

```bash
# If using git
git clone <repository-url>
cd llm_evaluator

# Or download and extract ZIP
```

#### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

#### Step 4: Install Dependencies

**Basic Installation (CPU only):**
```bash
pip install -r requirements.txt
```

**With GPU Support:**

**NVIDIA GPU (CUDA):**
```bash
# Install base packages
pip install PySide6 PyYAML

# Install llama-cpp-python with CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Apple Silicon (Metal):**
```bash
# Install base packages
pip install PySide6 PyYAML

# Install llama-cpp-python with Metal
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

#### Step 5: Download Models

Create the models directory:
```bash
mkdir -p models
```

Download GGUF models from Hugging Face. Recommended starter models:

**Small/Fast (Good for testing):**
- [Phi-2 2.7B Q4_K_M](https://huggingface.co/TheBloke/phi-2-GGUF) (~1.6GB)
- [TinyLlama 1.1B Q4_K_M](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (~600MB)

**Medium/Balanced:**
- [Mistral 7B Instruct Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) (~4.4GB)
- [Llama 2 7B Chat Q4_K_M](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) (~3.8GB)

**Specialized:**
- [CodeLlama 7B Q4_K_M](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF) (~3.8GB)

Place downloaded `.gguf` files in the `models/` directory.

#### Step 6: Configure Model Registry

Edit `config/model_registry.yaml` to add your downloaded models:

```yaml
models:
  - name: "phi-2"
    path: "models/phi-2.Q4_K_M.gguf"
    type: "chat"
    tags: ["general", "reasoning", "small"]
```

#### Step 7: Run the Application

```bash
python main.py
```

### Method 2: Using Conda

```bash
# Create conda environment
conda create -n llm_eval python=3.11
conda activate llm_eval

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

### Method 3: Standalone Executable (No Python Required)

If someone has already built the executable:

1. Download the `LLM_Evaluator` folder
2. Double-click `LLM_Evaluator.exe` (Windows) or `LLM_Evaluator` (Linux/Mac)
3. Add models to the `models/` folder
4. Configure `config/model_registry.yaml`

## Building from Source

### Prerequisites for Building

**Windows:**
- Visual Studio Build Tools 2019+
- CMake

**macOS:**
- Xcode Command Line Tools
- CMake

**Linux:**
- GCC/G++ compiler
- CMake

### Build Steps

1. Install PyInstaller:
```bash
pip install pyinstaller
```

2. Run build script:

**Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

**Windows:**
```batch
build.bat
```

3. Find executable in `dist/LLM_Evaluator/`

## Troubleshooting Installation

### Issue: "pip install llama-cpp-python" fails

**Windows:**
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Install "Desktop development with C++"

# Then retry
pip install llama-cpp-python
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Then retry
pip install llama-cpp-python
```

**Linux:**
```bash
# Install build dependencies
sudo apt install build-essential cmake

# Then retry
pip install llama-cpp-python
```

### Issue: "ModuleNotFoundError: No module named 'PySide6'"

```bash
# Ensure virtual environment is activated
pip install PySide6
```

### Issue: CUDA/Metal not detected

**NVIDIA:**
```bash
# Verify CUDA installation
nvidia-smi

# Check CUDA version and install matching toolkit
# https://developer.nvidia.com/cuda-downloads

# Reinstall with CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
```

**Apple Silicon:**
```bash
# Verify Metal support
system_profiler SPDisplaysDataType | grep Metal

# Reinstall with Metal
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
```

### Issue: Application crashes on startup

1. Run from terminal to see error messages:
```bash
python main.py
```

2. Check Python version:
```bash
python --version  # Should be 3.8+
```

3. Verify all dependencies:
```bash
pip list
```

4. Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: Model file not found

1. Check file path in `config/model_registry.yaml`
2. Use absolute path instead of relative:
```yaml
path: "/full/path/to/models/phi-2.Q4_K_M.gguf"
```

3. Verify file exists:
```bash
ls -la models/
```

## Verification

After installation, verify everything works:

```bash
# Activate virtual environment if using one
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Run application
python main.py
```

You should see the application window open with:
- Task & Models tab
- Prompt & Run tab
- Evaluations tab
- Settings tab

## Updating

To update to a new version:

```bash
# Pull latest changes (if using git)
git pull

# Update dependencies
pip install -r requirements.txt --upgrade

# Run application
python main.py
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove application directory
rm -rf llm_evaluator/

# Or on Windows
rmdir /s llm_evaluator
```

## Next Steps

After installation:
1. Read [QUICKSTART.md](QUICKSTART.md) for first-time usage
2. Download at least one model
3. Configure model registry
4. Run your first evaluation

## Getting Help

If you encounter issues:
1. Check this installation guide
2. Review [README.md](README.md) troubleshooting section
3. Ensure all dependencies are installed correctly
4. Check Python and package versions
5. Open an issue with error logs

---

**Installation complete? Continue to [QUICKSTART.md](QUICKSTART.md)**
