# Purpose-Driven Local LLM Evaluator

A lightweight Python desktop application for evaluating and comparing local LLM models based on specific tasks.

## Features

- 🎯 **Task-Driven Model Discovery** - Define your task and automatically find suitable models
- 🧠 **Local LLM Integration** - Run GGUF models locally via llama-cpp-python
- 📊 **Multi-Model Comparison** - Compare multiple models side-by-side
- 🧪 **Evaluation Framework** - Run task-specific test sets
- 💾 **Export Results** - Export comparison results to CSV
- 🖥️ **Desktop GUI** - Clean PySide6 interface

## Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- GGUF model files

## Installation

### 1. Clone or download this repository

```bash
cd llm_evaluator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note for GPU acceleration:**

- **NVIDIA GPU (CUDA):**
  ```bash
  CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```

- **Mac (Metal):**
  ```bash
  CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
  ```

### 3. Download GGUF models

Place your GGUF model files in the `models/` directory. You can download models from:
- [Hugging Face](https://huggingface.co/models?library=gguf)
- [TheBloke's Models](https://huggingface.co/TheBloke)

Example models:
- `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- `llama-2-7b-chat.Q4_K_M.gguf`
- `phi-2.Q4_K_M.gguf`

## Usage

### Running the Application

```bash
python main.py
```

### Quick Start Guide

1. **Define Your Task** (Task & Models tab)
   - Describe what you want the model to do
   - Select a task category or leave as auto-detect
   - Click "Discover Matching Models"

2. **Configure Model** (Settings tab)
   - Browse and select a GGUF model file
   - Adjust parameters (threads, context, temperature)
   - Click "Load Model"

3. **Run Prompts** (Prompt & Run tab)
   - Enter system and user prompts
   - Click "Run on Selected Model" for single model
   - Click "Run on All Matched Models" for comparison

4. **Evaluate Models** (Evaluations tab)
   - Select an evaluation YAML file
   - Click "Run Evaluation"
   - View comparison table
   - Export results to CSV

## Configuration Files

### Model Registry (`config/model_registry.yaml`)

```yaml
models:
  - name: "mistral-7b-instruct"
    path: "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    type: "chat"
    tags: ["general", "reasoning", "creative"]
```

**Fields:**
- `name`: Display name for the model
- `path`: Path to GGUF file (relative or absolute)
- `type`: Model category (chat, code, summarization, etc.)
- `tags`: Keywords for task matching

### Evaluation Sets (`evals/*.yaml`)

```yaml
task: summarization
description: Test summarization ability

cases:
  - input: "Long text to summarize..."
    expected: "key_word"
```

**Fields:**
- `task`: Task type
- `description`: Brief description
- `cases`: List of test cases with input and expected output

## Project Structure

```
llm_evaluator/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── build.spec             # PyInstaller spec file
├── build.sh/bat           # Build scripts
├── ui/
│   └── main_window.py     # Main UI window
├── core/
│   ├── model_manager.py   # Model loading & inference
│   ├── evaluator.py       # Evaluation logic
│   └── config_manager.py  # Configuration handling
├── config/
│   ├── model_registry.yaml  # Model definitions
│   └── config.yaml          # App state (auto-generated)
├── evals/
│   ├── summarization.yaml
│   ├── classification.yaml
│   └── creative.yaml
└── models/                # Place GGUF files here
    └── (your .gguf files)
```

## Building Standalone Executable

### Using PyInstaller

**Linux/Mac:**
```bash
chmod +x build.sh
./build.sh
```

**Windows:**
```batch
build.bat
```

The executable will be in `dist/LLM_Evaluator/`

### Manual Build

```bash
pip install pyinstaller
pyinstaller build.spec
```

## Customization

### Adding Custom Task Categories

Edit the task category dropdown in `ui/main_window.py`:

```python
self.task_category.addItems([
    "auto-detect", "summarization", "classification",
    "your-custom-task"  # Add here
])
```

### Creating Custom Evaluation Sets

Create a new YAML file in `evals/`:

```yaml
task: my_custom_task
description: My custom evaluation

cases:
  - input: "Test input"
    expected: "expected_output"
```

### Modifying Model Parameters

Default parameters can be changed in the Settings tab or directly in the code:

```python
# ui/main_window.py - create_settings_tab()
self.n_threads.setValue(8)     # CPU threads
self.n_ctx.setValue(4096)      # Context window
self.temperature.setValue(0.7)  # Temperature
self.max_tokens.setValue(1024)  # Max output tokens
```

## Troubleshooting

### Model Loading Issues

**Error: "DLL load failed" or "Library not found"**
- Reinstall llama-cpp-python with proper build flags (see Installation)
- Ensure you have Visual Studio Build Tools (Windows) or Xcode (Mac)

**Error: "Model file not found"**
- Check the path in `model_registry.yaml`
- Use absolute paths if relative paths don't work

### Memory Issues

**Out of memory errors:**
- Use smaller quantized models (Q4_K_M instead of Q8_0)
- Reduce `n_ctx` (context window)
- Close other applications

### Performance Issues

**Slow inference:**
- Reduce `max_tokens`
- Increase `n_threads` (but don't exceed CPU cores)
- Enable GPU acceleration (see Installation)
- Use smaller models

## Advanced Usage

### Running Without GUI (Programmatic)

```python
from core.model_manager import ModelManager
from core.config_manager import ConfigManager

config_mgr = ConfigManager()
model_mgr = ModelManager(config_mgr)

# Load model
model_mgr.load_model("models/your-model.gguf")

# Generate
messages = [{"role": "user", "content": "Hello!"}]
result = model_mgr.generate(messages)
print(result['text'])
```

### Batch Processing

```python
from core.evaluator import Evaluator

evaluator = Evaluator(model_mgr)
results = evaluator.run_eval_set(
    "evals/summarization.yaml",
    matched_models
)
```

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Streaming response support
- [ ] Model quantization tools
- [ ] Advanced evaluation metrics (BLEU, ROUGE)
- [ ] Multi-language support
- [ ] Dark theme
- [ ] Model download manager
- [ ] Chart/graph visualizations

## License

MIT License - feel free to use and modify for your purposes.

## Acknowledgments

- Built with [PySide6](https://doc.qt.io/qtforpython/)
- LLM inference via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Inspired by the need for simple, local LLM evaluation tools

## FAQ

**Q: Can I use models from Ollama?**
A: This tool uses GGUF files directly. You can find the GGUF files in Ollama's model directory and reference them.

**Q: What model sizes work best?**
A: 7B models with Q4_K_M quantization offer the best balance of quality and speed on consumer hardware.

**Q: Can I run multiple models simultaneously?**
A: Currently, models are loaded sequentially. Parallel execution would require significant RAM.

**Q: How do I add my fine-tuned model?**
A: Convert your model to GGUF format, place it in `models/`, and add an entry to `model_registry.yaml`.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section

---

**Happy Evaluating! 🚀**
