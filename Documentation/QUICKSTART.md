# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies (2 minutes)

```bash
cd llm_evaluator
pip install -r requirements.txt
```

### 2. Download a Test Model (2 minutes)

Download a small model for testing. Recommended starter:

**Phi-2 (2.7B parameters, ~1.6GB)**
- Go to: https://huggingface.co/TheBloke/phi-2-GGUF
- Download: `phi-2.Q4_K_M.gguf`
- Place in `models/` folder

Alternative quick models:
- TinyLlama-1.1B (~600MB)
- Mistral-7B-Instruct (~4GB)

### 3. Update Registry (30 seconds)

Edit `config/model_registry.yaml` and add your model:

```yaml
models:
  - name: "phi-2"
    path: "models/phi-2.Q4_K_M.gguf"
    type: "chat"
    tags: ["general", "reasoning", "small"]
```

### 4. Run the App (30 seconds)

```bash
python main.py
```

## First Evaluation

### Test 1: Single Prompt

1. Go to **Settings** tab
2. Click **Browse** and select your model
3. Click **Load Model** (wait ~10 seconds)
4. Go to **Prompt & Run** tab
5. Enter in User Prompt: `Explain quantum computing in one sentence`
6. Click **Run on Selected Model**
7. View response and stats

### Test 2: Model Comparison

1. Go to **Task & Models** tab
2. Enter task: `creative writing`
3. Click **Discover Matching Models**
4. Go to **Prompt & Run** tab
5. Enter prompt: `Write a haiku about coding`
6. Click **Run on All Matched Models**
7. View comparison in **Evaluations** tab

### Test 3: Evaluation Set

1. Go to **Evaluations** tab
2. Click **Browse** and select `evals/creative.yaml`
3. Click **Run Evaluation**
4. View results table
5. Click **Export Results to CSV**

## Tips for Best Results

### Model Selection
- Start with 3B-7B parameter models
- Q4_K_M quantization is the sweet spot
- Match model type to your task

### Performance Tuning
- **Threads**: Set to your CPU cores minus 1
- **Context**: Start with 2048, increase if needed
- **Temperature**: 0.7 for balanced, 0.3 for focused, 1.0 for creative
- **Max Tokens**: 256-512 for most tasks

### Task Matching
Use specific keywords in task description:
- "summarize legal documents" → matches "legal", "summarization"
- "generate Python code" → matches "code", "programming"
- "creative story writing" → matches "creative", "writing"

## Common First-Time Issues

### Issue: Model won't load
**Solution**: Check the path in registry is correct (try absolute path)

### Issue: Very slow inference
**Solution**: Reduce max_tokens to 256, ensure you're using Q4_K_M quantization

### Issue: Out of memory
**Solution**: Use smaller model or reduce n_ctx to 1024

### Issue: No models matched
**Solution**: Add more tags to your models in the registry

## Next Steps

1. **Add more models** to compare performance
2. **Create custom eval sets** for your use case
3. **Fine-tune parameters** for your hardware
4. **Export results** and analyze patterns

## Building Executable

Once comfortable with the app:

```bash
./build.sh  # Linux/Mac
build.bat   # Windows
```

Executable will be in `dist/LLM_Evaluator/`

---

Need help? Check README.md for detailed documentation.
