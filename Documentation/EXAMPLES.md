# Usage Examples

Practical examples for using the Purpose-Driven Local LLM Evaluator.

## Example 1: Evaluating Summarization Models

### Scenario
You need to find the best model for summarizing technical documentation.

### Steps

1. **Define the Task**
   - Go to "Task & Models" tab
   - Enter: "Summarize technical documentation and research papers"
   - Category: "summarization"
   - Click "Discover Matching Models"

2. **Review Matched Models**
   - System shows models tagged with "summarization"
   - Example: "legal-summarizer", "mistral-7b-instruct"

3. **Load Evaluation Set**
   - Go to "Evaluations" tab
   - Browse to `evals/summarization.yaml`
   - Click "Run Evaluation"

4. **Review Results**
   ```
   Model Name            | Pass/Fail | Latency | Rating
   ---------------------|-----------|---------|-------
   legal-summarizer     | 3/3       | 12.5s   | 10.0
   mistral-7b-instruct  | 2/3       | 8.2s    | 6.7
   ```

5. **Export Results**
   - Click "Export Results to CSV"
   - Save as `summarization_comparison.csv`

### Outcome
You identified that "legal-summarizer" has better accuracy but "mistral-7b-instruct" is faster.

---

## Example 2: Testing Creative Writing Capabilities

### Scenario
Compare multiple models for creative writing tasks.

### Steps

1. **Define Task**
   - Task: "Creative story and poem writing"
   - Category: "creative"

2. **Create Custom Prompt**
   - Go to "Prompt & Run" tab
   - System Prompt: "You are a creative writer focused on concise storytelling."
   - User Prompt: "Write a three-sentence science fiction story about time travel."

3. **Run on Multiple Models**
   - Click "Run on All Matched Models"
   - Wait for all models to complete

4. **Compare Outputs**
   - Review responses in comparison table
   - Note differences in creativity, coherence, and length

5. **Save Best Prompt**
   - Click "Save Prompt"
   - Save as `creative_time_travel.md`

### Sample Output Comparison

**Model A (Mistral-7B):**
> "Dr. Chen stepped through the portal and found herself in ancient Rome, but with one problem—she'd forgotten the return coordinates. As centurions approached, she realized her smartwatch, still connected to quantum satellites that wouldn't exist for 2000 years, was her only hope. With seconds to spare, she activated the emergency beacon, creating a paradox that would reshape history itself."

**Model B (Llama-2-7B):**
> "The time machine worked, but Sarah arrived in the wrong year. She was supposed to visit 2050, but she ended up in 1950 instead. Now she had to find a way to get back home without changing history."

### Analysis
- Model A: More creative, better vocabulary, engaging narrative
- Model B: Simpler, more direct, less detailed

---

## Example 3: Code Generation Comparison

### Scenario
Evaluate which model is best for Python code generation.

### Custom Eval Set

Create `evals/python_code.yaml`:

```yaml
task: code_generation
description: Python programming tasks

cases:
  - input: "Write a function to calculate fibonacci numbers recursively"
    expected: "fibonacci"

  - input: "Create a function to reverse a string"
    expected: "reverse"

  - input: "Write a function to check if a number is prime"
    expected: "prime"
```

### Steps

1. **Update Registry**
   Add a code model to `config/model_registry.yaml`:
   ```yaml
   - name: "codellama-7b"
     path: "models/codellama-7b-instruct.Q4_K_M.gguf"
     type: "code"
     tags: ["code", "programming", "python"]
   ```

2. **Define Task**
   - Task: "Python programming and code generation"
   - Category: "auto-detect"

3. **Run Evaluation**
   - Load `evals/python_code.yaml`
   - Run on matched models

4. **Review Code Quality**
   - Check for correct syntax
   - Verify logic correctness
   - Compare code readability

---

## Example 4: Parameter Optimization

### Scenario
Find optimal inference parameters for a specific model.

### Experiment: Temperature Testing

**Test 1: Temperature 0.3 (Focused)**
```
Settings:
- Temperature: 0.3
- Max Tokens: 256

Prompt: "Explain quantum computing"
Result: Technical, precise, somewhat repetitive
```

**Test 2: Temperature 0.7 (Balanced)**
```
Settings:
- Temperature: 0.7
- Max Tokens: 256

Prompt: "Explain quantum computing"
Result: Clear, varied, good balance
```

**Test 3: Temperature 1.2 (Creative)**
```
Settings:
- Temperature: 1.2
- Max Tokens: 256

Prompt: "Explain quantum computing"
Result: Very creative, sometimes off-topic
```

### Conclusion
Temperature 0.7 provides the best balance for general use.

---

## Example 5: Batch Processing Multiple Prompts

### Scenario
Test a model against multiple prompts quickly.

### Method

Create `prompts.md`:
```markdown
# System Prompt

You are a helpful assistant.

# User Prompt

1. What is machine learning?
2. Explain neural networks.
3. Describe deep learning.
```

### Process

1. Load prompt from file
2. Run on selected model
3. Copy first response
4. Modify prompt to second question
5. Run again
6. Repeat for all questions

### Optimization Tip
Create separate eval YAML files for different prompt sets to automate this.

---

## Example 6: Model Selection for Production

### Scenario
Choose the best model for a production application.

### Evaluation Criteria

1. **Accuracy** (from eval results)
2. **Speed** (latency)
3. **Resource Usage** (model size)
4. **Consistency** (multiple runs)

### Testing Process

**Round 1: Quick Test**
```
Models: Phi-2, Mistral-7B, Llama-2-7B
Prompts: 10 simple questions
Metric: Pass rate and latency
```

**Round 2: Deep Test**
```
Models: Top 2 from Round 1
Eval Set: Full 50-case evaluation
Metric: Detailed accuracy metrics
```

**Round 3: Stress Test**
```
Model: Winner from Round 2
Test: 100 prompts, various contexts
Metric: Consistency, error rate
```

### Sample Results Table

| Model | Pass Rate | Avg Latency | Size | Recommendation |
|-------|-----------|-------------|------|----------------|
| Phi-2 | 85% | 2.3s | 1.6GB | Development |
| Mistral-7B | 92% | 4.1s | 4.4GB | **Production** |
| Llama-2-7B | 88% | 4.8s | 3.8GB | Backup |

### Decision
Mistral-7B offers best accuracy-speed tradeoff for production.

---

## Example 7: Creating a Domain-Specific Evaluator

### Scenario
Evaluate models for medical question answering.

### Custom Setup

**1. Create Medical Eval Set** (`evals/medical.yaml`):
```yaml
task: medical_qa
description: Basic medical knowledge questions

cases:
  - input: "What is hypertension?"
    expected: "blood pressure"

  - input: "What causes diabetes?"
    expected: "insulin"

  - input: "What are symptoms of flu?"
    expected: "fever"
```

**2. Tag Models in Registry**:
```yaml
- name: "medllama"
  path: "models/medllama.gguf"
  type: "medical"
  tags: ["medical", "healthcare", "qa"]
```

**3. Run Specialized Evaluation**:
- Task: "Medical question answering"
- Load medical eval set
- Compare medical vs. general models

**4. Analyze Results**:
- Medical-specific models should score higher
- General models may provide less accurate medical info

---

## Example 8: Prompt Engineering Workflow

### Scenario
Iteratively improve a prompt for best results.

### Iteration Process

**Version 1 (Basic)**:
```
User: "Summarize this article"
Result: Generic, misses key points
```

**Version 2 (With Context)**:
```
System: "You are an expert summarizer."
User: "Summarize this article in 3 sentences"
Result: Better, but still generic
```

**Version 3 (Detailed)**:
```
System: "You are an expert summarizer. Focus on key findings and implications."
User: "Summarize this research article in 3 sentences, highlighting: 1) Main findings, 2) Methodology, 3) Impact"
Result: Precise, well-structured
```

### Tracking Improvements

Save each version:
- `prompt_v1.md`
- `prompt_v2.md`
- `prompt_v3.md`

Run each on same model, compare results.

---

## Example 9: Multi-Language Model Testing

### Scenario
Test models for multilingual capabilities.

### Setup

Create `evals/multilingual.yaml`:
```yaml
task: translation
description: Basic multilingual understanding

cases:
  - input: "Translate to Spanish: Hello, how are you?"
    expected: "Hola"

  - input: "Translate to French: Good morning"
    expected: "Bonjour"
```

### Testing
1. Run on general models
2. Compare which handles multiple languages better
3. Note: Specialized multilingual models perform best

---

## Example 10: Automated Daily Testing

### Scenario
Set up daily regression testing for model updates.

### Workflow

**Morning Routine**:
1. Open LLM Evaluator
2. Load standard eval set: `evals/daily_check.yaml`
3. Run on all models
4. Export results: `results_2024_01_15.csv`

**Weekly Analysis**:
```bash
# Compare weekly results
compare_results.py results_*.csv
```

**Benefits**:
- Track model performance over time
- Detect any degradation
- Validate model updates

---

## Tips for All Examples

### Performance Tips
1. Start with small models for testing
2. Use larger models for production
3. Adjust max_tokens to control response length
4. Use threading = CPU cores - 1

### Evaluation Tips
1. Create diverse eval sets
2. Test edge cases
3. Run multiple times for consistency
4. Compare apples-to-apples (same prompt, settings)

### Organization Tips
1. Name prompts descriptively
2. Version eval sets (v1, v2, etc.)
3. Keep results organized by date
4. Document findings in CSVs

---

## Common Patterns

### Pattern 1: Quick Test
```
Task Definition → Single Prompt → Run → Review
```

### Pattern 2: Comparison
```
Task Definition → Discover Models → Multi-Model Run → Compare Table → Export
```

### Pattern 3: Evaluation
```
Task Definition → Load Eval Set → Run Evaluation → Review Scores → Export
```

### Pattern 4: Optimization
```
Load Model → Adjust Parameters → Test Prompt → Iterate → Save Best Config
```

---

## Next Steps

After mastering these examples:
1. Create your own domain-specific eval sets
2. Build a library of optimized prompts
3. Establish baseline performance metrics
4. Automate regular testing workflows
5. Share findings with your team

---

**Happy Evaluating!** 🚀
