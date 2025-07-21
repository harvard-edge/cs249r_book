# LLM Cross-Reference Explanation Optimization Experiments

This framework provides systematic testing and evaluation of different LLM models and explanation lengths for generating cross-reference explanations in the ML Systems textbook.

## 🎯 What This Tests

### Model Comparison
- **Multiple Ollama Models**: Tests available models (qwen2.5, llama3.1, mistral, gemma2, etc.)
- **Consistent Evaluation**: All models tested on identical test cases
- **Performance Metrics**: Comprehensive scoring across 6 criteria

### Length Optimization  
- **5 Length Targets**: ultra_short (3-5 words) → extended (10-15 words)
- **Quality vs Brevity**: Finds optimal balance for margin space
- **Adherence Tracking**: Monitors if models follow length constraints

### LLM-as-Judge Evaluation
- **6 Evaluation Criteria**:
  - **Relevance**: Captures actual relationship between sections
  - **Clarity**: Clear and understandable for students  
  - **Conciseness**: Appropriate length without verbosity
  - **Usefulness**: Helps readers decide to follow the link
  - **Accuracy**: Factually correct about content domains
  - **Uniqueness**: Adds value beyond section titles

## 📁 Framework Components

```
scripts/llm_experiments/
├── test_cases.py          # 8 realistic cross-reference test cases
├── llm_judge.py           # LLM-based evaluation system
├── experiment_runner.py   # Main orchestration system
├── run_experiments.py     # Automated runner script
├── results/               # Experiment outputs (JSON files)
└── README.md             # This documentation
```

## 🚀 Quick Start

### Prerequisites
1. **Ollama installed and running**
2. **At least one model pulled** (recommended: `qwen2.5:7b`, `qwen2.5:32b`)
3. **Python packages**: `requests` (already in requirements)

### Run Experiments
```bash
cd scripts/llm_experiments
python run_experiments.py
```

The script will:
1. ✅ Check available Ollama models
2. 🧪 Test each model on standardized test cases
3. 📏 Optimize explanation length with best model
4. 📊 Generate data-driven recommendations
5. 💾 Save detailed results to `results/` directory

**Expected Duration**: 30-60 minutes depending on available models

## 📊 Understanding Results

### Key Output Files
- **`recommendations_latest.json`**: Main recommendations and analysis
- **`model_comparison_latest.json`**: Detailed model performance data
- **`length_optimization_latest.json`**: Optimal length analysis

### Sample Recommendation Output
```json
{
  "recommendations": {
    "model": {
      "recommended": "qwen2.5:14b",
      "confidence": "high",
      "reasoning": "qwen2.5:14b significantly outperforms other models with 8.2 average score vs 6.8 for worst model"
    },
    "length": {
      "recommended": "medium",
      "reasoning": "Length target 'medium' achieved highest score of 8.1 with 7.8 average words"
    }
  }
}
```

## 🧪 Test Cases Overview

The framework uses 8 carefully designed test cases covering:

1. **Introductory Connections**: AI Pervasiveness → Neural Networks
2. **Technical Depth**: Training → Hardware Acceleration  
3. **Advanced Topics**: Adversarial Attacks → Privacy
4. **Practical Applications**: Frameworks → Deployment
5. **Backward References**: Optimization → Training Fundamentals
6. **Complex Technical**: Transformers → Efficient Attention
7. **Real-world Applications**: Edge Computing → Deployment
8. **Short Content**: CNN Basics → Image Classification

Each test case includes realistic content excerpts and represents different difficulty levels and domains.

## 🔬 Methodology

### Model Testing Process
1. **Generate explanations** using each available model
2. **Evaluate with LLM judge** (powerful model like qwen2.5:32b)
3. **Score across 6 criteria** (1-10 scale)
4. **Calculate statistics** (mean, median, std dev)
5. **Rank models** by overall performance

### Length Optimization Process
1. **Use best-performing model** from comparison phase
2. **Test 5 length targets** on diverse test cases
3. **Measure quality vs length trade-offs**
4. **Check adherence** to length constraints
5. **Recommend optimal range**

### Evaluation Reliability
- **Low temperature** (0.1) for consistent judge scoring
- **Multiple test cases** per condition for statistical validity
- **Retry logic** for network reliability
- **Comprehensive criteria** covering all important aspects

## 📈 Expected Outcomes

The experiments will determine:

1. **Best Model**: Which Ollama model generates highest-quality explanations
2. **Optimal Length**: Sweet spot between informativeness and conciseness  
3. **Performance Gaps**: How much difference model choice makes
4. **Length Sensitivity**: How explanation length affects quality
5. **Deployment Recommendations**: Data-driven guidance for production

## 🛠️ Customization

### Adding New Models
Edit `experiment_runner.py`:
```python
self.test_models = [
    "qwen2.5:7b",
    "your-new-model:version",  # Add here
    # ... existing models
]
```

### Adding Test Cases
Edit `test_cases.py`:
```python
TEST_CASES.append({
    "id": "your_test_case",
    "source_title": "Source Section",
    "source_content": "Content...",
    "target_title": "Target Section", 
    "target_content": "Content...",
    "connection_type": "Preview",
    "domain": "your_domain",
    "difficulty": "intermediate"
})
```

### Adjusting Length Targets
Edit `test_cases.py`:
```python
LENGTH_TARGETS.append({
    "min_words": 5, 
    "max_words": 8, 
    "description": "custom_length"
})
```

## 🚨 Troubleshooting

### Common Issues

**No models available**
```bash
ollama list                    # Check installed models
ollama pull qwen2.5:7b        # Install a model
ollama serve                   # Start Ollama daemon
```

**Import errors**
```bash
cd scripts/llm_experiments
python -c "import requests; print('✅ OK')"
```

**Slow performance**
- Use smaller models for faster testing
- Reduce test cases in `run_model_comparison_experiment()`
- Increase timeouts in `_make_ollama_request()`

### Debug Mode
For detailed debugging, run individual components:
```python
from experiment_runner import ExperimentRunner
runner = ExperimentRunner()
models = runner.check_available_models()
print(f"Available models: {models}")
```

## 📝 Next Steps After Experiments

1. **Review recommendations** in `recommendations_latest.json`
2. **Update `cross_refs.py`** with optimal model
3. **Adjust prompt** for optimal explanation length
4. **Test on real data** with a small batch
5. **Deploy to production** if results are satisfactory

The framework provides the data-driven foundation for making informed decisions about LLM model selection and explanation generation parameters. 