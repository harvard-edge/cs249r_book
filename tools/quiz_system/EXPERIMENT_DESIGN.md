# AutoQuiz Experimental Design

## Overview

This document outlines 5 scientifically-designed experiments to optimize quiz generation for the ML Systems textbook, based on educational research and LLM best practices.

## Experimental Methodology

### Control Variables
- **Test Chapters**: Introduction (foundational), Benchmarking (technical), Responsible AI (conceptual)
- **Evaluation Metrics**: Standardized across all experiments
- **Human Evaluators**: 3 ML educators for consistency

### Independent Variables
Each experiment manipulates specific aspects of the generation process

### Dependent Variables
- Quality Score (0-100)
- Bloom's Distribution
- MCQ Balance (Chi-square statistic)
- Cognitive Load Score
- Generation Time
- Cost per Question

---

## Experiment 1: Baseline Performance Assessment

### Objective
Establish baseline metrics for the current quiz generation system to enable meaningful comparisons.

### Hypothesis
The existing system generates educationally valid questions but lacks optimization for Bloom's taxonomy distribution and cognitive load management.

### Method
```python
def experiment_1_baseline():
    chapters = ["introduction", "benchmarking", "responsible_ai"]
    
    for chapter in chapters:
        # Use existing quiz.py with default settings
        quiz = generate_quiz_baseline(chapter)
        
        # Measure all metrics
        metrics = {
            "bloom_distribution": analyze_bloom_levels(quiz),
            "mcq_balance": calculate_chi_square(quiz),
            "cognitive_load": measure_cognitive_load(quiz),
            "question_diversity": count_question_types(quiz),
            "quality_score": expert_evaluation(quiz)
        }
        
        save_results(f"exp1_baseline_{chapter}", metrics)
```

### Expected Outcomes
- Uneven Bloom's distribution (over-emphasis on Remember/Understand)
- MCQ imbalance (chi-square > 7.815)
- Variable cognitive load across questions
- Quality score: 60-70/100

---

## Experiment 2: Bloom's Taxonomy Optimization

### Objective
Test whether explicit Bloom's level targeting improves question quality and educational value.

### Hypothesis
Prompting with specific Bloom's levels will generate more diverse questions that better assess higher-order thinking.

### Method

#### Condition A: Standard Prompting (Control)
```python
standard_prompt = """
Generate quiz questions for this section...
"""
```

#### Condition B: Bloom's-Targeted Prompting
```python
bloom_prompt = """
Generate quiz questions targeting these Bloom's levels:
- 1 Remember question (factual recall)
- 1 Understand question (explain concepts)
- 1 Apply question (use in new situation)
- 1 Analyze question (break down systems)
- 1 Evaluate/Create question (judgment or design)

For each question, explicitly state the Bloom's level targeted.
"""
```

#### Condition C: Bloom's with Examples
```python
bloom_example_prompt = bloom_prompt + """

Examples by level:
Remember: "What is the time complexity of gradient descent?"
Understand: "Explain why batch normalization improves training stability"
Apply: "Calculate the memory requirements for deploying this model"
Analyze: "Compare the tradeoffs between model accuracy and latency"
Evaluate: "Design a deployment strategy for a resource-constrained environment"
"""
```

### Measurements
- Bloom's level accuracy (expert rating)
- Question quality scores
- Diversity index
- Student performance simulation

### Expected Outcomes
- Condition C > Condition B > Condition A for Bloom's distribution
- 30% improvement in higher-order questions
- Quality score increase of 15-20 points

---

## Experiment 3: Cognitive Load Optimization

### Objective
Minimize extraneous cognitive load while maintaining question validity.

### Hypothesis
Questions designed with cognitive load principles will be clearer and more accurately assess knowledge.

### Method

#### Condition A: Current Format
- No specific cognitive load constraints

#### Condition B: Cognitive Load Optimized
```python
cognitive_load_rules = {
    "max_words": 50,
    "no_negatives": True,
    "single_concept": True,
    "clear_stem": True,
    "parallel_options": True  # MCQ options have parallel structure
}
```

#### Condition C: Progressive Scaffolding
```python
scaffolding_rules = cognitive_load_rules + {
    "difficulty_progression": "easy_to_hard",
    "concept_building": True,  # Later questions build on earlier
    "context_provided": "minimal_necessary"
}
```

### Measurements
- Flesch Reading Ease Score
- Average completion time (simulated)
- Error rate analysis
- Cognitive load self-report scale

### Expected Outcomes
- 25% reduction in average question length
- 15% improvement in clarity scores
- Reduced error rates on complex topics

---

## Experiment 4: Domain-Specific Optimization

### Objective
Customize generation strategies for different chapter types in ML Systems.

### Hypothesis
Domain-aware prompting will generate more relevant and technically accurate questions.

### Method

#### Technical Chapters Strategy
```python
technical_prompt_additions = {
    "emphasis": ["calculations", "performance_metrics", "tradeoffs"],
    "question_types": ["CALC", "MCQ", "SHORT"],
    "examples": load_technical_examples(),
    "metrics": ["latency", "throughput", "memory", "accuracy"]
}
```

#### Conceptual Chapters Strategy
```python
conceptual_prompt_additions = {
    "emphasis": ["principles", "implications", "best_practices"],
    "question_types": ["SHORT", "TF", "MCQ"],
    "examples": load_conceptual_examples(),
    "focus": ["ethics", "design", "impact", "philosophy"]
}
```

#### Hybrid Approach
```python
def select_strategy(chapter_content):
    technical_score = count_technical_terms(chapter_content)
    conceptual_score = count_conceptual_terms(chapter_content)
    
    if technical_score > conceptual_score * 1.5:
        return technical_prompt_additions
    elif conceptual_score > technical_score * 1.5:
        return conceptual_prompt_additions
    else:
        return merge_strategies(technical, conceptual)
```

### Measurements
- Domain relevance scores
- Technical accuracy validation
- Question-content alignment
- Expert assessment by domain specialists

### Expected Outcomes
- 40% increase in CALC questions for technical chapters
- 30% increase in SHORT questions for conceptual chapters
- Improved technical accuracy (>95%)

---

## Experiment 5: Model Architecture Comparison

### Objective
Identify optimal models for different question types and chapter contexts.

### Hypothesis
Different models excel at different question types; a hybrid approach will optimize quality and cost.

### Method

#### Models to Test
1. **GPT-4o**: Current standard
2. **GPT-4o-mini**: Cost-effective alternative
3. **Claude-3-Opus**: Alternative architecture
4. **Llama-3.2 (Local)**: Privacy-preserving option
5. **Fine-tuned GPT-3.5**: BloomLLM approach

#### Test Protocol
```python
def model_comparison_test():
    test_sections = select_representative_sections()
    
    for model in models:
        for section in test_sections:
            # Generate with same prompt
            quiz = generate_with_model(model, section)
            
            # Measure performance
            metrics = {
                "quality": expert_evaluation(quiz),
                "bloom_accuracy": check_bloom_alignment(quiz),
                "generation_time": measure_time(quiz),
                "cost": calculate_cost(model, quiz),
                "consistency": measure_consistency(multiple_runs=3)
            }
            
            save_results(f"exp5_{model}_{section}", metrics)
```

#### Hybrid Strategy Testing
```python
def hybrid_model_strategy():
    strategies = {
        "quality_first": {
            "MCQ": "gpt-4o",
            "CALC": "gpt-4o",
            "SHORT": "claude-3-opus",
            "others": "gpt-4o-mini"
        },
        "cost_optimized": {
            "MCQ": "gpt-4o-mini",
            "CALC": "gpt-4o",
            "SHORT": "llama-local",
            "others": "llama-local"
        },
        "balanced": {
            "MCQ": "gpt-4o-mini",
            "CALC": "gpt-4o",
            "SHORT": "gpt-4o",
            "others": "gpt-4o-mini"
        }
    }
    
    return test_strategies(strategies)
```

### Measurements
- Quality scores by question type
- Cost per question
- Generation speed
- Consistency across runs
- Hallucination rate

### Expected Outcomes
- GPT-4o best for complex reasoning
- Local models viable for simple questions
- Hybrid approach reduces cost by 40% with <5% quality loss
- Fine-tuned models excel at Bloom's alignment

---

## Analysis Plan

### Statistical Methods
1. **ANOVA**: Compare means across conditions
2. **Chi-square Test**: MCQ distribution analysis
3. **Cohen's d**: Effect size calculations
4. **Inter-rater Reliability**: Kappa coefficient for human evaluations

### Success Criteria
Each experiment succeeds if:
- p < 0.05 for primary hypothesis
- Effect size (Cohen's d) > 0.5
- Quality score improvement > 10%

### Data Collection
```python
class ExperimentTracker:
    def __init__(self):
        self.results = {}
        self.timestamps = {}
        self.metadata = {}
    
    def record_experiment(self, exp_id, condition, metrics):
        self.results[exp_id] = {
            "condition": condition,
            "metrics": metrics,
            "timestamp": datetime.now(),
            "version": get_code_version()
        }
    
    def analyze_results(self):
        return {
            "summary_statistics": calculate_summary_stats(),
            "hypothesis_tests": run_statistical_tests(),
            "visualizations": generate_plots(),
            "recommendations": derive_recommendations()
        }
```

---

## Implementation Timeline

### Week 1: Baseline & Setup
- Day 1-2: Implement measurement framework
- Day 3-4: Run Experiment 1 (Baseline)
- Day 5: Analyze baseline results

### Week 2: Pedagogical Experiments
- Day 1-2: Run Experiment 2 (Bloom's)
- Day 3-4: Run Experiment 3 (Cognitive Load)
- Day 5: Comparative analysis

### Week 3: Domain & Model Tests
- Day 1-2: Run Experiment 4 (Domain-Specific)
- Day 3-4: Run Experiment 5 (Model Comparison)
- Day 5: Integration testing

### Week 4: Analysis & Implementation
- Day 1-2: Statistical analysis
- Day 3: Generate final recommendations
- Day 4-5: Implement optimal configuration

---

## Ethical Considerations

1. **Student Privacy**: No actual student data used
2. **Bias Detection**: Check for demographic biases in questions
3. **Accessibility**: Ensure questions are screen-reader friendly
4. **Transparency**: Document all AI involvement

---

## Expected Deliverables

1. **Comprehensive Results Report**: Statistical analysis of all experiments
2. **Optimal Configuration File**: Best settings for production
3. **Prompt Library**: Validated prompts for different scenarios
4. **Implementation Guide**: Step-by-step deployment instructions
5. **Future Research Recommendations**: Next steps for improvement

---

## Risk Mitigation

### Technical Risks
- **API Failures**: Implement retry logic and fallbacks
- **Model Changes**: Version-lock models during experiments
- **Data Loss**: Continuous backup of results

### Quality Risks
- **Human Evaluation Bias**: Use multiple evaluators and blind review
- **Overfitting**: Test on held-out chapters
- **Generalization**: Validate on different textbook sections

---

## Success Metrics Summary

The experimental program succeeds if:
1. Quality scores improve by >20% from baseline
2. Bloom's distribution matches target within 10%
3. MCQ chi-square < 7.815 achieved
4. Cognitive load reduced by >25%
5. Cost per question reduced by >30%
6. Generation time < 30 seconds per section

This experimental framework provides a scientific approach to optimizing quiz generation while maintaining pedagogical integrity.