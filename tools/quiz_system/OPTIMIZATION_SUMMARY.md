# Quiz Optimization Summary

## Current Problems Identified

Based on analysis of 684 questions across 20 chapters:

1. **Severe MCQ Answer Imbalance**
   - B: 60.2% of all answers (97 out of 161)
   - A: 9.3% (severely underused)
   - D: 3.7% (severely underused)
   - Chi-square: 125.19 (threshold for balance: 7.815)

2. **Almost No CALC Questions**
   - Only 6 out of 684 questions (0.9%)
   - Technical chapters need 10-15% CALC questions
   - Missing opportunities in Optimizations, Training, HW Acceleration chapters

3. **Question Type Imbalance**
   - Too many MCQ/SHORT/TF (75% combined)
   - Not enough variety in ORDER, CALC, FILL

4. **No Knowledge Progression**
   - Questions don't build on prior chapter knowledge
   - Missing opportunities to reinforce earlier concepts

## Optimization Strategy

### Core Principle
These are **self-check quizzes for learning reinforcement**, not grading. They should:
- Help students verify their understanding
- Motivate continued learning
- Build on prior knowledge progressively
- Provide variety to maintain engagement

### Targeted Fixes

#### 1. MCQ Answer Balance
- Force rotation: A, D, C, A, D, C... (avoiding B)
- Explicit prompt instructions to use underused choices
- Track distribution across generation

#### 2. CALC Questions for Technical Chapters
**Optimizations Chapter**:
- Memory savings: "7B FP32 model → INT8 quantization = ?"
- Pruning impact: "70% sparsity → theoretical speedup = ?"
- Compression ratio: "175B teacher → 7B student = ?"

**Training Chapter**:
- Batch memory: "Model 500MB + batch 32 + activation 10MB/sample = ?"
- Learning rate: "gradient 0.5, lr 0.001 → weight update = ?"
- Gradient accumulation: "micro-batch 8, accumulate 4 steps = ?"

**HW Acceleration Chapter**:
- Arithmetic intensity: "1000 FLOPs / 100 bytes = ?"
- Roofline model: "10 TFLOPS peak, 100 GB/s bandwidth = ?"
- GPU utilization: "8 GPUs, batch 256, 50ms/batch = ?"

#### 3. Knowledge Map Integration
- Track concepts introduced in each chapter
- Reference earlier concepts in later chapters
- Example: "Using gradient descent from Chapter 7, calculate..."
- Build complexity progressively through the book

#### 4. Question Type Targets

**Technical Chapters** (training, optimizations, hw_acceleration, benchmarking):
- CALC: 25-30%
- SHORT: 25-30%
- MCQ: 20-25%
- TF: 10-15%
- Others: 10-15%

**Conceptual Chapters** (responsible_ai, privacy_security, ai_for_good):
- SHORT: 30-35%
- TF: 20-25%
- MCQ: 20-25%
- FILL: 10-15%
- Others: 10-15%

## Implementation Approach

### Autonomous System
The system works by:
1. Analyzing existing quiz distribution
2. Modifying prompts to enforce corrections
3. Using GPT-4o for best quality
4. Tracking improvements across chapters
5. Building on prior knowledge using knowledge map

### Key Files Created

1. **`knowledge_map.py`**: Tracks concept progression through chapters
2. **`autoquiz_optimizer.py`**: Main optimization logic
3. **`run_optimization.sh`**: Executable script for batch processing
4. **`analyze_current_quizzes.py`**: Analysis tool

## Results Expected

After optimization:
- MCQ distribution: ~25% each for A, B, C, D (chi-square < 7.815)
- CALC questions: 10-15% for technical chapters, 20-25% where applicable
- Better variety: All 6 question types used appropriately
- Knowledge building: Questions reference and build on prior chapters
- Engagement: More interactive and varied assessment

## Next Steps

1. **Test Run** (3 priority chapters):
   ```bash
   ./tools/quiz_system/run_optimization.sh
   ```

2. **Review Results**: Check the optimized quizzes in `experiments/optimized_quizzes/`

3. **Apply to All Chapters** (if satisfied):
   ```bash
   python tools/scripts/genai/quizzes.py --mode generate -d quarto/contents/core/ --model gpt-4o
   ```

4. **Continuous Improvement**:
   - Monitor which questions students find most helpful
   - Adjust prompts based on patterns
   - Update knowledge map as chapters evolve

## Quality Metrics

Each optimized quiz is evaluated on:
- **Question Diversity** (25 points): Number of different question types
- **MCQ Balance** (25 points): Chi-square statistic < 7.815
- **CALC Presence** (25 points): Appropriate percentage for chapter type
- **Total Coverage** (25 points): Adequate number of questions

Target: 80+/100 quality score

## Pedagogical Benefits

1. **Balanced Assessment**: No single question type dominates
2. **Active Calculation**: Students practice quantitative reasoning
3. **Progressive Learning**: Concepts build systematically
4. **Self-Directed**: Students can identify knowledge gaps
5. **Motivating**: Variety maintains engagement

The optimization maintains the core purpose of self-check quizzes while fixing technical issues that were limiting their effectiveness.