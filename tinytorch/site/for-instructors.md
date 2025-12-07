#  For Instructors & TAs

**Complete guide for teaching ML Systems Engineering with TinyTorch**

<div style="background: #f0f9ff; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; margin: 2rem 0;">
<h3 style="margin: 0 0 0.5rem 0;">📋 Quick Course Assessment</h3>
<p style="margin: 0.5rem 0;">
<strong>Duration:</strong> 14-16 weeks (flexible pacing)<br>
<strong>Prerequisites:</strong> Python + basic linear algebra<br>
<strong>Student Outcome:</strong> Complete ML framework supporting vision AND language models<br>
<strong>Grading:</strong> 70% auto-graded (NBGrader), 30% manual (systems thinking)
</p>
</div>

## For Instructors: Course Setup

### 30-Minute Initial Setup

**Step 1: Environment Setup (10 minutes)**
```bash
# Clone repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
pip install nbgrader

# Verify installation
tito system health
```

**Step 2: Initialize Grading (10 minutes)**
```bash
# Setup NBGrader integration
tito grade setup

# Verify grading commands
tito grade --help
```

**Step 3: Prepare First Assignment (10 minutes)**
```bash
# Generate instructor version (with solutions)
tito grade generate 01_tensor

# Create student version (solutions removed)
tito grade release 01_tensor

# Student assignments ready in: release/01_tensor/
```

### Assignment Workflow

TinyTorch wraps NBGrader behind simple `tito grade` commands:

**1. Prepare Assignments**
```bash
# Generate instructor version with solutions
tito grade generate MODULE_NAME

# Create student version (auto-removes solutions)
tito grade release MODULE_NAME
```

**2. Distribute to Students**
- **Option A: GitHub Classroom** (recommended)
  - Create assignment repository from TinyTorch template
  - Students clone and work in their repos
  - Automatic submission via GitHub
  
- **Option B: Direct Distribution**
  - Share `release/` directory contents
  - Students download and submit via LMS

**3. Collect Submissions**
```bash
# Collect all students
tito grade collect MODULE_NAME

# Or specific student
tito grade collect MODULE_NAME --student student_id
```

**4. Auto-Grade**
```bash
# Grade all submissions
tito grade autograde MODULE_NAME

# Grade specific student
tito grade autograde MODULE_NAME --student student_id
```

**5. Manual Review**
```bash
# Open browser-based grading interface
tito grade manual MODULE_NAME
```

**6. Generate Feedback**
```bash
# Create feedback files for students
tito grade feedback MODULE_NAME
```

**7. Export Grades**
```bash
# Export all grades to CSV
tito grade export

# Or specific module
tito grade export --module MODULE_NAME --output grades.csv
```

### Grading Components

**Auto-Graded (70%)**
- Code implementation correctness
- Test passing
- Function signatures
- Output validation
- Edge case handling

**Manually Graded (30%)**
- ML Systems Thinking questions (3 per module)
- Each question: 10 points
- Focus on understanding, not perfection

### Grading Rubric for Systems Thinking Questions

| Points | Criteria |
|--------|----------|
| 9-10 | Deep understanding, specific code references, discusses systems implications (memory, scaling, trade-offs) |
| 7-8 | Good understanding, some code references, basic systems thinking |
| 5-6 | Surface understanding, generic response, limited systems perspective |
| 3-4 | Attempted but misses key concepts |
| 0-2 | No attempt or completely off-topic |

**What to Look For:**
- References to actual implemented code
- Memory/performance analysis
- Scaling considerations
- Production system comparisons (PyTorch, TensorFlow)
- Understanding of trade-offs

### Sample 16-Week Schedule

| Week | Module | Focus | Teaching Notes |
|------|--------|-------|----------------|
| 1 | 01 Tensor | Data Structures, Memory | Demo: memory profiling, copying behavior |
| 2 | 02 Activations | Non-linearity, Stability | Demo: gradient vanishing/exploding |
| 3 | 03 Layers | Neural Components | Demo: forward/backward passes |
| 4 | 04 Losses | Optimization Objectives | Demo: loss landscapes |
| 5 | 05 Autograd | Auto Differentiation | ⚠ Most challenging - allocate extra TA hours |
| 6 | 06 Optimizers | Training Algorithms | Demo: optimizer comparisons |
| 7 | 07 Training | Complete Training Loop | Milestone: Train first network! |
| 8 | **Midterm Project** | Build and Train Network | Assessment: End-to-end system |
| 9 | 08 DataLoader | Data Pipeline | Demo: batching, shuffling |
| 10 | 09 Spatial | Convolutions, CNNs | ⚠ High demand - O(N²) complexity |
| 11 | 10 Tokenization | Text Processing | Demo: vocabulary building |
| 12 | 11 Embeddings | Word Representations | Demo: embedding similarity |
| 13 | 12 Attention | Attention Mechanisms | ⚠ Moderate-high demand |
| 14 | 13 Transformers | Transformer Architecture | Milestone: Text generation! |
| 15 | 14-19 Optimization | Profiling, Quantization | Focus on production trade-offs |
| 16 | 20 Capstone | **Torch Olympics** | Final Competition |

### Critical Modules (Extra TA Support Needed)

1. **Module 05: Autograd** - Most conceptually challenging
   - Pre-record debugging walkthroughs
   - Create FAQ document
   - Schedule additional office hours

2. **Module 09: Spatial (CNNs)** - Complex nested loops
   - Focus on memory profiling
   - Loop optimization strategies
   - Padding/stride calculations

3. **Module 12: Attention** - Attention mechanisms
   - Scaling factor importance
   - Numerical stability
   - Positional encoding issues

### Module-Specific Teaching Notes

**Module 01: Tensor**
- **Key Concept:** Memory layout is crucial for ML performance
- **Demo:** Show `memory_footprint()`, compare copying vs views
- **Watch For:** Students hardcoding float32 instead of using `dtype`

**Module 05: Autograd**
- **Key Concept:** Computational graphs enable deep learning
- **Demo:** Visualize computational graphs, show gradient flow
- **Watch For:** Gradient shape mismatches, disconnected graphs

**Module 09: Spatial (CNNs)**
- **Key Concept:** O(N²) operations become bottlenecks
- **Demo:** Profile convolution memory usage
- **Watch For:** Index out of bounds, missing padding

**Module 12: Attention**
- **Key Concept:** Attention is compute-intensive but powerful
- **Demo:** Profile attention with different sequence lengths
- **Watch For:** Missing scaling factor (1/√d_k), softmax errors

**Module 20: Capstone**
- **Key Concept:** Production requires optimization across ALL components
- **Project:** Torch Olympics Competition (4 tracks: Speed, Compression, Accuracy, Efficiency)

### Assessment Strategy

**Continuous Assessment (70%)**
- Module completion: 4% each × 16 modules = 64%
- Checkpoint achievements: 6%

**Projects (30%)**
- Midterm: Build and train CNN on CIFAR-10 (15%)
- Final: Torch Olympics Competition (15%)

### Tracking Student Progress

```bash
# Check specific student
tito checkpoint status --student student_id

# Export class progress
tito checkpoint export --output class_progress.csv

# View module completion rates
tito module status --comprehensive
```

**Identify Struggling Students:**
- Missing checkpoint achievements
- Low scores on systems thinking questions
- Incomplete module submissions
- Late milestone completions

---

## For Teaching Assistants: Student Support

### TA Preparation

**Develop Deep Familiarity With:**
1. **Module 05: Autograd** - Most student questions
2. **Module 09: CNNs** - Complex implementation
3. **Module 13: Transformers** - Advanced concepts

**Preparation Process:**
1. Complete all three critical modules yourself
2. Introduce bugs intentionally
3. Practice debugging scenarios
4. Review past student submissions

### Common Student Errors

#### Module 05: Autograd

**Error 1: Gradient Shape Mismatches**
- Symptom: `ValueError: shapes don't match for gradient`
- Cause: Incorrect gradient accumulation
- Debug: Check gradient shapes match parameter shapes

**Error 2: Disconnected Computational Graph**
- Symptom: Gradients are None or zero
- Cause: Operations not tracked
- Debug: Verify `requires_grad=True`, check graph construction

**Error 3: Broadcasting Failures**
- Symptom: Shape errors during backward pass
- Cause: Incorrect handling of broadcasted operations
- Debug: Check gradient accumulation for broadcasted dims

#### Module 09: CNNs (Spatial)

**Error 1: Index Out of Bounds**
- Symptom: `IndexError` in convolution loops
- Cause: Incorrect padding/stride calculations
- Debug: Verify output shape calculations

**Error 2: Memory Issues**
- Symptom: Out of memory errors
- Cause: Creating unnecessary intermediate arrays
- Debug: Profile memory, look for unnecessary copies

#### Module 13: Transformers

**Error 1: Attention Scaling Issues**
- Symptom: Attention weights don't sum to 1
- Cause: Missing softmax or incorrect scaling
- Debug: Verify softmax, check scaling factor (1/√d_k)

**Error 2: Positional Encoding Errors**
- Symptom: Model doesn't learn positional information
- Cause: Incorrect implementation
- Debug: Verify sinusoidal patterns

### Debugging Strategy

**Guide students with questions, not answers:**

1. "What error message are you seeing?" - Read full traceback
2. "What did you expect to happen?" - Clarify mental model
3. "What actually happened?" - Compare expected vs actual
4. "What have you tried?" - Avoid repeating failed approaches
5. "Can you test with a simpler case?" - Reduce complexity

### Productive vs Unproductive Struggle

**Productive Struggle (encourage):**
- Trying different approaches
- Making incremental progress
- Understanding error messages
- Passing more tests over time

**Unproductive Frustration (intervene):**
- Repeated identical errors
- Random code changes
- Unable to articulate the problem
- No progress after 30+ minutes

### Office Hour Patterns

**Expected Demand Spikes:**

- **Weeks 5-6 (Module 05: Autograd)**: Highest demand
  - Schedule 2× TA capacity
  - Pre-record debugging walkthroughs
  - Create FAQ document

- **Week 10 (Module 09: CNNs)**: High demand
  - Focus on memory profiling
  - Loop optimization
  - Padding/stride help

- **Week 13 (Module 13: Transformers)**: Moderate-high
  - Attention debugging
  - Scaling problems
  - Architecture questions

### Manual Review Focus

While auto-grading handles 70%, focus manual review on:

1. **Code Quality**
   - Readability
   - Design choices
   - Documentation

2. **Edge Case Handling**
   - Appropriate checks
   - Error handling
   - Boundary conditions

3. **Systems Thinking**
   - Memory analysis
   - Performance understanding
   - Scaling awareness

### Teaching Tips

1. **Encourage Exploration** - Let students try different approaches
2. **Connect to Production** - Reference PyTorch equivalents
3. **Make Systems Visible** - Profile memory, analyze complexity together
4. **Build Confidence** - Acknowledge progress and validate understanding

---

## Troubleshooting Common Issues

### Environment Problems
```bash
# Student fix:
tito system health
tito system reset
```

### Module Import Errors
```bash
# Rebuild package
tito module complete N
```

### Test Failures
```bash
# Detailed test output
tito module test N --verbose
```

### NBGrader Issues

**Database Locked**
```bash
# Clear and reinitialize
rm gradebook.db
tito grade setup
```

**Missing Submissions**
```bash
# Check submission directory
ls submitted/*/MODULE/
```

---

## Additional Resources

- **[Complete Course Structure](chapters/00-introduction.md)** - Full curriculum overview
- **[Student Getting Started](getting-started.md)** - Send this to students
- **[CLI Documentation](tito/overview.md)** - Detailed command reference
- **[Troubleshooting Guide](tito/troubleshooting.md)** - Common issues and solutions
- **[GitHub Discussions](https://github.com/mlsysbook/TinyTorch/discussions)** - Community support
- **[Issue Tracker](https://github.com/mlsysbook/TinyTorch/issues)** - Report bugs

---

## Contact & Support

**Need help?**
- Open an issue on GitHub
- Join discussions forum
- Email: support@mlsysbook.ai (if available)

**Contributing:**
- Sample solutions welcome
- Teaching material improvements
- Bug fixes and enhancements

---

<div style="background: #d4edda; border: 1px solid #28a745; padding: 1.5rem; border-radius: 0.5rem; margin: 2rem 0;">
<h3 style="margin: 0 0 0.5rem 0; color: #155724;">✓ You're Ready to Teach!</h3>
<p style="margin: 0; color: #155724;">
With NBGrader integration, automated grading, and comprehensive teaching materials, you have everything needed to run a successful ML systems course.
</p>
</div>

