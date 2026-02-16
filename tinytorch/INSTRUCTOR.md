# 👩‍🏫 TinyTorch Instructor Guide

Complete guide for teaching ML Systems Engineering with TinyTorch.

## 🎯 Course Overview

TinyTorch teaches ML systems engineering through building, not just using. Students construct a complete ML framework from tensors to transformers, understanding memory, performance, and scaling at each step.

## 🛠️ Instructor Setup

### **1. Initial Setup**
```bash
# Clone and setup
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/tinytorch

# Virtual environment (MANDATORY)
python -m venv .venv
source .venv/bin/activate

# Install with instructor tools
pip install -r requirements.txt
pip install nbgrader

# Setup grading infrastructure
tito nbgrader init
```

### **2. Verify Installation**
```bash
tito system health
# Should show all green checkmarks

tito nbgrader
# Should show available NBGrader commands
```

## 📝 Assignment Workflow

### **⚠️ Experimental Feature**
The NBGrader integration is under active development. Use for testing only.

### **Using NBGrader via Tito**
We provide `tito nbgrader` commands for grading workflows.

### **1. Prepare Assignments**
```bash
# Generate instructor version (with solutions)
tito nbgrader generate 01_tensor

# Create student version (solutions removed)
tito nbgrader release 01_tensor

# Student version will be in: assignments/release/01_tensor/
```

### **2. Distribute to Students**
```bash
# Option A: GitHub Classroom (recommended)
# 1. Create assignment repository from TinyTorch
# 2. Remove solutions from modules
# 3. Students clone and work

# Option B: Direct distribution
# Share the release/ directory contents
```

### **3. Collect Submissions**
```bash
# Collect all students
tito nbgrader collect 01_tensor

# Or specific student
tito nbgrader collect 01_tensor --student student_id
```

### **4. Auto-Grade**
```bash
# Grade all submissions
tito nbgrader autograde 01_tensor

# Grade specific student
tito nbgrader autograde 01_tensor --student student_id
```

### **5. Manual Review**
```bash
# Use NBGrader's formgrader for manual review
# This launches a web interface for:
# - Reviewing ML Systems question responses
# - Adding feedback comments
# - Adjusting auto-grades
nbgrader formgrader
```

### **6. Generate Feedback**
```bash
# Create feedback files for students
tito nbgrader feedback 01_tensor
```

### **7. Export Grades**
```bash
# Export grades report
tito nbgrader report

# Or specific module
tito nbgrader report --module 01_tensor
```

## 📊 Grading Components

### **Auto-Graded (70%)**
- Code implementation correctness
- Test passing
- Function signatures
- Output validation

### **Manually Graded (30%)**
- ML Systems Thinking questions (3 per module)
- Each question: 10 points
- Focus on understanding, not perfection

### **Grading Rubric for ML Systems Questions**

| Points | Criteria |
|--------|----------|
| 9-10 | Demonstrates deep understanding, references specific code, discusses systems implications |
| 7-8 | Good understanding, some code references, basic systems thinking |
| 5-6 | Surface understanding, generic response, limited systems perspective |
| 3-4 | Attempted but misses key concepts |
| 0-2 | No attempt or completely off-topic |

**What to Look For:**
- References to actual implemented code
- Memory/performance analysis
- Scaling considerations
- Production system comparisons
- Understanding of trade-offs

## 📋 Sample Solutions for Grading Calibration

This section provides sample solutions to help calibrate grading standards. Use these as reference points when evaluating student submissions.

### Module 01: Tensor - Memory Footprint

**Excellent Solution (9-10 points)**:
```python
def memory_footprint(self):
    """Calculate tensor memory in bytes."""
    return self.data.nbytes
```
**Why Excellent**:
- Concise and correct
- Uses NumPy's built-in `nbytes` property
- Clear docstring
- Handles all tensor shapes correctly

**Good Solution (7-8 points)**:
```python
def memory_footprint(self):
    """Calculate memory usage."""
    return np.prod(self.data.shape) * self.data.dtype.itemsize
```
**Why Good**:
- Correct implementation
- Manually calculates (shows understanding)
- Works but less efficient than using `nbytes`
- Minor: docstring could be more specific

**Acceptable Solution (5-6 points)**:
```python
def memory_footprint(self):
    size = 1
    for dim in self.data.shape:
        size *= dim
    return size * 4  # Assumes float32
```
**Why Acceptable**:
- Correct logic but hardcoded dtype size
- Works for float32 but fails for other dtypes
- Shows understanding of memory calculation
- Missing proper dtype handling

### Module 06: Autograd - Backward Pass

**Excellent Solution (9-10 points)**:
```python
def backward(self, gradient=None):
    """Backward pass through computational graph."""
    if gradient is None:
        gradient = np.ones_like(self.data)

    self.grad = gradient

    if self.grad_fn is not None:
        # Compute gradients for inputs
        input_grads = self.grad_fn.backward(gradient)

        # Propagate to input tensors
        if isinstance(input_grads, tuple):
            for input_tensor, input_grad in zip(self.grad_fn.inputs, input_grads):
                if input_tensor.requires_grad:
                    input_tensor.backward(input_grad)
        else:
            if self.grad_fn.inputs[0].requires_grad:
                self.grad_fn.inputs[0].backward(input_grads)
```
**Why Excellent**:
- Handles both scalar and tensor gradients
- Properly checks `requires_grad` before propagating
- Handles tuple returns from grad_fn
- Clear variable names and structure

**Good Solution (7-8 points)**:
```python
def backward(self, gradient=None):
    if gradient is None:
        gradient = np.ones_like(self.data)
    self.grad = gradient
    if self.grad_fn:
        grads = self.grad_fn.backward(gradient)
        for inp, grad in zip(self.grad_fn.inputs, grads):
            inp.backward(grad)
```
**Why Good**:
- Correct logic
- Missing `requires_grad` check (minor issue)
- Assumes grads is always iterable (may fail for single input)
- Works for most cases but less robust

**Acceptable Solution (5-6 points)**:
```python
def backward(self, grad):
    self.grad = grad
    if self.grad_fn:
        self.grad_fn.inputs[0].backward(self.grad_fn.backward(grad))
```
**Why Acceptable**:
- Basic backward pass works
- Only handles single input (fails for multi-input operations)
- Missing None gradient handling
- Shows understanding but incomplete

### Module 09: Spatial - Convolution Implementation

**Excellent Solution (9-10 points)**:
```python
def forward(self, x):
    """Forward pass with explicit loops for clarity."""
    batch_size, in_channels, height, width = x.shape
    out_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
    out_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

    output = np.zeros((batch_size, self.out_channels, out_height, out_width))

    # Apply padding
    if self.padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                      (self.padding, self.padding)), mode='constant')

    # Explicit convolution loops
    for b in range(batch_size):
        for oc in range(self.out_channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    h_start = oh * self.stride
                    w_start = ow * self.stride
                    h_end = h_start + self.kernel_size
                    w_end = w_start + self.kernel_size

                    window = x[b, :, h_start:h_end, w_start:w_end]
                    output[b, oc, oh, ow] = np.sum(
                        window * self.weight[oc] + self.bias[oc]
                    )

    return Tensor(output, requires_grad=x.requires_grad)
```
**Why Excellent**:
- Clear output shape calculation
- Proper padding handling
- Explicit loops make O(kernel_size²) complexity visible
- Correct gradient tracking setup
- Well-structured and readable

**Good Solution (7-8 points)**:
```python
def forward(self, x):
    B, C, H, W = x.shape
    out_h = (H - self.kernel_size) // self.stride + 1
    out_w = (W - self.kernel_size) // self.stride + 1
    out = np.zeros((B, self.out_channels, out_h, out_w))

    for b in range(B):
        for oc in range(self.out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h = i * self.stride
                    w = j * self.stride
                    out[b, oc, i, j] = np.sum(
                        x[b, :, h:h+self.kernel_size, w:w+self.kernel_size]
                        * self.weight[oc]
                    ) + self.bias[oc]
    return Tensor(out)
```
**Why Good**:
- Correct implementation
- Missing padding support (works only for padding=0)
- Less clear variable names
- Missing requires_grad propagation

**Acceptable Solution (5-6 points)**:
```python
def forward(self, x):
    out = np.zeros((x.shape[0], self.out_channels, x.shape[2]-2, x.shape[3]-2))
    for b in range(x.shape[0]):
        for c in range(self.out_channels):
            for i in range(out.shape[2]):
                for j in range(out.shape[3]):
                    out[b, c, i, j] = np.sum(x[b, :, i:i+3, j:j+3] * self.weight[c])
    return Tensor(out)
```
**Why Acceptable**:
- Basic convolution works
- Hardcoded kernel_size=3 (not general)
- No stride or padding support
- Shows understanding but incomplete

### Module 12: Attention - Scaled Dot-Product Attention

**Excellent Solution (9-10 points)**:
```python
def forward(self, query, key, value, mask=None):
    """Scaled dot-product attention with numerical stability."""
    # Compute attention scores
    scores = np.dot(query, key.T) / np.sqrt(self.d_k)

    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    # Softmax with numerical stability
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Apply attention to values
    output = np.dot(attention_weights, value)

    return output, attention_weights
```
**Why Excellent**:
- Proper scaling factor (1/√d_k)
- Numerical stability with max subtraction
- Mask handling
- Returns both output and attention weights
- Clear and well-documented

**Good Solution (7-8 points)**:
```python
def forward(self, q, k, v):
    scores = np.dot(q, k.T) / np.sqrt(q.shape[-1])
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    return np.dot(weights, v)
```
**Why Good**:
- Correct implementation
- Missing numerical stability (may overflow)
- Missing mask support
- Works but less robust

**Acceptable Solution (5-6 points)**:
```python
def forward(self, q, k, v):
    scores = np.dot(q, k.T)
    weights = np.exp(scores) / np.sum(np.exp(scores))
    return np.dot(weights, v)
```
**Why Acceptable**:
- Basic attention mechanism
- Missing scaling factor
- Missing numerical stability
- Incorrect softmax (should be per-row)

### Grading Guidelines Using Sample Solutions

**When Evaluating Student Code**:

1. **Correctness First**: Does it pass all tests?
   - If no: Maximum 6 points (even if well-written)
   - If yes: Proceed to quality evaluation

2. **Code Quality**:
   - **Excellent (9-10)**: Production-ready, handles edge cases, well-documented
   - **Good (7-8)**: Correct and functional, minor improvements possible
   - **Acceptable (5-6)**: Works but incomplete or has issues

3. **Systems Thinking**:
   - **Excellent**: Discusses memory, performance, scaling implications
   - **Good**: Some systems awareness
   - **Acceptable**: Focuses only on correctness

4. **Common Patterns**:
   - Look for: Proper error handling, edge case consideration, documentation
   - Red flags: Hardcoded values, missing checks, unclear variable names

**Remember**: These are calibration examples. Adjust based on your course level and learning objectives. The goal is consistent evaluation, not perfection.

## 📚 Module Teaching Notes

### **Module 01: Tensor**
- **Focus**: Memory layout, data structures
- **Key Concept**: Understanding memory is crucial for ML performance
- **Demo**: Show memory profiling, copying behavior

### **Module 02: Activations**
- **Focus**: Vectorization, numerical stability
- **Key Concept**: Small details matter at scale
- **Demo**: Gradient vanishing/exploding

### **Module 04-05: Layers & Networks**
- **Focus**: Composition, parameter management
- **Key Concept**: Building blocks combine into complex systems
- **Project**: Build a small CNN

### **Module 06-07: Spatial & Attention**
- **Focus**: Algorithmic complexity, memory patterns
- **Key Concept**: O(N²) operations become bottlenecks
- **Demo**: Profile attention memory usage

### **Module 08-11: Training Pipeline**
- **Focus**: End-to-end system integration
- **Key Concept**: Many components must work together
- **Project**: Train a real model

### **Module 12-15: Production**
- **Focus**: Deployment, optimization, monitoring
- **Key Concept**: Academic vs production requirements
- **Demo**: Model compression, deployment

### **Module 16: TinyGPT**
- **Focus**: Framework generalization
- **Key Concept**: 70% component reuse from vision to language
- **Capstone**: Build a working language model

## 🎯 Learning Objectives

By course end, students should be able to:

1. **Build** complete ML systems from scratch
2. **Analyze** memory usage and computational complexity
3. **Debug** performance bottlenecks
4. **Optimize** for production deployment
5. **Understand** framework design decisions
6. **Apply** systems thinking to ML problems

## 📈 Tracking Progress

### **Individual Progress**
```bash
# Check specific student progress
tito module status --student student_id
```

### **Class Overview**
```bash
# Export all module progress
tito module status --export class_progress.csv
```

### **Identify Struggling Students**
Look for:
- Missing module completions
- Low scores on ML Systems questions
- Incomplete module submissions

## 💡 Teaching Tips

### **1. Emphasize Building Over Theory**
- Have students type every line of code
- Run tests immediately after implementation
- Break and fix things intentionally

### **2. Connect to Production Systems**
- Show PyTorch/TensorFlow equivalents
- Discuss real-world bottlenecks
- Share production war stories

### **3. Make Performance Visible**
```python
# Use profilers liberally
with TimeProfiler("operation"):
    result = expensive_operation()

# Show memory usage
print(f"Memory: {get_memory_usage():.2f} MB")
```

### **4. Encourage Systems Questions**
- "What would break at 1B parameters?"
- "How would you distribute this?"
- "What's the bottleneck here?"

## 🔧 Troubleshooting

### **Common Student Issues**

**Environment Problems**
```bash
# Student fix:
tito system health
tito module reset XX  # Reset specific module if needed
```

**Module Import Errors**
```bash
# Rebuild package
tito export --all
```

**Test Failures**
```bash
# Detailed test output
tito module test MODULE --verbose
```

### **NBGrader Issues**

**Database Locked**
```bash
# Clear NBGrader database
rm gradebook.db
tito nbgrader init
```

**Missing Submissions**
```bash
# Check submission directory
ls submitted/*/MODULE/
```

## 📊 Sample Schedule (16 Weeks)

| Week | Module | Focus |
|------|--------|-------|
| 1 | 01 Tensor | Data Structures, Memory |
| 2 | 02 Activations | Non-linearity Functions |
| 3 | 03 Layers | Neural Network Components |
| 4 | 04 Losses | Optimization Objectives |
| 5 | 05 DataLoader | Data Pipeline |
| 6 | 06 Autograd | Automatic Differentiation |
| 7 | 07 Optimizers | Training Algorithms |
| 8 | 08 Training | Complete Training Loop |
| 9 | Midterm Project | Build and Train Network |
| 10 | 09 Spatial | Convolutions, CNNs |
| 11 | 10 Tokenization | Text Processing |
| 12 | 11 Embeddings | Word Representations |
| 13 | 12 Attention | Attention Mechanisms |
| 14 | 13 Transformers | Transformer Architecture |
| 15 | 14-19 Optimization | Profiling, Quantization, etc. |
| 16 | 20 Capstone | Torch Olympics Competition |

## 🎓 Assessment Strategy

### **Continuous Assessment (70%)**
- Module completion: 4% each × 16 = 64%
- Checkpoint achievements: 6%

### **Projects (30%)**
- Midterm: Build and train CNN (15%)
- Final: Extend TinyGPT (15%)

## 📚 Additional Resources

- [MLSys Book](https://mlsysbook.ai) - Companion textbook
- [Course Discussions](https://github.com/harvard-edge/cs249r_book/discussions)
- [Issue Tracker](https://github.com/harvard-edge/cs249r_book/issues)

---

**Need help? Open an issue or contact the TinyTorch team!**
