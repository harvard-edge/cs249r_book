# Teaching Assistant Guide for TinyTorch

Complete guide for TAs supporting TinyTorch courses, covering common student errors, debugging strategies, and effective support techniques.

## 🎯 TA Preparation

### Critical Modules for Deep Familiarity

TAs should develop deep familiarity with modules where students commonly struggle:

1. **Module 05: Autograd** - Most conceptually challenging
2. **Module 09: CNNs (Spatial)** - Complex nested loops and memory patterns
3. **Module 13: Transformers** - Attention mechanisms and scaling

### Preparation Process

1. **Complete modules yourself** - Implement all three critical modules
2. **Introduce bugs intentionally** - Understand common error patterns
3. **Practice debugging** - Work through error scenarios
4. **Review student submissions** - Familiarize yourself with common mistakes

## 🐛 Common Student Errors

### Module 05: Autograd

#### Error 1: Gradient Shape Mismatches
**Symptom**: `ValueError: shapes don't match for gradient`
**Common Cause**: Incorrect gradient accumulation or shape handling
**Debugging Strategy**:
- Check gradient shapes match parameter shapes
- Verify gradient accumulation logic
- Look for broadcasting issues

**Example**:
```python
# Wrong: Gradient shape mismatch
param.grad = grad  # grad might be wrong shape

# Right: Ensure shapes match
assert grad.shape == param.shape
param.grad = grad
```

#### Error 2: Disconnected Computational Graph
**Symptom**: Gradients are None or zero
**Common Cause**: Operations not tracked in computational graph
**Debugging Strategy**:
- Verify `requires_grad=True` on input tensors
- Check that operations create new Tensor objects
- Ensure backward() is called on leaf nodes

**Example**:
```python
# Wrong: Graph disconnected
x = Tensor([1, 2, 3])  # requires_grad=False by default
y = x * 2
y.backward()  # No gradients!

# Right: Enable gradient tracking
x = Tensor([1, 2, 3], requires_grad=True)
y = x * 2
y.backward()  # Gradients flow correctly
```

#### Error 3: Broadcasting Failures
**Symptom**: Shape errors during backward pass
**Common Cause**: Incorrect handling of broadcasted operations
**Debugging Strategy**:
- Understand NumPy broadcasting rules
- Check gradient accumulation for broadcasted dimensions
- Verify gradient shapes match original tensor shapes

### Module 09: CNNs (Spatial)

#### Error 1: Index Out of Bounds
**Symptom**: `IndexError` in convolution loops
**Common Cause**: Incorrect padding or stride calculations
**Debugging Strategy**:
- Verify output shape calculations
- Check padding logic
- Test with small examples first

#### Error 2: Memory Issues
**Symptom**: Out of memory errors
**Common Cause**: Creating unnecessary intermediate arrays
**Debugging Strategy**:
- Profile memory usage
- Look for unnecessary copies
- Optimize loop structure

### Module 13: Transformers

#### Error 1: Attention Scaling Issues
**Symptom**: Attention weights don't sum to 1
**Common Cause**: Missing softmax or incorrect scaling
**Debugging Strategy**:
- Verify softmax is applied
- Check scaling factor (1/sqrt(d_k))
- Test attention weights sum to 1

#### Error 2: Positional Encoding Errors
**Symptom**: Model doesn't learn positional information
**Common Cause**: Incorrect positional encoding implementation
**Debugging Strategy**:
- Verify sinusoidal patterns
- Check encoding is added correctly
- Test with simple sequences

## 🔧 Debugging Strategies

### Structured Debugging Questions

When students ask for help, guide them with questions rather than giving answers:

1. **What error message are you seeing?**
   - Read the full traceback
   - Identify the specific line causing the error

2. **What did you expect to happen?**
   - Clarify their mental model
   - Identify misconceptions

3. **What actually happened?**
   - Compare expected vs actual
   - Look for patterns

4. **What have you tried?**
   - Avoid repeating failed approaches
   - Build on their attempts

5. **Can you test with a simpler case?**
   - Reduce complexity
   - Isolate the problem

### Productive vs Unproductive Struggle

**Productive Struggle** (encourage):
- Trying different approaches
- Making incremental progress
- Understanding error messages
- Passing additional tests over time

**Unproductive Frustration** (intervene):
- Repeated identical errors
- Random code changes
- Unable to articulate the problem
- No progress after 30+ minutes

### When to Provide Scaffolding

Offer scaffolding modules when students reach unproductive frustration:

- **Before Autograd**: Numerical gradient checking module
- **Before Tensor Autograd**: Scalar autograd module
- **Before CNNs**: Simple 1D convolution exercises

## 📊 Office Hour Patterns

### Expected Demand Spikes

**Module 05 (Autograd)**: Highest demand
- Schedule additional TA capacity
- Pre-record debugging walkthroughs
- Create FAQ document

**Module 09 (CNNs)**: High demand
- Focus on memory profiling
- Loop optimization strategies
- Padding/stride calculations

**Module 13 (Transformers)**: Moderate-high demand
- Attention mechanism debugging
- Positional encoding issues
- Scaling problems

### Support Channels

1. **Synchronous**: Office hours, lab sessions
2. **Asynchronous**: Discussion forums, email
3. **Self-service**: Common errors documentation, FAQ

## 🎓 Grading Support

### Manual Review Focus Areas

While NBGrader automates 70-80% of assessment, focus manual review on:

1. **Code Clarity and Design Choices**
   - Is code readable?
   - Are design decisions justified?
   - Is the implementation clean?

2. **Edge Case Handling**
   - Does code handle edge cases?
   - Are there appropriate checks?
   - Is error handling present?

3. **Computational Complexity Analysis**
   - Do students understand complexity?
   - Can they analyze their code?
   - Do they recognize bottlenecks?

4. **Memory Profiling Insights**
   - Do students understand memory usage?
   - Can they identify memory issues?
   - Do they optimize appropriately?

### Grading Rubrics

See `INSTRUCTOR.md` for detailed grading rubrics for:
- ML Systems Thinking questions
- Code quality assessment
- Systems analysis evaluation

## 💡 Teaching Tips

### 1. Encourage Exploration
- Let students try different approaches
- Support learning from mistakes
- Celebrate incremental progress

### 2. Connect to Production
- Reference PyTorch equivalents
- Discuss real-world debugging scenarios
- Share production war stories

### 3. Make Systems Visible
- Profile memory usage together
- Analyze computational complexity
- Visualize computational graphs

### 4. Build Confidence
- Acknowledge when students are on the right track
- Validate their understanding
- Provide encouragement during struggle

## 📚 Resources

- **INSTRUCTOR.md**: Complete instructor guide with grading rubrics
- **Common Errors**: This document (expanded as needed)
- **Module Documentation**: Each module's ABOUT.md file
- **Student Forums**: Community discussion areas

## 🔄 Continuous Improvement

### Feedback Collection

- Track common errors in office hours
- Document new error patterns
- Update this guide regularly
- Share insights with instructor team

### TA Training

- Regular TA meetings
- Share debugging strategies
- Review student submissions together
- Practice debugging sessions

---

**Last Updated**: November 2024  
**For Questions**: See INSTRUCTOR.md or contact course instructor

