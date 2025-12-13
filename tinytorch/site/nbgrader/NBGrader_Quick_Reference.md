# 📋 NBGrader Quick Reference Card

**TinyTorch + NBGrader Essential Commands for Instructors**


##  **One-Time Setup**
```bash
# 1. Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Initialize NBGrader
./bin/tito nbgrader init

# 3. Verify setup
./bin/tito system health
```


## 📝 **Weekly Assignment Workflow**

### **Monday: Release New Assignment**
```bash
# Generate assignment from TinyTorch module
./bin/tito nbgrader generate 03_activations

# Create student version
./bin/tito nbgrader release 03_activations

# Upload assignments/release/03_activations/03_activations.ipynb to LMS
```

### **Friday: Grade Submissions**
```bash
# After downloading student submissions to assignments/submitted/
./bin/tito nbgrader autograde 03_activations
./bin/tito nbgrader feedback 03_activations

# Return assignments/feedback/ files to students
```


## 🔧 **Essential Commands**

### **Status & Monitoring**
```bash
./bin/tito module status --comprehensive    # System health
./bin/tito nbgrader status                  # Assignment status
./bin/tito nbgrader analytics MODULE_NAME   # Student progress
```

### **Batch Operations**
```bash
./bin/tito nbgrader generate --all          # All assignments
./bin/tito nbgrader generate --range 01-04  # Module range
./bin/tito nbgrader autograde --all         # Grade everything
./bin/tito nbgrader feedback --all          # Generate all feedback
```

### **Export & Cleanup**
```bash
./bin/tito nbgrader report --format csv     # Export gradebook
./bin/tito clean                            # Clean temp files
```


## 📁 **Directory Structure**
```
assignments/
├── source/        # Generated assignments (git tracked)
├── release/       # Student versions (git tracked)
├── submitted/     # Student submissions (git ignored)
├── autograded/    # Graded submissions (git ignored)
└── feedback/      # Student feedback (git ignored)
```


## 🆘 **Quick Troubleshooting**
```bash
# Environment issues
source .venv/bin/activate
./bin/tito system health

# Module not found
ls modules/                          # Check available modules
./bin/tito nbgrader generate 02_tensor      # Use exact name

# Validation failures (normal for student notebooks)
# Students have unimplemented functions = expected behavior
```


## 📚 **Course Planning17 TinyTorch Modules:**
- **00-02**: Foundation (intro, setup, tensors)
- **03-07**: Building Blocks (activations, layers, dense, spatial, attention)  
- **08-11**: Training (dataloader, autograd, optimizers, training)
- **12-16**: Production (compression, kernels, benchmarking, mlops, capstone)

**Recommended Pacing:** 1 module per week = 16-week semester


** For complete details: See [Instructor Guide](book/instructor-guide.md)**