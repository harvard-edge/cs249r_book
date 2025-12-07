# Tiny🔥Torch Documentation Site

This directory contains the TinyTorch course website and documentation.

## 🚀 Building the Site

All builds are managed through the Makefile:

```bash
cd tinytorch/site

# Build HTML website
make html

# Build PDF (requires LuaLaTeX)
make pdf

# Clean build artifacts
make clean

# Install dependencies
make install
```

## 📚 User Documentation

- **`STUDENT_QUICKSTART.md`** - Getting started guide for students
- **`INSTRUCTOR_GUIDE.md`** - Setup and grading guide for instructors
- **`cifar10-training-guide.md`** - Complete guide to achieving the north star goal (75% CIFAR-10 accuracy)

## 🔧 Development Documentation

- **`tinytorch-assumptions.md`** - **CRITICAL**: TinyTorch complexity framework and implementation guidelines

### Development Standards
- **`development/module-rules.md`** - Module development standards and patterns

### NBGrader Integration
- **`nbgrader/NBGrader_Quick_Reference.md`** - Daily use commands and workflow
- **`nbgrader/NBGRADER_STYLE_GUIDE.md`** - Style guide for NBGrader cells
- **`nbgrader/NBGrader_Text_Response_Technical_Implementation.md`** - Technical implementation details

---

**Start here**:
- **Students**: Read `STUDENT_QUICKSTART.md`
- **Instructors**: Read `INSTRUCTOR_GUIDE.md`
- **Developers**: Read `tinytorch-assumptions.md` FIRST, then `development/module-rules.md`