# Tiny\raisebox{-0.1em}{\includegraphics[height=1em]{../_static/logos/fire-emoji.png}}Torch Documentation Site

This directory contains the TinyTorch course website and documentation.

##  Building the Site

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
- **`instructor-guide.md`** - Setup and grading guide for instructors
- **`quickstart-guide.md`** - Quick start guide for all users

## 🔧 Development Documentation

### Development Standards
- **`development/module-rules.md`** - Module development standards and patterns
- **`development/DEVELOPER_SETUP.md`** - Developer environment setup
- **`development/MODULE_ABOUT_TEMPLATE.md`** - Template for module documentation

### NBGrader Integration
- **`nbgrader/NBGrader_Quick_Reference.md`** - Daily use commands and workflow
- **`nbgrader/NBGrader_Text_Response_Technical_Implementation.md`** - Technical implementation details


**Start here**:
- **Students**: Read `STUDENT_QUICKSTART.md`
- **Instructors**: Read `instructor-guide.md`
- **Developers**: Read `development/module-rules.md`
