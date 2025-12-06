# TinyTorch Modules Directory

This directory contains student-facing Jupyter notebooks for learning ML systems from scratch.

## 📦 Module Structure

Each module directory contains:
- `{module}_dev.py` - Jupytext Python file (source of truth)
- `{module}.ipynb` - Jupyter notebook (auto-generated)
- `README.md` - Module overview and learning objectives

## 🔄 How Modules Are Created

Modules are **automatically exported from `src/`** using the following workflow:

1. **Source notebooks** live in `src/{module}/` as `.ipynb` files
2. **Run export**: `tito system export {module}` or `nbdev_export`
3. **Auto-generated files** appear in `modules/{module}/`

The `src/` directory is where development happens. The `modules/` directory is what students use.

## 📚 Available Modules

Modules will be populated as you complete the TinyTorch learning path:

- ✅ `01_tensor` - Tensor fundamentals and operations
- ✅ `02_activations` - Activation functions (ReLU, Sigmoid, etc.)
- ✅ `04_losses` - Loss functions for training
- ✅ `06_optimizers` - Optimization algorithms (SGD, Adam, etc.)
- 🔒 Additional modules unlock as you progress...

## 🚀 Getting Started

1. **Check module status**: `tito module status`
2. **Start a module**: `tito module start 01`
3. **Work on the module**: Opens Jupyter Lab automatically
4. **Complete the module**: `tito module complete 01`

Each module builds on previous ones, creating a complete ML framework from scratch!
