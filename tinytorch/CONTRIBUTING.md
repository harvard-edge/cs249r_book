# Contributing to TinyTorch 🔥

Thank you for your interest in contributing to TinyTorch! This educational ML framework is designed to teach systems engineering principles through hands-on implementation.

## 🎯 Contributing Philosophy

TinyTorch is an **educational framework** where every contribution should:
- **Enhance learning** - Make concepts clearer for students
- **Maintain pedagogical flow** - Preserve the learning progression
- **Follow systems thinking** - Emphasize memory, performance, and scaling
- **Keep it simple** - Educational clarity over production complexity

## 🚀 Getting Started

### Development Setup

1. **Clone and setup environment**:
   ```bash
   git clone https://github.com/harvard-edge/cs249r_book.git
   cd TinyTorch
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   tito system health
   tito module status
   ```

3. **Read the development guidelines**:
   - `CLAUDE.md` - Complete development standards
   - `docs/INSTRUCTOR_GUIDE.md` - Educational context
   - `docs/development/` - Technical guidelines

## 🛠️ Types of Contributions

### 1. **Module Improvements**
- Fix bugs in educational implementations
- Improve documentation and explanations
- Add better examples or visualizations
- Enhance systems analysis sections

### 2. **Testing & Validation**
- Add test cases for edge conditions
- Improve checkpoint validation
- Enhance integration tests
- Fix failing test cases

### 3. **Documentation**
- Improve module explanations
- Add better ML systems insights
- Create additional examples
- Fix typos and clarity issues

### 4. **Examples & Demos**
- Create new working examples
- Improve existing example performance
- Add visualization and analysis
- Fix broken demonstrations

## 📋 Development Process

### **MANDATORY: Follow Git Workflow Standards**

```bash
# 1. Always use virtual environment
source .venv/bin/activate

# 2. Create feature branch (NEVER work on dev/main directly)
git checkout dev
git pull origin dev
git checkout -b feature/your-improvement

# 3. Make changes following standards in CLAUDE.md
# 4. Test thoroughly
python tests/run_all_modules.py
tito module test 01

# 5. Commit with descriptive messages (NO auto-attribution)
git add .
git commit -m "Fix tensor broadcasting bug in Module 02

- Resolve shape mismatch in batch operations
- Add comprehensive test cases
- Update documentation with edge cases"

# 6. Merge to dev when complete
git checkout dev
git merge feature/your-improvement
git branch -d feature/your-improvement
```

### **Critical Policies - NO EXCEPTIONS**
- ✅ Always use virtual environment (`.venv`)
- ✅ Always work on feature branches
- ✅ Always test before committing
- 🚨 **NEVER add Co-Authored-By or automated attribution**
- 🚨 **NEVER add "Generated with Claude Code"**
- 🚨 **Only project owner adds attribution when needed**

## 🧪 Testing Requirements

All contributions must pass:

1. **Module Tests**:
   ```bash
   python tests/module_XX/run_all_tests.py
   ```

2. **Integration Tests**:
   ```bash
   python tests/integration/run_integration_tests.py
   ```

3. **Module Testing**:
   ```bash
   tito module test XX
   ```

4. **Example Verification**:
   ```bash
   cd examples/xornet && python train.py
   cd examples/cifar10 && python train_cifar10_mlp.py
   ```

## 📝 Code Standards

### Module Development

**For Students** (using the framework):
- **File Format**: Edit `*_dev.ipynb` notebooks in Jupyter Lab
- **Location**: Notebooks are in `modules/NN_name/` directories
- **Testing**: Run tests inline as you build
- **Export**: Use `tito module complete N` to export to package

**For Contributors** (improving the framework):
- **Source Files**: Edit `*_dev.py` files (source of truth)
- **Notebooks**: Generated from source files using `tito src export`
- **Structure**: Follow the standardized module structure
- **Testing**: Include immediate testing after each implementation
- **Systems Analysis**: MANDATORY memory and performance analysis
- **Documentation**: Clear explanations for educational value

### Code Quality
- **Clean Code**: Readable, well-commented implementations
- **Educational Focus**: Prioritize clarity over optimization
- **Error Handling**: Helpful error messages for students
- **Type Hints**: Where they enhance understanding

## 🎓 Educational Guidelines

### What Makes a Good Contribution

✅ **Good Examples**:
- Fixes a bug that confuses students
- Adds memory profiling to show systems concepts
- Improves explanation of complex ML concepts
- Creates working example that achieves good performance

❌ **Avoid These**:
- Overly complex optimizations that obscure learning
- Breaking changes that disrupt module progression
- Adding dependencies that complicate setup
- Removing educational scaffolding

### Systems Focus
Every contribution should emphasize:
- **Memory usage** and optimization
- **Computational complexity** analysis
- **Performance characteristics**
- **Scaling behavior** and bottlenecks
- **Production implications**

## 🐛 Bug Reports

When reporting bugs, include:

1. **Environment**: OS, Python version, virtual environment status
2. **Module**: Which module/checkpoint is affected
3. **Steps to Reproduce**: Exact commands and inputs
4. **Expected vs Actual**: What should happen vs what happens
5. **Error Messages**: Full stack traces if applicable
6. **Testing**: Did you run the module tests?

```bash
# Always include this information
python --version
echo $VIRTUAL_ENV
tito system health
```

## 🌟 Feature Requests

For new features, please:

1. **Check existing issues** - Avoid duplicates
2. **Explain educational value** - How does this help students learn?
3. **Consider module progression** - Where does this fit?
4. **Propose implementation** - High-level approach
5. **Systems implications** - Memory, performance, scaling considerations

## 💬 Communication

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions and ideas
- **Documentation**: Check `docs/` directory for detailed guides
- **Development**: Follow `CLAUDE.md` for complete standards

## 🏆 Recognition

Contributors who follow these guidelines and make valuable educational improvements will be acknowledged in:
- Module documentation where appropriate
- Release notes for significant contributions
- Course materials when contributions enhance learning

## 📚 Resources

### Essential Reading
- **`CLAUDE.md`** - Complete development standards and workflow
- **`docs/INSTRUCTOR_GUIDE.md`** - Educational context and teaching approach
- **`docs/development/`** - Technical implementation guidelines

### Quick References
- **Module Structure**: See any `modules/XX_name/` directory
- **Testing Patterns**: Check `tests/module_template/`
- **Example Code**: Look at `examples/xornet/` and `examples/cifar10/`

---

**Remember**: TinyTorch is about teaching students to understand ML systems by building them. Every contribution should enhance that educational mission! 🎓🔥

**Questions?** Check the docs or open a GitHub Discussion.
