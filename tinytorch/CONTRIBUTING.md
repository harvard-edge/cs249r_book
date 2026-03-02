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
   cd cs249r_book/tinytorch
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   tito --version       # Check TinyTorch version
   tito system health   # Verify environment
   tito module status   # See module progress
   ```

3. **Read the development guidelines**:
   - `CONTRIBUTING.md` - Development standards (this file)
   - `INSTRUCTOR.md` - Educational context and teaching approach
   - `README.md` - Repository structure and project overview

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

# 3. Make changes following standards in CONTRIBUTING.md
# 4. Test thoroughly
pytest tests/
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

1. **Module Tests** (run tests for a specific module):
   ```bash
   pytest tests/NN_name/             # e.g., pytest tests/01_tensor/
   tito module test NN               # e.g., tito module test 01
   ```

2. **Integration Tests**:
   ```bash
   pytest tests/integration/
   ```

3. **Milestone Verification** (end-to-end examples):
   ```bash
   python milestones/02_1969_xor/02_xor_solved.py
   python milestones/04_1998_cnn/01_lecun_tinydigits.py
   ```

## 📝 Code Standards

### Module Development

**For Students** (using the framework):
- **File Format**: Work in `modules/NN_name/name.ipynb` notebooks in Jupyter Lab
- **Location**: Notebooks are in `modules/NN_name/` directories (e.g., `modules/01_tensor/tensor.ipynb`)
- **Testing**: Run tests inline as you build
- **Export**: Use `tito module complete N` to export to package

**For Contributors** (improving the framework):
- **Source Files**: Edit `src/NN_name/NN_name.py` files (source of truth, e.g., `src/01_tensor/01_tensor.py`)
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

1. **Version**: Run `tito --version` to get TinyTorch version
2. **Environment**: OS, Python version, virtual environment status
3. **Module**: Which module/checkpoint is affected
4. **Steps to Reproduce**: Exact commands and inputs
5. **Expected vs Actual**: What should happen vs what happens
6. **Error Messages**: Full stack traces if applicable
7. **Testing**: Did you run the module tests?

```bash
# Always include this information
tito --version
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
- **Documentation**: Check `README.md` for project structure and guides
- **Development**: Follow `CONTRIBUTING.md` for complete standards

## 🏆 Recognition

Contributors who follow these guidelines and make valuable educational improvements will be acknowledged in:
- Module documentation where appropriate
- Release notes for significant contributions
- Course materials when contributions enhance learning

## 🏷️ Releases (Maintainers Only)

TinyTorch follows [semantic versioning](https://semver.org/):

| Release Type | Version Change | When to Use |
|--------------|----------------|-------------|
| **patch** | 0.1.0 → 0.1.1 | Bug fixes, typos, small updates |
| **minor** | 0.1.x → 0.2.0 | New features, module improvements |
| **major** | 0.x.x → 1.0.0 | Breaking changes, stable API |

### Release Process

Releases are created via the `tinytorch-publish-live.yml` GitHub Actions workflow:

1. Maintainer triggers workflow from GitHub Actions
2. Select release type (patch/minor/major)
3. Enter release description
4. Workflow automatically:
   - Bumps version in code
   - Runs tests and preflight checks
   - Merges dev → main
   - Deploys to tinytorch.org
   - Creates git tag (e.g., v0.1.1)
   - Creates GitHub Release with notes
   - Publishes to PyPI

### For Contributors

**You don't need to bump versions.** Maintainers handle versioning during the release process. Just focus on:
- Writing good code
- Following the contribution guidelines
- Using conventional commit messages (`fix:`, `feat:`, `docs:`)

Your commits will be included in the next release with appropriate version bump.

## 📚 Resources

### Essential Reading
- **`CONTRIBUTING.md`** - Development standards and workflow (this file)
- **`INSTRUCTOR.md`** - Educational context and teaching approach
- **`README.md`** - Repository structure and project overview

### Quick References
- **Module Structure**: See any `src/NN_name/` directory (e.g., `src/01_tensor/`)
- **Testing Patterns**: Check `tests/NN_name/` directories (e.g., `tests/01_tensor/`)
- **Example Code**: Look at `milestones/` for end-to-end working examples

---

## 🏆 Contributor Recognition

We use [All Contributors](https://allcontributors.org) to recognize everyone who helps improve TinyTorch.

### How to Recognize a Contributor

After merging a PR or resolving an issue, comment:

```
@all-contributors please add @username for TYPE
```

### Contribution Types

| Type | Emoji | Use For |
|------|-------|---------|
| `bug` | 🐛 | Found a bug or issue |
| `code` | 💻 | Submitted code |
| `doc` | 📖 | Improved documentation |
| `ideas` | 💡 | Suggested improvements |
| `test` | 🧪 | Added tests |
| `review` | 👀 | Reviewed PRs |

### Examples

```
@all-contributors please add @AmirAlasady for bug
@all-contributors please add @student123 for code, doc
```

---

**Remember**: TinyTorch is about teaching students to understand ML systems by building them. Every contribution should enhance that educational mission! 🎓🔥

**Questions?** Check the docs or open a GitHub Discussion.
