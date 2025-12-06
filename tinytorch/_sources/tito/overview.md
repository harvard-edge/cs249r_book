# TITO Command Reference

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Master the TinyTorch CLI</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Complete command reference for building ML systems efficiently</p>
</div>

**Purpose**: Quick reference for all TITO commands. Find the right command for every task in your ML systems engineering journey.

## Quick Start: Three Commands You Need

<div style="display: grid; grid-template-columns: 1fr; gap: 1rem; margin: 2rem 0;">

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
<h4 style="margin: 0 0 0.5rem 0; color: #1976d2;">1. Check Your Environment</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito system health</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Verify your setup is ready for development</p>
</div>

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
<h4 style="margin: 0 0 0.5rem 0; color: #d97706;">2. Build & Export Modules</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito module complete 01</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Export your module to the TinyTorch package</p>
</div>

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0;">
<h4 style="margin: 0 0 0.5rem 0; color: #7b1fa2;">3. Run Historical Milestones</h4>
<code style="background: #263238; color: #ffffff; padding: 0.5rem; border-radius: 0.25rem; display: block; margin: 0.5rem 0;">tito milestone run 03</code>
<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #64748b;">Recreate ML history with YOUR code</p>
</div>

</div>

---

## 👥 Commands by User Role

TinyTorch serves three types of users. Choose your path:

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 2rem 0;">

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
<h3 style="margin: 0 0 1rem 0; color: #1976d2;">🎓 Student / Learner</h3>
<p style="margin: 0 0 1rem 0; font-size: 0.9rem; color: #37474f;">You're learning ML systems by building from scratch</p>

**Your Workflow:**
```bash
# Start learning
tito module start 01

# Complete modules  
tito module complete 01

# Validate with history
tito milestone run 03

# Track progress
tito status
```

**Key Commands:**
- `tito module` - Build components
- `tito milestone` - Validate
- `tito status` - Track progress

</div>

<div style="background: #fff3e0; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f57c00;">
<h3 style="margin: 0 0 1rem 0; color: #e65100;">👨‍🏫 Instructor</h3>
<p style="margin: 0 0 1rem 0; font-size: 0.9rem; color: #37474f;">You're teaching ML systems engineering</p>

**Your Workflow:**
```bash
# Generate assignments
tito nbgrader generate 01

# Distribute to students
tito nbgrader release 01

# Collect & grade
tito nbgrader collect 01
tito nbgrader autograde 01

# Provide feedback
tito nbgrader feedback 01
```

**Key Commands:**
- `tito nbgrader` - Assignment management
- `tito module` - Test implementations
- `tito milestone` - Validate setups

</div>

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0;">
<h3 style="margin: 0 0 1rem 0; color: #7b1fa2;">👩‍💻 Developer / Contributor</h3>
<p style="margin: 0 0 1rem 0; font-size: 0.9rem; color: #37474f;">You're contributing to TinyTorch modules</p>

**Your Workflow:**
```bash
# Edit source code
# src/01_tensor/01_tensor.py

# Export to notebooks & package
tito src export 01_tensor
tito src export --all

# Test implementations
tito src test 01_tensor

# Validate changes
tito milestone run 03
```

**Key Commands:**
- `tito src` - Developer workflow
- `tito module` - Test as student
- `tito milestone` - Validate

</div>

</div>

---

## Complete Command Reference

### System Commands

**Purpose**: Environment health, validation, and configuration

| Command | Description | Guide |
|---------|-------------|-------|
| `tito system health` | Quick environment health check (status only) | [Module Workflow](modules.md) |
| `tito system check` | Comprehensive validation with 60+ tests | [Module Workflow](modules.md) |
| `tito system info` | System resources (paths, disk, memory) | [Module Workflow](modules.md) |
| `tito system version` | Show all package versions | [Module Workflow](modules.md) |
| `tito system clean` | Clean workspace caches and temp files | [Module Workflow](modules.md) |
| `tito system report` | Generate JSON diagnostic report | [Module Workflow](modules.md) |
| `tito system jupyter` | Start Jupyter Lab server | [Module Workflow](modules.md) |
| `tito system protect` | Student protection system | [Module Workflow](modules.md) |

### Module Commands

**Purpose**: Build-from-scratch workflow (your main development cycle)

| Command | Description | Guide |
|---------|-------------|-------|
| `tito module start XX` | Begin working on a module (first time) | [Module Workflow](modules.md) |
| `tito module resume XX` | Continue working on a module | [Module Workflow](modules.md) |
| `tito module complete XX` | Test, export, and track module completion | [Module Workflow](modules.md) |
| `tito module status` | View module completion progress | [Module Workflow](modules.md) |
| `tito module reset XX` | Reset module to clean state | [Module Workflow](modules.md) |

**See**: [Module Workflow Guide](modules.md) for complete details

### Milestone Commands

**Purpose**: Run historical ML recreations with YOUR implementations

| Command | Description | Guide |
|---------|-------------|-------|
| `tito milestone list` | Show all 6 historical milestones (1957-2018) | [Milestone System](milestones.md) |
| `tito milestone run XX` | Run milestone with prerequisite checking | [Milestone System](milestones.md) |
| `tito milestone info XX` | Get detailed milestone information | [Milestone System](milestones.md) |
| `tito milestone status` | View milestone progress and achievements | [Milestone System](milestones.md) |
| `tito milestone timeline` | Visual timeline of your journey | [Milestone System](milestones.md) |

**See**: [Milestone System Guide](milestones.md) for complete details

### Progress & Data Commands

**Purpose**: Track progress and manage user data

| Command | Description | Guide |
|---------|-------------|-------|
| `tito status` | View all progress (modules + milestones) | [Progress & Data](data.md) |
| `tito reset all` | Reset all progress and start fresh | [Progress & Data](data.md) |
| `tito reset progress` | Reset module completion only | [Progress & Data](data.md) |
| `tito reset milestones` | Reset milestone achievements only | [Progress & Data](data.md) |

**See**: [Progress & Data Management](data.md) for complete details

### Community Commands

**Purpose**: Join the global TinyTorch community and track your progress

| Command | Description | Guide |
|---------|-------------|-------|
| `tito community join` | Join the community (optional info) | [Community Guide](../community.md) |
| `tito community update` | Update your community profile | [Community Guide](../community.md) |
| `tito community profile` | View your community profile | [Community Guide](../community.md) |
| `tito community stats` | View community statistics | [Community Guide](../community.md) |
| `tito community leave` | Remove your community profile | [Community Guide](../community.md) |

**See**: [Community Guide](../community.md) for complete details

### Benchmark Commands

**Purpose**: Validate setup and measure performance

| Command | Description | Guide |
|---------|-------------|-------|
| `tito benchmark baseline` | Quick setup validation ("Hello World") | [Community Guide](../community.md) |
| `tito benchmark capstone` | Full Module 20 performance evaluation | [Community Guide](../community.md) |

**See**: [Community Guide](../community.md) for complete details

### Developer Commands

**Purpose**: Source code development and contribution (for developers only)

| Command | Description | Use Case |
|---------|-------------|----------|
| `tito src export <module>` | Export src/ → modules/ → tinytorch/ | After editing source files |
| `tito src export --all` | Export all modules | After major refactoring |
| `tito src test <module>` | Run tests on source files | During development |

**Note**: These commands work with `src/XX_name/XX_name.py` files and are for TinyTorch contributors/developers.  
**Students** use `tito module` commands to work with generated notebooks.

**Directory Structure:**
```
src/              ← Developers edit here (Python source)
modules/          ← Students use these (generated notebooks)
tinytorch/        ← Package code (auto-generated)
```

---

## Command Groups by Task

### First-Time Setup

```bash
# Clone and setup
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch
./setup-environment.sh
source activate.sh

# Verify environment
tito system health
```

### Student Workflow (Learning)

```bash
# Start or continue a module
tito module start 01      # First time
tito module resume 01     # Continue later

# Export when complete
tito module complete 01

# Check progress
tito module status
```

### Developer Workflow (Contributing)

```bash
# Edit source files in src/
vim src/01_tensor/01_tensor.py

# Export to notebooks + package
tito src export 01_tensor

# Test implementation
python -c "from tinytorch import Tensor; print(Tensor([1,2,3]))"

# Validate with milestones
tito milestone run 03
```

### Achievement & Validation

```bash
# See available milestones
tito milestone list

# Get details
tito milestone info 03

# Run milestone
tito milestone run 03

# View achievements
tito milestone status
```

### Progress Management

```bash
# View all progress
tito status

# Reset if needed
tito reset all --backup
```

---

## Typical Session Flow

Here's what a typical TinyTorch session looks like:

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**1. Start Session**
```bash
cd TinyTorch
source activate.sh
tito system health         # Verify environment
```

**2. Work on Module**
```bash
tito module start 03       # Or: tito module resume 03
# Edit in Jupyter Lab...
```

**3. Export & Test**
```bash
tito module complete 03
```

**4. Run Milestone (when prerequisites met)**
```bash
tito milestone list        # Check if ready
tito milestone run 03      # Run with YOUR code
```

**5. Track Progress**
```bash
tito status                # See everything
```

</div>

---

## Command Help

Every command has detailed help text:

```bash
# Top-level help
tito --help

# Command group help
tito module --help
tito milestone --help

# Specific command help
tito module complete --help
tito milestone run --help
```

---

## Detailed Guides

- **[Module Workflow](modules.md)** - Complete guide to building and exporting modules
- **[Milestone System](milestones.md)** - Running historical ML recreations
- **[Progress & Data](data.md)** - Managing your learning journey
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

---

## Related Resources

- **[Getting Started Guide](../getting-started.md)** - Complete setup and first steps
- **[Module Workflow](modules.md)** - Day-to-day development cycle
- **[Datasets Guide](../datasets.md)** - Understanding TinyTorch datasets

---

*Master these commands and you'll build ML systems with confidence. Every command is designed to accelerate your learning and keep you focused on what matters: building production-quality ML frameworks from scratch.*
