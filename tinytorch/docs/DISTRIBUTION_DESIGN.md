# TinyTorch Distribution Design

> **Status:** Draft / Design Phase  
> **Branch:** `feature/tinytorch-pip-distribution`  
> **Goal:** Enable `pip install tinytorch` + `tito init` workflow

---

## The Problem

TinyTorch currently lives inside `harvard-edge/cs249r_book`. Students cannot:
- `pip install tinytorch` (not on PyPI)
- `git clone tinytorch` (no standalone repo)

Current workaround requires cloning the entire book repository, which is not ideal UX.

---

## Design Considerations

### 1. What Are We Distributing?

TinyTorch has **three distinct components**:

| Component | Description | Changes Often? | Size |
|-----------|-------------|----------------|------|
| **tito CLI** | Command line tool | Sometimes (bug fixes) | ~100KB |
| **tinytorch package** | Base ML framework (student builds on this) | Rarely | ~50KB |
| **Learning Materials** | modules/, tests/, milestones/ | Sometimes (content fixes) | ~4MB |

**Key insight:** The CLI and package change together (pip upgrade). But learning materials are "forked" into student's workspace.

---

### 2. The Update Problem

When we fix a bug in Module 05, what happens to students who already ran `tito init`?

#### Scenario A: Student hasn't started Module 05 yet
- **Ideal:** They get the fix automatically
- **Reality:** Their workspace has the old version

#### Scenario B: Student is in the middle of Module 05
- **Ideal:** They get the fix without losing their work
- **Reality:** Merging our fix with their edits is complex

#### Scenario C: Student already completed Module 05
- **Ideal:** They probably don't care about the fix
- **Reality:** Their exported code might have the bug

---

### 3. How Do Other Projects Handle This?

| Project | Distribution | Update Strategy |
|---------|-------------|-----------------|
| **Rustlings** | Bundled in rustup | `rustlings update` replaces exercises, preserves progress file |
| **Exercism** | Download on demand | Fresh download each exercise, solutions stored separately |
| **freeCodeCamp** | Web-based | No local files, always current |
| **fast.ai** | pip package | Content in package, `pip upgrade` updates everything |
| **nbdev** | Template on init | No updates to existing projects |
| **Create React App** | Template on init | No updates to existing projects (ejected) |

**Observation:** Most tools treat the initialized project as "student-owned" and don't update it.

---

### 4. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ tinytorch (pip package)                                         │
├─────────────────────────────────────────────────────────────────┤
│ tito/                    # CLI (versioned, updated via pip)     │
│ tinytorch/               # Base package (versioned)             │
│ _templates/              # Bundled learning materials           │
│   ├── modules/           # 20 module notebooks                  │
│   ├── tests/             # Validation tests                     │
│   ├── milestones/        # Historical achievements              │
│   └── VERSION            # Template version for migrations      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ tito init
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Student Workspace (owned by student)                            │
├─────────────────────────────────────────────────────────────────┤
│ modules/                 # Student's working copies             │
│ tests/                   # Test framework                       │
│ milestones/              # Achievement scripts                  │
│ tinytorch/               # Student's exported code              │
│ progress.json            # Progress tracking                    │
│ .tinytorch/              # Config and metadata                  │
│   ├── version            # Workspace version (for migrations)   │
│   └── config.json        # User preferences                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Update Strategies

### Option A: No Updates (Simplest)

**Philosophy:** Once `tito init` runs, the workspace is student-owned. No automatic updates.

```bash
pip install tinytorch     # Get CLI + templates
tito init                  # Create workspace (one-time)
# ... student works ...
pip install --upgrade tinytorch  # Updates CLI only
# Workspace unchanged
```

**Handling bugs:**
- Critical fixes: Announce on course website, provide manual patch instructions
- Students who haven't started affected module: `tito module reset 05` could restore from templates

**Pros:**
- Simplest to implement
- No risk of corrupting student work
- Predictable behavior

**Cons:**
- Bug fixes don't reach existing users easily
- Potential version drift between students

---

### Option B: Selective Updates (Recommended)

**Philosophy:** CLI updates automatically. Workspace content updates are opt-in and surgical.

```bash
pip install tinytorch
tito init

# Later, when we release a fix...
tito update --check        # "Updates available for: tests/05_autograd"
tito update                # Applies updates to tests (not modules)
tito update --module 05    # Resets module 05 to latest (with backup)
```

**Update categories:**

| Content | Auto-update? | Command |
|---------|-------------|---------|
| CLI (tito) | Yes (pip upgrade) | `pip install --upgrade tinytorch` |
| Test files | Opt-in | `tito update` |
| Milestones | Opt-in | `tito update` |
| Module notebooks | Never auto, manual only | `tito module reset 05` |

**Pros:**
- Fixes reach students who need them
- Student work in modules is protected
- Clear separation of concerns

**Cons:**
- More complex to implement
- Need to track what's been modified

---

### Option C: Git-Based Workspace (Most Flexible)

**Philosophy:** Workspace is a git repo that tracks upstream for updates.

```bash
pip install tinytorch
tito init                  # Creates workspace as git repo
# ... student works, commits ...

tito update                # Fetches upstream, attempts merge
# If conflict in module student edited:
#   "Conflict in modules/05_autograd. Resolve manually or skip."
```

**Pros:**
- Full version control
- Students learn git
- Merge conflicts are visible

**Cons:**
- Most complex to implement
- Requires git knowledge from students
- Merge conflicts can be confusing

---

## Recommendation: Option B (Selective Updates)

### Implementation Details

#### 1. Version Tracking

```json
// .tinytorch/version.json
{
  "workspace_version": "1.0.0",
  "template_version": "1.0.0",
  "created": "2025-01-15T10:30:00Z",
  "last_updated": "2025-01-15T10:30:00Z",
  "files": {
    "tests/05_autograd/test_progressive_integration.py": {
      "version": "1.0.0",
      "modified_by_student": false
    },
    "modules/05_autograd/05_autograd.py": {
      "version": "1.0.0", 
      "modified_by_student": true
    }
  }
}
```

#### 2. Update Command

```bash
tito update --check
# Output:
# 📦 Package version: 1.1.0 (you have 1.0.0)
# 📝 Available updates:
#    tests/05_autograd/test_progressive_integration.py (bug fix)
#    milestones/04_1998_cnn/02_lecun_cifar10.py (new feature)
# 
# Run 'tito update' to apply, or 'tito update --dry-run' to preview.

tito update
# Output:
# ✅ Updated tests/05_autograd/test_progressive_integration.py
# ✅ Updated milestones/04_1998_cnn/02_lecun_cifar10.py
# ⏭️  Skipped modules/05_autograd (student modified)
#
# 💡 To reset a module to latest: tito module reset 05
```

#### 3. Module Reset

```bash
tito module reset 05
# Output:
# ⚠️  This will reset modules/05_autograd to the latest template.
# 
# Current file: modules/05_autograd/05_autograd.py
#   Last modified: 2025-01-14 14:30:22
#   Size: 12.4 KB
#   Changes: 342 lines added/modified
#
# Options:
#   1. Backup and reset (recommended)
#   2. View diff first
#   3. Cancel
#
# Choice [1]: 1
#
# ✅ Backed up to .tinytorch/backups/05_autograd_20250115_103000/
# ✅ Reset modules/05_autograd to version 1.1.0
```

---

## Maintenance Workflow

### For Maintainers (Us)

```bash
# 1. Make fix in tinytorch/
cd tinytorch
# Edit modules/05_autograd/05_autograd.py

# 2. Bump version
# Edit _templates/VERSION

# 3. Update changelog
# Edit CHANGELOG.md

# 4. Commit and push
git commit -m "fix(module-05): correct gradient accumulation bug"
git push

# 5. Release (when ready)
# Bump version in pyproject.toml
# Create GitHub release
# Publish to PyPI (future)
```

### For Students

```bash
# Check for updates periodically
tito update --check

# Apply updates (safe, doesn't touch modules)
tito update

# If told to reset a module they haven't modified
tito module reset 05

# If they have modifications, decide:
# - Keep their version (might have bug)
# - Reset and redo work (gets fix)
# - Manually apply fix (advanced)
```

---

## Open Questions

### Q1: What about datasets?

Datasets (CIFAR-10, MNIST) are large. Options:
- **A:** Download on demand (`tito data download cifar10`)
- **B:** Reference external URLs (students download manually)
- **C:** Subset in package, full on demand

**Recommendation:** Option A. `tito data download` fetches when needed.

### Q2: What about Jupyter notebooks vs Python files?

Currently: Students work in notebooks, we maintain .py source files.

Options:
- **A:** Bundle notebooks (students work in them directly)
- **B:** Bundle .py, generate notebooks on init (`jupytext`)
- **C:** Bundle both (redundant, but simple)

**Recommendation:** Option A. Bundle notebooks. Simpler for students.

### Q3: How do instructors customize for their course?

Instructors might want:
- Different module order
- Subset of modules
- Additional assignments

Options:
- **A:** Fork the repo (current approach)
- **B:** `tito init --config course.yml` with customization
- **C:** Separate instructor workflow

**Recommendation:** Start with Option A. Consider B later.

### Q4: What about NBGrader integration?

NBGrader expects specific directory structure. Need to ensure workspace is compatible.

**Action:** Verify NBGrader works with proposed structure.

---

## Implementation Plan

### Phase 1: Basic Distribution (This Branch)

1. [ ] Restructure package to include `_templates/`
2. [ ] Implement `tito init` command
3. [ ] Update `pyproject.toml` for package data
4. [ ] Test install from GitHub subdirectory
5. [ ] Update demo GIFs for new flow

### Phase 2: Update Mechanism

1. [ ] Implement version tracking in workspace
2. [ ] Implement `tito update --check`
3. [ ] Implement `tito update` for tests/milestones
4. [ ] Implement `tito module reset` with backups

### Phase 3: Polish

1. [ ] Add `tito doctor` for common issues
2. [ ] Add `tito data download` for datasets
3. [ ] Documentation and guides
4. [ ] PyPI publication

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-XX | Bundle materials in pip package | Offline reliability, university lab compatibility |
| 2025-01-XX | Use selective update strategy | Protects student work, allows bug fixes |
| 2025-01-XX | ... | ... |

---

## References

- [Rustlings Update Mechanism](https://github.com/rust-lang/rustlings)
- [Cookiecutter Template Updates](https://cookiecutter.readthedocs.io/)
- [Poetry Project Management](https://python-poetry.org/)
- [nbdev Documentation](https://nbdev.fast.ai/)


