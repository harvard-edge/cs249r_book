# TinyTorch Modules: Education Review Action Plan

**Generated**: December 18, 2025
**Revised**: December 18, 2025 (after pattern analysis)
**Purpose**: Conservative action plan respecting established module patterns

---

## CRITICAL: Established Naming Conventions

After reviewing the codebase, **all modules already follow consistent patterns**. Do NOT introduce new function types.

### Allowed Function Types

| Prefix | Purpose | Example |
|--------|---------|---------|
| `test_unit_*` | Unit tests for specific features | `test_unit_sigmoid`, `test_unit_linear_layer` |
| `test_module` | Integration test for whole module | `test_module()` |
| `analyze_*` | Systems analysis with output | `analyze_memory_layout`, `analyze_layer_performance` |
| `demo_*` | Demonstrations at end | `demo_tensor`, `demo_activations` |

### FORBIDDEN (Do NOT Add)

- `calculate_*` - use `analyze_*` instead
- `exercise_*` - breaks established patterns
- `compare_*` - use `analyze_*` instead
- `measure_*` - use `analyze_*` instead
- Any standalone functions for students to implement outside of classes

### Key Rules

1. **Preserve "Introduce → Implement → Test Immediately" flow**
2. **Systems exercises belong in `analyze_*` functions or markdown reflection sections**
3. **Micro-reflections go in markdown cells, not as code exercises**
4. **NBGrader blocks are for class methods and test functions only**

---

## File Locations

**Source of Truth**: `tinytorch/src/NN_name/NN_name.py` (Jupytext format)
**Generated Notebooks**: `tinytorch/modules/NN_name/*.ipynb` (auto-generated)

**ALWAYS edit files in `src/`** then run `tito src export` to regenerate notebooks.

---

## Revised Assessment: Most Modules Are Already Good

After detailed review of Modules 01, 02, 03, and 16:

| Module | Original Priority | Revised Status |
|--------|------------------|----------------|
| 01 Tensor | Low | **Already excellent** - no changes needed |
| 02 Activations | Medium | **Already excellent** - has strong systems sections |
| 03 Layers | Low | **Already excellent** - has `analyze_layer_memory`, `analyze_layer_performance` |
| 16 Compression | Critical | **Actually well-structured** - has proper flow and guards |

**The original action items were over-prescriptive** and would have broken established patterns.

---

## What Actually Needs Checking

### For Each Module: Verification Checklist

- [ ] NBGrader markers (`### BEGIN SOLUTION` / `### END SOLUTION`) on all implementations
- [ ] `if __name__ == "__main__":` guards after test and analysis functions
- [ ] Proper function naming (`test_unit_*`, `analyze_*`, `demo_*`)
- [ ] Clear "Introduce → Implement → Test" flow

### If Content Gaps Are Found

**For systems analysis**: Enhance existing `analyze_*` functions, don't add new standalone exercises

**For reflections**: Add to existing markdown "Reflection Questions" sections

**For quantitative exercises**: Include in `analyze_*` functions with printed output

---

## Module Status Summary

### Verified Good (No Action Needed)

| Module | Status | Notes |
|--------|--------|-------|
| 01 Tensor | Excellent | Complete NBGrader, good `analyze_memory_layout` |
| 02 Activations | Excellent | Strong systems thinking section, all patterns correct |
| 03 Layers | Excellent | Has `analyze_layer_memory`, `analyze_layer_performance` |

### Need Verification (Quick Check Only)

| Module | Check For |
|--------|-----------|
| 04 Losses | NBGrader completeness |
| 05 Autograd | NBGrader completeness, guard coverage |
| 06 Optimizers | NBGrader completeness |
| 07 Training | NBGrader completeness, guard coverage |
| 08 DataLoader | NBGrader completeness, guard coverage |
| 09 Convolutions | NBGrader completeness |
| 10 Tokenization | NBGrader completeness |
| 11 Embeddings | NBGrader completeness |
| 12 Attention | NBGrader completeness |
| 13 Transformers | NBGrader completeness |
| 14 Profiling | NBGrader completeness |
| 15 Quantization | NBGrader completeness |
| 16 Compression | Already verified good |
| 17 Memoization | NBGrader completeness |
| 18 Acceleration | NBGrader completeness |
| 19 Benchmarking | NBGrader completeness |
| 20 Capstone | Structure review (special case - project module) |

---

## Quick Verification Commands

### Check NBGrader Markers
```bash
# Count BEGIN/END SOLUTION pairs
grep -c "BEGIN SOLUTION" tinytorch/src/NN_name/NN_name.py
grep -c "END SOLUTION" tinytorch/src/NN_name/NN_name.py
# These should be equal
```

### Check Function Patterns
```bash
# List all functions
grep "^def " tinytorch/src/NN_name/NN_name.py

# Check for non-standard names (should find nothing)
grep "^def " tinytorch/src/NN_name/NN_name.py | grep -v "test_unit_\|test_module\|analyze_\|demo_\|__"
```

### Check Guards
```bash
# Count functions vs guards
grep -c "^def " tinytorch/src/NN_name/NN_name.py
grep -c "^if __name__" tinytorch/src/NN_name/NN_name.py
```

---

## Workflow for Any Needed Changes

1. **Read the module first** - understand its current structure
2. **Check against patterns** - does it follow naming conventions?
3. **Only fix actual issues**:
   - Missing NBGrader markers
   - Missing `if __name__` guards
   - Broken tests
4. **Do NOT add**:
   - New function types
   - Standalone exercise functions
   - Content that duplicates existing sections
5. **Test after changes**: `python3 tinytorch/src/NN_name/NN_name.py`

---

## Summary

**Original estimate**: 80-110 hours of work
**Revised estimate**: 5-10 hours (verification only, minimal fixes)

Most modules are already in excellent shape. The original action plan would have introduced inconsistencies by adding new function types and breaking the established pedagogical flow.

**Key insight**: The modules already implement "Build → Use → Reflect" correctly through:
- Class implementations (Build)
- Immediate `test_unit_*` functions (Use)
- Markdown reflection sections and `analyze_*` functions (Reflect)

---

## Notes for Future Development

When adding new modules or features:

1. Follow existing naming patterns exactly
2. Test immediately after each implementation
3. Put systems analysis in `analyze_*` functions
4. Put reflections in markdown sections
5. Every function gets an `if __name__` guard
6. Every implementation gets NBGrader markers
