# Milestone System Design

## Clean Architecture Achieved ✅

The milestone system is now **self-contained** with **zero code duplication**.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Student Workflow                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 1. Complete module
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              tito module complete 01                        │
│         (tito/commands/module_workflow.py)                  │
│                                                             │
│  • Runs inline tests                                        │
│  • Exports to package                                       │
│  • Updates module progress                                  │
│  • Calls: check_module_export(module_name, console)  ◄──┐   │
└─────────────────────────────────────────────────────────────┘
                            │                                │
                            │ 2. Check unlocks              │
                            ▼                                │
┌─────────────────────────────────────────────────────────────┐
│         Milestone Tracker (Single Source of Truth)          │
│        tests/milestones/milestone_tracker.py                │
│                                                             │
│  check_module_export(module_name, console):                 │
│    1. Mark module complete                                  │
│    2. Check all milestone requirements                      │
│    3. Detect newly unlocked milestones                      │
│    4. Show unlock notifications                             │
│    5. Return results                                        │
│                                                             │
│  Progress stored in: ~/.tinytorch/progress.json             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 3. Milestone unlocked!
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Student sees notification                  │
│                                                             │
│  🔓 MILESTONE UNLOCKED!                                     │
│  1957 - The Perceptron                                      │
│  Run: tito milestone run perceptron                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 4. Run milestone test
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              tito milestone run perceptron                  │
│           (tito/commands/milestone.py)                      │
│                                                             │
│  • Verify requirements met                                  │
│  • Run pytest test                                          │
│  • Show learning metrics                                    │
│  • Mark complete if passed                                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Single Source of Truth

**All milestone logic in ONE place**: `tests/milestones/milestone_tracker.py`

This includes:
- Milestone definitions (`MILESTONES` dict)
- Requirement checking
- Progress tracking
- Unlock detection
- Message formatting

### 2. Clean API

**Other code calls through simple functions**:

```python
# That's it! Just one call:
from milestone_tracker import check_module_export
check_module_export(module_name, console)
```

No need to:
- Import `MILESTONES` dict
- Check requirements manually
- Track progress separately
- Format unlock messages
- Duplicate any logic

### 3. Separation of Concerns

**Milestone Tracker** (tests/milestones/milestone_tracker.py):
- Defines what milestones exist
- Tracks which modules are complete
- Determines when to unlock
- Formats messages

**CLI Commands** (tito/commands/):
- Handle user interaction
- Call milestone API
- Display results
- Run pytest tests

**No overlap!**

### 4. Fail Gracefully

```python
try:
    from milestone_tracker import check_module_export
    check_module_export(module_name, console)
except ImportError:
    pass  # Don't break workflow if milestone system unavailable
```

## File Organization

```
tests/milestones/
├── milestone_tracker.py           # Core system (API)
│   ├── MilestoneTracker class
│   ├── MILESTONES dict
│   ├── check_module_export()     # Main API
│   ├── show_progress()
│   └── list_tests()
│
├── test_learning_verification.py  # Pytest tests
│   ├── test_perceptron_learning()
│   ├── test_xor_learning()
│   ├── test_mlp_digits_learning()
│   ├── test_cnn_learning()
│   └── test_transformer_learning()
│
└── Documentation/
    ├── API.md                     # API reference
    ├── WORKFLOW.md                # Complete workflow
    ├── STUDENT_GUIDE.md           # Student docs
    ├── PROGRESSION.md             # Historical context
    ├── QUICKSTART.md              # Quick reference
    ├── SYSTEM_DESIGN.md           # This file
    └── README.md                  # Overview
```

## Data Flow

### Module Completion Flow

```
Student completes module
    ↓
tito module complete 01
    ↓
Run tests ✓
    ↓
Export to package ✓
    ↓
check_module_export("01_tensor", console)
    ↓
Milestone Tracker:
  • Add "01_tensor" to completed_modules
  • Check all milestones:
    - perceptron: needs [00_setup, 01_tensor, 02_autograd]
      → 2/3 complete, not yet
    - xor: needs [00_setup, 01_tensor, 02_autograd, 03_nn]
      → 2/4 complete, not yet
  • Save progress
  • Return: {'newly_unlocked': [], 'messages': []}
    ↓
No unlocks yet, continue
```

### Unlock Flow

```
Student completes 02_autograd
    ↓
check_module_export("02_autograd", console)
    ↓
Milestone Tracker:
  • Add "02_autograd" to completed_modules
  • Check all milestones:
    - perceptron: needs [00_setup, 01_tensor, 02_autograd]
      → 3/3 complete! ✓ UNLOCK!
  • Add "perceptron" to unlocked_milestones
  • Save progress
  • Show unlock notification
  • Return: {'newly_unlocked': ['perceptron'], 'messages': [...]}
    ↓
Student sees:
  🔓 MILESTONE UNLOCKED!
  1957 - The Perceptron
  Run: tito milestone run perceptron
```

### Test Run Flow

```
Student runs: tito milestone run perceptron
    ↓
MilestonesCommand.run()
    ↓
Check if unlocked:
  tracker.can_run_milestone("perceptron")
  → Yes, it's in unlocked_milestones
    ↓
Run pytest:
  pytest tests/milestones/test_learning_verification.py::test_perceptron_learning -v
    ↓
Test runs, shows metrics:
  ✅ Loss decreases >50%
  ✅ Accuracy >90%
  ✅ Gradients flow
  ✅ Weights update
    ↓
Test passes!
    ↓
tracker.mark_milestone_complete("perceptron")
    ↓
Show completion message:
  🏆 MILESTONE COMPLETED!
  Your neural network actually learns. 🎓
```

## Progress Tracking

### Storage Location

`~/.tinytorch/progress.json`

### Structure

```json
{
  "completed_modules": [
    "00_setup",
    "01_tensor",
    "02_autograd"
  ],
  "unlocked_milestones": [
    "perceptron"
  ],
  "completed_milestones": []
}
```

### Why Separate from Module Progress?

Module progress (`progress.json` in project root):
- Tracks which modules student started/completed
- Used by module workflow
- Project-specific

Milestone progress (`~/.tinytorch/progress.json`):
- Tracks milestone unlocks/completions
- Used by milestone system
- User-specific (persists across projects)

## Adding New Milestones

**Only need to edit ONE file**: `milestone_tracker.py`

```python
# 1. Add to MILESTONES dict
MILESTONES["new_milestone"] = {
    "name": "2025 - New Breakthrough",
    "requires": ["00_setup", "01_tensor", "15_new_module"],
    "test": "test_new_milestone_learning",
    "description": "Description",
    "unlock_message": "🎉 You can now...",
}

# 2. Add to MILESTONE_ORDER
MILESTONE_ORDER = [
    "perceptron", "xor", "mlp_digits", "cnn", "transformer",
    "new_milestone"  # Add here
]
```

Then create the pytest test in `test_learning_verification.py`. That's it!

## Testing the System

### Unit Test (Milestone Logic)

```bash
# Test unlock detection
python3 tests/milestones/milestone_tracker.py complete 00_setup
python3 tests/milestones/milestone_tracker.py complete 01_tensor
python3 tests/milestones/milestone_tracker.py complete 02_autograd
# Should show unlock message

# Check progress
python3 tests/milestones/milestone_tracker.py progress
```

### Integration Test (Full Workflow)

```bash
# Complete module through CLI
tito module complete 01
# Should automatically check for unlocks

# Run milestone test
tito milestone run perceptron
# Should verify requirements and run test
```

### Verification Tests (Learning)

```bash
# Run all milestone tests
pytest tests/milestones/test_learning_verification.py -v

# Run specific milestone
pytest tests/milestones/test_learning_verification.py::test_perceptron_learning -v
```

## Benefits of This Design

### ✅ No Code Duplication
- Milestone logic in ONE place
- Other code just calls API
- Changes only need to happen once

### ✅ Clean Separation
- Milestone system is self-contained
- CLI commands are thin wrappers
- Easy to understand and maintain

### ✅ Easy to Extend
- Add new milestone: edit one file
- Add new requirement: edit one dict
- Add new test: create pytest function

### ✅ Testable
- Milestone logic can be tested independently
- CLI integration can be tested separately
- Learning verification tests are isolated

### ✅ Fail Gracefully
- If milestone system unavailable, workflow continues
- Errors don't break module completion
- Silent fallback for missing dependencies

## Summary

**The milestone system is now clean, self-contained, and has zero code duplication.**

Key points:
1. **Single source of truth**: `milestone_tracker.py`
2. **Simple API**: `check_module_export(module_name, console)`
3. **Clean separation**: Milestone logic vs. CLI interaction
4. **Easy to extend**: Add milestones in one place
5. **Well documented**: Multiple docs for different audiences

The system automatically:
- Tracks module completion
- Detects when milestones unlock
- Shows unlock notifications
- Verifies requirements before running tests
- Marks milestones complete when tests pass

Students get a **gamified learning experience** with clear progression through 60+ years of ML history! 🚀
