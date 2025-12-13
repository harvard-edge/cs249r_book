# Milestone System API

## Overview

The milestone system is **self-contained** in `tests/milestones/milestone_tracker.py`. Other parts of TinyTorch call it through a clean API—no code duplication.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  TinyTorch CLI                          │
│  (tito/commands/module_workflow.py, milestones.py)     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   │ Calls API functions
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│           Milestone Tracker (Single Source of Truth)    │
│        tests/milestones/milestone_tracker.py            │
│                                                         │
│  • Tracks progress (~/.tinytorch/progress.json)        │
│  • Defines milestone requirements                       │
│  • Checks unlock conditions                            │
│  • Displays unlock messages                            │
│  • Runs verification tests                             │
└─────────────────────────────────────────────────────────┘
```

## Public API Functions

### 1. `check_module_export(module_name: str, console=None) -> dict`

**Purpose**: Check if completing a module unlocks any milestones.

**When to call**: After a student successfully exports a module.

**Parameters**:
- `module_name` (str): Module name (e.g., "01_tensor", "02_autograd")
- `console` (Optional[Console]): Rich console for displaying messages

**Returns**:
```python
{
    'newly_unlocked': ['perceptron', 'xor'],  # List of milestone IDs
    'messages': [
        "🔓 MILESTONE UNLOCKED!\n\n1957 - The Perceptron\n...",
        "🔓 MILESTONE UNLOCKED!\n\n1986 - Backpropagation (XOR)\n..."
    ]
}
```

**Example**:
```python
from milestone_tracker import check_module_export

# After exporting a module
result = check_module_export("02_autograd", console=console)

if result['newly_unlocked']:
    # Milestones were unlocked!
    for milestone_id in result['newly_unlocked']:
        print(f"Unlocked: {milestone_id}")
```

### 2. `show_progress()`

**Purpose**: Display current milestone progress.

**When to call**: When student runs `tito milestone status`.

**Example**:
```python
from milestone_tracker import show_progress

show_progress()
```

**Output**:
```
🎯 TinyTorch Milestone Progress
┌────────────────────────────────┬─────────────┬──────────────┐
│ Milestone                      │   Status    │ Requirements │
├────────────────────────────────┼─────────────┼──────────────┤
│ 1957 - The Perceptron          │ 🔓 Unlocked │ 3/3 modules  │
│ 1986 - Backpropagation (XOR)   │  🔒 Locked  │ 3/4 modules  │
...
```

### 3. `list_tests()`

**Purpose**: List all unlocked milestone tests.

**When to call**: When student runs `tito milestone list`.

**Example**:
```python
from milestone_tracker import list_tests

list_tests()
```

## Data Structures

### Milestone Definition

```python
MILESTONES = {
    "perceptron": {
        "name": "1957 - The Perceptron",
        "requires": ["00_setup", "01_tensor", "02_autograd"],
        "test": "test_perceptron_learning",
        "description": "First learning algorithm with automatic weight updates",
        "unlock_message": "🎉 You can now verify that gradient descent actually works!",
    },
    # ... more milestones
}
```

### Progress File (`~/.tinytorch/progress.json`)

```json
{
  "completed_modules": ["00_setup", "01_tensor", "02_autograd"],
  "unlocked_milestones": ["perceptron"],
  "completed_milestones": []
}
```

## Integration Points

### Module Workflow (`tito module complete`)

```python
# In module_workflow.py
def complete_module(self, module_number):
    # ... run tests, export ...

    # Check for milestone unlocks
    self._check_milestone_unlocks(module_name)

def _check_milestone_unlocks(self, module_name):
    """Simple wrapper - milestone tracker does all the work."""
    from milestone_tracker import check_module_export
    check_module_export(module_name, console=self.console)
```

### Milestone Command (`tito milestone`)

```python
# In milestones.py
class MilestonesCommand(BaseCommand):
    def run(self, args):
        from milestone_tracker import MilestoneTracker, MILESTONES

        tracker = MilestoneTracker()

        if args.action == "progress":
            tracker.show_progress()

        elif args.action == "run":
            # Verify requirements
            if not tracker.can_run_milestone(milestone_id):
                # Show what's needed
                return 1

            # Run pytest test
            # Mark complete if passed
```

## Design Principles

### 1. Single Source of Truth

**All milestone logic lives in `milestone_tracker.py`:**
- Milestone definitions
- Requirement checking
- Progress tracking
- Unlock detection
- Message formatting

### 2. Clean Separation

**CLI commands only:**
- Call API functions
- Pass console for display
- Handle user interaction

**They do NOT:**
- Duplicate milestone logic
- Track progress themselves
- Check requirements directly

### 3. No Code Duplication

**Before** (bad):
```python
# In export.py
if module in completed:
    for milestone_id in MILESTONES:
        if requirements_met(milestone_id):
            show_unlock_message(milestone_id)

# In module_workflow.py
if module in completed:
    for milestone_id in MILESTONES:  # DUPLICATE!
        if requirements_met(milestone_id):  # DUPLICATE!
            show_unlock_message(milestone_id)  # DUPLICATE!
```

**After** (good):
```python
# In export.py
check_module_export(module_name, console)

# In module_workflow.py
check_module_export(module_name, console)

# All logic in milestone_tracker.py (ONE PLACE!)
```

### 4. Fail Gracefully

If milestone tracker isn't available:
```python
try:
    from milestone_tracker import check_module_export
    check_module_export(module_name, console)
except ImportError:
    # Skip silently - don't break the workflow
    pass
```

## Testing the API

### Test Module Completion

```bash
# Simulate completing modules
python3 tests/milestones/milestone_tracker.py complete 00_setup
python3 tests/milestones/milestone_tracker.py complete 01_tensor
python3 tests/milestones/milestone_tracker.py complete 02_autograd

# Should show unlock message for perceptron
```

### Test Progress Display

```bash
python3 tests/milestones/milestone_tracker.py progress
```

### Test in Integration

```bash
# Complete a module through CLI
tito module complete 01

# Should automatically check for unlocks
```

## File Structure

```
tests/milestones/
├── milestone_tracker.py      # Core system (API)
├── test_learning_verification.py  # Pytest tests
├── API.md                     # This file
├── WORKFLOW.md               # User workflow
├── STUDENT_GUIDE.md          # Student documentation
├── PROGRESSION.md            # Historical context
├── QUICKSTART.md             # Quick reference
└── README.md                 # Overview
```

## Adding New Milestones

1. **Define in `MILESTONES` dict**:
```python
MILESTONES["new_milestone"] = {
    "name": "2025 - New Breakthrough",
    "requires": ["00_setup", "01_tensor", "15_new_module"],
    "test": "test_new_milestone_learning",
    "description": "Description of the breakthrough",
    "unlock_message": "🎉 You can now...",
}
```

2. **Add to `MILESTONE_ORDER`**:
```python
MILESTONE_ORDER = ["perceptron", "xor", "mlp_digits", "cnn", "transformer", "new_milestone"]
```

3. **Create pytest test**:
```python
# In test_learning_verification.py
def test_new_milestone_learning():
    # ... test implementation ...
    pass
```

That's it! The system automatically:
- Checks requirements
- Shows unlock messages
- Allows running the test
- Tracks completion

## Summary

**The milestone system is self-contained.** Other code calls it through clean API functions. No duplication, single source of truth, clean separation of concerns.

```python
# This is all you need to remember:
from milestone_tracker import check_module_export

# After module export:
check_module_export(module_name, console)

# That's it! Milestone tracker handles everything else.
```
