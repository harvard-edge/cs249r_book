# End-to-End User Journey Tests

This directory contains tests that simulate the complete student experience with TinyTorch.

## Philosophy

Unlike unit tests (test components) or integration tests (test interactions), E2E tests verify:
- **The complete user journey works from start to finish**
- **Commands chain together correctly**
- **Progress tracking persists across operations**
- **Milestones unlock at the right time**

## Test Levels

### Level 1: Quick Verification (~30 seconds)
```bash
pytest tests/e2e/test_user_journey.py -k quick -v
```
- Verifies CLI commands work
- Checks module/milestone structure exists
- No actual training

### Level 2: Module Flow (~2 minutes)
```bash
pytest tests/e2e/test_user_journey.py -k module_flow -v
```
- Tests module start → complete → export cycle
- Verifies progress tracking
- Tests prerequisite enforcement

### Level 3: Full Journey (~5-10 minutes)
```bash
pytest tests/e2e/test_user_journey.py -k full_journey -v
```
- Complete flow from setup to first milestone
- Actual module completion
- Milestone unlock verification

### Level 4: Release Validation (~30 minutes)
```bash
./tests/e2e/validate_release.sh
```
- Simulates fresh git clone
- Full setup through milestone 01
- Tests pip installability
- Comprehensive verification

## Running Before Release

Before any release, run:

```bash
# Quick sanity check
make e2e-quick

# Full validation (recommended before release)
make e2e-full
```

## What Gets Tested

1. **Setup Flow**
   - `tito setup --skip-profile` works non-interactively
   - Environment validation passes
   - Package is importable

2. **Module Workflow**
   - `tito module start 01` works
   - `tito module complete 01` exports correctly
   - Progress tracking updates
   - `tito module status` shows correct state

3. **Prerequisite Enforcement**
   - Can't start module 02 without completing 01
   - Can't run milestone 02 without prerequisites

4. **Milestone Flow**
   - Milestones list correctly
   - `tito milestone run 01` executes
   - Completion is tracked

5. **Error Handling**
   - Graceful failures for invalid commands
   - Helpful error messages
