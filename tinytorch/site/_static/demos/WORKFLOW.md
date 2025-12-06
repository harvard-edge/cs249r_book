# Demo Generation Workflow

Simple 3-step process to generate TinyTorch demo GIFs.

## Prerequisites

Install VHS:
```bash
# macOS
brew install vhs

# Linux
go install github.com/charmbracelet/vhs@latest
```

## Workflow

### Step 1: Validate (Required)

Make sure all commands work before recording:

```bash
./docs/_static/demos/scripts/validate_demos.sh
```

This will:
- Clone TinyTorch to `/tmp/TinyTorch_validate`
- Test all demo workflows (setup, module completion, milestones, etc.)
- Report pass/fail for each command

**Output:**
```
🔥 TinyTorch Demo Validation
========================================

📋 Demo 01: Zero to Ready
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Testing: git clone
  ✓ PASS

Testing: setup-environment.sh
  ✓ PASS

...

📊 Validation Summary
========================================
  Passed:   15
  Warnings: 0
  Failed:   0

✅ All critical tests passed!
```

### Step 2: Calibrate (Optional)

Measure command timings on your machine to optimize tape files:

```bash
./docs/_static/demos/scripts/demo.sh --calibrate
```

This will:
- Run each command and measure execution time
- Save timings to `docs/_static/demos/.timings.json`
- Suggest timeout values for your tape files

**When to calibrate:**
- First time generating demos on a new machine
- After changing commands in tape files
- If demos are timing out or waiting too long

**When to skip:**
- VHS tape files already have 120s timeouts
- Your machine is reasonably fast
- You're just testing changes

### Step 3: Generate (Record)

Create the actual demo GIF:

```bash
# Generate a specific demo
./docs/_static/demos/scripts/demo.sh --tape 01

# Test demo (quick 5-second test)
./docs/_static/demos/scripts/demo.sh --tape 00
```

**Available demos:**
- `00` - Quick test (5 seconds)
- `01` - Zero to Ready (clone → setup → activate)
- `02` - Build, Test, Ship (module completion)
- `03` - Milestone Unlocked (achievement system)
- `04` - Share Journey (community features)

**Output:**
```
🎬 TinyTorch Demo Generator
======================================
📹 Demo 01: 01-zero-to-ready

🧹 Cleaning /tmp/TinyTorch...
🎬 Recording demo (this may take 1-2 minutes)...

✅ Success!
   Created: 01-zero-to-ready.gif (2.3M)
   Moved to: docs/_static/demos/01-zero-to-ready.gif

💡 Preview:
  open docs/_static/demos/01-zero-to-ready.gif
```

### Step 4: Preview

```bash
# macOS
open docs/_static/demos/01-zero-to-ready.gif

# Linux
xdg-open docs/_static/demos/01-zero-to-ready.gif
```

## Quick Reference

```bash
# Full workflow
./docs/_static/demos/scripts/validate_demos.sh              # Validate
./docs/_static/demos/scripts/demo.sh --calibrate            # [Optional] Calibrate
./docs/_static/demos/scripts/demo.sh --tape 01              # Generate
open docs/_static/demos/01-zero-to-ready.gif                # Preview

# Quick test workflow (no validation)
./docs/_static/demos/scripts/demo.sh --tape 00              # Test demo only
```

## Troubleshooting

### Validation fails

```bash
❌ Some tests failed. Please fix issues before generating demos.

Debug by running the failing command manually:
  cd /tmp/TinyTorch_validate
  source activate.sh
  # Run the failing command
```

**Common issues:**
- Environment not set up correctly
- Missing dependencies
- Changed command names/flags

### Demo times out

**Symptom:** VHS waits 120s then fails

**Fix:** Check your network/machine speed:
```bash
# Run commands manually to see timing
cd /tmp
rm -rf TinyTorch
time git clone https://github.com/mlsysbook/TinyTorch.git
# If > 120s, increase timeout in tape file
```

### GIF is too large

**Symptom:** File > 5MB

**Fix:** Reduce framerate or dimensions in tape file:
```vhs
Set Framerate 24  # Lower from 30
Set Width 1024    # Reduce from 1280
Set Height 576    # Reduce from 720
```

## File Structure

```
docs/_static/demos/
├── WORKFLOW.md              # This file
├── README.md                # Technical details (VHS syntax, patterns)
├── scripts/
│   ├── validate_demos.sh   # Step 1: Validate
│   └── demo.sh             # Step 2-3: Calibrate + Generate
├── tapes/
│   ├── 00-test.tape        # Quick test
│   ├── 01-zero-to-ready.tape
│   ├── 02-build-test-ship.tape
│   ├── 03-milestone-unlocked.tape
│   └── 04-share-journey.tape
└── *.gif                   # Generated demos (gitignored)
```

## Tips

1. **Always validate first** - Saves time by catching issues early
2. **Calibration is optional** - Only needed for timing optimization
3. **Test with Demo 00** - Quick 5-second check before recording long demos
4. **Clean slate** - Scripts automatically clean `/tmp/TinyTorch*`
5. **Preview before committing** - Make sure GIF looks good

## CI/CD (Future)

The validation script can run in CI to catch breaking changes:

```yaml
- name: Validate demos
  run: ./docs/_static/demos/scripts/validate_demos.sh
```

