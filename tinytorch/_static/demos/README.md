# TinyTorch Demo Generation

One script to rule them all.

## Quick Start

```bash
# Interactive mode (asks questions)
./docs/_static/demos/scripts/tito-demo.sh
```

That's it! The script will ask what you want to do and guide you through.

## What It Does

The script handles everything in one go:

### **Full Workflow** (Recommended)
1. **Validate** - Tests all demo commands work (clones TinyTorch to `/tmp`, runs setup, etc.)
2. **Time** - Measures command execution times during validation (smart - no duplicate runs!)
3. **Generate** - Creates demo GIF using VHS

### **Individual Steps** (If Needed)
- **Validate only** - Just test commands without timing or generation
- **Generate only** - Create GIF without validation (risky if commands changed)

## Interactive Mode

```bash
./docs/_static/demos/scripts/tito-demo.sh
```

You'll see:

```
  ╔═══════════════════════════════════════╗
  ║   🔥 TinyTorch Demo Studio 🎬        ║
  ╚═══════════════════════════════════════╝

What would you like to do?

  1) Validate only (test all commands work)
  2) Generate demo GIF only
  3) Full workflow (validate + timing + generate) ← Recommended
  4) Exit

Choose [1-4]:
```

Pick option 3 (Full workflow), answer which demo you want, done.

## Live Progress

The script shows live output as commands run (not silent!):

```
📋 Step 1: Validation + Timing Collection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏳ Testing: git clone

  │ Cloning into 'TinyTorch_validate'...
  │ remote: Enumerating objects: 1234, done.
  │ remote: Counting objects: 100% (456/456), done.
  │ remote: Compressing objects: 100% (234/234), done.
  │ remote: Total 1234 (delta 123), reused 789 (delta 56)
  │ Receiving objects: 100% (1234/1234), 2.34 MiB | 1.23 MiB/s, done.
  │ Resolving deltas: 100% (567/567), done.

  ✓ PASS (12.45s)

⏳ Testing: setup-environment.sh

  │ 🔥 TinyTorch Environment Setup
  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │ 
  │ 📦 Creating virtual environment...
  │   ✓ Virtual environment created
  │ 
  │ 📦 Installing dependencies...
  │   ✓ numpy installed
  │   ✓ pytest installed
  │   ...
  │ 
  │ ✅ TinyTorch environment setup complete

  ✓ PASS (45.23s)

⏳ Testing: tito module status

  │ Module Status
  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  │ 01_tensor          ⬜ Not Started
  │ 02_activations     ⬜ Not Started
  │ ...

  ✓ PASS (0.87s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱  Timing Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Command                        Time (s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
git clone                         12.45s
setup-environment.sh              45.23s
tito module status                 0.87s

💡 VHS wait syntax for tape files:
   Wait+Line@10ms /profvjreddi/

✅ All tests passed!

🎬 Step 2: Generate Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏳ Step 2.1: Cleaning /tmp/TinyTorch...
  ✓ Clean

⏳ Step 2.2: Recording with VHS (1-2 minutes)...

  Setting up terminal...
  Executing commands...
  Recording frames...
  Generating GIF...

✅ Recording complete! (took 87s)

⏳ Step 2.3: Moving to docs/_static/demos/
  ✓ Saved: docs/_static/demos/01-zero-to-ready.gif (2.3M)

💡 Preview with:
  open docs/_static/demos/01-zero-to-ready.gif

🎉 Complete! All steps done successfully.
```

**You see everything happen in real-time** - no silent waiting! Perfect for long-running commands like git clone and setup.

## Command Line Mode (Optional)

If you prefer non-interactive:

```bash
# Full workflow (recommended)
./docs/_static/demos/scripts/tito-demo.sh full 01

# Just validate (no timing, no generation)
./docs/_static/demos/scripts/tito-demo.sh validate

# Just generate demo 01 (skip validation)
./docs/_static/demos/scripts/tito-demo.sh generate 01
```

### Debug Mode (Skip Git Clone)

If you have slow internet or are iterating quickly, skip the git clone:

```bash
# Interactive mode will ask if you want to skip clone
./docs/_static/demos/scripts/tito-demo.sh

# Or use --skip-clone flag directly
./docs/_static/demos/scripts/tito-demo.sh validate --skip-clone
./docs/_static/demos/scripts/tito-demo.sh full 01 --skip-clone
```

This will:
- Skip the git clone step (saves 10-30s depending on internet)
- Use existing `/tmp/TinyTorch_validate` if present
- Otherwise copy from your current repo directory
- Run all other validation tests normally

**Perfect for:** Debugging, slow internet, rapid iteration

**Tip:** Use `full 01` for the safest workflow - validates, times, and generates in one command.

## Available Demos

- `00` - Quick test (5 seconds, verifies VHS setup)
- `01` - Zero to Ready (clone → setup → activate)
- `02` - Build, Test, Ship (module completion workflow)
- `03` - Milestone Unlocked (achievement system)
- `04` - Share Your Journey (community features)

## Prerequisites

Install VHS (terminal recorder):

```bash
# macOS
brew install vhs

# Linux
go install github.com/charmbracelet/vhs@latest
```

## File Structure

```
docs/_static/demos/
├── README.md                # This file
├── scripts/
│   ├── tito-demo.sh        # 🎯 ONE SCRIPT (interactive)
│   ├── validate_demos.sh   # [Legacy - use tito-demo.sh instead]
│   └── demo.sh             # [Legacy - use tito-demo.sh instead]
├── tapes/                  # VHS tape files (source of truth)
│   ├── 00-test.tape
│   ├── 01-zero-to-ready.tape
│   ├── 02-build-test-ship.tape
│   ├── 03-milestone-unlocked.tape
│   └── 04-share-journey.tape
└── *.gif                   # Generated demos (gitignored)
```

## VHS Tape Files

Each `.tape` file is a script for VHS to record a terminal session:

```vhs
# Example: 01-zero-to-ready.tape
Output "01-zero-to-ready.gif"

Set Width 1280
Set Height 720
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "

Type "git clone https://github.com/mlsysbook/TinyTorch.git"
Enter
Wait+Line@10ms /profvjreddi/ 120s  # Wait for clone (max 120s)

Type "cd TinyTorch"
Enter
Wait+Line@10ms /profvjreddi/

Type "./setup-environment.sh"
Enter
Wait+Line@10ms /profvjreddi/ 120s  # Wait for setup

# ... more commands
```

### Key Patterns

**Robust Waiting:**
```vhs
Wait+Line@10ms /profvjreddi/ 120s  # Wait for prompt (max 120s)
```

Instead of fixed `Sleep` times, wait for the prompt to return. This works regardless of machine speed.

**Custom Prompt:**
```vhs
Env PS1 "@profvjreddi 🔥 › "  # Sets prompt in the recording
```

Makes it easy to detect when commands finish.

## Troubleshooting

### Validation fails

The script will show which test failed and suggest debug commands:

```bash
❌ Some tests failed

Debug:
  cd /tmp/TinyTorch_validate
  source activate.sh
  # Run failing command manually
```

### Demo times out

If VHS waits 120s then fails, your network/machine might be slow:

```bash
# Test manually to see timing
cd /tmp
rm -rf TinyTorch
time git clone https://github.com/mlsysbook/TinyTorch.git

# If > 120s, edit the tape file and increase timeout
```

### GIF is too large (>5MB)

Edit the tape file and reduce quality:

```vhs
Set Framerate 24  # Lower from 30
Set Width 1024    # Reduce from 1280
Set Height 576    # Reduce from 720
```

## Manual Recording (Alternative Tools)

If you prefer to use Terminalizer, Asciinema, or other recording tools instead of VHS:

### Extract Command List

Use the converter script to extract commands from VHS tape files:

```bash
# Convert VHS tape to Terminalizer config
./docs/_static/demos/scripts/vhs-to-terminalizer.sh docs/_static/demos/tapes/01-zero-to-ready.tape

# This creates a .yml file with:
# - All commands extracted
# - Timing information converted
# - Terminal settings (dimensions, theme)
```

### Manual Recording Workflow

1. **Extract commands** from the tape file (see above)
2. **Review the .yml config** to see the command sequence
3. **Record manually** with your preferred tool:
   ```bash
   # With Terminalizer
   terminalizer record demo-01 -c 01-zero-to-ready.yml

   # With Asciinema
   asciinema rec demo-01.cast

   # Or just read the tape file directly - it's human-readable!
   cat docs/_static/demos/tapes/01-zero-to-ready.tape
   ```
4. **Type commands** from the sequence during recording
5. **Render to GIF** using your tool's output format

### Why Use VHS?

- **Fully automated** - No manual typing during recording
- **Reproducible** - Same GIF every time
- **Version controlled** - Tape files track command changes
- **Fast iteration** - Edit tape, re-record, done

### Why Use Manual Tools?

- **More polish** - Fine-tune pauses and interactions
- **Custom workflows** - Your own recording preferences
- **Tool familiarity** - Stick with what you know

**Tip:** The VHS tape files are human-readable scripts. You can use them as a reference for manual recording even without the converter!

## Development Tips

1. **Edit tape files directly** - They're in `tapes/*.tape`
2. **Test with Demo 00** - Quick 5-second validation
3. **Calibrate if timing issues** - Only needed if demos timeout
4. **Preview before committing** - Always check the GIF looks good

## CI/CD (Future)

The validation can run in GitHub Actions:

```yaml
- name: Validate demos
  run: ./docs/_static/demos/scripts/tito-demo.sh validate
```

## Resources

- [VHS Documentation](https://github.com/charmbracelet/vhs)
- [VHS Examples](https://github.com/charmbracelet/vhs/tree/main/examples)
- [Tape File Format](https://github.com/charmbracelet/vhs#tape-file-format)
