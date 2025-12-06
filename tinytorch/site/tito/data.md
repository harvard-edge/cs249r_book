# Progress & Data Management

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h2 style="margin: 0 0 1rem 0; color: #495057;">Track Your Journey</h2>
<p style="margin: 0; font-size: 1.1rem; color: #6c757d;">Understanding progress tracking, data management, and reset commands</p>
</div>

**Purpose**: Learn how TinyTorch tracks your progress, where your data lives, and how to manage it effectively.

## Your Learning Journey: Two Tracking Systems

TinyTorch uses a clean, simple approach to track your ML systems engineering journey:

```{mermaid}
graph LR
    A[Build Modules] --> B[Complete 01-20]
    B --> C[Export to Package]
    C --> D[Unlock Milestones]
    D --> E[Achieve 1957-2018]
    E --> F[Track Progress]

    style A fill:#e3f2fd
    style B fill:#fffbeb
    style C fill:#f0fdf4
    style D fill:#fef3c7
    style E fill:#f3e5f5
    style F fill:#e8eaf6
```

### The Two Systems

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 2rem 0;">

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3;">
<h4 style="margin: 0 0 0.5rem 0; color: #1976d2;">📦 Module Progress</h4>
<p style="margin: 0.5rem 0; font-size: 0.95rem; color: #37474f;">What you BUILD (01-20)</p>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem; font-size: 0.9rem; color: #546e7a;">
<li>Tensor, Autograd, Optimizers</li>
<li>Layers, Training, DataLoader</li>
<li>Convolutions, Transformers</li>
<li>Your complete ML framework</li>
</ul>
</div>

<div style="background: #f3e5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #9c27b0;">
<h4 style="margin: 0 0 0.5rem 0; color: #7b1fa2;">🏆 Milestone Achievements</h4>
<p style="margin: 0.5rem 0; font-size: 0.95rem; color: #37474f;">What you ACHIEVE (01-06)</p>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem; font-size: 0.9rem; color: #546e7a;">
<li>Perceptron (1957)</li>
<li>MLP Revival (1986)</li>
<li>CNN Revolution (1998)</li>
<li>AlexNet Era (2012)</li>
<li>Transformer Era (2017)</li>
<li>MLPerf (2018)</li>
</ul>
</div>

</div>

**Simple relationship**:
- Complete modules → Unlock milestones → Achieve historical ML recreations
- Build capabilities → Validate with history → Track achievements

---

## Where Your Data Lives

All your progress is stored in the `.tito/` folder:

```
TinyTorch/
├── .tito/                    ← Your progress data
│   ├── config.json           ← User preferences
│   ├── progress.json         ← Module completion (01-20)
│   ├── milestones.json       ← Milestone achievements (01-06)
│   └── backups/              ← Automatic safety backups
│       ├── 01_tensor_YYYYMMDD_HHMMSS.py
│       ├── 02_activations_YYYYMMDD_HHMMSS.py
│       └── ...
├── modules/                  ← Where you edit
├── tinytorch/                ← Where code exports
└── ...
```

### Understanding Each File

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**`config.json`** - User Preferences
```json
{
  "logo_theme": "standard"
}
```
- UI preferences
- Display settings
- Personal configuration

**`progress.json`** - Module Completion
```json
{
  "version": "1.0",
  "completed_modules": [1, 2, 3, 4, 5, 6, 7],
  "completion_dates": {
    "1": "2025-11-16T10:00:00",
    "2": "2025-11-16T11:00:00",
    ...
  }
}
```
- Tracks which modules (01-20) you've completed
- Records when you completed each
- Updated by `tito module complete XX`

**`milestones.json`** - Milestone Achievements
```json
{
  "version": "1.0",
  "completed_milestones": ["03"],
  "completion_dates": {
    "03": "2025-11-16T15:00:00"
  }
}
```
- Tracks which milestones (01-06) you've achieved
- Records when you achieved each
- Updated by `tito milestone run XX`

**`backups/`** - Module Backups
- Automatic backups before operations
- Timestamped copies of your implementations
- Safety net for module development
- Format: `XX_name_YYYYMMDD_HHMMSS.py`

</div>

---

## Unified Progress View

### See Everything: `tito status`

<div style="background: #e8eaf6; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #5e35b1; margin: 1.5rem 0;">

```bash
tito status
```

**Shows your complete learning journey in one view**:

```
╭─────────────── 📊 TinyTorch Progress ────────────────╮
│                                                      │
│  📦 Modules Completed: 7/20 (35%)                    │
│  🏆 Milestones Achieved: 1/6 (17%)                   │
│  📍 Last Activity: Module 07 (2 hours ago)           │
│                                                      │
│  Next Steps:                                         │
│    • Complete modules 08-09 to unlock Milestone 04   │
│                                                      │
╰──────────────────────────────────────────────────────╯

Module Progress:
  ✅ 01 Tensor
  ✅ 02 Activations
  ✅ 03 Layers
  ✅ 04 Losses
  ✅ 05 Autograd
  ✅ 06 Optimizers
  ✅ 07 Training
  🔒 08 DataLoader
  🔒 09 Convolutions
  🔒 10 Normalization
  ...

Milestone Achievements:
  ✅ 03 - MLP Revival (1986)
  🎯 04 - CNN Revolution (1998) [Ready after modules 08-09]
  🔒 05 - Transformer Era (2017)
  🔒 06 - MLPerf (2018)
```

**Use this to**:
- Check overall progress
- See next recommended steps
- Understand milestone prerequisites
- Track your learning journey

</div>

---

## Data Management Commands

### Reset Your Progress

<div style="background: #fff5f5; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e74c3c; margin: 1.5rem 0;">

**Starting fresh?** Reset commands let you start over cleanly.

#### Reset Everything

```bash
tito reset all
```

**What this does**:
- Clears all module completion
- Clears all milestone achievements
- Resets configuration to defaults
- Keeps your code in `modules/` safe
- Asks for confirmation before proceeding

**Example output**:
```
⚠️  Warning: This will reset ALL progress

This will clear:
  • Module completion (7 modules)
  • Milestone achievements (1 milestone)
  • Configuration settings

Your code in modules/ will NOT be deleted.

Continue? [y/N]: y

✅ Creating backup at .tito_backup_20251116_143000/
✅ Clearing module progress
✅ Clearing milestone achievements
✅ Resetting configuration

🔄 Reset Complete!

You're ready to start fresh.
Run: tito module start 01
```

#### Reset Module Progress Only

```bash
tito reset progress
```

**What this does**:
- Clears module completion tracking only
- Keeps milestone achievements
- Keeps configuration
- Useful for re-doing module workflow

#### Reset Milestone Achievements Only

```bash
tito reset milestones
```

**What this does**:
- Clears milestone achievements only
- Keeps module completion
- Keeps configuration
- Useful for re-running historical recreations

#### Safety: Automatic Backups

```bash
# Create backup before reset
tito reset all --backup
```

**What this does**:
- Creates timestamped backup: `.tito_backup_YYYYMMDD_HHMMSS/`
- Contains complete copy of `.tito/` folder
- Allows manual restore if needed
- Automatic before any destructive operation

</div>

---

## Data Safety & Recovery

### Automatic Backups

TinyTorch automatically backs up your work:

<div style="background: #f0fdf4; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #22c55e; margin: 1.5rem 0;">

**When backups happen**:
1. **Before module start**: Backs up existing work
2. **Before reset**: Creates full `.tito/` backup
3. **Before module reset**: Saves current implementation

**Where backups go**:
```
.tito/backups/
├── 01_tensor_20251116_100000.py
├── 01_tensor_20251116_143000.py
├── 03_layers_20251115_180000.py
└── ...
```

**How to use backups**:
```bash
# Backups are timestamped - find the one you need
ls -la .tito/backups/

# Manually restore if needed
cp .tito/backups/03_layers_20251115_180000.py modules/03_layers/layers_dev.py
```

</div>

### What If .tito/ Is Deleted?

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">

**No problem!** TinyTorch recovers gracefully:

```bash
# If .tito/ is deleted, next command recreates it
tito system health
```

**What happens**:
1. TinyTorch detects missing `.tito/` folder
2. Creates fresh folder structure
3. Initializes empty progress tracking
4. Your code in `modules/` and `tinytorch/` is safe
5. You can continue from where you left off

**Important**: Your actual code (source in `src/`, notebooks in `modules/`, package in `tinytorch/`) is separate from progress tracking (in `.tito/`). Deleting `.tito/` only resets progress tracking, not your implementations.

</div>

---

## Data Health Checks

### Verify Data Integrity

<div style="background: #e3f2fd; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1.5rem 0;">

```bash
tito system health
```

**Now includes data health checks**:

```
╭────────── 🔍 TinyTorch System Check ──────────╮
│                                               │
│  ✅ Environment setup                         │
│  ✅ Dependencies installed                    │
│  ✅ TinyTorch in development mode             │
│  ✅ Data files intact                         │
│    ✓ .tito/progress.json valid               │
│    ✓ .tito/milestones.json valid             │
│    ✓ .tito/config.json valid                 │
│  ✅ Backups directory exists                  │
│                                               │
╰───────────────────────────────────────────────╯

All systems ready! 🚀
```

**If data is corrupted**:
```
❌ Data files corrupted
  ✗ .tito/progress.json is malformed

Fix:
  tito reset progress

Or restore from backup:
  cp .tito_backup_YYYYMMDD/.tito/progress.json .tito/
```

</div>

---

## Best Practices

### Regular Progress Checks

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

**Good habits**:

1. **Check status regularly**:
   ```bash
   tito status
   ```
   See where you are, what's next

2. **Verify environment before work**:
   ```bash
   tito system health
   ```
   Catch issues early

3. **Let automatic backups work**:
   - Don't disable them
   - They're your safety net
   - Cleanup happens automatically

4. **Backup before experiments**:
   ```bash
   tito reset all --backup  # If trying something risky
   ```

5. **Version control for code**:
   ```bash
   git commit -m "Completed Module 05: Autograd"
   ```
   `.tito/` is gitignored - use git for code versions

</div>

---

## Understanding What Gets Tracked

### Modules (Build Progress)

**Tracked when**: You run `tito module complete XX`

**What's recorded**:
- Module number (1-20)
- Completion timestamp
- Test results (passed/failed)

**Visible in**:
- `tito module status`
- `tito status`
- `.tito/progress.json`

### Milestones (Achievement Progress)

**Tracked when**: You run `tito milestone run XX`

**What's recorded**:
- Milestone ID (01-06)
- Achievement timestamp
- Number of attempts (if multiple runs)

**Visible in**:
- `tito milestone status`
- `tito status`
- `.tito/milestones.json`

### What's NOT Tracked

<div style="background: #fffbeb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b; margin: 1.5rem 0;">

**TinyTorch does NOT track**:
- Your actual code implementations (source in `src/`, notebooks in `modules/`, package in `tinytorch/`)
- How long you spent on each module
- How many times you edited files
- Your test scores or grades
- Personal information
- Usage analytics

**Why**: TinyTorch is a local, offline learning tool. Your privacy is protected. All data stays on your machine.

</div>

---

## Common Data Scenarios

### Scenario 1: "I want to start completely fresh"

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

```bash
# Create backup first (recommended)
tito reset all --backup

# Or just reset
tito reset all

# Start from Module 01
tito module start 01
```

**Result**: Clean slate, progress tracking reset, your code untouched

</div>

### Scenario 2: "I want to re-run milestones but keep module progress"

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

```bash
# Reset only milestone achievements
tito reset milestones

# Re-run historical recreations
tito milestone run 03
tito milestone run 04
```

**Result**: Module completion preserved, milestone achievements reset

</div>

### Scenario 3: "I accidentally deleted .tito/"

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

```bash
# Just run any tito command
tito system health

# OR

# If you have a backup
cp -r .tito_backup_YYYYMMDD/ .tito/
```

**Result**: `.tito/` folder recreated, either fresh or from backup

</div>

### Scenario 4: "I want to share my progress with a friend"

<div style="background: #f8f9fa; padding: 1.5rem; border: 1px solid #dee2e6; border-radius: 0.5rem; margin: 1.5rem 0;">

```bash
# Create backup with timestamp
tito reset all --backup  # (then cancel when prompted)

# Share the backup folder
cp -r .tito_backup_YYYYMMDD/ ~/Desktop/my-tinytorch-progress/
```

**Result**: Friend can see your progress by copying to their `.tito/` folder

</div>

---

## FAQ

### Q: Will resetting delete my code?

**A**: No! Reset commands only affect progress tracking in `.tito/`. Your source code in `src/`, notebooks in `modules/`, and exported code in `tinytorch/` are never touched.

### Q: Can I manually edit progress.json?

**A**: Yes, but not recommended. Use `tito` commands instead. Manual edits might break validation.

### Q: What if I want to re-export a module?

**A**: Just run `tito module complete XX` again. It will re-run tests and re-export. Progress tracking remains unchanged.

### Q: How do I see my completion dates?

**A**: Run `tito status` for a formatted view, or check `.tito/progress.json` and `.tito/milestones.json` directly.

### Q: Can I delete backups?

**A**: Yes, backups in `.tito/backups/` can be deleted manually. They're safety nets, not requirements.

### Q: Is my data shared anywhere?

**A**: No. TinyTorch is completely local. No data leaves your machine. No tracking, no analytics, no cloud sync.

---

## Next Steps

<div style="background: #f8f9fa; padding: 2rem; border-radius: 0.5rem; margin: 2rem 0; text-align: center;">
<h3 style="margin: 0 0 1rem 0; color: #495057;">Keep Building!</h3>
<p style="margin: 0 0 1.5rem 0; color: #6c757d;">Now that you understand data management, focus on what matters: building ML systems</p>
<a href="modules.html" style="display: inline-block; background: #007bff; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500; margin-right: 1rem;">Module Workflow →</a>
<a href="milestones.html" style="display: inline-block; background: #9c27b0; color: white; padding: 0.75rem 1.5rem; border-radius: 0.25rem; text-decoration: none; font-weight: 500;">Milestone System →</a>
</div>

---

*Your progress is tracked, your data is safe, and your journey is yours. TinyTorch keeps track of what you've built and achieved - you focus on learning ML systems engineering.*
