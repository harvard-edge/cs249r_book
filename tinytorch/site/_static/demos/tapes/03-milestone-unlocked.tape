# VHS Tape: 🏆 Milestone Unlocked - Recreate ML History
# Purpose: Show milestone unlock and training with historical context
# Duration: 30-35 seconds

Output "gifs/03-milestone-unlocked.gif"

# Window bar for realistic terminal look
Set WindowBar Colorful

# Carousel-optimized dimensions (16:9 aspect ratio)
Set Width 1280
Set Height 720
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme { "name": "TinyTorch", "black": "#1E1E2E", "red": "#F38BA8", "green": "#A6E3A1", "yellow": "#F9E2AF", "blue": "#89B4FA", "magenta": "#CBA6F7", "cyan": "#94E2D5", "white": "#CDD6F4", "brightBlack": "#585B70", "brightRed": "#F38BA8", "brightGreen": "#A6E3A1", "brightYellow": "#F9E2AF", "brightBlue": "#89B4FA", "brightMagenta": "#CBA6F7", "brightCyan": "#94E2D5", "brightWhite": "#CDD6F4", "background": "#1E1E2E", "foreground": "#CDD6F4", "selection": "#585B70", "cursor": "#F5E0DC" }
Set Padding 30
Set Framerate 30
Set Margin 20
Set MarginFill "#1E1E2E"
Set BorderRadius 10
Set LoopOffset 10%

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "
Set TypingSpeed 100ms

# ==============================================================================
# SETUP PHASE (HIDDEN): Unit test pattern - prepare clean environment
# ==============================================================================
Hide

# 1. Navigate to project root
Type "cd /Users/VJ/GitHub/TinyTorch"
Enter
Sleep 1s

# 2. Activate the TinyTorch environment
Type "source activate.sh"
Enter
Sleep 3s

# 3. Reset all modules to pristine state (git checkout)
Type "git checkout -- modules/"
Enter
Sleep 2s

# 4. Clean cached/built files
Type "rm -rf tinytorch/__pycache__ tinytorch/*.pyc"
Enter
Sleep 1s

# 5. Complete prerequisite modules 01-06 (needed to unlock module 07)
Type "tito module complete 01"
Enter
Sleep 5s
Type "tito module complete 02"
Enter
Sleep 5s
Type "tito module complete 03"
Enter
Sleep 5s
Type "tito module complete 04"
Enter
Sleep 5s
Type "tito module complete 05"
Enter
Sleep 5s
Type "tito module complete 06"
Enter
Sleep 5s

Show

Ctrl+L  # Clear screen to hide setup command history

# ==============================================================================
# OPENING: Introduce the milestone concept
# ==============================================================================

Type "# Milestone: Recreate ML History 🏆"
Sleep 2s
Enter
Sleep 500ms

# ==============================================================================
# STEP 1: CHECK PROGRESS - Show where we are in the journey
# ==============================================================================

Type "tito module status"
Sleep 400ms
Enter
Sleep 4s  # Let viewers see modules 01-06 are completed

# ==============================================================================
# STEP 2: UNLOCK MILESTONE - Complete module 07 (Training)
# ==============================================================================

Type "tito module complete 07"
Sleep 400ms
Enter
Sleep 10s  # Watch tests pass and module export to TinyTorch

# ==============================================================================
# STEP 3: LIST MILESTONES - See what historical achievements are available
# ==============================================================================

Type "tito milestones list"
Sleep 400ms
Enter
Sleep 3s  # Read the milestone descriptions

# ==============================================================================
# STEP 4: RUN MILESTONE - Recreate Rumelhart's 1986 breakthrough
# ==============================================================================

Type "echo '' | tito milestones run 03"
Sleep 400ms
Enter
Sleep 60s  # Watch the training recreate history (training takes ~30s)

# ==============================================================================
# CLOSING: Celebrate recreating ML history
# ==============================================================================

Type "# You recreated Rumelhart's 1986 breakthrough! 🎉"
Sleep 3s

