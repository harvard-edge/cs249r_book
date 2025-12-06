# VHS Tape: ⚡ Zero to Ready in 30 Seconds
# Purpose: Show how easy TinyTorch setup is
# Duration: 15-20 seconds

Output "gifs/01-zero-to-ready.gif"

# Window bar for realistic terminal look (must be at top)
Set WindowBar Colorful

# Carousel-optimized dimensions (16:9 aspect ratio)
Set Width 1280
Set Height 720
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme "Catppuccin Mocha"
Set Padding 60
Set Framerate 30
Set TypingSpeed 100ms
Set LoopOffset 0%

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "

# ==============================================================================
# STEP 1: START IN CLEAN DIRECTORY
# ==============================================================================

Type "cd /tmp"
Sleep 300ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 500ms

# ==============================================================================
# STEP 2: CLONE REPOSITORY - Get TinyTorch from GitHub
# ==============================================================================

Type "git clone https://github.com/mlsysbook/TinyTorch.git"
Sleep 400ms
Enter
Wait+Line@120s /profvjreddi/  # Wait for clone to complete (up to 120s)
Sleep 500ms

# ==============================================================================
# STEP 3: ENTER PROJECT - Navigate into TinyTorch
# ==============================================================================

Type "cd TinyTorch"
Sleep 300ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 500ms

# ==============================================================================
# STEP 4: SETUP ENVIRONMENT - Automatic detection (conda/venv)
# ==============================================================================

Type "./setup-environment.sh"
Sleep 400ms
Enter
Wait+Line@120s /profvjreddi/  # Setup creates virtual environment (up to 120s)
Sleep 1s

# ==============================================================================
# STEP 5: ACTIVATE - Get access to tito commands
# ==============================================================================

Type "source activate.sh"
Sleep 300ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 1s

# ==============================================================================
# STEP 6: CHECK STATUS - See all 20 modules ready to build
# ==============================================================================

Type "tito module status"
Sleep 400ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 2s

# ==============================================================================
# STEP 7: READ THE STORY - Understand the TinyTorch philosophy
# ==============================================================================

Type "tito logo"
Sleep 400ms
Enter
Wait+Line@10ms /profvjreddi/
Sleep 3s

# ==============================================================================
# CLOSING: You're ready to start building!
# ==============================================================================

Type "# Ready to build your ML framework from scratch 🔥"
Sleep 2s
