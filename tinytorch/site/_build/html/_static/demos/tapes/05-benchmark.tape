# VHS Tape: ⚡ Benchmark Your Build - Performance Validation
# Purpose: Show how to benchmark TinyTorch implementations
# Duration: 25-30 seconds

Output "gifs/05-benchmark.gif"

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
# SETUP PHASE (HIDDEN): Like a unit test, set up the environment cleanly first
# ==============================================================================
Hide

# Navigate to project and activate environment
Type "cd /Users/VJ/GitHub/TinyTorch"
Enter
Sleep 1s
Type "source activate.sh"
Enter
Sleep 3s

# Reset all modules to pristine state using git
Type "git checkout -- modules/"
Enter
Sleep 2s

# Clean any cached/built files
Type "rm -rf tinytorch/__pycache__ tinytorch/*.pyc"
Enter
Sleep 1s

# Complete modules 01-03 to have some implementations to benchmark
Type "tito module complete 01"
Enter
Sleep 5s
Type "tito module complete 02"
Enter
Sleep 5s
Type "tito module complete 03"
Enter
Sleep 5s

Show
# ==============================================================================
# DEMO BEGINS (VISIBLE): Now show the benchmarking workflow
# ==============================================================================

# Opening comment explaining what we'll demonstrate
Type "# Benchmark Your Build ⚡"
Sleep 2s
Enter
Sleep 500ms

# Run the baseline benchmark
Type "tito benchmark baseline"
Sleep 400ms
Enter
Sleep 15s  # Wait for benchmark to complete

# Final message
Type "# Fast code. Built by you. 🚀"
Sleep 3s
