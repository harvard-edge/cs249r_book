# VHS Tape: 🔥 Welcome to TinyTorch - The Big Logo
# Purpose: Show the beautiful branded welcome screen with full command reference
# Duration: 10-12 seconds

Output "gifs/00-welcome.gif"

# Window bar for realistic terminal look (must be at top)
Set WindowBar Colorful

# Taller dimensions to show full tito output
Set Width 1280
Set Height 1000
Set FontSize 18
Set FontFamily "JetBrains Mono, Monaco, Menlo, monospace"
Set Theme { "name": "TinyTorch", "black": "#1E1E2E", "red": "#F38BA8", "green": "#A6E3A1", "yellow": "#F9E2AF", "blue": "#89B4FA", "magenta": "#CBA6F7", "cyan": "#94E2D5", "white": "#CDD6F4", "brightBlack": "#585B70", "brightRed": "#F38BA8", "brightGreen": "#A6E3A1", "brightYellow": "#F9E2AF", "brightBlue": "#89B4FA", "brightMagenta": "#CBA6F7", "brightCyan": "#94E2D5", "brightWhite": "#CDD6F4", "background": "#1E1E2E", "foreground": "#CDD6F4", "selection": "#585B70", "cursor": "#F5E0DC" }
Set Padding 30
Set Framerate 30
Set Margin 20
Set MarginFill "#1E1E2E"
Set BorderRadius 10
Set LoopOffset 10%
Set TypingSpeed 100ms

# Set shell with custom prompt for reliable waiting
Set Shell bash
Env PS1 "@profvjreddi 🔥 › "
Env COLUMNS "128"

# ==============================================================================
# OPENING: Welcome message to set the tone
# ==============================================================================

Type "# Welcome to Tiny🔥Torch!"
Sleep 2s
Enter
Sleep 500ms

# ==============================================================================
# SETUP: Show the activation process (visible for this demo)
# ==============================================================================

Type "cd /Users/VJ/GitHub/TinyTorch"
Sleep 400ms
Enter
Sleep 1s

Type "source activate.sh"
Sleep 400ms
Enter
Sleep 3s

# ==============================================================================
# MAIN FEATURE: Display the complete tito command reference
# ==============================================================================

Type "tito"
Sleep 400ms
Enter
Sleep 8s  # Let viewers read the full feature list

# ==============================================================================
# CLOSING: Inspire them to start the journey
# ==============================================================================

Type "# Let's recreate PyTorch together! 🔥"
Sleep 3s
