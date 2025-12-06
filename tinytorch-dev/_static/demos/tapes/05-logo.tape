# VHS Tape: 🔥 The TinyTorch Story - Philosophy & Vision
# Purpose: Show the beautiful story behind TinyTorch with tito logo
# Duration: 45-50 seconds

Output "gifs/05-logo.gif"

# Window bar for realistic terminal look
Set WindowBar Colorful

# Standard carousel dimensions (16:9 aspect ratio)
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
# SETUP: Activate environment to access tito command (hidden)
# ==============================================================================

Hide
Type "cd /Users/VJ/GitHub/TinyTorch"
Enter
Sleep 500ms
Type "source activate.sh"
Enter
Sleep 3s
Show

Ctrl+L  # Clear screen to hide setup command history

# ==============================================================================
# OPENING: Set the stage with a comment about the story
# ==============================================================================

Type "# The Story Behind TinyTorch 🔥"
Sleep 2s
Enter
Sleep 500ms

# ==============================================================================
# SHOW THE LOGO: Display the complete philosophy
# ==============================================================================

Type "tito logo"
Sleep 400ms
Enter
Sleep 3s  # Let the output complete

# ==============================================================================
# SCROLL THROUGH THE LOGO: Use Up/Down arrows to navigate like a real user
# ==============================================================================

# Scroll way up to see the top (the ASCII art logo)
Up 50
Sleep 3s

# Now scroll down slowly through each section
Down 8
Sleep 3s

Down 8
Sleep 3s

Down 8
Sleep 3s

Down 8
Sleep 3s

Down 8
Sleep 3s

Down 8
Sleep 4s  # Longer pause for the final message

# ==============================================================================
# CLOSING: Reinforce the core message
# ==============================================================================

Type "# Don't import it. Build it. ✨🔥"
Sleep 3s
