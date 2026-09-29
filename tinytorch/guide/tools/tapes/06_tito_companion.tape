# ==============================================================================
# 06_tito_companion.tape: Tito CLI Companion & System Health Diagnostics
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-06-tito-companion.gif"

Set Shell "bash"
Set FontSize 14
Set FontFamily "Menlo"
Set Width 1200
Set Height 860
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set CursorBlink false

# Prepare environment invisibly
Hide
Type "source ./tinytorch/guide/tools/tape_env.sh && clear"
Enter
Sleep 500ms
Show

# Step 1: Launch Tito companion CLI
Type "tito"
Sleep 300ms
Enter
Sleep 3500ms

# Step 2: Clear screen for environment diagnostics
Type "clear"
Sleep 200ms
Enter
Sleep 300ms

# Step 3: Run comprehensive system health check
Type "tito system health"
Sleep 300ms
Enter
Sleep 7000ms
