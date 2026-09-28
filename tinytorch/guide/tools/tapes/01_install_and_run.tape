# ==============================================================================
# 01_install_and_run.tape: One-Line Install to Running Milestone 01 Out of the Box
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-01-install-and-run.gif"

Set Shell "bash"
Set FontSize 12
Set FontFamily "Menlo"
Set Width 880
Set Height 680
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# Prepare environment invisibly
Hide
Type "source ./tinytorch/guide/tools/tape_env.sh && clear"
Enter
Sleep 500ms
Show

# Step 1: One-line install command
Type "curl -fsSL https://mlsysbook.ai/tinytorch/install.sh | bash"
Sleep 300ms
Enter
Sleep 2500ms

# Step 2: First-time setup
Type "tito setup"
Sleep 300ms
Enter
Sleep 2500ms

# Step 3: Run Milestone 01 out of the box
Type "clear"
Sleep 200ms
Enter
Sleep 300ms
Type "tito milestone run 01"
Sleep 300ms
Enter
# Generous hold time for viewer to read parameters, decision plot, and achievement banner
Sleep 6000ms
