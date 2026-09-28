# ==============================================================================
# 07_tito_olympics.tape: TinyTorch Olympics & Competition Events
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-07-tito-olympics.gif"

Set Shell "bash"
Set FontSize 15
Set FontFamily "Menlo"
Set Width 1200
Set Height 820
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms
Set CursorBlink false

# Prepare environment invisibly
Hide
Type "source ./tinytorch/guide/tools/tape_env.sh && clear"
Enter
Sleep 500ms
Show

# Run Tito Olympics competition board
Type "tito olympics"
Sleep 300ms
Enter
Sleep 6000ms
