# ==============================================================================
# 07_tito_olympics.tape: the Olympics command as it stands today
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-07-tito-olympics.gif"

Set Shell "bash"
Set FontSize 15
Set FontFamily "Menlo"
Set Width 1200
Set Height 820
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# Prepare environment invisibly
Hide
Type 'source ./tinytorch/guide/tools/tape_env.sh && clear'
Enter
Sleep 800ms
Show

Type "tito olympics"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 6s
