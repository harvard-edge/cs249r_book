# ==============================================================================
# 06_tito_companion.tape: the tito CLI and its health check
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-06-tito-companion.gif"

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

Type "tito"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 4s
Type "clear"
Sleep 200ms
Enter
Sleep 300ms
Type "tito system health"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 7s
