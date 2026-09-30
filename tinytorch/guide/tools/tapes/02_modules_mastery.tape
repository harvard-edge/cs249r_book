# ==============================================================================
# 02_modules_mastery.tape: complete a module, then the student's real progress
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-02-modules-mastery.gif"

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

Type "tito module complete 06"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 3s
Type "clear"
Sleep 200ms
Enter
Sleep 300ms
Type "tito module status"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 7s
