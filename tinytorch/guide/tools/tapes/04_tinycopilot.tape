# ==============================================================================
# 04_tinycopilot.tape: TinyCopilot Code Generation on TinyPy with AST Validation
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-04-tinycopilot.gif"

Set Shell "bash"
Set FontSize 15
Set FontFamily "Menlo"
Set Width 1200
Set Height 820
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# Prepare environment invisibly
Hide
Type "source ./tinytorch/guide/tools/tape_env.sh && clear"
Enter
Sleep 500ms
Show

# Train TinyCopilot on Python code and evaluate syntax with AST compiler
Type "tito milestone run 05 --part 3"
Sleep 300ms
Enter
# Wait for the real run to finish (the shell prompt returns), however long
# training takes on this machine, then hold on the result.
Wait@600s /^\$ *$/
Sleep 6s
