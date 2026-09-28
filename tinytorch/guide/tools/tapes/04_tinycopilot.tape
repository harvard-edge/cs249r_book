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
Type "tito milestone run tinycopilot"
Sleep 300ms
Enter
# Generous hold time to study code completions, compiler diagnostics, and 100% AST validity scorecard
Sleep 19000ms
