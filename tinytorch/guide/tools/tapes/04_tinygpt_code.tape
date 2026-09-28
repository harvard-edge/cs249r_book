# ==============================================================================
# 04_tinygpt_code.tape: TinyGPT Code Completion on TinyPy with AST Validation
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-04-tinygpt-code.gif"

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

# Train TinyGPT on Python code and evaluate syntax with AST compiler
Type "python3 tinytorch/milestones/05_2017_transformer/03_tinygpt_code_completion.py --quick"
Sleep 300ms
Enter
# Generous hold time to study code completions, compiler diagnostics, and 100% AST validity scorecard
Sleep 19000ms
