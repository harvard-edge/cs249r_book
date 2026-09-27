# ==============================================================================
# 03_tinygpt_chat.tape: TinyGPT Conversational Chat & Overfitting Detective
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-03-tinygpt-chat.gif"

Set Shell "bash"
Set FontSize 13
Set FontFamily "Menlo"
Set Width 900
Set Height 600
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

# Prepare environment invisibly
Hide
Type "source ./tinytorch/guide/tools/tape_env.sh && clear"
Enter
Sleep 500ms
Show

# Run Milestone 05 conversational assistant
Type "tito milestone run 05"
Sleep 300ms
Enter
# Generous hold time to study loss drop, Overfitting Detective gap, and generated Q&A dialogue
Sleep 5500ms
