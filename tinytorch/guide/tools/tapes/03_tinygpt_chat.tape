# ==============================================================================
# 03_tinygpt_chat.tape: TinyGPT Conversational Chat & Overfitting Detective
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-03-tinygpt-chat.gif"

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

# Run Milestone 05 conversational assistant with live token streaming
Type "tito milestone run 05"
Sleep 300ms
Enter
# Generous hold time to study loss drop, Overfitting Detective gap, live streaming, and dialogue
Sleep 13000ms
