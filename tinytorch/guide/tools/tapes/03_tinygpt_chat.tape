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
Type "tito milestone run 05 --part 4"
Sleep 300ms
Enter
# Wait for the real run to finish (the shell prompt returns), however long
# training takes on this machine, then hold on the result.
Wait@600s /^\$ *$/
Sleep 6s
