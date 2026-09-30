# ==============================================================================
# 05_mlperf_opt.tape: MLPerf Hardware Optimization Olympics & Pareto Frontier
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-05-mlperf-opt.gif"

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

# Run Milestone 06: MLPerf Optimization Olympics
Type "tito milestone run 06"
Sleep 300ms
Enter
# Wait for the real run to finish (the shell prompt returns), however long
# training takes on this machine, then hold on the result.
Wait@600s /^\$ *$/
Sleep 6s
