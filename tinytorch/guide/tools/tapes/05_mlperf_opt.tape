# ==============================================================================
# 05_mlperf_opt.tape: MLPerf Hardware Optimization Olympics & Pareto Frontier
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-05-mlperf-opt.gif"

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

# Run Milestone 06: MLPerf Optimization Olympics
Type "tito milestone run 06"
Sleep 300ms
Enter
# Generous hold time to study 3 divisions (INT8, im2col, KV cache) and Pareto frontier
Sleep 5500ms
