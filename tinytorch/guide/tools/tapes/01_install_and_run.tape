# ==============================================================================
# 01_install_and_run.tape: one-line install, then a student's first module, all real
# ==============================================================================

Output "tinytorch/guide/assets/images/demos/tinytorch-01-install-and-run.gif"

Set Shell "bash"
Set FontSize 15
Set FontFamily "Menlo"
Set Width 1200
Set Height 820
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# Fresh shell in an empty directory: system PATH, no inherited venv or PYTHONPATH
Hide
Type 'source ./tinytorch/guide/tools/tape_env.sh && export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin && unset PYTHONPATH VIRTUAL_ENV && cd "$(mktemp -d)" && clear'
Enter
Sleep 800ms
Show

# The real installer: clone, virtual environment, dependencies, tito check
Type "curl -fsSL https://mlsysbook.ai/tinytorch/install.sh | bash"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 3s
Type "cd tinytorch && source .venv/bin/activate"
Sleep 300ms
Enter
Sleep 1s
Type "tito setup"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 3s
Type "clear"
Sleep 200ms
Enter
Sleep 300ms
# A student's first module: create the notebook, then test and export it
Type "tito module start 01 --no-jupyter"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 3s
Type "clear"
Sleep 200ms
Enter
Sleep 300ms
Type "tito module complete 01"
Sleep 300ms
Enter
Wait@900s /^\$ *$/
Sleep 6s
