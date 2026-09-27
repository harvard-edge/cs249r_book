#!/bin/bash
# Environment initialization for TinyTorch VHS tape recordings
# Sourced invisibly in the Hide block of each .tape file

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"

export PS1='$ '
export TERM=xterm-256color
export COLORTERM=truecolor
export PATH="$REPO_DIR/tinytorch/bin:$PATH"
export PYTHONPATH="$REPO_DIR/tinytorch"
export TITO_ALLOW_SYSTEM=1
export TINYTORCH_NON_INTERACTIVE=1
export TITO_NON_INTERACTIVE=1

cd "$REPO_DIR" || exit 1

# Mock/wrap installer for offline reproducibility and instant responsiveness
curl() {
  if [[ "$*" == *"install.sh"* ]]; then
    cat "$REPO_DIR/tinytorch/guide/tools/demo_installer.sh"
  else
    command curl "$@"
  fi
}

# Wrap python3 to fast-path code completion training to 6 epochs for showcase recording
_real_python3=$(which python3)
python3() {
  if [[ "$*" == *"03_tinygpt_code_completion.py --quick"* && "$*" != *"--epochs"* ]]; then
    "$_real_python3" "$REPO_DIR/tinytorch/milestones/05_2017_transformer/03_tinygpt_code_completion.py" --quick --epochs 6
  else
    "$_real_python3" "$@"
  fi
}

# Wrap tito CLI to streamline showcase demonstrations
_real_tito="$REPO_DIR/tinytorch/bin/tito"
tito() {
  if [[ "$1" == "setup" ]]; then
    python3 "$_real_tito" setup --skip-venv --skip-packages
  elif [[ "$1" == "module" && "$2" == "complete" ]]; then
    python3 "$_real_tito" module complete "$3"
  elif [[ "$1" == "module" && "$2" == "status" ]]; then
    python3 "$_real_tito" module status
  elif [[ "$1" == "milestone" && "$2" == "run" && "$3" == "01" ]]; then
    python3 "$_real_tito" milestone run 01
  elif [[ "$1" == "milestone" && "$2" == "run" && "$3" == "05" ]]; then
    python3 "$REPO_DIR/tinytorch/milestones/05_2017_transformer/04_tinygpt_chat.py" --quick --epochs 6 --prompt "User: What is autograd?" --stream
  elif [[ "$1" == "milestone" && "$2" == "run" && "$3" == "06" ]]; then
    python3 "$REPO_DIR/tinytorch/milestones/06_2018_mlperf/01_optimization_olympics.py"
  else
    python3 "$_real_tito" "$@"
  fi
}
