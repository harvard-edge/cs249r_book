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
export VIRTUAL_ENV="$REPO_DIR/.venv"
# Put the project venv first so `tito` and `python3` are the real ones.
for _venv in "$REPO_DIR/.venv" "$REPO_DIR/tinytorch/.venv"; do
  if [[ -x "$_venv/bin/python3" ]]; then export PATH="$_venv/bin:$PATH"; export VIRTUAL_ENV="$_venv"; break; fi
done

# Students run tito inside their tinytorch/ checkout.
cd "$REPO_DIR/tinytorch" || exit 1

# Keep recordings away from the user's real Jupyter kernels: `tito setup`
# registers a kernel named "tinytorch", and with a user-level data dir that
# would repoint the recorder's own kernel at a throwaway venv (it did once,
# 2026-09-29).
export JUPYTER_DATA_DIR="$REPO_DIR/.tt_demo_jupyter"

# Everything a tape shows runs for real: no command on screen is rewritten.
#
# The one substitution: when TT_LOCAL_INSTALLER is set (recording before a
# release is published), `curl .../install.sh` returns this checkout's real
# install.sh (the same file the site serves once published) instead of the
# copy currently live on the site. With TT_INSTALL_REPO_URL and
# TT_INSTALL_BRANCH also set, that real installer clones this unpushed branch
# from a local repository. After publishing, record without these variables
# and curl fetches from the site like any student's does.
#
# 2026-09-29: this file used to pipe a hand-written fake installer
# (demo_installer.sh: echo lines and sleeps, no clone, no venv, no install),
# rewrite `tito setup` to `tito setup --skip-venv --skip-packages`, and swap
# `tito milestone run` for direct script calls with tuned flags. All removed.
if [[ -n "${TT_LOCAL_INSTALLER:-}" ]]; then
  curl() {
    if [[ "$*" == *"mlsysbook.ai/tinytorch/install.sh"* ]]; then
      cat "$REPO_DIR/tinytorch/guide/install.sh"
    else
      command curl "$@"
    fi
  }
  [[ -n "${TT_INSTALL_REPO_URL:-}" ]] && export TINYTORCH_REPO_URL="$TT_INSTALL_REPO_URL"
  [[ -n "${TT_INSTALL_BRANCH:-}" ]] && export TINYTORCH_BRANCH="$TT_INSTALL_BRANCH"
fi

# Keep the prompt "$ " after `source .venv/bin/activate`, so tapes can wait
# for it.
export VIRTUAL_ENV_DISABLE_PROMPT=1
