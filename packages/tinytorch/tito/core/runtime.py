"""
Runtime environment detection for the TinyTorch CLI.

Single source of truth for two *separate* questions that the rest of the CLI
must never conflate:

1. ``is_ci()``          -- are we running inside automation (CI, a pipeline)?
2. ``is_interactive()`` -- can we safely prompt the user and read an answer?

Why these are separate
----------------------
A previous version gated progress *syncing* on ``sys.stdin.isatty()``. That
silently broke sync for real students on Windows Git Bash / MinTTY, where the
shell pipes stdin to ``python.exe`` and ``sys.stdin.isatty()`` returns ``False``
*even in an interactive terminal*. The same happens in some IDE-integrated
terminals on any OS.

The lesson encoded here: "can I prompt?" is NOT the same as "should I do the
network action?". Whether to *prompt* depends on having a usable TTY; whether to
*sync* depends only on being a real (non-CI) user who is logged in. Callers
decide whether to prompt with ``is_interactive()`` and decide whether to act
with ``is_ci()`` -- never the other way around.
"""

import os
import sys

# Environment variables set by common CI / automation providers.
_CI_ENV_VARS = (
    "CI",              # generic; set by GitHub Actions, GitLab, CircleCI, Travis, ...
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "JENKINS_URL",
    "BUILDKITE",
    "TF_BUILD",        # Azure Pipelines
    "TEAMCITY_VERSION",
)


def is_ci() -> bool:
    """Return ``True`` when running in a CI / automation environment.

    Used to suppress *all* interactive behavior and any "helpful" background
    network calls (like auto-syncing progress) that have no place in automation.
    """
    return any(os.environ.get(var) for var in _CI_ENV_VARS)


def is_interactive() -> bool:
    """Return ``True`` only when it is safe to prompt the user.

    Requires a real console on **both** stdin and stdout, and that we are not in
    CI. This is intentionally conservative: when it returns ``False`` the caller
    should proceed *without* a prompt (e.g. auto-confirm), not skip the action.

    See the module docstring for why this must never be used to decide whether
    to perform a network sync.
    """
    if is_ci():
        return False
    try:
        return bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError):
        # Streams can be replaced (e.g. by test harnesses) or closed.
        return False
