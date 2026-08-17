"""
Malformed ~/.tinytorch/credentials.json tests for load_credentials()
and its callers (get_token, is_logged_in, get_user_email,
get_refresh_token).

Same bug class already found and fixed in tito/commands/milestone.py,
tito/commands/module/workflow.py, and tito/core/submission.py:
json.load() succeeds on syntactically valid JSON of the wrong shape,
and code that assumes a dict crashes. Here callers guard with
`if creds:` before calling .get(), but a non-empty wrong-type value
(a list, a non-empty string) is truthy, so that guard doesn't actually
protect against the crash.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tito.core.auth as auth_mod


def _patch_credentials_path(monkeypatch, tmp_path):
    creds_path = tmp_path / "credentials.json"
    monkeypatch.setattr(auth_mod, "_credentials_path", lambda: creds_path)
    return creds_path


class TestLoadCredentialsWrongType:
    def test_list_type_returns_none(self, tmp_path, monkeypatch):
        creds_path = _patch_credentials_path(monkeypatch, tmp_path)
        creds_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

        assert auth_mod.load_credentials() is None

    def test_nonempty_string_type_returns_none(self, tmp_path, monkeypatch):
        """A non-empty string is truthy, so `if creds:` alone would not
        have caught this before the isinstance check was added."""
        creds_path = _patch_credentials_path(monkeypatch, tmp_path)
        creds_path.write_text(json.dumps("corrupted"), encoding="utf-8")

        assert auth_mod.load_credentials() is None

    def test_valid_dict_still_returned(self, tmp_path, monkeypatch):
        creds_path = _patch_credentials_path(monkeypatch, tmp_path)
        creds_path.write_text(json.dumps({"access_token": "abc123"}), encoding="utf-8")

        assert auth_mod.load_credentials() == {"access_token": "abc123"}


class TestAuthCallersWithWrongTypeCredentials:
    def test_get_token_does_not_crash(self, tmp_path, monkeypatch):
        creds_path = _patch_credentials_path(monkeypatch, tmp_path)
        creds_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

        assert auth_mod.get_token() is None

    def test_is_logged_in_does_not_crash(self, tmp_path, monkeypatch):
        creds_path = _patch_credentials_path(monkeypatch, tmp_path)
        creds_path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")

        assert auth_mod.is_logged_in() is False

    def test_get_user_email_does_not_crash(self, tmp_path, monkeypatch):
        creds_path = _patch_credentials_path(monkeypatch, tmp_path)
        creds_path.write_text(json.dumps("corrupted"), encoding="utf-8")

        assert auth_mod.get_user_email() is None

    def test_get_refresh_token_does_not_crash(self, tmp_path, monkeypatch):
        creds_path = _patch_credentials_path(monkeypatch, tmp_path)
        creds_path.write_text(json.dumps(42), encoding="utf-8")

        assert auth_mod.get_refresh_token() is None
