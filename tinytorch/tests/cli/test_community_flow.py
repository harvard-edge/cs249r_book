"""
Tests for the TinyTorch community submission and progress syncing flow.

These tests validate:
1. Retrieval of authentication status and tokens.
2. Creation and structure of the progress payload.
3. Execution of progress syncing POST requests with mock servers.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Ensure tito is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.core.config import CLIConfig
from tito.core.submission import SubmissionHandler


class TestCommunitySyncFlow:
    """Test suite for validating submission and syncing features."""

    @pytest.fixture
    def mock_env_paths(self, tmp_path):
        """Creates a temporary .tito directory with progress and milestone mock data."""
        tito_dir = tmp_path / ".tito"
        tito_dir.mkdir(parents=True, exist_ok=True)

        progress_file = tito_dir / "progress.json"
        progress_data = {
            "started_modules": [],
            "completed_modules": ["01", "02"],
            "last_worked": "02",
            "last_completed": "02",
            "last_updated": "2026-06-16T22:32:26"
        }
        progress_file.write_text(json.dumps(progress_data))

        milestones_file = tito_dir / "milestones.json"
        milestone_data = {
            "unlocked_milestones": ["01"],
            "completed_milestones": ["01"],
            "unlock_dates": {"01": "2026-06-16T22:32:26"},
            "completion_dates": {"01": "2026-06-16T22:32:26"},
            "total_unlocked": 1
        }
        milestones_file.write_text(json.dumps(milestone_data))

        config = CLIConfig(
            project_root=tmp_path,
            assignments_dir=tmp_path / "assignments",
            modules_dir=tmp_path / "src",
            tinytorch_dir=tmp_path / "tinytorch",
            bin_dir=tmp_path / "bin"
        )
        return config, progress_file, milestones_file

    def test_assemble_payload(self, mock_env_paths):
        """Verify the JSON payload matches the Supabase Edge Function schema."""
        config, _, _ = mock_env_paths
        console = MagicMock()
        handler = SubmissionHandler(config, console)

        with patch("tito.core.auth.get_user_email", return_value="test_student@example.com"):
            payload = handler.assemble_payload(total_modules=20)

        # Assert correct schema mapping
        assert payload["user_id"] == "test_student@example.com"
        assert payload["version"] == "1.0"
        assert payload["module_progress"]["completed_count"] == 2
        assert payload["module_progress"]["completed_modules"] == ["01", "02"]
        assert payload["module_progress"]["completion_percentage"] == 10.0
        assert payload["milestone_progress"]["unlocked_count"] == 1
        
        milestones = payload["milestone_progress"]["unlocked_milestones"]
        assert len(milestones) == 1
        assert milestones[0]["id"] == "01"
        assert milestones[0]["completed"] is True

    @patch("urllib.request.urlopen")
    @patch("tito.core.auth.get_token")
    @patch("tito.core.auth.get_user_email")
    def test_sync_progress_success(self, mock_email, mock_token, mock_urlopen, mock_env_paths):
        """Test successful progress synchronization (HTTP 200)."""
        config, _, _ = mock_env_paths
        mock_email.return_value = "test_student@example.com"
        mock_token.return_value = "mock_jwt_access_token"

        # Mock urllib response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"synced_modules": 2}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        console = MagicMock()
        handler = SubmissionHandler(config, console)
        success = handler.sync_progress(total_modules=20)

        # sync_progress returns a SyncResult (see #1849); check the .ok flag.
        assert success.ok is True
        mock_urlopen.assert_called_once()
        
        # Verify request contents
        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.method == "POST"
        assert request_obj.headers["Authorization"] == "Bearer mock_jwt_access_token"
        assert request_obj.headers["Content-type"] == "application/json"
        
        # Verify printed messages
        console.print.assert_any_call("✅ [bold green]Sync successful![/bold green]")

    @patch("urllib.request.urlopen")
    @patch("tito.core.auth.get_token")
    @patch("tito.core.auth.get_user_email")
    def test_sync_progress_unauthorized_token_refresh(self, mock_email, mock_token, mock_urlopen, mock_env_paths):
        """Verify that an expired token triggers token refresh logic."""
        config, _, _ = mock_env_paths
        mock_email.return_value = "test_student@example.com"
        mock_token.return_value = "expired_token"

        import urllib.error
        # Raise HTTP 401 Unauthorized for the first call
        http_error = urllib.error.HTTPError(
            url="http://mock-url",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=MagicMock()
        )
        http_error.read = MagicMock(return_value=b'{"error": "Invalid token"}')

        # Second call response (after refresh token)
        retry_response = MagicMock()
        retry_response.__enter__.return_value.status = 200
        retry_response.__enter__.return_value.read.return_value = b'{"synced_modules": 2}'

        mock_urlopen.side_effect = [http_error, retry_response]

        console = MagicMock()
        handler = SubmissionHandler(config, console)

        with patch("tito.core.auth.refresh_token", return_value="new_fresh_token") as mock_refresh:
            handler.sync_progress(total_modules=20)
            mock_refresh.assert_called_once_with(console)
