"""
Live E2E integration test for TinyTorch progress submission to Supabase.

This test runs ONLY if TINYTORCH_TEST_EMAIL and TINYTORCH_TEST_PASSWORD are set
in the environment or in a local.env file.

It automates the entire flow:
1. Programmatic login to Supabase auth API using email/password + active Anon Key.
2. Temporary credentials file setup.
3. Seeding dummy progress data.
4. Sending the sync payload to the Supabase Edge Function.
5. Verifying a successful response from the server.
"""

import os
import json
import urllib.request
import urllib.error
import pytest
from pathlib import Path
import sys

# Ensure tito is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.core.config import CLIConfig
from tito.core.submission import SubmissionHandler
from tito.core import auth

# Helper to load local.env if present
def load_local_env():
    env_path = Path(__file__).parent.parent.parent / "local.env"
    if env_path.exists():
        for line in env_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            os.environ[key.strip()] = val.strip()

load_local_env()

TEST_EMAIL = os.environ.get("TINYTORCH_TEST_EMAIL")
TEST_PASSWORD = os.environ.get("TINYTORCH_TEST_PASSWORD")

# Dynamic retrieval of Supabase Anon Key and Project URL from config.js
def get_supabase_keys():
    config_js = Path(__file__).parent.parent.parent / "quarto" / "community" / "modules" / "config.js"
    if not config_js.exists():
        return None, None
    
    content = config_js.read_text(encoding='utf-8')
    import re
    url_match = re.search(r'SUPABASE_PROJECT_URL\s*=\s*["\']([^"\']+)["\']', content)
    key_match = re.search(r'SUPABASE_ANON_KEY\s*=\s*["\']([^"\']+)["\']', content)
    
    return (
        url_match.group(1) if url_match else None,
        key_match.group(1) if key_match else None
    )

@pytest.mark.skipif(not TEST_EMAIL or not TEST_PASSWORD, reason="Requires TINYTORCH_TEST_EMAIL and TINYTORCH_TEST_PASSWORD in local.env or environment")
class TestLiveCommunitySync:
    """Live E2E test verifying progress syncing with Supabase Auth & Edge functions."""

    @pytest.fixture(autouse=True)
    def setup_sandbox_paths(self, tmp_path):
        """Creates clean .tito folder and backs up / restores existing credentials."""
        # Seeding mock data in clean directory
        self.tito_dir = tmp_path / ".tito"
        self.tito_dir.mkdir(parents=True, exist_ok=True)
        
        self.progress_file = self.tito_dir / "progress.json"
        self.progress_file.write_text(json.dumps({
            "completed_modules": ["01"],
            "last_worked": "01",
            "last_completed": "01",
            "last_updated": "2026-06-16T22:32:26"
        }))
        
        self.milestones_file = self.tito_dir / "milestones.json"
        self.milestones_file.write_text(json.dumps({
            "unlocked_milestones": [],
            "completed_milestones": [],
            "unlock_dates": {},
            "completion_dates": {},
            "total_unlocked": 0
        }))
        
        self.config = CLIConfig(
            project_root=tmp_path,
            assignments_dir=tmp_path / "assignments",
            modules_dir=tmp_path / "src",
            tinytorch_dir=tmp_path / "tinytorch",
            bin_dir=tmp_path / "bin"
        )
        
        # Override credentials storage directory to temporary path
        self.orig_creds_dir = auth.CREDENTIALS_DIR
        auth.CREDENTIALS_DIR = str(tmp_path / "creds")
        
        yield
        
        # Restore credentials path
        auth.CREDENTIALS_DIR = self.orig_creds_dir

    def test_live_auth_and_sync(self):
        """Performs programmatic login, writes credentials, and syncs progress to Supabase."""
        project_url, anon_key = get_supabase_keys()
        assert project_url and anon_key, "Could not extract Supabase Project URL or Anon Key from config.js"
        
        # Step 1: Login programmatically using Supabase password grant type
        auth_url = f"{project_url}/auth/v1/token?grant_type=password"
        headers = {
            "apikey": anon_key,
            "Content-Type": "application/json"
        }
        body = {
            "email": TEST_EMAIL,
            "password": TEST_PASSWORD
        }
        
        req = urllib.request.Request(
            auth_url,
            data=json.dumps(body).encode('utf-8'),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            pytest.fail(f"Supabase Authentication failed (HTTP {e.code}): {e.read().decode('utf-8')}")
        except Exception as e:
            pytest.fail(f"Failed to reach Supabase Auth API: {e}")
            
        assert "access_token" in data, "No access_token returned from Supabase Auth"
        
        # Step 2: Store credentials locally in the temp credentials path
        auth.save_credentials({
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "user_email": data["user"]["email"]
        })
        
        assert auth.is_logged_in() is True
        assert auth.get_user_email() == TEST_EMAIL
        
        # Step 3: Run the submission handler to sync progress
        from rich.console import Console
        console = Console(record=True)
        handler = SubmissionHandler(self.config, console)
        
        sync_success = handler.sync_progress(total_modules=20)

        # Assert syncing returns success from the Edge Function (SyncResult, #1849)
        assert sync_success.ok is True
        console_output = console.export_text()
        assert "Sync Successful!" in console_output

        # Step 4: Verify persistence via /get-profile-details
        details_url = f"{project_url}/functions/v1/get-profile-details"
        details_req = urllib.request.Request(
            details_url,
            headers={
                "Authorization": f"Bearer {data['access_token']}",
                "Content-Type": "application/json"
            },
            method="GET"
        )
        try:
            with urllib.request.urlopen(details_req, timeout=10) as details_response:
                details_data = json.loads(details_response.read().decode('utf-8'))
        except Exception as e:
            pytest.fail(f"Failed to fetch profile details from get-profile-details: {e}")

        completed_modules = details_data.get("completed_modules", [])
        assert (1 in completed_modules or "01" in completed_modules), (
            "Backend Sync Bug: The module progress was uploaded successfully (Edge Function returned 200), "
            "but did not persist in the database's completed_modules list. "
            "This indicates that user_stats row is missing or not updated. "
            f"Returned completed_modules: {completed_modules}"
        )
