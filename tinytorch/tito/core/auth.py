"""Simple secure JSON credentials storage system for TinyTorch CLI."""
from __future__ import annotations

import http.server
import threading
import json
import os
import ssl
import time
import socket
import webbrowser
from pathlib import Path
from typing import Optional, Dict
from urllib.parse import urlparse, parse_qs

import certifi

# --- Configuration Constants ---
API_BASE_URL = "https://tinytorch.netlify.app"

# API Endpoints
ENDPOINTS = {
    "login": f"{API_BASE_URL}/api/auth/login",
    "leaderboard": f"{API_BASE_URL}/api/leaderboard",
    "submissions": f"{API_BASE_URL}/api/submissions",
    "cli_login": f"{API_BASE_URL}/cli-login",
}

# Defaults
LOCAL_SERVER_HOST = "0.0.0.0"  # Listen on all interfaces for WSL compatibility
AUTH_START_PORT = 54321
AUTH_PORT_HUNT_RANGE = 100
AUTH_CALLBACK_PATH = "/callback"
CREDENTIALS_FILE_NAME = "credentials.json"

# Determine credentials directory (Standard Python way)
CREDENTIALS_DIR = os.getenv("TINYTORCH_CREDENTIALS_DIR", str(Path.home() / ".tinytorch"))


# --- Storage Logic ---

def _credentials_dir() -> Path:
    return Path(os.path.expanduser(CREDENTIALS_DIR))

def _credentials_path() -> Path:
    return _credentials_dir() / CREDENTIALS_FILE_NAME

def _ensure_dir() -> None:
    d = _credentials_dir()
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass

def save_credentials(data: Dict[str, str]) -> None:
    """Persist credentials to disk safely and atomically."""
    from tito.core.console import get_console
    _ensure_dir()
    p = _credentials_path()
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass



def load_credentials() -> Optional[Dict[str, str]]:
    p = _credentials_path()
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

def delete_credentials() -> None:
    p = _credentials_path()
    try:
        p.unlink()
    except OSError:
        pass

# --- Public Auth Helpers ---

def get_token() -> Optional[str]:
    """Retrieve the access token if it exists."""
    creds = load_credentials()
    if creds:
        return creds.get("access_token")
    return None

def is_logged_in() -> bool:
    """Check if the user has valid credentials stored."""
    return get_token() is not None

def get_user_email() -> Optional[str]:
    """Retrieve the user's email if it exists."""
    creds = load_credentials()
    if creds:
        return creds.get("user_email")
    return None


def get_refresh_token() -> Optional[str]:
    """Retrieve the refresh token if it exists."""
    creds = load_credentials()
    if creds:
        return creds.get("refresh_token")
    return None

def refresh_token(console: "Console") -> Optional[str]:
    """Refresh the access token. If refresh fails, clear credentials to force re-login."""
    refresh_token_val = get_refresh_token()
    if not refresh_token_val:
        return None

    import urllib.request
    import urllib.error
    import json

    url = f"{API_BASE_URL}/api/auth/refresh"
    data = {"refreshToken": refresh_token_val}
    headers = {"Content-Type": "application/json"}

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers=headers,
        method="POST"
    )

    # Create SSL context with certifi certificates for macOS compatibility
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    try:
        with urllib.request.urlopen(req, context=ssl_context) as response:
            if response.status == 200:
                new_session = json.loads(response.read().decode('utf-8'))

                # Handle nested session structures (adjust based on your actual API response)
                # Some APIs return { session: { ... } }, others return { access_token: ... } direct
                session_data = new_session.get('session', new_session)

                if 'access_token' in session_data:
                    new_access_token = session_data['access_token']
                    # IMPORTANT: Always grab the new refresh token if the server rotates it
                    new_refresh_token = session_data.get('refresh_token', refresh_token_val)

                    creds = load_credentials() or {}
                    creds.update({
                        "access_token": new_access_token,
                        "refresh_token": new_refresh_token,
                    })
                    save_credentials(creds)
                    return new_access_token
                else:
                    console.print("[red]Token refresh response is missing session data.[/red]")
                    return None
            else:
                console.print(f"[red]Token refresh failed with status: {response.status}[/red]")
                return None

    except urllib.error.HTTPError as e:
        # --- CRITICAL FIX HERE ---
        # If we get a 400 (Bad Request) or 401 (Unauthorized), the refresh token is dead.
        # We must delete the credentials so the user is forced to log in again.
        if e.code in [400, 401, 403]:
            console.print("[yellow]Session expired. Please log in again.[/yellow]")
            delete_credentials() # This deletes the JSON file
            return None

        console.print(f"[red]Token refresh failed (HTTP {e.code}): {e.reason}[/red]")
        try:
            error_body = e.read().decode('utf-8')
            error_json = json.loads(error_body)
            console.print(f"   [dim red]Error details: {error_json.get('error', 'No description provided.')}[/dim red]")
        except (json.JSONDecodeError, Exception):
            pass
        return None

    except urllib.error.URLError as e:
        console.print(f"[red]Token refresh failed (Network error): {e.reason}[/red]")
        return None

# --- Auth Server Logic ---

class CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/logout":
            self.server.logout_requested = True  # Signal that logout was hit
            self.send_response(302)
            self.send_header('Location', f"{API_BASE_URL}/cli/logged-out")
            self.end_headers()
            return

        if parsed_path.path != AUTH_CALLBACK_PATH:
            self.send_error(404, "Not Found")
            return

        query_params = parse_qs(parsed_path.query)

        if 'access_token' in query_params and 'refresh_token' in query_params:
            self.server.auth_data = {
                'access_token': query_params['access_token'][0],
                'refresh_token': query_params['refresh_token'][0],
                'user_email': query_params.get('email', [''])[0]
            }

            # Redirect to the branded "Logged In" page
            user_email = self.server.auth_data['user_email']
            redirect_url = f"{API_BASE_URL}/cli/logged-in?email={user_email}"

            self.send_response(302)
            self.send_header('Location', redirect_url)
            self.end_headers()

            # Persist immediately
            try:
                save_credentials(self.server.auth_data)
            except Exception:
                pass
        else:
            self.send_error(400, "Missing tokens in callback URL")

    def log_message(self, format, *args):
        pass

class LocalAuthServer(http.server.HTTPServer):
    def __init__(self, server_address, RequestHandlerClass):
        super().__init__(server_address, RequestHandlerClass)
        self.auth_data: Optional[Dict[str, str]] = None
        self.logout_requested: bool = False  # Initialize the logout flag

def _is_wsl() -> bool:
    """Check if running in WSL environment."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

def _get_callback_host() -> str:
    """Get the appropriate host for callback URL (handles WSL)."""
    if _is_wsl():
        import subprocess
        try:
            # Get WSL IP that Windows can reach
            result = subprocess.run(
                ['hostname', '-I'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                wsl_ip = result.stdout.strip().split()[0]
                if wsl_ip:
                    return wsl_ip
        except Exception:
            pass
    return "127.0.0.1"

class AuthReceiver:
    def __init__(self, start_port: int = None):
        self.start_port = start_port if start_port is not None else AUTH_START_PORT
        self.server: Optional[LocalAuthServer] = None
        self.thread: Optional[threading.Thread] = None
        self.port: int = 0
        self.callback_host: str = "127.0.0.1"

    def start(self) -> int:
        port = self.start_port
        max_port = self.start_port + AUTH_PORT_HUNT_RANGE

        while True:
            try:
                self.server = LocalAuthServer((LOCAL_SERVER_HOST, port), CallbackHandler)
                self.port = self.server.server_address[1]
                break
            except OSError:
                port += 1
                if port > max_port:
                    raise Exception("Could not find an open port for authentication.")

        def serve_with_error_handling():
            try:
                self.server.serve_forever()
            except Exception:
                pass

        self.thread = threading.Thread(target=serve_with_error_handling, daemon=True)
        self.thread.start()
        time.sleep(0.2)

        # Determine the callback host for URL construction
        self.callback_host = _get_callback_host()

        # Check if server is ready
        max_wait = 2.0
        waited = 0.0
        server_ready = False

        while waited < max_wait:
            try:
                if not self.thread.is_alive():
                    break

                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(0.2)
                # Test on localhost since we're checking from within WSL
                result = test_socket.connect_ex(("127.0.0.1", self.port))
                test_socket.close()
                if result == 0:
                    server_ready = True
                    break
            except Exception:
                pass
            time.sleep(0.1)
            waited += 0.1

        if not server_ready:
            self.stop()
            raise Exception(f"Server failed to start on port {self.port}")

        return self.port

    def get_redirect_url(self) -> str:
        """Get the full redirect URL that should work from the browser."""
        return f"http://{self.callback_host}:{self.port}{AUTH_CALLBACK_PATH}"

    def wait_for_tokens(self, timeout: int = 120) -> Optional[Dict[str, str]]:
        start_time = time.time()
        try:
            while getattr(self.server, "auth_data", None) is None:
                if time.time() - start_time > timeout:
                    return None
                time.sleep(0.25)

            try:
                save_credentials(self.server.auth_data)
            except Exception:
                pass

            time.sleep(1.0)

            return self.server.auth_data
        finally:
            self.stop()

    def wait_for_logout(self, timeout: int = 20) -> bool:
        """Wait for the /logout endpoint to be hit."""
        start_time = time.time()
        try:
            while not getattr(self.server, "logout_requested", False):
                if time.time() - start_time > timeout:
                    return False  # Timed out
                time.sleep(0.25)
            return True  # Logout signal received
        finally:
            self.stop()

    def stop(self):
        if self.server:
            try:
                # Just close the socket - the daemon thread will handle cleanup
                # Don't call shutdown() as it blocks waiting for serve_forever() to finish
                self.server.server_close()
            except Exception:
                pass
