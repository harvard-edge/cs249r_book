import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
SERVE_DIRECTORY = "test_website/mlsys_book_removed_most"

class COOPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, format, *args):
        # Suppress the default logging to keep the output clean
        return

if __name__ == "__main__":
    project_root = Path.cwd()
    serve_path = project_root / SERVE_DIRECTORY

    if not (project_root / "vite.config.mjs").exists():
        print(f"Error: This script must be run from the project root directory.")
        print(f"Current directory: {project_root}")
        exit(1)

    if not serve_path.is_dir():
        print(f"Error: The directory to serve does not exist: {serve_path}")
        exit(1)

    # Change the current working directory to the one we want to serve
    os.chdir(serve_path)

    with socketserver.TCPServer(("", PORT), COOPHandler) as httpd:
        print(f"Serving from directory: {serve_path.resolve()}")
        print(f"Server running at http://localhost:{PORT}")
        print("Required headers (Cross-Origin-Opener-Policy, etc.) are being served.")
        httpd.serve_forever()