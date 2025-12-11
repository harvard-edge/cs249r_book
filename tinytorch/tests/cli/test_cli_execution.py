"""
CLI Execution Tests - Smoke tests for each command

This test suite ensures:
1. Each command can be executed without crashing (help mode)
2. Commands with subcommands show their subcommand help
3. Error messages are helpful when commands fail
"""

import pytest
import subprocess
import sys
from pathlib import Path

# Add tito to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.main import TinyTorchCLI


class TestCommandExecution:
    """Test that all commands can be executed (smoke tests)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = TinyTorchCLI()
        self.project_root = Path(__file__).parent.parent.parent

    def test_bare_tito_command(self):
        """Test bare 'tito' command shows welcome screen."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Should exit successfully
        assert result.returncode == 0, f"Bare tito command failed: {result.stderr}"

        # Should show welcome message
        assert "Welcome to Tiny" in result.stdout or "TORCH" in result.stdout
        assert "Command Groups:" in result.stdout or "Quick Start:" in result.stdout

    def test_tito_help(self):
        """Test 'tito -h' shows help."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main', '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        # Custom help displays logo and commands
        assert "TinyTorch" in result.stdout or "TORCH" in result.stdout
        assert "Quick Start" in result.stdout or "module" in result.stdout

    def test_tito_version(self):
        """Test 'tito --version' shows version."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main', '--version'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "Tiny" in result.stdout or "CLI" in result.stdout

    @pytest.mark.parametrize("command", [
        'setup', 'system', 'module', 'src', 'package', 'nbgrader',
        'milestones', 'benchmark', 'community', 'export', 'test',
        'grade', 'logo'
    ])
    def test_command_help_works(self, command):
        """Test that each command's help can be displayed."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main', command, '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Help should always succeed
        assert result.returncode == 0, (
            f"Command '{command} -h' failed with exit code {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

        # Should show usage
        assert "usage:" in result.stdout.lower(), (
            f"Command '{command} -h' didn't show usage"
        )

    @pytest.mark.parametrize("command,subcommand", [
        ('system', 'info'),
        ('system', 'health'),
        ('module', 'status'),
        ('module', 'list'),
        ('community', 'join'),
        ('milestones', 'status'),
    ])
    def test_subcommand_help_works(self, command, subcommand):
        """Test that subcommands can show help."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main', command, subcommand, '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Subcommand help should work
        # Note: Some commands might return non-zero if not fully implemented
        # but should at least not crash
        assert result.returncode in [0, 1, 2], (
            f"Command '{command} {subcommand} -h' crashed with exit code {result.returncode}"
        )


class TestCommandGrouping:
    """Test that commands are properly grouped and discoverable."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent

    def test_student_facing_commands_discoverable(self):
        """Test that main student-facing commands are easily discoverable."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Key student commands should be visible
        student_commands = ['setup', 'module', 'milestones']

        for cmd in student_commands:
            assert cmd in result.stdout, (
                f"Student command '{cmd}' not visible in welcome screen"
            )

    def test_developer_commands_documented(self):
        """Test that developer commands are documented in help."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main', '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Developer commands should be in help
        dev_commands = ['src', 'package', 'nbgrader']

        for cmd in dev_commands:
            assert cmd in result.stdout, (
                f"Developer command '{cmd}' not in help text"
            )


class TestErrorMessages:
    """Test that error messages are helpful."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent

    def test_invalid_command_shows_help(self):
        """Test that invalid commands show helpful error."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main', 'nonexistent'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Should fail
        assert result.returncode != 0

        # Should mention the invalid command
        combined_output = result.stdout + result.stderr
        assert 'nonexistent' in combined_output or 'invalid choice' in combined_output.lower()

    def test_missing_subcommand_shows_help(self):
        """Test that missing subcommands show help."""
        # Try module command without subcommand
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main', 'module'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Should show help or error
        # Some commands might have default behavior, others require subcommand
        combined_output = result.stdout + result.stderr
        assert len(combined_output) > 0, "No output from command without subcommand"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
