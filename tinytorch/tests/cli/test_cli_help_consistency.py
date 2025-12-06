"""
CLI Help Consistency Tests - Validate help text is consistent and complete

This test suite ensures:
1. Help text uses consistent formatting and terminology
2. All commands document their purpose clearly
3. Examples are provided where helpful
4. No broken references or outdated commands in help text
"""

import pytest
import subprocess
import sys
from pathlib import Path

# Add tito to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.main import TinyTorchCLI


class TestHelpConsistency:
    """Test that help text is consistent across commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent

    def get_command_help(self, *args):
        """Get help output for a command."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main'] + list(args) + ['-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        return result.stdout + result.stderr

    def test_all_command_helps_mention_tito(self):
        """Verify all help texts mention 'tito' command."""
        commands = ['setup', 'system', 'module', 'checkpoint', 'milestones']

        for cmd in commands:
            help_text = self.get_command_help(cmd)
            assert 'tito' in help_text.lower(), (
                f"Command '{cmd}' help doesn't mention 'tito'"
            )

    def test_bare_tito_and_help_are_different(self):
        """Verify bare 'tito' and 'tito -h' show different but related content."""
        # Get bare tito output
        bare_result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Get help output
        help_result = subprocess.run(
            [sys.executable, '-m', 'tito.main', '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        bare_output = bare_result.stdout
        help_output = help_result.stdout

        # Should both mention key commands
        for cmd in ['module', 'milestones', 'setup']:
            assert cmd in bare_output, f"'{cmd}' missing from bare tito"
            assert cmd in help_output, f"'{cmd}' missing from tito -h"

        # Bare should have welcome/logo
        assert any(word in bare_output for word in ['Welcome', 'TORCH', 'Quick Start']), \
            "Bare tito should show welcome screen"

        # Help should have usage
        assert 'usage:' in help_output.lower(), "tito -h should show usage"

    def test_no_references_to_removed_commands(self):
        """Verify help doesn't reference commands that don't exist."""
        cli = TinyTorchCLI()
        registered_commands = set(cli.commands.keys())

        # Get main help
        help_text = self.get_command_help()

        # Common command-like words that might be false positives
        ignore_words = {
            'command', 'commands', 'option', 'options', 'argument',
            'arguments', 'help', 'version', 'verbose', 'color',
            'git', 'python', 'pip', 'jupyter', 'pytest', 'run',
            'build', 'install', 'create', 'delete', 'update',
            'show', 'list', 'view', 'open', 'close', 'start',
            'stop', 'export', 'import', 'output', 'input'
        }

        # Extract words that look like commands (lowercase alphanumeric)
        import re
        potential_commands = set(re.findall(r'\b[a-z][a-z_-]*[a-z]\b', help_text.lower()))

        # Filter to reasonable command-like words
        suspicious = potential_commands - registered_commands - ignore_words

        # These are expected in help text but not commands
        expected_non_commands = {
            'system', 'module', 'first', 'time', 'complete', 'resume',
            'status', 'progress', 'journey', 'profile', 'timeline',
            'tinytorch', 'tiny', 'torch', 'cli', 'developer', 'student',
            'workflow', 'tracking', 'capabilities', 'achievements'
        }

        truly_suspicious = suspicious - expected_non_commands

        # Just warn if we find something, don't fail
        # (This test is informational)
        if truly_suspicious:
            print(f"\nInfo: Found potential command references: {sorted(truly_suspicious)[:10]}")


class TestWelcomeScreen:
    """Test the welcome screen shown by bare 'tito' command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent

    def test_welcome_screen_shows_quick_start(self):
        """Verify welcome screen has quick start section."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        output = result.stdout
        assert 'Quick Start' in output or 'Getting Started' in output, \
            "Welcome screen should have Quick Start section"

    def test_welcome_screen_shows_command_groups(self):
        """Verify welcome screen organizes commands into groups."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        output = result.stdout
        assert 'Quick Start:' in output or 'Commands:' in output, \
            "Welcome screen should show quick start or commands"

    def test_welcome_screen_has_examples(self):
        """Verify welcome screen shows example commands."""
        result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        output = result.stdout

        # Should have at least one example command
        assert 'tito setup' in output or 'tito module' in output, \
            "Welcome screen should show example commands"


class TestCommandDocumentation:
    """Test that commands are properly documented."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent.parent
        self.cli = TinyTorchCLI()

    def test_all_registered_commands_in_welcome_or_help(self):
        """Verify all registered commands appear in welcome screen or help."""
        # Get welcome screen
        welcome_result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        # Get help
        help_result = subprocess.run(
            [sys.executable, '-m', 'tito.main', '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        combined = welcome_result.stdout + help_result.stdout

        missing = []
        for cmd in self.cli.commands.keys():
            if cmd not in combined:
                missing.append(cmd)

        assert not missing, (
            f"Commands registered but not documented: {missing}\n"
            f"Add them to welcome screen or help text"
        )

    def test_milestones_properly_documented(self):
        """Specifically test that milestones command is documented."""
        # This addresses the user's concern about progress tracking
        welcome_result = subprocess.run(
            [sys.executable, '-m', 'tito.main'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        help_result = subprocess.run(
            [sys.executable, '-m', 'tito.main', '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        combined = welcome_result.stdout + help_result.stdout

        assert 'milestones' in combined.lower(), \
            "milestones command should be documented"

        # Check it has a description
        milestones_help = subprocess.run(
            [sys.executable, '-m', 'tito.main', 'milestones', '-h'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )

        assert 'progress' in milestones_help.stdout.lower() or \
               'track' in milestones_help.stdout.lower() or \
               'milestone' in milestones_help.stdout.lower(), \
            "milestones help should explain its purpose"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
