"""
CLI Registry Tests - Validate all commands are properly registered and accessible

This test suite ensures:
1. All commands in TinyTorchCLI.commands are valid BaseCommand subclasses
2. All commands have proper metadata (name, description)
3. All commands can be invoked via argparse
4. No commands are missing from registration
5. No orphaned command files exist without registration
"""

import pytest
import argparse
from pathlib import Path
import sys

# Add tito to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tito.main import TinyTorchCLI
from tito.commands.base import BaseCommand
from tito.core.config import CLIConfig


class TestCLIRegistry:
    """Test that all commands are properly registered in the CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = TinyTorchCLI()
        self.config = CLIConfig.from_project_root()

    def test_all_commands_are_base_command_subclasses(self):
        """Verify all registered commands inherit from BaseCommand."""
        for cmd_name, cmd_class in self.cli.commands.items():
            assert issubclass(cmd_class, BaseCommand), (
                f"Command '{cmd_name}' ({cmd_class.__name__}) must inherit from BaseCommand"
            )

    def test_all_commands_have_description(self):
        """Verify all commands have a description."""
        for cmd_name, cmd_class in self.cli.commands.items():
            cmd_instance = cmd_class(self.config)
            assert hasattr(cmd_instance, 'description'), (
                f"Command '{cmd_name}' must have a 'description' attribute"
            )
            assert cmd_instance.description, (
                f"Command '{cmd_name}' has empty description"
            )
            assert len(cmd_instance.description) > 10, (
                f"Command '{cmd_name}' description too short: '{cmd_instance.description}'"
            )

    def test_all_commands_implement_execute(self):
        """Verify all commands implement execute() method."""
        for cmd_name, cmd_class in self.cli.commands.items():
            cmd_instance = cmd_class(self.config)
            assert hasattr(cmd_instance, 'execute'), (
                f"Command '{cmd_name}' must implement execute() method"
            )
            assert callable(cmd_instance.execute), (
                f"Command '{cmd_name}' execute must be callable"
            )

    def test_all_commands_implement_add_arguments(self):
        """Verify all commands implement add_arguments() method."""
        for cmd_name, cmd_class in self.cli.commands.items():
            cmd_instance = cmd_class(self.config)
            assert hasattr(cmd_instance, 'add_arguments'), (
                f"Command '{cmd_name}' must implement add_arguments() method"
            )
            assert callable(cmd_instance.add_arguments), (
                f"Command '{cmd_name}' add_arguments must be callable"
            )

    def test_parser_creation_succeeds(self):
        """Verify the argument parser can be created without errors."""
        parser = self.cli.create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_all_commands_registered_in_parser(self):
        """Verify all registered commands appear in the parser."""
        parser = self.cli.create_parser()

        # Get all subparsers
        subparsers_actions = [
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]

        assert len(subparsers_actions) == 1, "Should have exactly one subparsers group"

        # Get registered command names from parser
        subparser_choices = subparsers_actions[0].choices.keys()

        # Verify all commands in self.cli.commands are in parser
        for cmd_name in self.cli.commands.keys():
            assert cmd_name in subparser_choices, (
                f"Command '{cmd_name}' registered in TinyTorchCLI.commands but not in parser"
            )

    def test_no_duplicate_command_names(self):
        """Verify no duplicate command names in registry."""
        cmd_names = list(self.cli.commands.keys())
        unique_names = set(cmd_names)
        assert len(cmd_names) == len(unique_names), (
            f"Duplicate command names found: {[n for n in cmd_names if cmd_names.count(n) > 1]}"
        )

    def test_command_help_text_accessible(self):
        """Verify all commands can generate help text without errors."""
        parser = self.cli.create_parser()

        for cmd_name in self.cli.commands.keys():
            # This should not raise any exceptions
            help_text = parser.format_help()
            assert cmd_name in help_text or cmd_name == 'src', (
                f"Command '{cmd_name}' not found in help text"
            )


class TestCommandFiles:
    """Test that command files match registry."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = TinyTorchCLI()
        self.commands_dir = Path(__file__).parent.parent.parent / "tito" / "commands"

    def test_command_files_exist(self):
        """Verify all registered commands have corresponding files."""
        # Map command names to expected file paths (now with subfolders!)
        cmd_to_file = {
            'setup': 'setup.py',
            'system': 'system/__init__.py',  # Now in subfolder
            'module': 'module/__init__.py',   # Now in subfolder
            'src': 'src.py',
            'package': 'package/__init__.py', # Now in subfolder
            'nbgrader': 'nbgrader.py',
            'milestones': 'milestone.py',
            'leaderboard': 'leaderboard.py',
            'olympics': 'olympics.py',
            'benchmark': 'benchmark.py',
            'community': 'community.py',
            'export': 'export.py',
            'test': 'test.py',
            'book': 'book.py',
            'grade': 'grade.py',
            'demo': 'demo.py',
            'logo': 'logo.py',
        }

        for cmd_name, expected_file in cmd_to_file.items():
            if cmd_name in self.cli.commands:
                file_path = self.commands_dir / expected_file
                assert file_path.exists(), (
                    f"Command '{cmd_name}' registered but file missing: {expected_file}"
                )

    def test_no_orphaned_command_files(self):
        """Warn about command files that aren't registered."""
        # Get all Python files in commands directory (excluding special files)
        command_files = [
            f for f in self.commands_dir.glob("*.py")
            if f.name not in ['__init__.py', 'base.py']
        ]

        # Files we expect to see (registered commands + internal helpers)
        expected_files = {
            # Registered top-level commands
            'setup.py', 'src.py', 'nbgrader.py',
            'milestone.py', 'benchmark.py',
            'community.py', 'export.py', 'test.py',
            'grade.py', 'logo.py',
            # Known internal/subcommand files (not top-level)
            'login.py',  # Subcommand of community
            'clean_workspace.py', 'version.py', 'check.py', 'view.py',
            'protect.py', 'report.py'
        }

        orphaned = []
        for cmd_file in command_files:
            if cmd_file.name not in expected_files:
                orphaned.append(f"{cmd_file.name} -> not in expected files")

        if orphaned:
            pytest.fail(
                f"Found {len(orphaned)} orphaned command files:\n" +
                "\n".join(f"  - {item}" for item in orphaned) +
                "\n\nEither register these commands or move to subfolders"
            )


class TestEpilogDocumentation:
    """Test that epilog in parser matches actual available commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cli = TinyTorchCLI()

    def test_epilog_mentions_registered_commands(self):
        """Verify epilog mentions all registered command groups."""
        parser = self.cli.create_parser()
        epilog = parser.epilog

        # Key command groups that should be mentioned
        expected_groups = [
            'system',
            'module',
            'package',
            'nbgrader',
            'milestones',
            'leaderboard',
            'olympics'
        ]

        missing = []
        for group in expected_groups:
            if group in self.cli.commands:
                if group not in epilog:
                    missing.append(group)

        if missing:
            pytest.fail(
                f"Commands registered but not in epilog: {missing}\n"
                f"Update epilog in tito/main.py create_parser() method"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
