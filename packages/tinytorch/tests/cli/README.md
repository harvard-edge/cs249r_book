# CLI Test Suite

Comprehensive test suite for the TinyTorch CLI (`tito`). Ensures all commands are properly registered, documented, and executable.

## Test Files

### 1. `test_cli_registry.py` - Command Registration Tests
Validates the command registry in [tito/main.py](../../tito/main.py#L74-L99):

- **TestCLIRegistry**: Core registry validation
  - All commands inherit from `BaseCommand`
  - All commands have descriptions (>10 chars)
  - All commands implement `execute()` and `add_arguments()`
  - Parser creation succeeds
  - No duplicate command names
  - Help text accessible for all commands

- **TestCommandFiles**: File system consistency
  - All registered commands have corresponding files
  - No orphaned command files (files without registration)
  - Detects commands that need cleanup

- **TestEpilogDocumentation**: Help text consistency
  - Epilog mentions all registered command groups
  - No outdated references

### 2. `test_cli_execution.py` - Command Execution Tests
Smoke tests for actual command execution:

- **TestCommandExecution**: Basic execution
  - Bare `tito` command shows welcome screen
  - `tito -h` shows help
  - `tito --version` works
  - All 18 commands can show help (`tito <cmd> -h`)
  - All subcommands can show help

- **TestCommandGrouping**: Discoverability
  - Student-facing commands visible in welcome screen
  - Developer commands documented in help

- **TestErrorMessages**: Error handling
  - Invalid commands show helpful errors
  - Missing subcommands show help

### 3. `test_cli_help_consistency.py` - Help Text Quality
Ensures help text is consistent and complete:

- **TestHelpConsistency**: Formatting consistency
  - All command helps mention 'tito'
  - Bare `tito` vs `tito -h` serve different purposes
  - No references to removed commands

- **TestWelcomeScreen**: Welcome screen quality
  - Shows Quick Start section
  - Organizes commands into groups
  - Includes example commands

- **TestCommandDocumentation**: Documentation completeness
  - All registered commands documented
  - Checkpoint command specifically validated

## Running the Tests

Run all CLI tests:
```bash
pytest tests/cli/ -v
```

Run specific test file:
```bash
pytest tests/cli/test_cli_registry.py -v
```

Run with output:
```bash
pytest tests/cli/ -v -s
```

## What These Tests Catch

### ✅ Prevents

1. **Orphaned Commands**: Command files that aren't registered
2. **Missing Documentation**: Commands without descriptions
3. **Broken Help**: Commands that crash when showing help
4. **Inconsistent UX**: Different help formats across commands
5. **Stale References**: Help text mentioning removed commands

### ✅ Validates

1. **Registration**: All commands in `TinyTorchCLI.commands` dict
2. **Implementation**: All commands inherit from `BaseCommand`
3. **Documentation**: All commands have clear descriptions
4. **Execution**: All commands can run without crashing
5. **Discoverability**: Key commands visible to users

## Test Coverage

**52 tests** covering:
- 18 registered commands
- 8 subcommand groups
- Registry validation
- Execution smoke tests
- Help text consistency
- Welcome screen UX

## How Modern CLIs Handle Testing

These tests follow industry best practices from tools like:

1. **Click** (Python): Command registration validation
2. **Git**: Comprehensive help text testing
3. **Docker**: Smoke tests for each command
4. **kubectl**: Subcommand hierarchy validation

Key patterns used:
- **Registry-based validation**: Single source of truth in code
- **Smoke tests**: Don't test functionality, just "does it run?"
- **Help text parsing**: Ensure documentation stays current
- **Snapshot testing**: Compare outputs (could be added later)

## Maintenance

When adding a new command:

1. Add to `TinyTorchCLI.commands` dict in [tito/main.py](../../tito/main.py#L74-L99)
2. Create command file in `tito/commands/`
3. Add to epilog if it's a major command group
4. Tests will automatically validate it!

When removing a command:

1. Remove from `TinyTorchCLI.commands` dict
2. Delete command file OR add to `known_internal` in tests
3. Update epilog/welcome screen

## Future Enhancements

Consider adding:

- **Snapshot tests**: Save known-good help output, compare changes
- **Integration tests**: Test actual command workflows
- **Performance tests**: Ensure CLI startup is fast
- **Completion tests**: Validate shell completion scripts
- **Config tests**: Test config file parsing and validation
