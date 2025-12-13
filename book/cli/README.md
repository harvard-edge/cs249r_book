# MLSysBook CLI v2.0 - Modular Architecture

This directory contains the refactored, modular CLI for the MLSysBook project. The CLI has been broken down from a monolithic 4000+ line script into maintainable, testable modules.

## Architecture

```
cli/
├── __init__.py           # Package initialization
├── main.py              # Main CLI application class
├── core/                # Core functionality
│   ├── config.py        # Configuration management
│   └── discovery.py     # Chapter/file discovery
├── commands/            # Command implementations
│   └── build.py         # Build operations
├── formats/             # Format-specific handlers (future)
└── utils/               # Shared utilities (future)
```

## Key Components

### ConfigManager (`core/config.py`)
- Manages Quarto configuration files for HTML, PDF, and EPUB
- Handles symlink creation and switching between formats
- Provides output directory detection from config files

### ChapterDiscovery (`core/discovery.py`)
- Discovers and validates chapter files
- Supports fuzzy matching for chapter names
- Provides chapter listing and dependency detection

### BuildCommand (`commands/build.py`)
- Handles build operations for all formats
- Supports both full book and individual chapter builds
- Includes progress indication and error handling

### MLSysBookCLI (`main.py`)
- Main application class that orchestrates all components
- Provides command routing and help system
- Beautiful Rich-based UI with organized command tables

## Usage

The modular CLI has replaced the original binder and is available as `./binder` in the project root:

```bash
# Show help
./binder help

# Build commands
./binder build                    # Build full book (HTML)
./binder build intro,ml_systems   # Build specific chapters (HTML)
./binder pdf intro                # Build chapter as PDF
./binder epub                     # Build full book as EPUB

# Preview commands
./binder preview                  # Start live dev server for full book
./binder preview intro            # Start live dev server for chapter

# Management
./binder list                     # List all chapters
./binder status                   # Show current status
./binder doctor                   # Run comprehensive health check
```

## Benefits of Modular Architecture

1. **Maintainability**: Each component has a single responsibility
2. **Testability**: Individual modules can be unit tested
3. **Debuggability**: Issues can be isolated to specific modules
4. **Extensibility**: New formats and commands are easy to add
5. **Code Reuse**: Shared functionality is properly modularized
6. **Collaboration**: Multiple developers can work on different components

## Migration Complete

The modular CLI has successfully replaced the original monolithic binder script:

- **`./binder`** - Modular CLI (4000+ lines → organized modules)

All functionality has been preserved and enhanced:
- All existing commands work the same way
- Same configuration files and output directories
- Same build processes and quality
- Enhanced error handling and progress indication
- New preview and doctor commands added

## Future Enhancements

The modular architecture enables easy addition of:
- Format-specific handlers in `formats/`
- Additional commands in `commands/`
- Shared utilities in `utils/`
- Plugin system for custom extensions
- Comprehensive unit test suite
