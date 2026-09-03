"""
TinyTorch CLI Color Theme

Consistent color palette for all CLI output.
Logo-inspired but terminal-safe for both dark and light backgrounds.
"""


class Theme:
    """Centralized color constants for CLI styling."""

    # ==========================================
    # BRAND COLORS
    # ==========================================
    BRAND_ACCENT = "orange1"          # Primary brand color (matches logo "tiny")
    BRAND_PRIMARY = "bold white"      # Main text (TORCH - navy doesn't work on dark terminals)
    BRAND_FLAME = "yellow"            # Flame emoji styling

    # ==========================================
    # SEMANTIC COLORS (Status Messages)
    # ==========================================
    SUCCESS = "green"                 # ✅ Success messages, completed status
    WARNING = "yellow"                # ⚠️ Warnings, caution messages
    ERROR = "red"                     # ❌ Errors, failures
    INFO = "cyan"                     # ℹ️ Info messages, general information

    # ==========================================
    # UI COLORS (Help & Navigation)
    # ==========================================
    COMMAND = "bright_green"          # Command names in help text
    OPTION = "yellow"                 # CLI options, arguments, flags
    SECTION = "bold cyan"             # Section headers
    DIM = "dim"                       # Secondary text, descriptions
    EMPHASIS = "bold white"           # Important highlighted text

    # ==========================================
    # PANEL BORDERS
    # ==========================================
    BORDER_DEFAULT = "bright_blue"    # Default panel border
    BORDER_SUCCESS = "green"          # Success panel border
    BORDER_WARNING = "yellow"         # Warning panel border
    BORDER_ERROR = "red"              # Error panel border
    BORDER_INFO = "cyan"              # Info panel border
    BORDER_WELCOME = "bright_green"   # Welcome screen border

    # ==========================================
    # CATEGORY COLORS (Command Groups)
    # ==========================================
    CAT_QUICKSTART = "bright_green"   # Quick start commands
    CAT_PROGRESS = "yellow"           # Progress tracking commands
    CAT_COMMUNITY = "cyan"            # Community commands
    CAT_HELP = "magenta"              # Help & docs commands


# Convenience aliases for common patterns
COLORS = Theme
