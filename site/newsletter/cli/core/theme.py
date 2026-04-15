"""Semantic color palette for the news CLI.

Mirrors the Tito CLI theme pattern: a single Theme class with constants for
status, category, and panel borders. Using Rich color names so it adapts to
both dark and light terminals.
"""


class Theme:
    # Brand
    BRAND_ACCENT = "bright_blue"
    BRAND_PRIMARY = "bold white"

    # Semantic status
    SUCCESS = "green"
    WARNING = "yellow"
    ERROR = "red"
    INFO = "cyan"

    # UI (help and navigation)
    COMMAND = "bright_green"
    OPTION = "yellow"
    SECTION = "bold cyan"
    DIM = "dim"
    EMPHASIS = "bold white"

    # Panel borders
    BORDER_DEFAULT = "bright_blue"
    BORDER_SUCCESS = "green"
    BORDER_WARNING = "yellow"
    BORDER_ERROR = "red"
    BORDER_INFO = "cyan"

    # Category colors (command groups)
    CAT_DRAFT = "bright_green"
    CAT_PUBLISH = "bright_magenta"
    CAT_ARCHIVE = "yellow"
    CAT_INFO = "cyan"
