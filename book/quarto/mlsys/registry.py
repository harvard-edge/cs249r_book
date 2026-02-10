"""
registry.py
Transparent execution tape for chapter calculations.
"""

TAPE = []


def start_chapter(chapter_id):
    """Initialize a new tape for a chapter."""
    TAPE.clear()
    TAPE.append({"type": "chapter_start", "chapter": chapter_id})


def record(name, value, units=None, context=None):
    """Record a calculation step and return the value."""
    entry = {
        "name": name,
        "value": value,
        "units": units,
        "context": context,
    }
    TAPE.append(entry)
    return value


def dump_tape():
    """Return the full tape (for inline display)."""
    return TAPE
