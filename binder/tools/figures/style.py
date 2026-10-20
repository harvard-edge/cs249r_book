"""Figure style for book cells: a re-export of ``mlsysim.viz.style``.

The style (palette, fonts, font sizes, figure size) has one source,
``mlsysim/mlsysim/viz/style.py``. This module keeps the import path chapter
cells and margin tooling already use; define nothing here.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from mlsysim.viz import style as _style
except ImportError:
    # Folder-based resolution: mlsysim is never pip-installed. With only the
    # repo root on sys.path, the outer mlsysim/ folder imports as an empty
    # namespace package; drop it and point at the real package.
    for _name in [m for m in sys.modules if m == "mlsysim" or m.startswith("mlsysim.")]:
        del sys.modules[_name]
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "mlsysim"))
    from mlsysim.viz import style as _style

BODY_FIGSIZE = _style.BODY_FIGSIZE
COLORS = _style.COLORS
MIN_FONT_PT = _style.MIN_FONT_PT
PRINT_SCALE_RANGE = _style.PRINT_SCALE_RANGE
PRINT_TEXT_WIDTH_IN = _style.PRINT_TEXT_WIDTH_IN
WEB_FIG_DPI = _style.WEB_FIG_DPI
plt = _style.plt
bar_compare = _style.bar_compare
finalize_web_figure = _style.finalize_web_figure
set_book_style = _style.set_book_style
setup_plot = _style.setup_plot
