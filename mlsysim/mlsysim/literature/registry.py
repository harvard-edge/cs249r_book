"""Published literature anchors cited in the book (MFU, scaling, Chinchilla, …).

Each anchor is a provenance-carrying scalar; the values live as YAML under
``literature/data/<category>.yaml`` and are loaded via ``load_sourced_registry``
(provenance referenced by catalog key). See ``.claude/rules/mlsysim.md`` →
*Storage format*. Literature holds ONLY genuinely-cited field figures.
"""
from pathlib import Path

from ..core.registry import Registry
from ..core.loader import load_sourced_registry

_DATA = Path(__file__).parent / "data"


def _load(stem: str, name: str, doc: str = "") -> type:
    return load_sourced_registry(_DATA / f"{stem}.yaml", name=name, doc=doc)


Training = _load("training", "Training")
Scaling = _load("scaling", "Scaling")
Overheads = _load("overheads", "Overheads")
Chinchilla = _load("chinchilla", "Chinchilla")
Communication = _load("communication", "Communication")
BatchSize = _load("batchsize", "BatchSize", "McCandlish et al. (2018) critical batch size estimates.")
Energy = _load(
    "energy", "Energy",
    """Simplified pedagogical energy hierarchy for the sustainability chapter.

    Architecture-class EFFECTIVE energy per FLOP (CPU->ASIC) and per-byte
    data-movement cost (register->network). Order-of-magnitude teaching figures,
    distinct from the device-level Horowitz raw-MAC/per-access constants in
    ``core.constants`` (e.g. ``ENERGY_FLOP_FP32_PJ``). DRAM per-byte is NOT
    duplicated here -- it uses the canonical device value
    ``constants.ENERGY_DRAM_PJ_PER_BYTE`` (160).
    """,
)


class Literature(Registry):
    """Registry namespace for Literature."""
    Training = Training
    Scaling = Scaling
    Overheads = Overheads
    Chinchilla = Chinchilla
    Communication = Communication
    BatchSize = BatchSize
    Energy = Energy
