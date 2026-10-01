"""Embodied package for physical AI platforms and cyber-physical machine embodiments."""

from .types import EmbodiedPlatform
from .registry import Embodied
from .datasheet import data_sheet_lookup, machine_card, machine_data_sheet

__all__ = ["EmbodiedPlatform", "Embodied", "data_sheet_lookup", "machine_card", "machine_data_sheet"]
