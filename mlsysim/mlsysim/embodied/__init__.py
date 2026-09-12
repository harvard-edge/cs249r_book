"""Embodied package for physical AI platforms, cyber-physical machines, and robotic embodiments."""

from .types import EmbodiedPlatform, RobotPlatform
from .registry import Embodied, Robotics

__all__ = ["EmbodiedPlatform", "RobotPlatform", "Embodied", "Robotics"]
