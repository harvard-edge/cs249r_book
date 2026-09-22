# mlsysim/__init__.py
"""
mlsysim: Machine Learning Systems Infrastructure and Modeling Platform
"""

__version__ = "0.1.2"

from . import core
from . import engine
from . import hardware
from . import models
from . import platforms
from . import infrastructure
from . import systems
from . import sim
from . import physics
from . import embodied
from . import sensors
from . import actuators
from . import agents

# AUTHORITATIVE API ENTRY POINTS
from .engine.engine import Engine
from .engine.scenarios import Scenario, Scenarios
from .hardware.registry import Hardware
from .models.registry import Models
from .platforms.registry import Platforms
# Datasets loaded lazily via __getattr__ below.
from .systems.registry import Systems
from .infrastructure.registry import Infrastructure
from .literature.registry import Literature
# Non-executable sourced anchors used by examples and external analyses.
from .reference_stats.registry import ReferenceStats
from .embodied.registry import Embodied
from .sensors.registry import Sensors
from .actuators.registry import Actuators
from .agents.registry import Agents, ReferencePlatforms as AgentPlatforms
from .ops import Ops
from .engine import calibration

# AUTHORITATIVE MEASUREMENT (units + physics-only constants)
from .core.units import *  # noqa: F401,F403

# AUTHORITATIVE PHYSICS FORMULAS
from .physics import *  # noqa: F401,F403

# AUTHORITATIVE FORMATTING
from .fmt import (
    fmt, fmt_int, fmt_qty, fmt_usd, fmt_eur, fmt_percent, fmt_pp, fmt_multiple,
    fmt_multiple_range, fmt_time, fmt_rate, fmt_fps, fmt_count, fmt_params, fmt_tokens,
    fmt_ratio, fmt_range, fmt_magnitude, fmt_text, fmt_display_math, fmt_math,
    fmt_unit, fmt_val, fmt_frac, fmt_qty_int, fmt_sci, fmt_sci_flops,
    fmt_qty_range, fmt_time_range, fmt_count_range, fmt_usd_range,
    fmt_percent_range, fmt_sci_qty,
    fmt_power, fmt_energy, fmt_bandwidth, fmt_flop_rate, fmt_flops,
    fmt_arithmetic_intensity, fmt_ops_rate, fmt_compute_efficiency,
    fmt_area, fmt_heat_flux, fmt_specific_heat, fmt_memory, fmt_memory_capacity,
    fmt_length, fmt_emissions, fmt_carbon_intensity, fmt_water,
    fmt_water_rate, fmt_water_intensity, fmt_latency,
    fmt_frequency, fmt_acceleration, fmt_torque, fmt_torque_rate, fmt_torque_constant, fmt_velocity, fmt_jerk,
    fmt_inertia, fmt_voltage, fmt_current, fmt_resistance, fmt_token_rate,
    fmt_force, fmt_force_rate, fmt_mass, fmt_angular_velocity,
    fmt_temperature, fmt_temperature_rate, fmt_decibel, fmt_illuminance,
    fmt_energy_per_op, fmt_energy_per_bit, fmt_energy_per_byte, fmt_energy_per_flop,
    fmt_thermal_resistance, fmt_heat_capacity, fmt_density,
    fmt_mass_flow, fmt_mass_flow_rate, fmt_volumetric_flow, fmt_flow_rate,
    fmt_charge, fmt_battery_capacity, fmt_angle, fmt_inductance, fmt_capacitance, fmt_power_rate,
    fmt_stiffness, fmt_volume,
    assert_qty_close, check, MarkdownStr,
)


def plot_evaluation_scorecard(*args, **kwargs):
    """Render a system evaluation scorecard."""
    from .viz.plots import plot_evaluation_scorecard as _plot_evaluation_scorecard
    return _plot_evaluation_scorecard(*args, **kwargs)


def plot_roofline(*args, **kwargs):
    """Render a Roofline plot."""
    from .viz.plots import plot_roofline as _plot_roofline
    return _plot_roofline(*args, **kwargs)


# datasets imported at the end — after all other subpackages are registered.
# The .gitignore fix (mlsysim/.gitignore overrides root datasets/ exclusion)
# ensures the subpackage is included in the built wheel.
from . import datasets
from .datasets.registry import Datasets


__all__ = sorted(name for name in globals() if not name.startswith("_"))
