"""Physical AI and embodied ML systems physics and safety accounting formulas (Volume IV).

Domain scope:
- Sensor-to-actuator end-to-end latency budgets
- Dynamic stopping distance and kinetic inertia safety bounds
- Actuator stator thermal dissipation and power envelopes
- Action chunking rate adaptation and control loop timing
- Reflected rotor inertia and gear ratios
"""

from __future__ import annotations

from mlsysim.core.units import ureg
from mlsysim.core._validation import (
    validate_positive,
    validate_nonnegative,
    validate_at_least,
)


def calc_sensor_to_actuator_latency(
    sensor_latency,
    inference_latency,
    arbitration_latency,
    actuator_latency,
):
    """
    Calculate the total end-to-end perception-to-actuation loop latency.

    Parameters
    ----------
    sensor_latency : Quantity
        Sensor capture and transmission delay (e.g. camera frame exposure + USB/CSI bus).
    inference_latency : Quantity
        Policy inference duration on neural accelerator (e.g. VLA, Diffusion Policy).
    arbitration_latency : Quantity
        Deterministic nervous system / Control Barrier Function arbitration delay.
    actuator_latency : Quantity
        Motor controller response time and CAN/EtherCAT bus transit delay.

    Returns
    -------
    Quantity
        Total closed-loop latency in milliseconds or seconds.
    """
    total = sensor_latency + inference_latency + arbitration_latency + actuator_latency
    return total.to(ureg.millisecond)


def calc_safe_stopping_distance(velocity, total_latency, max_deceleration):
    """
    Calculate dynamic stopping distance under non-zero control latency.

    Equation:
        d_stop = v * tau_latency + (v^2) / (2 * a_max)

    Parameters
    ----------
    velocity : Quantity
        Current operational velocity (e.g. m/s).
    total_latency : Quantity
        End-to-end perception-to-brake latency (e.g. ms or s).
    max_deceleration : Quantity
        Maximum braking deceleration available (e.g. m/s^2).

    Returns
    -------
    Quantity
        Total stopping distance in meters.
    """
    validate_nonnegative(velocity, "velocity")
    validate_positive(total_latency, "total_latency")
    validate_positive(max_deceleration, "max_deceleration")

    reaction_distance = velocity * total_latency
    braking_distance = (velocity ** 2) / (2 * max_deceleration)
    total_distance = reaction_distance + braking_distance
    return total_distance.to(ureg.meter)


def calc_actuator_thermal_power(current, resistance, duty_cycle: float = 1.0):
    """
    Calculate average ohmic stator heat dissipation in electromagnetic actuators.

    Equation:
        P_loss = (I^2 * R) * duty_cycle

    Parameters
    ----------
    current : Quantity
        Phase current flowing through actuator windings (e.g. Amperes).
    resistance : Quantity
        Phase winding resistance (e.g. Ohms).
    duty_cycle : float, optional
        Operating duty cycle fraction (0.0 to 1.0).

    Returns
    -------
    Quantity
        Thermal loss power in Watts.
    """
    validate_positive(resistance, "resistance")
    validate_nonnegative(current, "current")

    power = (current ** 2) * resistance * duty_cycle
    return power.to(ureg.watt)


def calc_action_chunk_cadence(chunk_horizon_steps: int, control_loop_hz, brain_inference_hz):
    """
    Evaluate the freshness and execution headroom of an action chunking policy.

    Parameters
    ----------
    chunk_horizon_steps : int
        Number of consecutive actions predicted in a single model inference chunk.
    control_loop_hz : Quantity
        Execution frequency of the low-level nervous system / motor loop (e.g. Hz).
    brain_inference_hz : Quantity
        Inference rate of the high-level neural policy (e.g. Hz).

    Returns
    -------
    dict
        Execution window duration, inference budget, and headroom factor.
    """
    validate_at_least(chunk_horizon_steps, 1, "chunk_horizon_steps")
    validate_positive(control_loop_hz, "control_loop_hz")
    validate_positive(brain_inference_hz, "brain_inference_hz")

    step_period = (1.0 / control_loop_hz).to(ureg.second)
    chunk_duration = chunk_horizon_steps * step_period
    brain_period = (1.0 / brain_inference_hz).to(ureg.second)

    headroom_factor = (chunk_duration / brain_period).to_base_units().magnitude

    return {
        "chunk_duration": chunk_duration.to(ureg.millisecond),
        "brain_period": brain_period.to(ureg.millisecond),
        "headroom_factor": headroom_factor,
    }


def calc_reflected_inertia(gear_ratio: float, rotor_inertia):
    """
    Calculate reflected rotor inertia felt at the actuator output shaft.

    Equation:
        J_reflected = N^2 * J_rotor

    Parameters
    ----------
    gear_ratio : float
        Transmission reduction ratio N (>= 1.0).
    rotor_inertia : Quantity
        Unloaded rotor moment of inertia (e.g. kg * m^2).

    Returns
    -------
    Quantity
        Reflected inertia in kg * m^2.
    """
    validate_at_least(gear_ratio, 1.0, "gear_ratio")
    reflected = (gear_ratio ** 2) * rotor_inertia
    return reflected
