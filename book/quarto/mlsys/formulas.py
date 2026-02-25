# formulas.py
# Canonical equations for Machine Learning Systems
# centralizing the logic for TCO, Physics, and Performance math.

from .constants import ureg, SPEED_OF_LIGHT_FIBER_KM_S, MS, MB, GB, hour

def _ensure_unit(val, unit):
    """Helper to attach unit if value is a raw number."""
    if isinstance(val, (int, float)):
        return val * unit
    return val

def calc_network_latency_ms(distance_km):
    """Calculates round-trip time in milliseconds."""
    d = _ensure_unit(distance_km, ureg.kilometer)
    round_trip_s = (d * 2) / SPEED_OF_LIGHT_FIBER_KM_S
    return round_trip_s.to(ureg.millisecond).magnitude

def dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """
    Core training time calculation (physics-first).
    Returns a Pint Quantity in seconds.
    """
    # ops / (n * p * eta)
    effective_throughput = num_devices * peak_flops_per_device * efficiency_eta
    duration = total_ops / effective_throughput
    return duration.to(ureg.second)


def calc_training_time_days(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """Calculates training duration in days."""
    duration = dTime(total_ops, num_devices, peak_flops_per_device, efficiency_eta)
    return duration.to(ureg.day).magnitude

def calc_amdahls_speedup(p, s):
    """
    Calculates overall system speedup given:
    p: fraction of work that can be improved (0.0 to 1.0)
    s: speedup of that fraction
    """
    overall = 1 / ((1 - p) + (p / s))
    return overall

def calc_monthly_egress_cost(bytes_per_sec, cost_per_gb):
    """Calculates monthly cloud egress cost."""
    b_s = _ensure_unit(bytes_per_sec, ureg.byte / ureg.second)
    monthly_bytes = b_s * (30 * ureg.day)
    cost = monthly_bytes * cost_per_gb
    return cost.to(ureg.dollar).magnitude

def calc_fleet_tco(unit_cost, power_w, quantity, years, kwh_price):
    """Calculates Total Cost of Ownership (TCO)."""
    u_cost = _ensure_unit(unit_cost, ureg.dollar)
    p_w = _ensure_unit(power_w, ureg.watt)
    price = _ensure_unit(kwh_price, ureg.dollar / ureg.kilowatt_hour)
    time = _ensure_unit(years, ureg.year)
    fleet_capex = u_cost * quantity
    total_energy = p_w * quantity * time
    power_opex = total_energy * price
    total = fleet_capex + power_opex
    return total.to(ureg.dollar).magnitude

def calc_bottleneck(ops, model_bytes, device_flops, device_bw):
    """Roofline bottleneck analysis."""
    compute_time = ops / device_flops
    memory_time = model_bytes / device_bw
    t_comp_ms = compute_time.to(ureg.millisecond).magnitude
    t_mem_ms = memory_time.to(ureg.millisecond).magnitude
    is_memory_bound = t_mem_ms > t_comp_ms
    ratio = t_mem_ms / t_comp_ms if is_memory_bound else t_comp_ms / t_mem_ms
    intensity = ops / model_bytes
    return {
        "compute_ms": t_comp_ms,
        "memory_ms": t_mem_ms,
        "bottleneck": "Memory" if is_memory_bound else "Compute",
        "ratio": ratio,
        "intensity": intensity.magnitude
    }

def model_memory(params, bytes_per_param, unit=MB):
    """
    Calculate model memory footprint.

    Args:
        params: Number of parameters (pint Quantity with param unit, or raw number)
        bytes_per_param: Bytes per parameter (pint Quantity with byte unit, or raw number)
        unit: Target unit for output (default: MB)

    Returns:
        Memory footprint as float in the specified unit

    Example:
        >>> model_memory(RESNET50_PARAMS, BYTES_FP32)  # Returns ~102 (MB)
        >>> model_memory(GPT3_PARAMS, BYTES_FP16, GB)  # Returns ~350 (GB)
    """
    # Extract magnitudes to avoid param*byte dimension issues
    if hasattr(params, 'magnitude'):
        param_count = params.magnitude
    else:
        param_count = params

    if hasattr(bytes_per_param, 'magnitude'):
        bpp = bytes_per_param.magnitude
    else:
        bpp = bytes_per_param

    total_bytes = param_count * bpp * ureg.byte
    return total_bytes.to(unit).magnitude

