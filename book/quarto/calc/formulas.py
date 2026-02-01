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

def calc_training_time_days(total_ops, num_devices, peak_flops_per_device, efficiency_eta):
    """Calculates training duration in days."""
    # ops / (n * p * eta)
    effective_throughput = num_devices * peak_flops_per_device * efficiency_eta
    duration = total_ops / effective_throughput
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

def fmt(quantity, unit=None, precision=1, commas=True):
    """
    Format a Pint Quantity for narrative text.
    Returns ONLY the number string (no unit suffix).
    
    Usage:
        fmt(RESNET_FLOPS, "Gflop") -> "4.1"
        fmt(COST, "USD", 2) -> "0.09"
    """
    if unit:
        # Check if quantity is raw number, if so apply unit first to allow conversion
        # (This handles the case where we passed a raw int to a function)
        if isinstance(quantity, (int, float)):
            # This is ambiguous, better to assume the user handled it or fail.
            # But for safety in qmds, let's assume it's already in base units if raw.
            pass 
        else:
            quantity = quantity.to(unit)
    
    if hasattr(quantity, "magnitude"):
        val = quantity.magnitude
    else:
        val = quantity
        
    fmt_str = f",.{precision}f" if commas else f".{precision}f"
    return f"{val:{fmt_str}}"

def sci(val, precision=2):
    """
    Formats a number or Pint Quantity into LaTeX scientific notation.
    Example: 4.1e9 -> "4.10 \\times 10^{9}"
    """
    if hasattr(val, "magnitude"): val = val.magnitude
    s = f"{val:.{precision}e}"
    base, exp = s.split('e')
    # Remove leading zero from exponent if present, but handle sign
    exp_int = int(exp) 
    return f"{float(base):.{precision}f} \\times 10^{{{exp_int}}}"

class Namespace:
    """Helper to store calculation results for easy access in prose."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def fmt(self, name, unit=None, precision=1):
        return fmt(getattr(self, name), unit, precision)
