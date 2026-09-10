# units.py
# Measurement infrastructure for the ML Systems simulator.
# This module owns the pint unit registry and defines all unit aliases.
# It contains ONLY measurement plumbing — no domain knowledge or tuneable defaults.

from pathlib import Path

import pint

__all__ = [
    # Registry and Quantity constructor
    "ureg", "Q_",
    # Dimensionless scalars
    "QUADRILLION", "TRILLION", "BILLION", "MILLION", "THOUSAND", "HUNDRED",
    # Base units
    "byte", "bit", "second", "joule", "watt", "kilowatt", "milliwatt", "meter", "liter", "L", "hour", "day", "count",
    "kelvin", "K",
    # Data scale units
    "KB", "MB", "GB", "TB", "PB", "KiB", "MiB", "GiB", "TiB",
    # Precision sizes
    "BYTES_FP32", "BYTES_INT32", "BYTES_FP16", "BYTES_INT8", "BYTES_INT4", "BYTES_ADAM_STATE",
    # Bit widths, precision map, encoding/format facts
    "FP32_BITS", "INT8_BITS", "SIMD_REGISTER_BITS", "PRECISION_MAP",
    "normalize_precision", "resolve_precision",
    "COLOR_DEPTH_8BIT", "IMAGE_CHANNELS_RGB", "VIDEO_BYTES_PER_PIXEL_RGB",
    "VIDEO_1080P_WIDTH", "VIDEO_1080P_HEIGHT",
    "VIDEO_4K_WIDTH", "VIDEO_4K_HEIGHT", "VIDEO_FPS_STANDARD",
    # Time units
    "MS", "US", "NS", "ms", "MILLISECOND", "MICROSECOND", "NANOSECOND",
    "microsecond", "millisecond", "nanosecond", "minute", "week", "year",
    # Time scalars
    "SECONDS_PER_MINUTE", "MINUTES_PER_HOUR", "SEC_PER_HOUR", "HOURS_PER_DAY",
    "SEC_PER_DAY", "DAYS_PER_MONTH", "DAYS_PER_YEAR", "SEC_PER_YEAR", "SEC_PER_YEAR_LEAP",
    "HOURS_PER_YEAR", "MS_PER_SEC", "US_PER_MS", "NS_PER_US", "NS_PER_MS", "NS_PER_SEC",
    # Data size scalars
    "BITS_PER_BYTE", "KIB_TO_BYTES", "MIB_TO_BYTES", "GIB_TO_BYTES",
    # FLOPs and operation-rate units
    "flop", "KFLOP", "KFLOPs", "MFLOP", "MFLOPs", "GFLOP", "GFLOPs", "TFLOP", "TFLOPs",
    "PFLOP", "PFLOPs", "EFLOP", "EFLOPs", "ZFLOP", "ZFLOPs",
    "OPS", "KOPS", "MOPS", "GOPS", "TOPS",
    # Network
    "Gbps", "megabit", "gigabit", "terabit",
    # Currency
    "USD", "EUR",
    # Model parameters
    "param", "Kparam", "Mparam", "Bparam", "Tparam",
    # Energy and power aliases
    "J", "kJ", "MJ", "mJ", "uJ", "pJ", "Wh", "kWh", "MWh", "GWh",
    "GW", "MW", "kW", "mW", "uW",
    "kilojoule", "microjoule", "gigawatt", "megawatt", "microwatt",
    # Mass and length
    "gram", "kilogram", "kg", "metric_ton", "tonne", "kilometer", "km", "foot", "feet",
]

ureg = pint.UnitRegistry()
ureg.formatter.default_format = "~P"           # compact Pretty: "312 TFLOP/s" not "312.0 teraFLOPs / second"
pint.set_application_registry(ureg)  # canonical registry for the whole mlsysim package
Q_ = ureg.Quantity

# Custom MLSysIM units — single reviewable vocabulary file.
ureg.load_definitions(str(Path(__file__).with_name("mlsysim_units.txt")))

# --- Dimensionless Scalars (Helpers) ---
QUADRILLION = 1e15
TRILLION = 1e12
BILLION = 1e9
MILLION = 1e6
THOUSAND = 1e3
HUNDRED = 100

# --- Units ---
byte = ureg.byte
bit = ureg.bit
second = ureg.second
joule = ureg.joule
watt = ureg.watt
kilowatt = ureg.kilowatt
milliwatt = ureg.milliwatt
meter = ureg.meter
liter = ureg.liter
L = ureg.liter
hour = ureg.hour
day = ureg.day
count = ureg.count
kelvin = ureg.kelvin
K = ureg.kelvin

# --- Exported aliases from loaded/custom registry ---
KB = ureg.KB
MB = ureg.MB
GB = ureg.GB
TB = ureg.TB
PB = ureg.PB

# Common precision sizes
BYTES_FP32 = 4 * byte
BYTES_INT32 = 4 * byte
BYTES_FP16 = 2 * byte
BYTES_INT8 = 1 * byte
BYTES_INT4 = 0.5 * byte
# Adam FP32 first + second moments ONLY (4 + 4). EXCLUDES the FP32 master
# copy of the weights — mixed-precision accountings that keep the master add
# +4 B/param (engine/calibration.py: TRAINING_OPTIMIZER_BYTES_ADAM = 12,
# CHECKPOINT_BYTES_PER_PARAM_ADAM = 14 include it). Pick deliberately.
BYTES_ADAM_STATE = 8 * byte

# Bit widths (companion to the byte widths above)
FP32_BITS = 32
INT8_BITS = 8
SIMD_REGISTER_BITS = 512

# Precision string -> byte width. Shared by Engine, ServingModel, SynthesisSolver.
PRECISION_MAP = {
    "fp32": BYTES_FP32,   # 4 bytes
    "tf32": BYTES_FP32,   # 4 bytes storage, TF32 compute
    "bf16": BYTES_FP16,   # 2 bytes
    "fp16": BYTES_FP16,   # 2 bytes
    "fp8":  BYTES_INT8,   # 1 byte
    "int8": BYTES_INT8,   # 1 byte
    "int4": BYTES_INT4,   # 0.5 bytes
}


def normalize_precision(precision: str) -> str:
    """Return the canonical precision key or raise for unsupported formats."""
    if not isinstance(precision, str):
        raise TypeError(f"precision must be a string, got {type(precision).__name__}")
    key = precision.lower()
    if key not in PRECISION_MAP:
        supported = ", ".join(sorted(PRECISION_MAP))
        raise ValueError(f"precision '{precision}' is not supported. Supported precision values: {supported}")
    return key


def resolve_precision(precision: str):
    """Return ``(canonical_key, bytes_per_element)`` for a supported precision.

    The single entry point solvers use to turn a user-facing precision string
    into a storage width. ``bytes_per_element`` is a Pint byte Quantity from
    ``PRECISION_MAP`` — the *storage* width, not the compute format (so
    'tf32' maps to 4 bytes and 'fp8' to 1 byte). Case-insensitive; raises
    ``ValueError`` for unsupported formats, listing the supported keys.

    Examples
    --------
    >>> resolve_precision("FP16")
    ('fp16', <Quantity(2, 'byte')>)
    """
    key = normalize_precision(precision)
    return key, PRECISION_MAP[key]

# Color / image / video encoding + format facts
COLOR_DEPTH_8BIT = 256
IMAGE_CHANNELS_RGB = 3
VIDEO_BYTES_PER_PIXEL_RGB = 3 * byte
VIDEO_1080P_WIDTH = 1920
VIDEO_1080P_HEIGHT = 1080
VIDEO_4K_WIDTH = 3840   # UHD 4K (2160p) horizontal resolution
VIDEO_4K_HEIGHT = 2160  # UHD 4K (2160p) vertical resolution
VIDEO_FPS_STANDARD = Q_(30, 'Hz')

KiB = ureg.KiB
MiB = ureg.MiB
GiB = ureg.GiB
TiB = ureg.TiB

# --- Time (legacy uppercase MS/US/NS loaded from mlsysim_units.txt) ---
MS = ureg.MS
ms = ureg.ms          # pint built-in millisecond (alias for convenience)
US = ureg.US
NS = ureg.NS
MILLISECOND = MS
MICROSECOND = US
NANOSECOND = NS
# SI lowercase aliases (consumed by interactive examples and notebooks)
microsecond = ureg.microsecond
millisecond = ureg.millisecond
nanosecond = ureg.nanosecond
minute = ureg.minute
week = ureg.week
year = ureg.year

# Common time conversions (unitless scalars)
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
SEC_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR
HOURS_PER_DAY = 24
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY
DAYS_PER_MONTH = 30
DAYS_PER_YEAR = 365
SEC_PER_YEAR = SEC_PER_DAY * DAYS_PER_YEAR
SEC_PER_YEAR_LEAP = int(365.25 * SEC_PER_DAY)
HOURS_PER_YEAR = 8760

# Data size scalars
BITS_PER_BYTE = 8
KIB_TO_BYTES = 1024
MIB_TO_BYTES = 1024 * 1024
GIB_TO_BYTES = 1024 * 1024 * 1024

# Time scalars
MS_PER_SEC = 1000
US_PER_MS = 1000
NS_PER_US = 1000
NS_PER_MS = NS_PER_US * US_PER_MS
NS_PER_SEC = NS_PER_MS * MS_PER_SEC

# --- FLOPs and operation rates (loaded from mlsysim_units.txt) ---
flop = ureg.flop
KFLOP = ureg.KFLOP
KFLOPs = ureg.KFLOPs
MFLOP = ureg.MFLOP
MFLOPs = ureg.MFLOPs
GFLOP = ureg.GFLOP
GFLOPs = ureg.GFLOPs
TFLOP = ureg.TFLOP
TFLOPs = ureg.TFLOPs
PFLOPs = ureg.PFLOPs
PFLOP = ureg.PFLOP
EFLOP = ureg.EFLOP
EFLOPs = ureg.EFLOPs
ZFLOP = ureg.ZFLOP
ZFLOPs = ureg.ZFLOPs

OPS = ureg.OPS
KOPS = ureg.KOPS
MOPS = ureg.MOPS
GOPS = ureg.GOPS
TOPS = ureg.TOPS

Gbps = ureg.Gbps

USD = ureg.dollar
EUR = ureg.EUR

param = ureg.param
Kparam = ureg.Kparam
Mparam = ureg.Mparam
Bparam = ureg.Bparam
Tparam = ureg.Tparam

# --- Convenience aliases (prefer these over ureg.* in authored examples) ---
J = joule
kJ = ureg.kilojoule
MJ = ureg.megajoule
mJ = ureg.millijoule
uJ = ureg.microjoule
pJ = ureg.picojoule
Wh = ureg.watt_hour
kWh = ureg.kilowatt_hour
MWh = ureg.megawatt_hour
GWh = ureg.gigawatt_hour
GW = ureg.gigawatt
MW = ureg.megawatt
kW = kilowatt
mW = milliwatt
uW = ureg.microwatt
kilojoule = ureg.kilojoule
microjoule = ureg.microjoule
gigawatt = ureg.gigawatt
megawatt = ureg.megawatt
microwatt = ureg.microwatt
gram = ureg.gram
kilogram = ureg.kilogram
kg = ureg.kilogram
metric_ton = ureg.metric_ton
tonne = ureg.metric_ton
kilometer = ureg.kilometer
km = ureg.kilometer
foot = ureg.foot
feet = ureg.foot
megabit = ureg.megabit
gigabit = ureg.gigabit
terabit = ureg.terabit
