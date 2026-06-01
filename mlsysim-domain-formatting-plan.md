# MLSysIM Domain Formatting Plan

This plan proposes a small MLSysBook authoring layer on top of Pint and the
existing `mlsysim.fmt` helpers. The goal is to make LEGO cells shorter, more
consistent, and harder to get wrong, without losing the dimensional safety that
Pint already provides.

The plan is based on a read-only scan of all QMD files under:

- `book/quarto/contents/vol1/**/*.qmd`: 36 files
- `book/quarto/contents/vol2/**/*.qmd`: 39 files

That is 75 QMD files total. The scan was programmatic, with follow-up spot
checks in high-signal chapters such as `sustainable_ai`, `responsible_engr`,
`performance_engineering`, `inference`, `compute_infrastructure`, and
`hw_acceleration`.

## Executive Summary

The current foundation is good:

- Registry values are already Pint quantities.
- `fmt_qty`, `fmt_time`, `fmt_count`, `fmt_usd`, `fmt_percent`, and related
  helpers already centralize most rendering.
- `fmt_qty` already protects the most important invariant: physical prose output
  should still receive a Pint quantity, not a raw float.
- Pint's compact pretty formatter is already used in part through
  `ureg.formatter.default_format = "~P"` and `_compact_unit_suffix`.

The main gap is authoring ergonomics. LEGO cells still require authors to
remember details like:

- whether to use `GB`, `GiB`, `GB/second`, `TB/second`, or `Gbps`
- when to pass `precision=0`
- when to pass `unit_label="GB"` for vendor-style memory labels
- when to display `kWh` vs `MWh`
- how to consistently show `TFLOP/s` instead of variants such as `TFLOPs/s`
- how to format parameters, tokens, token rates, carbon intensity, and emissions

The proposed solution is not to replace Pint. It is to add domain-specific
formatters that use Pint and the current `fmt_*` functions underneath.

The target authoring style should be:

```python
bw_str = fmt_bandwidth(Hardware.Cloud.H100.memory.bandwidth)
mem_str = fmt_memory(Hardware.Cloud.H100.memory.capacity, vendor_label=True)
peak_str = fmt_flop_rate(Hardware.Cloud.H100.compute.peak_flops)
ttft_str = fmt_latency(result.ttft)
params_str = fmt_params(Models.Language.Llama2_70B.parameters)
tokens_str = fmt_tokens(prompt_tokens)
energy_str = fmt_energy(train_energy)
carbon_str = fmt_emissions(total_emissions)
```

rather than:

```python
bw_str = fmt_qty(bw, GB/second, precision=0, commas=False)
mem_str = fmt_qty(mem, GiB, precision=0, commas=False, unit_label="GB")
peak_str = fmt_qty(flops, TFLOP/second, precision=0, commas=False)
ttft_str = fmt_time(ttft, "millisecond", precision=1, commas=False)
params_str = fmt_count(params, scale="B", precision=0, commas=False, label="parameter")
energy_str = fmt(energy_mwh, precision=0, suffix=" MWh")
```

## Current Usage Patterns

Formatter usage across Vol 1 and Vol 2:

| Helper | Vol 1 | Vol 2 | Total |
|---|---:|---:|---:|
| `fmt_qty` | 239 | 279 | 518 |
| `fmt_qty_int` | 11 | 20 | 31 |
| `fmt_time` | 374 | 292 | 666 |
| `fmt_count` | 138 | 117 | 255 |
| `fmt_usd` | 194 | 77 | 271 |
| `fmt_percent` | 327 | 297 | 624 |
| `fmt_rate` | 52 | 17 | 69 |
| `fmt_multiple` | 102 | 5 | 107 |
| `fmt_int` | 145 | 265 | 410 |
| `unit_label=` | 31 | 50 | 81 |
| `per=` | 89 | 20 | 109 |

Most common `fmt_qty` display units:

| Display unit | Approximate call count |
|---|---:|
| `GB/second` | 92 |
| `GB` | 50 |
| `watt` | 47 |
| `MB` | 45 |
| `TB/second` | 38 |
| `TFLOP/second` | 34 |
| `Gbps` | 22 |
| `KB` | 22 |
| `GiB` | 20 |
| `ureg.megawatt` | 15 |
| `MB/second` | 12 |
| `TB` | 11 |
| `ms` | 11 |
| `kilowatt` | 10 |
| `TOPS` | 8 |

Most common `fmt_time` display units:

| Display unit | Approximate call count |
|---|---:|
| `millisecond` | 335 |
| `hour` | 100 |
| `second` | 87 |
| `minute` | 57 |
| `microsecond` | 22 |
| `week` | 14 |
| `year` | 14 |
| `month` | 13 |
| `nanosecond` | 7 |
| `day` | 9 |

Common `unit_label=` values:

| Unit label | Count |
|---|---:|
| `GB` | 31 |
| `Gb/s` | 20 |
| `MB` | 6 |
| `FLOP/byte` | 5 |
| `TFLOP/s` | 3 |
| `Gb` | 2 |
| `Mb/s` | 2 |
| `TFLOP/s per W` | 2 |

The corpus repeatedly formats the same concepts:

- memory and storage capacity
- memory bandwidth and network bandwidth
- compute rates and FLOP counts
- arithmetic intensity
- latency, duration, TTFT, ITL, dispatch, transfer time
- model parameters
- tokens and token rates
- power, energy, PUE, and carbon
- cost and price rates
- percentages, percentage points, utilization, MFU, goodput

This is exactly the shape where domain helpers help.

## Problem Examples

### Energy and Carbon

In `vol2/sustainable_ai/sustainable_ai.qmd`, there are comments like:

```python
energy_kwh = energy_mwh * THOUSAND  # MWh -> kWh (mlsysim does not export kilowatt_hour)
training_mwh_str = fmt(energy_mwh, precision=0, suffix=" MWh")
energy_kwh_str = fmt(energy_kwh, precision=0, suffix=" kWh")
```

This is fragile because the unit conversion lives in a scalar expression and the
unit label lives in a suffix string. A better style is:

```python
energy = Q_(energy_mwh, MWh)
training_mwh_str = fmt_energy(energy, MWh)
energy_kwh_str = fmt_energy(energy, kWh)
```

or, when no equation requires a specific unit:

```python
energy_str = fmt_energy(energy, unit="auto")
```

### Parameters and Forced Scale

Today, if the author writes:

```python
fmt_count(150e6, scale="B", precision=0, commas=False, label="parameter")
```

the formatter correctly raises because `150M = 0.15B`, and `precision=0` would
hide that as `0B`.

That guard is valuable. The better author-facing API is:

```python
fmt_params(150e6)       # "150M parameters"
fmt_params(1.2e9)       # "1.2B parameters"
fmt_params(70e9)        # "70B parameters"
```

If a caller explicitly forces `scale="B"`, the helper can either:

- render `0.15B parameters` with enough precision, or
- raise with a clear message: use `scale="M"` or `scale="auto"`.

The default should make the common case easy and safe.

### Latency

The corpus frequently writes:

```python
fmt_time(ttft, "millisecond", precision=1, commas=False)
fmt_time(dispatch_us, "microsecond", precision=0, commas=False)
```

That is correct, but it requires repetition. For metrics like TTFT and ITL, the
unit should still be time. TTFT and ITL should not become Pint units. They are
metric names whose values are Pint durations.

Better:

```python
ttft_str = fmt_latency(ttft)
itl_str = fmt_latency(itl)
dispatch_str = fmt_latency(dispatch_tax, unit="auto")
```

## Design Principles

### Keep Pint as the Source of Truth

The wrappers should never replace Pint math. They should:

1. accept Pint quantities where appropriate
2. convert with Pint
3. format through existing `fmt_*` helpers
4. return `MarkdownStr`

The wrappers should not silently accept raw numbers for physical dimensions,
except where the concept itself is intentionally a count, such as tokens.

### Make Domain Defaults Encode Book Policy

Generic helpers cannot know whether `precision=0` is appropriate. Domain helpers
can.

Examples:

- `fmt_tokens(4096)` can default to integer display because token counts are
  counts.
- `fmt_params(70e9)` can default to `70B parameters`.
- `fmt_latency(0.003 * second)` can default to milliseconds.
- `fmt_energy(1500 * kWh, unit="auto")` can choose `1.5 MWh`.
- `fmt_memory(80 * GiB, vendor_label=True)` can display `80 GB` while still
  dimension-checking the binary quantity.

### Keep Explicit Units for Equation Steps

Automatic scaling is useful in prose but can be harmful in derivations where the
reader needs to see matching units.

For example, in a carbon derivation:

```text
10,000 MWh = 10,000,000 kWh
10,000,000 kWh * 429 g/kWh = ...
```

the code should explicitly call:

```python
energy_mwh_str = fmt_energy(energy, MWh)
energy_kwh_str = fmt_energy(energy, kWh)
```

not rely on `unit="auto"`.

### Use Automatic Scaling for Summary Prose and Tables

Automatic scaling is appropriate when the prose only needs a readable magnitude:

```python
fmt_energy(1_500_000 * kWh, unit="auto")   # "1.5 GWh"
fmt_params(150e6)                          # "150M parameters"
fmt_tokens(1_200_000, scale="auto")        # "1.2M tokens"
fmt_emissions(12_500 * kilogram)           # "12.5 metric tons"
```

### Preserve Existing Generic Helpers

`fmt_qty`, `fmt_time`, `fmt_count`, `fmt_usd`, `fmt_percent`, and `fmt_rate`
should remain public. The domain helpers should be conveniences and policy
encoders, not a replacement for every unusual case.

## Proposed Unit Registry Changes

File: `mlsysim/mlsysim/core/units.py`

### Export Energy Units

Add stable exported aliases:

```python
Wh = ureg.watt_hour
kWh = ureg.kilowatt_hour
MWh = ureg.megawatt_hour
GWh = ureg.gigawatt_hour
```

If Pint's built-in aliases are not all present on every supported version, define
the missing ones explicitly:

```python
ureg.define("Wh = watt * hour")
ureg.define("kWh = kilowatt * hour")
ureg.define("MWh = megawatt * hour")
ureg.define("GWh = gigawatt * hour")
```

Add them to `__all__`.

### Export Mass Units

Add:

```python
gram = ureg.gram
kilogram = ureg.kilogram
kg = ureg.kilogram
metric_ton = ureg.metric_ton
tonne = ureg.metric_ton
```

Add them to `__all__`.

This enables carbon emissions to be represented as quantities rather than raw
floats:

```python
emissions = 429 * kilogram
emissions_t = emissions.to(metric_ton)
```

### Fix Time Alias Display

Current custom definitions:

```python
ureg.define("MS = 1e-3 * second")
ureg.define("US = 1e-6 * second")
ureg.define("NS = 1e-9 * second")
```

can display as `MS`, `US`, `NS`. That is not ideal in book prose.

Prefer aliases to Pint's canonical units:

```python
ureg.define("@alias millisecond = MS")
ureg.define("@alias microsecond = US")
ureg.define("@alias nanosecond = NS")
```

Expected display behavior:

```python
fmt_time(5, US, precision=0)   # "5 us" or "5 microseconds" depending policy
```

The exact microsecond glyph policy should be decided in `fmt.py`. The book may
prefer `us` for ASCII source and rendering stability, or the Greek-mu symbol for
typographic output.

### Normalize FLOP Aliases

Current units include:

```python
GFLOP
GFLOPs
TFLOP
TFLOPs
PFLOPs
```

This is useful for parsing existing data, but display should be canonical:

- FLOP count: `TFLOP`, `GFLOP`, `PFLOP`
- FLOP rate: `TFLOP/s`, `GFLOP/s`, `PFLOP/s`

Add parse compatibility aliases if desired:

```python
ureg.define("@alias flop = FLOP = FLOPs = flops")
ureg.define("@alias TFLOP = TFLOPs")
ureg.define("@alias GFLOP = GFLOPs")
ureg.define("@alias PFLOP = PFLOPs")
```

Do not encourage `TFLOPS` as canonical display. Industry uses it, but in a book
that distinguishes FLOPs from FLOP/s, the clearer output is `TFLOP/s`.

### Consider Token Units Carefully

Tokens are common in prose and formulas, but token counts are not physical units
in the same way bytes and seconds are. There are two reasonable paths:

1. Keep tokens as counts and format with `fmt_tokens` / `fmt_token_rate`.
2. Add Pint units `token`, `Ktoken`, `Mtoken`, `Btoken`, `Ttoken`.

Recommendation for first pass: do not make tokens a required Pint unit. Add
`fmt_tokens` and `fmt_token_rate` first. Add token Pint units later only if they
meaningfully simplify formulas without making authoring heavier.

## Proposed Formatter Internal Changes

File: `mlsysim/mlsysim/fmt.py`

### Use Pint Unit Formatting Directly

Current `_compact_unit_suffix` formats `1 * display_unit` and splits off the
leading magnitude:

```python
one = 1 * display_unit
formatted = f"{one:~P}"
parts = formatted.split(None, 1)
```

Pint can format units directly:

```python
label = f"{display_unit:~P}"
```

Proposed implementation shape:

```python
_UNIT_LABEL_NORMALIZATION = {
    "GFLOPs/s": "GFLOP/s",
    "TFLOPs/s": "TFLOP/s",
    "PFLOPs/s": "PFLOP/s",
    "ZFLOPs/s": "ZFLOP/s",
    "Gbps": "Gb/s",
    "US": "us",
    "MS": "ms",
    "NS": "ns",
}

def _compact_unit_suffix(display_unit) -> str:
    if str(display_unit) in {"dollar", "USD", "EUR"}:
        raise ValueError(...)
    display_unit = _coerce_unit(display_unit)
    label = f"{display_unit:~P}"
    label = _UNIT_LABEL_NORMALIZATION.get(label, label)
    return f" {label}"
```

This keeps Pint's `~P` formatting while allowing book-specific normalization.

### Keep Precision Guards

Do not remove `_check_fmt_precision`. It catches real mistakes, especially
hidden zeroes and forced wrong scales.

Domain helpers should call existing `fmt`, `fmt_int`, `fmt_qty`, and
`fmt_qty_int` so they inherit these guards.

## Proposed Domain Helpers

All helpers should return `MarkdownStr`.

### `fmt_memory`

Purpose: memory/storage capacity.

Signature:

```python
def fmt_memory(
    quantity,
    unit="auto",
    *,
    precision=None,
    commas=False,
    vendor_label=False,
    approx=False,
    lower_bound=False,
):
    ...
```

Policy:

- Requires a Pint quantity.
- `unit="auto"` chooses `KB`, `MB`, `GB`, or `TB` based on magnitude.
- If the quantity is binary vendor memory, the caller can pass
  `vendor_label=True` to display `80 GiB` as `80 GB` when the book intentionally
  matches vendor language.
- Default precision:
  - `precision=0` for integer-ish values
  - `precision=1` for non-integer values
- Use `fmt_qty` underneath.

Examples:

```python
fmt_memory(80 * GiB, vendor_label=True)     # "80 GB"
fmt_memory(85.899 * GB)                     # "85.9 GB"
fmt_memory(50 * MB)                         # "50 MB"
fmt_memory(2 * TB)                          # "2 TB"
```

### `fmt_bandwidth`

Purpose: byte bandwidth and network bandwidth.

Signature:

```python
def fmt_bandwidth(
    quantity,
    unit="auto",
    *,
    precision=None,
    commas=False,
    network=False,
):
    ...
```

Policy:

- Requires a Pint quantity.
- For byte bandwidth, auto chooses `MB/second`, `GB/second`, or `TB/second`.
- For network bit rates, `network=True` can choose `Gb/s` or `Tb/s`.
- Default precision:
  - `0` for integer-ish `GB/s`
  - `1` or `2` for `TB/s` depending magnitude
- Use `fmt_qty`.

Examples:

```python
fmt_bandwidth(900 * GB/second)       # "900 GB/s"
fmt_bandwidth(3.35 * TB/second)      # "3.35 TB/s"
fmt_bandwidth(400 * Gbps, network=True)  # "400 Gb/s"
```

### `fmt_flop_rate`

Purpose: hardware compute throughput.

Signature:

```python
def fmt_flop_rate(
    quantity,
    unit=TFLOP / second,
    *,
    precision=0,
    commas=False,
    approx=False,
):
    ...
```

Policy:

- Requires a Pint quantity.
- Default display is `TFLOP/s`.
- Canonical display should be singular `TFLOP/s`, not `TFLOPs/s`.
- For very large systems, allow explicit `PFLOP/second`.

Examples:

```python
fmt_flop_rate(Hardware.Cloud.H100.compute.peak_flops)      # "989 TFLOP/s"
fmt_flop_rate(system_flops, PFLOP/second, precision=1)     # "3.2 PFLOP/s"
```

### `fmt_flops`

Purpose: operation counts, not rates.

Signature:

```python
def fmt_flops(
    quantity,
    unit="auto",
    *,
    precision=None,
    commas=False,
):
    ...
```

Policy:

- Requires a Pint quantity with FLOP/count dimensionality.
- Auto chooses `KFLOP`, `MFLOP`, `GFLOP`, `TFLOP`, or `PFLOP`.
- Default precision:
  - `0` for integer-ish values
  - `1` for non-integer scaled values

Examples:

```python
fmt_flops(4.1 * GFLOP)       # "4.1 GFLOP"
fmt_flops(350 * TFLOP)       # "350 TFLOP"
```

### `fmt_intensity`

Purpose: arithmetic intensity and related ratios with units.

Signature:

```python
def fmt_intensity(quantity, *, precision=1, commas=False):
    return fmt_qty(quantity, flop / byte, precision=precision, commas=commas, unit_label="FLOP/byte")
```

Examples:

```python
fmt_intensity(Hardware.Cloud.H100.ridge_point())  # "295.2 FLOP/byte"
```

### `fmt_latency`

Purpose: latency-style durations such as TTFT, ITL, dispatch tax, transfer
time, p99 latency, and model inference latency.

Signature:

```python
def fmt_latency(
    duration,
    unit="auto",
    *,
    precision=None,
    commas=False,
    style="symbol",
    allow_negative=False,
):
    ...
```

Policy:

- If `duration` is a raw number, treat it as milliseconds only if the caller
  explicitly opts in, or reject raw numbers. Preferred: require Pint quantities
  for new code.
- `unit="auto"` chooses:
  - `ns` below 1 microsecond
  - `us` below 1 millisecond
  - `ms` below 1 second
  - `second` above 1 second
- Default precision:
  - `0` for integer-ish values
  - `1` or `2` for small fractional latencies
- Use `fmt_time`.

Examples:

```python
fmt_latency(35 * millisecond)      # "35 ms"
fmt_latency(0.42 * millisecond)    # "420 us" or "0.42 ms", depending policy
fmt_latency(1.5 * second)          # "1.5 s"
```

Decision needed: whether auto should prefer `420 us` or `0.42 ms`.

### `fmt_duration`

Purpose: human-scale time spans such as hours, days, weeks, months, years.

This may simply be an alias/wrapper around `fmt_time(..., style="word")` with
better defaults.

Examples:

```python
fmt_duration(3 * hour)     # "3 hours"
fmt_duration(14 * day)     # "14 days"
```

### `fmt_params`

Purpose: model parameter counts.

Signature:

```python
def fmt_params(
    value,
    scale="auto",
    *,
    precision=None,
    style="symbol",
    approx=False,
):
    ...
```

Inputs:

- Pint `param` quantities
- raw counts

Policy:

- `scale="auto"` chooses:
  - `K` for thousands
  - `M` for millions
  - `B` for billions
  - `T` for trillions
- Default precision:
  - `0` if scaled value is integer-ish
  - `1` if one decimal is needed
  - possibly `2` for values below 1 in a forced scale
- Label should be `parameter` / `parameters`.
- `style="symbol"` gives `70B parameters`.
- `style="word"` gives `70 billion parameters`.
- If forced scale would render `0`, raise or increase precision.

Examples:

```python
fmt_params(150e6)       # "150M parameters"
fmt_params(1.2e9)       # "1.2B parameters"
fmt_params(70e9)        # "70B parameters"
fmt_params(7e9, style="word")  # "7 billion parameters"
```

### `fmt_tokens`

Purpose: token counts.

Signature:

```python
def fmt_tokens(
    value,
    scale=None,
    *,
    precision=None,
    approx=False,
):
    ...
```

Policy:

- Treat tokens as counts in first implementation.
- Default `scale=None` for exact context windows and prompts.
- `scale="auto"` for corpus sizes.
- Default precision:
  - `0` for unscaled token counts
  - automatic scaled precision for large corpora

Examples:

```python
fmt_tokens(4096)              # "4,096 tokens"
fmt_tokens(1.2e12, scale="auto")  # "1.2T tokens"
fmt_tokens(300e9, scale="B")      # "300B tokens"
```

### `fmt_token_rate`

Purpose: token throughput.

Signature:

```python
def fmt_token_rate(
    value,
    per="second",
    *,
    scale=None,
    precision=None,
):
    ...
```

Policy:

- Wrap `fmt_rate`.
- Accept `per="second"` and `per="hour"`.
- Default precision:
  - `0` unless scaled value is fractional

Examples:

```python
fmt_token_rate(1200)                  # "1,200 tokens/s"
fmt_token_rate(45.2e6, per="hour", scale="million")  # "45.2 million tokens/hour"
```

### `fmt_power`

Purpose: watts, kilowatts, megawatts.

Signature:

```python
def fmt_power(
    quantity,
    unit="auto",
    *,
    precision=None,
    commas=False,
):
    ...
```

Policy:

- Requires a Pint quantity.
- Auto chooses `mW`, `W`, `kW`, or `MW`.
- Default precision:
  - `0` for W values like `700 W`
  - `1` for kW/MW values where fractional part matters

Examples:

```python
fmt_power(700 * watt)       # "700 W"
fmt_power(5_600 * watt)     # "5.6 kW"
fmt_power(1.2 * megawatt)   # "1.2 MW"
```

### `fmt_energy`

Purpose: joules and watt-hours, especially `Wh`, `kWh`, `MWh`, `GWh`.

Signature:

```python
def fmt_energy(
    quantity,
    unit="auto",
    *,
    precision=None,
    commas=False,
):
    ...
```

Policy:

- Requires a Pint quantity.
- Auto for electricity energy should choose `Wh`, `kWh`, `MWh`, or `GWh`.
- For operation energy, explicit `joule`, `millijoule`, `microjoule`, or
  `picojoule` should be supported.
- Default precision:
  - `0` for integer-ish values
  - `1` for scaled fractional values

Examples:

```python
fmt_energy(1287 * MWh)             # "1,287 MWh"
fmt_energy(1287 * MWh, kWh)        # "1,287,000 kWh"
fmt_energy(1_500_000 * kWh, unit="auto")  # "1.5 GWh"
fmt_energy(66 * ureg.millijoule)   # "66 mJ"
```

### `fmt_emissions`

Purpose: carbon emissions mass.

Signature:

```python
def fmt_emissions(
    quantity,
    unit="auto",
    *,
    precision=None,
    commas=False,
    co2e=False,
):
    ...
```

Policy:

- Requires a Pint mass quantity.
- Auto chooses `kg` or `metric tons`.
- Default precision:
  - `0` for large integer-ish kg
  - `1` for metric tons
- Decide output label policy:
  - `kg CO2e`
  - `metric tons CO2e`
  - or plain `kg` / `metric tons` with prose adding `CO2e`

Recommendation: default helper should include carbon context:

```python
fmt_emissions(552000 * kilogram)  # "552 metric tons CO2e"
```

but allow `co2e=False` if a table header already carries the label.

### `fmt_carbon_intensity`

Purpose: grid carbon intensity, commonly `g CO2e/kWh` or `kg CO2e/kWh`.

Signature:

```python
def fmt_carbon_intensity(
    value,
    unit="g/kWh",
    *,
    precision=None,
    commas=False,
):
    ...
```

Inputs:

- raw `carbon_intensity_g_kwh` float from existing grid profiles
- Pint quantity if grid profile evolves to expose one

Policy:

- `unit="g/kWh"` default for grid-intensity prose.
- `unit="kg/kWh"` supported for equations that multiply by `kWh` and produce kg.
- Default precision:
  - `0` for `g/kWh`
  - `3` for `kg/kWh`

Examples:

```python
fmt_carbon_intensity(Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh)
# "429 g CO2e/kWh"

fmt_carbon_intensity(Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh, unit="kg/kWh")
# "0.429 kg CO2e/kWh"
```

### `fmt_temperature`

Purpose: Celsius and temperature rates, used rarely but currently handled with
`unit_label`.

Signature:

```python
def fmt_temperature(quantity, unit=ureg.degC, *, precision=0, commas=False):
    ...

def fmt_temperature_rate(quantity, unit=ureg.delta_degC / second, *, precision=0, commas=False):
    ...
```

This can be deferred. It is less common than energy, memory, bandwidth, and
latency.

## Proposed Formula Helpers

Domain formatters solve output consistency. Formula helpers solve repeated unit
math.

These should likely live outside `fmt.py`, for example in:

- `mlsysim/mlsysim/physics/performance.py`
- `mlsysim/mlsysim/physics/energy.py`
- `mlsysim/mlsysim/physics/carbon.py`
- or a new `mlsysim/mlsysim/book/calculations.py` if these are primarily
  textbook-facing.

### `transfer_time`

```python
def transfer_time(size: Quantity, bandwidth: Quantity) -> Quantity:
    return (size / bandwidth).to(second)
```

Use cases:

- model transfer over PCIe/NVLink
- checkpoint write time
- data ingestion bandwidth

### `compute_time`

```python
def compute_time(ops: Quantity, peak_rate: Quantity, efficiency=1.0) -> Quantity:
    return (ops / (peak_rate * efficiency)).to(second)
```

Use cases:

- roofline checks
- prefill time
- training step estimates

### `energy_from_power`

```python
def energy_from_power(power: Quantity, duration: Quantity) -> Quantity:
    return (power * duration).to(kWh)
```

Use cases:

- GPU-hours to kWh
- facility energy
- carbon calculations

### `facility_energy`

```python
def facility_energy(it_energy: Quantity, pue: float) -> Quantity:
    return it_energy * pue
```

### `carbon_emissions`

```python
def carbon_emissions(energy: Quantity, grid: GridProfile) -> Quantity:
    intensity = grid.carbon_intensity_g_kwh * gram / kWh
    return (energy * intensity).to(kilogram)
```

This eliminates manual:

```python
energy_kwh * carbon_intensity_g_kwh / THOUSAND
```

patterns.

### `gpu_hours`

```python
def gpu_hours(num_gpus: int, duration: Quantity) -> Quantity:
    return num_gpus * duration.to(hour)
```

This may use a simple quantity, or remain a count/time calculation.

## Infrastructure Model Enhancements

File: `mlsysim/mlsysim/infrastructure/types.py`

Current grid profile:

```python
carbon_intensity_g_kwh: float

@property
def carbon_intensity_kg_kwh(self) -> float:
    return self.carbon_intensity_g_kwh / 1000.0

def carbon_kg(self, facility_energy_kwh: float) -> float:
    return facility_energy_kwh * self.carbon_intensity_kg_kwh
```

This preserves old code, but encourages float arithmetic.

Add quantity-returning properties while keeping old ones:

```python
@property
def carbon_intensity(self):
    return self.carbon_intensity_g_kwh * gram / kWh

@property
def carbon_intensity_kg_per_kwh(self):
    return self.carbon_intensity.to(kilogram / kWh)

def carbon(self, facility_energy):
    if not isinstance(facility_energy, ureg.Quantity):
        facility_energy = facility_energy * kWh
    return (facility_energy * self.carbon_intensity).to(kilogram)
```

Do not remove `carbon_kg` in the first pass. Keep compatibility.

## Automatic Precision Policy

Generic `fmt` should remain strict and explicit. Automatic precision should live
inside domain helpers.

### Helper: `choose_precision`

Internal utility:

```python
def _auto_precision(value, *, max_precision=2):
    if integer-like:
        return 0
    if one decimal preserves value:
        return 1
    return min(2, max_precision)
```

But this should be used carefully. The existing precision guard is valuable and
should still catch cases where automatic rules hide meaningful values.

### Counts

Counts should default to integer display unless scaled:

```python
fmt_tokens(4096)      # precision 0
fmt_params(70e9)      # precision 0 after B scaling
fmt_params(1.2e9)     # precision 1 after B scaling
```

### Physical Quantities

Physical quantities should use concept-specific defaults:

- memory: usually `0` or `1`
- bandwidth: `0` for `GB/s`, `2` for `TB/s` when needed
- flop rate: usually `0`
- latency: `0`, `1`, or `2` depending scale
- energy: `0` for equation-friendly kWh/MWh; `1` for auto-scaled GWh/MWh
- emissions: `0` for kg, `1` for metric tons

### Forced Scale Safety

If the caller forces a scale that would produce a hidden zero, either:

1. raise with a clear error, or
2. increase precision automatically only if `auto_precision=True`.

Recommendation:

- Default `scale="auto"` should choose a good scale.
- Explicit scale should respect the author's request but still refuse hidden
  zero unless the caller passes `precision`.

Examples:

```python
fmt_params(150e6, scale="auto")  # "150M parameters"
fmt_params(150e6, scale="B")     # raise or "0.15B parameters"
fmt_params(150e6, scale="B", precision=2)  # "0.15B parameters"
```

## Relationship to Pint Formatting

Pint supports formatting such as:

```python
f"{quantity:.2f~P}"
f"{unit:~P}"
```

The project should use this more internally, but not expose raw Pint f-strings
as the main LEGO-cell authoring style.

Recommended stack:

```text
Pint UnitRegistry and Quantity
  -> generic fmt_qty / fmt_time / fmt_count / fmt_usd
  -> domain helpers such as fmt_bandwidth, fmt_energy, fmt_params
  -> LEGO prose
```

Why not raw Pint f-strings everywhere?

- They do not return `MarkdownStr`.
- They do not know Quarto/Pandoc escaping rules.
- They do not enforce the book's precision guard.
- They do not know vendor memory label policy.
- They do not know when `kg` means carbon emissions.
- They do not know whether to write `TFLOP/s` rather than `TFLOPs/s`.

Why use Pint internally?

- It owns unit parsing and conversion.
- It can format compact unit labels.
- It prevents dimensional mistakes.
- It avoids duplicate local unit-string logic.

## Migration Strategy

### Stage 0: Wait for Suffix Cleanup

Another thread is already removing legacy `suffix=` unit formatting. Do not
duplicate that work.

This plan should focus on improving the stable helper surface and then migrating
call sites after suffix cleanup lands.

### Stage 1: Add Units and Tests

Files:

- `mlsysim/mlsysim/core/units.py`
- `mlsysim/tests/test_fmt.py`
- possibly `mlsysim/tests/test_hardware.py`

Add:

- energy aliases: `Wh`, `kWh`, `MWh`, `GWh`
- mass aliases: `gram`, `kilogram`, `kg`, `metric_ton`, `tonne`
- improved time aliases: `MS`, `US`, `NS`
- FLOP display tests

Tests:

```python
assert f"{GB/second:~P}" == "GB/s"
assert fmt_time(5, US, precision=0) in {"5 us", "5 μs"}
assert fmt_qty(989 * TFLOP/second, TFLOP/second, precision=0) == "989 TFLOP/s"
assert fmt_qty(1.5 * MWh, MWh, precision=1) == "1.5 MWh"
```

### Stage 2: Improve Generic Unit Labeling

File:

- `mlsysim/mlsysim/fmt.py`

Change `_compact_unit_suffix` to use direct Pint unit formatting and apply a
normalization map.

Add tests for:

- `TFLOPs/s` normalizes to `TFLOP/s`
- `Gbps` normalizes to `Gb/s`
- time aliases normalize correctly
- currency is still refused by `fmt_qty`

### Stage 3: Add Domain Helpers

File:

- `mlsysim/mlsysim/fmt.py`

Start with the highest leverage helpers:

1. `fmt_memory`
2. `fmt_bandwidth`
3. `fmt_flop_rate`
4. `fmt_flops`
5. `fmt_latency`
6. `fmt_params`
7. `fmt_tokens`
8. `fmt_token_rate`
9. `fmt_power`
10. `fmt_energy`
11. `fmt_emissions`
12. `fmt_carbon_intensity`

Add tests before migrating the book.

### Stage 4: Add Carbon/Energy Formula Helpers

Files:

- `mlsysim/mlsysim/physics/energy.py` or equivalent
- `mlsysim/mlsysim/physics/carbon.py` or equivalent
- `mlsysim/mlsysim/infrastructure/types.py`

Add:

- `energy_from_power`
- `facility_energy`
- `carbon_emissions`
- grid quantity properties

Keep old float properties for compatibility.

### Stage 5: Migrate Pilot Chapters

Start with chapters that exercise the new helpers:

1. `book/quarto/contents/vol2/sustainable_ai/sustainable_ai.qmd`
2. `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd`
3. `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd`
4. `book/quarto/contents/vol2/inference/inference.qmd`

Why these:

- Sustainability validates `kWh`, `MWh`, carbon intensity, emissions, power.
- Responsible engineering validates TCO, carbon, latency, power.
- Performance engineering validates FLOP rate, bandwidth, intensity.
- Inference validates latency, params, tokens, memory, TTFT/ITL patterns.

Render each chapter after migration and compare output.

### Stage 6: Migrate Remaining Volumes Opportunistically

After pilot chapters are stable, use mechanical replacements where obvious:

```python
fmt_qty(x, GB/second, ...)
```

to:

```python
fmt_bandwidth(x, GB/second, ...)
```

and:

```python
fmt_time(x, "millisecond", ...)
```

to:

```python
fmt_latency(x, ...)
```

Do not force every unusual `fmt_qty` call into a domain helper. Keep the generic
helper for unusual values.

## Acceptance Criteria

### Authoring Criteria

After this work:

- Authors should rarely type `GB/second` for common bandwidth prose.
- Authors should rarely type `precision=0` for token or parameter counts.
- Authors should not manually convert `MWh` to `kWh` with `THOUSAND`.
- Authors should not manually attach `kg/kWh` strings to carbon intensity values.
- Authors should not need `unit_label="GB"` except in unusual cases; memory
  helper should own vendor-label policy.
- New LEGO cells should look like domain prose, not unit plumbing.

### Safety Criteria

- Pint quantities remain attached through calculation.
- `fmt_qty` still refuses raw floats.
- Domain helpers reject dimensionally wrong quantities.
- Forced wrong scales do not silently render zero.
- Currency remains routed through `fmt_usd`.
- Rendered outputs remain Quarto-safe `MarkdownStr`.

### Test Criteria

Add tests for:

- domain helper defaults
- auto scaling
- forced scale behavior
- hidden-zero refusal
- time alias display
- energy and carbon unit conversions
- vendor memory display
- FLOP/FLOP-rate canonical display
- dimensionality errors for wrong helper input

## Effort Estimate

### Low Technical Risk

The existing foundation already has:

- Pint registry
- unit-bearing registries
- `fmt_qty` with dimensional checks
- precision guards
- `MarkdownStr`
- many existing tests

The proposed helpers are mostly thin wrappers around known-good code.

### Medium Migration Effort

Rough effort:

- Add units and formatter normalization: 0.5 to 1 day
- Add domain helpers and tests: 1 to 2 days
- Add carbon/energy formula helpers: 1 day
- Pilot migration of 3-4 chapters: 1 to 2 days
- Full corpus migration after suffix cleanup: 3 to 7 days depending on how
  aggressively old patterns are replaced

### Main Risks

The risk is not unit math. The risk is prose policy drift:

- `80 GiB` vs vendor-style `80 GB`
- `Gb/s` vs `GB/s`
- `metric tons` vs `tonnes` vs `tons`
- `CO2` vs `CO2e`
- `1,287 MWh` vs `1.287 GWh`
- `420 us` vs `0.42 ms`
- preserving exact equation-unit displays

This is why automatic scaling should be optional and domain-specific, and why
pilot renders are important.

## Recommended First Implementation Slice

The smallest valuable slice:

1. Export `kWh`, `MWh`, `GWh`, `kilogram`, `metric_ton`.
2. Fix `_compact_unit_suffix` to use direct Pint unit formatting plus
   normalization.
3. Add:
   - `fmt_energy`
   - `fmt_power`
   - `fmt_emissions`
   - `fmt_carbon_intensity`
   - `fmt_params`
   - `fmt_tokens`
   - `fmt_latency`
4. Add tests.
5. Convert the first 200 lines of `sustainable_ai` where `MWh/kWh` and carbon
   intensity are currently manually formatted.

That will validate the design quickly without committing to a full-volume
migration.

## Open Policy Decisions

These should be decided before full migration:

1. Should microseconds display as `us` or the Greek-mu symbol?
2. Should carbon output include `CO2e` inside helpers or leave it to prose?
3. Should auto energy scale `1,287 MWh` to `1.287 GWh`, or preserve MWh for
   training-run energy because the book often discusses MWh?
4. Should `fmt_params(150e6, scale="B")` raise, or render `0.15B parameters`?
5. Should `fmt_memory(..., vendor_label=True)` be default for hardware registry
   memory capacity, or explicit at every call?
6. Should token counts become Pint units, or remain count-formatting helpers?
7. Should TTFT and ITL get named formatter aliases such as `fmt_ttft` and
   `fmt_itl`, or is `fmt_latency` enough?

Recommended defaults:

- Use ASCII `us` in source-oriented outputs unless the book style explicitly
  wants the micro symbol.
- Include `CO2e` in carbon-specific helpers by default, but allow disabling.
- Preserve explicit energy units in equations; allow auto scaling in summaries.
- Raise on forced hidden-zero parameter scales unless precision is explicit.
- Keep `vendor_label=True` explicit for memory until policy is settled.
- Keep tokens as count helpers first.
- Use `fmt_latency` for TTFT and ITL; do not add `fmt_ttft` unless repeated
  code still feels noisy.

## Addendum: LEGO Semantic Output Contract

Added 2026-06-01 after the direct `fmt(...)` audit found remaining
unit-bearing outputs that are correct enough to execute, but not yet consistent
with the desired authoring standard.

The book should converge on one simple rule:

> If an output string represents a unit-bearing or semantic-unit value, the
> output name, formatter, and rendered unit must agree.

That means `facility_energy_kwh_str` is not just a name. It is an assertion that
the exported string is facility energy, rendered in `kWh`, by a formatter that
knows it is energy. The display-unit tokens embedded near the end of the
variable name must be the unit or scale that will actually appear in prose.

### L/E/G/O Standard

Use this as the default LEGO discipline for every chapter.

| Stage | Standard |
|---|---|
| `LOAD` | Load canonical facts from MLSysIM registries. If a reusable fact is missing, add it to MLSysIM with provenance instead of defining it locally. Scenario-local assumptions may live here, but they must be named and unitized immediately. |
| `EXECUTE` | Keep Pint quantities intact through physical calculations. Use scalar magnitudes only at true scalar boundaries: plotting, optimizer APIs, dimensionless ratios, or explicit guard comparisons. Compound units should stay visible, e.g. `1.9 * (TB / second)`. |
| `GUARD` | Check dimensions, ranges, precision-sensitive values, and prose contracts. Guards should catch wrong unit dimensions, hidden-zero formatting, duplicated prose units, and a closed output whose name does not match its rendered unit. |
| `OUTPUT` | Convert to prose-facing strings only once, using `fmt_qty` or a domain formatter. Output variable names encode the semantic quantity and display unit or scale. Prose references closed exports bare and does not repeat owned units. |

### Output Naming Rule

Prefer this grammar for all prose-facing LEGO exports:

```text
<semantic_quantity_tokens>_<display_unit_or_scale_tokens>_str
```

The number of semantic tokens is flexible. The last meaningful tokens before
`_str` should identify the display unit or display scale. This makes names
machine-checkable and human-readable:

- `facility_energy_kwh_str` means semantic quantity `facility_energy`, rendered
  in `kWh`.
- `it_power_kw_str` means semantic quantity `it_power`, rendered in `kW`.
- `model_flops_tflops_str` means semantic quantity `model_flops`, rendered as
  `TFLOPs` or `TFLOP` work/count, not as raw FLOPs.
- `h100_peak_flops_tflop_s_str` means semantic quantity `h100_peak_flops`,
  rendered as `TFLOP/s` throughput.
- `frontier_params_b_str` means semantic quantity `frontier_params`, rendered
  at billion-parameter scale.

Do not interpret the convention as exactly three tokens. It is a suffix
contract: the end of the name tells us the display unit/scale, and `_str` tells
us the value is already prose-ready.

Examples:

```python
facility_energy_kwh_str = fmt_energy(facility_energy, unit=kWh, precision=1)
it_power_kw_str = fmt_power(accelerator_power, unit=kilowatt, precision=1)
grid_ci_g_per_kwh_str = fmt_carbon_intensity(grid_ci, unit=gram / kWh, precision=0)
operational_tonnes_str = fmt_emissions(operational_carbon, unit=metric_ton, precision=1)
h100_peak_flops_tflop_s_str = fmt_flop_rate(h100.compute.peak_flops, unit=TFLOP / second)
model_flops_tflops_str = fmt_flops(model_flops, unit=TFLOPs)
ridge_flop_per_byte_str = fmt_arithmetic_intensity(ridge, unit=flop / byte, precision=1)
frontier_params_b_str = fmt_params(model.parameters, scale="B")
pue_str = fmt(pue, precision=2, commas=False)
speedup_x_str = fmt_multiple(speedup)
```

Use lowercase ASCII unit tokens in variable names even when prose renders the
unit with uppercase symbols. For example, use `_kwh_str` for `kWh`,
`_tflop_s_str` for `TFLOP/s`, `_tflops_str` for `TFLOPs`, `_gb_s_str` for
`GB/s`, and `_g_per_kwh_str` for `g/kWh`.

If the name includes a display unit, the formatter call should normally include
an explicit `unit=` or display unit argument. For example,
`facility_energy_kwh_str` should not call `fmt_energy(facility_energy)` if that
helper may auto-scale to `MWh` or `GWh`.

When auto-scaling is the intent, do not put a fixed unit in the output name:

```python
total_energy_str = fmt_energy(total_energy)
model_size_str = fmt_memory(model_size)
```

Those names say "the helper owns the display unit." Prose must not assume a
specific unit for auto-scaled exports.

### Semantic Capture Rule

Every output should make the value kind clear enough that another agent can
choose the formatter without reading the surrounding prose.

| Value kind | Preferred formatter | Preferred name signal |
|---|---|---|
| Memory/storage capacity | `fmt_memory` or `fmt_qty` | `_gb_str`, `_gib_str`, `_tb_str`, `_bytes_str` |
| Bandwidth | `fmt_bandwidth` | `_gb_s_str`, `_tb_s_str`, `_mb_s_str` |
| Service/event throughput | `fmt_rate` | `_samples_s_str`, `_tokens_s_str`, `_images_s_str`, `_qps_str` |
| FLOP throughput | `fmt_flop_rate` | `_tflop_s_str`, `_gflop_s_str`, `_eflop_s_str` |
| FLOP work/count | `fmt_flops` or `fmt_qty` | `_gflops_str`, `_mflops_str`, `_flops_str` |
| Operation throughput | `fmt_ops_rate` or `fmt_qty` | `_tops_str`, `_gops_str` |
| Arithmetic intensity | `fmt_arithmetic_intensity` or `fmt_qty` | `_flop_per_byte_str` |
| Power | `fmt_power` | `_w_str`, `_kw_str`, `_mw_str` |
| Energy | `fmt_energy` | `_j_str`, `_mj_str`, `_kwh_str`, `_mwh_str` |
| Carbon mass/emissions | `fmt_emissions` | `_kg_str`, `_tonnes_str`, `_t_str` |
| Carbon intensity | `fmt_carbon_intensity` | `_g_per_kwh_str`, `_kg_per_kwh_str` |
| Latency/duration | `fmt_latency` or `fmt_time` | `_ns_str`, `_us_str`, `_ms_str`, `_s_str`, `_h_str` |
| Parameters | `fmt_params` | `_params_m_str`, `_params_b_str`, `_params_str` |
| Tokens | `fmt_tokens` | `_tokens_str`, `_tokens_m_str`, `_tokens_b_str` |
| Percent/ratio | `fmt_percent`, `fmt_ratio`, or `fmt` | `_pct_str`, `_ratio_str`, `_pue_str` |
| Multiples/speedups | `fmt_multiple` | `_x_str`, `_speedup_str` |

Open scalar strings are still allowed for pure numbers such as PUE, ratios,
integer counts, index values, and named dimensionless quantities. The rule is
not "never use `fmt`." The rule is "do not use plain `fmt` for a physical or
semantic-unit value when a typed formatter exists or should exist."

### Precision Policy TODO

Domain helpers should infer safe precision whenever the display policy is
standard enough to do so.

Required behavior:

- `fmt_params(150e6)` should naturally produce a useful compact value such as
  `150M`, not require the author to remember `precision=0`.
- `fmt_params(150e6, scale="B")` must not silently print `0B`. It should either
  choose enough precision, such as `0.15B`, or raise with a clear message when
  explicit `precision=0` would hide the value.
- Count helpers for tokens, parameters, GPUs, nodes, requests, and examples
  should default to integer precision when the scaled value is integer-like.
- Physical helpers should reject precision choices that hide non-zero values or
  round non-integer values to integers unless the caller uses an explicit
  integer formatter.
- Equation outputs may force a unit and precision, but the name must record the
  forced unit.

### Ratio and Multiple Policy TODO

The formatter surface already has `fmt_ratio` and `fmt_multiple`; agents should
not collapse both meanings into plain `fmt(...)`.

Use this decision rule:

| Meaning | Formatter | Name pattern | Prose pattern |
|---|---|---|---|
| Multiplicative comparison, speedup, slowdown, "more/less than", cost gap | `fmt_multiple` | `_x_str`, `_speedup_str`, `_reduction_x_str`, `_gap_x_str` | `` `{python} value_str`$\times$ `` |
| Bare dimensionless diagnostic ratio, quotient, or index | `fmt_ratio` | `_ratio_str`, `_dp_ratio_str`, `_pue_str` when PUE-specific helper is unnecessary | prose supplies the ratio meaning, no `$\times$` |
| Share/fraction of a whole | `fmt_percent` | `_pct_str`, `_share_pct_str` | formatter/prose percent contract |
| Physical quotient with units | domain formatter or `fmt_qty` | `_flop_per_byte_str`, `_g_per_kwh_str`, `_j_per_token_str` | closed unit string |

Examples:

```python
carbon_spread_x_str = fmt_multiple(carbon_spread, precision=0)
tokens_per_param_ratio_str = fmt_ratio(tokens_per_param, precision=1)
training_share_pct_str = fmt_percent(training_share, precision=0)
ridge_flop_per_byte_str = fmt_arithmetic_intensity(ridge, unit=flop / byte)
```

So a line like:

```python
ratio_str = fmt(ratio, precision=0, commas=True)
```

should normally become either:

```python
ratio_x_str = fmt_multiple(ratio, precision=0, commas=True)
```

when prose says `$\times$`, or:

```python
ratio_str = fmt_ratio(ratio, precision=0, commas=True)
```

when the value is a bare diagnostic quotient.

Follow-up improvement: tighten `fmt_ratio` and `fmt_multiple` so that if they
receive a Pint quantity, it must be dimensionless. Non-dimensionless quotients
belong in a physical formatter, not in ratio/multiple helpers.

### Formatter Surface TODO

The latest audit shows that the current formatter surface is close but not
complete. Add or confirm these helpers before the final broad migration:

- `fmt_flops(quantity, unit=None, precision=None, commas=False)` for FLOP work
  or operation counts such as `GFLOPs`, distinct from throughput.
- `fmt_arithmetic_intensity(quantity, unit=None, precision=None, commas=False)`
  for `FLOP/byte` roofline quantities.
- `fmt_ops_rate(quantity, unit=None, precision=None, commas=False)` for TOPS,
  GOPS, and related operation-throughput displays when `fmt_flop_rate` is not
  semantically right.
- Keep `fmt_flop_rate` for FLOP/s throughput and `fmt_compute_efficiency` for
  FLOP/s/W.
- Keep `fmt_qty` as the explicit fallback for one-off Pint quantities, but
  prefer a named domain formatter when the pattern appears repeatedly.

### Audit and Lint TODO

Add checks that make the contract enforceable:

1. Flag direct `fmt(quantity, ...)` in LEGO cells when the quantity is not
   dimensionless.
2. Flag direct `fmt(x.to(unit).magnitude, ...)` in LEGO `OUTPUT` blocks unless
   the conversion is explicitly dimensionless with `.to("").magnitude`.
3. Flag output names whose unit suffix disagrees with the formatter display
   unit, such as `_kwh_str` backed by an auto-scaling energy call.
4. Flag closed formatter outputs whose prose repeats the same unit.
5. Flag unit-bearing outputs whose names are generic, such as `value_str`,
   `total_str`, or `result_str`, when a semantic name and unit are available.
6. Extend the audit report so every `*_str` export records:
   - source expression;
   - semantic value kind;
   - formatter used;
   - display unit or scale;
   - closed vs open prose contract;
   - whether precision was inferred or explicit.

### Reader-Facing Assumption Language TODO

Do not repeat "MLSysIM" in every worked example. That would make the prose feel
like implementation advertising. Instead:

1. Add one short explanation in front matter, likely `About This Book` or the
   shared conventions include, saying that worked examples are calculated from
   MLSysIM, the book's unit-aware registry of hardware, model, infrastructure,
   and scenario assumptions.
2. After that, use natural reader-facing input prose. Prefer:

   ```markdown
   **Inputs:** 10,000 MWh training run; Quebec and Poland grid-intensity assumptions.
   ```

   over:

   ```markdown
   **Inputs:** 10,000 MWh training run; Quebec and Poland grid-intensity assumptions from MLSysIM.
   ```

3. Say "registry" only when it helps the reader distinguish a fixed book
   assumption from a scenario-local assumption. For example, "Quebec and Poland
   grid-intensity assumptions from the book registry" is useful in a chapter
   that is teaching provenance or sensitivity analysis; it is unnecessary in
   routine examples after the front-matter explanation exists.
4. Use code comments for implementation provenance:

   ```python
   energy = 10_000 * MWh  # scenario assumption
   quebec_grid = Infrastructure.Grids.Quebec  # registry value
   poland_grid = Infrastructure.Grids.Poland  # registry value
   ```

5. Margin notes should explain context, not define calculation inputs. Use a
   margin note for why Quebec is low-carbon or why regional grid mix matters.
   Use an `Inputs` sentence or compact input list for values that change the
   calculation.

### Editorial Coherence and Reference Audit TODO

After the semantic formatter migration is stable, run an editorial coherence
pass over the rendered prose. This is separate from unit correctness: the math
can be right while the prose still feels repetitive, mechanically generated, or
awkward.

Issue classes to audit:

1. **Repeated nearby table/figure references.** A paragraph may mention the
   same table or figure twice within a few sentences, for example "Table 13.6
   quantifies..." followed immediately by "The resulting shift is evident in
   table 13.6." Sometimes this is useful re-anchoring; often the second mention
   should become "the table," "this transition," or be removed.
2. **Rendered multiplier duplication.** Ensure there are no outputs like
   `7x $\times$ speedup` or `7××`. The standard is:

   ```python
   speedup_x_str = fmt_multiple(speedup)
   ```

   with prose:

   ```markdown
   `{python} Example.speedup_x_str`$\times$ speedup
   ```

   Do not use `fmt(..., suffix="x")`, `fmt(..., suffix="×")`, or prose that
   adds both "times" and `$\times$` around the same value.
3. **Stale docs examples.** The source QMDs no longer appear to use
   `suffix="x"` for multiplier exports, but `mlsysim/docs/api/fmt.fmt.qmd`
   still contains an old example. Update formatter documentation after the
   helper policy is finalized.
4. **Unit reattachment and scalarization.** Continue flagging patterns such as
   `fmt_qty(value * GB/second, GB/second, ...)` and
   `fmt(quantity.to(unit).magnitude, ...)`. In the checkpoint bandwidth case,
   the main prose export now uses `fmt_bandwidth(R.ckpt_write_bw, unit=GB/second,
   ...)`, which is the desired shape, but remaining helper magnitudes such as
   `_bw_val = fmt(R.ckpt_write_bw.to(GB/second).magnitude, ...)` should be
   reviewed during the cleanup pass.
5. **Side figure plus table pedagogy.** A margin figure beside a detailed table
   can be good textbook design when it provides a visual intuition that the
   table does not. Keep it when it helps scan the concept quickly and does not
   overload the page. Remove, move, or promote it inline when it competes with
   the table or adds only decoration.
6. **Service-rate notation drift.** Event rates such as `samples/s`,
   `tokens/s`, `images/s`, and `QPS` should use `fmt_rate`, not physical
   `fmt_qty`. They are counted events per time, not memory bandwidth. In table
   labels, prefer one spelling per concept (`samples/s` rather than mixing
   `samples/sec`, `samples/second`, and `Samples/s`) unless the source quote or
   benchmark name requires otherwise. Output names should expose the rendered
   rate unit, such as `throughput_samples_s_str`,
   `decode_tokens_s_str`, or `serving_qps_str`.
7. **Format-specific prose splits.** Blocks such as
   `::: {.content-visible when-format="html"}` and
   `::: {.content-visible when-format="pdf"}` are legitimate when HTML and PDF
   truly need different structure. They are risky when used only to nudge line
   breaks because the duplicated prose can drift. Audit these blocks for:
   duplicated-but-not-identical sentences, missing references in one format,
   different labels or captions, and PDF-only fragments that read oddly on
   their own.
8. **Floating single-sentence paragraphs.** Single-sentence paragraphs are
   sometimes useful as transitions, but they can also look like orphaned text
   left behind after an edit. Audit all body paragraphs that contain only one
   sentence, excluding headings, captions, list items, callout titles, quiz
   prompts, and intentional emphasis lines. Classify each as keep, merge with
   previous paragraph, merge with next paragraph, expand, or remove.
9. **Generic glossary deferrals.** Sentences such as "For a complete glossary
   of foundational MLOps terminology, see the glossary" are usually weak
   textbook prose. They can overclaim completeness, interrupt the local
   argument, and point readers away instead of teaching the needed distinction.
   Keep a glossary pointer only when the chapter genuinely needs to tell the
   reader where definitions live. Prefer a local distinction in prose, such as
   "This chapter assumes the foundational MLOps vocabulary introduced earlier
   and focuses on Enterprise Fleet Operations." If a glossary pointer remains,
   avoid "complete" unless the glossary is intentionally exhaustive for that
   scope.

Suggested audit workflow:

1. **Static prefilter.**
   - Find repeated `@tbl-*` / `@fig-*` references within a paragraph or within
     a short line window.
   - Find rendered/plain references such as `Table 13.6` repeated close
     together.
   - Find `fmt(..., suffix="x")`, `fmt(..., suffix="×")`, and prose patterns
     where an inline Python value is followed by both text "times" and
     `$\times$`.
   - Find `fmt_qty(<expr> * UNIT, UNIT)` and `fmt(<quantity>.to(...).magnitude)`.
   - Find service-rate spelling variants:
     `samples/sec`, `samples/second`, `Samples/s`, `tokens/second`,
     `images/sec`, and compare against nearby `fmt_rate` outputs.
   - Find all `content-visible when-format=` blocks and diff matching HTML/PDF
     variants by local neighborhood.
   - Find one-sentence body paragraphs and queue only prose paragraphs, not
     tables, captions, callout headings, or lists.
   - Find generic glossary deferrals:
     "complete glossary", "see the glossary", "consult the glossary", and
     `@sec-glossary` outside front/back matter.
2. **LLM review packets.** For each candidate, provide the local paragraph,
   the table/figure label, nearby caption, and any LEGO output definitions. Ask
   for a narrow judgment: keep, rewrite, remove second reference, or escalate.
3. **Rendered review.** For accepted candidates, inspect the rendered HTML/PDF
   because reference repetition, margin figures, and multiplier spacing are
   visual/editorial issues.
4. **Manual fixes only after material freeze.** Do not polish these while the
   underlying calculations or output names are still moving.

### Deferred PDF Layout QA TODO

Defer page-layout cleanup until the material is semantically correct: source of
truth, Pint quantity flow, formatter output names, prose inputs, and render
truth come first. After the real content work is stable, run a final PDF layout
pass and make page-by-page judgment calls.

Known issue class: margin congestion. A page can become unreadable when several
sidenotes, citation notes, and a margin figure or chart compete for the same
vertical margin. This is a layout problem, not a LEGO math problem.

Policy for the final layout pass:

1. Margin notes are for short context, definitions, source anchors, or
   interpretive comments.
2. Calculation-defining assumptions belong in `Inputs`, body prose, or the
   callout itself, not in the margin.
3. Long notes should move to body prose, normal footnotes/endnotes, or be
   shortened.
4. Margin figures should not sit beside pages already crowded with sidenotes.
   Promote the figure inline or move it to a quieter page when needed.
5. Use judgment: the goal is a readable printed page, not preserving every
   margin placement.

Automation options to explore:

- Parse LaTeX logs for warnings such as `Overfull \\hbox`, `Underfull \\vbox`,
  `Marginpar`, `Float too large`, and related page-placement warnings. These
  are useful signals but will not catch every visual collision.
- Preserve or archive PDF build logs per volume and add a warning summary to
  the layout QA report.
- Add a PDF bounding-box audit using `pdftotext -bbox`, `mutool`, or a similar
  tool to detect unusually dense or overlapping right-margin content.
- Add a screenshot/image-density pass for candidate pages: high ink density in
  the margin plus body text nearby should queue the page for manual review.
- Treat automatic findings as triage, not truth. Final decisions should be
  made from rendered PDF pages.

### Agent Rules Update Gate TODO

Before the final layout-rendering pass, update the shared agent rules in:

```text
/Users/VJ/GitHub/AIConfigs/projects/MLSysBook/.claude/rules/
```

Purpose: preserve the practices that worked so future Claude/Codex sessions do
not rediscover or regress them.

Timing:

1. Finish the semantic/content work first: MLSysIM source-of-truth cleanup,
   formatter helpers, output naming, prose contracts, and audits.
2. Verify the rules against the actual final corpus, not against an interim
   migration state.
3. Update `.claude/rules/` before the final PDF layout pass, so layout agents
   and future content agents use the same stable conventions.

Scope discipline:

- Write only general rules that apply cleanly to future work.
- Do not encode one-off fixes, temporary migration hacks, or chapter-specific
  decisions as global rules.
- Prefer editing existing focused rule files when the content has a natural
  home.
- Create a new rule file only when the guidance would otherwise be scattered or
  ambiguous.
- Include short examples that show the desired pattern and one clear anti-
  pattern only when it prevents common mistakes.

Likely update targets:

| Rule file | Stable content to add or update |
|---|---|
| `lego-units.md` | LEGO `LOAD`/`EXECUTE`/`GUARD`/`OUTPUT` contract; Pint quantities stay attached; scalar boundaries; registry values vs scenario assumptions. |
| `fmt.md` | Domain formatter selection, `unit=` decision rule, output naming convention, ratio vs multiple policy, service-rate formatter policy. |
| `mlsysim.md` | MLSysIM as source of truth; when to add registry fields, units, provenance, or helper formulas instead of chapter-local literals. |
| `numbers-and-math-in-prose.md` | Closed vs open output strings, prose unit ownership, multiplier spacing, precision/hidden-zero expectations. |
| `book-prose.md` / `prose-craft.md` | Reader-facing `Inputs` language, avoiding repeated "MLSysIM" mentions, floating single-sentence paragraph review. |
| `cross-references.md` | Repeated nearby table/figure reference policy and generic glossary deferral policy. |
| `margin-figures.md` / `footnotes.md` | Margin congestion policy: assumptions in body/callout, context in margin, layout cleanup deferred. |
| `glossary.md` | Avoid "complete glossary" deferrals unless the pointer is genuinely useful. |

Potential new file:

- `lego-output-formatting.md` if `fmt.md` and `lego-units.md` become too broad.
  This file would own only prose-facing LEGO exports: naming, formatter choice,
  display-unit tokens, and closed/open prose contracts.

Acceptance criteria for the rules update:

- The rules describe the final intended workflow, not the migration history.
- Examples use the final helper names and naming convention.
- The rules are consistent with lint/audit gates.
- No rule tells agents to use deprecated patterns such as `.m_as()` for prose
  output, `fmt(..., suffix="x")`, `fmt(quantity.to(unit).magnitude)`, or
  `fmt_qty(<scalar>, unit)`.
- The rules remain short enough that agents can actually follow them.

### Phase 1 Implementation Decisions

The central formatter surface should be completed before chapter edits so
chapter agents do not invent local patterns. The following decisions are now
part of the plan:

1. **FLOP count and FLOP rate are different value kinds.**
   - Use `fmt_flop_rate(quantity, unit=TFLOP / second)` for throughput.
   - Use `fmt_flops(quantity, unit=GFLOP)` for work/count.
   - Do not use `fmt(... / MILLION, suffix="MFLOP")` or ambiguous
     `*_gflops_str` names for FLOP counts.
2. **Arithmetic intensity is not a bare ratio.**
   - Use `fmt_arithmetic_intensity(quantity, unit=flop / byte)`.
   - Output names should say `flop_per_byte`, for example
     `ridge_intensity_flop_per_byte_str`.
   - `fmt_ratio` is only for dimensionless scalars such as compression or
     overcommit ratios.
3. **Integer operation throughput gets its own formatter.**
   - Use `fmt_ops_rate(quantity, unit=TOPS)` for TOPS/GOPS/OPS values that are
     not FLOP rates.
4. **Fixed-unit output names are assertions.**
   - If a name ends with `_kwh_str`, `_gb_s_str`, `_tflop_s_str`,
     `_flop_per_byte_str`, `_usd_month_str`, or `_tokens_s_str`, the formatter
     call must force that same rendered unit or scale.
   - If a formatter auto-scales, use a generic name such as
     `facility_energy_str` or `model_memory_str`.
5. **Prefer `unit=` in new or touched output calls.**
   - Positional units remain valid for compatibility.
   - New/touched code should prefer `fmt_qty(x, unit=GB / second, ...)`,
     `fmt_time(x, unit=millisecond, ...)`, and
     `fmt_qty_range(lo, hi, unit=GB, ...)`.
6. **Token scaling remains explicit for now.**
   - `fmt_params` defaults to auto scale because model-size prose almost
     always wants K/M/B/T.
   - `fmt_tokens` keeps explicit `scale=` because exact context lengths such as
     4096 tokens should not silently become `4.1K`.
   - Use `fmt_tokens(tokens, scale="T")` or `scale="B"` for corpus/training
     totals.

### Phase 1 Implementation Checkpoint — 2026-06-01

The first semantic formatter pass is implemented in the working tree:

1. **Sanctioned scalar boundary.** `fmt_magnitude(quantity, unit=...)` exists
   for algebraic displays and value/unit split tables. It is not a prose
   formatter; ordinary prose should use `fmt_qty` or a domain formatter that
   emits the unit.
2. **Euro currency.** `fmt_eur(...)` exists for euro-denominated prose such as
   `EUR 390M`, so chapter prose no longer assembles `EUR` + number + `million`
   by hand.
3. **Direct scaled `fmt(...)` burn-down.** The current scanner for
   `fmt(x / MILLION)`, `fmt(x * THOUSAND)`, and formatter `suffix=` scale
   patterns reports zero QMD hits.
4. **Scalarized OUTPUT burn-down.** The current scanner for
   `fmt(q.to(unit).magnitude)` and `.m_as(...)` inside `_str = fmt(...)`
   reports zero QMD hits.
5. **Pint-safe multipliers.** Unit-bearing ratios are normalized to a common
   unit and collapsed to dimensionless scalars before `fmt_multiple`.
6. **Currency amount/rate discipline.** A rate such as `USD/hour` is formatted
   with `per=...` when it remains a rate, or multiplied by the physical
   denominator first when computing an amount.

Verification at this checkpoint:

- `book_check_lego_prose_units.py book/quarto/contents` → 81/81 OK.
- `audit_math_canonical.py book/quarto/contents` → 0 violations.
- Focused formatter/unit pytest → 166 passed.
- Headless LEGO execution over all 81 QMD files → 932 LEGO classes, 0
  failures.
- `pre-commit run --all-files` → pass.

### Current Static Audit Queue

Read-only agent audits on 2026-06-01 found the following source queues. These
are not all bugs, but they define the migration order:

| Queue | Count / scope | Action |
|---|---:|---|
| Closed unit names using plain `fmt()` | 42 L014 hits | Convert to `fmt_qty`, domain helpers, `fmt_time`, or `fmt_usd`; keep explicit fixed-unit names. |
| Scalarized quantity passed to `fmt()` | 191 high-confidence sites | Preserve Pint quantity until output; use `fmt_*` helper with `unit=`. |
| Manual decimal scale strings | 104 candidates | Use `fmt_params`, `fmt_tokens`, `fmt_count(scale=...)`, `fmt_flops`, or `fmt_usd(scale=...)`. |
| Scalarization then reunitization | 3 hard hits | Keep units attached in EXECUTE; add registry/helper if needed. |
| Closed formatter names that do not assert rendered unit/scale | 808 heuristic candidates | Rename only when touched or when prose ambiguity is real; do not churn every stable auto-scaled name in one pass. |
| Currency/count misuse | 3 hits | Replace `fmt_count` on dollars with `fmt_usd(scale=...)`. |
| Prose duplicate units after closed formatters | 0 current hits | Keep running the prose-unit checker after each migration wave. |

Editorial/layout queues are deferred until after semantic correctness:

- repeated nearby table/figure references;
- dynamic multiplier spacing and count-vs-multiplier ambiguity;
- service-rate notation drift such as `samples/sec`, `samples/second`,
  `samples/s`;
- generic glossary deferrals;
- format-specific HTML/PDF prose splits;
- floating one-sentence paragraphs;
- margin figure plus footnote congestion.

### Migration TODO

Use this order to avoid another mixed-state cleanup:

1. Add missing domain helpers and tests in MLSysIM.
2. Run the direct `fmt(...)` audit and classify each finding by value kind.
3. Migrate high-confidence physical leftovers first:
   energy, power, memory, bandwidth, latency, emissions, and carbon intensity.
4. Migrate the missing-domain categories after helpers exist:
   FLOP work/count, arithmetic intensity, and operation throughput.
5. Migrate parameter and token count scalarizations to `fmt_params` and
   `fmt_tokens`.
6. Run prose-unit and output-name audits after each wave.
7. Only then tighten lint rules from reporting to blocking.
8. Update shared `.claude/rules/` with stable future-facing rules.
9. Run the deferred PDF layout QA pass.
