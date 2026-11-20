"""Measured GPU waiting costs and a calibrated GPU power model (opt-in).

The values come from ``engine/data/measured_gpu_energy.yaml``, which
``paper/scripts/build_measured_gpu_energy.py`` builds from the per-run tables
in ``paper/data/serverless-gpu-energy/`` (GPU board energy on Modal serverless
GPUs, read from the NVML total-energy counter). Board energy only: host,
cooling and facility energy are not included. Nothing here changes
``Engine.solve`` or any default. Callers opt in.

``measured_power(gpu, hfu, bwu)``
    P = min(TDP, P_idle + (b x HFU + c x BWU) x TDP), with P_idle the measured
    loaded-idle power of that GPU and b, c two coefficients pooled over all
    measured GPUs. HFU and BWU are the shares of peak FLOP/s and peak memory
    bandwidth a run uses.

``standby_power(gpu)``
    Measured power of a GPU kept reserved with the model loaded. It is the
    ``standby_power`` input of ``TemporalShiftModel.solve``, which charges it
    over each waiting step at that step's carbon intensity.

``coldstart_energy(gpu)``
    Measured GPU energy of one cold start. A released GPU pays it when the job
    restarts, at the chosen start step, so it is not a ``deferral_energy``
    input: ``TemporalShiftModel.solve`` charges ``deferral_energy`` at the
    release step's carbon intensity.
"""

from functools import lru_cache
from pathlib import Path

import yaml

from ..core.units import Q_

_DATA = Path(__file__).parent / "data" / "measured_gpu_energy.yaml"


@lru_cache(maxsize=None)
def load_measured() -> dict:
    """Return the measured table as stored.

    Returns
    -------
    dict
        Keys ``provenance``, ``power_model`` and ``gpus``. GPUs are keyed by
        their ``Hardware.Cloud`` registry name.
    """
    return yaml.safe_load(_DATA.read_text())


def _gpu(gpu: str) -> dict:
    data = load_measured()["gpus"]
    if gpu not in data:
        raise KeyError(f"no measurement for {gpu!r}; measured: {sorted(data)}")
    return data[gpu]


def measured_power(gpu: str, hfu: float, bwu: float):
    """Return the calibrated average board power of a run on a measured GPU.

    Parameters
    ----------
    gpu : str
        ``Hardware.Cloud`` registry name of a measured GPU, e.g. ``"H100"``.
    hfu : float
        Share of the GPU's peak FLOP/s the run uses, in [0, 1].
    bwu : float
        Share of the GPU's peak memory bandwidth the run uses, in [0, 1].

    Returns
    -------
    Quantity
        Average board power in watts, capped at the GPU's TDP.

    Notes
    -----
    b and c were fitted to 148 runs on seven GPUs. The stored hold-out errors
    are refits within those runs. They are conditional on each GPU's measured
    idle power and on HFU and BWU scaled by measured throughput, and the
    model's form was chosen after looking at the first runs. They are not a
    prospective accuracy. With HFU and BWU taken from a roofline prediction
    instead, the prediction's throughput error adds to the power error.
    """
    if not (0.0 <= hfu <= 1.0 and 0.0 <= bwu <= 1.0):
        raise ValueError("hfu and bwu must be between 0 and 1")
    g, m = _gpu(gpu), load_measured()["power_model"]
    tdp = float(g["tdp_w"])
    watts = min(tdp, float(g["loaded_idle_power_w"]) + (m["b_hfu"] * hfu + m["c_bwu"] * bwu) * tdp)
    return Q_(watts, "W")


def standby_power(gpu: str):
    """Return the measured power of a GPU kept reserved between jobs.

    Parameters
    ----------
    gpu : str
        ``Hardware.Cloud`` registry name of a measured GPU, e.g. ``"H100"``.

    Returns
    -------
    Quantity
        Mean board power in watts over 600 s with the model loaded and no
        work, starting right after a job. Pass it as ``standby_power`` to
        ``TemporalShiftModel.solve``, which charges it over each waiting step
        at that step's carbon intensity.

    Notes
    -----
    This is the extra power of keeping the GPU compared with releasing it.
    If the GPU draws this power whether or not the job waits, for example an
    owned GPU that would idle anyway, the extra is zero and ``standby_power``
    should be zero.
    """
    return Q_(float(_gpu(gpu)["reserved_idle_power_w"]), "W")


def coldstart_energy(gpu: str):
    """Return the measured GPU energy of one cold start.

    Parameters
    ----------
    gpu : str
        ``Hardware.Cloud`` registry name of a measured GPU, e.g. ``"H100"``.

    Returns
    -------
    Quantity
        Mean board energy in joules from the first NVML reading in a fresh
        container to the first generated token. GPU energy before the
        function starts, and host energy, are not observed.

    Notes
    -----
    A job that releases its GPU and starts again at step k pays this energy
    at step k, so its carbon cost is this energy times the intensity at step
    k. ``TemporalShiftModel.solve`` charges ``deferral_energy`` at the
    release step instead, so this value is not a ``deferral_energy`` input.
    """
    return Q_(float(_gpu(gpu)["coldstart_energy_j"]), "J")
