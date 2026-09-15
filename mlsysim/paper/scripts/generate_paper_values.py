#!/usr/bin/env python3
"""Compute every number and data figure in the MLSys·im paper from mlsysim.

The paper never hand-types a result. This script runs each scenario the paper
quotes against the in-repo ``mlsysim`` package and writes:

* ``paper/generated/values.tex``  one ``\\newcommand`` per number, each with a
  trailing ``% source:`` comment naming the solver call that produced it;
* ``paper/generated/values.json`` the raw (unrounded) values, the printed
  strings, and the sources, for audit and for ``--check``;
* ``paper/generated/hw_catalog.tex`` and ``paper/generated/infra_systems.tex``
  table-row macros for the appendix registry snapshots;
* ``paper/figures/roofline-crossover.{pdf,svg}`` and
  ``paper/figures/carbon-comparison.{pdf,svg}`` (matplotlib, vector);
* in-place label updates for the conceptual diagrams: any ``<text>`` element
  in ``paper/figures/*.svg`` that carries ``data-template="... {MacroName} ..."``
  has its content rewritten from the computed values, so a diagram cannot
  quote a stale count or number.

Regenerate (from the mlsysim project root, the directory holding the
``mlsysim`` package; the package is found through ``PYTHONPATH``, never a pip
install)::

    cd mlsysim && PYTHONPATH=. uv run --no-project --python 3.11 \\
        --with "pint>=0.24.4" --with "pydantic>=2.10.5" --with "numpy>=2.0" \\
        --with typer --with rich --with pyyaml --with matplotlib \\
        --with pytest --with marimo --with scipy --with ortools \\
        python paper/scripts/generate_paper_values.py

or ``make values`` in ``paper/``. The test-count step collects ``tests/``,
which imports the lab and optimizer extras.

Drift gate: ``python paper/scripts/generate_paper_values.py --check`` recomputes
every value without writing anything and exits non-zero if any value differs
from the committed ``generated/values.json`` (relative tolerance 1e-6), if the
appendix tables differ, or if a diagram label is out of sync. It skips the
figures, the sweep timing, the machine description, and the test count, which
describe the machine and environment rather than the model.

Any solver error, any infeasible result where the paper needs a feasible run,
any malformed or duplicate macro name, and any diagram template naming an
unknown value is fatal: the script exits non-zero before writing.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import pkgutil
import platform
import re
import statistics
import subprocess
import sys
import time
import warnings
from itertools import product
from pathlib import Path

SCRIPT = Path(__file__).resolve()
PAPER_DIR = SCRIPT.parent.parent
PROJECT_ROOT = PAPER_DIR.parent  # directory that holds the mlsysim package
GENERATED_DIR = PAPER_DIR / "generated"
FIGURES_DIR = PAPER_DIR / "figures"

for p in (str(PROJECT_ROOT), str(SCRIPT.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import mlsysim  # noqa: E402
import mlsysim.physics as physics_pkg  # noqa: E402
import paper_scenarios as ps  # noqa: E402
from mlsysim import Hardware, Infrastructure, Models, Systems  # noqa: E402
from mlsysim.core import provenance_catalog  # noqa: E402
from mlsysim.core.provenance import Provenance  # noqa: E402
from mlsysim.core.registry import Registry  # noqa: E402
from mlsysim.core.units import Gbps, Q_, resolve_precision  # noqa: E402
from mlsysim.engine import calibration as cal  # noqa: E402
from mlsysim.engine import solvers as solver_pkg  # noqa: E402
from mlsysim.engine import walls as walls_mod  # noqa: E402
from mlsysim.engine.solvers.base import (  # noqa: E402
    BaseOptimizer,
    BaseResolver,
    BaseSolver,
    ForwardModel,
)
from mlsysim.models.types import TransformerWorkload  # noqa: E402
from mlsysim.solvers import (  # noqa: E402
    CompressionModel,
    ContinuousBatchingModel,
    DataModel,
    DistributedModel,
    EconomicsModel,
    InferenceScalingModel,
    OrchestrationModel,
    ParallelismOptimizer,
    ReliabilityModel,
    ScalingModel,
    SensitivitySolver,
    ServingModel,
    SingleNodeModel,
    SustainabilityModel,
    SynthesisSolver,
    TopologyModel,
    TrainingMemoryModel,
    TransformationModel,
    WeightStreamingModel,
)
from mlsysim.systems.types import Fleet, Node  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO INPUTS with no registry home (validation-anchor inputs and their
# published values live in paper_scenarios.py).
# ─────────────────────────────────────────────────────────────────────────────

# Case S2: per-worker ImageNet JPEG rates (decode only; decode + augmentation)
# and the host provisioning of four data-loader workers per GPU.
S2_DECODE_IMG_PER_WORKER_S = 1_200
S2_AUGMENT_IMG_PER_WORKER_S = 850
S2_WORKERS_PER_GPU = 4
S2_SAMPLE_SIZE = Q_("150 KB")  # compressed ImageNet JPEG, scenario assumption
# Appendix Scenario C: under-provisioned 16-worker pipeline at 40k images/s.
SC_DEMAND_IMG_PER_S = 40_000
SC_WORKERS = 16
SC_ACCEL_STEP = Q_("48 ms")
# Bound printed in the conceptual-diagram footnotes ("full analysis completes
# in <0.3 s on a laptop"); generation fails if a chain exceeds it.
CLAIMED_CHAIN_BOUND_S = 0.3
CHECK_REL_TOL = 1e-6

H100 = Hardware.Cloud.H100
A100 = Hardware.Cloud.A100
B200 = Hardware.Cloud.B200


class GenerationError(RuntimeError):
    """Raised for any condition that must stop generation."""


def fail(message: str) -> None:
    raise GenerationError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def _round_half_up(x: float, nd: int) -> float:
    q = 10 ** nd
    return math.floor(abs(x) * q + 0.5) / q * (1 if x >= 0 else -1)


def fmt_fixed(x: float, nd: int = 0) -> str:
    """Fixed decimals with thousands separators: 19007.4 -> '19,007'."""
    if not math.isfinite(x):
        fail(f"refusing to format non-finite value {x!r}")
    r = _round_half_up(x, nd)
    s = f"{abs(r):,.{nd}f}"
    if r < 0 and float(s.replace(",", "")) != 0.0:
        s = "-" + s
    return s


def fmt_table(x: float, sig: int = 3) -> str:
    """Compact table number: integers >= 100, else `sig` significant figures, no trailing zeros."""
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return fmt_fixed(x, 0)
    nd = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    s = fmt_fixed(x, nd)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def tex_escape(s: str) -> str:
    for ch, rep in (("\\", r"\textbackslash{}"), ("%", r"\%"), ("$", r"\$"),
                    ("&", r"\&"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(ch, rep)
    return s


def to_tex(plain: str) -> str:
    """Printed string -> LaTeX: '{,}' separators, math minus, escaped specials."""
    s = tex_escape(plain).replace(",", "{,}")
    if re.match(r"^-\d", s):
        s = r"\ensuremath{-}" + s[1:]
    return s


class Values:
    """Ordered store of every generated value."""

    NAME_RE = re.compile(r"^[A-Za-z]+$")

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.audit: dict[str, object] = {}
        self._section = "unsectioned"

    def section(self, title: str) -> None:
        self._section = title

    def add(self, name: str, raw, plain: str, source: str, tex: str | None = None,
            volatile: bool = False) -> None:
        if not self.NAME_RE.match(name):
            fail(f"macro name {name!r} must be letters only")
        if name in self.records:
            fail(f"duplicate macro name {name!r}")
        if "\n" in source or "\n" in plain:
            fail(f"macro {name}: newline in printed value or source")
        self.records[name] = {
            "section": self._section,
            "raw": raw,
            "printed": plain,
            "tex": tex if tex is not None else to_tex(plain),
            "source": source,
            "volatile": volatile,
        }

    def num(self, name: str, raw: float, source: str, nd: int = 0, volatile: bool = False) -> None:
        self.add(name, float(raw), fmt_fixed(raw, nd), source, volatile=volatile)

    def pct(self, name: str, fraction: float, source: str, nd: int = 1) -> None:
        """Percent value printed without the sign (prose writes \\%)."""
        self.add(name, float(fraction), fmt_fixed(100.0 * fraction, nd), source)

    def sci(self, name: str, raw: float, source: str, nd: int = 2) -> None:
        exp = int(math.floor(math.log10(abs(raw))))
        mant = raw / 10 ** exp
        if float(f"{mant:.{nd}f}") >= 10:
            mant /= 10
            exp += 1
        plain = f"{mant:.{nd}f}e{exp}"
        tex = rf"\ensuremath{{{mant:.{nd}f}\times 10^{{{exp}}}}}"
        self.add(name, float(raw), plain, source, tex=tex)

    def text(self, name: str, value: str, source: str, volatile: bool = False) -> None:
        self.add(name, value, value, source, volatile=volatile)

    def plain_map(self) -> dict[str, str]:
        return {k: v["printed"] for k, v in self.records.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BATCH_WORDS = {1: "One", 8: "Eight", 32: "ThirtyTwo", 128: "OneTwentyEight", 256: "TwoFiftySix"}


def median_seconds(fn, runs: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def gb(q) -> float:
    return q.to("GB").magnitude


def single_accelerator_fleet(name: str, accelerator, fabric, datacenter=None) -> Fleet:
    node = Node(name=f"{name} node", accelerator=accelerator, accelerators_per_node=1,
                intra_node_bw=accelerator.nvlink.bandwidth_per_direction, nics_per_node=1)
    return Fleet(name=name, node=node, count=1, fabric=fabric, datacenter=datacenter)


# ─────────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────────

def section_registry(V: Values) -> None:
    V.section("Registry facts quoted in prose")
    V.num("RegHhPeakTflops", H100.compute.peak_flops.m_as("TFLOPs/s"), "Hardware.Cloud.H100.compute.peak_flops (FP16)")
    V.num("RegHhBandwidthTBs", H100.memory.bandwidth.m_as("TB/s"), "Hardware.Cloud.H100.memory.bandwidth", nd=2)
    V.num("RegHhCapacityGiB", H100.memory.capacity.m_as("GiB"), "Hardware.Cloud.H100.memory.capacity")
    V.num("RegHhCapacityGB", H100.memory.capacity.m_as("GB"), "Hardware.Cloud.H100.memory.capacity in decimal GB", nd=1)
    V.num("RegHhUnitCostK", H100.unit_cost.m_as("USD") / 1e3, "Hardware.Cloud.H100.unit_cost / 1e3")
    V.num("RegHhTdpW", H100.tdp.m_as("W"), "Hardware.Cloud.H100.tdp")
    V.num("RegHhRidgePoint", (H100.compute.peak_flops / H100.memory.bandwidth).m_as("flop/byte"),
          "Hardware.Cloud.H100.compute.peak_flops / memory.bandwidth")
    V.num("RegHhNvlinkGBs", H100.nvlink.bandwidth.m_as("GB/s"), "Hardware.Cloud.H100.nvlink.bandwidth (bidirectional total)")
    V.num("RegHhNvlinkPerDirectionGBs", H100.nvlink.bandwidth_per_direction.m_as("GB/s"),
          "Hardware.Cloud.H100.nvlink.bandwidth_per_direction")
    V.num("RegAhPeakTflops", A100.compute.peak_flops.m_as("TFLOPs/s"), "Hardware.Cloud.A100.compute.peak_flops")
    V.num("RegAhBandwidthTBs", A100.memory.bandwidth.m_as("TB/s"), "Hardware.Cloud.A100.memory.bandwidth", nd=2)
    V.num("RegBhBandwidthTBs", B200.memory.bandwidth.m_as("TB/s"), "Hardware.Cloud.B200.memory.bandwidth", nd=1)
    V.num("RegBhCapacityGiB", B200.memory.capacity.m_as("GiB"), "Hardware.Cloud.B200.memory.capacity")
    V.num("RegHhOverAhBandwidthRatio", (H100.memory.bandwidth / A100.memory.bandwidth).m_as(""),
          "Hardware.Cloud.H100.memory.bandwidth / Hardware.Cloud.A100.memory.bandwidth", nd=1)
    V.num("RegHhOverAhFlopsRatio", (H100.compute.peak_flops / A100.compute.peak_flops).m_as(""),
          "Hardware.Cloud.H100.compute.peak_flops / Hardware.Cloud.A100.compute.peak_flops", nd=1)
    ndr, hdr = Systems.Fabrics.InfiniBand_NDR.bandwidth, Systems.Fabrics.InfiniBand_HDR.bandwidth
    V.num("RegIbNdrGbps", ndr.m_as(Gbps), "Systems.Fabrics.InfiniBand_NDR.bandwidth")
    V.num("RegIbNdrGBs", ndr.m_as("GB/s"), "Systems.Fabrics.InfiniBand_NDR.bandwidth in GB/s")
    V.num("RegIbHdrGbps", hdr.m_as(Gbps), "Systems.Fabrics.InfiniBand_HDR.bandwidth")
    V.num("RegNvlinkPerDirectionOverNdr", (H100.nvlink.bandwidth_per_direction / ndr).m_as(""),
          "Hardware.Cloud.H100.nvlink.bandwidth_per_direction / Systems.Fabrics.InfiniBand_NDR.bandwidth")
    for key, label in (("Iowa", "Iowa"), ("Quebec", "Quebec"), ("US_Avg", "UsAvg")):
        grid = getattr(Infrastructure.Grids, key)
        V.num(f"RegGrid{label}CarbonIntensity", grid.carbon_intensity_g_kwh, f"Infrastructure.Grids.{key}.carbon_intensity_g_kwh")
        V.num(f"RegGrid{label}Pue", grid.pue, f"Infrastructure.Grids.{key}.pue", nd=2)
        V.num(f"RegGrid{label}Wue", grid.wue, f"Infrastructure.Grids.{key}.wue", nd=1)
    V.text("RegGridIowaProvenanceKind", Infrastructure.Grids.Iowa.metadata.provenance.kind.value,
           "Infrastructure.Grids.Iowa.metadata.provenance.kind")
    grids = Infrastructure.Grids.list()
    cis = [g.carbon_intensity_g_kwh for g in grids]
    V.num("CountGridProfiles", len(grids), "len(Infrastructure.Grids.list())")
    V.num("RegGridCarbonSpread", max(cis) / min(cis), "max/min carbon_intensity_g_kwh over Infrastructure.Grids.list()")
    V.num("CountFabrics", len(Systems.Fabrics.list()), "len(Systems.Fabrics.list())")
    V.num("CountClusters", len(Systems.Clusters.list()), "len(Systems.Clusters.list())")
    V.num("CountNodes", len(Systems.Nodes.list()), "len(Systems.Nodes.list())")
    V.num("RegTokensPerReasoningStep", cal.TOKENS_PER_REASONING_STEP, "mlsysim.engine.calibration.TOKENS_PER_REASONING_STEP")


def section_counts(V: Values, check: bool) -> None:
    V.section("Counts computed from code")
    V.num("CountWalls", len(walls_mod.ALL_WALLS), "len(mlsysim.engine.walls.ALL_WALLS)")
    V.num("CountDomains", len(walls_mod.Domain), "len(mlsysim.engine.walls.Domain)")
    for dom, label in ((walls_mod.Domain.NODE, "Node"), (walls_mod.Domain.DATA, "Data"),
                       (walls_mod.Domain.ALGORITHM, "Algorithm"), (walls_mod.Domain.FLEET, "Fleet"),
                       (walls_mod.Domain.OPERATIONS, "Operations"), (walls_mod.Domain.ANALYSIS, "Analysis")):
        V.num(f"CountWalls{label}", len(walls_mod.walls_in_domain(dom)), f"len(walls_in_domain(Domain.{dom.name}))")

    tiers = {"Models": [], "Solvers": [], "Optimizers": []}
    for name in solver_pkg.__all__:
        cls = getattr(solver_pkg, name)
        if cls in (BaseResolver, ForwardModel, BaseSolver, BaseOptimizer):
            continue
        if issubclass(cls, BaseOptimizer):
            tiers["Optimizers"].append(name)
        elif issubclass(cls, BaseSolver):
            tiers["Solvers"].append(name)
        elif issubclass(cls, ForwardModel):
            tiers["Models"].append(name)
        else:
            fail(f"resolver {name} has no tier base class")
    V.num("CountResolvers", sum(len(v) for v in tiers.values()),
          "classes in mlsysim.engine.solvers.__all__ minus the four base classes")
    bases = {"Models": "ForwardModel", "Solvers": "BaseSolver", "Optimizers": "BaseOptimizer"}
    for tier, names in tiers.items():
        V.num(f"CountResolver{tier}", len(names), f"subclasses of {bases[tier]} in mlsysim.engine.solvers.__all__")
    V.num("CountWallResolvers", len({w.resolver_name for w in walls_mod.ALL_WALLS}), "distinct Wall.resolver_name over ALL_WALLS")
    V.audit["resolver_tiers"] = tiers

    V.num("CountHardwareDevices", len(Hardware.list()), "len(Hardware.list()) (Tech classes excluded)")
    for tier in ("Cloud", "Workstation", "Edge", "Mobile", "Tiny"):
        V.num(f"CountHardware{tier}", len(getattr(Hardware, tier).list()), f"len(Hardware.{tier}.list())")
    V.num("CountModels", len(Models.list()), "len(Models.list())")
    families = [n for n in dir(Models) if isinstance(getattr(Models, n), type) and issubclass(getattr(Models, n), Registry)]
    V.num("CountModelFamilies", len(families), "Registry subclasses on Models")
    provs = {v.id for v in vars(provenance_catalog).values() if isinstance(v, Provenance)}
    V.num("CountProvenanceRecords", len(provs), "distinct Provenance ids defined in mlsysim.core.provenance_catalog")

    registries = sorted(n for n in dir(mlsysim) if isinstance(getattr(mlsysim, n), type)
                        and issubclass(getattr(mlsysim, n), Registry))
    require(hasattr(mlsysim, "Scenarios"), "mlsysim.Scenarios missing")
    registries.append("Scenarios")
    auxiliary = ["Literature", "ReferenceStats"]
    for name in auxiliary + ["Agents", "Embodied"]:
        require(name in registries, f"registry {name} missing")
    curated = [r for r in registries if r not in auxiliary]
    V.num("CountRegistriesTotal", len(registries), "top-level Registry subclasses on mlsysim, plus Scenarios")
    V.num("CountRegistriesCurated", len(curated), "CountRegistriesTotal minus auxiliary Literature and ReferenceStats")
    V.num("CountRegistriesAuxiliary", len(auxiliary), "Literature and ReferenceStats")
    V.text("ListRegistriesCurated", ", ".join(curated), "curated registry names")
    V.audit["registries"] = {"curated": curated, "auxiliary": auxiliary}

    public = [m.name for m in pkgutil.iter_modules(physics_pkg.__path__) if not m.name.startswith("_")]
    supporting = ["constants", "quantities"]
    for s in supporting:
        require(s in public, f"physics.{s} missing")
    domain = [m for m in public if m not in supporting]
    V.num("CountPhysicsModules", len(public), "public modules in mlsysim.physics")
    V.num("CountPhysicsDomainModules", len(domain), "public mlsysim.physics modules minus constants and quantities")
    V.num("CountPhysicsSupportingModules", len(supporting), "mlsysim.physics.constants and quantities")
    V.text("ListPhysicsDomainModules", ", ".join(domain), "domain physics module names")

    if check:
        return  # the collected count depends on which optional extras are installed
    env = dict(os.environ, PYTHONPATH=str(PROJECT_ROOT))
    proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        fail("pytest --collect-only failed (install pytest and the optional test dependencies; "
             "see paper/Makefile):\n" + proc.stdout[-3000:] + proc.stderr[-2000:])
    m = re.search(r"^(\d+) tests? collected", proc.stdout, re.M)
    if m:
        n_tests = int(m.group(1))
    else:
        per_file = [int(x) for x in re.findall(r"^tests/\S+\.py: (\d+)$", proc.stdout, re.M)]
        require(bool(per_file), "could not parse pytest collection output")
        n_tests = sum(per_file)
    V.num("CountTests", n_tests, "python -m pytest --collect-only -q over tests/ with marimo, scipy, ortools installed", volatile=True)


def section_timing(V: Values) -> None:
    V.section("Sweep timing and machine (volatile: describes the machine that ran this)")
    hardware = [Hardware.Cloud.T4, Hardware.Cloud.V100, A100, H100, Hardware.Cloud.H200, B200,
                Hardware.Cloud.Gaudi2, Hardware.Cloud.Gaudi3, Hardware.Cloud.MI250X, Hardware.Cloud.MI300X]
    models = [Models.Vision.ResNet50, Models.Vision.MobileNetV2, Models.Vision.EfficientNetB0,
              Models.Language.BERT_Base, Models.Language.GPT2, Models.Language.Llama2_7B,
              Models.Language.Llama3_8B, Models.Language.Llama2_70B, Models.Language.Llama3_70B,
              Models.Language.GPT3]
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    configs = list(product(hardware, models, batches))
    require(len(configs) == 1000, "sweep must be 1,000 configurations")
    solver = SingleNodeModel()

    def sweep():
        for hw, model, b in configs:
            solver.solve(model, hw, batch_size=b, precision="fp16")

    med = median_seconds(sweep, runs=5, warmup=1)
    V.num("SweepTimingConfigs", len(configs), "10 Cloud accelerators x 10 models x 10 batch sizes, SingleNodeModel().solve", volatile=True)
    V.num("SweepTimingMedianMs", med * 1e3, "median of 5 warm runs of the 1,000-config SingleNodeModel sweep (ms)", volatile=True)
    V.num("SweepTimingPerConfigUs", med * 1e6 / len(configs), "SweepTimingMedianMs / 1,000 configs (microseconds)", volatile=True)

    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        os_name = f"macOS {platform.mac_ver()[0]}"
    else:
        os_name = f"{platform.system()} {platform.release()}"
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    V.text("MachineCpu", cpu, "sysctl machdep.cpu.brand_string or /proc/cpuinfo", volatile=True)
    V.num("MachineCores", os.cpu_count() or 0, "os.cpu_count()", volatile=True)
    V.text("MachineOs", os_name, "platform", volatile=True)
    V.text("MachinePython", platform.python_version(), "platform.python_version()", volatile=True)


def section_anchors(V: Values) -> None:
    V.section("Validation anchors (configurations in paper_scenarios.py)")
    a1 = ps.anchor_one()
    require(a1["feasible"], "A1: ResNet-50 training at batch 256 must fit on one A100")
    V.num("AnchorOneEta", ps.A1_CONFIG["efficiency"], "paper_scenarios.A1_CONFIG", nd=2)
    V.num("AnchorOnePerGpuBatch", ps.A1_CONFIG["batch_size"], "paper_scenarios.A1_CONFIG")
    V.num("AnchorOneGlobalBatch", ps.A1_GPUS * ps.A1_CONFIG["batch_size"], "8 GPUs x per-GPU batch")
    V.num("AnchorOneSimPerGpu", a1["per_gpu"], a1["source"] + ".throughput")
    V.num("AnchorOneSimThroughput", a1["node"], a1["source"] + ".throughput x 8")
    V.num("AnchorOneReported", a1["reported"], "paper_scenarios.A1_REPORTED_IMG_PER_S (NVIDIA DeepLearningExamples)")
    V.pct("AnchorOneErrorPct", a1["rel_error"], "|AnchorOneSimThroughput - AnchorOneReported| / AnchorOneReported")
    V.text("AnchorOneErrorDirection", "over" if a1["node"] > a1["reported"] else "under", "sign of predicted - reported")
    V.text("AnchorOneBottleneck", a1["profile"].bottleneck, a1["source"] + ".bottleneck")

    a2 = ps.anchor_two()
    V.num("AnchorTwoWeightsGB", gb(a2["weights"]), "Models.Language.Llama2_70B.size_in_bytes(FP16)")
    V.num("AnchorTwoTpDegree", a2["tp"], "paper_scenarios.A2_TP_DEGREE")
    V.num("AnchorTwoDecodeFloorMs", a2["floor_ms"], a2["source"], nd=1)

    a3 = ps.anchor_three()
    d, fleet = a3["result"], a3["fleet"]
    src = a3["source"]
    V.num("AnchorThreeGpus", fleet.total_accelerators, "Fleet.total_accelerators")
    V.num("AnchorThreeNodes", fleet.count, "Fleet.count")
    V.num("AnchorThreeTp", d.parallelism["tp"], src + ".parallelism['tp']")
    V.num("AnchorThreePp", d.parallelism["pp"], src + ".parallelism['pp']")
    V.num("AnchorThreeDp", d.parallelism["dp"], src + ".parallelism['dp']")
    V.num("AnchorThreeSeqLen", ps.A3_CONFIG["seq_len"], "paper_scenarios.A3_CONFIG")
    V.num("AnchorThreeBatch", ps.A3_CONFIG["batch_size"], "paper_scenarios.A3_CONFIG (sequences per step)")
    V.num("AnchorThreeMicrobatches", ps.A3_CONFIG["microbatch_count"], "paper_scenarios.A3_CONFIG")
    V.num("AnchorThreeFabricGbps", fleet.fabric.bandwidth.m_as(Gbps), "Systems.Fabrics.InfiniBand_NDR.bandwidth")
    V.num("AnchorThreeOverlap", ps.A3_CONFIG["overlap_efficiency"], "paper_scenarios.A3_CONFIG", nd=2)
    V.num("AnchorThreeEta", ps.A3_ETA, "paper_scenarios.A3_ETA", nd=2)
    V.pct("AnchorThreeScalingEffPct", d.scaling_efficiency, src + ".scaling_efficiency")
    V.pct("AnchorThreeMfuPct", a3["mfu"], src + ".scaling_efficiency x .node_profile.mfu")
    V.num("AnchorThreeZeroStage", ps.A3_CONFIG["zero_stage"], "paper_scenarios.A3_CONFIG")
    V.num("AnchorThreeMemoryGB", a3["memory_gb"], src + ".node_profile.memory_footprint (GB)", nd=1)
    V.text("AnchorThreeMemoryFits", "fits" if a3["feasible"] else "exceeds", src + ".node_profile.feasible")
    V.num("AnchorThreeCapacityGB", Hardware.Cloud.H100.memory.capacity.m_as("GB"), "Hardware.Cloud.H100.memory.capacity (GB)", nd=1)
    V.num("AnchorThreeComputeSeconds", d.node_profile.latency.m_as("s"), src + ".node_profile.latency", nd=1)
    V.num("AnchorThreeBubbleSeconds", d.pipeline_bubble_latency.m_as("s"), src + ".pipeline_bubble_latency", nd=1)
    V.pct("AnchorThreeBubbleFractionPct", d.bubble_fraction, src + ".bubble_fraction")
    V.pct("AnchorThreeBubbleShareOfStepPct", (d.pipeline_bubble_latency / d.step_latency_total).m_as(""),
          src + ".pipeline_bubble_latency / .step_latency_total")
    V.num("AnchorThreeStepSeconds", d.step_latency_total.m_as("s"), src + ".step_latency_total", nd=1)
    V.pct("AnchorThreeReportedMfuPct", a3["reported"], "paper_scenarios.A3_REPORTED_MFU (Llama 3 herd paper)", nd=0)
    V.pct("AnchorThreeErrorPct", a3["rel_error"], "|MFU - reported| / reported")
    V.num("AnchorThreeErrorPoints", a3["abs_error_points"], "|MFU - reported| in percentage points", nd=1)
    V.audit["A3_node_profile"] = {"feasible": d.node_profile.feasible, "bottleneck": d.node_profile.bottleneck}

    a4 = ps.anchor_four()
    d4 = a4["result"]
    V.num("AnchorFourChips", a4["fleet"].total_accelerators, "Fleet.total_accelerators")
    V.num("AnchorFourChipsPerHost", ps.A4_CHIPS_PER_HOST, "paper_scenarios.A4_CHIPS_PER_HOST")
    V.num("AnchorFourTp", ps.A4_CONFIG["tp_size"], "paper_scenarios.A4_CONFIG")
    V.num("AnchorFourTflops", Hardware.Cloud.TPUv4.compute.peak_flops.m_as("TFLOPs/s"), "Hardware.Cloud.TPUv4.compute.peak_flops")
    V.num("AnchorFourPodChips", ps.A4_POD_CHIPS, "paper_scenarios.A4_POD_CHIPS (PaLM Sec. 4)")
    V.num("AnchorFourIciGBs", Hardware.Cloud.TPUv4.nvlink.bandwidth_per_direction.m_as("GB/s"),
          "Hardware.Cloud.TPUv4.nvlink.bandwidth_per_direction (Jouppi et al. 2023, Table 4)")
    V.num("AnchorFourDcnBurstTbps", ps.A4_DCN_BURST_GBS * 8 / 1000, "paper_scenarios.A4_DCN_BURST_GBS (PaLM Sec. 4)")
    V.num("AnchorFourDcnPerChipGBs", ps.A4_DCN_PER_CHIP.m_as("GB/s"), "PaLM cross-pod burst / 6,144 chips (GB/s)", nd=2)
    V.num("AnchorFourBatch", ps.A4_CONFIG["batch_size"], "paper_scenarios.A4_CONFIG")
    V.num("AnchorFourEta", ps.A4_ETA, "paper_scenarios.A4_ETA", nd=2)
    V.pct("AnchorFourScalingEffPct", d4.scaling_efficiency, a4["source"] + ".scaling_efficiency")
    V.pct("AnchorFourMfuPct", a4["mfu"], a4["source"] + ".scaling_efficiency x .node_profile.mfu")
    V.num("AnchorFourDp", d4.parallelism["dp"], a4["source"] + ".parallelism['dp']")
    V.num("AnchorFourZeroStage", ps.A4_CONFIG["zero_stage"], "paper_scenarios.A4_CONFIG")
    V.num("AnchorFourSeqLen", ps.A4_CONFIG["seq_len"], "paper_scenarios.A4_CONFIG")
    V.num("AnchorFourMemoryGB", a4["memory_gb"], a4["source"] + ".node_profile.memory_footprint (GB)", nd=1)
    V.text("AnchorFourMemoryFits", "fits" if a4["feasible"] else "exceeds", a4["source"] + ".node_profile.feasible")
    V.pct("AnchorFourReportedMfuPct", a4["reported"], "paper_scenarios.A4_REPORTED_MFU (PaLM)")
    V.pct("AnchorFourReportedHfuPct", ps.A4_REPORTED_HFU, "paper_scenarios.A4_REPORTED_HFU (PaLM Sec. 4.1)")
    V.pct("AnchorFourReplicaMfuPct", d4.node_profile.mfu, a4["source"] + ".node_profile.mfu (excludes rematerialization)")
    V.pct("AnchorFourErrorPct", a4["rel_error"], "|MFU - reported| / reported")
    V.num("AnchorFourErrorPoints", a4["abs_error_points"], "|MFU - reported| in percentage points", nd=1)

    a5 = ps.anchor_five()
    V.sci("AnchorFiveComputeBudget", a5["budget"], "Literature.Chinchilla.ComputeConstant x 70e9 x 1.4e12 (FLOPs)")
    V.num("AnchorFiveOptimalParamsB", a5["p_star"] / 1e9, a5["source"] + ".optimal_parameters / 1e9", nd=1)
    V.num("AnchorFiveOptimalTokensT", a5["d_star"] / 1e12, a5["source"] + ".optimal_tokens / 1e12", nd=2)
    V.pct("AnchorFiveErrorPct", a5["rel_error"], "|P* - 70B| / 70B", nd=2)
    lit = mlsysim.Literature.Chinchilla
    s24 = ScalingModel().solve(compute_budget=Q_(1e24, "flop"))
    V.num("WallElevenPstarB", s24.optimal_parameters.m_as("count") / 1e9, "ScalingModel().solve(Q_(1e24,'flop')).optimal_parameters / 1e9")
    V.num("WallElevenDstarT", s24.optimal_tokens.m_as("count") / 1e12, "ScalingModel().solve(Q_(1e24,'flop')).optimal_tokens / 1e12", nd=1)
    V.num("WallElevenTokensPerParam", float(lit.TokensPerParam), "Literature.Chinchilla.TokensPerParam")

    a6 = ps.anchor_six()
    V.num("AnchorSixEnergyMWh", a6["energy_mwh"], "Models.Language.GPT3.training_energy_mwh")
    V.num("AnchorSixGridCi", a6["ci"], "Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh")
    V.num("AnchorSixCarbonTonnes", a6["tonnes"], a6["source"], nd=1)
    V.num("AnchorSixReportedTonnes", a6["reported"], "paper_scenarios.A6_REPORTED_TONNES (Patterson et al. 2021)")
    V.pct("AnchorSixErrorPct", a6["rel_error"], "|computed - reported| / reported", nd=2)

    a7 = ps.anchor_seven(fleet)
    opt, wide, second = a7["result"], a7["wide"], a7["runner_up"]
    src = a7["source"]
    V.num("AnchorSevenTp", opt.best_config["tp"], src + ".best_config")
    V.num("AnchorSevenPp", opt.best_config["pp"], src + ".best_config")
    V.num("AnchorSevenDp", opt.best_config["dp"], src + ".best_config")
    V.num("AnchorSevenCandidates", opt.total_searched, src + ".total_searched")
    V.pct("AnchorSevenBestMfuPct", opt.best_mfu, src + ".best_mfu")
    V.pct("AnchorSevenRunnerUpMfuPct", second["mfu"], src + ".top_candidates[1]['mfu']")
    V.num("AnchorSevenRunnerUpTp", second["config"]["tp"], src + ".top_candidates[1]")
    V.num("AnchorSevenRunnerUpPp", second["config"]["pp"], src + ".top_candidates[1]")
    V.num("AnchorSevenRunnerUpDp", second["config"]["dp"], src + ".top_candidates[1]")
    V.pct("AnchorSevenRunnerUpMarginPct", a7["margin_rel"], "(best MFU - runner-up MFU) / best MFU")
    V.num("AnchorSevenRunnerUpMarginPoints", a7["margin_points"], "best MFU - runner-up MFU, percentage points", nd=2)
    V.num("AnchorSevenWideMaxTp", ps.A7_WIDE_MAX_TP, "paper_scenarios.A7_WIDE_MAX_TP")
    V.num("AnchorSevenWideTp", wide.best_config["tp"], src + " with max_tp=64 .best_config")
    V.num("AnchorSevenWidePp", wide.best_config["pp"], src + " with max_tp=64 .best_config")
    V.num("AnchorSevenWideDp", wide.best_config["dp"], src + " with max_tp=64 .best_config")


def section_walls(V: Values) -> None:
    V.section("Walls")
    llama70, llama8 = Models.Language.Llama3_70B, Models.Language.Llama3_8B

    # Wall 4: attention share of prefill FLOPs, recovered from ServingModel's TTFT.
    peak16 = H100.compute.precision_flops.get("fp16", H100.compute.peak_flops)
    for seq, word in ((8192, "EightK"), (32768, "ThirtyTwoK")):
        eta = 0.5
        sv = ServingModel().solve(llama70, H100, seq_len=seq, batch_size=1, precision="fp16", efficiency=eta)
        ops = ((sv.ttft - H100.dispatch_tax) * peak16 * eta).m_as("flop")
        share = 1.0 - 2 * llama70.parameters.m_as("count") * seq / ops
        V.pct(f"WallFourAttnShare{word}Pct", share,
              f"1 - 2PS / prefill_ops, prefill_ops = (ServingModel().solve(Llama3_70B, H100, seq_len={seq}).ttft - dispatch_tax) x FP16 peak x 0.5", nd=0)
    V.num("WallFourAttnSeqShort", 8192, "scenario input")
    V.num("WallFourAttnSeqLong", 32768, "scenario input")

    # Wall 5: static max-length reservation vs PagedAttention.
    cap = 100_000
    for page, word in ((16, ""), (2048, "LargePage")):
        cb = ContinuousBatchingModel().solve(llama8, H100, max_seq_len=8192, mean_request_tokens=2048,
                                             max_batch_size=cap, page_size=page, precision="fp16")
        require(cb.feasible, f"W5: continuous batching infeasible at page {page}")
        require(cb.max_active_requests < cap and cb.static_max_active_requests < cap, "W5: max_batch_size cap binds")
        src = (f"ContinuousBatchingModel().solve(Models.Language.Llama3_8B, Hardware.Cloud.H100, max_seq_len=8192, "
               f"mean_request_tokens=2048, max_batch_size=100000, page_size={page})")
        if not word:
            V.num("WallFiveStaticUsers", cb.static_max_active_requests, src + ".static_max_active_requests")
            V.pct("WallFiveStaticWastePct", cb.static_internal_fragmentation, src + ".static_internal_fragmentation", nd=0)
            V.num("WallFiveStaticTokensPerS", cb.static_throughput_tokens_per_sec, src + ".static_throughput_tokens_per_sec")
        V.num(f"WallFive{word}PagedUsers", cb.max_active_requests, src + ".max_active_requests")
        V.pct(f"WallFive{word}PagedWastePct", cb.paged_internal_fragmentation, src + ".paged_internal_fragmentation")
        V.num(f"WallFive{word}PagedTokensPerS", cb.throughput_tokens_per_sec, src + ".throughput_tokens_per_sec")
        V.num(f"WallFive{word}Speedup", cb.speedup_vs_static, src + ".speedup_vs_static", nd=2)
    V.num("WallFiveMaxSeqLen", 8192, "scenario input")
    V.num("WallFiveMeanTokens", 2048, "scenario input")
    V.num("WallFivePageSize", 16, "scenario input (vLLM default)")
    V.num("WallFiveLargePageSize", 2048, "scenario input")

    # Wall 10: bisection fraction.
    ndr = Systems.Fabrics.InfiniBand_NDR
    for n, word in ((27, "Small"), (64, "Large")):
        t = TopologyModel().solve(ndr, topology="torus_3d", num_nodes=n)
        src = f"TopologyModel().solve(Systems.Fabrics.InfiniBand_NDR, topology='torus_3d', num_nodes={n})"
        V.num(f"WallTenTorusNodes{word}", n, "scenario input")
        V.num(f"WallTenTorusBeta{word}", t.bisection_bw_fraction, src + ".bisection_bw_fraction", nd=2)
        V.num(f"WallTenTorusEffGbps{word}", t.effective_bw.m_as(Gbps), src + ".effective_bw")
    ft = TopologyModel().solve(ndr, topology="fat_tree", num_nodes=27)
    df = TopologyModel().solve(ndr, topology="dragonfly", num_nodes=27)
    small = TopologyModel().solve(ndr, topology="torus_3d", num_nodes=27)
    V.num("WallTenLinkGbps", ndr.bandwidth.m_as(Gbps), "Systems.Fabrics.InfiniBand_NDR.bandwidth")
    V.num("WallTenFatTreeBeta", ft.bisection_bw_fraction, "TopologyModel().solve(InfiniBand_NDR, 'fat_tree').bisection_bw_fraction", nd=1)
    V.num("WallTenDragonflyBeta", df.bisection_bw_fraction, "TopologyModel().solve(InfiniBand_NDR, 'dragonfly').bisection_bw_fraction", nd=2)
    V.num("WallTenFatTreeAdvantageSmall", ft.bisection_bw_fraction / small.bisection_bw_fraction, "fat-tree beta / 27-node torus beta", nd=1)

    # Wall 13: INT4 (follows current code; a CompressionModel fix is in progress elsewhere).
    comp = CompressionModel().solve(llama70, H100, method="quantization", target_bitwidth=4)
    src = "CompressionModel().solve(Models.Language.Llama3_70B, Hardware.Cloud.H100, method='quantization', target_bitwidth=4)"
    V.num("WallThirteenIntFourCompressionRatio", comp.compression_ratio, src + ".compression_ratio", nd=1)
    V.num("WallThirteenIntFourSpeedup", comp.inference_speedup, src + ".inference_speedup", nd=1)
    V.pct("WallThirteenIntFourMemorySavingsPct", comp.memory_savings_pct / 100.0, src + ".memory_savings_pct")

    # Wall 16: M/D/1.
    fleet = Systems.Clusters.Research_256
    hi = OrchestrationModel().solve(fleet, arrival_rate_jobs_per_day=0.9, avg_job_duration_days=1.0)
    lo = OrchestrationModel().solve(fleet, arrival_rate_jobs_per_day=0.5, avg_job_duration_days=1.0)
    V.num("WallSixteenRhoHigh", hi.cluster_utilization, "OrchestrationModel().solve(Research_256, 0.9 jobs/day, 1 day).cluster_utilization", nd=1)
    V.num("WallSixteenRhoLow", lo.cluster_utilization, "OrchestrationModel().solve(Research_256, 0.5 jobs/day, 1 day).cluster_utilization", nd=1)
    V.num("WallSixteenWaitRatio", (hi.avg_wait_time_days / lo.avg_wait_time_days).m_as(""),
          "OrchestrationModel avg_wait_time_days at rho 0.9 / at rho 0.5", nd=1)
    V.num("WallSixteenQueueRatio", hi.avg_queue_length / lo.avg_queue_length,
          "OrchestrationModel avg_queue_length at rho 0.9 / at rho 0.5", nd=1)

    # Wall 17: capital.
    f1k = Systems.Clusters.Training_1K
    econ = EconomicsModel().solve(f1k, duration_days=30)
    src = "EconomicsModel().solve(Systems.Clusters.Training_1K, duration_days=30)"
    V.num("WallSeventeenGpus", f1k.total_accelerators, "Systems.Clusters.Training_1K.total_accelerators")
    V.num("WallSeventeenClusterCapexM", H100.unit_cost.m_as("USD") * f1k.total_accelerators / 1e6,
          "Hardware.Cloud.H100.unit_cost x Training_1K.total_accelerators / 1e6", nd=1)
    V.num("WallSeventeenDays", 30, "scenario input")
    V.num("WallSeventeenAmortYears", 3, "EconomicsModel default amortization_years")
    V.num("WallSeventeenRunCapexK", econ.capex_usd / 1e3, src + ".capex_usd / 1e3")
    V.num("WallSeventeenEnergyK", econ.opex_energy_usd / 1e3, src + ".opex_energy_usd / 1e3", nd=1)
    V.num("WallSeventeenMaintenanceK", econ.opex_maintenance_usd / 1e3, src + ".opex_maintenance_usd / 1e3")
    V.num("WallSeventeenTcoK", econ.tco_usd / 1e3, src + ".tco_usd / 1e3")
    V.pct("WallSeventeenElectricitySharePct", econ.opex_energy_usd / econ.tco_usd, src + ".opex_energy_usd / .tco_usd")
    V.num("WallSeventeenKwhPrice", Infrastructure.Pricing.Cloud.ElectricityPerKwh.rate.magnitude,
          "Infrastructure.Pricing.Cloud.ElectricityPerKwh.rate (USD/kWh)", nd=2)

    # Wall 21 (and Case R2): sensitivity.
    sens = SensitivitySolver().solve(llama8, A100)
    src = "SensitivitySolver().solve(Models.Language.Llama3_8B, Hardware.Cloud.A100)"
    V.num("WallTwentyOneBandwidthSensitivity", sens.sensitivities["memory_bandwidth"], src + ".sensitivities['memory_bandwidth']", nd=3)
    V.pct("WallTwentyOneBandwidthCutPct", -sens.sensitivities["memory_bandwidth"], "-(memory_bandwidth sensitivity)")
    V.num("WallTwentyOneFlopsSensitivity", sens.sensitivities["peak_flops"], src + ".sensitivities['peak_flops']", nd=1)
    V.text("WallTwentyOneBinding", sens.binding_constraint, src + ".binding_constraint")
    V.num("WallTwentyOnePerturbationPct", sens.perturbation_pct, src + ".perturbation_pct")

    # Wall 22: synthesis.
    syn = SynthesisSolver().solve(llama70, Q_("50 ms"), precision="fp16")
    src = "SynthesisSolver().solve(Models.Language.Llama3_70B, Q_('50 ms'), precision='fp16')"
    ratio = (syn.required_bw / A100.memory.bandwidth).m_as("")
    V.num("WallTwentyTwoTargetMs", 50, "scenario input")
    V.num("WallTwentyTwoWeightsGB", syn.model_size.m_as("GB"), src + ".model_size", nd=1)
    V.num("WallTwentyTwoRequiredBwGBs", syn.required_bw.m_as("GB/s"), src + ".required_bw")
    V.num("WallTwentyTwoRequiredBwTBs", syn.required_bw.m_as("TB/s"), src + ".required_bw", nd=2)
    V.num("WallTwentyTwoAhMultiple", ratio, src + ".required_bw / Hardware.Cloud.A100.memory.bandwidth", nd=2)
    V.num("WallTwentyTwoMinAhDevices", math.ceil(ratio), "ceil(WallTwentyTwoAhMultiple)")


def roofline_sweep(model, hw, eta, batches):
    return [SingleNodeModel().solve(model, hw, batch_size=b, precision="fp16", efficiency=eta) for b in batches]


def section_instructor(V: Values, check: bool) -> dict:
    V.section("Case I1: Roofline batch sweep (ResNet-50 on H100)")
    eta, batches, model = 0.5, [1, 8, 32, 128, 256], Models.Vision.ResNet50
    profiles = roofline_sweep(model, H100, eta, batches)
    peak = H100.compute.precision_flops.get("fp16", H100.compute.peak_flops)
    src = "SingleNodeModel().solve(Models.Vision.ResNet50, Hardware.Cloud.H100, batch_size={b}, precision='fp16', efficiency=0.5)"
    for b, p in zip(batches, profiles):
        require(p.feasible, f"I1: ResNet-50 batch {b} must fit on H100")
        w = BATCH_WORDS[b]
        V.num(f"CaseIOneAiBatch{w}", p.arithmetic_intensity.m_as("flop/byte"), src.format(b=b) + ".arithmetic_intensity")
        V.num(f"CaseIOneMfuBatch{w}", p.mfu, src.format(b=b) + ".mfu", nd=2)
        V.text(f"CaseIOneBottleneckBatch{w}", p.bottleneck, src.format(b=b) + ".bottleneck")
        V.num(f"CaseIOneAttainedTflopsBatch{w}", p.mfu * peak.m_as("TFLOPs/s"), src.format(b=b) + ".mfu x FP16 peak")
    ridge = (peak / H100.memory.bandwidth).m_as("flop/byte")
    V.num("CaseIOneEta", eta, "scenario input", nd=1)
    V.num("CaseIOneRidgePoint", ridge, "H100 FP16 peak / memory.bandwidth")
    V.num("CaseIOneEffectiveRidge", eta * ridge, "efficiency x H100 FP16 peak / memory.bandwidth")
    labels = [p.bottleneck for p in profiles]
    last_mem = max((b for b, lab in zip(batches, labels) if lab == "Memory"), default=None)
    first_comp = min((b for b, lab in zip(batches, labels) if lab == "Compute"), default=None)
    require(last_mem is not None and first_comp is not None and last_mem < first_comp,
            f"I1: expected a single memory-to-compute crossover, got {labels}")
    V.num("CaseIOneCrossoverLowBatch", last_mem, "largest memory-bound batch in the sweep")
    V.num("CaseIOneCrossoverHighBatch", first_comp, "smallest compute-bound batch in the sweep")
    if not check:
        sweep_s = median_seconds(lambda: roofline_sweep(model, H100, eta, batches))
        V.num("CaseIOneSweepMs", sweep_s * 1e3, "median of 5 warm runs of the 5-point sweep (ms)", nd=2, volatile=True)
    contrast = roofline_sweep(Models.Language.Llama2_7B, H100, eta, batches)
    V.num("CaseIOneContrastAiMax", max(p.arithmetic_intensity.m_as("flop/byte") for p in contrast),
          "max arithmetic_intensity of SingleNodeModel(Llama2_7B, H100, eta=0.5) over batches 1..256", nd=1)
    V.text("CaseIOneContrastBottleneck", "Memory" if all(p.bottleneck == "Memory" for p in contrast) else "Mixed",
           "bottleneck of every Llama2_7B point in the contrast sweep")
    return {"batches": batches, "profiles": profiles, "eta": eta, "peak": peak}


def section_carbon(V: Values) -> dict:
    V.section("Case I2 and carbon figure: Research_256, 30 days, MFU 0.42")
    fleet, days, mfu = Systems.Clusters.Research_256, 30, 0.42
    res = {}
    for key, label in (("Iowa_Reference", "Iowa"), ("Quebec_Hydro", "Quebec")):
        dc = getattr(Infrastructure.Datacenters, key)
        s = SustainabilityModel().solve(fleet, duration_days=days, datacenter=dc, mfu=mfu)
        src = f"SustainabilityModel().solve(Systems.Clusters.Research_256, duration_days=30, datacenter=Infrastructure.Datacenters.{key}, mfu=0.42)"
        V.num(f"CarbonFig{label}Tonnes", s.carbon_footprint_kg / 1e3, src + ".carbon_footprint_kg / 1e3", nd=1)
        V.num(f"CarbonFig{label}WaterKL", s.water_usage_liters / 1e3, src + ".water_usage_liters / 1e3")
        V.num(f"CarbonFig{label}EnergyMWh", s.total_energy_kwh.m_as("MWh"), src + ".total_energy_kwh", nd=1)
        V.num(f"CarbonFig{label}Pue", s.pue, src + ".pue", nd=2)
        res[label] = s
    V.num("CarbonFigRatio", res["Iowa"].carbon_footprint_kg / res["Quebec"].carbon_footprint_kg, "Iowa carbon / Quebec carbon")
    V.num("CarbonFigGpus", fleet.total_accelerators, "Systems.Clusters.Research_256.total_accelerators")
    V.num("CarbonFigDays", days, "scenario input")
    V.num("CarbonFigMfu", mfu, "scenario input", nd=2)
    return {"results": res}


def section_students(V: Values) -> None:
    V.section("Case S1 and Wall 12: chain-of-thought on B200")
    llama70 = Models.Language.Llama3_70B
    k8 = InferenceScalingModel().solve(llama70, B200, reasoning_steps=8, context_length=2048)
    k1 = InferenceScalingModel().solve(llama70, B200, reasoning_steps=1, context_length=2048)
    require(k8.feasible and k1.feasible, "S1: Llama-3 70B FP16 must fit on one B200")
    src = "InferenceScalingModel().solve(Models.Language.Llama3_70B, Hardware.Cloud.B200, reasoning_steps={k}, context_length=2048)"
    ratio = (k8.total_reasoning_time / k1.total_reasoning_time).m_as("")
    V.num("CaseSOneSteps", 8, "scenario input")
    V.num("CaseSOneTotalSeconds", k8.total_reasoning_time.m_as("s"), src.format(k=8) + ".total_reasoning_time", nd=1)
    V.num("CaseSOneSingleStepSeconds", k1.total_reasoning_time.m_as("s"), src.format(k=1) + ".total_reasoning_time", nd=2)
    V.num("CaseSOneTtftMs", k8.ttft.m_as("ms"), src.format(k=8) + ".ttft")
    V.num("CaseSOneItlMs", k8.itl.m_as("ms"), src.format(k=8) + ".itl", nd=1)
    V.num("CaseSOneLatencyMultiplier", ratio, "total_reasoning_time at K=8 / at K=1", nd=1)
    V.num("CaseSOneEnergyMultiplier", (k8.energy_per_query / k1.energy_per_query).m_as(""),
          "energy_per_query at K=8 / at K=1 (TDP x time)", nd=1)
    V.num("CaseSOneTokenMultiplier", k8.tokens_generated / k1.tokens_generated, "tokens_generated at K=8 / at K=1")
    V.num("WallTwelveLatencyRatio", ratio, "same call as CaseSOneLatencyMultiplier", nd=1)
    h8 = InferenceScalingModel().solve(llama70, H100, reasoning_steps=8, context_length=2048)
    h1 = InferenceScalingModel().solve(llama70, H100, reasoning_steps=1, context_length=2048)
    V.audit["S1_on_H100"] = {"feasible": h8.feasible, "latency_ratio": (h8.total_reasoning_time / h1.total_reasoning_time).m_as(""),
                             "total_s": h8.total_reasoning_time.m_as("s")}

    V.section("Case S2: DGX A100 input pipeline")
    fleet = Fleet(name="DGX A100 (1 node)", node=Systems.Nodes.DGX_A100, count=1, fabric=Systems.Fabrics.InfiniBand_HDR)
    eta, batch = 0.30, 2048
    d = DistributedModel().solve(Models.Vision.ResNet50, fleet, batch_size=batch, precision="fp16", efficiency=eta)
    require(d.node_profile.feasible, "S2: ResNet-50 per-GPU batch must fit on A100")
    src = "DistributedModel().solve(Models.Vision.ResNet50, Fleet(Systems.Nodes.DGX_A100 x 1), batch_size=2048, efficiency=0.30)"
    step = d.step_latency_total
    workers = S2_WORKERS_PER_GPU * fleet.total_accelerators
    supply_aug = workers * S2_AUGMENT_IMG_PER_WORKER_S
    tr = TransformationModel().solve(batch_size=batch, sample_size_bytes=S2_SAMPLE_SIZE,
                                     cpu_throughput=supply_aug * S2_SAMPLE_SIZE / Q_("1 s"), accelerator_step_time=step)
    V.num("CaseSTwoEta", eta, "scenario input", nd=2)
    V.num("CaseSTwoBatch", batch, "scenario input")
    V.num("CaseSTwoStepMs", step.m_as("ms"), src + ".step_latency_total")
    V.num("CaseSTwoDemandImgPerS", batch / step.m_as("s"), "batch / step_latency_total")
    V.pct("CaseSTwoScalingEffPct", d.scaling_efficiency, src + ".scaling_efficiency")
    V.num("CaseSTwoWorkersPerGpu", S2_WORKERS_PER_GPU, "scenario input")
    V.num("CaseSTwoWorkers", workers, "workers per GPU x fleet.total_accelerators")
    V.num("CaseSTwoAugmentRate", S2_AUGMENT_IMG_PER_WORKER_S, "scenario input: decode + augmentation images/s per worker")
    V.num("CaseSTwoDecodeRate", S2_DECODE_IMG_PER_WORKER_S, "scenario input: decode-only images/s per worker")
    V.num("CaseSTwoRawSupplyImgPerS", workers * S2_DECODE_IMG_PER_WORKER_S, "workers x decode-only rate")
    V.num("CaseSTwoEffectiveSupplyImgPerS", supply_aug, "workers x decode+augment rate")
    V.pct("CaseSTwoAugmentFactorPct", S2_AUGMENT_IMG_PER_WORKER_S / S2_DECODE_IMG_PER_WORKER_S, "augment rate / decode rate", nd=0)
    V.pct("CaseSTwoAccelUtilizationPct", tr.accelerator_utilization,
          "TransformationModel().solve(batch_size=2048, 150 KB samples, effective CPU supply, step).accelerator_utilization", nd=0)
    V.num("CaseSTwoEndToEndEta", eta * tr.accelerator_utilization, "eta x accelerator_utilization", nd=2)
    V.text("CaseSTwoCpuBottleneck", "yes" if tr.is_bottleneck else "no", "TransformationModel(...).is_bottleneck")


def section_researchers(V: Values) -> None:
    V.section("Case R1: user-defined 180B on Cerebras CS-3")
    cs3 = Hardware.Cloud.Cerebras_CS3
    m180 = TransformerWorkload(name="User-defined 180B", architecture="Transformer", parameters=Q_("180e9 param"),
                               layers=80, hidden_dim=14848, heads=232, kv_heads=8)
    eta, seq = 0.4, 2048
    ws = WeightStreamingModel()

    def run(b):
        return ws.solve(m180, cs3, seq_len=seq, batch_size=b, precision="fp16", efficiency=eta)

    r1 = run(1)
    require(r1.feasible, "R1: batch 1 must fit on the wafer")
    bstar = r1.optimal_batch_size
    rstar = run(bstar)
    lo, hi = 1, bstar
    while lo < hi:  # largest batch whose KV cache still fits on-wafer SRAM
        mid = (lo + hi + 1) // 2
        if run(mid).feasible:
            lo = mid
        else:
            hi = mid - 1
    rmax = run(lo)
    src = ("WeightStreamingModel().solve(TransformerWorkload(180B, 80 layers, d=14848, 232 heads, 8 KV heads), "
           "Hardware.Cloud.Cerebras_CS3, seq_len=2048, efficiency=0.4, batch_size={b})")
    V.num("CaseROneParamsB", 180, "scenario input")
    V.num("CaseROneLayers", 80, "scenario input")
    V.num("CaseROneHiddenDim", 14848, "scenario input")
    V.num("CaseROneHeads", 232, "scenario input")
    V.num("CaseROneKvHeads", 8, "scenario input")
    V.num("CaseROneSeqLen", seq, "scenario input")
    V.num("CaseROneEta", eta, "scenario input", nd=1)
    V.num("CaseROneInjectionTBs", cs3.interconnect.bandwidth.m_as("TB/s"), "Hardware.Cloud.Cerebras_CS3.interconnect.bandwidth", nd=1)
    V.num("CaseROneWaferSramGiB", cs3.memory.capacity.m_as("GiB"), "Hardware.Cloud.Cerebras_CS3.memory.capacity")
    V.num("CaseROneThroughputBatchOne", r1.throughput_tokens_per_sec, src.format(b=1) + ".throughput_tokens_per_sec", nd=2)
    V.num("CaseROneLatencyBatchOneMs", 1e3 / r1.throughput_tokens_per_sec, "1e3 / CaseROneThroughputBatchOne")
    V.text("CaseROneBottleneckBatchOne", r1.bottleneck, src.format(b=1) + ".bottleneck")
    V.num("CaseROneOptimalBatch", bstar, src.format(b=1) + ".optimal_batch_size")
    V.num("CaseROneWaferUtilizationAtOptimal", rstar.wafer_memory_utilization, src.format(b="B*") + ".wafer_memory_utilization")
    V.text("CaseROneFeasibleAtOptimal", "feasible" if rstar.feasible else "infeasible", src.format(b="B*") + ".feasible")
    V.num("CaseROneMaxFeasibleBatch", lo, "largest batch_size with feasible=True (bisection over WeightStreamingModel)")
    V.num("CaseROneThroughputMaxFeasible", rmax.throughput_tokens_per_sec, src.format(b="CaseROneMaxFeasibleBatch") + ".throughput_tokens_per_sec")
    _, bpp16 = resolve_precision("fp16")
    w180 = m180.size_in_bytes(bpp16)
    n_gpu = math.ceil((w180 / H100.memory.capacity).m_as(""))
    V.num("CaseROneWeightsGB", gb(w180), "size_in_bytes(FP16)")
    V.num("CaseROneGpuTp", n_gpu, "ceil(weights / Hardware.Cloud.H100.memory.capacity)")
    V.num("CaseROneGpuItlMs", (w180 / (n_gpu * H100.memory.bandwidth)).m_as("ms"), "weights / (CaseROneGpuTp x H100.memory.bandwidth)", nd=1)

    V.section("Case R3, Wall 14, fallacy, and architecture-stack figure: Llama-3 70B on Training_512_H100")
    fleet, llama70 = Systems.Clusters.Training_512_H100, Models.Language.Llama3_70B
    eta, days = 0.40, 30
    # TP within each node, ZeRO-1 optimizer sharding across DP, and full activation
    # recomputation: without the last two the TP=8 state does not fit an 80 GB H100.
    kw = dict(batch_size=1024, precision="fp16", efficiency=eta, tp_size=8, pp_size=1,
              zero_stage=1, activation_recomputation=True,
              overlap_comm=True, overlap_efficiency=0.85, seq_len=4096)
    d = DistributedModel().solve(llama70, fleet, **kw)
    src = ("DistributedModel().solve(Models.Language.Llama3_70B, Systems.Clusters.Training_512_H100, batch_size=1024, "
           "efficiency=0.40, tp_size=8, pp_size=1, zero_stage=1, activation_recomputation=True, "
           "overlap_comm=True, overlap_efficiency=0.85, seq_len=4096)")
    # Replica MFU counts model FLOPs only, so full recomputation lowers it to 3/4 of eta.
    mfu = d.scaling_efficiency * d.node_profile.mfu
    require(d.node_profile.feasible, "R3: the Llama-3 70B replica must fit in H100 memory")
    quebec, iowa = Infrastructure.Datacenters.Quebec_Hydro, Infrastructure.Datacenters.Iowa_Reference
    rel = ReliabilityModel().solve(fleet, job_duration_hours=days * 24)
    econ = EconomicsModel().solve(fleet, duration_days=days, datacenter=quebec, mfu=mfu)
    sq = SustainabilityModel().solve(fleet, duration_days=days, datacenter=quebec, mfu=mfu)
    si = SustainabilityModel().solve(fleet, duration_days=days, datacenter=iowa, mfu=mfu)
    steps = days * 86400 / d.step_latency_total.m_as("s")
    tokens = steps * kw["batch_size"] * kw["seq_len"]
    budget = float(mlsysim.Literature.Chinchilla.ComputeConstant) * llama70.parameters.m_as("count") * tokens
    scal = ScalingModel().solve(compute_budget=Q_(budget, "flop"))

    def chain():
        DistributedModel().solve(llama70, fleet, **kw)
        ReliabilityModel().solve(fleet, job_duration_hours=days * 24)
        EconomicsModel().solve(fleet, duration_days=days, datacenter=quebec, mfu=mfu)
        SustainabilityModel().solve(fleet, duration_days=days, datacenter=quebec, mfu=mfu)

    chain_s = median_seconds(chain)
    require(chain_s < CLAIMED_CHAIN_BOUND_S, f"R3 chain took {chain_s:.3f} s, over the claimed {CLAIMED_CHAIN_BOUND_S} s bound")

    V.num("CaseRThreeGpus", fleet.total_accelerators, "Training_512_H100.total_accelerators")
    V.num("CaseRThreeNodes", fleet.count, "Training_512_H100.count")
    V.text("CaseRThreeFabric", fleet.fabric.name, "Training_512_H100.fabric.name")
    V.num("CaseRThreeFabricGbps", fleet.fabric.bandwidth.m_as(Gbps), "Training_512_H100.fabric.bandwidth")
    V.num("CaseRThreeBatch", kw["batch_size"], "scenario input")
    V.num("CaseRThreeSeqLen", kw["seq_len"], "scenario input")
    V.num("CaseRThreeEta", eta, "scenario input", nd=2)
    V.num("CaseRThreeTp", d.parallelism["tp"], src + ".parallelism")
    V.num("CaseRThreeDp", d.parallelism["dp"], src + ".parallelism")
    V.num("CaseRThreeStepSeconds", d.step_latency_total.m_as("s"), src + ".step_latency_total", nd=2)
    V.num("CaseRThreeComputeSeconds", d.node_profile.latency.m_as("s"), src + ".node_profile.latency", nd=2)
    V.num("CaseRThreeDpCommMs", d.dp_communication_latency.m_as("ms"), src + ".dp_communication_latency")
    V.num("CaseRThreeDpExposedMs", d.dp_communication_latency.m_as("ms") * (1 - kw["overlap_efficiency"]),
          src + ".dp_communication_latency x (1 - overlap_efficiency)")
    V.num("CaseRThreeTpCommMs", d.tp_communication_latency.m_as("ms"), src + ".tp_communication_latency")
    V.num("CaseRThreeCommMs", d.communication_latency.m_as("ms"), src + ".communication_latency")
    V.pct("CaseRThreeScalingEffPct", d.scaling_efficiency, src + ".scaling_efficiency")
    V.pct("CaseRThreeMfuPct", mfu, src + ".scaling_efficiency x .node_profile.mfu")
    V.pct("CaseRThreeReplicaMfuPct", d.node_profile.mfu, src + ".node_profile.mfu")
    V.text("CaseRThreeNodeBottleneck", d.node_profile.bottleneck, src + ".node_profile.bottleneck")
    V.num("CaseRThreePerGpuMemoryGB", d.node_profile.memory_footprint.m_as("GB"),
          src + ".node_profile.memory_footprint (TrainingMemoryModel with the same sharding)", nd=1)
    V.text("CaseRThreeMemoryFits", "fits" if d.node_profile.feasible else "exceeds", src + ".node_profile.feasible")
    V.num("CaseRThreeFleetMtbfHours", rel.fleet_mtbf.m_as("hour"), "ReliabilityModel().solve(Training_512_H100, job_duration_hours=720).fleet_mtbf", nd=1)
    V.num("CaseRThreeCheckpointIntervalMin", rel.optimal_checkpoint_interval.m_as("minute"),
          "ReliabilityModel().solve(Training_512_H100, 720).optimal_checkpoint_interval")
    V.num("CaseRThreeExpectedFailures", rel.expected_failures, "ReliabilityModel(...).expected_failures")
    V.pct("CaseRThreeGoodputPct", rel.goodput_ratio, "ReliabilityModel(...).goodput_ratio")
    src_e = "EconomicsModel().solve(Training_512_H100, duration_days=30, datacenter=Infrastructure.Datacenters.Quebec_Hydro, mfu=CaseRThreeMfuPct/100)"
    V.num("CaseRThreeCapexK", econ.capex_usd / 1e3, src_e + ".capex_usd / 1e3")
    V.num("CaseRThreeEnergyK", econ.opex_energy_usd / 1e3, src_e + ".opex_energy_usd / 1e3", nd=1)
    V.num("CaseRThreeMaintenanceK", econ.opex_maintenance_usd / 1e3, src_e + ".opex_maintenance_usd / 1e3")
    V.num("CaseRThreeTcoK", econ.tco_usd / 1e3, src_e + ".tco_usd / 1e3")
    V.num("CaseRThreeTcoM", econ.tco_usd / 1e6, src_e + ".tco_usd / 1e6", nd=2)
    src_s = "SustainabilityModel().solve(Training_512_H100, duration_days=30, datacenter=Infrastructure.Datacenters.{dc}, mfu=CaseRThreeMfuPct/100)"
    V.num("CaseRThreeEnergyMWh", sq.total_energy_kwh.m_as("MWh"), src_s.format(dc="Quebec_Hydro") + ".total_energy_kwh")
    V.num("CaseRThreeCarbonTonnes", sq.carbon_footprint_kg / 1e3, src_s.format(dc="Quebec_Hydro") + ".carbon_footprint_kg / 1e3", nd=1)
    V.num("CaseRThreeWaterKL", sq.water_usage_liters / 1e3, src_s.format(dc="Quebec_Hydro") + ".water_usage_liters / 1e3")
    V.num("CaseRThreeIowaCarbonTonnes", si.carbon_footprint_kg / 1e3, src_s.format(dc="Iowa_Reference") + ".carbon_footprint_kg / 1e3", nd=1)
    V.num("CaseRThreeIowaWaterKL", si.water_usage_liters / 1e3, src_s.format(dc="Iowa_Reference") + ".water_usage_liters / 1e3")
    V.num("CaseRThreeIowaRatio", si.carbon_footprint_kg / sq.carbon_footprint_kg, "Iowa carbon / Quebec carbon")
    V.sci("CaseRThreeFlopBudget", budget, "Literature.Chinchilla.ComputeConstant x P x (30 days / step_latency_total) x 1024 x 4096 tokens")
    V.num("CaseRThreeTokensT", tokens / 1e12, "(30 days / step_latency_total) x 1024 x 4096 / 1e12", nd=2)
    V.num("CaseRThreeChinchillaPstarB", scal.optimal_parameters.m_as("count") / 1e9, "ScalingModel().solve(CaseRThreeFlopBudget).optimal_parameters / 1e9")
    V.num("CaseRThreeChainSeconds", chain_s, "median of 5 warm runs of Distributed+Reliability+Economics+Sustainability (s)", nd=3, volatile=True)
    V.audit["R3_node_profile"] = {"feasible": d.node_profile.feasible, "bottleneck": d.node_profile.bottleneck}

    V.pct("WallFourteenTpCommSharePct", (d.tp_communication_latency / d.step_latency_total).m_as(""),
          src + ".tp_communication_latency / .step_latency_total")
    V.pct("WallFourteenDpCommSharePct", (d.dp_communication_latency * (1 - kw["overlap_efficiency"]) / d.step_latency_total).m_as(""),
          src + " exposed DP communication / .step_latency_total")
    V.pct("FallacyCommCommFractionPct", 1 - d.scaling_efficiency, "1 - " + src + ".scaling_efficiency")
    V.pct("FallacyCommGoodputLossPct", 1 - rel.goodput_ratio, "1 - ReliabilityModel().solve(Training_512_H100, 720).goodput_ratio")

    V.section("Case R4: parallelism search for GPT-3 on Production_2K")
    gpt3, f2k = Models.Language.GPT3, Systems.Clusters.Production_2K
    okw = dict(batch_size=2048, precision="fp16", efficiency=0.5)
    opt = ParallelismOptimizer().solve(gpt3, f2k, **okw)
    src = "ParallelismOptimizer().solve(Models.Language.GPT3, Systems.Clusters.Production_2K, batch_size=2048, efficiency=0.5)"
    min_pp, p = None, 1
    while p <= f2k.total_accelerators:
        try:
            ParallelismOptimizer().solve(gpt3, f2k, max_pp=p, **okw)
            min_pp = p
            break
        except ValueError:
            p *= 2
    require(min_pp is not None, "R4: no pipeline depth is memory-feasible")
    tp_max, params = f2k.node.accelerators_per_node, gpt3.parameters.m_as("count")

    def state_gb(pp):  # replica training state (activations included) the optimizer checks at TP = node size
        dp = f2k.total_accelerators // (tp_max * pp)
        replica = DistributedModel().solve(
            gpt3, f2k, tp_size=tp_max, pp_size=pp, overlap_comm=True,
            microbatch_count=max(1, okw["batch_size"] // dp) if pp > 1 else 1, **okw)
        return replica.node_profile.memory_footprint.m_as("GB")

    V.num("CaseRFourGpus", f2k.total_accelerators, "Production_2K.total_accelerators")
    V.num("CaseRFourBatch", okw["batch_size"], "scenario input")
    V.num("CaseRFourEta", okw["efficiency"], "scenario input", nd=1)
    V.num("CaseRFourTp", opt.best_config["tp"], src + ".best_config")
    V.num("CaseRFourPp", opt.best_config["pp"], src + ".best_config")
    V.num("CaseRFourDp", opt.best_config["dp"], src + ".best_config")
    V.num("CaseRFourCandidates", opt.total_searched, src + ".total_searched (memory-feasible candidates evaluated)")
    V.pct("CaseRFourBestMfuPct", opt.best_mfu, src + ".best_mfu")
    V.num("CaseRFourMinPp", min_pp, "smallest max_pp for which ParallelismOptimizer finds a feasible split")
    V.num("CaseRFourStateAtMinPpGB", state_gb(min_pp), "DistributedModel replica memory_footprint at TP=node size, PP=CaseRFourMinPp (GB)", nd=1)
    V.num("CaseRFourStateBelowMinPpGB", state_gb(max(1, min_pp // 2)), "DistributedModel replica memory_footprint at PP=CaseRFourMinPp/2 (GB)", nd=1)
    V.num("CaseRFourCapacityGB", H100.memory.capacity.m_as("GB"), "Hardware.Cloud.H100.memory.capacity (GB)", nd=1)
    V.num("CaseRFourScreenCapGB", 0.9 * H100.memory.capacity.m_as("GB"), "0.9 x Hardware.Cloud.H100.memory.capacity (GB)", nd=1)


def section_overview(V: Values, check: bool) -> None:
    V.section("Overview figure panel (d): Llama-3 70B")
    llama70, f1k = Models.Language.Llama3_70B, Systems.Clusters.Training_1K
    us = Infrastructure.Datacenters.US_Hyperscale
    t0 = time.perf_counter()
    node = SingleNodeModel().solve(llama70, B200, batch_size=1, precision="fp16")
    sens = SensitivitySolver().solve(llama70, B200)
    syn = SynthesisSolver().solve(llama70, Q_("50 ms"), precision="fp16")
    sust = SustainabilityModel().solve(f1k, duration_days=30, datacenter=us, mfu=OVERVIEW_MFU)
    econ = EconomicsModel().solve(f1k, duration_days=30, datacenter=us, mfu=OVERVIEW_MFU)
    rel = ReliabilityModel().solve(f1k, job_duration_hours=720)
    elapsed = time.perf_counter() - t0
    require(node.feasible, "overview (d): Llama-3 70B FP16 must fit on one B200")
    require(elapsed < CLAIMED_CHAIN_BOUND_S, f"overview (d) took {elapsed:.3f} s, over the claimed bound")
    wall = {"Memory": 2, "Compute": 1}.get(node.bottleneck)
    require(wall is not None, f"overview (d): unexpected bottleneck {node.bottleneck}")
    V.num("OverviewDBindingWall", wall, "SingleNodeModel().solve(Llama3_70B, Hardware.Cloud.B200, batch_size=1).bottleneck (Memory is Wall 2)")
    V.text("OverviewDBottleneck", node.bottleneck, "SingleNodeModel().solve(Llama3_70B, B200, batch_size=1).bottleneck")
    V.num("OverviewDBandwidthTBs", B200.memory.bandwidth.m_as("TB/s"), "Hardware.Cloud.B200.memory.bandwidth", nd=1)
    V.num("OverviewDSensBandwidth", sens.sensitivities["memory_bandwidth"], "SensitivitySolver().solve(Llama3_70B, B200).sensitivities['memory_bandwidth']", nd=3)
    V.num("OverviewDSensFlops", sens.sensitivities["peak_flops"], "SensitivitySolver().solve(Llama3_70B, B200).sensitivities['peak_flops']", nd=1)
    V.num("OverviewDSynthBwTBs", syn.required_bw.m_as("TB/s"), "SynthesisSolver().solve(Llama3_70B, 50 ms).required_bw", nd=2)
    src = "Systems.Clusters.Training_1K, duration_days=30, datacenter=Infrastructure.Datacenters.US_Hyperscale, mfu=0.42"
    V.num("OverviewDGpus", f1k.total_accelerators, "Training_1K.total_accelerators")
    V.num("OverviewDCarbonTonnes", sust.carbon_footprint_kg / 1e3, f"SustainabilityModel().solve({src}).carbon_footprint_kg / 1e3")
    V.num("OverviewDWaterML", sust.water_usage_liters / 1e6, f"SustainabilityModel().solve({src}).water_usage_liters / 1e6", nd=2)
    V.num("OverviewDTcoM", econ.tco_usd / 1e6, f"EconomicsModel().solve({src}).tco_usd / 1e6", nd=2)
    V.num("OverviewDMtbfHours", rel.fleet_mtbf.m_as("hour"), "ReliabilityModel().solve(Training_1K, 720).fleet_mtbf", nd=1)
    V.num("OverviewDCheckpointMin", rel.optimal_checkpoint_interval.m_as("minute"), "ReliabilityModel().solve(Training_1K, 720).optimal_checkpoint_interval")
    if not check:
        V.num("OverviewDSolveSeconds", elapsed, "wall time of the six panel-(d) solves (s)", nd=3, volatile=True)


def section_chain_example(V: Values) -> None:
    V.section("Solver-chaining figure: Llama-2 7B serving on one H100 in Iowa for a year")
    model, iowa = Models.Language.Llama2_7B, Infrastructure.Datacenters.Iowa_Reference
    node = SingleNodeModel().solve(model, H100, batch_size=1, precision="fp16")
    require(node.feasible, "chain example: Llama-2 7B must fit on one H100")
    sv = ServingModel().solve(model, H100, seq_len=CHAIN_SEQ_LEN, batch_size=1, precision="fp16")
    require(sv.feasible, "chain example: serving must fit")
    fleet = single_accelerator_fleet("1x H100 serving replica", H100, Systems.Fabrics.Ethernet_100G, datacenter=iowa)
    econ = EconomicsModel().solve(fleet, duration_days=365, datacenter=iowa, mfu=node.mfu)
    sust = SustainabilityModel().solve(fleet, duration_days=365, datacenter=iowa, mfu=node.mfu)
    _, bpp16 = resolve_precision("fp16")
    V.text("ChainModelName", model.name, "Models.Language.Llama2_7B.name")
    V.num("ChainParamsB", model.parameters.m_as("count") / 1e9, "Models.Language.Llama2_7B.parameters / 1e9")
    V.num("ChainFlopsPerTokenG", model.lower(bpp16).total_ops.m_as("flop") / 1e9, "Llama2_7B.lower(FP16).total_ops / 1e9")
    V.num("ChainWeightsGB", gb(model.size_in_bytes(bpp16)), "Llama2_7B.size_in_bytes(FP16)")
    V.text("ChainBottleneck", node.bottleneck.lower(), "SingleNodeModel().solve(Llama2_7B, H100, batch_size=1).bottleneck")
    V.num("ChainMfu", node.mfu, "SingleNodeModel().solve(Llama2_7B, H100, batch_size=1).mfu", nd=3)
    V.num("ChainTtftMs", sv.ttft.m_as("ms"), "ServingModel().solve(Llama2_7B, H100, seq_len=2048).ttft")
    V.num("ChainItlMs", sv.itl.m_as("ms"), "ServingModel().solve(Llama2_7B, H100, seq_len=2048).itl", nd=1)
    V.num("ChainTcoKPerYear", econ.tco_usd / 1e3, "EconomicsModel().solve(1x H100 fleet, 365 days, Iowa_Reference, mfu=ChainMfu).tco_usd / 1e3", nd=1)
    V.num("ChainCarbonTPerYear", sust.carbon_footprint_kg / 1e3, "SustainabilityModel().solve(1x H100 fleet, 365 days, Iowa_Reference, mfu=ChainMfu).carbon_footprint_kg / 1e3", nd=2)
    V.num("ChainWaterKLPerYear", sust.water_usage_liters / 1e3, "SustainabilityModel(...).water_usage_liters / 1e3", nd=1)


def section_listings(V: Values) -> None:
    V.section("Appendix listings: printed outputs")
    llama8, llama70 = Models.Language.Llama3_8B, Models.Language.Llama3_70B

    for b in (1, 8, 32, 128, 256):  # Scenario A
        p = SingleNodeModel().solve(llama8, H100, batch_size=b, precision="fp16")
        require(p.feasible, f"Scenario A: batch {b} must fit")
        w = BATCH_WORDS[b]
        src = f"Engine.solve(Models.Language.Llama3_8B, Hardware.Cloud.H100, batch_size={b}, precision='fp16')"
        V.num(f"ScenarioAThroughputBatch{w}", p.throughput.m_as("1/s"), src + ".throughput")
        V.text(f"ScenarioABottleneckBatch{w}", p.bottleneck, src + ".bottleneck")
        V.num(f"ScenarioAMfuBatch{w}", p.mfu, src + ".mfu", nd=3)

    fleet = Systems.Clusters.Research_256  # Scenario B
    d = DistributedModel().solve(llama70, fleet, batch_size=1024, seq_len=2048, tp_size=8, pp_size=1, activation_recomputation=True)
    days = (d.step_latency_total * (1e12 / (1024 * 2048))).m_as("day")
    s = SustainabilityModel().solve(fleet, duration_days=days, datacenter=Infrastructure.Datacenters.Iowa_Reference, mfu=0.42)
    V.num("ScenarioBDays", days, "DistributedModel().solve(Llama3_70B, Research_256, batch_size=1024, seq_len=2048, tp_size=8, activation_recomputation=True).step_latency_total x 1e12/(1024 x 2048)")
    V.num("ScenarioBCarbonTonnes", s.carbon_footprint_kg / 1e3, "SustainabilityModel().solve(Research_256, ScenarioBDays, Iowa_Reference, mfu=0.42).carbon_footprint_kg / 1e3", nd=1)

    demand = Q_(SC_DEMAND_IMG_PER_S, "1/s") * S2_SAMPLE_SIZE  # Scenario C
    dm = DataModel().solve(workload_data_rate=demand, hardware=A100)
    tr = TransformationModel().solve(batch_size=2048, sample_size_bytes=S2_SAMPLE_SIZE,
                                     cpu_throughput=SC_WORKERS * S2_AUGMENT_IMG_PER_WORKER_S * S2_SAMPLE_SIZE / Q_("1 s"),
                                     accelerator_step_time=SC_ACCEL_STEP)
    V.text("ScenarioCIoStalled", str(dm.is_stalled), "DataModel().solve(40,000/s x 150 KB, Hardware.Cloud.A100).is_stalled")
    V.text("ScenarioCCpuStalled", str(tr.is_bottleneck), "TransformationModel().solve(2048, 150 KB, 16 x 850 img/s, 48 ms).is_bottleneck")
    V.pct("ScenarioCAccelUtilizationPct", tr.accelerator_utilization, "TransformationModel(...).accelerator_utilization", nd=0)

    syn = SynthesisSolver().solve(llama8, Q_("30 ms"), precision="fp16")  # Scenario D
    V.num("ScenarioDRequiredBwGBs", syn.required_bw.m_as("GB/s"), "SynthesisSolver().solve(Llama3_8B, 30 ms).required_bw", nd=1)
    V.num("ScenarioDRequiredTflops", syn.required_flops.m_as("TFLOPs/s"), "SynthesisSolver().solve(Llama3_8B, 30 ms).required_flops", nd=1)

    base = ServingModel().solve(llama70, B200, seq_len=2048, batch_size=1)  # Scenario E
    spec = ServingModel().solve(llama70, B200, seq_len=2048, batch_size=1, draft_model=llama8, draft_acceptance_rate=0.75)
    require(base.feasible and spec.feasible, "Scenario E: target + draft must fit on one B200")
    V.num("ScenarioEBaseItlMs", base.itl.m_as("ms"), "ServingModel().solve(Llama3_70B, Hardware.Cloud.B200, seq_len=2048).itl", nd=2)
    V.num("ScenarioESpecItlMs", spec.itl.m_as("ms"), "ServingModel().solve(Llama3_70B, B200, seq_len=2048, draft_model=Llama3_8B, draft_acceptance_rate=0.75).itl", nd=2)
    V.num("ScenarioESpeedup", (base.itl / spec.itl).m_as(""), "ScenarioEBaseItlMs / ScenarioESpecItlMs", nd=2)
    V.num("ScenarioEAcceptance", 0.75, "scenario input", nd=2)

    ev = mlsysim.Scenarios.ChatbotServing.evaluate()  # Scenario F
    for level, word in ((ev.feasibility, "Feasibility"), (ev.performance, "Performance"), (ev.macro, "Macro")):
        src_f = f"Scenarios.ChatbotServing.evaluate().{word.lower()}"
        V.text(f"ScenarioF{word}Status", level.status, src_f + ".status")
        V.text(f"ScenarioF{word}Summary", level.summary, src_f + ".summary")
    V.text("ScenarioFPassedAll", str(ev.passed_all), "Scenarios.ChatbotServing.evaluate().passed_all")

    g = ParallelismOptimizer().solve(llama70, fleet, batch_size=2048, precision="fp16", efficiency=0.5)  # Scenario G
    src = "ParallelismOptimizer().solve(Llama3_70B, Research_256, batch_size=2048, efficiency=0.5)"
    V.num("ScenarioGTp", g.best_config["tp"], src + ".best_config")
    V.num("ScenarioGPp", g.best_config["pp"], src + ".best_config")
    V.num("ScenarioGDp", g.best_config["dp"], src + ".best_config")
    V.num("ScenarioGBestMfu", g.best_mfu, src + ".best_mfu", nd=3)
    for i, word in ((1, "Second"), (2, "Third")):
        cfg = g.top_candidates[i]
        V.text(f"ScenarioG{word}Config", f"TP={cfg['config']['tp']}, PP={cfg['config']['pp']}, DP={cfg['config']['dp']}", src + f".top_candidates[{i}]")
        V.num(f"ScenarioG{word}Mfu", cfg["mfu"], src + f".top_candidates[{i}]['mfu']", nd=3)

    for (model, hw), word in (((Models.Vision.LeNet1, Hardware.Tiny.ESP32_S3), "Esp"),  # Scenario H
                              ((Models.Vision.MobileNetV2, Hardware.Edge.JetsonOrinNano), "OrinNano"),
                              ((Models.Vision.MobileNetV2_Alpha0_5, Hardware.Edge.Coral), "Coral")):
        p = SingleNodeModel().solve(model, hw, batch_size=1, precision="int8")
        require(p.feasible, f"Scenario H: {model.name} must fit on {hw.name}")
        src = f"Engine.solve({model.name}, {hw.name}, batch_size=1, precision='int8')"
        V.text(f"ScenarioH{word}Device", hw.name, "registry name")
        V.text(f"ScenarioH{word}Bottleneck", p.bottleneck, src + ".bottleneck")
        V.num(f"ScenarioH{word}LatencyMs", p.latency.m_as("ms"), src + ".latency", nd=2)
        V.num(f"ScenarioH{word}EnergyMJ", p.energy.m_as("mJ"), src + ".energy", nd=2)

    opt = ScalingModel().solve(compute_budget=Q_("4e24 flop"))  # composability listing
    frontier = TransformerWorkload(name="Frontier-Model", architecture="transformer", parameters=opt.optimal_parameters,
                                   layers=80, hidden_dim=8192, heads=64)
    f8k = Systems.Clusters.Frontier_8K
    # One-sequence microbatches: 4,096 sequences over DP = 8,192 / (8 x 4) = 256 leaves 16 per replica.
    compose_microbatches = 4096 // (f8k.total_accelerators // (8 * 4))
    perf = DistributedModel().solve(frontier, f8k, batch_size=4096, tp_size=8, pp_size=4, seq_len=4096,
                                    microbatch_count=compose_microbatches)
    duration = (perf.step_latency_total * (opt.optimal_tokens.m_as("count") / (4096 * 4096))).m_as("day")
    tco = EconomicsModel().solve(f8k, duration_days=duration)
    V.num("ListingComposePstarB", opt.optimal_parameters.m_as("count") / 1e9, "ScalingModel().solve(Q_('4e24 flop')).optimal_parameters / 1e9")
    V.pct("ListingComposeScalingEffPct", perf.scaling_efficiency, "DistributedModel().solve(Frontier-Model, Frontier_8K, batch_size=4096, tp_size=8, pp_size=4, seq_len=4096, microbatch_count=16).scaling_efficiency")
    V.num("ListingComposeMicrobatches", compose_microbatches, "4096 // (Frontier_8K.total_accelerators // 32)")
    V.num("ListingComposeDays", duration, "step_latency_total x optimal_tokens / (4096 x 4096)", nd=1)
    V.num("ListingComposeTcoUSD", tco.tco_usd, "EconomicsModel().solve(Frontier_8K, duration_days=ListingComposeDays).tco_usd")


# ─────────────────────────────────────────────────────────────────────────────
# Appendix tables
# ─────────────────────────────────────────────────────────────────────────────

TIERS = ("Cloud", "Workstation", "Edge", "Mobile", "Tiny")
SOURCE_LABELS = {"hydro": "Hydro", "coal": "Coal", "mixed": "Mixed", "nuclear": "Nuclear",
                 "geothermal": "Geothermal", "coal_gas": "Coal/gas"}


def _peak_cell(hw) -> tuple[float, str]:
    q = hw.compute.precision_flops.get("fp16", hw.compute.peak_flops)
    try:
        return q.m_as("TFLOPs/s"), fmt_table(q.m_as("TFLOPs/s"))
    except Exception:
        return q.m_as("TOPS"), fmt_table(q.m_as("TOPS")) + " (INT8)"


def build_tables() -> dict[str, str]:
    header = ["% Generated by mlsysim/paper/scripts/generate_paper_values.py. Do not edit by hand."]
    hw_lines = header + ["% \\input this file in the preamble, then use \\HwCatalogRows inside the hardware tabular",
                         "% (columns: tier, accelerator, peak FP16 TFLOP/s, memory GiB, bandwidth TB/s, TDP W, interconnect).",
                         "\\newcommand{\\HwCatalogRows}{%"]
    for i, tier in enumerate(TIERS):
        if i:
            hw_lines.append("\\midrule")
        hw_lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{tier}}}}} \\\\")
        rows = []
        for hw in getattr(Hardware, tier).list():
            peak, peak_txt = _peak_cell(hw)
            mem = fmt_table(hw.memory.capacity.m_as("GiB"))
            bw = fmt_table(hw.memory.bandwidth.m_as("TB/s"))
            tdp = fmt_table(hw.tdp.m_as("W")) if hw.tdp is not None else "--"
            if hw.nvlink is not None:
                ic = f"{hw.nvlink.name} ({fmt_table(hw.nvlink.bandwidth.m_as('GB/s'))} GB/s)"
            elif hw.interconnect is not None:
                ic = hw.interconnect.name
            else:
                ic = "---"
            rows.append((peak, hw.name, f" & {tex_escape(hw.name)} & {to_tex(peak_txt)} & {to_tex(mem)} & "
                                        f"{to_tex(bw)} & {to_tex(tdp)} & {tex_escape(ic)} \\\\"))
        hw_lines += [r for _, _, r in sorted(rows, key=lambda t: (t[0], t[1]))]
    hw_lines.append("}")

    infra = header + ["% \\input this file in the preamble, then use the row macros inside the matching tabulars."]
    infra.append("% Grid profiles, by carbon intensity (columns: profile, CI gCO2/kWh, PUE, WUE L/kWh, source).")
    infra.append("\\newcommand{\\InfraGridRows}{%")
    for g in sorted(Infrastructure.Grids.list(), key=lambda g: (g.carbon_intensity_g_kwh, g.name)):
        src = SOURCE_LABELS.get(g.primary_source, str(g.primary_source))
        infra.append(f"{tex_escape(g.name)} & {to_tex(fmt_table(g.carbon_intensity_g_kwh))} & {g.pue:.2f} & "
                     f"{g.wue:.1f} & {src} \\\\")
    infra.append("}")
    infra.append("% Compute nodes (columns: node, composition).")
    infra.append("\\newcommand{\\SystemsNodeRows}{%")
    for n in Systems.Nodes.list():
        link = n.accelerator.nvlink.name if n.accelerator.nvlink is not None else "intra-node link"
        infra.append(f"{tex_escape(n.name)} & {n.accelerators_per_node}$\\times$ {tex_escape(n.accelerator.name)}, "
                     f"{tex_escape(link)}, {to_tex(fmt_table(n.intra_node_bw.m_as('GB/s')))}\\,GB/s per direction \\\\")
    infra.append("}")
    infra.append("% Network fabrics (columns: fabric, bandwidth and latency).")
    infra.append("\\newcommand{\\SystemsFabricRows}{%")
    for f in sorted(Systems.Fabrics.list(), key=lambda f: (f.bandwidth.m_as(Gbps), f.name)):
        lat = f", {to_tex(fmt_table(f.latency.m_as('microsecond')))}\\,$\\mu$s" if f.latency is not None else ""
        infra.append(f"{tex_escape(f.name)} & {to_tex(fmt_table(f.bandwidth.m_as(Gbps)))}\\,Gb/s{lat} \\\\")
    infra.append("}")
    infra.append("% Fleet configurations, by size (columns: fleet, composition).")
    infra.append("\\newcommand{\\SystemsClusterRows}{%")
    for c in sorted(Systems.Clusters.list(), key=lambda c: (c.total_accelerators, c.name)):
        infra.append(f"{tex_escape(c.name)} & {to_tex(fmt_table(c.count))}$\\times$ {tex_escape(c.node.name)}, "
                     f"{tex_escape(c.fabric.name)} \\\\")
    infra.append("}")
    return {"hw_catalog.tex": "\n".join(hw_lines) + "\n", "infra_systems.tex": "\n".join(infra) + "\n"}


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

MEM_COLOR = "#eb6834"   # memory-bound / high-carbon (validated CVD-safe pair)
COMP_COLOR = "#2a78d6"  # compute-bound / low-carbon
INK = "#1f1f1f"
INK_2 = "#52514e"
GRID = "#e6e6e3"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
        "ytick.color": INK_2, "axes.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "path", "svg.hashsalt": "mlsysim-paper", "pdf.fonttype": 42, "savefig.dpi": 300,
    })


def save_figure(fig, stem: str) -> list[Path]:
    paths = []
    for ext, meta in (("pdf", {"CreationDate": None, "Creator": None, "Producer": None}),
                      ("svg", {"Date": None, "Creator": None})):
        path = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", pad_inches=0.02, metadata=meta)
        paths.append(path)
    plt.close(fig)
    return paths


def figure_roofline(ctx: dict) -> list[Path]:
    batches, profiles, eta = ctx["batches"], ctx["profiles"], ctx["eta"]
    peak_t = ctx["peak"].m_as("TFLOPs/s")
    bw = H100.memory.bandwidth.m_as("TB/s")  # TB/s x FLOP/byte = TFLOP/s
    ridge_eff, ridge_peak = eta * peak_t / bw, peak_t / bw

    attained = [p.mfu * peak_t for p in profiles]
    require(min(attained) > 4, "roofline figure: an attained point falls below the plotted y-range")
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.34)
    x_hi = 3000
    x = [10 ** (i / 100) for i in range(100, 349)]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(10, x_hi)
    ax.set_ylim(4, 2500)
    ax.grid(True, which="major", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)

    # Datasheet ceilings (dashed) and the ceilings the engine actually applies at eta (solid).
    ax.plot([ridge_peak, x_hi], [peak_t, peak_t], color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=2)
    xs = [xi for xi in x if xi <= ridge_peak] + [ridge_peak]
    ax.plot(xs, [bw * xi for xi in xs], color=MEM_COLOR, lw=0.9, ls=(0, (4, 2)), zorder=2)
    xm = [xi for xi in x if xi <= ridge_eff] + [ridge_eff]
    ax.plot(xm, [bw * xi for xi in xm], color=MEM_COLOR, lw=2.0, zorder=3, solid_capstyle="butt")
    ax.plot([ridge_eff, x_hi], [eta * peak_t] * 2, color=COMP_COLOR, lw=2.0, zorder=3, solid_capstyle="butt")

    ax.plot([ridge_eff], [eta * peak_t], marker="D", ms=5, color=INK, zorder=6)
    ax.annotate(f"effective ridge {ridge_eff:,.0f}", (ridge_eff, eta * peak_t), xytext=(-6, 5),
                textcoords="offset points", ha="right", va="bottom", fontsize=6.6, color=INK)
    ax.plot([ridge_peak], [peak_t], marker="D", ms=4.5, mfc="white", mec=INK_2, mew=1.0, zorder=6)
    ax.annotate(f"peak ridge {ridge_peak:,.0f}", (ridge_peak, peak_t), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.6, color=INK_2)

    label_offsets = {128: ((6, -5), "left", "top"), 256: ((7, 0), "left", "center")}
    for b, p, att in zip(batches, profiles, attained):
        ai = p.arithmetic_intensity.m_as("flop/byte")
        roof = min(bw * ai, eta * peak_t)
        mem = p.bottleneck == "Memory"
        ax.plot([ai, ai], [att, roof], color=INK_2, lw=0.6, ls=":", zorder=4)
        ax.plot([ai], [att], marker="o" if mem else "s", ms=5.5, mfc=MEM_COLOR if mem else COMP_COLOR,
                mec="white", mew=0.8, zorder=7)
        offset, ha, va = label_offsets.get(b, ((0, -7), "center", "top"))
        ax.annotate(f"B={b}", (ai, att), xytext=offset, textcoords="offset points", ha=ha, va=va,
                    fontsize=6.5, color=INK)

    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attained performance (TFLOP/s)")
    handles = [
        Line2D([], [], color=MEM_COLOR, lw=2.0, label=f"memory roof ({bw:.2f} TB/s)"),
        Line2D([], [], marker="o", ls="", ms=5, mfc=MEM_COLOR, mec="white", label="memory-bound batch"),
        Line2D([], [], color=COMP_COLOR, lw=2.0, label=f"compute ceiling at η = {eta:g}"),
        Line2D([], [], marker="s", ls="", ms=5, mfc=COMP_COLOR, mec="white", label="compute-bound batch"),
        Line2D([], [], color=INK_2, lw=0.9, ls=(0, (4, 2)), label=f"datasheet FP16 peak ({peak_t:,.0f})"),
        Line2D([], [], color=INK_2, lw=0.6, ls=":", label="gap to roof (framework tax)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.0),
               handlelength=1.8, handletextpad=0.4, columnspacing=1.0, fontsize=6.6, labelcolor=INK)
    return save_figure(fig, "roofline-crossover")


def figure_carbon(ctx: dict) -> list[Path]:
    iowa, quebec = ctx["results"]["Iowa"], ctx["results"]["Quebec"]
    gi, gq = Infrastructure.Grids.Iowa, Infrastructure.Grids.Quebec
    kind = gi.metadata.provenance.kind.value
    labels = [f"Iowa\n({kind}\nreference,\n{gi.carbon_intensity_g_kwh:,.0f} g/kWh)",
              f"Québec\n(hydro,\n{gq.carbon_intensity_g_kwh:,.0f} g/kWh)"]
    colors = [MEM_COLOR, COMP_COLOR]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.45))
    panels = [("Operational CO₂ (t)", [iowa.carbon_footprint_kg / 1e3, quebec.carbon_footprint_kg / 1e3], "{:,.1f}"),
              ("Water (kL)", [iowa.water_usage_liters / 1e3, quebec.water_usage_liters / 1e3], "{:,.0f}")]
    for ax, (title, vals, fmt) in zip(axes, panels):
        bars = ax.bar([0, 1], vals, width=0.62, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, color=INK, pad=4, loc="left")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=6.3, color=INK, linespacing=1.05)
        ax.tick_params(axis="x", length=0)
        top = max(vals) * 1.22
        ax.set_ylim(0, top)
        ax.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + top * 0.015, fmt.format(v), ha="center", va="bottom",
                    fontsize=7, color=INK)
    ratio = iowa.carbon_footprint_kg / quebec.carbon_footprint_kg
    axes[0].text(0.98, 0.55, f"{ratio:,.0f}×", transform=axes[0].transAxes, ha="right", va="center",
                 fontsize=10, fontweight="bold", color=INK)
    fig.tight_layout(w_pad=1.0)
    return save_figure(fig, "carbon-comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Diagram label sync, outputs, and the drift check
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_RE = re.compile(r'(<text\b[^>]*?\bdata-template="([^"]*)"[^>]*>)(.*?)(</text>)', re.S)
CHAIN_SEQ_LEN = 2048   # context length of the solver-chaining example (section_chain_example)
OVERVIEW_MFU = 0.42    # utilization of the overview panel (d) fleet (section_overview)
STACK_RESOLVERS_DRAWN = 4  # resolver boxes drawn in architecture-stack.svg


def section_diagram_labels(V: Values) -> None:
    """Values that exist only to label the conceptual diagrams."""
    V.section("Diagram labels")
    r256 = Systems.Clusters.Research_256
    llama70 = Models.Language.Llama3_70B
    _, bpp16 = resolve_precision("fp16")
    V.num("RegResearchNodes", r256.count, "Systems.Clusters.Research_256.count")
    V.text("RegResearchFabric", r256.fabric.name, "Systems.Clusters.Research_256.fabric.name")
    V.num("RegLlamaSeventyParamsB", llama70.parameters.m_as("count") / 1e9, "Models.Language.Llama3_70B.parameters / 1e9", nd=1)
    V.num("RegLlamaSeventyLayers", llama70.layers, "Models.Language.Llama3_70B.layers")
    V.num("RegLlamaSeventyWeightsGB", gb(llama70.size_in_bytes(bpp16)), "Models.Language.Llama3_70B.size_in_bytes(FP16)")
    V.num("DiagramStackMoreResolvers", V.records["CountResolvers"]["raw"] - STACK_RESOLVERS_DRAWN,
          "CountResolvers minus the resolver boxes drawn in architecture-stack.svg")
    bottleneck = V.records["CaseRThreeNodeBottleneck"]["raw"]
    wall = {"Memory": 2, "Compute": 1}.get(bottleneck)
    require(wall is not None, f"R3: unexpected node bottleneck {bottleneck!r}")
    V.num("CaseRThreeBindingWall", wall, "CaseRThreeNodeBottleneck mapped to its wall (Compute=1, Memory=2)")
    V.text("CaseRThreeBindingChip", bottleneck.upper(), "CaseRThreeNodeBottleneck, upper case")
    V.num("ChainSeqLen", CHAIN_SEQ_LEN, "scenario input (section_chain_example)")
    V.num("OverviewDMfu", OVERVIEW_MFU, "scenario input (section_overview)", nd=2)


def guard_diagram_names(V: Values) -> None:
    """Every physics domain module must be drawn in system-anatomy.svg."""
    text = (FIGURES_DIR / "system-anatomy.svg").read_text(encoding="utf-8")
    for name in V.records["ListPhysicsDomainModules"]["raw"].split(", "):
        if not re.search(rf">\s*{re.escape(name)}\s*<", text):
            fail(f"system-anatomy.svg does not draw physics module {name!r}")


def sync_diagrams(V: Values, write: bool) -> list[str]:
    guard_diagram_names(V)
    values = V.plain_map()
    changed, used = [], 0
    for svg in sorted(FIGURES_DIR.glob("*.svg")):
        original = svg.read_text(encoding="utf-8")

        def repl(m):
            nonlocal used
            try:
                content = html.unescape(m.group(2)).format_map(values)
            except KeyError as exc:
                fail(f"{svg.name}: data-template names unknown value {exc}")
            used += 1
            return m.group(1) + html.escape(content, quote=False) + m.group(4)

        updated = TEMPLATE_RE.sub(repl, original)
        if updated != original:
            changed.append(svg.name)
            if write:
                svg.write_text(updated, encoding="utf-8")
    V.audit["diagram_templates"] = used
    return changed


def write_outputs(V: Values, tables: dict[str, str]) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by mlsysim/paper/scripts/generate_paper_values.py. Do not edit by hand.",
             "% Regenerate with `make values` in mlsysim/paper/. Raw values: generated/values.json.",
             "% Percent macros (*Pct) print the number without the percent sign."]
    current = None
    for name, rec in V.records.items():
        if rec["section"] != current:
            current = rec["section"]
            lines += ["", f"% ===== {current} ====="]
        lines.append(f"\\newcommand{{\\{name}}}{{{rec['tex']}}}% source: {rec['source']}")
    (GENERATED_DIR / "values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"generator": "mlsysim/paper/scripts/generate_paper_values.py", "values": V.records, "audit": V.audit}
    (GENERATED_DIR / "values.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    for name, text in tables.items():
        (GENERATED_DIR / name).write_text(text, encoding="utf-8")


def check_drift(V: Values, tables: dict[str, str]) -> list[str]:
    problems = []
    committed_path = GENERATED_DIR / "values.json"
    if not committed_path.exists():
        return [f"{committed_path} is missing; run `make values`"]
    committed = json.loads(committed_path.read_text(encoding="utf-8"))["values"]
    for name, rec in V.records.items():
        if rec["volatile"]:
            continue
        old = committed.get(name)
        if old is None:
            problems.append(f"{name}: new value {rec['printed']!r} not in committed values.json")
            continue
        a, b = rec["raw"], old["raw"]
        if isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
            if not math.isclose(a, b, rel_tol=CHECK_REL_TOL, abs_tol=1e-12):
                problems.append(f"{name}: committed {old['printed']} ({b!r}), recomputed {rec['printed']} ({a!r})")
        elif a != b:
            problems.append(f"{name}: committed {b!r}, recomputed {a!r}")
    for name, old in committed.items():
        if name not in V.records and not old.get("volatile"):
            problems.append(f"{name}: in committed values.json but no longer generated")
    for fname, text in tables.items():
        path = GENERATED_DIR / fname
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            problems.append(f"generated/{fname} differs from the registry")
    for svg in sync_diagrams(V, write=False):
        problems.append(f"figures/{svg}: data-template labels out of sync")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="recompute without writing; exit 1 if any value differs from generated/values.json")
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    apply_style()
    V = Values()
    try:
        section_registry(V)
        section_counts(V, args.check)
        section_anchors(V)
        section_walls(V)
        roof = section_instructor(V, args.check)
        carbon = section_carbon(V)
        section_students(V)
        section_researchers(V)
        section_overview(V, args.check)
        section_chain_example(V)
        section_listings(V)
        section_diagram_labels(V)
        tables = build_tables()
        if args.check:
            problems = check_drift(V, tables)
            if problems:
                print(f"generate_paper_values --check: {len(problems)} difference(s):", file=sys.stderr)
                for p in problems:
                    print(f"  - {p}", file=sys.stderr)
                print("Regenerate with `make values` in mlsysim/paper and commit the result.", file=sys.stderr)
                return 1
            print(f"generate_paper_values --check: {sum(not r['volatile'] for r in V.records.values())} values match")
            return 0
        section_timing(V)
        figures = figure_roofline(roof) + figure_carbon(carbon)
        changed = sync_diagrams(V, write=True)
        write_outputs(V, tables)
    except GenerationError as exc:
        print(f"generate_paper_values: FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(V.records)} macros to {GENERATED_DIR / 'values.tex'}")
    print("figures:", ", ".join(p.name for p in figures))
    print("diagrams updated:", ", ".join(changed) if changed else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
