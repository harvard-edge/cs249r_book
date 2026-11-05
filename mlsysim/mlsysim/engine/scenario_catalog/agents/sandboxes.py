"""Named agents scenarios for sandboxes.

Inputs are composed here; arithmetic belongs to scenario definitions and solvers.
"""

from ....core.units import Q_
from ....hardware.registry import Hardware as _Hardware
from ....agents.registry import Agents as _Agents
from ... import scenario_types as _st


# --- Agent sandboxes (added 2026-10-20) ---
SyscallTaxPerMillionCalls = _st.SyscallTaxScenario(
    name="Added time of a million ordinary system calls in a container, microVM, and user-space kernel",
    description="Illustrative per-call terms: a 0.1 us trap, 0.05 us in-kernel filter, 1.5 us "
    "emulation, and a 2 us switch to the emulator; device exits not counted.",
    calls=1_000_000,  # scenario assumption: calls in a build or test run
    trap=Q_(0.1, "us"),  # scenario assumption: trap into a kernel
    filter=Q_(0.05, "us"),  # scenario assumption: in-kernel filter
    emulate=Q_(1.5, "us"),  # scenario assumption: user-space kernel emulation
    switch=Q_(2.0, "us"),  # scenario assumption: switch to the emulator
)


ParserCallsEnvelopeSetup = _st.EnvelopeAmortizationScenario(
    name="100 parser calls of 3 ms each, envelope setup per call or once per trajectory",
    description="Setup is the SWE-bench coding harness's snapshot restore; a 10 percent cap "
    "on setup's share of a fresh call.",
    calls=100,  # scenario assumption: sequential parser calls in one trajectory
    setup=_Agents.Coding.SWE_Bench_Runner.sandbox_snapshot_restore_latency.to("ms"),
    work_per_call=Q_(3, "ms"),  # scenario assumption: useful work per call
    overhead_cap=0.10,  # scenario assumption: setup at most this share of a fresh call
)


CopyOnWriteRepairWorkspaces = _st.WorkspaceProvisioningScenario(
    name="32 repair workspaces on a 2.4 GB base, by full copy or copy-on-write",
    description="Each task writes 40 MB; one PCIe Gen4 NVMe drive; a 1.2 GB shared fixture "
    "changed by one byte under file or 64 KB block copying.",
    workspaces=32,  # scenario assumption: concurrent workspaces for one task
    base_image=Q_(2.4, "GB"),  # scenario assumption: repository, toolchain, dependencies
    written=Q_(40, "MB"),  # scenario assumption: edits and build outputs per workspace
    bandwidth=_Hardware.Tech.Storage.NvmeGen4.bandwidth,
    fixture=Q_(1.2, "GB"),  # scenario assumption: shared binary test fixture
    block=Q_(64, "KB"),  # scenario assumption: block-granular copying unit
    modified=Q_(1, "byte"),  # scenario assumption: bytes each task overwrites
)


CodingSandboxPoolTwoPerSecond = _st.SandboxPoolReserveScenario(
    name="Coding sandbox pool at 2 trajectories per second, 300 s leases, snapshot restore",
    description="The SWE-bench coding harness's restore time and guest image; 64 MB written "
    "per lease, 8 MB per standby, a 0.1 percent request-level miss target.",
    arrival_rate=Q_(2.0, "1/s"),  # scenario assumption: offered trajectories per second
    hold=Q_(300, "s"),  # scenario assumption: lease held for the whole trajectory
    restore=_Agents.Coding.SWE_Bench_Runner.sandbox_snapshot_restore_latency,
    guest_memory=_Agents.Coding.SWE_Bench_Runner.sandbox_memory_footprint,
    dirty=Q_(64, "MB"),  # scenario assumption: private pages written per active lease
    standby=Q_(8, "MB"),  # scenario assumption: private state per standby instance
    target_miss=1e-3,  # scenario assumption: design target for finding standby empty
    initial_reserve=1,  # scenario assumption: candidate evaluated first
    slow_factor=10,  # scenario assumption: a cold boot an order of magnitude slower
)
