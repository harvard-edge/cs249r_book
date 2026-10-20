"""Agent-sandbox solvers: the cost of an isolation envelope (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. Each model is arithmetic
on stated per-call, per-byte, or per-lease inputs: the system-call tax each
envelope pays, setup paid per call or once per trajectory, workspace
provisioning by copy or copy-on-write, and a warm pool sized by Little's law
and a Poisson reserve. None models queueing inside a host, restore
contention, or a particular hypervisor.
"""

from __future__ import annotations

from ..results import (
    BootTimeCdfResult,
    EnvelopeAmortizationResult,
    SandboxPoolReserveResult,
    SyscallTaxResult,
    WorkspaceProvisioningResult,
)
from .base import ForwardModel
from ...core.types import Quantity
from ...physics.agents import calc_setup_share, calc_work_for_setup_cap
from ...physics.serving import calc_littles_law_occupancy
from ...physics.statistics import (
    calc_cdf_quantile,
    calc_poisson_mean_for_reserve,
    calc_poisson_reserve,
    calc_poisson_tail,
)


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class SyscallTaxModel(ForwardModel):
    """
    The added time of ``calls`` ordinary system calls under each envelope.

    A shared-kernel container pays a trap and an in-kernel filter per call, a
    microVM a native trap in its guest (device exits counted separately), and
    a user-space kernel a trap, emulation, and a switch to the emulator.
    """
    requires = ()
    produces = SyscallTaxResult

    def solve(self, calls: int, trap: Quantity, filter: Quantity, emulate: Quantity,
              switch: Quantity) -> SyscallTaxResult:
        container = trap + filter
        microvm = trap
        userspace = trap + emulate + switch
        return SyscallTaxResult(
            container_call=container, microvm_call=microvm, userspace_call=userspace,
            container_total=(calls * container).to("s"),
            microvm_total=(calls * microvm).to("s"),
            userspace_total=(calls * userspace).to("s"),
            userspace_vs_microvm=_ratio(userspace, microvm),
        )


class EnvelopeAmortizationModel(ForwardModel):
    """
    Setup paid on every call against once per trajectory, under equal useful work.

    Fresh per call: ``N (T_setup + W)``; retained: ``T_setup + N W``. Setup is
    ``T_setup / (T_setup + W)`` of a fresh call, and holding it to a share
    ``phi`` needs ``W >= T_setup (1 - phi) / phi``. Serialized, no overlap.
    """
    requires = ()
    produces = EnvelopeAmortizationResult

    def solve(self, calls: int, setup: Quantity, work_per_call: Quantity,
              overhead_cap: float) -> EnvelopeAmortizationResult:
        useful = calls * work_per_call
        per_call_setup = calls * setup
        per_call_total = per_call_setup + useful
        per_trajectory = setup + useful
        return EnvelopeAmortizationResult(
            useful_work=useful, per_call_setup=per_call_setup, per_call_total=per_call_total,
            per_trajectory_total=per_trajectory, setup_saved=per_call_setup - setup,
            reuse_speedup=_ratio(per_call_total, per_trajectory),
            setup_share=calc_setup_share(setup, work_per_call),
            work_for_cap=calc_work_for_setup_cap(setup, overhead_cap),
        )


class WorkspaceProvisioningModel(ForwardModel):
    """
    Bytes to provision ``workspaces`` by full copy or copy-on-write, and the
    first-write payload under file or block granularity.

    Payload only: drive times are lower bounds at the sequential rate, and
    metadata, journaling, and caching add traffic.
    """
    requires = ()
    produces = WorkspaceProvisioningResult

    def solve(self, workspaces: int, base_image: Quantity, written: Quantity,
              bandwidth: Quantity, fixture: Quantity, block: Quantity,
              modified: Quantity) -> WorkspaceProvisioningResult:
        full = workspaces * base_image
        cow = workspaces * written
        return WorkspaceProvisioningResult(
            full_copy=full, cow_written=cow,
            full_copy_time=(full / bandwidth).to("s"), cow_time=(cow / bandwidth).to("s"),
            copy_vs_cow=_ratio(full, cow),
            file_copy=workspaces * fixture, block_copy=workspaces * block,
            logical_changes=workspaces * modified,
            file_vs_block=_ratio(fixture, block),
        )


class SandboxPoolReserveModel(ForwardModel):
    """
    A warm sandbox pool under steady Poisson arrivals.

    Active leases follow Little's law, ``N = lambda T_hold``. With ``s``
    standby instances and a constant restore lead time ``T_init``, a request
    misses when at least ``s`` earlier replacements are outstanding:
    ``P(Poisson(lambda T_init) >= s)``. Resident guest memory is a private
    image per lease, or one shared image plus dirty pages per lease and
    private state per standby under copy-on-write.
    """
    requires = ()
    produces = SandboxPoolReserveResult

    def solve(self, arrival_rate: Quantity, hold: Quantity, restore: Quantity,
              guest_memory: Quantity, dirty: Quantity, standby: Quantity, target_miss: float,
              initial_reserve: int, slow_factor: float) -> SandboxPoolReserveResult:
        active = float(calc_littles_law_occupancy(arrival_rate, hold).to("").magnitude)
        mu = float((arrival_rate * restore).to("").magnitude)
        reserve = calc_poisson_reserve(mu, target_miss)
        private = (active * guest_memory).to("GB")
        pool = (guest_memory + active * dirty + reserve * standby).to("GB")
        mu_max = calc_poisson_mean_for_reserve(reserve, target_miss)
        mu_slow = mu * slow_factor
        return SandboxPoolReserveResult(
            active=active, restore_arrivals=mu,
            initial_miss=calc_poisson_tail(mu, initial_reserve),
            reserve=reserve, reserve_miss=calc_poisson_tail(mu, reserve),
            private_memory=private, pool_memory=pool, memory_ratio=_ratio(private, pool),
            max_restore=(mu_max / arrival_rate).to("ms"),
            slow_arrivals=mu_slow, slow_reserve=calc_poisson_reserve(mu_slow, target_miss),
        )


class BootTimeCdfModel(ForwardModel):
    """
    The median and a tail percentile of a measured boot-time CDF.

    Reads the first boot time whose cumulative fraction reaches each
    percentile, as a released CDF is read; no interpolation.
    """
    requires = ()
    produces = BootTimeCdfResult

    def solve(self, booting: int, boot_times, cumulative, median_fraction: float,
              tail_fraction: float) -> BootTimeCdfResult:
        return BootTimeCdfResult(
            booting=booting,
            median=calc_cdf_quantile(boot_times, cumulative, median_fraction),
            tail=calc_cdf_quantile(boot_times, cumulative, tail_fraction),
        )
