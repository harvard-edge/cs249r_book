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

    def solve(
        self, calls: int, trap: Quantity, filter: Quantity, emulate: Quantity, switch: Quantity
    ) -> SyscallTaxResult:
        """Compute serialized system-call overhead.

        Parameters
        ----------
        calls : int
            Number of calls, nonnegative.
        trap : Quantity
            Native trap duration, positive time quantity.
        filter : Quantity
            Filter duration per call, nonnegative time quantity.
        emulate : Quantity
            Emulation duration per call, nonnegative time quantity.
        switch : Quantity
            Context-switch duration per call, nonnegative time quantity.

        Returns
        -------
        SyscallTaxResult
            Per-call durations, total serialized seconds, and the dimensionless
            userspace-to-microVM time ratio.

        Notes
        -----
        Equation: T_container=N(trap+filter); T_guest=N trap; T_user=N(trap+emulate+switch).

        Assumptions: Each call pays the same stated cost, with no overlap.

        Excludes: Device exits, scheduling contention, and measured hypervisor behavior.

        References
        ----------
        Derivation: add the trap and envelope-specific costs paid by one call,
        then multiply that duration by the call count. The guest baseline pays only
        the native trap, so dividing user-space duration by trap gives its relative tax.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = SyscallTaxModel().solve(2, Q_(1,"us"), Q_(1,"us"), Q_(2,"us"), Q_(1,"us"))
        >>> r.container_total > r.microvm_total
        True
        """
        container = trap + filter
        microvm = trap
        userspace = trap + emulate + switch
        return SyscallTaxResult(
            container_call=container,
            microvm_call=microvm,
            userspace_call=userspace,
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

    def solve(
        self, calls: int, setup: Quantity, work_per_call: Quantity, overhead_cap: float
    ) -> EnvelopeAmortizationResult:
        """Compare fresh and retained envelope setup costs.

        Parameters
        ----------
        calls : int
            Positive number of calls.
        setup : Quantity
            Nonnegative setup time quantity.
        work_per_call : Quantity
            Positive useful time per call.
        overhead_cap : float
            Allowed setup share, dimensionless and between zero and one.

        Returns
        -------
        EnvelopeAmortizationResult
            Useful work, setup and total durations, reuse speedup, setup share,
            and useful-work duration required by the overhead cap.

        Notes
        -----
        Equation: T_fresh=N(S+W); T_retained=S+NW; W_min=S(1-phi)/phi.

        Assumptions: Equal useful work and serial setup across calls.

        Excludes: Idle retention cost, eviction, and overlapping work.

        References
        ----------
        Derivation: retaining an envelope removes all but one setup charge.
        Solving setup / (setup + work) <= overhead_cap for work gives the
        minimum useful-work duration; this is a cost-accounting identity.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = EnvelopeAmortizationModel().solve(2, Q_(1,"s"), Q_(2,"s"), 0.2)
        >>> r.per_call_total > r.per_trajectory_total
        True
        """
        useful = calls * work_per_call
        per_call_setup = calls * setup
        per_call_total = per_call_setup + useful
        per_trajectory = setup + useful
        return EnvelopeAmortizationResult(
            useful_work=useful,
            per_call_setup=per_call_setup,
            per_call_total=per_call_total,
            per_trajectory_total=per_trajectory,
            setup_saved=per_call_setup - setup,
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

    def solve(
        self,
        workspaces: int,
        base_image: Quantity,
        written: Quantity,
        bandwidth: Quantity,
        fixture: Quantity,
        block: Quantity,
        modified: Quantity,
    ) -> WorkspaceProvisioningResult:
        """Compare full-copy and copy-on-write payloads.

        Parameters
        ----------
        workspaces : int
            Positive workspace count.
        base_image : Quantity
            Base image size, positive byte quantity.
        written : Quantity
            Bytes written per workspace, positive byte quantity.
        bandwidth : Quantity
            Positive sequential storage bandwidth, bytes per second.
        fixture : Quantity
            File copied on first write, positive byte quantity.
        block : Quantity
            Block copied on first write, positive byte quantity.
        modified : Quantity
            Logical modification size, nonnegative byte quantity.

        Returns
        -------
        WorkspaceProvisioningResult
            Full-copy and copy-on-write byte payloads, sequential-transfer
            times in seconds, and file-to-block amplification ratios.

        Notes
        -----
        Equation: B_full=N B_image; B_cow=N B_written; T=B/bandwidth.

        Assumptions: Each workspace writes the supplied payload at a constant bandwidth.

        Excludes: Metadata, journaling, cache effects, and device contention.

        References
        ----------
        Derivation: each workspace contributes either the whole image or its
        written payload. Dividing aggregate bytes by sequential bytes per second
        gives a transfer-time lower bound. First-write amplification divides
        file-size payload by block-size payload for the same logical change.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = WorkspaceProvisioningModel().solve(2, Q_(10,"GB"), Q_(1,"GB"), Q_(1,"GB/s"), Q_(1,"MB"), Q_(4,"KiB"), Q_(1,"KiB"))
        >>> r.full_copy_time > r.cow_time
        True
        """
        full = workspaces * base_image
        cow = workspaces * written
        return WorkspaceProvisioningResult(
            full_copy=full,
            cow_written=cow,
            full_copy_time=(full / bandwidth).to("s"),
            cow_time=(cow / bandwidth).to("s"),
            copy_vs_cow=_ratio(full, cow),
            file_copy=workspaces * fixture,
            block_copy=workspaces * block,
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

    def solve(
        self,
        arrival_rate: Quantity,
        hold: Quantity,
        restore: Quantity,
        guest_memory: Quantity,
        dirty: Quantity,
        standby: Quantity,
        target_miss: float,
        initial_reserve: int,
        slow_factor: float,
    ) -> SandboxPoolReserveResult:
        """Size a warm pool against a Poisson miss target.

        Parameters
        ----------
        arrival_rate : Quantity
            Positive arrival rate, inverse time.
        hold : Quantity
            Positive lease holding time.
        restore : Quantity
            Positive restore lead time.
        guest_memory : Quantity
            Positive private guest memory, byte quantity.
        dirty : Quantity
            Nonnegative dirty memory per active guest, byte quantity.
        standby : Quantity
            Nonnegative private state per standby, byte quantity.
        target_miss : float
            Target miss probability, between zero and one.
        initial_reserve : int
            Nonnegative initial standby count.
        slow_factor : float
            Positive multiplicative restore-time slowdown.

        Returns
        -------
        SandboxPoolReserveResult
            Mean active leases, standby counts, Poisson miss probabilities,
            private and shared-pool memory in GB, and restore-time ceiling in ms.

        Notes
        -----
        Equation: N=lambda T_hold; mu=lambda T_restore; miss=P(Poisson(mu)>=reserve).

        Assumptions: Steady Poisson arrivals and constant restore lead time.

        Excludes: Burst correlation, restore contention, and a guaranteed tail bound for non-Poisson arrivals.

        References
        ----------
        Derivation: Little's occupancy identity multiplies arrivals per second
        by lease duration. During a fixed restore interval, Poisson arrivals have
        mean arrival_rate * restore; calc_poisson_reserve chooses the smallest
        standby count meeting its upper-tail target. Pool memory adds one shared
        image, dirty pages for active leases, and private standby state.

        Examples
        --------
        >>> from mlsysim.core.units import Q_
        >>> r = SandboxPoolReserveModel().solve(Q_(1,"1/s"), Q_(10,"s"), Q_(1,"s"), Q_(1,"GB"), Q_(0.1,"GB"), Q_(0.1,"GB"), 0.01, 1, 2)
        >>> r.reserve_miss <= 0.01
        True
        """
        active = float(calc_littles_law_occupancy(arrival_rate, hold).to("").magnitude)
        mu = float((arrival_rate * restore).to("").magnitude)
        reserve = calc_poisson_reserve(mu, target_miss)
        private = (active * guest_memory).to("GB")
        pool = (guest_memory + active * dirty + reserve * standby).to("GB")
        mu_max = calc_poisson_mean_for_reserve(reserve, target_miss)
        mu_slow = mu * slow_factor
        return SandboxPoolReserveResult(
            active=active,
            restore_arrivals=mu,
            initial_miss=calc_poisson_tail(mu, initial_reserve),
            reserve=reserve,
            reserve_miss=calc_poisson_tail(mu, reserve),
            private_memory=private,
            pool_memory=pool,
            memory_ratio=_ratio(private, pool),
            max_restore=(mu_max / arrival_rate).to("ms"),
            slow_arrivals=mu_slow,
            slow_reserve=calc_poisson_reserve(mu_slow, target_miss),
        )
