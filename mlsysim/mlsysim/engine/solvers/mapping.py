"""Volumetric-map solvers: clearance-query reads, map storage, update traffic, and encoder rate (added 2026-10-20).

Domain implementations behind ``mlsysim.solvers``. The equations live in
``physics/robotics/perception.py``; these models compose them for one stated
permission controller, workcell, depth stream, or encoder.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..results import (
    ClearanceQueryResult,
    ClearanceQueryRow,
    EncoderRateResult,
    HostMapSizingResult,
    RaymarchTrafficResult,
)
from ...core.types import Quantity
from ...core.units import ureg
from ...physics.robotics.perception import (
    calc_octree_depth,
    calc_sphere_scan_cells,
    calc_voxel_quantization_error,
)
from .base import ForwardModel


def _ratio(a, b) -> float:
    return float((a / b).to("").magnitude)


class ClearanceQueryModel(ForwardModel):
    """
    Reads a sphere-hull clearance query needs per map representation, against the permission tick's read budget.

    The collision allocation divided among the hulls gives the time per
    query; SRAM bandwidth over that time, in cells, is the read budget. An
    occupancy scan reads ``calc_sphere_scan_cells`` cells; an octree pays
    ``calc_octree_depth`` dependent reads for each; an ESDF reads the corners
    of one trilinear interpolation. A snapshot of ``slab_volume`` stores
    ``V / r^3`` cells, and a cell's center misplaces a point by up to
    ``calc_voxel_quantization_error``.
    """
    requires = ("tick_rate", "collision_allocation", "hulls", "sram_bandwidth")
    produces = ClearanceQueryResult

    def solve(self, tick_rate: Quantity, collision_allocation: Quantity, hulls: int, hull_radius: Quantity,
              protective_clearance: Quantity, sram_bandwidth: Quantity, esdf_cell_bytes: Quantity,
              occupancy_cell_bytes: Quantity, map_edge: Quantity, slab_volume: Quantity,
              pitches: Sequence[Quantity], trilinear_reads: int,
              wcet: Optional[Quantity] = None) -> ClearanceQueryResult:
        """Compare map-query reads and storage at each voxel pitch.

        Parameters
        ----------
        tick_rate : Quantity
            Positive permission-controller frequency, in Hz.
        collision_allocation : Quantity
            Time assigned to all collision queries per tick, in seconds.
        hulls : int
            Positive number of spherical hull queries sharing the allocation.
        hull_radius : Quantity
            Nonnegative radius of each collision hull, in meters.
        protective_clearance : Quantity
            Nonnegative extra clearance around each hull, in meters.
        sram_bandwidth : Quantity
            Positive sustained SRAM read bandwidth, in bytes per second.
        esdf_cell_bytes : Quantity
            Positive storage per signed-distance cell, in bytes.
        occupancy_cell_bytes : Quantity
            Positive storage per occupancy cell, in bytes.
        map_edge : Quantity
            Positive map side length used to estimate octree depth, in meters.
        slab_volume : Quantity
            Positive snapshot volume whose storage is counted, in cubic meters.
        pitches : sequence of Quantity
            Positive voxel widths in meters. Result rows are keyed by rounded
            centimeters, so pitches must have distinct keys to retain every row.
        trilinear_reads : int
            Nonnegative number of corner-cell reads for one ESDF interpolation.
        wcet : Quantity, optional
            Positive reference worst-case execution time, in seconds. If supplied,
            report allocation divided by this value; no WCET is inferred.

        Returns
        -------
        ClearanceQueryResult
            Tick and per-query times in microseconds, read budgets as counts, and
            pitch-indexed storage in kilobytes and spatial error in meters.
            Budget ratios describe this accounting model, not collision safety.

        See Also
        --------
        mlsysim.physics.robotics.perception.calc_sphere_scan_cells : Sphere-grid scan count.
        mlsysim.physics.robotics.perception.calc_octree_depth : Dependent octree levels.
        mlsysim.physics.robotics.perception.calc_voxel_quantization_error : Cell-center spatial error.

        Notes
        -----
        Equation: t_query = collision_allocation / hulls; reads_budget =
            t_query * sram_bandwidth / esdf_cell_bytes. Dense storage counts
            slab_volume / ``pitch**3`` cells; octree reads multiply sphere-scan cells
            by tree depth, while the ESDF reads trilinear_reads corners.
        Assumptions: one equal time allocation per hull, uniform cubic voxels,
            fully sustained SRAM bandwidth, and a fixed interpolation read count.
        Excludes: cache reuse, dependency stalls, tree traversal instructions,
            scene reconstruction error, and certification of a collision checker.

        References
        ----------
        Derivation: divide time and byte budgets by per-query and per-cell costs;
            the spatial counts and quantization bound are supplied by the linked
            sphere-scan, octree-depth, and voxel-quantization functions.

        Examples
        --------
        A constructed controller gives each of two hulls half its collision budget.

        >>> from mlsysim import Q_
        >>> result = ClearanceQueryModel().solve(
        ...     Q_(100, 'Hz'), Q_(2, 'ms'), 2, Q_(0.1, 'm'), Q_(0.05, 'm'),
        ...     Q_(1, 'GB/s'), Q_(4, 'byte'), Q_(1, 'byte'), Q_(1, 'm'),
        ...     Q_(1, 'm^3'), [Q_(0.1, 'm')], 8)
        >>> result.query_time == Q_(1, 'ms')
        True
        """
        tick = (1 / tick_rate).to(ureg.microsecond)
        t_query = (collision_allocation / hulls).to(ureg.microsecond)
        budget = _ratio(t_query * sram_bandwidth, esdf_cell_bytes)
        reach = (hull_radius + protective_clearance).to(ureg.meter)
        rows = {}
        for r in pitches:
            scan = calc_sphere_scan_cells(reach, r)
            depth = calc_octree_depth(map_edge, r)
            cells = _ratio(slab_volume, r ** 3)
            rows[round(r.to(ureg.centimeter).magnitude)] = ClearanceQueryRow(
                pitch=r.to(ureg.centimeter), occupancy_reads=scan, octree_depth=depth, octree_reads=scan * depth,
                esdf_reads=trilinear_reads, esdf_bytes=(cells * esdf_cell_bytes).to(ureg.kilobyte),
                occupancy_bytes=(cells * occupancy_cell_bytes).to(ureg.kilobyte),
                quantization=calc_voxel_quantization_error(r),
            )
        return ClearanceQueryResult(
            tick=tick, query_time=t_query, tick_spare_share=1 - _ratio(collision_allocation, tick),
            reads_budget=budget, reach=reach, slab_volume=slab_volume.to(ureg.meter ** 3), rows=rows,
            wcet_share=None if wcet is None else _ratio(collision_allocation, wcet),
        )


class HostMapSizingModel(ForwardModel):
    """
    A dense voxel grid of a workcell against a hashed narrow band around its surfaces.

    Dense storage is ``V / r^3`` voxels; the band of half-width ``delta``
    around surface area ``A`` has volume ``2 delta A`` and fills
    ``V_band / (B r)^3`` blocks of ``B^3`` voxels each.
    """
    requires = ("extent", "pitch", "voxel_bytes", "surface_area", "band_half_width", "block_edge")
    produces = HostMapSizingResult

    def solve(self, extent: Sequence[Quantity], pitch: Quantity, voxel_bytes: Quantity, surface_area: Quantity,
              band_half_width: Quantity, block_edge: int) -> HostMapSizingResult:
        """Count dense-grid storage and ideal narrow-band block storage.

        Parameters
        ----------
        extent : sequence of Quantity
            Three positive workspace side lengths, in meters.
        pitch : Quantity
            Positive cubic voxel width, in meters.
        voxel_bytes : Quantity
            Positive bytes stored per voxel, excluding hash-table metadata.
        surface_area : Quantity
            Positive mapped surface area, in square meters.
        band_half_width : Quantity
            Positive distance stored on each side of the surface, in meters.
        block_edge : int
            Positive voxels per side of each cubic block.

        Returns
        -------
        HostMapSizingResult
            Workspace and band volumes in cubic meters, ideal cell/block counts,
            dense storage in GB, band storage in MB, and dimensionless ratios.
            Fractional block counts are retained; memory admission is not checked.

        Notes
        -----
        Equation: V = extent[0] * extent[1] * extent[2]; N_dense = V / ``pitch**3``;
            V_band = 2 * band_half_width * surface_area. N_blocks =
            V_band / (block_edge * ``pitch)**3`` and each block stores
            ``block_edge**3`` * voxel_bytes.
        Assumptions: flat locally independent surface bands, identical voxel payloads,
            and continuously divisible block occupancy.
        Excludes: band overlap, workspace clipping, partial-block rounding, hash
            keys, allocator overhead, and GPU residency constraints.

        References
        ----------
        Derivation: volume divided by cubic-cell volume gives cell count;
            multiplying a surface area by its two-sided band thickness gives the
            narrow-band volume used in this geometric approximation.

        Examples
        --------
        A constructed cubic workspace exposes the volume accounting directly.

        >>> from mlsysim import Q_
        >>> result = HostMapSizingModel().solve(
        ...     [Q_(1, 'm')] * 3, Q_(0.1, 'm'), Q_(4, 'byte'),
        ...     Q_(1, 'm^2'), Q_(0.05, 'm'), 4)
        >>> result.workspace_volume == Q_(1, 'm^3')
        True
        """
        lx, ly, lz = extent
        v_ws = (lx * ly * lz).to(ureg.meter ** 3)
        n_dense = _ratio(v_ws, pitch ** 3)
        dense = (n_dense * voxel_bytes).to(ureg.gigabyte)
        v_band = (2 * band_half_width * surface_area).to(ureg.meter ** 3)
        blocks = _ratio(v_band, (block_edge * pitch) ** 3)
        hashed = (blocks * block_edge ** 3 * voxel_bytes).to(ureg.megabyte)
        return HostMapSizingResult(
            workspace_volume=v_ws, dense_voxels=n_dense, dense_bytes=dense, band_volume=v_band,
            band_share=_ratio(v_band, v_ws), blocks=blocks,
            block_bytes=(block_edge ** 3 * voxel_bytes).to(ureg.byte), hashed_bytes=hashed,
            ratio=_ratio(dense, hashed),
        )


class RaymarchTrafficModel(ForwardModel):
    """Naive ray-marching map-update traffic ``f N_rays (d / r) S`` against a memory bandwidth."""
    requires = ("width_px", "height_px", "frame_rate", "ray_length", "pitch", "bytes_per_step", "bandwidth")
    produces = RaymarchTrafficResult

    def solve(self, width_px: int, height_px: int, frame_rate: Quantity, ray_length: Quantity, pitch: Quantity,
              bytes_per_step: Quantity, bandwidth: Quantity) -> RaymarchTrafficResult:
        """Estimate uncached voxel-update traffic for a depth-image stream.

        Parameters
        ----------
        width_px : int
            Positive image width, in pixels.
        height_px : int
            Positive image height, in pixels.
        frame_rate : Quantity
            Positive depth-frame rate, in Hz.
        ray_length : Quantity
            Nonnegative traversed distance per ray, in meters.
        pitch : Quantity
            Positive distance between ray samples, in meters.
        bytes_per_step : Quantity
            Nonnegative bytes transferred per ray sample, in bytes.
        bandwidth : Quantity
            Positive available memory bandwidth, in bytes per second.

        Returns
        -------
        RaymarchTrafficResult
            Ray count per frame, steps per ray, traffic in GB/s, and traffic divided
            by available bandwidth. A share above one identifies a bandwidth deficit
            under the stated access model.

        Notes
        -----
        Equation: rays = width_px * height_px; steps = ray_length / pitch;
            traffic = frame_rate * rays * steps * bytes_per_step.
        Assumptions: every pixel emits one equal-length ray and every step causes
            the supplied byte transfer. Fractional step counts are retained.
        Excludes: repeated-voxel reuse, culling, empty-space skipping, compression,
            contention with other clients, and integer step rounding.

        References
        ----------
        Derivation: multiply frames per second, rays per frame, steps per ray,
            and bytes per step; the units reduce to bytes per second.

        Examples
        --------
        Doubling the constructed frame rate doubles the traffic.

        >>> from mlsysim import Q_
        >>> model = RaymarchTrafficModel()
        >>> args = (2, 2, Q_(1, 'Hz'), Q_(1, 'm'), Q_(0.1, 'm'),
        ...         Q_(4, 'byte'), Q_(1, 'GB/s'))
        >>> first = model.solve(*args)
        >>> second = model.solve(*args[:2], args[2] * 2, *args[3:])
        >>> second.traffic == first.traffic * 2
        True
        """
        rays = width_px * height_px
        steps = _ratio(ray_length, pitch)
        traffic = (frame_rate * rays * steps * bytes_per_step).to(ureg.gigabyte / ureg.second)
        return RaymarchTrafficResult(rays=rays, steps=steps, traffic=traffic, share=_ratio(traffic, bandwidth))


class EncoderRateModel(ForwardModel):
    """A per-frame encoder of ``flop_per_frame`` run at ``frame_rate``, against a ``peak`` rate."""
    requires = ("flop_per_frame", "frame_rate", "peak")
    produces = EncoderRateResult

    def solve(self, flop_per_frame: Quantity, frame_rate: Quantity, peak: Quantity) -> EncoderRateResult:
        """Compare frame-encoder work demand with an available compute rate.

        Parameters
        ----------
        flop_per_frame : Quantity
            Nonnegative arithmetic work for one frame, in FLOPs.
        frame_rate : Quantity
            Nonnegative frame arrival rate, in Hz.
        peak : Quantity
            Positive compute capacity, in FLOPs per second. Supply an effective
            rate instead if utilization losses should already be included.

        Returns
        -------
        EncoderRateResult
            Required and supplied rates in the units of peak, and their ratio.
            The field shortfall is demand/capacity, not a percentage or latency.

        Notes
        -----
        Equation: required = flop_per_frame * frame_rate; shortfall = required / peak.
        Assumptions: every frame executes the same work, and throughput can be
            compared with the supplied compute rate.
        Excludes: memory traffic, queueing, launch overhead, and deadline guarantees.

        References
        ----------
        Derivation: work per frame multiplied by frames per second is work per
            second; division by supply gives dimensionless capacity demand.

        Examples
        --------
        A constructed stream whose demand equals supply has unit demand ratio.

        >>> from mlsysim import Q_
        >>> result = EncoderRateModel().solve(Q_(10, 'flop'), Q_(2, 'Hz'), Q_(20, 'flop/s'))
        >>> result.shortfall == 1
        True
        """
        required = (flop_per_frame * frame_rate).to(peak.units)
        return EncoderRateResult(required=required, peak=peak, shortfall=_ratio(required, peak))
