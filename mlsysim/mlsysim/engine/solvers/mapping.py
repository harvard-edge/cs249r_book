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
        rays = width_px * height_px
        steps = _ratio(ray_length, pitch)
        traffic = (frame_rate * rays * steps * bytes_per_step).to(ureg.gigabyte / ureg.second)
        return RaymarchTrafficResult(rays=rays, steps=steps, traffic=traffic, share=_ratio(traffic, bandwidth))


class EncoderRateModel(ForwardModel):
    """A per-frame encoder of ``flop_per_frame`` run at ``frame_rate``, against a ``peak`` rate."""
    requires = ("flop_per_frame", "frame_rate", "peak")
    produces = EncoderRateResult

    def solve(self, flop_per_frame: Quantity, frame_rate: Quantity, peak: Quantity) -> EncoderRateResult:
        required = (flop_per_frame * frame_rate).to(peak.units)
        return EncoderRateResult(required=required, peak=peak, shortfall=_ratio(required, peak))
