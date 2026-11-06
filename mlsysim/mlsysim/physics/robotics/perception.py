"""Spatial memory footprints, evidence horizons, and teleoperation data budgets."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least, validate_nonnegative, validate_positive


def calc_spatial_information_age_displacement(velocity, information_age):
    """Compute displacement during a perception delay.

    Parameters
    ----------
    velocity : Quantity
        Nonnegative constant translational speed in length / time.
    information_age : Quantity
        Positive sensor-to-use age in time units.

    Returns
    -------
    Quantity
        Displacement in meters.

    Notes
    -----
    Equation: displacement = velocity * information_age.

    Assumptions: The supplied speed remains constant over the age interval.

    Excludes: Acceleration, object motion not included in velocity, and collision
    assessment.

    Equation:
        Delta_x = v * tau_age

    Source: OpenStax, University Physics Volume 1, Sec. 3.1. Physical AI perception
    pipeline
    waterfall (exposure, readout, transport, DMA, ISP, backbone, IPC).

    References
    ----------
    Derivation: Integrate constant velocity over the elapsed perception age.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_spatial_information_age_displacement
    >>> from mlsysim.core.units import ureg
    >>> distance = calc_spatial_information_age_displacement(1 * ureg.meter / ureg.second, 1 * ureg.second)
    >>> assert distance == 1 * ureg.meter
    """
    validate_nonnegative(velocity, "velocity")
    validate_positive(information_age, "information_age")
    dx = velocity * information_age
    return dx.to(ureg.meter)


def calc_3dgs_memory_footprint(num_gaussians: int, bytes_per_gaussian: int = 56):
    """Count Gaussian-state storage under an explicit packing.

    Parameters
    ----------
    num_gaussians : int
        Positive count of Gaussian state records.
    bytes_per_gaussian : int, optional
        Positive bytes per record; default 56 is an assumed packing, not a universal 3DGS
        layout.

    Returns
    -------
    Quantity
        Packed state storage in bytes.

    Notes
    -----
    Equation: memory = num_gaussians * bytes_per_gaussian.

    Assumptions: Each record uses the same caller-specified packing.

    Excludes: Optimizer state, allocator overhead, renderer workspaces, and additional
    harmonic coefficients.

    Equation:
        Memory = K_gaussians * bytes_per_gaussian

    References
    ----------
    Derivation: Count records and multiply by storage per record; the packing is an input
    convention.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_3dgs_memory_footprint
    >>> from mlsysim.core.units import ureg
    >>> small = calc_3dgs_memory_footprint(10)
    >>> large = calc_3dgs_memory_footprint(20)
    >>> assert large == 2 * small
    """
    validate_at_least(num_gaussians, 1, "num_gaussians")
    validate_at_least(bytes_per_gaussian, 1, "bytes_per_gaussian")

    total_bytes = num_gaussians * bytes_per_gaussian
    return total_bytes * ureg.byte


def calc_tsdf_voxel_grid_budget(
    workspace_volume,
    voxel_size,
    bytes_per_voxel: int = 4,
    sparsity_ratio: float = 0.03,
    block_size_voxels: int = 512,
    block_header_bytes: int = 32,
    ray_rate=None,
    dense_voxels_per_ray: int = 300,
    sparse_voxels_per_ray: int = 6,
    free_voxels_per_ray: int = 294,
    bytes_per_ray_voxel: int = 8,
    cache_hit_rate: float = 0.95,
):
    """Estimate TSDF payload storage and assumed traversal traffic.

    Parameters
    ----------
    workspace_volume : Quantity
        Positive workspace bounding volume, convertible to cubic meters.
    voxel_size : Quantity
        Positive isotropic voxel edge length.
    bytes_per_voxel : int, optional
        Positive payload bytes per voxel; default 4 stores distance and fusion weight.
    sparsity_ratio : float, optional
        Active voxel fraction in [0, 1]; default 0.03 is an assumed scenario.
    block_size_voxels : int, optional
        Positive voxel capacity per allocated leaf block; default 512.
    block_header_bytes : int, optional
        Nonnegative metadata bytes per block; default 32.
    ray_rate : Quantity or float, optional
        Nonnegative integration rays per second; None omits bandwidth outputs.
    dense_voxels_per_ray : int, optional
        Nonnegative dense lookups per ray; default 300.
    sparse_voxels_per_ray : int, optional
        Nonnegative TSDF-band updates per ray; default 6.
    free_voxels_per_ray : int, optional
        Nonnegative free-space occupancy updates per ray; default 294.
    bytes_per_ray_voxel : int, optional
        Nonnegative read/write bytes per lookup; default 8.
    cache_hit_rate : float, optional
        Assumed cache-hit fraction in [0, 1]; default 0.95, not an inherent sparse-map
        property.

    Returns
    -------
    dict
        Integer voxel/block counts, dense_memory in GB, sparse_memory in MB; optional
        dense_dram_bandwidth in GB/s and sparse_dram_bandwidth in MB/s.

    Notes
    -----
    Equation: N = round(V / ``pitch**3)``; A = ``round(N*sparsity)``; blocks = ceil(A /
    block_size); sparse bytes = ``blocks*(block_size*voxel_bytes`` + header); traffic =
    ``ray_rate*lookups*lookup_bytes``, discounted by sparse cache misses.

    Assumptions: Fixed voxel packing, rounded voxel counts, full leaf allocation, and
    caller-specified traversal/cache behavior.

    Excludes: Complete occupancy-state storage, evidence epochs, allocator/index metadata,
    cache capacity analysis, and measured traffic prediction.

    The four-byte voxel stores distance and fusion weight only. The sparse result
    includes the stipulated leaf header, but neither result includes a separate
    occupancy layer, per-cell evidence epochs, lookup/allocator metadata, or
    other full-map state. The hit rate is an assumed workload input, not a
    property of sparse allocation.

    Equation:
        N_dense = V_workspace / (Delta_v)^3
        M_dense = N_dense * B_voxel
        N_active = N_dense * S_sparsity
        N_blocks = ceil(N_active / B_block)
        M_sparse = N_blocks * (B_block * B_voxel + B_header)
        BW_dense_dram = R_rays * N_dense_lookups * B_lookup
        BW_sparse_dram = R_rays * (N_free_occupancy + N_TSDF_band) * B_lookup * (1 -
        Hit_cache)

    References
    ----------
    Derivation: Count a regular grid, round active occupancy, allocate whole blocks, and
    sum per-ray memory transactions; this is explicit byte accounting, not a claim about a
    particular mapper.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_tsdf_voxel_grid_budget
    >>> from mlsysim.core.units import ureg
    >>> budget = calc_tsdf_voxel_grid_budget(1 * ureg.meter**3, 0.1 * ureg.meter)
    >>> assert budget["dense_voxel_count"] == 1000
    >>> assert budget["sparse_block_count"] == 1
    """
    validate_positive(workspace_volume, "workspace_volume")
    validate_positive(voxel_size, "voxel_size")
    if not (0.0 <= cache_hit_rate <= 1.0):
        raise ValueError(f"cache_hit_rate must be in [0, 1], got {cache_hit_rate}")

    v_vol = workspace_volume.to(ureg.meter**3).magnitude
    v_vox = (voxel_size.to(ureg.meter).magnitude) ** 3
    n_dense = int(round(v_vol / v_vox))

    m_dense_bytes = n_dense * bytes_per_voxel * ureg.byte
    m_dense = m_dense_bytes.to(ureg.gigabyte)

    n_active = int(round(n_dense * sparsity_ratio))
    n_blocks = math.ceil(n_active / block_size_voxels)
    bytes_per_block = block_size_voxels * bytes_per_voxel + block_header_bytes
    m_sparse_bytes = n_blocks * bytes_per_block * ureg.byte
    m_sparse = m_sparse_bytes.to(ureg.megabyte)

    result = {
        "dense_voxel_count": n_dense,
        "dense_memory": m_dense,
        "active_voxel_count": n_active,
        "sparse_block_count": n_blocks,
        "sparse_memory": m_sparse,
    }

    if ray_rate is not None:
        r_rate = ray_rate.to(1 / ureg.second).magnitude if hasattr(ray_rate, "magnitude") else float(ray_rate)
        dense_dram_raw = r_rate * dense_voxels_per_ray * bytes_per_ray_voxel
        validate_at_least(free_voxels_per_ray, 0, "free_voxels_per_ray")
        sparse_dram_raw = (
            r_rate * (free_voxels_per_ray + sparse_voxels_per_ray) * bytes_per_ray_voxel * (1.0 - cache_hit_rate)
        )

        result["dense_dram_bandwidth"] = (dense_dram_raw * (ureg.byte / ureg.second)).to(ureg.gigabyte / ureg.second)
        result["sparse_dram_bandwidth"] = (sparse_dram_raw * (ureg.byte / ureg.second)).to(ureg.megabyte / ureg.second)
        result["free_occupancy_updates_per_ray"] = free_voxels_per_ray
        result["tsdf_band_updates_per_ray"] = sparse_voxels_per_ray
        result["traversed_voxels_per_ray"] = free_voxels_per_ray + sparse_voxels_per_ray

    return result


def calc_target_evidence_horizon(
    tolerance_radius,
    sensor_noise,
    drift_velocity,
):
    """Find when constant target drift consumes an error margin.

    Parameters
    ----------
    tolerance_radius : Quantity
        Positive spatial tolerance in length units.
    sensor_noise : Quantity
        Positive initial error allowance below tolerance_radius; a quantile is not a hard
        bound.
    drift_velocity : Quantity
        Positive drift-speed allowance in length / time.

    Returns
    -------
    Quantity
        Target-evidence validity horizon in milliseconds.

    Notes
    -----
    Equation: horizon = (tolerance_radius - sensor_noise) / drift_velocity.

    Assumptions: An initial scalar error plus constant bounded drift; interpretation
    follows the supplied allowances.

    Excludes: Braking, fallback feasibility, clearance checks, and safety certification.

    This is not a stopping-time or stopping-distance bound. Motion admission also
    needs current clearance, total pre-brake delay, achievable deceleration, and
    a feasible fallback after this evidence expires.

    Equation:
        tau = (r_tol - e_0) / v_drift

    References
    ----------
    Derivation: Solve initial error + ``drift*time`` = tolerance for time.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_target_evidence_horizon
    >>> from mlsysim.core.units import ureg
    >>> h = calc_target_evidence_horizon(2 * ureg.meter, 1 * ureg.meter, 1 * ureg.meter / ureg.second)
    >>> assert h == 1 * ureg.second
    """
    validate_positive(tolerance_radius, "tolerance_radius")
    validate_positive(sensor_noise, "sensor_noise")
    validate_positive(drift_velocity, "drift_velocity")

    r_tol = tolerance_radius.to(ureg.meter)
    sig = sensor_noise.to(ureg.meter)
    if sig >= r_tol:
        raise ValueError("sensor_noise must be strictly less than tolerance_radius.")

    v_drift = drift_velocity.to(ureg.meter / ureg.second)
    tau = (r_tol - sig) / v_drift
    return tau.to(ureg.millisecond)


def calc_tracked_evidence_horizon(tolerance_radius, initial_error, slip_acceleration):
    """Find when residual acceleration consumes a tracked error margin.

    Parameters
    ----------
    tolerance_radius : Quantity
        Positive spatial tolerance in length units.
    initial_error : Quantity
        Nonnegative initial error below tolerance_radius.
    slip_acceleration : Quantity
        Positive residual relative acceleration in length / time**2.

    Returns
    -------
    Quantity
        Tracked-evidence horizon in milliseconds.

    Notes
    -----
    Equation: horizon = ``sqrt(2*(tolerance_radius`` - initial_error)/slip_acceleration).

    Assumptions: Tracking removes initial relative velocity; residual acceleration is
    bounded by the input.

    Excludes: Velocity-estimation error, braking, and proof of an acceleration bound.

    A tracker that follows a moving target removes its steady velocity, so the
    remaining error growth comes from a bounded residual acceleration a_slip.
    A segment executed open-loop for a time T then drifts by a_slip T^2 / 2;
    setting that drift equal to the error margin r_tol - e_0 gives

    Equation:
        tau_ev = sqrt(2 (r_tol - e_0) / a_slip)

    This is the constant-acceleration counterpart of
    ``calc_target_evidence_horizon`` (constant drift velocity, a_slip = 0).
    Like that function it is not a stopping-time bound.

    Source: first-passage time of a constant-acceleration displacement
    (d = a T^2 / 2, solved for T at d = r_tol - e_0); kinematics, derived here.

    References
    ----------
    Derivation: Integrate constant acceleration from zero residual velocity, then solve
    displacement = remaining margin.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_tracked_evidence_horizon
    >>> from mlsysim.core.units import ureg
    >>> h = calc_tracked_evidence_horizon(2 * ureg.meter, 0 * ureg.meter, 1 * ureg.meter / ureg.second**2)
    >>> assert h == 2 * ureg.second
    """
    validate_positive(tolerance_radius, "tolerance_radius")
    validate_nonnegative(initial_error, "initial_error")
    validate_positive(slip_acceleration, "slip_acceleration")

    margin = (tolerance_radius - initial_error).to(ureg.meter)
    if margin.magnitude <= 0:
        raise ValueError("initial_error must be strictly less than tolerance_radius.")
    a = slip_acceleration.to(ureg.meter / ureg.second**2)
    return ((2 * margin / a) ** 0.5).to(ureg.millisecond)


def calc_linear_invalidation_horizon(initial_error, error_growth_rate, error_limit):
    """Find the first passage of a linearly growing error allowance.

    Parameters
    ----------
    initial_error : Quantity
        Initial error allowance in length units.
    error_growth_rate : Quantity
        Positive error-growth rate in length / time.
    error_limit : Quantity
        Allowed error in length units; a spent limit returns zero.

    Returns
    -------
    Quantity
        Invalidation horizon in milliseconds, floored at zero.

    Notes
    -----
    Equation: horizon = max(error_limit - initial_error, 0) / error_growth_rate.

    Assumptions: Monotone constant-rate growth of a scalar error allowance.

    Excludes: Acceleration, new evidence, stopping capability, and confidence calibration.

    Equation:
        E(dt) = E_0 + r * dt ;  T_inv = max(0, (E_max - E_0) / r)

    A zero result means the belief never supported the task: its initial error
    already consumes the tolerance. Unlike ``calc_target_evidence_horizon`` this
    does not raise when E_0 >= E_max; it floors the deadline at zero.

    Source: first-passage time of a linear error bound (definition of the
    invalidation horizon, T_inv = inf{dt >= 0 : E(dt) >= E_max}); the
    reachable-set lineage for occluded free space is Orzechowski, Meyer, and
    Lauer 2018 (IEEE ITSC).

    References
    ----------
    Derivation: Set a linear error envelope equal to the limit; clamp past crossings to
    time zero.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_linear_invalidation_horizon
    >>> from mlsysim.core.units import ureg
    >>> h = calc_linear_invalidation_horizon(2 * ureg.meter, 1 * ureg.meter / ureg.second, 1 * ureg.meter)
    >>> assert h.magnitude == 0
    """
    validate_positive(error_growth_rate, "error_growth_rate")
    e0 = initial_error.to(ureg.meter).magnitude
    e_max = error_limit.to(ureg.meter).magnitude
    r = error_growth_rate.to(ureg.meter / ureg.second).magnitude
    return (max(e_max - e0, 0.0) / r * ureg.second).to(ureg.millisecond)


def calc_free_space_expiry(free_depth, loc_bound, protective_clearance, approach_speed):
    """Compute a free-space claim's erosion horizon.

    Parameters
    ----------
    free_depth : Quantity
        Observed free depth in length units.
    loc_bound : Quantity
        Nonnegative localization allowance in length units.
    protective_clearance : Quantity
        Nonnegative clearance reserved before an approaching object reaches the task
        region.
    approach_speed : Quantity
        Positive assumed approach speed in length / time; supplied by the caller.

    Returns
    -------
    Quantity
        Expiry time in milliseconds, floored at zero.

    Notes
    -----
    Equation: expiry = max(free_depth - loc_bound - protective_clearance, 0) /
    approach_speed.

    Assumptions: An unseen object enters from the free-space boundary at a bounded
    constant speed.

    Excludes: Occlusion geometry beyond a scalar depth, braking, speed-bound validation,
    and standards compliance.

    Equation:
        T_inv = max(0, (d_free - delta_loc - delta_margin) / v_h)

    The free region erodes from its unobserved boundary at the approach speed
    v_h; localization error delta_loc and the protective clearance delta_margin
    are spent up front. It is ``calc_linear_invalidation_horizon`` with
    E_0 = delta_loc, r = v_h, and E_max = d_free - delta_margin.

    Source: linear invalidation horizon applied to occluded free space; the
    walking-speed bound v_h is ISO 13855's K (1.6 m/s).

    References
    ----------
    Derivation: Apply the linear invalidation horizon with initial_error = loc_bound and
    error_limit = free_depth - protective_clearance.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_free_space_expiry
    >>> from mlsysim.core.units import ureg
    >>> h = calc_free_space_expiry(3 * ureg.meter, 1 * ureg.meter, 1 * ureg.meter, 1 * ureg.meter / ureg.second)
    >>> assert h == 1 * ureg.second
    """
    validate_positive(approach_speed, "approach_speed")
    return calc_linear_invalidation_horizon(loc_bound, approach_speed, free_depth - protective_clearance)


def calc_teleop_ingestion_budget(
    num_cameras: int,
    rgb_width: int,
    rgb_height: int,
    rgb_fps: float,
    rgb_bytes_per_pixel: int = 3,
    depth_width: int = 0,
    depth_height: int = 0,
    depth_fps: float = 0.0,
    depth_bytes_per_pixel: int = 2,
    kinematics_bytes_per_sec=None,
    tactile_bytes_per_sec=None,
):
    """Count uncompressed teleoperation traffic and hourly storage.

    Parameters
    ----------
    num_cameras : int
        Positive count of identical camera streams.
    rgb_width : int
        Positive RGB image width in pixels.
    rgb_height : int
        Positive RGB image height in pixels.
    rgb_fps : float
        Positive RGB frame frequency in frames per second.
    rgb_bytes_per_pixel : int, optional
        Positive RGB bytes per pixel; default 3.
    depth_width : int, optional
        Depth width in pixels; default zero disables depth.
    depth_height : int, optional
        Depth height in pixels; default zero disables depth.
    depth_fps : float, optional
        Depth frames per second; depth is enabled only when width, height, and frequency
        are positive.
    depth_bytes_per_pixel : int, optional
        Positive depth bytes per pixel; default 2.
    kinematics_bytes_per_sec : Quantity, optional
        Nonnegative kinematics bandwidth in bytes / time; None means zero.
    tactile_bytes_per_sec : Quantity, optional
        Nonnegative tactile bandwidth in bytes / time; None means zero.

    Returns
    -------
    dict
        RGB, depth, vision, kinematics, tactile, and total bandwidth in MB/s;
        storage_per_hour in TB.

    Notes
    -----
    Equation: camera bandwidth = ``cameras*width*height*bytes_per_pixel*fps``; total = RGB +
    depth + kinematics + tactile; hourly storage = ``total*3600`` seconds.

    Assumptions: Identical continuous camera streams and uncompressed fixed-size payloads.

    Excludes: Compression, transport headers, indexes, replication, and storage write
    amplification.

    Equation:
        BW_rgb = num_cameras * (W_rgb * H_rgb * Bpp_rgb) * FPS_rgb
        BW_depth = num_cameras * (W_depth * H_depth * Bpp_depth) * FPS_depth
        BW_kinematics = bytes_per_sec_kinematics
        BW_tactile = bytes_per_sec_tactile
        BW_total = BW_rgb + BW_depth + BW_kinematics + BW_tactile
        Storage_1hr = BW_total * 3600 s

    References
    ----------
    Derivation: Multiply pixels per frame by bytes per pixel and frames per second, sum
    independent streams, and integrate over one hour.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_teleop_ingestion_budget
    >>> from mlsysim.core.units import ureg
    >>> budget = calc_teleop_ingestion_budget(1, 10, 10, 1)
    >>> assert budget["bw_depth"].magnitude == 0
    >>> assert budget["storage_per_hour"] == (budget["bw_total"] * ureg.hour).to("terabyte")
    """
    validate_positive(num_cameras, "num_cameras")
    validate_positive(rgb_width, "rgb_width")
    validate_positive(rgb_height, "rgb_height")
    validate_positive(rgb_fps, "rgb_fps")

    bw_rgb_raw = num_cameras * (rgb_width * rgb_height * rgb_bytes_per_pixel) * rgb_fps
    bw_rgb = (bw_rgb_raw * (ureg.byte / ureg.second)).to(ureg.megabyte / ureg.second)

    if depth_width > 0 and depth_height > 0 and depth_fps > 0:
        bw_depth_raw = num_cameras * (depth_width * depth_height * depth_bytes_per_pixel) * depth_fps
        bw_depth = (bw_depth_raw * (ureg.byte / ureg.second)).to(ureg.megabyte / ureg.second)
    else:
        bw_depth = 0.0 * (ureg.megabyte / ureg.second)

    bw_vision = bw_rgb + bw_depth

    bw_kin = (
        (0.0 * (ureg.megabyte / ureg.second))
        if kinematics_bytes_per_sec is None
        else kinematics_bytes_per_sec.to(ureg.megabyte / ureg.second)
    )
    bw_tac = (
        (0.0 * (ureg.megabyte / ureg.second))
        if tactile_bytes_per_sec is None
        else tactile_bytes_per_sec.to(ureg.megabyte / ureg.second)
    )

    bw_total = bw_vision + bw_kin + bw_tac
    storage_1hr = (bw_total * (3600.0 * ureg.second)).to(ureg.terabyte)

    return {
        "bw_rgb": bw_rgb,
        "bw_depth": bw_depth,
        "bw_vision": bw_vision,
        "bw_kinematics": bw_kin,
        "bw_tactile": bw_tac,
        "bw_total": bw_total,
        "storage_per_hour": storage_1hr,
    }


# --- Clearance queries against volumetric maps (added 2026-10-20) ---


def calc_sphere_scan_cells(reach, pitch):
    """Estimate voxel reads from a spherical query volume.

    Parameters
    ----------
    reach : Quantity
        Positive sphere radius including any clearance allowance.
    pitch : Quantity
        Positive cubic voxel edge length.

    Returns
    -------
    float
        Dimensionless continuum estimate of cells, not an integer enumeration.

    Notes
    -----
    Equation: cells = ``(4*pi/3)`` * (reach / ``pitch)**3``.

    Assumptions: A dense read of every voxel within the spherical volume.

    Excludes: Boundary-cell rounding, early exits, spatial indexes, and clearance
    certification.

    Equation:
        N = (4/3) pi (reach / r)^3

    ``reach`` is the sphere radius plus the protective clearance; every cell
    within it must be read, because an occupancy cell answers only whether
    that one cell is occupied.

    Source: volume of a sphere in units of the voxel volume r^3.

    References
    ----------
    Derivation: Divide sphere volume by cubic voxel volume.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_sphere_scan_cells
    >>> from mlsysim.core.units import ureg
    >>> small = calc_sphere_scan_cells(1 * ureg.meter, 0.1 * ureg.meter)
    >>> large = calc_sphere_scan_cells(2 * ureg.meter, 0.1 * ureg.meter)
    >>> assert large == 8 * small
    """
    validate_positive(reach, "reach")
    validate_positive(pitch, "pitch")
    return 4 / 3 * math.pi * float((reach / pitch).to("").magnitude) ** 3


def calc_octree_depth(map_edge, pitch):
    """Count halvings from a map edge to a target voxel edge.

    Parameters
    ----------
    map_edge : Quantity
        Positive cubic map edge; the intended hierarchy has map_edge >= pitch.
    pitch : Quantity
        Positive target leaf edge in the same length dimension.

    Returns
    -------
    int
        Ceiling of the base-two edge ratio; can be negative for out-of-regime pitch >
        map_edge.

    Notes
    -----
    Equation: depth = ceil(log2(map_edge / pitch)).

    Assumptions: Each level halves an edge and a query descends the full hierarchy.

    Excludes: Allocation sparsity, early termination, memory latency, and automatic
    clamping of out-of-regime input.

    Equation:
        depth = ceil(log2(L / r))

    Each level is one dependent read in a point query.

    Source: Hornung et al., "OctoMap," Autonomous Robots 34(3), 2013, Sec. III.

    References
    ----------
    Derivation: After d binary subdivisions the edge is ``map_edge/2**d``; choose the least
    integer d attaining pitch.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_octree_depth
    >>> from mlsysim.core.units import ureg
    >>> assert calc_octree_depth(8 * ureg.meter, 1 * ureg.meter) == 3
    """
    validate_positive(map_edge, "map_edge")
    validate_positive(pitch, "pitch")
    return math.ceil(math.log2(float((map_edge / pitch).to("").magnitude)))


def calc_voxel_quantization_error(pitch):
    """Compute the maximum center-representation error of a cubic voxel.

    Parameters
    ----------
    pitch : Quantity
        Positive cubic voxel edge length.

    Returns
    -------
    Quantity
        Half the voxel diagonal in millimeters.

    Notes
    -----
    Equation: error = sqrt(3) * pitch / 2.

    Assumptions: A point anywhere in a cubic voxel is represented by its center.

    Excludes: Sensor error, map registration error, and noncubic cells.

    Equation:
        e_q = (sqrt(3) / 2) r

    Source: geometry of a cube of edge r.

    References
    ----------
    Derivation: Apply the three-dimensional Pythagorean theorem to a corner-to-center
    offset.

    Examples
    --------
    >>> from mlsysim.physics.robotics.perception import calc_voxel_quantization_error
    >>> from mlsysim.core.units import ureg
    >>> a = calc_voxel_quantization_error(1 * ureg.millimeter)
    >>> b = calc_voxel_quantization_error(2 * ureg.millimeter)
    >>> assert b == 2 * a
    """
    validate_positive(pitch, "pitch")
    return (math.sqrt(3) / 2 * pitch).to(ureg.millimeter)
