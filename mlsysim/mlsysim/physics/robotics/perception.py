"""Spatial memory footprints, evidence horizons, and teleoperation data budgets."""

from __future__ import annotations

import math

from mlsysim.core.units import ureg
from mlsysim.core._validation import validate_at_least, validate_nonnegative, validate_positive


def calc_spatial_information_age_displacement(velocity, information_age):
    """
    Calculate unmodeled spatial displacement of a moving body across perception pipeline delay.

    Equation:
        Delta_x = v * tau_age

    Source: OpenStax, University Physics Volume 1, Sec. 3.1. Physical AI perception pipeline
    waterfall (exposure, readout, transport, DMA, ISP, backbone, IPC).

    Parameters
    ----------
    velocity : Quantity
        Operational forward or end-effector translational velocity (e.g. m/s).
    information_age : Quantity
        Total sensor-to-inference latency / information age tau_age (e.g. ms or s).

    Returns
    -------
    Quantity
        Unmodeled physical displacement in meters.
    """
    validate_nonnegative(velocity, "velocity")
    validate_positive(information_age, "information_age")
    dx = velocity * information_age
    return dx.to(ureg.meter)


def calc_3dgs_memory_footprint(num_gaussians: int, bytes_per_gaussian: int = 56):
    """
    Calculate spatial state memory footprint for 3D Gaussian Splatting (3DGS).

    Equation:
        Memory = K_gaussians * bytes_per_gaussian

    Parameters
    ----------
    num_gaussians : int
        Number of 3D Gaussian ellipsoids in the spatial memory buffer.
    bytes_per_gaussian : int, optional
        Storage footprint per Gaussian (position 12B, covariance/scale/rotation 16B,
        opacity 4B, color coefficients 24B = 56 bytes in this assumed packing;
        actual storage varies with harmonic degree, precision, and implementation).

    Returns
    -------
    Quantity
        Memory footprint in bytes.
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
    """
    Calculate partial TSDF storage and modeled DRAM traffic for dense vs sparse grids.

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
        BW_sparse_dram = R_rays * (N_free_occupancy + N_TSDF_band) * B_lookup * (1 - Hit_cache)

    Parameters
    ----------
    workspace_volume : Quantity
        Total bounding volume of operational workcell (e.g. 300 m^3).
    voxel_size : Quantity
        Isotropic voxel side length (e.g. 1.0 cm or 5.0 mm).
    bytes_per_voxel : int
        TSDF payload bytes per voxel (default 4: 16-bit distance + 16-bit weight).
    sparsity_ratio : float
        Surface occupancy fraction within bounding volume (default 0.03 = 3%).
    block_size_voxels : int
        Voxel count per hierarchical leaf block (default 512 for 8x8x8).
    block_header_bytes : int
        Memory overhead per block metadata/pointer header (default 32 B).
    ray_rate : Quantity, optional
        Raycast integration query rate (e.g., 640x480x60 = 18.432e6 rays/s).
    dense_voxels_per_ray : int
        Raycast integration depth across dense volume (default 300 lookups).
    sparse_voxels_per_ray : int
        TSDF fusion updates in the narrow surface band (default 6 lookups).
    free_voxels_per_ray : int
        Separate occupancy free-ray updates before the hit (default 294). Sparse
        TSDF allocation does not remove this traversal or its occupancy work.
    bytes_per_ray_voxel : int
        Read-modify-write memory traffic per ray step (default 8 B).
    cache_hit_rate : float
        Assumed measured SRAM/L2 hit fraction for both occupancy and TSDF updates
        (default 0.95 for the illustrative scenario; not a sparse-map property).

    Returns
    -------
    dict
        Voxel count, dense TSDF payload, sparse blocks, sparse TSDF payload
        plus leaf headers, and modeled DRAM bus bandwidths. The legacy
        ``dense_memory`` and ``sparse_memory`` keys are not full-map totals.
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
        sparse_dram_raw = r_rate * (free_voxels_per_ray + sparse_voxels_per_ray) * bytes_per_ray_voxel * (1.0 - cache_hit_rate)

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
    """
    Calculate the target-evidence validity horizon under a bounded drift model.

    This is not a stopping-time or stopping-distance bound. Motion admission also
    needs current clearance, total pre-brake delay, achievable deceleration, and
    a feasible fallback after this evidence expires.

    Equation:
        tau = (r_tol - e_0) / v_drift

    Parameters
    ----------
    tolerance_radius : Quantity
        Defended task spatial tolerance radius (e.g. mm or m).
    sensor_noise : Quantity
        Initial target-error allowance e_0 (legacy parameter name). It is a hard
        bound only when the sensing model justifies one; a quantile gives a
        validity horizon at its declared tail risk. Must be < tolerance_radius.
    drift_velocity : Quantity
        Bounded relative target drift under the declared operating envelope
        (e.g. m/s), or a calibrated rate at a declared tail risk.

    Returns
    -------
    Quantity
        Maximum target-evidence age in milliseconds.
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
    """
    Calculate the tracked target-evidence horizon under residual slip acceleration.

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

    Parameters
    ----------
    tolerance_radius : Quantity
        Defended task spatial tolerance radius r_tol (e.g. mm).
    initial_error : Quantity
        Initial target-error allowance e_0; must be >= 0 and < tolerance_radius.
    slip_acceleration : Quantity
        Bounded residual target acceleration a_slip after tracking (m/s^2).

    Returns
    -------
    Quantity
        Tracked evidence horizon in milliseconds.
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
    """
    Calculate the invalidation horizon of a belief whose error bound grows linearly.

    Equation:
        E(dt) = E_0 + r * dt ;  T_inv = max(0, (E_max - E_0) / r)

    A zero result means the belief never supported the task: its initial error
    already consumes the tolerance. Unlike ``calc_target_evidence_horizon`` this
    does not raise when E_0 >= E_max; it floors the deadline at zero.

    Source: first-passage time of a linear error bound (definition of the
    invalidation horizon, T_inv = inf{dt >= 0 : E(dt) >= E_max}); the
    reachable-set lineage for occluded free space is Orzechowski, Meyer, and
    Lauer 2018 (IEEE ITSC).
    """
    validate_positive(error_growth_rate, "error_growth_rate")
    e0 = initial_error.to(ureg.meter).magnitude
    e_max = error_limit.to(ureg.meter).magnitude
    r = error_growth_rate.to(ureg.meter / ureg.second).magnitude
    return (max(e_max - e0, 0.0) / r * ureg.second).to(ureg.millisecond)


def calc_free_space_expiry(free_depth, loc_bound, protective_clearance, approach_speed):
    """
    Calculate how long a free-space claim survives before an unseen person could breach the clearance.

    Equation:
        T_inv = max(0, (d_free - delta_loc - delta_margin) / v_h)

    The free region erodes from its unobserved boundary at the approach speed
    v_h; localization error delta_loc and the protective clearance delta_margin
    are spent up front. It is ``calc_linear_invalidation_horizon`` with
    E_0 = delta_loc, r = v_h, and E_max = d_free - delta_margin.

    Source: linear invalidation horizon applied to occluded free space; the
    walking-speed bound v_h is ISO 13855's K (1.6 m/s).
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
    """
    Calculate sustained data ingestion bandwidth and hourly/shift storage accumulation
    for multi-camera robotic teleoperation rigs.

    Equation:
        BW_rgb = num_cameras * (W_rgb * H_rgb * Bpp_rgb) * FPS_rgb
        BW_depth = num_cameras * (W_depth * H_depth * Bpp_depth) * FPS_depth
        BW_kinematics = bytes_per_sec_kinematics
        BW_tactile = bytes_per_sec_tactile
        BW_total = BW_rgb + BW_depth + BW_kinematics + BW_tactile
        Storage_1hr = BW_total * 3600 s

    Parameters
    ----------
    num_cameras : int
        Number of active camera streams (e.g., 4).
    rgb_width, rgb_height : int
        Spatial resolution of RGB streams (e.g., 1920, 1080).
    rgb_fps : float
        Frame rate of RGB streams in Hz (e.g., 30.0).
    rgb_bytes_per_pixel : int
        Color encoding byte depth (default 3 bytes for 24-bit RGB).
    depth_width, depth_height : int
        Spatial resolution of depth streams (e.g., 1280, 720).
    depth_fps : float
        Frame rate of depth streams in Hz (e.g., 30.0).
    depth_bytes_per_pixel : int
        Depth encoding byte depth (default 2 bytes for uint16).
    kinematics_bytes_per_sec : Quantity, optional
        Data rate for joint positions, velocities, torques (e.g., 128 B * 1000 Hz).
    tactile_bytes_per_sec : Quantity, optional
        Data rate for tactile array telemetry (e.g., 2048 B * 100 Hz).

    Returns
    -------
    dict
        Breakdown of RGB, depth, vision total, kinematics, tactile, total bandwidth,
        and hourly uncompressed storage consumption.
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

    bw_kin = (0.0 * (ureg.megabyte / ureg.second)) if kinematics_bytes_per_sec is None else kinematics_bytes_per_sec.to(ureg.megabyte / ureg.second)
    bw_tac = (0.0 * (ureg.megabyte / ureg.second)) if tactile_bytes_per_sec is None else tactile_bytes_per_sec.to(ureg.megabyte / ureg.second)

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
