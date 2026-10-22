#!/usr/bin/env python3
"""Every number on the Lecture 11b (Hardware Acceleration, Part 2) slides and notes, read from MLSysIM.

MLSysIM holds the inputs (Hardware, Hardware.Tech, Models, Literature) and the
calculations (the mapping Scenarios.*: systolic fill and drain, matrix tiles,
waves, mapping choices, register occupancy, memory round trips); the Hardware
Acceleration chapter reads the same scenarios. This file only selects values,
converts units, and formats them; it types no numbers. Each value is recorded
with v() under the key the deck prints as \\val{key}; running this file writes
images/numbers.tex. make_figs.py imports this module, so figures and slides
share one computation.

Run from this folder: python3 calcs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "assets" / "kit"))
from deckvals import v, write, mlsysim  # noqa: E402
from mlsysim.core.series_loader import load_series  # noqa: E402
from mlsysim.engine.serving_profile import resnet50_h100_profile  # noqa: E402

Hardware, Literature, Scenarios = mlsysim.Hardware, mlsysim.Literature, mlsysim.Scenarios
TECH, CR, DM = Hardware.Tech, Literature.CompilerResults, Literature.DeviceMeasurements
H100, A100, TPU4 = Hardware.Cloud.H100, Hardware.Cloud.A100, Hardware.Cloud.TPUv4
RESNET = mlsysim.Models.Vision.ResNet50

# Boehm's worklog, kernel by kernel (label, GFLOP/s), in the article's order.
BOEHM = [("1 Naive", CR.BoehmSgemmNaiveGflops), ("2 Coalesced loads", CR.BoehmSgemmCoalescedGflops),
         ("3 Shared-memory tiles", CR.BoehmSgemmSharedMemGflops),
         ("4 1D register tiles", CR.BoehmSgemm1dBlocktilingGflops),
         ("5 2D register tiles", CR.BoehmSgemm2dBlocktilingGflops),
         ("6 Vector loads", CR.BoehmSgemmVectorizedGflops), ("9 Autotuned", CR.BoehmSgemmAutotunedGflops),
         ("10 Warp tiles", CR.BoehmSgemmWarptilingGflops), ("cuBLAS", CR.BoehmSgemmCublasGflops)]
BOEHM = [(label, float(g)) for label, g in BOEHM]
NAIVE, CUBLAS = BOEHM[0][1], BOEHM[-1][1]

S3, S3_INT8, S3_MIXED = (Scenarios.ResNet50Stage3ConvBatch32, Scenarios.ResNet50Stage3ConvBatch32Int8,
                         Scenarios.ResNet50Stage3ConvBatch32Int8Weights)
S2, S2_INT8, S2_FP32 = (Scenarios.ResNet50Stage2ConvBatch32, Scenarios.ResNet50Stage2ConvBatch32Int8,
                        Scenarios.ResNet50Stage2ConvBatch32Fp32)
TRACE2, SYS4 = Scenarios.SystolicArray2x2TwoRows, Scenarios.SystolicArray4x4EightRows
MXU_ROWS = (Scenarios.TpuV4MxuOneRow, Scenarios.TpuV4Mxu128Rows, Scenarios.TpuV4Mxu1024Rows,
            Scenarios.TpuV4Mxu4096Rows)
LAYER = Scenarios.Dense256To512Batch32WeightTiles
EX114 = Scenarios.Conv3x3Filters16On8x8
K1, K3 = Scenarios.BoehmSgemmNaiveKernel, Scenarios.BoehmSgemmSharedMemKernel
MM, MM_DBL, MM_K256, MM_256 = (Scenarios.Matmul4096Fp16Tile128, Scenarios.Matmul4096Fp16Tile128Double,
                               Scenarios.Matmul4096Fp16Tile128K256Double, Scenarios.Matmul4096Fp16Tile256x128)
FRINGE = Scenarios.Width129On128Array
WAVE1, WAVE2 = Scenarios.A100GemmOneWave, Scenarios.A100GemmTailWave
SPACE = Scenarios.MappingChoicesOneLevel
PROFILE = resnet50_h100_profile()
FLOOR16 = Scenarios.ResNet50Batch1H100Fp16PeakRate
FUSION = Scenarios.ElementwisePairFusion
REGS = (Scenarios.H100Registers32PerThread, Scenarios.H100Registers64PerThread,
        Scenarios.H100Registers128PerThread, Scenarios.H100RegistersMaxPerThread)
PAD = Scenarios.Width1000OnA100Fp16Multiple
TF32, FP32 = Scenarios.PytorchTf32MatmulGa100, Scenarios.PytorchFp32MatmulGa100
READ1, B_INT8, B_FP16 = (Scenarios.ResNet50Stage2MapInt8ReadOnce, Scenarios.ResNet50Stage2MapInt8Boundary,
                         Scenarios.ResNet50Stage2MapFp16Boundary)
B_HBM_LO, B_HBM_HI, B_DDR = (Scenarios.ResNet50Stage2MapInt8BoundaryHbm2Low,
                             Scenarios.ResNet50Stage2MapInt8BoundaryHbm2High,
                             Scenarios.ResNet50Stage2MapInt8BoundaryDdr7nm)
TILE_SERIES = load_series("slides/vol1/11b_hw_acceleration/images/data/nvidia_gemm_tile_quantization_a100.csv")


def m(q, unit):
    """Magnitude of a Pint quantity in ``unit``."""
    return q.to(unit).magnitude


def pct(x):
    """A fraction as percent."""
    return 100 * x


def lit(entry):
    """A Literature value as a plain number."""
    return float(entry)


print("== Hook: Boehm's SGEMM ladder (Literature.CompilerResults.BoehmSgemm*)")
v("boehm_naive", NAIVE, "{:.0f}", "naive kernel, GFLOP/s")
v("boehm_naive_1", NAIVE, "{:.1f}", "the same, one decimal")
v("boehm_cublas_1", CUBLAS, "{:,.1f}", "cuBLAS, GFLOP/s")
v("boehm_cublas", round(CUBLAS, -1), "{:,.0f}", "cuBLAS, to the ten")
v("boehm_total", CUBLAS / NAIVE, "{:.1f}", "cuBLAS over naive")
v("boehm_total_int", CUBLAS / NAIVE, "{:.0f}", "the same, whole")
STEPS = {}
for (a, ga), (b, gb) in zip(BOEHM, BOEHM[1:]):
    STEPS[b] = gb / ga
v("step_coalesced", STEPS["2 Coalesced loads"], "{:.1f}", "kernel 2 over kernel 1")
v("step_shared", STEPS["3 Shared-memory tiles"], "{:.1f}", "kernel 3 over kernel 2")
v("step_reg1d", STEPS["4 1D register tiles"], "{:.1f}", "kernel 4 over kernel 3")
v("step_reg2d", STEPS["5 2D register tiles"], "{:.1f}", "kernel 5 over kernel 4")

print("== Recall: H100 FP16 roofline and the stage-3 conv (Scenarios.ResNet50Stage3ConvBatch32*)")
r16 = m(H100.ridge_point(), "flop/byte")
r8 = m(H100.ridge_point("int8"), "flop/byte")
v("h100_fp16", m(H100.compute.peak_flops, "TFLOP/s"), "{:,.0f}", "H100 dense FP16, TFLOP/s")
v("h100_bw", m(H100.memory.bandwidth, "TB/s"), "{:.2f}", "H100 HBM, TB/s")
v("ridge16", r16, "{:.1f}", "FP16 ridge")
v("ridge16_int", r16, "{:.0f}", "FP16 ridge, whole")
v("ridge8", r8, "{:.1f}", "INT8 ridge")
v("ridge8_int", r8, "{:.0f}", "INT8 ridge, whole")
v("conv_ai", S3.traffic_intensity, "{:.1f}", "stage-3 conv intensity, FP16")
v("conv_ai_int", S3.traffic_intensity, "{:.0f}", "the same, whole")
v("conv_ai8", S3_INT8.traffic_intensity, "{:,.0f}", "stage-3 conv intensity, INT8")
v("conv_ai_mixed", S3_MIXED.traffic_intensity, "{:.0f}", "INT8 weights, FP16 activations")
v("conv_batch", S3.batch_size, "{:.0f}", "batch")

print("== The array: 2x2 trace and 4x4 fill and drain (Scenarios.SystolicArray*)")
v("trace2_cycles", TRACE2.cycles, "{:.0f}", "2x2 array, two rows: cycles")
v("trace2_cycles_words", TRACE2.cycles, "words", "the same, in prose")
v("trace2_rows", TRACE2.rows, "{:.0f}", "rows in the 2x2 trace")
v("sys4_dim", SYS4.array_dim, "{:.0f}", "4x4 array width")
v("sys4_rows", SYS4.rows, "{:.0f}", "input rows")
v("sys4_rows_words", SYS4.rows, "words", "the same, in prose")
v("sys4_cells", SYS4.cells, "{:.0f}", "cells")
fill_t = next(t for t in range(SYS4.cycles) if SYS4.busy_cells(t) > SYS4.array_dim)  # first snapshot past one row
drain_t = SYS4.cycles - 1 - fill_t                                                    # its mirror in the drain
v("sys4_fill_t", fill_t, "{:.0f}", "filling snapshot cycle")
v("sys4_fill_busy", SYS4.busy_cells(fill_t), "{:.0f}", "busy cells then")
v("sys4_drain_t", drain_t, "{:.0f}", "draining snapshot cycle")
v("sys4_drain_busy", SYS4.busy_cells(drain_t), "{:.0f}", "busy cells then")
v("sys4_full_first", SYS4.first_full_cycle, "{:.0f}", "first full cycle")
v("sys4_full_last", SYS4.first_full_cycle + SYS4.full_cycles - 1, "{:.0f}", "last full cycle")
v("sys4_last_cycle", SYS4.cycles - 1, "{:.0f}", "last cycle index")
v("c4", SYS4.cycles, "{:.0f}", "cycles for 8 rows")
v("busy4", pct(SYS4.busy_fraction), "{:.0f}", "average busy share, %")

print("== A real array: TPUv4i (Literature.DeviceMeasurements.TpuV4i*; Hardware.Cloud.TPUv4 MXU)")
v("mxu_dim", TPU4.compute.systolic_array_dim, "{:.0f}", "MXU width (TPU v4 and TPUv4i)")
v("tpuv4i_mxus", lit(DM.TpuV4iMxusPerCore), "{:.0f}", "MXUs per core")
v("tpuv4i_mxus_words", lit(DM.TpuV4iMxusPerCore), "words", "the same, in prose")
v("tpuv4i_macs", lit(DM.TpuV4iMxusPerCore) * TPU4.compute.systolic_macs_per_cycle, "{:,.0f}",
  "MACs per cycle per core")
v("tpuv4i_mxu_pct", lit(DM.TpuV4iDieMxuPct), "{:.0f}", "die in MXUs, %")
v("tpuv4i_cmem_pct", lit(DM.TpuV4iDieCmemPct), "{:.0f}", "die in CMEM, %")
v("tpuv4i_util_pct", lit(DM.TpuV4iMeanFlopsUtilizationPct), "{:.0f}", "average share of peak, %")

print("== Busy fraction of a 128x128 array (Scenarios.TpuV4Mxu*Rows)")
for s, key, fmt in zip(MXU_ROWS, ("b1", "b128", "b1024", "b4096"), ("{:.1f}", "{:.0f}", "{:.0f}", "{:.0f}")):
    v(f"rows_{key}", s.rows, "{:,.0f}", "input rows")
    v(f"busy_{key}", pct(s.busy_fraction), fmt, "busy, %")

print("== Loop orders and their traffic (Scenarios.Dense256To512Batch32WeightTiles)")
v("w_rows", LAYER.m, "{:.0f}", "weight rows")
v("w_cols", LAYER.k, "{:.0f}", "weight columns (reduction)")
v("x_cols", LAYER.n, "{:.0f}", "batch columns")
v("layer_macs", LAYER.macs, "{:,.0f}", "MACs")
v("layer_macs_m", LAYER.macs / 10 ** 6, "{:.1f}", "MACs, millions")
v("w_el", LAYER.a_elements, "{:,.0f}", "weights")
v("x_el", LAYER.b_elements, "{:,.0f}", "inputs")
v("z_el", LAYER.c_elements, "{:,.0f}", "outputs")
v("psum_moves", LAYER.partial_sum_moves, "{:,.0f}", "partial-sum reads and writes")
v("psum_moves_m", LAYER.partial_sum_moves / 10 ** 6, "{:.1f}", "the same, millions")
v("ws_total_m", LAYER.a_stationary_moves / 10 ** 6, "{:.1f}", "weight stationary, millions")
v("os_total_m", LAYER.output_stationary_moves / 10 ** 6, "{:.1f}", "output stationary, millions")
v("is_total_m", LAYER.b_stationary_moves / 10 ** 6, "{:.1f}", "input stationary, millions")
v("row_cells", TPU4.compute.systolic_array_dim, "{:.0f}", "cells in one array row")
v("row_x_reads", LAYER.row_b_reads(TPU4.compute.systolic_array_dim), "{:,.0f}", "input reads, row of cells")

print("== Example 11.4 (Scenarios.Conv3x3Filters16On8x8)")
v("ex_filters", EX114.out_channels, "{:.0f}", "filters")
v("ex_kernel", EX114.kernel, "{:.0f}", "filter edge")
v("ex_map", EX114.height, "{:.0f}", "map edge")
v("ex_positions", EX114.positions, "{:.0f}", "output positions")
v("ex_held", EX114.weights, "{:.0f}", "weight reads, held")
v("ex_spilled", EX114.macs, "{:,.0f}", "weight reads, spilled")

print("== Eyeriss (Literature.DeviceMeasurements.Eyeriss*)")
v("eyeriss_rows", lit(DM.EyerissPeRows), "{:.0f}", "PE rows")
v("eyeriss_cols", lit(DM.EyerissPeColumns), "{:.0f}", "PE columns")
v("eyeriss_pes", lit(DM.EyerissPeRows) * lit(DM.EyerissPeColumns), "{:.0f}", "PEs")
v("eyeriss_conv_lo", lit(DM.EyerissConvEnergyGainLow), "{:.1f}", "conv energy gain, low")
v("eyeriss_conv_hi", lit(DM.EyerissConvEnergyGainHigh), "{:.1f}", "conv energy gain, high")
v("eyeriss_fc", lit(DM.EyerissFcEnergyGainMin), "{:.1f}", "FC energy gain, at least")
v("eyeriss_batch", lit(DM.EyerissFcMinBatch), "{:.0f}", "FC batch")

print("== Four weight tiles on one H100 SM (Scenarios.Dense256To512Batch32WeightTiles)")
KIB = "KiB"
v("smem_kb", m(H100.memory.shared_memory_per_sm, KIB), "{:.0f}", "shared memory per SM, labeled KB")
v("w_all_kb", m(LAYER.a_bytes, KIB), "{:.0f}", "all weights at FP16, KB")
v("w_tile_kb", m(LAYER.a_tile_bytes, KIB), "{:.0f}", "one weight tile, KB")
v("x_kb", m(LAYER.b_tile_bytes, KIB), "{:.0f}", "the input, KB")
v("z_tile_kb", m(LAYER.accumulator_tile_bytes, KIB), "{:.0f}", "one FP32 output tile, KB")
v("live_kb", m(LAYER.live_bytes, KIB), "{:.0f}", "live total, KB")
v("w_tile_rows", LAYER.tile_m, "{:.0f}", "rows per weight tile")
v("w_tiles", LAYER.tiles_m, "{:.0f}", "weight tiles")
v("w_tiles_words", LAYER.tiles_m, "words", "the same, in prose")
v("x_moves", LAYER.b_moves, "{:,.0f}", "input reads, tiled")
v("tiled", LAYER.tiled_moves, "{:,.0f}", "element moves, tiled")
v("untiled", LAYER.output_stationary_moves, "{:,.0f}", "element moves, untiled output stationary")
v("tile_gain", LAYER.output_stationary_moves / LAYER.tiled_moves, "{:.0f}", "untiled over tiled")

print("== Boehm kernels 1 and 3 (Scenarios.BoehmSgemm*Kernel)")
v("boehm_k3", BOEHM[2][1], "{:,.0f}", "kernel 3, GFLOP/s")
v("boehm_k3_1", BOEHM[2][1], "{:,.1f}", "kernel 3, one decimal")
v("k1_loads", K1.global_loads_per_mac, "{:.0f}", "kernel 1 global loads per multiply-add")
v("k3_block", K3.tile_m, "{:.0f}", "kernel 3 tile edge (BLOCKSIZE)")
v("k3_fewer", K1.global_loads_per_mac / K3.global_loads_per_mac, "{:.0f}", "kernel 3 fewer loads")
v("k3_gain", BOEHM[2][1] / NAIVE, "{:.1f}", "kernel 3 over kernel 1")
v("k5_over_k3", BOEHM[4][1] / BOEHM[2][1], "{:.1f}", "kernel 5 over kernel 3")

print("== Tile intensity (Scenarios.Matmul4096Fp16Tile*)")
v("mm_dim", MM.m, "{:,.0f}", "matrix edge")
v("tile_dim", MM.tile_m, "{:.0f}", "tile edge")
v("mm_out_tiles", MM.output_tiles, "{:,.0f}", "output tiles")
v("mm_tiles_k", MM.tiles_k, "{:.0f}", "tile products per output tile")
v("mm_tile_products", MM.tile_products, "{:,.0f}", "tile products")
v("i_none", MM.unblocked_intensity, "{:.1f}", "no reuse, FLOP/byte")
v("op_bytes", m(MM.operand_bytes, "byte"), "{:.0f}", "bytes per FP16 operand")
v("tile_bytes", m(MM.tile_product_bytes, "byte"), "{:,.0f}", "bytes per tile product")
v("tile_mflop", m(MM.tile_product_flops, "flop") / 10 ** 6, "{:.2f}", "FLOP per tile product, millions")
v("i128", MM.tile_intensity, "{:.0f}", "128x128 intensity")
v("i256", MM_256.tile_intensity, "{:.0f}", "256x128 intensity")
v("ionce", MM.compulsory_intensity, "{:,.0f}", "each matrix once")
v("big_tile_m", MM_256.tile_m, "{:.0f}", "cuBLAS largest tile, rows")
v("big_tile_n", MM_256.tile_n, "{:.0f}", "cuBLAS largest tile, columns")
v("small_tile", lit(CR.NvidiaGemmSmallestTileEdge), "{:.0f}", "cuBLAS smallest tile edge")

print("== Equation 6 capacity (Scenarios.Matmul4096Fp16Tile128*)")
v("cap_a", m(MM.a_tile_bytes, KIB), "{:.0f}", "A tile, single buffered, KB")
v("cap_acc", m(MM.accumulator_tile_bytes, KIB), "{:.0f}", "accumulator, KB")
v("cap_single", m(MM.live_bytes, KIB), "{:.0f}", "single buffered, KB")
v("cap_single_b", m(MM.live_bytes, "byte"), "{:,.0f}", "single buffered, bytes")
v("cap_double", m(MM_DBL.live_bytes, KIB), "{:.0f}", "double buffered, KB")
v("cap_k256", m(MM_K256.live_bytes, KIB), "{:.0f}", "T_K = 256 double buffered, KB")
v("tk_big", MM_K256.tile_k, "{:.0f}", "the longer reduction tile")

print("== The width-129 cliff (Scenarios.Width129On128Array)")
v("w129", FRINGE.pruned_channels, "{:.0f}", "layer width")
v("w129_pad", FRINGE.padded_channels, "{:.0f}", "lanes occupied")
v("w129_tiles", FRINGE.tiles, "{:.0f}", "tiles along the width")
v("w129_pct", pct(FRINGE.utilization), "{:.0f}", "array use along one dimension, %")
v("w129_pct2", pct(FRINGE.utilization ** 2), "{:.0f}", "both dimensions, %")
v("w129_last", FRINGE.last_tile_lanes, "words", "busy lanes in the second tile, in prose")
v("mm_tiles_dim", MM.tiles_m, "{:.0f}", "tiles per dimension")

print("== Measured tile quantization (slides/vol1/11b_hw_acceleration/images/data/nvidia_gemm_tile_quantization_a100.csv)")
tile_n = MM_256.tile_n                               # the forced tile is 128 wide along N
rows = TILE_SERIES.entries
at = next(i for i, r in enumerate(rows) if r["n"] == tile_n)
n0, t0 = rows[at]["n"], rows[at]["tflops"]
n1, t1 = rows[at + 1]["n"], rows[at + 1]["tflops"]
work = n1 / n0
time_growth = work * t0 / t1
v("nv_step", n1 - n0, "{:.0f}", "sweep step in N")
v("nv_n0", n0, "{:.0f}", "N at one full tile")
v("nv_n1", n1, "{:.0f}", "N one step past")
v("nv_t0", t0, "{:.0f}", "TFLOP/s at the tile")
v("nv_t0_1", t0, "{:.1f}", "the same, one decimal")
v("nv_t1", t1, "{:.0f}", "TFLOP/s one step past")
v("nv_t1_1", t1, "{:.1f}", "the same, one decimal")
v("nv_useful", pct((n1 - n0) / tile_n), "{:.2f}", "useful share of the extra tile, %")
v("nv_work", work, "{:.4f}", "work growth")
v("nv_work_pct", pct(work - 1), "{:.0f}", "work growth, %")
v("time36", time_growth, "{:.2f}", "time growth")
v("time36_pct", pct(time_growth - 1), "{:.0f}", "time growth, %")
v("nv_m", lit(CR.NvidiaGemmTileQuantizationM), "{:,.0f}", "M of the sweep")
v("nv_k", lit(CR.NvidiaGemmQuantizationK), "{:,.0f}", "K of the sweep")
edges = [r["n"] for r in rows if r["n"] % tile_n == 0 and n0 < r["n"] < rows[-1]["n"]]
v("nv_edge2", edges[0], "{:.0f}", "second tile edge in the sweep")
v("nv_edge3", edges[1], "{:.0f}", "third tile edge in the sweep")

print("== Wave quantization (Scenarios.A100GemmOneWave, A100GemmTailWave)")
v("wave_m", WAVE1.m, "{:,.0f}", "M")
v("wave_n1", WAVE1.n, "{:,.0f}", "N, one wave")
v("wave_n2", WAVE2.n, "{:,.0f}", "N, wave plus tail")
v("wave_rows", WAVE1.tile_rows, "{:.0f}", "tiles per column")
v("wave_cols1", WAVE1.tile_cols, "{:.0f}", "tiles per row, one wave")
v("wave_cols2", WAVE2.tile_cols, "{:.0f}", "tiles per row, with tail")
v("wave_tiles1", WAVE1.tiles, "{:.0f}", "tiles, one wave")
v("wave_tiles2", WAVE2.tiles, "{:.0f}", "tiles, with tail")
v("a100_sms", A100.compute.sm_count, "{:.0f}", "A100 SMs")
v("wave_tail", WAVE2.tail_tiles, "{:.0f}", "tail tiles")
v("wave_tail_words", WAVE2.tail_tiles, "words", "the same, in prose")
v("wave_idle", WAVE2.sms - WAVE2.tail_tiles, "{:.0f}", "idle SMs in the tail")
v("wave_tail_pct", pct(WAVE2.tail_share), "{:.2f}", "tail share of SMs, %")
v("wave_extra_pct", pct(WAVE2.n / WAVE1.n - 1), "{:.1f}", "extra work, %")
v("dim_mult", lit(CR.NvidiaGemmFp16DimMultiple), "{:.0f}", "FP16 dimension multiple")
v("dim_mult_a100", lit(CR.NvidiaGemmA100Fp16DimMultiple), "{:.0f}", "A100 FP16 dimension multiple")

print("== Mapping space (Scenarios.MappingChoicesOneLevel; Literature.CompilerResults.Ansor*)")
v("sp_tiers", SPACE.tiers, "{:.0f}", "placement choices per tensor")
v("sp_tensors", SPACE.tensors, "{:.0f}", "tensors placed")
v("sp_tensors_words", SPACE.tensors, "words", "the same, in prose")
v("sp_tiers_words", SPACE.tiers, "words", "placement choices, in prose")
v("sp_place", SPACE.placements, "{:,.0f}", "placements")
v("sp_depth", SPACE.loop_depth, "{:.0f}", "loop depth")
v("sp_orders", SPACE.loop_orders, "{:,.0f}", "loop orders")
v("sp_cand", SPACE.parallel_candidates, "{:.0f}", "loops that could run in parallel")
v("sp_pick", SPACE.parallel_loops, "{:.0f}", "loops chosen")
v("sp_cand_rest", SPACE.parallel_candidates - SPACE.parallel_loops, "{:.0f}", "the factorial below")
v("sp_par", SPACE.parallelizations, "{:.0f}", "parallelizations")
v("space", SPACE.per_level, "{:,.0f}", "mappings per level")
v("space_m", round(SPACE.per_level / 10 ** 6, -1), "{:.0f}", "the same, millions, to the ten")
v("ansor_intel", lit(CR.AnsorIntelCpuSpeedup), "{:.1f}", "Ansor, Intel CPU")
v("ansor_arm", lit(CR.AnsorArmCpuSpeedup), "{:.1f}", "Ansor, ARM CPU")
v("ansor_gpu", lit(CR.AnsorNvidiaGpuSpeedup), "{:.1f}", "Ansor, NVIDIA GPU")

print("== Compiler pipeline on ResNet-50 (engine.serving_profile; Scenarios.ResNet50Batch1H100Fp16PeakRate)")
v("eager_kernels", PROFILE.eager_kernels, "{:.0f}", "framework kernels")
v("fused_kernels", PROFILE.fused_kernels, "{:.0f}", "fused kernels")
v("compiled_ms", m(PROFILE.tensorrt("fp16", batch=1), "ms"), "{:.2f}", "compiled FP16 batch one, ms")
v("resnet_gflop", m(RESNET.inference_flops, "GFLOP"), "{:.1f}", "ResNet-50 GFLOP per image")
v("floor16_us", m(FLOOR16.result.latency_compute, "us"), "{:.0f}", "arithmetic at the FP16 peak, us")
v("floor16_us_1", m(FLOOR16.result.latency_compute, "us"), "{:.1f}", "the same, one decimal")

print("== Inductor fused BN and ReLU (Scenarios.ElementwisePairFusion on the stage-2 map)")
v("map_c", S2.out_channels, "{:.0f}", "channels")
v("map_h", S2.height, "{:.0f}", "map edge")
v("passes_unfused_words", FUSION.unfused_trips, "words", "map passes unfused, in prose")
v("passes_fused_words", FUSION.fused_trips, "words", "map passes fused, in prose")
saved = (FUSION.unfused_trips - FUSION.fused_trips) * S2_FP32.activation_bytes
v("fusion_saved_mb", m(saved, "MB"), "{:.2f}", "FP32 bytes saved per image, MB")

print("== Registers and resident threads (Scenarios.H100Registers*; Literature TVM)")
v("tvm_lo", lit(CR.TvmFusedOperatorSpeedupLow), "{:.1f}", "TVM fusion, low")
v("tvm_hi", lit(CR.TvmFusedOperatorSpeedupHigh), "{:.0f}", "TVM fusion, high")
v("regs_sm", REGS[0].registers_per_sm, "{:,.0f}", "registers per SM")
v("max_threads", H100.compute.max_threads_per_sm, "{:,.0f}", "resident-thread limit")
v("warp", H100.compute.warp_size, "{:.0f}", "threads per warp")
for s, key in zip(REGS, ("r32", "r64", "r128", "rmax")):
    v(f"{key}_regs", s.registers_per_thread, "{:.0f}", "registers per thread")
    v(f"{key}_threads", s.resident_threads, "{:,.0f}", "resident threads")
    v(f"{key}_pct", pct(s.occupancy), "{:g}", "occupancy, %")
v("rmax_raw", REGS[-1].threads_by_registers, "{:.0f}", "65,536 / 255 before rounding to warps")

print("== Diagnose (Scenarios.Width1000OnA100Fp16Multiple)")
v("pad_width", PAD.pruned_channels, "{:,.0f}", "illustrative width")
v("pad_blocks", PAD.full_tiles, "{:.0f}", "whole blocks of 64")
v("pad_rest", PAD.remainder, "{:.0f}", "left over")
v("pad_to", PAD.padded_channels, "{:,.0f}", "padded width")
v("pad_tiles", PAD.tiles, "{:.0f}", "blocks after padding")

print("== TF32 (Scenarios.Pytorch*MatmulGa100; Hardware.Cloud.A100)")
v("tf32_dim", TF32.dim, "{:,.0f}", "matrix edge")
v("tf32_s", m(TF32.time, "s"), "{:.3f}", "TF32 time, s")
v("fp32_s", m(FP32.time, "s"), "{:.2f}", "FP32 time, s")
v("tf32_ratio", (FP32.time / TF32.time).to("").magnitude, "{:.1f}", "time ratio")
v("tf32_rate", m(TF32.rate, "TFLOP/s"), "{:.0f}", "TF32 rate, TFLOP/s")
v("fp32_rate", m(FP32.rate, "TFLOP/s"), "{:.1f}", "FP32 rate, TFLOP/s")
v("a100_fp32", m(A100.compute.precision_flops["fp32"], "TFLOP/s"), "{:.1f}", "A100 FP32 peak")
v("a100_tf32", m(A100.compute.precision_flops["tf32"], "TFLOP/s"), "{:.0f}", "A100 TF32 dense peak")

print("== One activation map's trip (stage-2 scenarios; Hardware.Cloud.H100; Hardware.Tech.Memory)")
v("map_int8_kb", m(S2_INT8.activation_bytes, "KB"), "{:.0f}", "INT8 map, KB")
v("map_fp16_kb", m(S2.activation_bytes, "KB"), "{:.0f}", "FP16 map, KB")
v("read_uj", m(READ1.energy, "uJ"), "{:.1f}", "read once at INT8, uJ")
v("dram_pj_b", m(TECH.Memory.DRAM.energy_per_byte, "pJ/B"), "{:.0f}", "DRAM energy per byte, pJ")
v("dram_pj_word", m(TECH.Memory.DRAM.energy_per_access, "pJ"), "{:.0f}", "DRAM 32-bit read, pJ")
v("l2_mb", m(H100.memory.l2_cache, "MiB"), "{:.0f}", "H100 L2, labeled MB")
v("batch_mb", m(S2_INT8.batch_output_bytes, "MB"), "{:.1f}", "batch of 32 at INT8, MB")
v("e_reg", m(TECH.Memory.Register.energy_per_access, "pJ"), "{:.1f}", "register access, pJ")
v("b_int8", m(B_INT8.energy, "uJ"), "{:.0f}", "INT8 boundary, uJ")
v("b_fp16", m(B_FP16.energy, "uJ"), "{:.0f}", "FP16 boundary, uJ")
v("hbm_pj_lo", m(B_HBM_LO.energy_per_byte, "pJ/B"), "{:.0f}", "HBM2 7 nm, pJ/byte, low")
v("hbm_pj_hi", m(B_HBM_HI.energy_per_byte, "pJ/B"), "{:.0f}", "HBM2 7 nm, pJ/byte, high")
v("ddr_pj_b", m(B_DDR.energy_per_byte, "pJ/B"), "{:.0f}", "DDR3/4 7 nm, pJ/byte")
v("b_hbm_lo", m(B_HBM_LO.energy, "uJ"), "{:.1f}", "INT8 boundary at HBM2, low, uJ")
v("b_hbm_hi", m(B_HBM_HI.energy, "uJ"), "{:.1f}", "INT8 boundary at HBM2, high, uJ")
v("hbm_fall_lo", (B_INT8.energy / B_HBM_HI.energy).to("").magnitude, "{:.1f}", "45 nm over HBM2, low")
v("hbm_fall_hi", (B_INT8.energy / B_HBM_LO.energy).to("").magnitude, "{:.1f}", "45 nm over HBM2, high")

print("== Five layers on the H100 FP16 roofline (thumbnail of the next-lecture frame; Part 1's scenarios)")
ROOF_DECODE, ROOF_LN, ROOF_HEAD = (Scenarios.Gpt2XlDecodeH100Roofline, Scenarios.Gpt2XlLayerNormH100Roofline,
                                   Scenarios.ResNet50ClassifierHeadH100Roofline)
v("dec_ai", m(ROOF_DECODE.point_intensity, "flop/byte"), "{:.0f}", "decode intensity")
v("dec_pct", pct(ROOF_DECODE.peak_fraction), "{:.1f}", "decode ceiling, % of peak")
v("ln_ai", Scenarios.Gpt2XlLayerNormPrefill.intensity, "{:.1f}", "LayerNorm intensity")
v("ln_pct", pct(ROOF_LN.peak_fraction), "{:.1f}", "LayerNorm ceiling, % of peak")
v("head_ai", Scenarios.ResNet50ClassifierHeadBatch32.traffic_intensity, "{:.1f}", "FC head intensity")
v("head_pct", pct(ROOF_HEAD.peak_fraction), "{:.1f}", "FC head ceiling, % of peak")
v("qkv_ai", Scenarios.Gpt2XlQkvProjectionPrefill.traffic_intensity, "{:,.1f}", "QKV intensity")

if __name__ == "__main__":
    write(__file__)
