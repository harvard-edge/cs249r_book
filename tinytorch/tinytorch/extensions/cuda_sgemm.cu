/**
 * TinyTorch CUDA SGEMM Kernels: an example extension
 *
 * Single-precision GEMM (General Matrix Multiply) on NVIDIA GPUs, written as
 * three kernels that compute the same C = A * B and differ only in how threads
 * map onto memory. Each has its own extern "C" entry point, so you can time and
 * profile them side by side and see what each change buys.
 *
 * C = A * B
 * A: [M x K], B: [K x N], C: [M x N], row-major float32
 *
 * Why GEMM on a GPU is a memory problem
 * -------------------------------------
 * Every element of C is a dot product of one row of A with one column of B:
 * K multiply-adds that need 2K floats from global memory (DRAM). Read straight
 * from DRAM, that is 2 FLOPs per 8 bytes, or 0.25 FLOP/byte. A modern GPU can do
 * tens of FLOPs in the time it takes to fetch one byte, so a kernel that fetches
 * every operand from DRAM leaves almost all of the arithmetic units idle. The
 * three stages attack that in order:
 *
 *   1. naive      the obvious mapping, one thread per element of C, with a warp
 *                 that reads A in 32 scattered places
 *   2. coalesced  the same arithmetic with a warp that reads 32 adjacent floats,
 *                 so each memory transaction carries 32 useful values, not 1
 *   3. tiled      tiles of A and B staged in on-chip shared memory, so each float
 *                 fetched from DRAM is used TILE (32) times: about 8 FLOP/byte
 *
 * Measured on an RTX 4060 Laptop GPU at n = 2048 (kernel only, CUDA events):
 * about 120, 840 and 1,050 GFLOP/s. Coalescing is worth 7x. Tiling cuts DRAM
 * traffic 32x on paper, yet adds only about 25% over stage 2, because the L1/L2
 * caches were already catching much of stage 2's re-reading, and because the
 * tiled inner loop now waits on shared memory instead (two shared-memory loads
 * per multiply-add). Measure; do not assume the napkin math.
 *
 * Where to go next: register tiling (each thread computes a small block of C held
 * in registers, so each value loaded from shared memory feeds several
 * multiply-adds) is the step after stage 3, and the same idea as the register
 * micro-tiling project for the C++ SIMD kernel in cpp_simd_gemm.cpp.
 */

#include <cuda_runtime.h>
#define TILE 32

/**
 * Stage 1: naive.
 *
 * One thread computes one element C[row][col], and threadIdx.x selects the ROW.
 * The 32 threads of a warp run in lockstep, and here they sit in 32 different
 * rows with the same column. At each step k they all read A[row*K + k]: 32
 * addresses K floats apart, so the warp touches 32 different 32-byte memory
 * sectors to use 4 bytes from each. Their reads of B[k*N + col] hit one
 * address, which is broadcast, and their writes to C are again N floats apart.
 *
 * Stage 2 keeps the arithmetic and swaps the mapping.
 */
__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int K, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // The grid is rounded up to whole blocks, so edge threads can fall outside C
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

/**
 * Stage 2: coalesced.
 *
 * Same arithmetic as stage 1, but threadIdx.x now selects the COLUMN. A warp is
 * 32 threads with the same row and 32 consecutive columns, so at each step k it
 * reads B[k*N + col .. col+31]: 32 adjacent floats, 128 contiguous bytes served
 * as four full 32-byte sectors. A[row*K + k] is now the single broadcast
 * address, and the writes to C are adjacent too.
 *
 * Nothing about the math changed. The only change is which index the hardware's
 * thread numbering walks along, and on the RTX 4060 that alone is about 7x.
 */
__global__ void sgemm_coalesced(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

/**
 * Stage 3: shared-memory tiling.
 *
 * Stage 2 still fetches every operand from global memory once per use: each
 * float of A is read by N different threads and each float of B by M. Tiling
 * turns those re-reads into reuse. The block walks along K one TILE x TILE tile
 * at a time:
 *
 *   1. every thread loads one element of the A tile and one of the B tile into
 *      shared memory, a small on-chip scratchpad visible to the whole block
 *      (coalesced, since threadIdx.x runs along a row)
 *   2. the block synchronizes, so the whole tile is in place
 *   3. each thread does TILE multiply-adds out of shared memory
 *   4. the block synchronizes again before the next tile overwrites this one
 *
 * Each float fetched from DRAM is now used TILE times instead of once: 64 FLOPs
 * for 8 bytes loaded per thread per tile, about 8 FLOP/byte against stage 2's
 * 0.25. The two tiles take 2 x 32 x 32 x 4 bytes = 8 KB of shared memory per block.
 *
 * The inner loop is free of shared-memory bank conflicts: As[ty][i] is the same
 * address for the whole warp (a broadcast), and Bs[i][tx] spreads the warp over
 * 32 consecutive words, one per bank. Out-of-range elements of partial tiles are
 * loaded as 0, so they add nothing to the sum and the edges need no special case.
 */
__global__ void sgemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        // Load one tile of A and one of B; zero-fill past the edges
        As[ty][tx] = (row < M && t + tx < K) ? A[row * K + (t + tx)] : 0.0f;
        Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.0f;
        __syncthreads();  // the whole tile is loaded before anyone reads it

        for (int i = 0; i < TILE; ++i) {
            acc += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();  // everyone is done before the next tile overwrites it
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

enum Stage { STAGE_NAIVE = 0, STAGE_COALESCED = 1, STAGE_TILED = 2 };

/**
 * Host side, shared by every stage: one complete round trip to the GPU.
 *
 * A, B and C are host pointers owned by NumPy. Each call allocates device
 * memory, copies A and B to the GPU over PCIe, launches one kernel, copies C
 * back, and frees everything. A timing taken from Python therefore includes
 * both transfers, as the MPS numbers in the extensions chapter do. Those copies
 * move 3n^2 floats for 2n^3 FLOPs (n/6 FLOP/byte), so they dominate at small n
 * and fade at large n.
 *
 * Every CUDA call reports failure through its return value. The function stops
 * at the first error and returns it, so Python can warn instead of silently
 * handing back a NumPy result labeled as GPU work.
 */
static int run_sgemm(Stage stage, const float* A, const float* B, float* C, int M, int K, int N) {
    // An empty C has nothing to compute, and a zero grid dimension is an
    // invalid launch. K == 0 needs no guard: the loop is skipped and C is zeros.
    if (M == 0 || N == 0) return 0;

    size_t bytes_a = (size_t)M * K * sizeof(float);
    size_t bytes_b = (size_t)K * N * sizeof(float);
    size_t bytes_c = (size_t)M * N * sizeof(float);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaError_t err = cudaSuccess;

    // Bail out on the first failure; cudaFree(nullptr) is a no-op, so cleanup is safe
    if ((err = cudaMalloc(&dA, bytes_a)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&dB, bytes_b)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&dC, bytes_c)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(dA, A, bytes_a, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(dB, B, bytes_b, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;

    {
        // 32 x 32 = 1,024 threads per block, the CUDA maximum; the grid has
        // enough blocks to cover all of C, rounded up
        dim3 block(TILE, TILE);
        switch (stage) {
            case STAGE_NAIVE: {
                // x covers rows, y covers columns (see the kernel comment)
                dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE);
                sgemm_naive<<<grid, block>>>(dA, dB, dC, M, K, N);
                break;
            }
            case STAGE_COALESCED: {
                // x covers columns, y covers rows
                dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
                sgemm_coalesced<<<grid, block>>>(dA, dB, dC, M, K, N);
                break;
            }
            case STAGE_TILED: {
                // same mapping as stage 2: x covers columns, y covers rows
                dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
                sgemm_tiled<<<grid, block>>>(dA, dB, dC, M, K, N);
                break;
            }
        }
    }
    // A launch is asynchronous: cudaGetLastError catches a bad launch
    // configuration, and the blocking cudaMemcpy below waits for the kernel
    if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(C, dC, bytes_c, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;

cleanup:
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return (int)err;  // 0 on success; Python falls back to NumPy otherwise
}

// C-ABI entry points, loaded from Python with ctypes (see cuda_ops.py)

extern "C" {

/** Stage 1: one thread per element of C, uncoalesced reads of A. Returns a cudaError_t. */
int tinytorch_cuda_gemm_naive(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    return run_sgemm(STAGE_NAIVE, A, B, C, M, K, N);
}

/** Stage 2: the same arithmetic with coalesced reads of B. Returns a cudaError_t. */
int tinytorch_cuda_gemm_coalesced(const float* A, const float* B, float* C,
                                  int M, int K, int N) {
    return run_sgemm(STAGE_COALESCED, A, B, C, M, K, N);
}

/** Stage 3: 32 x 32 tiles staged in shared memory and reused. Returns a cudaError_t. */
int tinytorch_cuda_gemm_tiled(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    return run_sgemm(STAGE_TILED, A, B, C, M, K, N);
}

/** GPUs the CUDA runtime can see, or 0 when there is none or no usable driver. */
int tinytorch_cuda_device_count(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
}

} // extern "C"
