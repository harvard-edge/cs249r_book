/**
 * TinyTorch CUDA SGEMM Kernels
 *
 * Implements single-precision GEMM (General Matrix Multiply) on NVIDIA GPUs
 * in three teaching stages, each a separate kernel with its own extern "C"
 * entry point so learners can time and profile them side by side:
 *   1. naive      - one thread per element of C
 *   2. coalesced  - adjacent threads read adjacent addresses of B
 *   3. tiled      - tiles of A and B staged in shared memory and reused
 *
 * C = A * B
 * A: [M x K], B: [K x N], C: [M x N], row-major float32
 */

#include <cuda_runtime.h>
#define TILE 32 

// Stage 1 - Naive

//the thread computes the element ([row][col],threadIdx.x selects a row)
//so the 32 threads of the warp sit in 32 different rows , at each step k they
//read A[row*K +k] , addresses K floats apart so every thread touches its 
//own cache line. Stage 2 fixes this by swapping the mapping 

__global__ void sgemm_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N){
    int row  = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y +threadIdx.y;

    //The grid is rounded up to whole blocks so the edge threads might fall outside C
    if(row<M && col<N){
        float acc = 0.0f;
        for(int k = 0 ; k<K;++k){
            acc+=A[row*K+k] * B[k*N+col];
        }
        C[row * N + col]= acc;
    }
}
// Stage 2 - Coalesced
//
// Same arithmetic as stage 1, but threadIdx.x now selects the COLUMN. A warp
// is 32 threads with the same row and 32 consecutive columns, so at each step k
// it reads B[k*N + col .. col+31], 32 adjacent floats in one memory transaction,
// while A[row*K + k] is one address broadcast to the whole warp. The writes to C
// are adjacent too.

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
// Stage 3 - Shared-memory tiling
//
// The block walks along K one TILE x TILE tile at a time. Every thread loads
// one element of the A tile and one of the B tile into shared memory (coalesced,
// since threadIdx.x runs along a row), the block synchronizes, and each thread
// then does TILE multiply-adds out of shared memory. Each float fetched from
// global memory is reused TILE times instead of once. Out-of-range elements of
// partial tiles are loaded as 0 so they add nothing to the sum.

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



// Host side, shared by every stage
// A, B and C are host pointers owned by NumPy. Each call copies A and B to
// the GPU, runs one kernel, and copies C back, so a timing from Python
// includes both PCIe transfers, as the MPS numbers in the chapter do.


enum Stage { STAGE_NAIVE = 0, STAGE_COALESCED = 1, STAGE_TILED = 2 };

static int run_sgemm(Stage stage , const float* A, const float * B , float* C, int M , int K , int N){
    // An empty C has nothing to compute, and a zero grid dimension is an
    // invalid launch. K == 0 needs no guard: the loop is skipped and C is zeros.
    if (M == 0 || N == 0) return 0;
    
    size_t bytes_a = (size_t)M * K * sizeof(float);
    size_t bytes_b = (size_t)K * N * sizeof(float);
    size_t bytes_c = (size_t)M*N * sizeof(float);

    float *dA = nullptr , *dB = nullptr , *dC = nullptr;
    cudaError_t err = cudaSuccess;

    // Bail out on the first failure; cudaFree(nullptr) is a no-op, so cleanup is safe
    if ((err = cudaMalloc(&dA, bytes_a)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&dB, bytes_b)) != cudaSuccess) goto cleanup;
    if ((err = cudaMalloc(&dC, bytes_c)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(dA, A, bytes_a, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;
    if ((err = cudaMemcpy(dB, B, bytes_b, cudaMemcpyHostToDevice)) != cudaSuccess) goto cleanup;

    {
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
    if ((err = cudaGetLastError()) != cudaSuccess) goto cleanup;  // bad launch config
    if ((err = cudaMemcpy(C, dC, bytes_c, cudaMemcpyDeviceToHost)) != cudaSuccess) goto cleanup;

cleanup:
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return (int)err;  // 0 on success; Python falls back to NumPy otherwise
}

// C-ABI entry points, loaded from Python with ctypes

extern "C" {

int tinytorch_cuda_gemm_naive(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    return run_sgemm(STAGE_NAIVE, A, B, C, M, K, N);
}

int tinytorch_cuda_gemm_coalesced(const float* A, const float* B, float* C,
                                  int M, int K, int N) {
    return run_sgemm(STAGE_COALESCED, A, B, C, M, K, N);
}
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





