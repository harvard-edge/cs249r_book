/**
 * TinyTorch C++ SIMD Matrix Multiplication Kernel
 *
 * Implements cache-blocked SIMD GEMM (General Matrix Multiply) for CPUs in
 * standard C++17. The inner loop is written so the compiler vectorizes it
 * (AVX2 on x86, NEON on ARM); OpenMP spreads the tiles across cores when the
 * library is built with -fopenmp, and the pragmas are ignored otherwise.
 *
 * C = A * B
 * A: [M x K], B: [K x N], C: [M x N]
 */

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(_OPENMP)
#include <omp.h>
#endif

extern "C" {

/**
 * Tiled GEMM kernel; allocates nothing, writes into the caller's C buffer.
 * Designed to be called directly via Python ctypes.
 */
void tinytorch_cpp_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    constexpr int BLOCK_SIZE = 64;

    // Zero output buffer
    std::memset(C, 0, sizeof(float) * M * N);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int sj = 0; sj < N; sj += BLOCK_SIZE) {
        for (int si = 0; si < M; si += BLOCK_SIZE) {
            for (int sk = 0; sk < K; sk += BLOCK_SIZE) {

                int i_max = std::min(si + BLOCK_SIZE, M);
                int j_max = std::min(sj + BLOCK_SIZE, N);
                int k_max = std::min(sk + BLOCK_SIZE, K);

                for (int i = si; i < i_max; ++i) {
                    const float* a_row = &A[i * K];
                    float* c_row = &C[i * N];

                    for (int k = sk; k < k_max; ++k) {
                        float a_ik = a_row[k];
                        const float* b_row = &B[k * N];

                        #pragma omp simd
                        for (int j = sj; j < j_max; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * Fused Bias + GELU Elementwise Kernel
 * Computes: y = 0.5 * (x + bias) * (1.0 + tanh(sqrt(2/pi) * ((x + bias) + 0.044715 * (x + bias)^3)))
 */
void tinytorch_cpp_fused_bias_gelu(
    const float* __restrict__ X,
    const float* __restrict__ bias,
    float* __restrict__ Y,
    int total_elements,
    int inner_dim
) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total_elements; ++idx) {
        int col = idx % inner_dim;
        float val = X[idx] + bias[col];
        float cube = val * val * val;
        float inner = SQRT_2_OVER_PI * (val + COEFF * cube);
        
        float tanh_val = std::tanh(inner);
        Y[idx] = 0.5f * val * (1.0f + tanh_val);
    }
}

/** Threads OpenMP will use, or 1 when built without OpenMP. */
int tinytorch_cpp_num_threads(void) {
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

} // extern "C"
