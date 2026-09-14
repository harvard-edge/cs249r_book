/**
 * TinyTorch C++ SIMD Matrix Multiplication Kernel
 *
 * Implements cache-blocked, multi-threaded SIMD GEMM (General Matrix Multiply)
 * for CPU backends using standard C++17, OpenMP, and auto-vectorizable memory layouts.
 *
 * C = A * B
 * A: [M x K], B: [K x N], C: [M x N]
 */

#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

#if defined(_OPENMP)
#include <omp.h>
#endif

extern "C" {

/**
 * Tiled, multi-threaded GEMM kernel with zero allocation overhead.
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
        
        // Fast approximation of tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
        float exp_2z = std::exp(2.0f * inner);
        float tanh_val = (exp_2z - 1.0f) / (exp_2z + 1.0f);
        Y[idx] = 0.5f * val * (1.0f + tanh_val);
    }
}

} // extern "C"
