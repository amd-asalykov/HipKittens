/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {


#ifdef KITTENS_CDNA4
__device__ static inline void mfma323216(      float2 (&D)[8],
                                         const bf16_2 (&A)[8],
                                         const bf16_2 (&B)[8],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    
    *(floatx16_t*)C = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)A,
        *(bf16x8_t*)B,
        *(floatx16_t*)C,
        0, 0, 0
    );

    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_bf16(
        *(bf16x8_t*)(A + 4),
        *(bf16x8_t*)(B + 4),
        *(floatx16_t*)C,
        0, 0, 0
    );
}

__device__ static inline void mfma323216(      float2 (&D)[8],
                                         const half_2 (&A)[8],
                                         const half_2 (&B)[8],
                                         const float2 (&C)[8]) {
    // Cast to the correct vector types that the intrinsic expects
    typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 fp16x8_t;
    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;
    
    *(floatx16_t*)C = __builtin_amdgcn_mfma_f32_32x32x16_f16(
        *(fp16x8_t*)A,
        *(fp16x8_t*)B,
        *(floatx16_t*)C,
        0, 0, 0
    );

    *(floatx16_t*)D = __builtin_amdgcn_mfma_f32_32x32x16_f16(
        *(fp16x8_t*)(A + 4),
        *(fp16x8_t*)(B + 4),
        *(floatx16_t*)C,
        0, 0, 0
    );
}

__device__ static inline void mfma323264_fp6(      float2 (&D)[8],
                                             const fp6_e2m3_4 (&A)[8], 
                                             const fp6_e2m3_4 (&B)[8],
                                             const float2 (&C)[8]) {

    // Each lane provides 32 FP6 values per operand. AMD ISA requires dense 6-bit packing
    // across six contiguous 32-bit registers, but the builtin expects 8x int32 vector (32 bytes)
    auto repack_8x_fp6x4_to_int8 = [] __device__ (const fp6_e2m3_4 (&src)[8]) {
        uint32_t dense[8] = {0}; // 8 words to match builtin expectation, pad with zeros
        
        // Extract 32 FP6 values from 8 packs. Treat each pack as 4 bytes
        // and take the low 6 bits of each byte.
        uint32_t vals6[32];
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            uint32_t raw = std::bit_cast<uint32_t>(src[p]);
            vals6[p * 4 + 0] = (raw >> 0)  & 0x3Fu;   // byte0 low 6
            vals6[p * 4 + 1] = (raw >> 8)  & 0x3Fu;   // byte1 low 6
            vals6[p * 4 + 2] = (raw >> 16) & 0x3Fu;   // byte2 low 6
            vals6[p * 4 + 3] = (raw >> 24) & 0x3Fu;   // byte3 low 6
        }

        // Pack 32 x 6-bit values contiguously into first 6 u32 words (192 bits)
        // The hardware only uses the first 192 bits, last 2 words are padding
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const uint32_t v = vals6[i] & 0x3Fu;
            const int bit_pos = i * 6;               // 0..186
            const int word_idx = bit_pos >> 5;       // /32
            const int bit_off  = bit_pos & 31;       // %32

            if (word_idx < 6) { // Only pack into first 6 words
                dense[word_idx] |= (v << bit_off);
                const int spill = bit_off + 6 - 32;
                if (spill > 0 && word_idx + 1 < 6) {
                    dense[word_idx + 1] |= (v >> (6 - spill));
                }
            }
        }

        // Return as the expected 8x int32 vector type
        typedef __attribute__((__vector_size__(8 * sizeof(int)))) int int8_t;
        
        // Use a union to safely convert
        union {
            uint32_t arr[8];
            int8_t vec;
        } converter;
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            converter.arr[i] = dense[i];
        }
        
        return converter.vec;
    };

    typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16_t;

    auto a_packed = repack_8x_fp6x4_to_int8(A);
    auto b_packed = repack_8x_fp6x4_to_int8(B);

    *(floatx16_t*)D = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_packed,
        b_packed,
        *(floatx16_t*)C,
        2, // cbsz for FP6 E2M3 (log2(4) chunks)
        2, // blgp for FP6 E2M3 permutation
        0, // OpselA
        0, // scale_a (0 means no scaling)
        0, // OpselB
        0  // scale_b (0 means no scaling)
    );
}
#else
__device__ static inline void mfma161616(      float2 (&D)[2],
                                         const half_2 (&A)[2],
                                         const half_2 (&B)[2],
                                         const float2 (&C)[2]) {
    (*(float4*)D).data = {__builtin_amdgcn_mfma_f32_16x16x16f16(
        (*(short4*)A).data,
        (*(short4*)B).data,
        (*(float4*)C).data,
        0, 0, 0
    )};
}

__device__ static inline void mfma161616(      float2 (&D)[2],
                                         const bf16_2 (&A)[2],
                                         const bf16_2 (&B)[2],
                                         const float2 (&C)[2]) {
    (*(float4*)D).data = {__builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
        (*(short4*)A).data,
        (*(short4*)B).data,
        (*(float4*)C).data,
        0, 0, 0
    )};
}
#endif


/**
 * @brief Base matrix multiply-accumulate operation for row layout.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
#else
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<half, ducks::rt_layout::row> &a,
                                    const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AB_base(rt_base<float, ducks::rt_layout::col> &d,
                                    const rt_base<bf16, ducks::rt_layout::row> &a,
                                    const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                    const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Base dot product operation for row layout.
 *
 * This function performs the base dot product operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, row_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                    const rt_base<fp6_e2m3, ducks::rt_layout::row> &a,
                                    const rt_base<fp6_e2m3, ducks::rt_layout::row> &b,
                                    const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323264_fp6(d.data, a.data, b.data, c.data);
}
#else
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::row> &a,
                                     const rt_base<half, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::row> &a,
                                     const rt_base<bf16, ducks::rt_layout::row> &b, // in row-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                        const rt_base<half, ducks::rt_layout::col> &a,
                                        const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                        const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                        const rt_base<bf16, ducks::rt_layout::col> &a,
                                        const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                        const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
#else
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<half, ducks::rt_layout::col> &a,
                                     const rt_base<half, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtB_base(rt_base<float, ducks::rt_layout::col> &d,
                                     const rt_base<bf16, ducks::rt_layout::col> &a,
                                     const rt_base<bf16, ducks::rt_layout::col> &b, // in col-major mode
                                     const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Base matrix multiply-accumulate operation for row layout with transposed A and B.
 *
 * This function performs the base matrix multiply-accumulate operation
 * using the `hmma16816` function for matrices in row layout.
 *
 * @param[out] d The output rt_base<float2, row_layout> accumulator.
 * @param[in] a The first input rt_base<bf16_2, col_layout> matrix.
 * @param[in] b The second input rt_base<bf16_2, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_base<float2, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                        const rt_base<half, ducks::rt_layout::col> &a,
                                        const rt_base<half, ducks::rt_layout::row> &b, // in col-major mode
                                        const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                        const rt_base<bf16, ducks::rt_layout::col> &a,
                                        const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                        const rt_base<float, ducks::rt_layout::accumulator> &c) {
    mfma323216(d.data, a.data, b.data, c.data);
}
#else
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<half, ducks::rt_layout::col> &a,
                                      const rt_base<half, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
__device__ static inline void mma_AtBt_base(rt_base<float, ducks::rt_layout::col> &d,
                                      const rt_base<bf16, ducks::rt_layout::col> &a,
                                      const rt_base<bf16, ducks::rt_layout::row> &b, // in col-major mode
                                      const rt_base<float, ducks::rt_layout::col> &c) {
    mfma161616(d.data, a.data, b.data, c.data);
}
#endif
/**
 * @brief Matrix multiply-accumulate operation.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_hf<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_hf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_hf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_hf<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
template<ducks::rt::accumulator_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::accumulator_layout C>
#else
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
#endif
__device__ static inline void mma_AB(D &d,
                               const A &a,
                               const B &b,
                               const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::cols); // Check D matches A, B
    static_assert(A::cols == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AB_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_AB_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
template<ducks::rt::accumulator_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::accumulator_layout C>
#else
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
#endif
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    #ifdef KITTENS_CDNA4
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp6_e2m3> &&
            std::is_same_v<typename B::T, fp6_e2m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
/**
 * @brief Matrix multiply-accumulate operation with transposed A.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, row_layout> matrix.
 * @param[in] b The second input rt_bf<K, M, col_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
template<ducks::rt::accumulator_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::accumulator_layout C>
#else
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::col_layout B, ducks::rt::col_layout C>
#endif
__device__ static inline void mma_AtB(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::cols); // Check D matches A, B
    static_assert(A::rows == B::rows); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtB_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[0][m],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtB_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[k][m],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Matrix multiply-accumulate operation with transposed A and B.
 *
 * This function performs the matrix multiply-accumulate operation
 * using the `hmma16816` instruction.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<K, N, col_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in column-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
template<ducks::rt::accumulator_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::accumulator_layout C>
#else
template<ducks::rt::col_layout D, ducks::rt::col_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
#endif
__device__ static inline void mma_AtBt(D &d,
                                 const A &a,
                                 const B &b,
                                 const C &c) {
    static_assert(D::rows == A::cols && D::cols == B::rows); // Check D matches A, B
    static_assert(A::rows == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_AtBt_base(
                d.tiles[n][m],
                a.tiles[0][n],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::height; k++) {
                mma_AtBt_base(
                    d.tiles[n][m],
                    a.tiles[k][n],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}
}