/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Loads data from global memory into a shared memory vector.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination shared memory vector.
 * @param[in] src The source global memory array.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>, int N_THREADS=WARP_THREADS>
__device__ static inline void load(SV &dst, const GL &src, const COORD &idx) {
    // constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    // constexpr int total_calls = (SV::length + WARP_THREADS*elem_per_transfer - 1) / (WARP_THREADS*elem_per_transfer); // round up
    // typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    // #pragma unroll
    // for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
    //     if(i * elem_per_transfer < SV::length) {
    //         *(float4*)&dst.data[i*elem_per_transfer] = *(float4*)&src_ptr[i*elem_per_transfer];
    //     }
    // }
    using T = typename SV::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int num_memcpys = (SV::length * sizeof(T) + N_THREADS*bytes_per_thread - 1) / (N_THREADS*bytes_per_thread);
    static_assert(num_memcpys > 0, "num_memcpys must be greater than 0. Please decrease the number of threads.");

    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp = elem_per_thread * kittens::WARP_THREADS;
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid() % N_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[(idx.template unit_coord<-1, 3>())];
    i32x4 srsrc = make_srsrc(src_ptr, SV::length * sizeof(T));

    const T* lds_base = &dst.data[0] + (warpid * elem_per_warp);

    #pragma unroll
    for (int i = 0; i < num_memcpys; i++) {
        const int warp_offset = warpid + i * num_warps;
        const int lane_offset = laneid % kittens::WARP_THREADS;
        const int lane_byte_offset = (warp_offset * elem_per_warp + lane_offset * elem_per_thread) * sizeof(T);

        const T* lds_elem_ptr = lds_base + (i * num_warps * elem_per_warp);

        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc,
            lds_ptr,
            16,
            lane_byte_offset,
            0,
            0,
            static_cast<index_t>(coherency::cache_all));
    }
}

/**
 * @brief Stores data from a shared memory vector into global memory.
 *
 * @tparam ST The shared memory vector type.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory vector.
 * @param[in] idx The coord of the global memory array.
 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD=coord<SV>>
__device__ static inline void store(const GL &dst, const SV &src, const COORD &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = (SV::length + WARP_THREADS*elem_per_transfer-1) / (WARP_THREADS*elem_per_transfer); // round up
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst[(idx.template unit_coord<-1, 3>())];
    #pragma unroll
    for(int iter = 0, i = ::kittens::laneid(); iter < total_calls; iter++, i+=WARP_THREADS) {
        if(i * elem_per_transfer < SV::length) {
            *(float4*)&dst_ptr[i*elem_per_transfer] = *(float4*)&src.data[i*elem_per_transfer];
        }
    }
}

}