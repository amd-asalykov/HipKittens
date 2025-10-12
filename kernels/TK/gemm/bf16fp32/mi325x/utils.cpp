#include "kittens.cuh"
using namespace kittens;


// Load from global memory to registers with proper batching for cache locality
template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load_global_to_registers(
    float4* reg_buffer, int buffer_size,
    const GL& src, const COORD& idx, const ST& dst_template, int offset, int split)
{
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    const int big_calls_start = (big_calls / split) * offset;
    const int big_calls_end = big_calls_start + (big_calls / split);

    const int row_stride = src.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* base_ptr = (T*)&src[unit_coord];  // global memory pointer
    const int laneid = threadIdx.x % N_THREADS;

    // buffer resource
    const int total_bytes = row_stride * ST::rows * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

    int buf_idx = 0;
    for (int i = 0; i < big_calls && buf_idx < buffer_size; ++i) {
        const int offset = i * small_calls;
        #pragma unroll
        for (int j = 0; j < small_calls; ++j) {
            const int chunk_idx = (offset + j) * N_THREADS + laneid;
            if (chunk_idx < total_chunks && buf_idx < buffer_size) {
                int row = chunk_idx / memcpy_per_row;
                int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
                int flat_offset = row * row_stride + col;
                int byte_offset = flat_offset * sizeof(T);
                __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
                reg_buffer[buf_idx] = *reinterpret_cast<float4*>(&raw);
                buf_idx++;
            }
        }
    }
}


// Store from registers to shared memory (preserving the batched pattern)
template<ducks::st::all ST, int N_THREADS = WARP_THREADS>
__device__ inline void store_registers_to_shared(
    const float4* reg_buffer, ST& dst)
{
    using T = typename ST::dtype;
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(T);
    constexpr int elem_per_half_memcpy = sizeof(float2)/sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    
    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;
    
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 16;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    int buf_idx = 0;
    
    // Store in the same batched pattern to maintain locality
    #pragma unroll
    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;
            if (row < dst.rows && buf_idx < 64) { // Safety check - use fixed limit
                const float4& buf_val = reg_buffer[buf_idx];
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf_val.x, buf_val.y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf_val.z, buf_val.w});
                buf_idx++;
            }
        } // Wait for this batch of stores to complete
    }
}
