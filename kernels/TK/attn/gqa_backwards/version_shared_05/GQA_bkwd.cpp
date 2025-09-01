#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_N = 1024; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int BLOCK_SIZE_QO = 16; // block size for QO
constexpr int BLOCK_SIZE_KV = 256; // block size for KV
constexpr int WARP_SIZE_QO = 16; // warp size for QO
constexpr int WARP_SIZE_KV = 32; // warp size for KV

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

using namespace kittens;

template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using qo_tile = rt<T, WARP_SIZE_QO, D, L, M>;
template<int D, typename T=bf16, typename L=row_l, typename M=mfma_16x16x32> using kv_tile = rt<T, WARP_SIZE_KV, D, L, M>;
template<int D, typename T=float, typename L=accum_col_l, typename M=mfma_16x16x32> using attn_tile = rt<T, WARP_SIZE_QO, WARP_SIZE_KV, L, M>;


template<int D> struct attn_prep_globals { 
    gl<bf16, -1, -1, -1, -1> Og;
    gl<bf16, -1, -1, -1, -1> dOg; 
    gl<float, -1, -1, -1, -1> delta;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / (WARP_SIZE_QO * NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_prep_ker(const attn_prep_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    qo_tile<D, bf16, row_l> dO, O;
    qo_tile<D, float, row_l> dO_float, O_float;
    typename qo_tile<D, float, row_l>::col_vec delta_vec;

    load(dO, g.dOg, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid,0});
    load(O,  g.Og,  {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid,0});
    copy(O_float, O);
    copy(dO_float, dO);
    
    // Δ_i = row_sum(dO ⊙ O) 
    mul(dO_float, dO_float, O_float);
    row_sum(delta_vec, dO_float); 
    store(g.delta, delta_vec, {batch_idx, head_idx, 0, seq_idx * NUM_WARPS + warpid});
}

template<int D>
void dispatch_prep(attn_prep_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_prep_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_prep_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

template<int D> struct attn_bwd_combined_globals { 
    gl<bf16, -1, -1, -1, -1> Q, K, V, O, dS_ij;
    gl<bf16, -1, -1, -1, -1> dOg, dQg, dKg, dVg;
    gl<float, -1, -1, -1, -1> L_vec, delta_vec;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / BLOCK_SIZE_KV); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int axis, ducks::rt::accumulator_col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_store(const GL &dst, const RT &src, const COORD &idx) { 
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using U = typename GL::dtype;

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    const uint32_t buffer_size = row_stride * RT::rows * sizeof(U); 
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    int col_offset = laneid%(src.tile_size_col);
    int row_offset = laneid/(src.tile_size_col);

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col*j + col_offset;
            int row = src.tile_size_row*i + row_offset * 4;
            const U val_0 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].x);
            const U val_1 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[0].y);
            const U val_2 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].x);
            const U val_3 = base_types::convertor<U, T>::convert(src.tiles[i][j].data[1].y);

            uint32_t byte_offset_0 = static_cast<uint32_t>(((row + 0) * row_stride + col) * sizeof(U));
            uint32_t byte_offset_1 = static_cast<uint32_t>(((row + 1) * row_stride + col) * sizeof(U));
            uint32_t byte_offset_2 = static_cast<uint32_t>(((row + 2) * row_stride + col) * sizeof(U));
            uint32_t byte_offset_3 = static_cast<uint32_t>(((row + 3) * row_stride + col) * sizeof(U));

            uint32_t val_0_bits = *reinterpret_cast<const uint32_t*>(&val_0);
            uint32_t val_1_bits = *reinterpret_cast<const uint32_t*>(&val_1);
            uint32_t val_2_bits = *reinterpret_cast<const uint32_t*>(&val_2);
            uint32_t val_3_bits = *reinterpret_cast<const uint32_t*>(&val_3);

            asm volatile(
                "buffer_atomic_add_f32 %0, %1, %2, 0 offen\n"
                :
                : "v"(val_0_bits), "v"(byte_offset_0),      // %0, %1
                  "s"(*(i32x4*)&br)                         // %8
                : "memory"
            );

            asm volatile(
                "buffer_atomic_add_f32 %0, %1, %2, 0 offen\n"
                :
                : "v"(val_1_bits), "v"(byte_offset_1),      // %2, %3
                  "s"(*(i32x4*)&br)                         // %8
                : "memory"
            );

            asm volatile(
                "buffer_atomic_add_f32 %0, %1, %2, 0 offen\n"
                :
                : "v"(val_2_bits), "v"(byte_offset_2),      // %4, %5
                  "s"(*(i32x4*)&br)                         // %8
                : "memory"
            );

            asm volatile(
                "buffer_atomic_add_f32 %0, %1, %2, 0 offen\n"
                : 
                : "v"(val_3_bits), "v"(byte_offset_3),      // %6, %7
                  "s"(*(i32x4*)&br)                         // %8
                : "memory"
            );
        }
    }
}

template<int axis, ducks::rt::accumulator_col_layout RT, ducks::gl::all GL, ducks::coord::tile COORD=coord<RT>>
__device__ inline static void atomic_pk_add_bf16_shuffled(const GL &dst, const RT &src, const COORD &idx) { 
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    using T2 = base_types::packing<typename RT::dtype>::packed_type;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;

    static_assert(std::is_same_v<U, bf16>, "atomic_pk_add_bf16 is only supported for bf16");

    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    const int row_stride = dst.template stride<axis>();
    int laneid = kittens::laneid();

    const uint32_t buffer_size = row_stride * RT::rows * sizeof(U); 
    std::uintptr_t as_int = reinterpret_cast<std::uintptr_t>(dst_ptr);
    std::uint64_t  as_u64 = static_cast<std::uint64_t>(as_int);
    buffer_resource br = make_buffer_resource(as_u64, buffer_size, 0x00020000);

    int col_offset = (laneid/src.tile_size_row) * 4;
    int row_offset = laneid%(src.tile_size_row);

    #pragma unroll
    for(int i = 0; i < src.height; i++) {
        #pragma unroll
        for(int j = 0; j < src.width; j++) {
            int col = src.tile_size_col*j + col_offset;
            int row = src.tile_size_row*i + row_offset;

            const U2 val_0 = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[0]);
            const U2 val_1 = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[1]);

            uint32_t byte_offset_0 = static_cast<uint32_t>((row * row_stride + col + 0) * sizeof(U));
            uint32_t byte_offset_1 = static_cast<uint32_t>((row * row_stride + col + 2) * sizeof(U));

            uint32_t val_0_bits = *reinterpret_cast<const uint32_t*>(&val_0);
            uint32_t val_1_bits = *reinterpret_cast<const uint32_t*>(&val_1);

            asm volatile(
                "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen\n"
                :
                : "v"(val_0_bits), "v"(byte_offset_0),      // %0, %1
                  "s"(*(i32x4*)&br)                         // %8
                : "memory"
            );

            asm volatile(
                "buffer_atomic_pk_add_bf16 %0, %1, %2, 0 offen\n"
                :
                : "v"(val_1_bits), "v"(byte_offset_1),      // %2, %3
                  "s"(*(i32x4*)&br)                         // %8
                : "memory"
            );
        }
    }
}


template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_bwd_combined_ker(const attn_bwd_combined_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    const float scale_factor = 1.0f / sqrt(D);

    // Shared tiles
    // extern __shared__ alignment_dummy __shm[];
    // shared_allocator al((int*)&__shm[0]);
    // st_bf<BLOCK_SIZE_KV, D, ducks::st_matrix::mfma_16x16x32> K_j_smem = al.allocate<st_bf<BLOCK_SIZE_KV, D, ducks::st_matrix::mfma_16x16x32>>();
    // G::load(K_j_smem, g.K, {batch_idx, head_idx, seq_idx, 0});
    // __builtin_amdgcn_s_waitcnt(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);

    // Register tiles
    kv_tile<D, bf16, row_l, mfma_16x16x32> K_j, V_j;
    qo_tile<D, float, accum_col_l> dQ_i;
    kv_tile<D, float, accum_col_l, mfma_32x32x16> dK_j, dV_j;
    qo_tile<D, bf16, row_l, mfma_16x16x32> Q_i;
    qo_tile<D, bf16, col_l, mfma_32x32x16> dO_i;
    attn_tile<D, float, accum_col_l, mfma_16x16x32>::col_vec L_i, delta_i;

    attn_tile<D, float, accum_col_l> P_ij;
    attn_tile<D, bf16, accum_col_l> P_ij_bf16;
    attn_tile<D, float, accum_col_l> dP_ij;
    attn_tile<D, bf16, accum_col_l> dP_ij_bf16;

    attn_tile<D, bf16, col_l, mfma_32x32x16> P_ij_bf16_col;
    attn_tile<D, bf16, col_l, mfma_32x32x16> dP_ij_bf16_col;

    // 6. Load K_j and V_j from HBM to registers
    const int j = seq_idx * NUM_WARPS + warpid;
    load(V_j, g.V, {batch_idx, head_idx, j, 0});
    // load(K_j, subtile_inplace<WARP_SIZE_KV, D>(K_j_smem, {warpid, 0}));
    // __builtin_amdgcn_s_waitcnt(0);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);
    
    // 7. Initialize dK_j = 0 and dV_j = 0
    zero(dK_j);
    zero(dV_j);

    // 8. for 1 <= i <= T_r (1024 / 32 = 32)
    for (int i = 0; i < ATTN_N / BLOCK_SIZE_QO; ++i) {

        // 10. S_ij = Q_i K_j^T * scale
        // 11. P_ij = exp(S_ij - L_i)
        load(K_j, g.K, {batch_idx, head_idx, j, 0});
        load(Q_i, g.Q, {batch_idx, head_idx, i, 0});
        load(L_i, g.L_vec, {batch_idx, head_idx, 0, i});
        zero(P_ij);
        mma_ABt(P_ij, Q_i, K_j, P_ij);
        mul(P_ij, P_ij, scale_factor);
        sub_row(P_ij, P_ij, L_i);
        exp(P_ij, P_ij);
        copy(P_ij_bf16, P_ij);

        // 13. dP_ij = dO_i @ V_j^T
        load(*(qo_tile<D, bf16, row_l, mfma_16x16x32>*) (&dO_i), g.dOg, {batch_idx, head_idx, i, 0}); // TODO: replace with SMEM load;
        zero(dP_ij);
        mma_ABt(dP_ij, *(qo_tile<D, bf16, row_l, mfma_16x16x32>*) (&dO_i), V_j, dP_ij);

        // 14. dS_ij = P_ij o (dP_ij - delta_i)
        load(delta_i, g.delta_vec, {batch_idx, head_idx, 0, i});
        sub_row(dP_ij, dP_ij, delta_i);
        mul(dP_ij, dP_ij, P_ij);
        mul(dP_ij, dP_ij, scale_factor);
        copy(dP_ij_bf16, dP_ij);
        store(g.dS_ij, dP_ij_bf16, {batch_idx,head_idx,i,j}); // TODO: replace with SMEM store

        // 12. dV_j += P_ij^T @ dO_i
        load(dO_i, g.dOg, {batch_idx, head_idx, i, 0});
        P_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(P_ij_bf16);
        mma_AtB(dV_j, P_ij_bf16_col, dO_i, dV_j);

        // 16. dK_j += dS_ij^T @ Q_i   (128x64)=(128x16)x(16x64)
        load(*(qo_tile<D, bf16, col_l, mfma_32x32x16>*) (&Q_i), g.Q, {batch_idx, head_idx, i, 0});
        dP_ij_bf16_col = swap_layout_inplace<col_l, mfma_32x32x16>(dP_ij_bf16);
        mma_AtB(dK_j, dP_ij_bf16_col, *(qo_tile<D, bf16, col_l, mfma_32x32x16>*) (&Q_i), dK_j);

        // 15. dQ_i += dS_ij @ K_j (32x16)=(32x256)x(256x16)
        load(*(attn_tile<D, bf16, row_l>*) (&dP_ij_bf16_col), g.dS_ij, {batch_idx,head_idx,i,j});
        load(*(kv_tile<D, bf16, col_l>*) (&K_j), g.K, {batch_idx, head_idx, j, 0});  // TODO: replace with SMEM load
        zero(dQ_i);
        mma_AB(dQ_i, *(attn_tile<D, bf16, row_l>*) (&dP_ij_bf16_col), *(kv_tile<D, bf16, col_l>*) (&K_j), dQ_i);
        atomic_pk_add_bf16_shuffled<2>(g.dQg, dQ_i, {batch_idx,head_idx,i,0});
    }

    // 18. Write dK_j and dV_j back to HBM
    store(g.dKg, dK_j, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
    store(g.dVg, dV_j, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
}

template<int D>
void dispatch_bwd_combined(attn_bwd_combined_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_bwd_combined_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_bwd_combined_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

template<int D> struct attn_dq_shuffle_globals { 
    gl<bf16, -1, -1, -1, -1> dQg;
    dim3 grid() { return dim3(ATTN_B, ATTN_H, ATTN_N / (WARP_SIZE_QO * NUM_WARPS)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY - 32000; }
};

template<int D> __launch_bounds__(NUM_THREADS, 1)
__global__ void attend_dq_shuffle_ker(const attn_dq_shuffle_globals<D> g) {
    
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.z;

    const int warpid = kittens::warpid();

    qo_tile<D, bf16, accum_row_l> dQg;

    load(dQg, g.dQg, {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid,0});
    store(g.dQg, *(qo_tile<D, bf16, accum_col_l>*) (&dQg), {batch_idx, head_idx, seq_idx * NUM_WARPS + warpid, 0});
}

template<int D>
void dispatch_dq_shuffle(attn_dq_shuffle_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_dq_shuffle_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_dq_shuffle_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";

    py::bind_function<dispatch_prep<ATTN_D>>(m, "dispatch_prep", 
        &attn_prep_globals<ATTN_D>::Og, 
        &attn_prep_globals<ATTN_D>::dOg,
        &attn_prep_globals<ATTN_D>::delta
    );

    py::bind_function<dispatch_bwd_combined<ATTN_D>>(m, "dispatch_bwd_combined", 
        &attn_bwd_combined_globals<ATTN_D>::Q, 
        &attn_bwd_combined_globals<ATTN_D>::K, 
        &attn_bwd_combined_globals<ATTN_D>::V, 
        &attn_bwd_combined_globals<ATTN_D>::O, 
        &attn_bwd_combined_globals<ATTN_D>::dS_ij,
        &attn_bwd_combined_globals<ATTN_D>::dOg, 
        &attn_bwd_combined_globals<ATTN_D>::dQg,
        &attn_bwd_combined_globals<ATTN_D>::dKg,
        &attn_bwd_combined_globals<ATTN_D>::dVg,
        &attn_bwd_combined_globals<ATTN_D>::L_vec, 
        &attn_bwd_combined_globals<ATTN_D>::delta_vec
    );

    py::bind_function<dispatch_dq_shuffle<ATTN_D>>(m, "dispatch_dq_shuffle", 
        &attn_dq_shuffle_globals<ATTN_D>::dQg
    );
}
