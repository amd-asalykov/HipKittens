#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 64; // number of heads
constexpr int ATTN_H_KV = 8; // number of heads for key and value
constexpr int ATTN_N = 8192; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size
constexpr bool causal = true;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

template<int D, typename T=bf16, typename L=row_l> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L>;
template<int D, typename T=bf16, typename L=row_l> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L>;
template<int D, typename T=bf16, typename L=col_l> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L>;
template<int D, typename T=float, typename L=accum_col_l> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L>;


/**********************************************************/

template<ducks::rt::accumulator_col_layout RT>
__device__ inline static void mask_kv_tile_original(
    RT &dst,
    const int q_abs,
    const int k_abs
) {
    const int lane = laneid();
    const int col = lane & 31;
    const int q_pos = q_abs * Q_BLOCK_SIZE + col;

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int base_row = (i * 32 + ii * 8 + ((lane >> 5) << 2));
                const int k_pos = k_abs * KV_BLOCK_SIZE + base_row;
                if (k_pos + 0 > q_pos) dst.tiles[i][j].data[ii*2].x = kittens::base_types::constants<float>::neg_infty();
                if (k_pos + 1 > q_pos) dst.tiles[i][j].data[ii*2].y = kittens::base_types::constants<float>::neg_infty();
                if (k_pos + 2 > q_pos) dst.tiles[i][j].data[ii*2 + 1].x = kittens::base_types::constants<float>::neg_infty();
                if (k_pos + 3 > q_pos) dst.tiles[i][j].data[ii*2 + 1].y = kittens::base_types::constants<float>::neg_infty();
            }
        }
    }
}

// template<ducks::rt::accumulator_col_layout RT>
// __device__ inline static void mask_kv_tile(
//     RT &dst,
//     const int q_abs,
//     const int k_abs
// ) {
//     const int lane = laneid();
//     const int col = lane & 31;
//     const int q_pos = q_abs * Q_BLOCK_SIZE + col;

//     const float neg_inf = kittens::base_types::constants<float>::neg_infty();

//     #pragma unroll
//     for (int i = 0; i < dst.height; ++i) {
//         #pragma unroll
//         for (int j = 0; j < dst.width; ++j) {
//             #pragma unroll
//             for (int ii = 0; ii < 4; ++ii) {
//                 const int base_row = (i * 32 + ii * 8 + ((lane >> 5) << 2));
//                 const int k_pos = k_abs * KV_BLOCK_SIZE + base_row;
                
//                 int k_pos_0 = k_pos;
//                 int k_pos_1 = k_pos + 1;
//                 int k_pos_2 = k_pos + 2;
//                 int k_pos_3 = k_pos + 3;
                
//                 // Element 0: if k_pos_0 > q_pos then neg_inf else original
//                 {
//                     float result;
//                     asm volatile(
//                         "v_cmp_gt_i32 vcc, %1, %2\n\t"
//                         "v_cndmask_b32 %0, %3, %4, vcc"
//                         : "=v"(result)
//                         : "v"(k_pos_0), "v"(q_pos), "v"(dst.tiles[i][j].data[ii*2].x), "v"(neg_inf)
//                         : "vcc"
//                     );
//                     dst.tiles[i][j].data[ii*2].x = result;
//                 }
                
//                 // Element 1
//                 {
//                     float result;
//                     asm volatile(
//                         "v_cmp_gt_i32 vcc, %1, %2\n\t"
//                         "v_cndmask_b32 %0, %3, %4, vcc"
//                         : "=v"(result)
//                         : "v"(k_pos_1), "v"(q_pos), "v"(dst.tiles[i][j].data[ii*2].y), "v"(neg_inf)
//                         : "vcc"
//                     );
//                     dst.tiles[i][j].data[ii*2].y = result;
//                 }
                
//                 // Element 2
//                 {
//                     float result;
//                     asm volatile(
//                         "v_cmp_gt_i32 vcc, %1, %2\n\t"
//                         "v_cndmask_b32 %0, %3, %4, vcc"
//                         : "=v"(result)
//                         : "v"(k_pos_2), "v"(q_pos), "v"(dst.tiles[i][j].data[ii*2 + 1].x), "v"(neg_inf)
//                         : "vcc"
//                     );
//                     dst.tiles[i][j].data[ii*2 + 1].x = result;
//                 }
                
//                 // Element 3
//                 {
//                     float result;
//                     asm volatile(
//                         "v_cmp_gt_i32 vcc, %1, %2\n\t"
//                         "v_cndmask_b32 %0, %3, %4, vcc"
//                         : "=v"(result)
//                         : "v"(k_pos_3), "v"(q_pos), "v"(dst.tiles[i][j].data[ii*2 + 1].y), "v"(neg_inf)
//                         : "vcc"
//                     );
//                     dst.tiles[i][j].data[ii*2 + 1].y = result;
//                 }
//             }
//         }
//     }
// }

template<ducks::rt::accumulator_col_layout RT>
__device__ inline static void mask_kv_tile(
    RT &dst,
    const int q_abs,
    const int k_abs
) {
    const int lane = laneid();
    const int col = lane & 31;
    const int q_pos = q_abs * Q_BLOCK_SIZE + col;

    const float neg_inf = kittens::base_types::constants<float>::neg_infty();

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            #pragma unroll
            for (int ii = 0; ii < 4; ++ii) {
                const int base_row = (i * 32 + ii * 8 + ((lane >> 5) << 2));
                
                int k_pos_0 = k_abs * KV_BLOCK_SIZE + base_row + 0;
                int k_pos_1 = k_abs * KV_BLOCK_SIZE + base_row + 1;
                int k_pos_2 = k_abs * KV_BLOCK_SIZE + base_row + 2;
                int k_pos_3 = k_abs * KV_BLOCK_SIZE + base_row + 3;
                
                // Process elements 0 and 1 together
                {
                    float result0, result1;
                    asm volatile(
                        "v_cmp_gt_i32_e64 s[68:69], %2, %6\n\t"
                        "v_cmp_gt_i32_e64 s[70:71], %3, %6\n\t"
                        "v_cndmask_b32_e64 %0, %4, %7, s[68:69]\n\t"
                        "v_cndmask_b32_e64 %1, %5, %7, s[70:71]"
                        : "=v"(result0), "=v"(result1)
                        : "v"(k_pos_0), "v"(k_pos_1),
                          "v"(dst.tiles[i][j].data[ii*2].x),     // %4 - input for element 0
                          "v"(dst.tiles[i][j].data[ii*2].y),     // %5 - input for element 1
                          "v"(q_pos),                             // %6
                          "v"(neg_inf)                            // %7
                        : "s68", "s69", "s70", "s71"
                    );
                    dst.tiles[i][j].data[ii*2].x = result0;
                    dst.tiles[i][j].data[ii*2].y = result1;
                }
                
                // Process elements 2 and 3 together
                {
                    float result2, result3;
                    asm volatile(
                        "v_cmp_gt_i32_e64 s[68:69], %2, %6\n\t"
                        "v_cmp_gt_i32_e64 s[70:71], %3, %6\n\t"
                        "v_cndmask_b32_e64 %0, %4, %7, s[68:69]\n\t"
                        "v_cndmask_b32_e64 %1, %5, %7, s[70:71]"
                        : "=v"(result2), "=v"(result3)
                        : "v"(k_pos_2), "v"(k_pos_3),
                          "v"(dst.tiles[i][j].data[ii*2 + 1].x), // %4 - input for element 2
                          "v"(dst.tiles[i][j].data[ii*2 + 1].y), // %5 - input for element 3
                          "v"(q_pos),                             // %6
                          "v"(neg_inf)                            // %7
                        : "s68", "s69", "s70", "s71"
                    );
                    dst.tiles[i][j].data[ii*2 + 1].x = result2;
                    dst.tiles[i][j].data[ii*2 + 1].y = result3;
                }
            }
        }
    }
}


/**********************************************************/


template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og; 
    gl<float, -1, -1, -1, -1> L_vec;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::row>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, ducks::st_layout::accumulator_col>, 2>();
     
    const int head_idx = (blockIdx.x % 8) * 8 + (blockIdx.x / 8);
    const int batch_idx = blockIdx.z;
    const int GROUP_SIZE = ATTN_H / ATTN_H_KV;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    int k_idx_buf0 = -1, k_idx_buf1 = -1;
    int k_curr_idx = -1; 

    constexpr int num_tiles = ATTN_N / KV_BLOCK_SIZE;
    const int max_tile_idx = block_tile_idx * NUM_WARPS + NUM_WARPS - 1;
    const int max_q_end_pos = (max_tile_idx + 1) * Q_BLOCK_SIZE;
    int max_num_tiles = (max_q_end_pos + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    max_num_tiles = min(max_num_tiles + 1, num_tiles);

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Initialize all of the register tiles.
    qo_tile<D, bf16> q_reg;
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;
    kv_tile<D, bf16, accum_col_l> v_reg;
    qo_tile_transposed<D, float, accum_col_l> o_reg;
    attn_tile<D, float, accum_col_l> att_block[2];
    attn_tile<D, bf16, accum_col_l> att_block_bf16;
    typename attn_tile<D, float, accum_col_l>::row_vec max_vec, norm_vec, max_vec_prev;

    using T = typename st_bf<KV_BLOCK_SIZE, ATTN_D>::dtype;
    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);
    const lds_lane_ofs lane_offs = prefill_swizzled_offsets(k_reg, k_smem[0]);

    zero(o_reg);
    zero(norm_vec);
    neg_infty(max_vec_prev); 
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K);
    // __builtin_amdgcn_s_waitcnt(0);
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    {
        qo_tile<D, float> q_reg_fl;
        load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, g.Qg, {batch_idx, tile_idx, head_idx, 0});
        mul(q_reg_fl, q_reg_fl, TEMPERATURE_SCALE);
        copy(q_reg, q_reg_fl);
    }
    swap_layout_and_transpose(q_reg_transposed, q_reg);

    // All warps collaboratively load K1 and V0
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K);
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V);
    load(k_reg, k_smem[0], lane_offs);
    k_idx_buf1 = 1; 
    k_curr_idx = 0; 
    // __builtin_amdgcn_s_waitcnt(0);
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Process first tile always
    zero(att_block[0]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    __builtin_amdgcn_s_setprio(1);
    if constexpr (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
    __builtin_amdgcn_s_setprio(0);
    col_max(max_vec, att_block[0]);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // Load K1 and prepare for next iteration
    load(k_reg, k_smem[1], lane_offs);
    k_curr_idx = 1; 
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K);
    k_idx_buf0 = 2; 
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // hot loop
    // #pragma unroll  // for some reason unroll makes it slower
    for (int j = 3; j < max_num_tiles - 1; j += 2) {
        // Cluster 0: QK1
        zero(att_block[1]);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[0], norm_vec);
        copy(att_block_bf16, att_block[0]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 1: ALL warps must participate in collective loads
        // Load K(j) if in bounds, otherwise repeat a safe index
        __builtin_amdgcn_s_setprio(1);
        if constexpr (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
        __builtin_amdgcn_s_setprio(0);
        G::load<1, false>(k_smem[1], g.Kg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_K);
        k_idx_buf1 = j;
        load(v_reg, v_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 2: A0V0
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[1], max_vec);
        sub_col(att_block[1], att_block[1], max_vec);
        exp2(att_block[1], att_block[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 3: Loads
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V);
        load(k_reg, k_smem[0], lane_offs);
        k_curr_idx = k_idx_buf0;
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 4: QK2
        zero(att_block[0]);
        swap_layout_and_transpose(k_reg_transposed, k_reg);
        mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
        sub(max_vec_prev, max_vec_prev, max_vec); 
        exp2(max_vec_prev, max_vec_prev);  
        mul(norm_vec, norm_vec, max_vec_prev);
        col_sum(norm_vec, att_block[1], norm_vec);
        copy(att_block_bf16, att_block[1]);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 5: Loads
        __builtin_amdgcn_s_setprio(1);
        if constexpr (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
        __builtin_amdgcn_s_setprio(0);
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, swizzled_offsets_K);
        k_idx_buf0 = j + 1;
        load(v_reg, v_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 6: A1V1
        __builtin_amdgcn_s_setprio(1);
        mul_col(o_reg, o_reg, max_vec_prev);
        mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
        copy(max_vec_prev, max_vec);
        col_max(max_vec, att_block[0], max_vec);
        sub_col(att_block[0], att_block[0], max_vec);
        exp2(att_block[0], att_block[0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    
        // Cluster 7: Loads
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V);
        load(k_reg, k_smem[1], lane_offs);
        k_curr_idx = k_idx_buf1;
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0:
    //      QK3
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK2
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 1:
    //      Load K5 into shared
    __builtin_amdgcn_s_setprio(1);
    if constexpr (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
    __builtin_amdgcn_s_setprio(0);
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K);
    k_idx_buf1 = max_num_tiles - 1;
    //      Load V2 into registers
    load(v_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    //      A2V2
    __builtin_amdgcn_s_setprio(1);
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Partial softmax for QK3
    // mul(att_block[1], att_block[1], TEMPERATURE_SCALE);
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 3:
    //      Load V4 into shared
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, max_num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V);
    //      Load K4 into registers
    load(k_reg, k_smem[0], lane_offs);
    k_curr_idx = k_idx_buf0;
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    //      QK4
    zero(att_block[0]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[0], k_reg_transposed, q_reg_transposed, att_block[0]);
    //      Finish softmax for QK3
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 5:
    //      Load V3 into registers
    __builtin_amdgcn_s_setprio(1);
    if constexpr (causal) mask_kv_tile(att_block[0], tile_idx, k_curr_idx);
    __builtin_amdgcn_s_setprio(0);
    load(v_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6:
    //      A3V3
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Partial softmax for QK4
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[0], max_vec);
    sub_col(att_block[0], att_block[0], max_vec);
    exp2(att_block[0], att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 7:
    //      Load V5 into shared
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, max_num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V);
    //      Load K5 into registers
    load(k_reg, k_smem[1], lane_offs);
    k_curr_idx = k_idx_buf1;
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    //      QK5
    zero(att_block[1]);
    swap_layout_and_transpose(k_reg_transposed, k_reg);
    mma_AtB(att_block[1], k_reg_transposed, q_reg_transposed, att_block[1]);
    //      Finish softmax for QK4
    sub(max_vec_prev, max_vec_prev, max_vec); 
    exp2(max_vec_prev, max_vec_prev); 
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[0], norm_vec);
    copy(att_block_bf16, att_block[0]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 9:
    //      Load V4 into registers
    load(v_reg, v_smem[0]);
    __builtin_amdgcn_s_setprio(1);
    if constexpr (causal) mask_kv_tile(att_block[1], tile_idx, k_curr_idx);
    __builtin_amdgcn_s_setprio(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10:
    //      A4V4
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    //      Full softmax for QK5
    copy(max_vec_prev, max_vec);
    col_max(max_vec, att_block[1], max_vec);
    sub_col(att_block[1], att_block[1], max_vec);
    exp2(att_block[1], att_block[1]);
    sub(max_vec_prev, max_vec_prev, max_vec);
    exp2(max_vec_prev, max_vec_prev);  
    mul(norm_vec, norm_vec, max_vec_prev);
    col_sum(norm_vec, att_block[1], norm_vec);
    copy(att_block_bf16, att_block[1]);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 11:
    //      Load V5 into registers
    load(v_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    //      A5V5
    mul_col(o_reg, o_reg, max_vec_prev);
    mma_AtB(o_reg, v_reg, att_block_bf16, o_reg);
    div_col(o_reg, o_reg, norm_vec);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    qo_tile<D, float, accum_row_l> o_reg_transposed;
    swap_layout_and_transpose(o_reg_transposed, o_reg);
    store<1>(g.Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});

    // multiply by ln(2)
    mul(max_vec, max_vec, 0.69314718056f);
    log(norm_vec, norm_vec);
    add(norm_vec, norm_vec, max_vec);
    store(g.L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});

}


template<int D>
void dispatch_fwd(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_fwd_causal_kernel, m) {
    m.doc() = "tk_fwd_causal_kernel python module";
    py::bind_function<dispatch_fwd<ATTN_D>>(m, "dispatch_fwd", 
        &attn_globals<ATTN_D>::Qg, 
        &attn_globals<ATTN_D>::Kg, 
        &attn_globals<ATTN_D>::Vg, 
        &attn_globals<ATTN_D>::Og,
        &attn_globals<ATTN_D>::L_vec
    );
}