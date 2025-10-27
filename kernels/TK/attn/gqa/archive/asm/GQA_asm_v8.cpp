#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 16; // batch size
constexpr int ATTN_H = 16; // number of heads
constexpr int ATTN_H_KV = 16; // number of heads for key and value
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV; // queries per KV head group
constexpr int ATTN_N = 8192; // sequence length
constexpr int ATTN_D = 128; // dimension
constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

template<int D> struct attn_globals { 
    _gl_QKVO Qg, Kg, Vg, Og;
    gl<float, -1, -1, -1, -1> L_vec;
    dim3 grid() { return dim3(ATTN_H, ((ATTN_N / Q_BLOCK_SIZE + NUM_WARPS - 1) / NUM_WARPS), ATTN_B); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

template<int D> __launch_bounds__(NUM_THREADS, 2)
__global__ void __attribute__((amdgpu_num_vgpr(22))) attend_ker(const attn_globals<D> g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s> (&k_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>, 2>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s> (&v_smem)[2] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s>, 2>();
    
    const int head_idx = (blockIdx.x % GROUP_SIZE) * GROUP_SIZE + (blockIdx.x / GROUP_SIZE);
    const int batch_idx = blockIdx.z;
    const int head_idx_kv = head_idx / GROUP_SIZE;
    const int block_tile_idx = blockIdx.y;
    const int tile_idx = block_tile_idx * NUM_WARPS + warpid();
    const int stagger = warpid() / 4;

    const int num_tiles = ATTN_N / KV_BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;

    // Register tiles
    using KV_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<192, 255>>, 4>; // 64 registers - v[192:255]
    using Q_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<160, 191>>, 4>; // 32 registers - v[160:191]
    using Q_ranges_float = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<192, 255>>, 8>; // 64 registers - v[192:255]
    using O_ranges_transposed = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<96, 159>>, 16>; // 64 registers - v[96:159]
    using O_ranges = ducks::rt::transpose_2d<O_ranges_transposed, 4, 1>; // 64 registers - v[96:159]
    using O_ranges_bf16_transposed = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<96, 127>>, 8>; // 32 registers - v[96:127]
    using O_ranges_bf16 = ducks::rt::transpose_2d<O_ranges_bf16_transposed, 4, 1>; // 32 registers - v[96:127]
    
    using ATTN_0_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<32, 63>>, 16>; // 32 registers - v[32:63]
    using ATTN_0_bf16_out_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<32, 47>>, 8>; // 16 registers - v[32:47]
    using ATTN_0_bf16_in_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<32, 47>>, 4>; // 16 registers - v[32:47]

    using ATTN_1_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<64, 95>>, 16>; // 32 registers - v[64:95]
    using ATTN_1_bf16_out_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<64, 79>>, 8>; // 16 registers - v[64:79]
    using ATTN_1_bf16_in_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<64, 79>>, 4>; // 16 registers - v[64:79]

    constexpr int scale_reg = 30; // scale register
    constexpr int prev_max_reg = 28; // previous max register
    constexpr int max_reg = 26; // max register
    constexpr int norm_reg = 24; // norm register
    constexpr int other_norm_reg = 22; // other norm register

    using pinned_registers = ducks::rt::type_list<ducks::rt::range<22, 255>>;
    ducks::rt::clobber<pinned_registers>();

    // Initialize all of the register tiles.
    rt<bf16, KV_BLOCK_SIZE, D, row_l, rt_32x16_s, KV_ranges> K_reg; // 64x128
    rt<bf16, Q_BLOCK_SIZE,  D, row_l, rt_32x16_s, Q_ranges>  Q_reg; // 32x128
    rt<float, Q_BLOCK_SIZE, D, row_l, rt_32x16_s, Q_ranges_float> Q_reg_float; // 32x128
    rt<bf16, KV_BLOCK_SIZE, D, col_l, rt_16x32_4_s, KV_ranges> V_reg; // 64x128

    rt<float, D, Q_BLOCK_SIZE, col_l, rt_32x32_s, O_ranges_transposed> O_reg_float_transposed; // 128x32
    rt<bf16, D, Q_BLOCK_SIZE, col_l, rt_32x32_s, O_ranges_bf16_transposed> O_reg_bf16_transposed; // 128x32
    rt<bf16, Q_BLOCK_SIZE, D, row_l, rt_32x32_s, O_ranges_bf16> O_reg; // 128x32

    rt<float, KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_32x32_s, ATTN_0_ranges> ATTN_0_reg; // 64x32
    rt<bf16,  KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_32x32_s, ATTN_0_bf16_out_ranges> ATTN_0_bf16_out_reg; // 64x32
    rt<bf16,  KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_16x32_4_s, ATTN_0_bf16_in_ranges> ATTN_0_bf16_in_reg; // 64x32

    rt<float, KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_32x32_s, ATTN_1_ranges> ATTN_1_reg; // 64x32
    rt<bf16,  KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_32x32_s, ATTN_1_bf16_out_ranges> ATTN_1_bf16_out_reg; // 64x32
    rt<bf16,  KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_16x32_4_s, ATTN_1_bf16_in_ranges> ATTN_1_bf16_in_reg; // 64x32

    // Initialize registers
    zero(O_reg_float_transposed);
    zero<norm_reg>();
    zero<scale_reg>();
    smol<max_reg>();

    using T = typename st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    /********** Readfirstlane hoisting **********/
    // Create base buffer resources once
    const bf16* k_base = (bf16*)&g.Kg[{batch_idx, 0, head_idx_kv, 0}];
    const bf16* v_base = (bf16*)&g.Vg[{batch_idx, 0, head_idx_kv, 0}];
    const int k_row_stride = g.Kg.template stride<1>() * sizeof(bf16);
    const int v_row_stride = g.Vg.template stride<1>() * sizeof(bf16);
    i32x4 k_srsrc_base = make_srsrc(k_base, k_row_stride * ATTN_N, k_row_stride);
    i32x4 v_srsrc_base = make_srsrc(v_base, v_row_stride * ATTN_N, v_row_stride);

    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp   = elem_per_thread * kittens::WARP_THREADS;
    const int num_warps = NUM_THREADS / kittens::WARP_THREADS;
    const int wid = warpid() % num_warps;
    uint32_t k_0_lds_base = to_sgpr_u32(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[0].data[0]) + wid * elem_per_warp * sizeof(T)
        ));
    uint32_t v_0_lds_base = to_sgpr_u32(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[0].data[0]) + wid * elem_per_warp * sizeof(T)
        ));
    uint32_t k_1_lds_base = to_sgpr_u32(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&k_smem[1].data[0]) + wid * elem_per_warp * sizeof(T)
        ));
    uint32_t v_1_lds_base = to_sgpr_u32(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&v_smem[1].data[0]) + wid * elem_per_warp * sizeof(T)
        ));
    /**************Swizzling**************/
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;

    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], g.Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], g.Vg, swizzled_offsets_V);

    // Load K0 and V0 into shared memory
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_0_lds_base);

    // Load Q into registers
    load<1>(Q_reg, g.Qg, {batch_idx, 0, head_idx, 0}, {0, tile_idx, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    copy(Q_reg_float, Q_reg);
    mul(Q_reg_float, Q_reg_float, TEMPERATURE_SCALE);
    copy(Q_reg, Q_reg_float);

    // Load K0 into registers
    load(K_reg, k_smem[0]);
    // All warps then collaboratively load in the first slice of V (V0) and the second slice of K (K1) into shared memory
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_1_lds_base);
    // All warps then load in the first slice of K (K0)
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_0_lds_base);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Each warp performs QK0
    mma_ABt(ATTN_0_reg, K_reg, Q_reg);
    col_max<max_reg>(ATTN_0_reg);
    copy<prev_max_reg, max_reg>();
    exp2<scale_reg, scale_reg>();
    sub_col<max_reg>(ATTN_0_reg, ATTN_0_reg);
    exp2<0, 0>(ATTN_0_reg, ATTN_0_reg);

    if (stagger) {
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    __builtin_amdgcn_sched_barrier(0);
    // All warps then load in the second slice of K (K1)
    load(K_reg, k_smem[1]);

    // All warps then collaboratively load in the third slice of K (K2) into shared memory
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_0_lds_base);
    // All warps then collaboratively load in the second slice of V (V1) into shared memory 
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_1_lds_base);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // hot loop
    for (int j = 3; j < num_tiles - 1; j += 2) {
        // Cluster 0:
        // QK1
        // Finish softmax for QK0
        __builtin_amdgcn_s_setprio(1);
        {
            mma_ABt<0, 0, 0>(ATTN_1_reg, K_reg, Q_reg);
            exp2<1, 0, 0>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 1>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 2>(ATTN_0_reg, ATTN_0_reg);
            mma_ABt<0, 0, 1>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            exp2<1, 0, 3>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 4>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 5>(ATTN_0_reg, ATTN_0_reg);
            mma_ABt<0, 0, 2>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            exp2<1, 0, 6>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 7>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 8>(ATTN_0_reg, ATTN_0_reg);
            mma_ABt<0, 0, 3>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            exp2<1, 0, 9>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 10>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 11>(ATTN_0_reg, ATTN_0_reg);
            mma_ABt<0, 0, 4>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            exp2<1, 0, 12>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 13>(ATTN_0_reg, ATTN_0_reg);
            exp2<1, 0, 14>(ATTN_0_reg, ATTN_0_reg);
            mma_ABt<0, 0, 5>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            exp2<1, 0, 15>(ATTN_0_reg, ATTN_0_reg);
            mul<norm_reg, norm_reg, scale_reg>();
            zero<other_norm_reg>();
            col_sum<other_norm_reg, 0, 0>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 1>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 2>(ATTN_0_reg);
            mma_ABt<0, 0, 6>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            col_sum<other_norm_reg, 0, 3>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 4>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 5>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 6>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 7>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 8>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 9>(ATTN_0_reg);
            mma_ABt<0, 0, 7>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            col_sum<other_norm_reg, 0, 10>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 11>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 12>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 13>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 14>(ATTN_0_reg);
            col_sum<other_norm_reg, 0, 15>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 0>(ATTN_0_reg);
            mma_ABt<1, 0, 0>(ATTN_1_reg, K_reg, Q_reg);
            col_sum<other_norm_reg, 1, 1>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 2>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 3>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 4>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 5>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 6>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 7>(ATTN_0_reg);
            mma_ABt<1, 0, 1>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            col_sum<other_norm_reg, 1, 8>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 9>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 10>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 11>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 12>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 13>(ATTN_0_reg);
            col_sum<other_norm_reg, 1, 14>(ATTN_0_reg);
            mma_ABt<1, 0, 2>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            col_sum<other_norm_reg, 1, 15>(ATTN_0_reg);
            add<norm_reg, norm_reg, other_norm_reg>();
            mul_col<scale_reg, 3, 0>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 1>(O_reg_float_transposed, O_reg_float_transposed);
            mma_ABt<1, 0, 3>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            mul_col<scale_reg, 3, 2>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 3>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 4>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 5>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 6>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 7>(O_reg_float_transposed, O_reg_float_transposed);
            mma_ABt<1, 0, 4>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            mul_col<scale_reg, 3, 8>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 9>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 10>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 11>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 12>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 13>(O_reg_float_transposed, O_reg_float_transposed);
            mma_ABt<1, 0, 5>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            mul_col<scale_reg, 3, 14>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 15>(O_reg_float_transposed, O_reg_float_transposed);
            copy<0, 0, 0>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<0, 0, 1>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<0, 0, 2>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<0, 0, 3>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            mma_ABt<1, 0, 6>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            copy<0, 0, 4>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<0, 0, 5>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<0, 0, 6>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<0, 0, 7>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<1, 0, 0>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<1, 0, 1>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            mma_ABt<1, 0, 7>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
            copy<1, 0, 2>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<1, 0, 3>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<1, 0, 4>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<1, 0, 5>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<1, 0, 6>(ATTN_0_bf16_out_reg, ATTN_0_reg);
            copy<1, 0, 7>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // Cluster 1:
        // Load K3 into shared memory
        G::load<1, false>(k_smem[1], g.Kg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_1_lds_base);
        macros::s_nop<16>();
        // Load V0 into registers
        load(V_reg, v_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 2:
        // A0V0
        // Partial softmax for QK3
        __builtin_amdgcn_s_setprio(1);
        {
            mma_AtB<0, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 0, 0>(ATTN_1_reg);
            col_max<max_reg, 0, 1>(ATTN_1_reg);
            col_max<max_reg, 0, 2>(ATTN_1_reg);
            mma_AtB<0, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 0, 3>(ATTN_1_reg);
            col_max<max_reg, 0, 4>(ATTN_1_reg);
            col_max<max_reg, 0, 5>(ATTN_1_reg);
            col_max<max_reg, 0, 6>(ATTN_1_reg);
            mma_AtB<0, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 0, 7>(ATTN_1_reg);
            col_max<max_reg, 1, 0>(ATTN_1_reg);
            col_max<max_reg, 1, 1>(ATTN_1_reg);
            col_max<max_reg, 1, 2>(ATTN_1_reg);
            col_max<max_reg, 1, 3>(ATTN_1_reg);
            mma_AtB<0, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 1, 4>(ATTN_1_reg);
            col_max<max_reg, 1, 5>(ATTN_1_reg);
            col_max<max_reg, 1, 6>(ATTN_1_reg);
            col_max<max_reg, 1, 7>(ATTN_1_reg);
            mma_AtB<1, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            sub<scale_reg, prev_max_reg, max_reg>();
            copy<prev_max_reg, max_reg>();
            exp2<scale_reg, scale_reg>();
            sub_col<max_reg, 0, 0>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 1>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<1, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 0, 2>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 3>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 4>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 5>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 6>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 7>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<1, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 0, 8>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 9>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 10>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 11>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 12>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 13>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<1, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 0, 14>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 0, 15>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 0>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 1>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 2>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 3>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<2, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 1, 4>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 5>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 6>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 7>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 8>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<2, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 1, 9>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 10>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 11>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 12>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 13>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<2, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 1, 14>(ATTN_1_reg, ATTN_1_reg);
            sub_col<max_reg, 1, 15>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 0>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 1>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<2, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 2>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 3>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 4>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<3, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 5>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 6>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 7>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<3, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 8>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 9>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 10>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<3, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 11>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 12>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 13>(ATTN_1_reg, ATTN_1_reg);
            mma_AtB<3, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 14>(ATTN_1_reg, ATTN_1_reg);
            exp2<0, 0, 15>(ATTN_1_reg, ATTN_1_reg);
        }
        // Rescale O
        mul_col<scale_reg, 0>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 1>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 2>(O_reg_float_transposed, O_reg_float_transposed);
        macros::s_nop<0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // Cluster 3:
        // Load V2 into shared memory
        G::load<1, false>(v_smem[0], g.Vg, {batch_idx, j - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_0_lds_base);
        macros::s_nop<16>();
        // Load K2 into registers
        load(K_reg, k_smem[0]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 4:
        // QK2
        // Finish softmax for QK1
        __builtin_amdgcn_s_setprio(1);
        {
            mma_ABt<0, 0, 0>(ATTN_0_reg, K_reg, Q_reg);
            exp2<1, 0, 0>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 1>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 2>(ATTN_1_reg, ATTN_1_reg);
            mma_ABt<0, 0, 1>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            exp2<1, 0, 3>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 4>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 5>(ATTN_1_reg, ATTN_1_reg);
            mma_ABt<0, 0, 2>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            exp2<1, 0, 6>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 7>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 8>(ATTN_1_reg, ATTN_1_reg);
            mma_ABt<0, 0, 3>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            exp2<1, 0, 9>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 10>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 11>(ATTN_1_reg, ATTN_1_reg);
            mma_ABt<0, 0, 4>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            exp2<1, 0, 12>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 13>(ATTN_1_reg, ATTN_1_reg);
            exp2<1, 0, 14>(ATTN_1_reg, ATTN_1_reg);
            mma_ABt<0, 0, 5>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            exp2<1, 0, 15>(ATTN_1_reg, ATTN_1_reg);
            mul<norm_reg, norm_reg, scale_reg>();
            zero<other_norm_reg>();
            col_sum<other_norm_reg, 0, 0>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 1>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 2>(ATTN_1_reg);
            mma_ABt<0, 0, 6>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            col_sum<other_norm_reg, 0, 3>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 4>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 5>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 6>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 7>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 8>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 9>(ATTN_1_reg);
            mma_ABt<0, 0, 7>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            col_sum<other_norm_reg, 0, 10>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 11>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 12>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 13>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 14>(ATTN_1_reg);
            col_sum<other_norm_reg, 0, 15>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 0>(ATTN_1_reg);
            mma_ABt<1, 0, 0>(ATTN_0_reg, K_reg, Q_reg);
            col_sum<other_norm_reg, 1, 1>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 2>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 3>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 4>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 5>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 6>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 7>(ATTN_1_reg);
            mma_ABt<1, 0, 1>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            col_sum<other_norm_reg, 1, 8>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 9>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 10>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 11>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 12>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 13>(ATTN_1_reg);
            col_sum<other_norm_reg, 1, 14>(ATTN_1_reg);
            mma_ABt<1, 0, 2>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            col_sum<other_norm_reg, 1, 15>(ATTN_1_reg);
            add<norm_reg, norm_reg, other_norm_reg>();
            mul_col<scale_reg, 3, 0>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 1>(O_reg_float_transposed, O_reg_float_transposed);
            mma_ABt<1, 0, 3>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            mul_col<scale_reg, 3, 2>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 3>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 4>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 5>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 6>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 7>(O_reg_float_transposed, O_reg_float_transposed);
            mma_ABt<1, 0, 4>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            mul_col<scale_reg, 3, 8>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 9>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 10>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 11>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 12>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 13>(O_reg_float_transposed, O_reg_float_transposed);
            mma_ABt<1, 0, 5>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            mul_col<scale_reg, 3, 14>(O_reg_float_transposed, O_reg_float_transposed);
            mul_col<scale_reg, 3, 15>(O_reg_float_transposed, O_reg_float_transposed);
            copy<0, 0, 0>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<0, 0, 1>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<0, 0, 2>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<0, 0, 3>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            mma_ABt<1, 0, 6>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            copy<0, 0, 4>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<0, 0, 5>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<0, 0, 6>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<0, 0, 7>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<1, 0, 0>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<1, 0, 1>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            mma_ABt<1, 0, 7>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
            copy<1, 0, 2>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<1, 0, 3>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<1, 0, 4>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<1, 0, 5>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<1, 0, 6>(ATTN_1_bf16_out_reg, ATTN_1_reg);
            copy<1, 0, 7>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        }
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // Cluster 5:
        // Load K4 into shared memory
        G::load<1, false>(k_smem[0], g.Kg, {batch_idx, j + 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_0_lds_base);
        macros::s_nop<16>();
        // Load V1 into registers
        load(V_reg, v_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // Cluster 6:
        // A1V1
        // Partial softmax for QK2
        __builtin_amdgcn_s_setprio(1);
        {
            mma_AtB<0, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 0, 0>(ATTN_0_reg);
            col_max<max_reg, 0, 1>(ATTN_0_reg);
            col_max<max_reg, 0, 2>(ATTN_0_reg);
            mma_AtB<0, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 0, 3>(ATTN_0_reg);
            col_max<max_reg, 0, 4>(ATTN_0_reg);
            col_max<max_reg, 0, 5>(ATTN_0_reg);
            col_max<max_reg, 0, 6>(ATTN_0_reg);
            mma_AtB<0, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 0, 7>(ATTN_0_reg);
            col_max<max_reg, 1, 0>(ATTN_0_reg);
            col_max<max_reg, 1, 1>(ATTN_0_reg);
            col_max<max_reg, 1, 2>(ATTN_0_reg);
            col_max<max_reg, 1, 3>(ATTN_0_reg);
            mma_AtB<0, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            col_max<max_reg, 1, 4>(ATTN_0_reg);
            col_max<max_reg, 1, 5>(ATTN_0_reg);
            col_max<max_reg, 1, 6>(ATTN_0_reg);
            col_max<max_reg, 1, 7>(ATTN_0_reg);
            mma_AtB<1, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            sub<scale_reg, prev_max_reg, max_reg>();
            copy<prev_max_reg, max_reg>();
            exp2<scale_reg, scale_reg>();
            sub_col<max_reg, 0, 0>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 1>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<1, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 0, 2>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 3>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 4>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 5>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 6>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 7>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<1, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 0, 8>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 9>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 10>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 11>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 12>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 13>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<1, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 0, 14>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 0, 15>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 0>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 1>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 2>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 3>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<2, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 1, 4>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 5>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 6>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 7>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 8>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<2, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 1, 9>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 10>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 11>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 12>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 13>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<2, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            sub_col<max_reg, 1, 14>(ATTN_0_reg, ATTN_0_reg);
            sub_col<max_reg, 1, 15>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 0>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 1>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<2, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 2>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 3>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 4>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<3, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 5>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 6>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 7>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<3, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 8>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 9>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 10>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<3, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 11>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 12>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 13>(ATTN_0_reg, ATTN_0_reg);
            mma_AtB<3, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
            exp2<0, 0, 14>(ATTN_0_reg, ATTN_0_reg);
            exp2<0, 0, 15>(ATTN_0_reg, ATTN_0_reg);
        }
        // Rescale O
        mul_col<scale_reg, 0>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 1>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 2>(O_reg_float_transposed, O_reg_float_transposed);
        macros::s_nop<0>();
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();

        // Cluster 7:
        // Load V3 into shared memory
        G::load<1, false>(v_smem[1], g.Vg, {batch_idx, j, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_1_lds_base);
        macros::s_nop<16>();
        // Load K3 into registers
        load(K_reg, k_smem[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("s_waitcnt vmcnt(4)");
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    // Cluster 0:
    // QK3
    // Finish softmax for QK2
    __builtin_amdgcn_s_setprio(1);
    {
        mma_ABt<0, 0, 0>(ATTN_1_reg, K_reg, Q_reg);
        exp2<1, 0, 0>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 1>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 2>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 1>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 3>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 4>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 5>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 2>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 6>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 7>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 8>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 3>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 9>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 10>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 11>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 4>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 12>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 13>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 14>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 5>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 15>(ATTN_0_reg, ATTN_0_reg);
        mul<norm_reg, norm_reg, scale_reg>();
        zero<other_norm_reg>();
        col_sum<other_norm_reg, 0, 0>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 1>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 2>(ATTN_0_reg);
        mma_ABt<0, 0, 6>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 0, 3>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 4>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 5>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 6>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 7>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 8>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 9>(ATTN_0_reg);
        mma_ABt<0, 0, 7>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 0, 10>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 11>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 12>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 13>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 14>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 15>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 0>(ATTN_0_reg);
        mma_ABt<1, 0, 0>(ATTN_1_reg, K_reg, Q_reg);
        col_sum<other_norm_reg, 1, 1>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 2>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 3>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 4>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 5>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 6>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 7>(ATTN_0_reg);
        mma_ABt<1, 0, 1>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 1, 8>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 9>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 10>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 11>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 12>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 13>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 14>(ATTN_0_reg);
        mma_ABt<1, 0, 2>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 1, 15>(ATTN_0_reg);
        add<norm_reg, norm_reg, other_norm_reg>();
        mul_col<scale_reg, 3, 0>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 1>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 3>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        mul_col<scale_reg, 3, 2>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 3>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 4>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 5>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 6>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 7>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 4>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        mul_col<scale_reg, 3, 8>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 9>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 10>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 11>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 12>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 13>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 5>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        mul_col<scale_reg, 3, 14>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 15>(O_reg_float_transposed, O_reg_float_transposed);
        copy<0, 0, 0>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 1>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 2>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 3>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        mma_ABt<1, 0, 6>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        copy<0, 0, 4>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 5>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 6>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 7>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 0>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 1>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        mma_ABt<1, 0, 7>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        copy<1, 0, 2>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 3>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 4>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 5>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 6>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 7>(ATTN_0_bf16_out_reg, ATTN_0_reg);
    }
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Cluster 1:
    // Load K5 into shared memory
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_K, k_srsrc_base, k_base, k_1_lds_base);
    macros::s_nop<16>();
    // Load V2 into registers
    load(V_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    // A2V2
    // Partial softmax for QK3
    __builtin_amdgcn_s_setprio(1);
    {
        mma_AtB<0, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 0, 0>(ATTN_1_reg);
        col_max<max_reg, 0, 1>(ATTN_1_reg);
        col_max<max_reg, 0, 2>(ATTN_1_reg);
        col_max<max_reg, 0, 3>(ATTN_1_reg);
        col_max<max_reg, 0, 4>(ATTN_1_reg);
        mma_AtB<0, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 0, 5>(ATTN_1_reg);
        col_max<max_reg, 0, 6>(ATTN_1_reg);
        col_max<max_reg, 0, 7>(ATTN_1_reg);
        col_max<max_reg, 1, 0>(ATTN_1_reg);
        col_max<max_reg, 1, 1>(ATTN_1_reg);
        mma_AtB<0, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 1, 2>(ATTN_1_reg);
        col_max<max_reg, 1, 3>(ATTN_1_reg);
        col_max<max_reg, 1, 4>(ATTN_1_reg);
        col_max<max_reg, 1, 5>(ATTN_1_reg);
        col_max<max_reg, 1, 6>(ATTN_1_reg);
        mma_AtB<0, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 1, 7>(ATTN_1_reg);
        sub<scale_reg, prev_max_reg, max_reg>();
        copy<prev_max_reg, max_reg>();
        exp2<scale_reg, scale_reg>();
        mma_AtB<1, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 0>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 1>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 2>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 3>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 4>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<1, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 5>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 6>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 7>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 8>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 9>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<1, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 10>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 11>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 12>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 13>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 14>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<1, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 15>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 0>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 1>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 2>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 3>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 4>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 5>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 6>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 7>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 8>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 9>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 10>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 11>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 12>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 13>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 14>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 15>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 0>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 1>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 2>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 3>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 4>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 5>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 6>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 7>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 8>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 9>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 10>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 11>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 12>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 13>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 14>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 15>(ATTN_1_reg, ATTN_1_reg);
    }
    // Rescale O
    mul_col<scale_reg, 0>(O_reg_float_transposed, O_reg_float_transposed);
    mul_col<scale_reg, 1>(O_reg_float_transposed, O_reg_float_transposed);
    mul_col<scale_reg, 2>(O_reg_float_transposed, O_reg_float_transposed);
    macros::s_nop<0>();
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Cluster 3:
    // Load V4 into shared memory
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, num_tiles - 2, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_0_lds_base);
    macros::s_nop<16>();
    // Load K4 into registers
    load(K_reg, k_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    // QK4
    // Finish softmax for QK3
    __builtin_amdgcn_s_setprio(1);
    {
        mma_ABt<0, 0, 0>(ATTN_0_reg, K_reg, Q_reg);
        exp2<1, 0, 0>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 1>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 2>(ATTN_1_reg, ATTN_1_reg);
        mma_ABt<0, 0, 1>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        exp2<1, 0, 3>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 4>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 5>(ATTN_1_reg, ATTN_1_reg);
        mma_ABt<0, 0, 2>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        exp2<1, 0, 6>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 7>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 8>(ATTN_1_reg, ATTN_1_reg);
        mma_ABt<0, 0, 3>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        exp2<1, 0, 9>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 10>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 11>(ATTN_1_reg, ATTN_1_reg);
        mma_ABt<0, 0, 4>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        exp2<1, 0, 12>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 13>(ATTN_1_reg, ATTN_1_reg);
        exp2<1, 0, 14>(ATTN_1_reg, ATTN_1_reg);
        mma_ABt<0, 0, 5>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        exp2<1, 0, 15>(ATTN_1_reg, ATTN_1_reg);
        mul<norm_reg, norm_reg, scale_reg>();
        zero<other_norm_reg>();
        col_sum<other_norm_reg, 0, 0>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 1>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 2>(ATTN_1_reg);
        mma_ABt<0, 0, 6>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        col_sum<other_norm_reg, 0, 3>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 4>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 5>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 6>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 7>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 8>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 9>(ATTN_1_reg);
        mma_ABt<0, 0, 7>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        col_sum<other_norm_reg, 0, 10>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 11>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 12>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 13>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 14>(ATTN_1_reg);
        col_sum<other_norm_reg, 0, 15>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 0>(ATTN_1_reg);
        mma_ABt<1, 0, 0>(ATTN_0_reg, K_reg, Q_reg);
        col_sum<other_norm_reg, 1, 1>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 2>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 3>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 4>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 5>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 6>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 7>(ATTN_1_reg);
        mma_ABt<1, 0, 1>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        col_sum<other_norm_reg, 1, 8>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 9>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 10>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 11>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 12>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 13>(ATTN_1_reg);
        col_sum<other_norm_reg, 1, 14>(ATTN_1_reg);
        mma_ABt<1, 0, 2>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        col_sum<other_norm_reg, 1, 15>(ATTN_1_reg);
        add<norm_reg, norm_reg, other_norm_reg>();
        mul_col<scale_reg, 3, 0>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 1>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 3>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        mul_col<scale_reg, 3, 2>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 3>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 4>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 5>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 6>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 7>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 4>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        mul_col<scale_reg, 3, 8>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 9>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 10>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 11>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 12>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 13>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 5>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        mul_col<scale_reg, 3, 14>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 15>(O_reg_float_transposed, O_reg_float_transposed);
        copy<0, 0, 0>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<0, 0, 1>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<0, 0, 2>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<0, 0, 3>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        mma_ABt<1, 0, 6>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        copy<0, 0, 4>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<0, 0, 5>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<0, 0, 6>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<0, 0, 7>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<1, 0, 0>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<1, 0, 1>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        mma_ABt<1, 0, 7>(ATTN_0_reg, K_reg, Q_reg, ATTN_0_reg);
        copy<1, 0, 2>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<1, 0, 3>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<1, 0, 4>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<1, 0, 5>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<1, 0, 6>(ATTN_1_bf16_out_reg, ATTN_1_reg);
        copy<1, 0, 7>(ATTN_1_bf16_out_reg, ATTN_1_reg);
    }
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Cluster 5:
    // Load V3 into registers
    load(V_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 6:
    // A3V3
    // Partial softmax for QK4
    __builtin_amdgcn_s_setprio(1);
    {
        mma_AtB<0, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 0, 0>(ATTN_0_reg);
        col_max<max_reg, 0, 1>(ATTN_0_reg);
        col_max<max_reg, 0, 2>(ATTN_0_reg);
        col_max<max_reg, 0, 3>(ATTN_0_reg);
        col_max<max_reg, 0, 4>(ATTN_0_reg);
        mma_AtB<0, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 0, 5>(ATTN_0_reg);
        col_max<max_reg, 0, 6>(ATTN_0_reg);
        col_max<max_reg, 0, 7>(ATTN_0_reg);
        col_max<max_reg, 1, 0>(ATTN_0_reg);
        col_max<max_reg, 1, 1>(ATTN_0_reg);
        mma_AtB<0, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 1, 2>(ATTN_0_reg);
        col_max<max_reg, 1, 3>(ATTN_0_reg);
        col_max<max_reg, 1, 4>(ATTN_0_reg);
        col_max<max_reg, 1, 5>(ATTN_0_reg);
        col_max<max_reg, 1, 6>(ATTN_0_reg);
        mma_AtB<0, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 1, 7>(ATTN_0_reg);
        sub<scale_reg, prev_max_reg, max_reg>();
        copy<prev_max_reg, max_reg>();
        exp2<scale_reg, scale_reg>();
        mma_AtB<1, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 0>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 1>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 2>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 3>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 4>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<1, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 5>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 6>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 7>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 8>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 9>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<1, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 10>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 11>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 12>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 13>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 0, 14>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<1, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 15>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 0>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 1>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 2>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 3>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<2, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 4>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 5>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 6>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 7>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 8>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<2, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 9>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 10>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 11>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 12>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 13>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<2, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 14>(ATTN_0_reg, ATTN_0_reg);
        sub_col<max_reg, 1, 15>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 0>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 1>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<2, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 2>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 3>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 4>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<3, 0, 0>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 5>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 6>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 7>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<3, 0, 1>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 8>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 9>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 10>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<3, 0, 2>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 11>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 12>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 13>(ATTN_0_reg, ATTN_0_reg);
        mma_AtB<3, 0, 3>(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 14>(ATTN_0_reg, ATTN_0_reg);
        exp2<0, 0, 15>(ATTN_0_reg, ATTN_0_reg);
    }
    // Rescale O
    mul_col<scale_reg, 0>(O_reg_float_transposed, O_reg_float_transposed);
    mul_col<scale_reg, 1>(O_reg_float_transposed, O_reg_float_transposed);
    mul_col<scale_reg, 2>(O_reg_float_transposed, O_reg_float_transposed);
    macros::s_nop<0>();
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Cluster 7:
    // Load V5 into shared memory
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, num_tiles - 1, head_idx_kv, 0}, swizzled_offsets_V, v_srsrc_base, v_base, v_1_lds_base);
    macros::s_nop<16>();
    // Load K5 into registers
    load(K_reg, k_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    // QK5
    // Finish softmax for QK4
    __builtin_amdgcn_s_setprio(1);
    {
        mma_ABt<0, 0, 0>(ATTN_1_reg, K_reg, Q_reg);
        exp2<1, 0, 0>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 1>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 2>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 1>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 3>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 4>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 5>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 2>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 6>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 7>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 8>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 3>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 9>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 10>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 11>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 4>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 12>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 13>(ATTN_0_reg, ATTN_0_reg);
        exp2<1, 0, 14>(ATTN_0_reg, ATTN_0_reg);
        mma_ABt<0, 0, 5>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        exp2<1, 0, 15>(ATTN_0_reg, ATTN_0_reg);
        mul<norm_reg, norm_reg, scale_reg>();
        zero<other_norm_reg>();
        col_sum<other_norm_reg, 0, 0>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 1>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 2>(ATTN_0_reg);
        mma_ABt<0, 0, 6>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 0, 3>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 4>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 5>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 6>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 7>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 8>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 9>(ATTN_0_reg);
        mma_ABt<0, 0, 7>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 0, 10>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 11>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 12>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 13>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 14>(ATTN_0_reg);
        col_sum<other_norm_reg, 0, 15>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 0>(ATTN_0_reg);
        mma_ABt<1, 0, 0>(ATTN_1_reg, K_reg, Q_reg);
        col_sum<other_norm_reg, 1, 1>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 2>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 3>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 4>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 5>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 6>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 7>(ATTN_0_reg);
        mma_ABt<1, 0, 1>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 1, 8>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 9>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 10>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 11>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 12>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 13>(ATTN_0_reg);
        col_sum<other_norm_reg, 1, 14>(ATTN_0_reg);
        mma_ABt<1, 0, 2>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        col_sum<other_norm_reg, 1, 15>(ATTN_0_reg);
        add<norm_reg, norm_reg, other_norm_reg>();
        mul_col<scale_reg, 3, 0>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 1>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 3>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        mul_col<scale_reg, 3, 2>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 3>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 4>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 5>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 6>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 7>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 4>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        mul_col<scale_reg, 3, 8>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 9>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 10>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 11>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 12>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 13>(O_reg_float_transposed, O_reg_float_transposed);
        mma_ABt<1, 0, 5>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        mul_col<scale_reg, 3, 14>(O_reg_float_transposed, O_reg_float_transposed);
        mul_col<scale_reg, 3, 15>(O_reg_float_transposed, O_reg_float_transposed);
        copy<0, 0, 0>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 1>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 2>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 3>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        mma_ABt<1, 0, 6>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        copy<0, 0, 4>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 5>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 6>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<0, 0, 7>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 0>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 1>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        mma_ABt<1, 0, 7>(ATTN_1_reg, K_reg, Q_reg, ATTN_1_reg);
        copy<1, 0, 2>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 3>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 4>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 5>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 6>(ATTN_0_bf16_out_reg, ATTN_0_reg);
        copy<1, 0, 7>(ATTN_0_bf16_out_reg, ATTN_0_reg);
    }
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Cluster 9:
    // Load V4 into registers
    load(V_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 10:
    // A4V4
    // Full softmax for QK5
    __builtin_amdgcn_s_setprio(1);
    {
        mma_AtB<0, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 0, 0>(ATTN_1_reg);
        col_max<max_reg, 0, 1>(ATTN_1_reg);
        col_max<max_reg, 0, 2>(ATTN_1_reg);
        col_max<max_reg, 0, 3>(ATTN_1_reg);
        col_max<max_reg, 0, 4>(ATTN_1_reg);
        mma_AtB<0, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 0, 5>(ATTN_1_reg);
        col_max<max_reg, 0, 6>(ATTN_1_reg);
        col_max<max_reg, 0, 7>(ATTN_1_reg);
        col_max<max_reg, 1, 0>(ATTN_1_reg);
        col_max<max_reg, 1, 1>(ATTN_1_reg);
        mma_AtB<0, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 1, 2>(ATTN_1_reg);
        col_max<max_reg, 1, 3>(ATTN_1_reg);
        col_max<max_reg, 1, 4>(ATTN_1_reg);
        col_max<max_reg, 1, 5>(ATTN_1_reg);
        col_max<max_reg, 1, 6>(ATTN_1_reg);
        mma_AtB<0, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        col_max<max_reg, 1, 7>(ATTN_1_reg);
        sub<scale_reg, prev_max_reg, max_reg>();
        copy<prev_max_reg, max_reg>();
        exp2<scale_reg, scale_reg>();
        mma_AtB<1, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 0>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 1>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 2>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 3>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 4>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<1, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 5>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 6>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 7>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 8>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 9>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<1, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 10>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 11>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 12>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 13>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 0, 14>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<1, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 0, 15>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 0>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 1>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 2>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 3>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 4>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 5>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 6>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 7>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 8>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 9>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 10>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 11>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 12>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 13>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        sub_col<max_reg, 1, 14>(ATTN_1_reg, ATTN_1_reg);
        sub_col<max_reg, 1, 15>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 0>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 1>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<2, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 2>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 3>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 4>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 0>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 5>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 6>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 7>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 1>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 8>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 9>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 10>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 2>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 11>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 12>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 13>(ATTN_1_reg, ATTN_1_reg);
        mma_AtB<3, 0, 3>(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
        exp2<0, 0, 14>(ATTN_1_reg, ATTN_1_reg);
        exp2<0, 0, 15>(ATTN_1_reg, ATTN_1_reg);
    }
    // Rescale O
    mul_col<scale_reg>(O_reg_float_transposed, O_reg_float_transposed);

    exp2<1, 0>(ATTN_1_reg, ATTN_1_reg);
    mul<norm_reg, norm_reg, scale_reg>();

    zero<other_norm_reg>();
    col_sum<other_norm_reg>(ATTN_1_reg);
    add<norm_reg, norm_reg, other_norm_reg>();
    copy(ATTN_1_bf16_out_reg, ATTN_1_reg);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Cluster 11:
    // Load V5 into registers
    load(V_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    // A5V5
    __builtin_amdgcn_s_setprio(1);
    mma_AtB(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
    copy<other_norm_reg, norm_reg>();
    div_col<norm_reg>(O_reg_float_transposed, O_reg_float_transposed);
    copy(O_reg_bf16_transposed, O_reg_float_transposed);
    __builtin_amdgcn_s_setprio(0);
    
    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    store<1>(g.Og, O_reg, {batch_idx, 0, head_idx, 0}, {0, tile_idx, 0, 0});

    mul<max_reg, max_reg>(0.69314718056f);
    log<other_norm_reg, other_norm_reg>();
    mul<other_norm_reg, other_norm_reg>(0.69314718056f);
    add<other_norm_reg, other_norm_reg, max_reg>();
    store<other_norm_reg, typename rt<float, KV_BLOCK_SIZE, Q_BLOCK_SIZE, col_l, rt_32x32_s, ATTN_0_ranges>::row_vec>(g.L_vec, {batch_idx, head_idx, 0, 0}, {0, 0, 0, tile_idx});

}

template<int D>
void dispatch_micro(attn_globals<D> g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker<D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_asm, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro<ATTN_D>>(m, "dispatch_micro", 
        &attn_globals<ATTN_D>::Qg, 
        &attn_globals<ATTN_D>::Kg, 
        &attn_globals<ATTN_D>::Vg, 
        &attn_globals<ATTN_D>::Og,
        &attn_globals<ATTN_D>::L_vec
    );
}
