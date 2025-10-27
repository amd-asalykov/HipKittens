#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

constexpr int ATTN_B = 1; // batch size
constexpr int ATTN_H = 1; // number of heads
constexpr int ATTN_H_KV = 1; // number of heads for key and value
constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV; // queries per KV head group
constexpr int ATTN_N = 256; // sequence length
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

    constexpr float TEMPERATURE_SCALE = 0.08838834764f*1.44269504089f;

    // Register tiles
    using KV_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<192, 255>>, 4>; // 64 registers - v[192:255]
    using Q_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<160, 191>>, 4>; // 32 registers - v[160:191]
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
    constexpr int scale_max_reg = 22; // scale max register
    constexpr int other_norm_reg = 28; // other norm register

    using pinned_registers = ducks::rt::type_list<ducks::rt::range<20, 255>>;
    ducks::rt::clobber<pinned_registers>();

    // Initialize all of the register tiles.
    rt<bf16, KV_BLOCK_SIZE, D, row_l, rt_32x16_s, KV_ranges> K_reg; // 64x128
    rt<bf16, Q_BLOCK_SIZE,  D, row_l, rt_32x16_s, Q_ranges>  Q_reg; // 32x128
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

    // Load in the firt slice of K into shared memory
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 0, head_idx_kv, 0});

    // Load Q into registers
    load<1>(Q_reg, g.Qg, {batch_idx, 0, head_idx, 0}, {0, tile_idx, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // All warps then collaboratively load in the first slice of V (V0) and the second slice of K (K1) into shared memory
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, 1, head_idx_kv, 0});
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, 0, head_idx_kv, 0});

    // Load K0 into registers
    load(K_reg, k_smem[0]);
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(2)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Each warp performs QK0
    mma_ABt(ATTN_0_reg, K_reg, Q_reg);

    // Each warp performs a partial softmax of QK0
    col_max<max_reg>(ATTN_0_reg);
    mul<scale_max_reg, max_reg>(TEMPERATURE_SCALE);
    copy<prev_max_reg, max_reg>();
    fma_col<scale_max_reg>(ATTN_0_reg, ATTN_0_reg, TEMPERATURE_SCALE);
    exp2<0, 0>(ATTN_0_reg, ATTN_0_reg);

    if (stagger) {
        __builtin_amdgcn_s_barrier();
    }

    __builtin_amdgcn_sched_barrier(0);
    // All warps then load in the second slice of K (K1)
    load(K_reg, k_smem[1]);
    // All warps then collaboratively load in the third slice of K (K2) into shared memory
    G::load<1, false>(k_smem[0], g.Kg, {batch_idx, 2, head_idx_kv, 0});
    // All warps then collaboratively load in the second slice of V (V1) into shared memory
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, 1, head_idx_kv, 0});
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    // Epilogue
    // Cluster 0:
    // QK3
    mma_ABt(ATTN_1_reg, K_reg, Q_reg);
    // Finish softmax for QK2
    exp2<1, 0>(ATTN_0_reg, ATTN_0_reg);
    mul<norm_reg, norm_reg, scale_reg>();
    col_sum<norm_reg>(ATTN_0_reg);
    copy(ATTN_0_bf16_out_reg, ATTN_0_reg);
    __builtin_amdgcn_s_barrier();

    // Cluster 1:
    // Load K5 into shared memory
    G::load<1, false>(k_smem[1], g.Kg, {batch_idx, num_tiles - 1, head_idx_kv, 0});
    // Load V2 into registers
    load(V_reg, v_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 2:
    // A2V2
    mma_AtB(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
    // Partial softmax for QK3
    col_max<max_reg>(ATTN_1_reg);
    sub<scale_reg, prev_max_reg, max_reg>();
    copy<prev_max_reg, max_reg>();
    mul<scale_reg, scale_reg>(TEMPERATURE_SCALE);
    mul<scale_max_reg, max_reg>(TEMPERATURE_SCALE);
    exp2<scale_reg, scale_reg>();
    fma_col<scale_max_reg>(ATTN_1_reg, ATTN_1_reg, TEMPERATURE_SCALE);
    exp2<0, 0>(ATTN_1_reg, ATTN_1_reg);
    // Rescale O
    mul_col<scale_reg>(O_reg_float_transposed, O_reg_float_transposed);
    __builtin_amdgcn_s_barrier();

    // Cluster 3:
    // Load V4 into shared memory
    G::load<1, false>(v_smem[0], g.Vg, {batch_idx, num_tiles - 2, head_idx_kv, 0});
    // Load K4 into registers
    load(K_reg, k_smem[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 4:
    // QK4
    mma_ABt(ATTN_0_reg, K_reg, Q_reg);
    // Finish softmax for QK3
    exp2<1, 0>(ATTN_1_reg, ATTN_1_reg);
    mul<norm_reg, norm_reg, scale_reg>();
    col_sum<norm_reg>(ATTN_1_reg);
    copy(ATTN_1_bf16_out_reg, ATTN_1_reg);
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
    mma_AtB(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
    // Partial softmax for QK4
    col_max<max_reg>(ATTN_0_reg);
    sub<scale_reg, prev_max_reg, max_reg>();
    copy<prev_max_reg, max_reg>();
    mul<scale_reg, scale_reg>(TEMPERATURE_SCALE);
    mul<scale_max_reg, max_reg>(TEMPERATURE_SCALE);
    exp2<scale_reg, scale_reg>();
    fma_col<scale_max_reg>(ATTN_0_reg, ATTN_0_reg, TEMPERATURE_SCALE);
    exp2<0, 0>(ATTN_0_reg, ATTN_0_reg);
    // Rescale O
    mul_col<scale_reg>(O_reg_float_transposed, O_reg_float_transposed);
    __builtin_amdgcn_s_barrier();

    // Cluster 7:
    // Load V5 into shared memory
    G::load<1, false>(v_smem[1], g.Vg, {batch_idx, num_tiles - 1, head_idx_kv, 0});
    // Load K5 into registers
    load(K_reg, k_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    asm volatile("s_waitcnt vmcnt(4)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 8:
    // QK5
    mma_ABt(ATTN_1_reg, K_reg, Q_reg);
    // Finish softmax for QK4
    exp2<1, 0>(ATTN_0_reg, ATTN_0_reg);
    mul<norm_reg, norm_reg, scale_reg>();
    col_sum<norm_reg>(ATTN_0_reg);
    copy(ATTN_0_bf16_out_reg, ATTN_0_reg);
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
    mma_AtB(O_reg_float_transposed, V_reg, ATTN_0_bf16_in_reg, O_reg_float_transposed);
    // Full softmax for QK5
    col_max<max_reg>(ATTN_1_reg);
    sub<scale_reg, prev_max_reg, max_reg>();
    copy<prev_max_reg, max_reg>();
    mul<scale_reg, scale_reg>(TEMPERATURE_SCALE);
    mul<scale_max_reg, max_reg>(TEMPERATURE_SCALE);
    exp2<scale_reg, scale_reg>();

    fma_col<scale_max_reg>(ATTN_1_reg, ATTN_1_reg, TEMPERATURE_SCALE);
    exp2(ATTN_1_reg, ATTN_1_reg);
    mul<norm_reg, norm_reg, scale_reg>();
    col_sum<norm_reg>(ATTN_1_reg);
    copy(ATTN_1_bf16_out_reg, ATTN_1_reg);
    // Rescale O
    mul_col<scale_reg>(O_reg_float_transposed, O_reg_float_transposed);
    __builtin_amdgcn_s_barrier();

    // Cluster 11:
    load(V_reg, v_smem[1]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    // Cluster 12:
    // A5V5
    mma_AtB(O_reg_float_transposed, V_reg, ATTN_1_bf16_in_reg, O_reg_float_transposed);
    copy<other_norm_reg, norm_reg>();
    div_col<norm_reg>(O_reg_float_transposed, O_reg_float_transposed);
    __builtin_amdgcn_s_barrier();

    // Conclusion
    if (!stagger) {
        __builtin_amdgcn_s_barrier();
    }

    copy(O_reg_bf16_transposed, O_reg_float_transposed);
    store<1>(g.Og, O_reg, {batch_idx, 0, head_idx, 0}, {0, tile_idx, 0, 0});

    mul<scale_max_reg, scale_max_reg>(0.69314718056f);
    log<other_norm_reg, other_norm_reg>();
    __builtin_amdgcn_s_waitcnt(0);
    mul<other_norm_reg, other_norm_reg>(0.69314718056f);
    add<other_norm_reg, other_norm_reg, scale_max_reg>();
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
