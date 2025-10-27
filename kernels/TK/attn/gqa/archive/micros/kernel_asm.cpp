#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define NUM_WARPS 2
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using namespace kittens;

using G = kittens::group<NUM_WARPS>;

struct attn_globals { 
    gl<bf16, -1, -1, -1, -1> A;
    gl<bf16, -1, -1, -1, -1> B;
    gl<float, -1, -1, -1, -1> C;
    dim3 grid() { return dim3(1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<64, 128, st_8x32_s> (&A_smem) = al.allocate<st_bf<64, 128, st_8x32_s>>();
    st_bf<64, 64, st_8x32_s> (&B_smem) = al.allocate<st_bf<64, 64, st_8x32_s>>();

    int warp_idx = warpid();

    using A_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<192, 255>>, 4>; // 64 registers - v[192:255]
    using B_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<176, 191>>, 4>; // 16 registers - v[176:191]
    using C_ranges = ducks::rt::split_many_t<ducks::rt::type_list<ducks::rt::range<112, 175>>, 16>; // 64 registers - v[112:175]
    ducks::rt::clobber<A_ranges>();
    ducks::rt::clobber<B_ranges>();
    ducks::rt::clobber<C_ranges>();

    rt<bf16, 64, 128, col_l, rt_16x32_4_s, A_ranges> A_reg;
    rt<bf16, 64, 32, col_l, rt_16x32_4_s, B_ranges> B_reg;
    rt<float, 128, 32, col_l, rt_32x32_s, C_ranges> C_reg_transposed;
    rt<float, 32, 128, row_l, rt_32x32_s, ducks::rt::transpose_2d<C_ranges, 4, 1>> C_reg;

    // Load A and B into shared memory
    G::load(A_smem, g.A, {0, 0, 0, 0});
    G::load(B_smem, g.B, {0, 0, 0, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    load(A_reg, A_smem);
    load(B_reg, subtile_inplace<64, 32>(B_smem, {0, warp_idx}));
    zero(C_reg_transposed);
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    mma_AtB(C_reg_transposed, A_reg, B_reg, C_reg_transposed);

    store(g.C, C_reg, {0, 0, 0, 0}, {0, 0, warp_idx, 0});
}

void dispatch_micro(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel_asm, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", 
        &attn_globals::A,
        &attn_globals::B,
        &attn_globals::C
    );
}
