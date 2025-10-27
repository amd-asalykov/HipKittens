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
    size_t dynamic_shared_memory() { return 0; }
};

__launch_bounds__(NUM_THREADS, 2)
__global__ void attend_ker(const attn_globals g) {

    int warp_idx = warpid();

    rt<bf16, 64, 128, col_l, rt_16x32_4_s> A_reg;
    rt<bf16, 64, 32, col_l, rt_16x32_4_s> B_reg;
    rt<float, 128, 32, col_l, rt_32x32_s> C_reg_transposed;

    load(A_reg, g.A, {0, 0, 0, 0});
    load(B_reg, g.B, {0, 0, 0, warp_idx});
    zero(C_reg_transposed);

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();

    mma_AtB(C_reg_transposed, A_reg, B_reg, C_reg_transposed);

    rt<float, 32, 128, row_l, rt_32x32_s> C_reg;
    swap_layout_and_transpose(C_reg, C_reg_transposed);
    store(g.C, C_reg, {0, 0, warp_idx, 0});
}

void dispatch_micro(attn_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)attend_ker, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    attend_ker<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", 
        &attn_globals::A,
        &attn_globals::B,
        &attn_globals::C
    );
}
