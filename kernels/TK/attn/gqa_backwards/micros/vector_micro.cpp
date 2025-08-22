#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;


constexpr int b = 1;
constexpr int h = 1;
constexpr int n = 32;
constexpr int d = 32;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> in;
    gl<bf16, -1, -1, -1, -1> in_vec;
    gl<bf16, -1, -1, -1, -1> out;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {


    rt_fl<n, d> tile;
    rt_fl<n, d>::col_vec vec;
    load(tile, g.in, {0, 0, 0, 0});
    load(vec, g.in_vec, {0, 0, 0, 0});
    __syncthreads();
    mul_row(tile, tile, vec);
    // row_sum(vec, tile);
    __syncthreads();
    store(g.out, vec, {0, 0, 0, 0});
    __syncthreads();

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::in, &micro_globals::in_vec, &micro_globals::out);
}

