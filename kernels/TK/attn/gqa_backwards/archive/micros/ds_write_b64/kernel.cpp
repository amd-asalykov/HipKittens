#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#define NUM_WARPS 1
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using G = kittens::group<NUM_WARPS>;

struct micro_globals {
    gl<float, -1, -1, -1, -1> should_I_write;
    gl<float, -1, -1, -1, -1> offset;
    dim3 grid()  { return dim3(1); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return MAX_SHARED_MEMORY; }
};

__global__ __launch_bounds__(NUM_THREADS, 1)
void micro_tk(const micro_globals g) {
    extern __shared__ int smem[];

    const int tid = threadIdx.x;

    uint32_t should_I_write = reinterpret_cast<uint32_t*>(g.should_I_write.raw_ptr)[tid];
    uint32_t shared_memory_offset = reinterpret_cast<uint32_t*>(g.offset.raw_ptr)[tid];

    const uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&smem[0]);
    uint64_t garbage = 0x1234567890abcdef;

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    printf("threadId: %d, should_I_write: %d, shared_memory_offset: %d\n", tid, should_I_write, shared_memory_offset);

    if (should_I_write) {
        const uint32_t addr = dst_ptr + shared_memory_offset;


        asm volatile("ds_write_b64 %0, %1, offset:%2"
        :
        : "v"(addr), "v"(garbage),"i"(0));
    }
}

void dispatch_micro(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::should_I_write, &micro_globals::offset);
}

