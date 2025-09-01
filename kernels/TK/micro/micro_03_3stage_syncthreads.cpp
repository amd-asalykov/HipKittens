#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;

#define M 8192
#define K 8192
#define N 8192

struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a, b;
    gl<bf16, -1, -1, -1, -1> c;
    dim3 grid()  { return dim3(N / NEW_COL_BLOCK_SIZE, M / NEW_ROW_BLOCK_SIZE); } 
    dim3 block() { return dim3(NUM_THREADS); } 
    size_t dynamic_shared_memory() { return 3 * (M_BLOCK + N_BLOCK) * BLOCK_SIZE * BLOCK_SIZE * sizeof(bf16); } 
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_tk(const micro_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&As)[3][M_BLOCK] =
    al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 3, M_BLOCK>();
    st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row> (&Bs)[3][N_BLOCK] =
    al.allocate<st_bf<BLOCK_SIZE, BLOCK_SIZE, ducks::st_layout::row>, 3, N_BLOCK>();
    rt_fl<BLOCK_SIZE, BLOCK_SIZE, accum_col_l> C_accum;

    int row = blockIdx.y * M_BLOCK;
    int col = blockIdx.x * N_BLOCK;

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = kittens::warpgroupid();
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;
    
    int s = 0, n1 = 1, n2 = 2;
    if (is_producer) {
        // preload tile 0 into stage s
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[s][m],  g.a, {0,0, row+m, 0});
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[s][n],  g.b, {0,0, col+n, 0});
        // preload tile 1 into stage n1
        for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[n1][m], g.a, {0,0, row+m, 1});
        for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[n1][n], g.b, {0,0, col+n, 1});
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();


    if (is_consumer) {zero(C_accum);}
    const int num_tiles = K / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {

        // producers: prefetch tile+2 into n2
        if (is_producer && (tile + 2) < num_tiles) {
            for (int m=0; m<M_BLOCK; ++m) G::load<2,false>(As[n2][m], g.a, {0,0, row+m, tile+2});
            for (int n=0; n<N_BLOCK; ++n) G::load<2,false>(Bs[n2][n], g.b, {0,0, col+n, tile+2});
            __builtin_amdgcn_s_waitcnt(0);
        }
        // consumers: compute from current stage s
        else if (is_consumer) {
            rt_bf<BLOCK_SIZE,BLOCK_SIZE,row_l> a_reg, b_reg;
            load(a_reg, As[s][consumer_idx]);
            load(b_reg, Bs[s][local_warp_id]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum, a_reg, b_reg, C_accum);
            __builtin_amdgcn_s_setprio(0);
        }
        __syncthreads();

        int tmp = s;
        s = n1; n1 = n2; n2 = tmp;
    }
    if (is_consumer) {
        store(g.c, C_accum, {0, 0, row + consumer_idx, col + local_warp_id});
    }
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    hipEventRecord(start);
    micro_tk<<<g.grid(), g.block(), mem_size>>>(g);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float ms=0.f; hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start); hipEventDestroy(stop);
    printf("kernel_ms=%.3f\n", ms);
    hipDeviceSynchronize();
  }

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", &micro_globals::a, &micro_globals::b, &micro_globals::c);
}

