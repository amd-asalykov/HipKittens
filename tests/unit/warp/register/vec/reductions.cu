#include "reductions.cuh"

#ifdef TEST_WARP_REGISTER_VEC_REDUCTIONS

#define LENGTH 16

struct vec_norm {
    using dtype = float;
    template<int S, int NW, kittens::ducks::rv_layout::all L>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_vec_norm";
    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        float f = 1.f;
        for(int i = 0; i < o_ref.size(); i++) f += i_ref[i];
        for(int i = 0; i < o_ref.size(); i++) o_ref[i] = f;
    }
    template<int S, int NW, gl_t GL, kittens::ducks::rv_layout::all L>
    __device__ static void device_func(const GL &input, const GL &output) {
        kittens::rv_fl<LENGTH*S, L> vec;
        kittens::load(vec, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        float f = 1.f;
        kittens::sum(f, vec, f);
        kittens::zero(vec);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
        kittens::add(vec, vec, f);
        kittens::store(output, vec, {});
    }
};

void warp::reg::vec::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/vec/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    // sweep_size_1d_warp<vec_norm, SIZE, kittens::ducks::rv_layout::align>::run(results); // not supported
    // sweep_size_1d_warp<vec_norm, SIZE, kittens::ducks::rv_layout::ortho>::run(results); // not supported
    sweep_size_1d_warp<vec_norm, SIZE, kittens::ducks::rv_layout::naive>::run(results);
}

#endif