#include "shared_to_register.cuh"

#ifdef TEST_WARP_MEMORY_TILE_SHARED_TO_REGISTER

template<typename T>
struct sharedreg_load_store {
    using dtype = T;

    // NOTE: 'valid' does NOT take dtype; it uses the enclosing T
    template<typename RT_SHAPE, typename ST_SHAPE, int H, int W, int NW, kittens::ducks::rt_layout::all RL>
    using valid = std::bool_constant<
        (NW == 1 && W*H <= 16) &&
        kittens::ducks::st_shape::supports_dtype<ST_SHAPE, T>::value &&
        (W*H*ST_SHAPE::cols*ST_SHAPE::rows*sizeof(T) <= kittens::MAX_SHARED_MEMORY)
    >;

    static inline const std::string test_identifier =
        std::is_same_v<T, kittens::bf16> ? "shared_reg_loadstore_gmem=bf16" :
        std::is_same_v<T, kittens::half> ? "shared_reg_loadstore_gmem=half" :
                                           "shared_reg_loadstore_gmem=float";

    template<int H, int W, int NW,
             kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all RL>
    __host__ static void host_func(const std::vector<float>& i_ref, std::vector<float>& o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }

    template<typename RT_SHAPE, typename ST_SHAPE, typename DTYPE,
             int H, int W, int NW,
             kittens::ducks::gl::all GL, kittens::ducks::rt_layout::all RL>
    __device__ static void device_func(const GL input, const GL output) {
        static_assert(std::is_same_v<DTYPE, T>, "dtype mismatch");

        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<16> al((int*)&__shm[0]);

        using ST_TILE = kittens::st<T, ST_SHAPE::rows*H, ST_SHAPE::cols*W, ST_SHAPE>;
        ST_TILE& shared_tile = al.allocate<ST_TILE>();

        kittens::load(shared_tile, input, {0,0,0,0});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        kittens::rt<T, ST_SHAPE::rows*H, ST_SHAPE::cols*W, RL> reg_tile;
        kittens::load(reg_tile, shared_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        kittens::store(shared_tile, reg_tile);
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();

        kittens::store(output, shared_tile, {0,0,0,0});
    }
};

void warp::memory::tile::shared_to_register::tests(test_data& results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1 :
                         INTENSITY_1 ? 2 :
                         INTENSITY_2 ? 4 :
                         INTENSITY_3 ? 8 :
                         INTENSITY_4 ? 16 : -1;

    using RT_SHAPE_1 = kittens::ducks::rt_shape::rt_32x16;
    using ST_SHAPE_1 = kittens::ducks::st_shape::st_32x16;
    sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_1, ST_SHAPE_1,
                            SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    // sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_1, ST_SHAPE_1,
    //                         SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results); // NOTE: col layout is failing

    using RT_SHAPE_2 = kittens::ducks::rt_shape::rt_32x16_4;
    using ST_SHAPE_2 = kittens::ducks::st_shape::st_32x16;
    sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_2, ST_SHAPE_2,
                            SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    // sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_2, ST_SHAPE_2,
    //                         SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);


    using RT_SHAPE_7     = kittens::ducks::rt_shape::rt_16x32_4;
    using ST_SHAPE_7 = kittens::ducks::st_shape::st_16x32;
    sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_7, ST_SHAPE_7,
                            SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    // sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_7, ST_SHAPE_7,
    //                         SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);


    using RT_SHAPE_3 = kittens::ducks::rt_shape::rt_16x32;
    using ST_SHAPE_3 = kittens::ducks::st_shape::st_16x32;
    sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_3, ST_SHAPE_3,
                            SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    // sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_3, ST_SHAPE_3,
    //                         SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);

    using RT_SHAPE_4 = kittens::ducks::rt_shape::rt_16x16;
    using ST_SHAPE_4 = kittens::ducks::st_shape::st_16x16;
    // sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_4, ST_SHAPE_4,
                            // SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results); // NOTE: unsupported (< 1024 bytes)

    using RT_SHAPE_5 = kittens::ducks::rt_shape::rt_32x32;
    using ST_SHAPE_5 = kittens::ducks::st_shape::st_32x32;
    sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_5, ST_SHAPE_5,
                            SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    // sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_5, ST_SHAPE_5,
    //                         SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    

    using RT_SHAPE_6 = kittens::ducks::rt_shape::rt_32x32_8;
    using ST_SHAPE_6 = kittens::ducks::st_shape::st_32x32;
    sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_6, ST_SHAPE_6,
                            SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    // sweep_gmem_type_2d_warp<sharedreg_load_store, RT_SHAPE_6, ST_SHAPE_6,
    //                         SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
}
#endif
