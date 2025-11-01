


template <typename T, typename ST, int N_THREADS>
__device__ inline void buffer_load_lds(int i, const T* lds_base, i32x4 srsrc, int row_stride) {
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(fp8e4m3) / (16 * N_THREADS); // 8
    static_assert(memcpy_per_tile > 0, "memcpy_per_tile must be greater than 0. Please decrease the number of threads.");

    constexpr int elem_per_thread = 16 / sizeof(fp8e4m3);
    constexpr int col_dim_threads = ST::cols / elem_per_thread;
    int row_offset = i * elem_per_thread * N_THREADS / ST::cols + threadIdx.x / col_dim_threads;
    int col_offset = threadIdx.x % col_dim_threads * elem_per_thread;

    uintptr_t offset_in_st = (row_offset * ST::underlying_cols + col_offset) * sizeof(T);
    offset_in_st ^= (((offset_in_st % (256*8)) >> 8) << 4);
 
    row_offset = offset_in_st / (ST::underlying_cols * sizeof(T));
    col_offset = offset_in_st % (ST::underlying_cols * sizeof(T)) / sizeof(T);

    uintptr_t offset_in_global = (row_offset * row_stride + col_offset) * sizeof(T);

    const T* lds_elem_ptr = lds_base + (i * N_THREADS * elem_per_thread);
    uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
    as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

    llvm_amdgcn_raw_buffer_load_lds(
        srsrc, // buffer resource
        lds_ptr,
        16, // 16 bytes
        offset_in_global,
        0, 
        0, // instruction offset
        static_cast<int>(coherency::cache_all)); // cache coherency
}

/************************************************************************/

template<typename D, typename A, typename B, typename C>
__device__ inline void mma_ABt_base_wrapper(D& d_mma, const A& a_mma, const B& b_mma, const C& c_mma, int n, int m, int k) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C
    static_assert(std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp8e4m3> &&
                  std::is_same_v<typename B::T, fp8e4m3> && std::is_same_v<typename C::T, float>);
    
    mma_ABt_base(
        d_mma.tiles[n][m],
        a_mma.tiles[n][k],
        b_mma.tiles[m][k],
        c_mma.tiles[n][m]
    );
}
