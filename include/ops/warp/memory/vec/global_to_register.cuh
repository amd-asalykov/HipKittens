/**
 * @file
 * @brief Functions for transferring data directly between global memory and registers and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

/**
 * @brief Load data into a register vector from a source array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the source array.
 * @param[out] dst The destination register vector to load data into.
 * @param[in] src The source array in global memory to load data from.
 */
#ifdef KITTENS_CDNA4
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    // TODO: this uses no inter-thread communication and is therefore not optimal.
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*32 + 8*(laneid/32);
            // this should be a maximally coalesced load.
            #pragma unroll
            for(int i = 0; i < 2; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    dst[w][i * 4 + j] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx + i * 16 + j * 2]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, accum_align_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            // int idx = w*16 + 4*(laneid/16);
            // // this should be a maximally coalesced load.
            // #pragma unroll
            // for(int i = 0; i < 2; i++) {
            //     dst[w][i] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx + i * 2]);
            // }
            int idx = w*16 + 4*(laneid/16) + laneid%4;
            dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            int o_dim = w*2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                dst[o_dim][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
            }
        }


        #pragma unroll
        for(auto w = 0; w < dst.outer_dim/2; w++) {
            const int o_dim = w*2;
            const int other_o_dim = o_dim + 1;
            if constexpr (std::is_same_v<T, float>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(dst[o_dim][0]), __float_as_uint(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __uint_as_float(res.x);
                dst[other_o_dim][0] = __uint_as_float(res.y);
            }
            else if constexpr (std::is_same_v<T, bf16>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(dst[o_dim][0]), __bfloat16_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_bfloat16(res.x);
                dst[other_o_dim][0] = __ushort_as_bfloat16(res.y);
            }
            else if constexpr (std::is_same_v<T, half>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(dst[o_dim][0]), __half_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_half(res.x);
                dst[other_o_dim][0] = __ushort_as_half(res.y);
            } else {
                static_assert(false, "Unsupported type");
            }
        }

        if constexpr (RV::outer_dim % 2 == 1) {
            const int o_dim = dst.outer_dim - 1;
            if constexpr (std::is_same_v<T, float>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(dst[o_dim][0]), __float_as_uint(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __uint_as_float(res.x);
            }
            else if constexpr (std::is_same_v<T, bf16>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(dst[o_dim][0]), __bfloat16_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_bfloat16(res.x);
            }
            else if constexpr (std::is_same_v<T, half>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(dst[o_dim][0]), __half_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_half(res.x);
            } else {
                static_assert(false, "Unsupported type");
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
            }
        }
    }
}
#else
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void load(RV &dst, const GL &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *src_ptr = (U*)&src[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*128 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced load.
            if(idx < dst.length)
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src_ptr[idx]);
        }
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = 16*(w%4) + (laneid%8); // repeats every 128 columns
            dst[w][0] = packed_shfl(MASK_ALL, dst[w][0], leader);
            dst[w][1] = packed_shfl(MASK_ALL, dst[w][1], leader+8);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid%8)*8 + (laneid/8);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                T tmp = base_types::convertor<T, U>::convert(src_ptr[idx]);
                if(laneid%2==0) dst[o_dim][0].x = tmp;
                else dst[o_dim][0].y = tmp;
            }
        }
        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid/4)*4 + 2*(w%2); // repeats every 128 columns
            dst[w][0].x = packed_shfl(MASK_ALL, dst[w][0].x, leader);
            dst[w][0].y = packed_shfl(MASK_ALL, dst[w][0].y, leader+1);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src_ptr[idx]);
            }
        }
    }
}
#endif

/**
 * @brief Store data from a register vector to a destination array in global memory.
 *
 * @tparam RV The register vector type.
 * @tparam U The data type of the destination array.
 * @param[out] dst The destination array in global memory to store data into.
 * @param[in] src The source register vector to store data from.
 */

#ifdef KITTENS_CDNA4
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*2*kittens::WARP_THREADS + 16*((laneid%32)/4) + 8*(laneid/32) + 2*(laneid%4);
            int o_dim = w*4 + ((laneid%32)/8);
            int i_dim = (laneid%8);
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length)
                *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, accum_align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            // int idx = w*2*kittens::WARP_THREADS + 8*((laneid%32)/2) + 4*(laneid/32) + 2*(laneid%2);
            // int o_dim = w*4 + ((laneid%32)/8);
            // int i_dim = (laneid%8);
            // // this should be a maximally coalesced store. I hope!
            // if(idx < src.length)
            //     *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
            int idx = w*64 + ((laneid%16)/4)*16 + (laneid/16)*4 + (laneid%4);
            int o_dim = w*4 + ((laneid%16)/4);
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length)
                dst_ptr[idx] = base_types::convertor<U, T>::convert(src[o_dim][0]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            int o_dim = w*2 + laneid/32;
            // this should be a maximally coalesced load.
            if(idx < src.length) {
                dst_ptr[idx] = base_types::convertor<U, T>::convert(src[o_dim][0]);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < src.length) {
                dst_ptr[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}
#else
template<ducks::rv::all RV, ducks::gl::all GL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void store(const GL &dst, const RV &src, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename GL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<-1, 3>())];
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*128 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length)
                *(U2*)&dst_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        // really hoping https://stackoverflow.com/questions/15029765/is-coalescing-triggered-for-accessing-memory-in-reverse-order is still true
        // otherwise there will be some pain :/
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid%8)*8 + (laneid/8);
            int o_dim = w*2 + (laneid%4) / 2;
            // this should be a maximally coalesced load.
            if(idx < src.length) {
                U tmp;
                if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                dst_ptr[idx] = tmp;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < src.length) {
                dst_ptr[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}
#endif

} // namespace kittens