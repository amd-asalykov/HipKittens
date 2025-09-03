/**
 * @file
 * @brief Functions for transferring data directly between shared memory and registers and back.
 */

#pragma once

#include <type_traits>

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"
#include "../util/util.cuh"

namespace kittens {

/**
 * @brief Load data from a shared vector into a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination register vector.
 * @param src[in]  The source shared vector.
 */
#ifdef KITTENS_CDNA4
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);
    
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
                    dst[w][i * 4 + j] = base_types::convertor<T2, U2>::convert(*(U2*)&src.data[idx + i * 16 + j * 2]);
                }
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, accum_align_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            // int idx = w*32 + 4*(laneid/32);
            // // this should be a maximally coalesced load.
            // #pragma unroll
            // for(int i = 0; i < 4; i++) {
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         dst[w][i * 2 + j] = base_types::convertor<T2, U2>::convert(*(U2*)&src.data[idx + i * 8 + j * 2]);
            //     }
            // }
            int idx = w*16 + 4*(laneid/16) + laneid%4;
            dst[w][0] = base_types::convertor<T, U>::convert(src.data[idx]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            int o_dim = w*2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                dst[o_dim][0] = base_types::convertor<T, U>::convert(src.data[idx]);
            }
        }


        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            const int o_dim = w*2;
            const int other_o_dim = o_dim + 1;
            if constexpr (std::is_same_v<T, float>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__float_as_uint(dst[o_dim][0]), __float_as_uint(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __uint_as_float(res.x);
                if (other_o_dim < dst.outer_dim) {
                    dst[other_o_dim][0] = __uint_as_float(res.y);
                }
            }
            else if constexpr (std::is_same_v<T, bf16>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__bfloat16_as_ushort(dst[o_dim][0]), __bfloat16_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_bfloat16(res.x);
                if (other_o_dim < dst.outer_dim) {
                    dst[other_o_dim][0] = __ushort_as_bfloat16(res.y);
                }
            }
            else if constexpr (std::is_same_v<T, half>) {
                uint2_t res = __builtin_amdgcn_permlane32_swap(__half_as_ushort(dst[o_dim][0]), __half_as_ushort(dst[o_dim][0]), false, true);
                dst[o_dim][0] = __ushort_as_half(res.x);
                if (other_o_dim < dst.outer_dim) {
                    dst[other_o_dim][0] = __ushort_as_half(res.y);
                }
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
                dst[w][0] = base_types::convertor<T, U>::convert(src.data[idx]);
            }
        }
    }
}
#else
template<ducks::rv::all RV, ducks::sv::all SV>
__device__ inline static void load(RV &dst, const SV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);
    
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*128 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced load.
            if(idx < dst.length) {
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(*(U2*)&src.data[idx]);
            }
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
                T tmp = base_types::convertor<T, U>::convert(src.data[idx]);
                if(laneid%2==0) dst[o_dim][0].x =  tmp;
                else dst[o_dim][0].y = tmp;
            }
        }

        // now we need to do a bunch of shuffle_sync's to make sure everyone has everything they need.
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid/4)*4 + 2*(w%2); // repeats every 64 columns
            dst[w][0].x = packed_shfl(MASK_ALL, dst[w][0].x, leader);
            dst[w][0].y = packed_shfl(MASK_ALL, dst[w][0].y, leader+1);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < dst.length) {
                dst[w][0] = base_types::convertor<T, U>::convert(src.data[idx]);
            }
        }
    }
}
#endif

/**
 * @brief Store data into a shared vector from a register vector.
 *
 * @tparam RV The register vector type
 * @tparam SV The shared vector type
 * @param dst[out] The destination shared vector.
 * @param src[in]  The source register vector.
 */
#ifdef KITTENS_CDNA4
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);
    
    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*2*kittens::WARP_THREADS + 16*((laneid%32)/4) + 8*(laneid/32) + 2*(laneid%4);
            int o_dim = w*4 + ((laneid%32)/8);
            int i_dim = (laneid%8);
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length)
                *(U2*)&dst.data[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, accum_align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*2*kittens::WARP_THREADS + 8*((laneid%32)/2) + 4*(laneid/32) + 2*(laneid%2);
            int o_dim = w*4 + ((laneid%32)/8);
            int i_dim = (laneid%8);
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length)
                *(U2*)&dst.data[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*kittens::WARP_THREADS + laneid;
            int o_dim = w*2 + laneid/32;
            // this should be a maximally coalesced load.
            if(idx < src.length) {
                dst.data[idx] = base_types::convertor<U, T>::convert(src[o_dim][0]);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < src.length) {
                dst.data[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}
#else
template<ducks::sv::all SV, ducks::rv::all RV>
__device__ inline static void store(SV &dst, const RV &src) {
    using T2 = RV::dtype;
    using U = SV::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    static_assert(SV::length == RV::length);
    
    int laneid = ::kittens::laneid();

    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*128 + 2 * laneid;
            int o_dim = w*4 + (laneid/8) / 2;
            int i_dim = (laneid/8) % 2;
            // this should be a maximally coalesced store. I hope!
            if(idx < src.length) 
                *(U2*)&dst.data[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
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
                if(laneid%2==0) dst.data[idx] = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                else dst.data[idx] = base_types::convertor<U, T>::convert(src[o_dim][0].y);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            int idx = w*64 + laneid;
            if(idx < src.length) {
                dst.data[idx] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}
#endif
}