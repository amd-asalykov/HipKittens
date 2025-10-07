/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
* @namespace rt_shape
* 
* @brief A namespace for template metaprogramming with register tile layouts.
* Assumption below is that the col is the reduction dimension
*/
namespace rt_shape {
 
template<int _rows, int _cols, int _stride>
struct rt_shape {
    static constexpr int rows = _rows;
    static constexpr int cols = _cols;
    static constexpr int stride = _stride;
    static constexpr int num_elements = rows*cols;
    static constexpr int elements_per_thread = num_elements / kittens::WARP_THREADS;
    static constexpr int num_strides = elements_per_thread / stride;
};

using rt_16x16 = rt_shape<16, 16, 4>;
using rt_32x32 = rt_shape<32, 32, 4>;
using rt_16x32 = rt_shape<16, 32, 8>;
using rt_32x16 = rt_shape<32, 16, 8>;
using rt_32x16_4 = rt_shape<32, 16, 4>;

template<typename T>
concept all = std::is_same_v<T, rt_16x16> || std::is_same_v<T, rt_32x32> || std::is_same_v<T, rt_16x32> || std::is_same_v<T, rt_32x16> || std::is_same_v<T, rt_32x16_4>;

} // namespace rt_shape
} // namespace ducks
} // namespace kittens