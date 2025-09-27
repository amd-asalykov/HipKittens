/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
 * @namespace rt_layout
 * 
 * @brief A namespace for template metaprogramming with register tile layouts.
 */
namespace rt_layout {

/**
 * @brief A dummy type used to identify a row-major layout for a register tile.
 */
struct row {}; // for most matrices
/**
 * @brief A dummy type used to identify a col-major layout for a register tile.
 */
struct col {}; // for the B-matrix of MMA ops.
/**
 * @brief A dummy type used to identify an accumulator col-major layout for a register tile.
 */
struct accumulator_col {};
/**
 * @brief A dummy type used to identify an accumulator row-major layout for a register tile.
 */
 struct accumulator_row {};

/**
 * @brief A concept to check if a type is a register tile layout.
 */

template<typename T>
concept all = std::is_same_v<T, row> || std::is_same_v<T, col> || std::is_same_v<T, accumulator_col> || std::is_same_v<T, accumulator_row>;

/**
 * @brief A struct to generate a transposed layout.
 * Note: on CDNA4, the accumulator layout becomes the col layout when transposed.
 */
template<all L> struct transpose      { using type = col; };
template<>      struct transpose<col> { using type = row; };
template<>      struct transpose<accumulator_col> { using type = accumulator_row; };
template<>      struct transpose<accumulator_row> { using type = accumulator_col; };

template<all L> struct shuffle{ using type = col; };
template<>      struct shuffle<accumulator_row> { using type = row; };
template<>      struct shuffle<col> { using type = accumulator_col; };
template<>      struct shuffle<row> { using type = accumulator_row; };

} // namespace rt_layout
} // namespace ducks
} // namespace kittens