/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

 #pragma once

 #include <concepts>
 
 namespace kittens {
 namespace ducks {
 /**
  * @namespace rt_matrix
  * 
  * @brief A namespace for template metaprogramming with register tile layouts.
  * Assumption below is that the col is the reduction dimension
  */
 namespace rt_matrix {
 
 /**
  * @brief Base tile dimensions for the mfma_32x32x16 matrix layout.
  */
 struct mfma_32x32x16 {
    static constexpr int tile_size_row_in = 32; 
    static constexpr int tile_size_col_in = 16; 
    static constexpr int tile_size_row_out = 32;
    static constexpr int tile_size_col_out = 32;
 }; 
 /**
  * @brief Base tile dimensions for the mfma_16x16x32 matrix layout.
  */
 struct mfma_16x16x32 {
    static constexpr int tile_size_row_in = 16;
    static constexpr int tile_size_col_in = 32;
    static constexpr int tile_size_row_out = 16;
    static constexpr int tile_size_col_out = 16;
 }; 

 
 template<typename T>
 concept all = std::is_same_v<T, mfma_32x32x16> || std::is_same_v<T, mfma_16x16x32>;

 } // namespace rt_matrix
 } // namespace ducks
 } // namespace kittens

