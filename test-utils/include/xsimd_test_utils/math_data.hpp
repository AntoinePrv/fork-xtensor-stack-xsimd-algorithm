/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_TEST_UTILS_MATH_DATA_HPP
#define XSIMD_ALGORITHM_TEST_UTILS_MATH_DATA_HPP

#include <cmath>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "xsimd_algorithm/builder.hpp"
#include "xsimd_algorithm/math.hpp"

#include "xsimd_test_utils/utils.hpp"

namespace xsimd::test
{
    /// Derives the scalar range application from the element-wise Derived::apply.
    template <typename Derived, typename T>
    struct unary_op
    {
        using value_type = T;

        static void apply_range_scalar(std::span<T const> in, std::span<T> out)
        {
            for (std::size_t i = 0; i < in.size(); ++i)
            {
                out[i] = Derived::apply(in[i]);
            }
        }

        template <typename Alloc>
        static std::pair<std::vector<T, Alloc>, std::vector<T, Alloc>> make_input_output(std::size_t size)
        {
            auto input = Derived::template make_input<Alloc>(size);
            auto output = std::vector<T, Alloc>(input.size());
            return { std::move(input), std::move(output) };
        }
    };

    /*******************
     *  Test fixtures  *
     *******************/

    template <typename T>
    struct sqrt_op : unary_op<sqrt_op<T>, T>
    {
        static constexpr auto name = "sqrt";

        static T apply(T x)
        {
            return std::sqrt(x);
        }

        template <xsimd::builder::alignment_options aligned = {}>
        static void apply_range_simd(std::span<T const> in, std::span<T> out)
        {
            xsimd::algo::sqrt<aligned>(in, out);
        }

        template <typename Alloc>
        static std::vector<T, Alloc> make_input(std::size_t size)
        {
            return make_arange<T, Alloc>(size);
        }
    };

    template <typename T>
    struct abs_op : unary_op<abs_op<T>, T>
    {
        static constexpr auto name = "abs";

        static T apply(T x)
        {
            return std::abs(x);
        }

        template <xsimd::builder::alignment_options aligned = {}>
        static void apply_range_simd(std::span<T const> in, std::span<T> out)
        {
            xsimd::algo::abs<aligned>(in, out);
        }

        template <typename Alloc>
        static std::vector<T, Alloc> make_input(std::size_t size)
        {
            return make_arange<T, Alloc>(size, -static_cast<T>(size) / 2);
        }
    };
}

#endif
