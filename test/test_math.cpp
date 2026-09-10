/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include "xsimd_algorithm/math.hpp"

#ifndef XSIMD_NO_SUPPORTED_ARCHITECTURE

#include "doctest/doctest.h"

#include <cmath>
#include <cstddef>
#include <span>

#include "utils.hpp"

using namespace xsimd::test;

namespace
{
    template <typename T>
    struct sqrt_op
    {
        using value_type = T;

        template <xsimd::builder::alignment_options aligned = {}>
        static void apply(std::span<T const> in, std::span<T> out)
        {
            xsimd::algo::sqrt<aligned>(in, out);
        }

        static T scalar(T x)
        {
            return std::sqrt(x);
        }

        template <typename Alloc>
        static std::vector<T, Alloc> input(std::size_t size)
        {
            return make_arange<T, Alloc>(size);
        }
    };

    template <typename T>
    struct abs_op
    {
        using value_type = T;

        template <xsimd::builder::alignment_options aligned = {}>
        static void apply(std::span<T const> in, std::span<T> out)
        {
            xsimd::algo::abs<aligned>(in, out);
        }

        static T scalar(T x)
        {
            return std::abs(x);
        }

        template <typename Alloc>
        static std::vector<T, Alloc> input(std::size_t size)
        {
            return make_arange<T, Alloc>(size, -static_cast<T>(size) / 2);
        }
    };

    template <typename Op, typename Alloc, xsimd::builder::alignment_options aligned = {}>
    void check_unary_math()
    {
        using value_type = typename Op::value_type;
        // Not a multiple of the batch size, to exercise the tail.
        constexpr std::size_t test_size = 94;

        const auto input = Op::template input<Alloc>(test_size);
        auto output = std::vector<value_type, Alloc>(input.size());

        Op::template apply<aligned>(as_span(input), as_span(output));

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            CAPTURE(i);
            CHECK(output[i] == doctest::Approx(Op::scalar(input[i])));
        }
    }
}

TEST_CASE_TEMPLATE(
    "unary math",
    Op,
    sqrt_op<float>, sqrt_op<double>,
    abs_op<float>, abs_op<double>)
{
    using value_type = typename Op::value_type;

    SUBCASE("aligned without header")
    {
        using allocator = typename aligned_vector<value_type>::allocator_type;
        check_unary_math<Op, allocator, { .start_aligned = true }>();
    }

    SUBCASE("aligned with header")
    {
        using allocator = typename aligned_vector<value_type>::allocator_type;
        check_unary_math<Op, allocator>();
    }

    SUBCASE("unaligned with header")
    {
        using allocator = typename unaligned_vector<value_type>::allocator_type;
        check_unary_math<Op, allocator>();
    }
}

#endif
