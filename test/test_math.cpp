/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>
#include <type_traits>

#include <doctest/doctest.h>

#include <xsimd_test_utils/map_unary_data.hpp>
#include <xsimd_test_utils/utils.hpp>

namespace
{
    template <typename Op, typename Alloc, xsimd::builder::alignment aligned = xsimd::builder::alignment{}>
    void check_unary_math()
    {
        // Not a multiple of the batch size, to exercise the tail.
        constexpr std::size_t test_size = 94;

        auto [input, output] = Op::template make_input_output<Alloc>(test_size);

        Op::template apply_range_simd<aligned>(xsimd::test::as_span(input), xsimd::test::as_span(output));

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            CAPTURE(i);
            if constexpr (std::is_floating_point_v<typename Op::output_t>)
            {
                CHECK(output[i] == doctest::Approx(Op::apply_scalar(input[i])));
            }
            else
            {
                CHECK(output[i] == Op::apply_scalar(input[i]));
            }
        }
    }
}

TEST_CASE_TEMPLATE(
    "unary math",
    Op,
    xsimd::test::sqrt_op<float>,
    xsimd::test::sqrt_op<double>,
    xsimd::test::abs_op<float>,
    xsimd::test::abs_op<double>,
    xsimd::test::exp_op<float>,
    xsimd::test::exp_op<double>,
    xsimd::test::widen_op<std::int32_t>)
{
    using input_t = typename Op::input_t;
    using aligned_allocator = typename xsimd::test::aligned_vector<input_t>::allocator_type;
    using unaligned_allocator = typename xsimd::test::unaligned_vector<input_t>::allocator_type;

    SUBCASE("aligned without header")
    {
        check_unary_math<Op, aligned_allocator, xsimd::builder::alignment { .start_aligned = true }>();
    }

    SUBCASE("aligned with header")
    {
        check_unary_math<Op, aligned_allocator>();
    }

    SUBCASE("unaligned with header")
    {
        check_unary_math<Op, unaligned_allocator>();
    }
}
