/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>
#include <cstdint>

#include <doctest/doctest.h>
#include <xsimd_test_utils/utils.hpp>

#include "xsimd_algorithm/builder.hpp"

/// Map unary test that turned int32 into half has many int64.
TEST_CASE("map_unary int32 to int64")
{
    using input_type = std::int32_t;
    using output_type = std::int64_t;

    // Not a multiple of the batch size, to exercise the tail.
    static constexpr std::size_t input_size = 94;
    static constexpr std::size_t output_size = input_size * sizeof(input_type) / sizeof(output_type);

    const auto input = xsimd::test::make_arange<input_type>(input_size);
    auto output = xsimd::test::aligned_vector<output_type>(output_size);

    const auto func = [](auto const& x)
    { return xsimd::widen(x + input_type { 1 })[0]; };

    xsimd::builder::map_unary(xsimd::test::as_span(input), xsimd::test::as_span(output), func);

    static constexpr std::size_t in_batch_size = xsimd::batch<input_type>::size;
    static constexpr std::size_t out_batch_size = xsimd::batch<output_type>::size;

    for (std::size_t i = 0; i < output_size; ++i)
    {
        const auto in_index = (i / out_batch_size) * in_batch_size + (i % out_batch_size);
        CAPTURE(i);
        CHECK(output[i] == static_cast<output_type>(input[in_index] + 1));
    }
}
