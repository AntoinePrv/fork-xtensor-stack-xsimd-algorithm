/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_MATH_HPP
#define XSIMD_ALGORITHM_MATH_HPP

#include <xsimd/xsimd.hpp>

#include "./builder.hpp"

namespace xsimd::algo
{
    template <
        xsimd::builder::alignment_options aligned = {},
        typename Arch = xsimd::default_arch,
        typename T>
    void sqrt(std::span<T const> in, std::span<T> out)
    {
        return xsimd::builder::map_unary<aligned, Arch>(
            in, out, [](auto x)
            { return sqrt(x); });
    }

    template <
        xsimd::builder::alignment_options aligned = {},
        typename Arch = xsimd::default_arch,
        typename T>
    void abs(std::span<T const> in, std::span<T> out)
    {
        return xsimd::builder::map_unary<aligned, Arch>(
            in, out, [](auto x)
            { return abs(x); });
    }
}

#endif
