/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>

#include <benchmark/benchmark.h>

#include "map_unary_utils.hpp"
#include "xsimd_test_utils/map_unary_data.hpp"
#include "xsimd_test_utils/utils.hpp"

namespace
{
    using xsimd::bench::bench_map_unary;
    using xsimd::bench::bench_scalar;
    using xsimd::bench::bench_transform;
    using xsimd::bench::register_bench;
    using xsimd::builder::alignment;

    template <typename Op>
    void register_benches()
    {
        using input_t = typename Op::input_t;
        using arch = xsimd::default_arch;
        using aligned_alloc = typename xsimd::test::aligned_vector<input_t, arch>::allocator_type;
        using unaligned_alloc = typename xsimd::test::unaligned_vector<input_t, arch>::allocator_type;

        register_bench<Op, arch>("aligned/scalar", bench_scalar<Op, aligned_alloc>);
        register_bench<Op, arch>("aligned/simd/map:header+trailer", bench_map_unary<Op, aligned_alloc, arch>);
        register_bench<Op, arch>("aligned/simd/map:trailer", bench_map_unary<Op, aligned_alloc, arch, alignment { .start_aligned = true }>);
        register_bench<Op, arch>("aligned/simd/transform:header+trailer", bench_transform<Op, aligned_alloc, arch>);

        register_bench<Op, arch>("unaligned/scalar", bench_scalar<Op, unaligned_alloc>);
        register_bench<Op, arch>("unaligned/simd/map:header+trailer", bench_map_unary<Op, unaligned_alloc, arch>);
        register_bench<Op, arch>("unaligned/simd/map:trailer", bench_map_unary<Op, unaligned_alloc, arch, alignment { .start_aligned = true }>);
        register_bench<Op, arch>("unaligned/simd/transform:header+trailer", bench_transform<Op, unaligned_alloc, arch>);
    }

    bool const registered = []
    {
        register_benches<xsimd::test::sqrt_op<float>>();
        register_benches<xsimd::test::sqrt_op<double>>();
        register_benches<xsimd::test::abs_op<float>>();
        register_benches<xsimd::test::abs_op<double>>();
        register_benches<xsimd::test::exp_op<float>>();
        register_benches<xsimd::test::exp_op<double>>();
        return true;
    }();
}
