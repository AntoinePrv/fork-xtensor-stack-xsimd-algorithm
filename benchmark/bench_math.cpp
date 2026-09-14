/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>
#include <format>
#include <string_view>

#include <benchmark/benchmark.h>

#include "bench_utils.hpp"
#include "xsimd_test_utils/map_unary_data.hpp"
#include "xsimd_test_utils/utils.hpp"

using xsimd::builder::alignment;

namespace
{
    template <typename Op, typename Alloc, typename Apply>
    void bench_unary(benchmark::State& state, Apply apply)
    {
        using input_t = typename Op::input_t;

        auto const size = static_cast<std::size_t>(state.range(0));
        auto [input, output] = Op::template make_input_output<Alloc>(size);

        for (auto _ : state)
        {
            apply(xsimd::test::as_span(input), xsimd::test::as_span(output));
            benchmark::DoNotOptimize(output.data());
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * size));
        state.SetBytesProcessed(
            static_cast<std::int64_t>(state.iterations() * size * 2 * sizeof(input_t)));
    }

    template <typename Op, typename Alloc, typename Arch, alignment aligned = alignment {}>
    void bench_map_unary(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(
            state,
            [](auto in, auto out)
            { Op::template range_apply_map_unary<aligned, Arch>(in, out); });
    }

    template <typename Op, typename Alloc, typename Arch>
    void bench_transform(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(
            state,
            [](auto in, auto out)
            { Op::template range_apply_transform<Arch>(in, out); });
    }

    template <typename Op, typename Alloc>
    void bench_scalar(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(state, [](auto in, auto out)
                               { Op::range_apply_scalar(in, out); });
    }

    template <typename Op, typename Arch, typename Bench>
    void register_bench(std::string_view variant, Bench bench_fn)
    {
        using input_t = typename Op::input_t;

        auto* bench = benchmark::RegisterBenchmark(
            std::format("{}/{}/{}/{}", Arch::name(), Op::name, xsimd::bench::type_name<input_t>(), variant),
            bench_fn);
        for (auto const size : xsimd::bench::bench_sizes<input_t>())
        {
            bench->Arg(size);
        }
    }

    template <typename Op>
    void register_benches()
    {
        using input_t = typename Op::input_t;
        using arch = xsimd::default_arch;
        using aligned_alloc = typename xsimd::test::aligned_vector<input_t, arch>::allocator_type;
        using unaligned_alloc = typename xsimd::test::unaligned_vector<input_t, arch>::allocator_type;

        // To avoid an explosion of benchmarks, we probagly want to only benchmark aligned for
        // math ops, and benchmark map_unary/transform setups (alignment...) separately on a
        // few ops.
        register_bench<Op, arch>("scalar/aligned", bench_scalar<Op, aligned_alloc>);
        register_bench<Op, arch>("simd-map/aligned", bench_map_unary<Op, aligned_alloc, arch, alignment { .start_aligned = true }>);
        register_bench<Op, arch>("simd-map/unaligned", bench_map_unary<Op, unaligned_alloc, arch>);
        register_bench<Op, arch>("simd-transform/aligned", bench_map_unary<Op, aligned_alloc, arch>);
        register_bench<Op, arch>("simd-transform/unaligned", bench_map_unary<Op, unaligned_alloc, arch>);
    }

    bool const registered = []
    {
        register_benches<xsimd::test::sqrt_op<float>>();
        register_benches<xsimd::test::sqrt_op<double>>();
        register_benches<xsimd::test::abs_op<float>>();
        register_benches<xsimd::test::abs_op<double>>();
        register_benches<xsimd::test::exp_op<float>>();
        register_benches<xsimd::test::exp_op<double>>();
        register_benches<xsimd::test::widen_op<std::int32_t>>();
        return true;
    }();
}
