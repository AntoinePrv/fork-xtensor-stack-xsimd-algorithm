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
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include "xsimd_test_utils/math_data.hpp"
#include "xsimd_test_utils/utils.hpp"

using xsimd::builder::alignment;

namespace
{
    template <typename Op, typename Alloc, typename Apply>
    void bench_unary(benchmark::State& state, Apply apply)
    {
        using value_type = typename Op::value_type;

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
            static_cast<std::int64_t>(state.iterations() * size * 2 * sizeof(value_type)));
    }

    /// Sizes spanning L1-resident to memory-bound, each with and without a scalar tail.
    template <typename T>
    std::vector<std::int64_t> bench_sizes()
    {
        constexpr auto batch_size = static_cast<std::int64_t>(xsimd::batch<T>::size);

        auto sizes = std::vector<std::int64_t> {};
        for (std::int64_t size : { 64, 1024, 65536, 1 << 21 })
        {
            auto const whole = size - (size % batch_size);
            sizes.push_back(whole);
            sizes.push_back(whole + batch_size / 2 + 1);
        }
        return sizes;
    }

    template <typename T>
    constexpr auto type_name()
    {
        if constexpr (std::is_same_v<T, float>)
        {
            return "f32";
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return "f64";
        }
    }

    template <typename Op, typename Alloc, alignment aligned = {}>
    void bench_simd(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(
            state,
            [](auto in, auto out) { Op::template apply_range_simd<aligned>(in, out); });
    }

    template <typename Op, typename Alloc>
    void bench_scalar(benchmark::State& state)
    {
        bench_unary<Op, Alloc>(state, [](auto in, auto out) { Op::apply_range_scalar(in, out); });
    }

    template <typename Op, typename Bench>
    void register_bench(std::string_view variant, Bench bench_fn)
    {
        using value_type = typename Op::value_type;

        auto* bench = benchmark::RegisterBenchmark(
            std::format("{}/{}/{}", Op::name, type_name<value_type>(), variant),
            bench_fn);
        for (auto const size : bench_sizes<value_type>())
        {
            bench->Arg(size);
        }
    }

    template <typename Op>
    void register_benches()
    {
        using value_type = typename Op::value_type;
        using aligned_alloc = typename xsimd::test::aligned_vector<value_type>::allocator_type;
        using unaligned_alloc = typename xsimd::test::unaligned_vector<value_type>::allocator_type;

        register_bench<Op>("scalar/aligned", bench_scalar<Op, aligned_alloc>);
        register_bench<Op>("simd/aligned", bench_simd<Op, aligned_alloc, { .start_aligned = true }>);
        register_bench<Op>("simd/unaligned", bench_simd<Op, unaligned_alloc>);
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
