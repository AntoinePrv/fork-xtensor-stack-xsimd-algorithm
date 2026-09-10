/****************************************************************************
 * Copyright (c) xsimd-algorithm contributors                               *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_ALGORITHM_BUILDER_HPP
#define XSIMD_ALGORITHM_BUILDER_HPP

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include <xsimd/xsimd.hpp>

#include "./macros.hpp"

namespace xsimd::builder
{
    struct alignment
    {
        bool start_aligned = false;
        bool end_aligned = false;
    };

    template <typename T>
    XSIMD_INLINE auto prev_aligned(T* ptr, std::size_t alignment) -> T*
    {
        assert(std::has_single_bit(alignment));
        auto const address = reinterpret_cast<std::size_t>(ptr);
        return reinterpret_cast<T*>(address & ~(alignment - 1));
    }

    template <typename T>
    XSIMD_INLINE auto next_aligned(T* ptr, std::size_t alignment) -> T*
    {
        assert(std::has_single_bit(alignment));
        auto const address = reinterpret_cast<std::size_t>(ptr);
        return reinterpret_cast<T*>((address + alignment - 1) & ~(alignment - 1));
    }

    template <typename T>
    XSIMD_INLINE auto bytes_to_next_aligned(T* ptr, std::size_t alignment) -> std::size_t
    {
        assert(std::has_single_bit(alignment));
        auto const address = reinterpret_cast<std::uintptr_t>(ptr);
        return (alignment - (address & (alignment - 1))) & (alignment - 1);
    }

    template <typename T, typename U>
    XSIMD_INLINE auto are_aliased(std::span<T> lhs, std::span<U> rhs) -> bool
    {
        // Comparing pointers from unrelated objects is unspecified, integers are not.
        auto const lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data());
        auto const rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data());
        return (lhs_begin < rhs_begin + rhs.size_bytes()) && (rhs_begin < lhs_begin + lhs.size_bytes());
    }

    struct unary_options
    {
        std::size_t unroll_factor = 4;
        bool pure = false;
    };

    template <
        typename Arch = xsimd::default_arch,
        typename T, typename U, typename Func>
    XSIMD_INLINE void map_unary_batch(
        T const* XSIMD_RESTRICT begin,
        T const* XSIMD_RESTRICT end,
        U* XSIMD_RESTRICT out,
        Func&& func)
    {
        using input_batch = xsimd::batch<T, Arch>;
        using output_batch = xsimd::batch<U, Arch>;

        assert(begin <= end);
        assert(static_cast<std::size_t>(end - begin) <= input_batch::size);

        if (begin == end) [[unlikely]]
        {
            return;
        }

        alignas(Arch::alignment()) T input_buffer[input_batch::size] {};
        alignas(Arch::alignment()) U output_buffer[output_batch::size];

        const std::size_t in_count = static_cast<std::size_t>(end - begin);
        std::memcpy(input_buffer, begin, in_count * sizeof(T));
        func(input_batch::load_aligned(input_buffer)).store_aligned(output_buffer);
        std::memcpy(out, output_buffer, in_count * sizeof(T));
    }

    template <typename T, typename A, bool aligned>
    XSIMD_INLINE xsimd::batch<T, A> load_batch(T const* ptr)
    {
        if constexpr (aligned)
        {
            return xsimd::batch<T, A>::load_aligned(ptr);
        }
        else
        {
            return xsimd::batch<T, A>::load_unaligned(ptr);
        }
    }

    template <typename T, typename A, bool aligned>
    XSIMD_INLINE void store_batch(xsimd::batch<T, A> x, T* ptr)
    {
        if constexpr (aligned)
        {
            x.store_aligned(ptr);
        }
        else
        {
            x.store_unaligned(ptr);
        }
    }

    template <
        alignment align = {},
        unary_options opts = {},
        typename Arch = xsimd::default_arch,
        typename T, typename U, typename Func>
    XSIMD_INLINE void map_unary(std::span<T const> in, std::span<U> out, Func&& func)
    {
        using input_batch = xsimd::batch<T, Arch>;
        using output_batch = xsimd::batch<U, Arch>;

        // Both sides can only be split at an element boundary.
        // If input is not guarenteed aligned, we will try to align preferably the
        // output (more expensive unaligned stores) or otherwise the input.
        constexpr bool align_output = sizeof(U) >= sizeof(T);
        constexpr bool load_is_aligned = align.start_aligned || !align_output;
        constexpr bool store_is_aligned = align.start_aligned || align_output;

        assert(in.size() * sizeof(T) == out.size() * sizeof(U));
        assert(!are_aliased(in, out));

        if (in.empty()) [[unlikely]]
        {
            return;
        }

        auto ot = out.data();
        auto it = in.data();
        auto const iend = in.data() + in.size();

        // Input and output may not have the same alignment so it may be impossible
        // to get both aligned, so we align a single side.
        if constexpr (!align.start_aligned)
        {
            // The span may be too short to reach the next alignment boundary.
            const auto head_bytes = std::min(
                align_output ? bytes_to_next_aligned(ot, Arch::alignment())
                             : bytes_to_next_aligned(it, Arch::alignment()),
                in.size_bytes());
            assert(head_bytes % sizeof(T) == 0);
            assert(head_bytes % sizeof(U) == 0);

            map_unary_batch<Arch>(it, it + head_bytes / sizeof(T), ot, func);
            it += head_bytes / sizeof(T);
            ot += head_bytes / sizeof(U);
        }

        // Unrolled loop processing multiple batches at a time
        while (static_cast<std::size_t>(iend - it) >= opts.unroll_factor * input_batch::size)
        {
            input_batch x[opts.unroll_factor];
            for (std::size_t u = 0; u < opts.unroll_factor; ++u)
            {
                x[u] = load_batch<T, Arch, load_is_aligned>(it + u * input_batch::size);
            }
            for (std::size_t u = 0; u < opts.unroll_factor; ++u)
            {
                store_batch<U, Arch, store_is_aligned>(func(x[u]), ot + u * output_batch::size);
            }

            it += opts.unroll_factor * input_batch::size;
            ot += opts.unroll_factor * output_batch::size;
        }

        while (static_cast<std::size_t>(iend - it) >= input_batch::size)
        {
            const auto x = load_batch<T, Arch, load_is_aligned>(it);
            store_batch<U, Arch, store_is_aligned>(func(x), ot);
            it += input_batch::size;
            ot += output_batch::size;
        }

        // Unlikely to be skipped, meant for users that know they allocate
        // a multiple of the batch size, such as in a local buffer
        if constexpr (!align.end_aligned)
        {
            auto const oend = out.data() + out.size();
            // Stepping back shifts the batch boundary, which pairs lanes differently
            // than starting from the front unless both sides have the same lane count.
            constexpr bool can_step_back = input_batch::size == output_batch::size;
            if (can_step_back && it != iend && (in.size() >= input_batch::size)) [[likely]]
            {
                // Recompute overlapping data, this time starting from the end.
                const auto x = load_batch<T, Arch, false>(iend - input_batch::size);
                store_batch<U, Arch, false>(func(x), oend - output_batch::size);
            }
            else
            {
                map_unary_batch<Arch>(it, iend, ot, func);
            }
        }
    }
}

#endif
