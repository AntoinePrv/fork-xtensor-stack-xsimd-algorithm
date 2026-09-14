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
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>

#include <xsimd/xsimd.hpp>

#include "./macros.hpp"

namespace xsimd::builder
{
    struct alignment
    {
        bool start_aligned = false;
        bool end_aligned = false;
    };

    /// Return the pointer before the input with the given alignment or itself if aligned.
    template <typename T>
    XSIMD_INLINE auto prev_aligned(T* ptr, std::size_t alignment) -> T*
    {
        assert(std::has_single_bit(alignment));
        auto const address = reinterpret_cast<std::size_t>(ptr);
        return reinterpret_cast<T*>(address & ~(alignment - 1));
    }

    /// Return the pointer after the input with the given alignment or itself if aligned.
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

    /// Check if two spans are aliasing each others (overlapping).
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

    /// Number of batches of T spanning as many elements as one batch of the widest of T and U.
    ///
    /// Pairing that many batches on each side lets both sides advance by the same number of
    /// elements, so a mapping stays elementwise regardless of the respective lane counts.
    template <typename T, typename U>
    inline constexpr std::size_t batch_arity = sizeof(T) / std::min(sizeof(T), sizeof(U));

    /// Load batch wrapper with an alignment as template parameter.
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

    /// Store batch wrapper with an alignment as template parameter.
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

    /// Load an array of batches.
    template <std::size_t N, typename A, bool aligned, typename T>
    XSIMD_INLINE auto load_batches(T const* ptr) -> std::array<xsimd::batch<T, A>, N>
    {
        std::array<xsimd::batch<T, A>, N> x;
        for (std::size_t i = 0; i < N; ++i)
        {
            x[i] = load_batch<T, A, aligned>(ptr + i * xsimd::batch<T, A>::size);
        }
        return x;
    }

    /// Store an array of batches.
    template <std::size_t N, typename A, bool aligned, typename T>
    XSIMD_INLINE void store_batches(std::array<xsimd::batch<T, A>, N> const& x, T* ptr)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            store_batch<T, A, aligned>(x[i], ptr + i * xsimd::batch<T, A>::size);
        }
    }

    namespace internal
    {
        template <typename T>
        inline constexpr bool is_array = false;

        template <typename T, std::size_t N>
        inline constexpr bool is_array<std::array<T, N>> = true;

        /// If an array contains only one element, return it.
        template <typename B, std::size_t N>
        XSIMD_INLINE auto const& unwrap_array(std::array<B, N> const& x)
        {
            if constexpr (N == 1)
            {
                return x[0];
            }
            else
            {
                return x;
            }
        }

        /// Wrap user function to handle ``xsimd::batch`` as 1D array.
        ///
        /// Transform 1D input array as batch from alogrithm functions to batch for to
        /// the user function, and user batch result as 1D arrays for the algorithm
        /// functions.
        template <typename Func>
        XSIMD_INLINE auto wrap_params_as_1d_arrays(Func&& func)
        {
            return [func = std::forward<Func>(func)](auto const&... x)
            {
                auto res = func(internal::unwrap_array(x)...);
                if constexpr (internal::is_array<decltype(res)>)
                {
                    return res;
                }
                else
                {
                    return std::array { res };
                }
            };
        }
    }

    /// Map fewer elements than a full step through a scratch buffer.
    template <
        typename Arch = xsimd::default_arch,
        typename T, typename U, typename Func>
    XSIMD_INLINE void map_unary_batch(
        T const* XSIMD_RESTRICT begin,
        T const* XSIMD_RESTRICT end,
        U* XSIMD_RESTRICT out,
        Func&& func)
    {
        constexpr std::size_t in_arity = batch_arity<T, U>;
        constexpr std::size_t out_arity = batch_arity<U, T>;
        constexpr std::size_t step = in_arity * xsimd::batch<T, Arch>::size;
        static_assert(step == out_arity * xsimd::batch<U, Arch>::size);
        auto mapper = internal::wrap_params_as_1d_arrays(std::forward<Func>(func));

        assert(begin <= end);
        assert(static_cast<std::size_t>(end - begin) <= step);

        if (begin == end) [[unlikely]]
        {
            return;
        }

        alignas(Arch::alignment()) std::array<T, step> input_buffer {};
        alignas(Arch::alignment()) std::array<U, step> output_buffer;

        const std::size_t count = static_cast<std::size_t>(end - begin);
        std::memcpy(input_buffer.data(), begin, count * sizeof(T));
        store_batches<out_arity, Arch, true>(
            mapper(load_batches<in_arity, Arch, true>(input_buffer.data())),
            output_buffer.data());
        std::memcpy(out, output_buffer.data(), count * sizeof(U));
    }

    /// Apply func elementwise over in, writing as many elements to out.
    ///
    /// Func maps a std::array<batch<T, Arch>, batch_arity<T, U>> to a
    /// std::array<batch<U, Arch>, batch_arity<U, T>>, both spanning the same element count.
    /// When arity is one, a callback over plain batches is accepted as well.
    template <
        alignment align = alignment{},
        unary_options opts = unary_options{},
        typename Arch = xsimd::default_arch,
        typename T, typename U, typename Func>
    XSIMD_INLINE void map_unary(std::span<T const> in, std::span<U> out, Func&& func)
    {
        constexpr std::size_t in_arity = batch_arity<T, U>;
        constexpr std::size_t out_arity = batch_arity<U, T>;
        // Elements consumed and produced by a single call to func.
        constexpr std::size_t step = in_arity * xsimd::batch<T, Arch>::size;
        static_assert(step == out_arity * xsimd::batch<U, Arch>::size);
        auto mapper = internal::wrap_params_as_1d_arrays(std::forward<Func>(func));

        // Input and output may not have the same alignment so it may be impossible
        // to get both aligned. We align preferably the output (more expensive
        // unaligned stores) or otherwise the input.
        constexpr bool align_output = sizeof(U) >= sizeof(T);
        constexpr bool load_is_aligned = align.start_aligned || !align_output;
        constexpr bool store_is_aligned = align.start_aligned || align_output;

        assert(in.size() == out.size());
        assert(!are_aliased(in, out));

        if (in.empty()) [[unlikely]]
        {
            return;
        }

        auto ot = out.data();
        auto it = in.data();
        auto const iend = in.data() + in.size();

        if constexpr (!align.start_aligned)
        {
            // The span may be too short to reach the next alignment boundary.
            const auto head = std::min(
                align_output ? bytes_to_next_aligned(ot, Arch::alignment()) / sizeof(U)
                             : bytes_to_next_aligned(it, Arch::alignment()) / sizeof(T),
                in.size());

            if (opts.pure && (head != 0) && (in.size() >= step))
            {
                // Recompute the head as a full step, the body overwrites the excess.
                store_batches<out_arity, Arch, false>(
                    mapper(load_batches<in_arity, Arch, false>(it)),
                    ot);
            }
            else
            {
                map_unary_batch<Arch>(it, it + head, ot, func);
            }
            it += head;
            ot += head;
        }

        // Unrolled loop processing multiple steps at a time
        while (static_cast<std::size_t>(iend - it) >= opts.unroll_factor * step)
        {
            std::array<std::array<xsimd::batch<T, Arch>, in_arity>, opts.unroll_factor> x;
            for (std::size_t u = 0; u < opts.unroll_factor; ++u)
            {
                x[u] = load_batches<in_arity, Arch, load_is_aligned>(it + u * step);
            }
            for (std::size_t u = 0; u < opts.unroll_factor; ++u)
            {
                store_batches<out_arity, Arch, store_is_aligned>(mapper(x[u]), ot + u * step);
            }

            it += opts.unroll_factor * step;
            ot += opts.unroll_factor * step;
        }

        while (static_cast<std::size_t>(iend - it) >= step)
        {
            auto x = load_batches<in_arity, Arch, load_is_aligned>(it);
            store_batches<out_arity, Arch, store_is_aligned>(mapper(x), ot);
            it += step;
            ot += step;
        }

        // Unlikely to be skipped, meant for users that know they allocate
        // a multiple of the batch size, such as in a local buffer
        if constexpr (!align.end_aligned)
        {
            auto const oend = out.data() + out.size();
            if (opts.pure && (it != iend) && (in.size() >= step)) [[likely]]
            {
                // Recompute overlapping data, this time starting from the end.
                auto x = load_batches<in_arity, Arch, false>(iend - step);
                store_batches<out_arity, Arch, false>(mapper(x), oend - step);
            }
            else
            {
                map_unary_batch<Arch>(it, iend, ot, func);
            }
        }
    }
}

#endif
