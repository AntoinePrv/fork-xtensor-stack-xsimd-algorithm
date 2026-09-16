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
        /// Transform 1D input array as batch from algorithm functions to batch for to
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

    template <typename T, typename A, std::size_t N>
    struct alignas(A::alignment()) alignas(T) aligned_array : std::array<T, N>
    {
    };

    template <typename A, typename Out, typename... In>
    struct map_helper
    {
        static constexpr std::size_t min_elem_size = std::min({ sizeof(Out), sizeof(In)... });

        /// Number of batches of T spanning as many elements as one batch of the widest element.
        ///
        /// Pairing that many batches on all side lets both sides advance by the same number of
        /// elements, so a mapping stays elementwise regardless of the respective lane counts.
        template <typename T>
        static constexpr std::size_t batch_arity()
        {
            return sizeof(T) / min_elem_size;
        }

        template <typename T>
        using batch_array = std::array<xsimd::batch<T, A>, batch_arity<T>()>;

        static constexpr std::size_t chunk_size = batch_arity<Out>() * xsimd::batch<Out, A>::size;

        /// Load an array of batches.
        template <bool aligned, typename T>
        XSIMD_INLINE static auto load_batches(T const* ptr) -> batch_array<T>
        {
            batch_array<T> x;
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                x[i] = load_batch<T, A, aligned>(ptr + i * xsimd::batch<T, A>::size);
            }
            return x;
        }

        /// Store an array of batches.
        template <bool aligned, typename T>
        XSIMD_INLINE static void store_batches(batch_array<T> const& x, T* ptr)
        {
            for (std::size_t i = 0; i < x.size(); ++i)
            {
                store_batch<T, A, aligned>(x[i], ptr + i * xsimd::batch<T, A>::size);
            }
        }

        /// Map fewer elements than a full chunk through a scratch buffer.
        template <typename Func>
        XSIMD_INLINE static void map_chunk(
            In const* XSIMD_RESTRICT... begin,
            Out* XSIMD_RESTRICT out,
            std::size_t count,
            Func&& func)
        {
            assert(count <= chunk_size);
            if (count == 0) [[unlikely]]
            {
                return;
            }

            constexpr auto read = []<typename T>(T const* in, std::size_t cnt)
            {
                aligned_array<T, A, chunk_size> in_buffer = {};
                std::memcpy(in_buffer.data(), in, cnt * sizeof(T));
                return load_batches<true>(in_buffer.data());
            };

            aligned_array<Out, A, chunk_size> out_buffer;
            store_batches<true>(func(read(begin, count)...), out_buffer.data());
            std::memcpy(out, out_buffer.data(), count * sizeof(Out));
        }
    };

    /// Apply func elementwise over in, writing as many elements to out.
    ///
    /// Func maps a std::array<batch<T, Arch>, batch_arity<T, U>> to a
    /// std::array<batch<U, Arch>, batch_arity<U, T>>, both spanning the same element count.
    /// When arity is one, a callback over plain batches is accepted as well.
    template <
        alignment align = alignment {},
        unary_options opts = unary_options {},
        typename Arch = xsimd::default_arch,
        typename T, typename U, typename Func>
    XSIMD_INLINE void map_unary(std::span<T const> in, std::span<U> out, Func&& func)
    {
        using H = map_helper<Arch, U, T>;

        // Input and output may not have the same alignment so it may be impossible
        // to get both aligned. We align preferably the output (more expensive
        // unaligned stores) or otherwise the input.
        constexpr bool align_output = sizeof(U) >= sizeof(T);
        constexpr bool load_is_aligned = align.start_aligned || !align_output;
        constexpr bool store_is_aligned = align.start_aligned || align_output;

        assert(in.size() == out.size());
        assert(!are_aliased(in, out));

        auto mapper = internal::wrap_params_as_1d_arrays(std::forward<Func>(func));

        if (in.empty()) [[unlikely]]
        {
            return;
        }

        auto out_iter = out.data();
        auto in_iter = in.data();
        auto const in_end = in.data() + in.size();

        if constexpr (!align.start_aligned)
        {
            // The span may be too short to reach the next alignment boundary.
            const auto head = std::min(
                align_output ? bytes_to_next_aligned(out_iter, Arch::alignment()) / sizeof(U)
                             : bytes_to_next_aligned(in_iter, Arch::alignment()) / sizeof(T),
                in.size());

            if (opts.pure && (head != 0) && (in.size() >= H::chunk_size))
            {
                // Recompute the head as a full step, the body overwrites the excess.
                auto x = H::template load_batches<false>(in_iter);
                H::template store_batches<false>(mapper(x), out_iter);
            }
            else
            {
                H::map_chunk(in_iter, out_iter, head, mapper);
            }
            in_iter += head;
            out_iter += head;
        }

        // Unrolled loop processing multiple steps at a time
        while (static_cast<std::size_t>(in_end - in_iter) >= opts.unroll_factor * H::chunk_size)
        {
            std::array<typename H::template batch_array<T>, opts.unroll_factor> x;
            for (std::size_t u = 0; u < opts.unroll_factor; ++u)
            {
                x[u] = H::template load_batches<load_is_aligned>(in_iter + u * H::chunk_size);
            }
            for (std::size_t u = 0; u < opts.unroll_factor; ++u)
            {
                H::template store_batches<store_is_aligned>(mapper(x[u]), out_iter + u * H::chunk_size);
            }

            in_iter += opts.unroll_factor * H::chunk_size;
            out_iter += opts.unroll_factor * H::chunk_size;
        }

        while (static_cast<std::size_t>(in_end - in_iter) >= H::chunk_size)
        {
            auto x = H::template load_batches<load_is_aligned>(in_iter);
            H::template store_batches<store_is_aligned>(mapper(x), out_iter);
            in_iter += H::chunk_size;
            out_iter += H::chunk_size;
        }

        // Unlikely to be skipped, meant for users that know they allocate
        // a multiple of the batch size, such as in a local buffer
        if constexpr (!align.end_aligned)
        {
            auto const out_end = out.data() + out.size();
            if (opts.pure && (in_iter != in_end) && (in.size() >= H::chunk_size)) [[likely]]
            {
                // Recompute overlapping data, this time starting from the end.
                auto x = H::template load_batches<false>(in_end - H::chunk_size);
                H::template store_batches<false>(mapper(x), out_end - H::chunk_size);
            }
            else
            {
                H::map_chunk(in_iter, out_iter, in_end - in_iter, mapper);
            }
        }
    }
}

#endif
