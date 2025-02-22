#pragma once
#define RASTERIO_RVV_HPP_INCLUDED

#include <riscv_vector.h>

#include <type_traits>
#include <limits>

namespace rasterio_rvv
{
namespace detail
{

template <size_t S>
using size_to_uint = std::conditional_t<
    (S == 1), uint8_t,
    std::conditional_t<
        (S == 2), uint16_t,
        std::conditional_t<(S == 4), uint32_t,
                           std::conditional_t<(S == 8), uint64_t, void>>>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t LOGM> struct rvv_helper;

#define RVV_HELPER(T, NAME, SIZE, EEW, EMUL, LOGM)                             \
    template <> struct rvv_helper<T, LOGM>                                     \
    {                                                                          \
        using type = v##NAME##EMUL##_t;                                        \
        static size_t setvl(size_t n)                                          \
        {                                                                      \
            return __riscv_vsetvl_e##SIZE##EMUL(n);                            \
        }                                                                      \
        static type le(const T *ptr, size_t vl)                                \
        {                                                                      \
            return __riscv_vle##SIZE##_v_##EEW##EMUL(ptr, vl);                 \
        }                                                                      \
    }

RVV_HELPER(uint8_t, uint8, 8, u8, m1, 0);
RVV_HELPER(uint8_t, uint8, 8, u8, m2, 1);
RVV_HELPER(uint8_t, uint8, 8, u8, m4, 2);
RVV_HELPER(uint8_t, uint8, 8, u8, m8, 3);

RVV_HELPER(uint16_t, uint16, 16, u16, m1, 0);
RVV_HELPER(uint16_t, uint16, 16, u16, m2, 1);
RVV_HELPER(uint16_t, uint16, 16, u16, m4, 2);
RVV_HELPER(uint16_t, uint16, 16, u16, m8, 3);

RVV_HELPER(uint32_t, uint32, 32, u32, m1, 0);
RVV_HELPER(uint32_t, uint32, 32, u32, m2, 1);
RVV_HELPER(uint32_t, uint32, 32, u32, m4, 2);
RVV_HELPER(uint32_t, uint32, 32, u32, m8, 3);

RVV_HELPER(uint64_t, uint64, 64, u64, m1, 0);
RVV_HELPER(uint64_t, uint64, 64, u64, m2, 1);
RVV_HELPER(uint64_t, uint64, 64, u64, m4, 2);
RVV_HELPER(uint64_t, uint64, 64, u64, m8, 3);

#undef RVV_HELPER

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct rvv_traits;

#define RVV_TRAITS(T, NAME, SIZE, EMUL, LOGM)                                  \
    template <> struct rvv_traits<v##NAME##EMUL##_t>                           \
    {                                                                          \
        using scalar = T;                                                      \
        using type = v##NAME##EMUL##_t;                                        \
        static constexpr size_t logm = LOGM;                                   \
        static void se(scalar *base, type value, size_t vl)                    \
        {                                                                      \
            return __riscv_vse##SIZE(base, value, vl);                         \
        }                                                                      \
    }

// decltype(__riscv_vmv_x(declval<type>))

RVV_TRAITS(uint8_t, uint8, 8, mf8, -3);
RVV_TRAITS(uint8_t, uint8, 8, mf4, -2);
RVV_TRAITS(uint8_t, uint8, 8, mf2, -1);
RVV_TRAITS(uint8_t, uint8, 8, m1, 0);
RVV_TRAITS(uint8_t, uint8, 8, m2, 1);
RVV_TRAITS(uint8_t, uint8, 8, m4, 2);
RVV_TRAITS(uint8_t, uint8, 8, m8, 3);

RVV_TRAITS(uint16_t, uint16, 16, m1, 0);
RVV_TRAITS(uint16_t, uint16, 16, m2, 1);
RVV_TRAITS(uint16_t, uint16, 16, m4, 2);
RVV_TRAITS(uint16_t, uint16, 16, m8, 3);

RVV_TRAITS(uint32_t, uint32, 32, m1, 0);
RVV_TRAITS(uint32_t, uint32, 32, m2, 1);
RVV_TRAITS(uint32_t, uint32, 32, m4, 2);
RVV_TRAITS(uint32_t, uint32, 32, m8, 3);

RVV_TRAITS(float, float32, 32, m1, 0);
RVV_TRAITS(float, float32, 32, m2, 1);
RVV_TRAITS(float, float32, 32, m4, 2);
RVV_TRAITS(float, float32, 32, m8, 3);

RVV_TRAITS(double, float64, 64, m1, 0);
RVV_TRAITS(double, float64, 64, m2, 1);
RVV_TRAITS(double, float64, 64, m4, 2);
RVV_TRAITS(double, float64, 64, m8, 3);

RVV_TRAITS(int8_t, int8, 8, m1, 0);
RVV_TRAITS(int8_t, int8, 8, m2, 1);
RVV_TRAITS(int8_t, int8, 8, m4, 2);
RVV_TRAITS(int8_t, int8, 8, m8, 3);

RVV_TRAITS(int16_t, int16, 16, m1, 0);
RVV_TRAITS(int16_t, int16, 16, m2, 1);
RVV_TRAITS(int16_t, int16, 16, m4, 2);
RVV_TRAITS(int16_t, int16, 16, m8, 3);

#undef RVV_TRAITS

////////////////////////////////////////////////////////////////////////////////

template <typename DstS, typename SrcT>
inline auto rvv_uint_cvt(SrcT src, size_t vl)
{
    static_assert(std::is_integral_v<DstS>);

    using src_scalar = typename rvv_traits<SrcT>::scalar;
    static constexpr size_t src_size = sizeof(src_scalar);
    static constexpr size_t dst_size = sizeof(DstS);

    if constexpr (dst_size > src_size)
        return rvv_uint_cvt<DstS>(__riscv_vwcvtu_x(src, vl), vl);
    // TODO: use vzext?

    else if constexpr (dst_size < src_size)
        return rvv_uint_cvt<DstS>(__riscv_vncvt_x(src, vl), vl);

    else
        return src;
}

template <typename DstS, typename SrcT>
inline auto rvv_fp_cvt(SrcT src, size_t vl)
{
    static_assert(std::is_floating_point_v<DstS>);

    using src_scalar = typename rvv_traits<SrcT>::scalar;
    static constexpr size_t src_size = sizeof(src_scalar);
    static constexpr size_t dst_size = sizeof(DstS);

    if constexpr (dst_size > src_size)
        return rvv_fp_cvt<DstS>(__riscv_vfwcvt_f(src, vl), vl);

    else if constexpr (dst_size < src_size)
        return rvv_fp_cvt<DstS>(__riscv_vfncvt_f(src, vl), vl);

    else
        return src;
}

template <typename DstS, typename SrcT>
inline auto rvv_uint_to_fp(SrcT src, size_t vl)
{
    using src_scalar = typename rvv_traits<SrcT>::scalar;
    static constexpr size_t dst_size = sizeof(DstS);
    static constexpr size_t src_size = sizeof(src_scalar);

    if constexpr (dst_size > src_size)
        return __riscv_vfwcvt_f(
            rvv_uint_cvt<size_to_uint<dst_size / 2>>(src, vl), vl);

    else if constexpr (dst_size < src_size)
        return __riscv_vfncvt_f(
            rvv_uint_cvt<size_to_uint<dst_size * 2>>(src, vl), vl);

    else
        return __riscv_vfcvt_f(src, vl);
}

template <typename DstS, typename SrcT>
inline auto rvv_fp_to_uint(SrcT src, size_t vl)
{
}

template <typename DstS, typename SrcT> inline auto rvv_cvt(SrcT src, size_t vl)
{
    using src_scalar = typename rvv_traits<SrcT>::scalar;
    using dst_scalar = DstS;
    if constexpr (std::is_integral_v<src_scalar> &&
                  std::is_integral_v<dst_scalar>)
    {
        return rvv_uint_cvt<DstS>(src, vl);
    }
    else if constexpr (std::is_integral_v<src_scalar> &&
                       std::is_floating_point_v<dst_scalar>)
    {
        return rvv_uint_to_fp<DstS>(src, vl);
    }
    else
        return src;  // TODO: more cases
}

////////////////////////////////////////////////////////////////////////////////

template <typename TIn, typename TOut>
struct copy_words_lmul_hint : std::integral_constant<int, 0>
{
};

/*
// Add specializations if you have found the optimal lmul for FOO -> BAR
template <>
struct copy_words_lmul_hint<FOO, BAR> : std::integral_constant<int, BLAH>
{
};
*/

template <typename TIn, typename TOut, typename Enable = void>
struct copy_words_fn;

template <typename In, typename Out>
struct copy_words_fn<
    In, Out
    //, std::enable_if_t<std::is_integral_v<UintIn> && std::is_integral_v<UintOut>>
    >
{
    static auto apply(const In *__restrict in, Out *__restrict out, ptrdiff_t n)
    {
        using rvv = rvv_helper<In, copy_words_lmul_hint<In, Out>::value>;
        using scalar_in = In;
        using scalar_out = Out;

        for (; n > 0;)
        {
            const size_t vl = rvv::setvl(n);
            auto src = rvv::le(in, vl);

            if constexpr (std::is_integral_v<scalar_in> &&
                          std::is_integral_v<scalar_out> &&
                          std::numeric_limits<scalar_in>::max() >
                              std::numeric_limits<scalar_out>::max())
            {
                src = __riscv_vminu(src, std::numeric_limits<scalar_out>::max(),
                                    vl);
            }
            // FIXME: add more cases

            auto res = rvv_cvt<scalar_out>(src, vl);

            static_assert(
                std::is_same_v<typename rvv_traits<decltype(res)>::scalar,
                               scalar_out>);

            rvv_traits<decltype(res)>::se(out, res, vl);

            in += vl;
            out += vl;
            n -= vl;
        }
    }
};

template <> struct copy_words_fn<uint16_t, int16_t>
{
    static auto apply(const uint16_t *__restrict in, int16_t *__restrict out,
                      ptrdiff_t n)
    {
        for (; n > 0;)
        {
            const size_t vl = __riscv_vsetvl_e16m1(n);
            auto src = __riscv_vle16_v_u16m1(in, vl);
            auto dst =
                __riscv_vreinterpret_i16m1(__riscv_vminu(src, 32767U, vl));
            __riscv_vse16(out, dst, vl);

            in += vl;
            out += vl;
            n -= vl;
        }
    }
};

}  // namespace detail

template <typename TIn, typename TOut>
inline void copy_words(const TIn *__restrict in, TOut *__restrict out,
                       ptrdiff_t n)
{
    return detail::copy_words_fn<TIn, TOut>::apply(in, out, n);
}

template <ptrdiff_t Stride>
inline void unroll_copy_s_1(uint8_t *__restrict out,
                            const uint8_t *__restrict in, ptrdiff_t n)
{
    for (; n > 0;)
    {
        const size_t vl = __riscv_vsetvl_e8m1(n);
        auto v = __riscv_vlse8_v_u8m1(in, Stride, vl);
        __riscv_vse8(out, v, vl);

        in += vl * Stride;
        out += vl;
        n -= vl;
    }
}

}  // namespace rasterio_rvv
