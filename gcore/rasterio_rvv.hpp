#pragma once
#define RASTERIO_RVV_HPP_INCLUDED

#include <riscv_vector.h>
#include <limits>
#include <utility>

#include "gdalrvv.hpp"

namespace rasterio_rvv
{
namespace detail
{

using namespace gdalrvv;

// TODO: using std::cmp_less;
template <class T, class U> constexpr bool cmp_less(T t, U u) noexcept
{
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
        return t < u;
    else if constexpr (std::is_signed_v<T>)
        return t < 0 || std::make_unsigned_t<T>(t) < u;
    else
        return u >= 0 && t < std::make_unsigned_t<U>(u);
}

template <typename DstS, typename SrcT>
inline auto rvv_clamp_value(SrcT src, size_t vl)
{
    using scalar_in = typename rvv_traits<SrcT>::scalar;
    using scalar_out = DstS;

    static constexpr auto in_max = std::numeric_limits<scalar_in>::max();
    static constexpr auto in_min = std::numeric_limits<scalar_in>::lowest();
    static constexpr auto out_max = std::numeric_limits<scalar_out>::max();
    static constexpr auto out_min = std::numeric_limits<scalar_out>::lowest();

    if constexpr (cmp_less(out_max, in_max))
    {
        if constexpr (std::is_integral_v<scalar_in>)
        {
            if constexpr (std::is_unsigned_v<scalar_in>)
                src = __riscv_vminu(src, out_max, vl);
            else
                src = __riscv_vmin(src, out_max, vl);
        }
        else
        {
            static_assert(std::is_floating_point_v<scalar_in>);
            src = __riscv_vfmin(src, out_max, vl);
        }
    }

    if constexpr (cmp_less(in_min, out_min))
    {
        if constexpr (std::is_integral_v<scalar_in>)
        {
            if constexpr (std::is_unsigned_v<scalar_in>)
                // src = __riscv_vmaxu(src, max(out_min, 0), vl);
                ;
            else
                src = __riscv_vmax(src, out_min, vl);
        }
        else
        {
            static_assert(std::is_floating_point_v<scalar_in>);
            src = __riscv_vfmax(src, out_min, vl);
        }
    }
    return src;
}

////////////////////////////////////////////////////////////////////////////////

// Convert between different sizes
template <size_t DstScalarSize, typename SrcT>
inline auto rvv_integral_size_cvt(SrcT src, size_t vl)
{
    using src_scalar = typename rvv_traits<SrcT>::scalar;
    static constexpr size_t src_size = sizeof(src_scalar);
    static constexpr size_t dst_size = DstScalarSize;

    // TODO: use vnclip(src, 0, 0, vl)?

    if constexpr (dst_size > src_size)
        return vext<dst_size>(src, vl);

    else if constexpr (dst_size < src_size)
        return rvv_integral_size_cvt<dst_size>(__riscv_vncvt_x(src, vl), vl);

    else
        return src;
}

// Convert between different sizes and signed/unsigned
template <typename DstS, typename SrcT>
inline auto rvv_integral_cvt(SrcT src, size_t vl)
{

    using src_scalar = typename rvv_traits<SrcT>::scalar;
    using dst_scalar = DstS;

    auto tmp = rvv_integral_size_cvt<sizeof(dst_scalar)>(src, vl);

    using cur_type = decltype(tmp);
    using helper_dst_type = rvv_helper<dst_scalar, rvv_traits<cur_type>::logm>;

    static_assert(sizeof(rvv_scalar_t<cur_type>) == sizeof(DstS));

    if constexpr (std::is_unsigned_v<src_scalar> !=
                  std::is_unsigned_v<dst_scalar>)
        return helper_dst_type::reinterpret(tmp);
    else
        return tmp;
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
inline auto rvv_integral_to_fp(SrcT src, size_t vl)
{
    using src_scalar = typename rvv_traits<SrcT>::scalar;
    static constexpr size_t dst_size = sizeof(DstS);
    static constexpr size_t src_size = sizeof(src_scalar);

    if constexpr (dst_size > src_size)
        return __riscv_vfwcvt_f(rvv_integral_size_cvt<dst_size / 2>(src, vl),
                                vl);

    else if constexpr (dst_size < src_size)
        return __riscv_vfncvt_f(rvv_integral_size_cvt<dst_size * 2>(src, vl),
                                vl);

    else
        return __riscv_vfcvt_f(src, vl);
}

template <typename DstS, typename SrcT>
inline auto rvv_fp_to_integral(SrcT src, size_t vl)
{
    using src_scalar = typename rvv_traits<SrcT>::scalar;
    using dst_scalar = DstS;
    static constexpr size_t dst_scalar_size = sizeof(dst_scalar);
    if constexpr (std::is_unsigned_v<dst_scalar>)
    {
        if constexpr (sizeof(dst_scalar) > sizeof(src_scalar))
            return rvv_integral_size_cvt<dst_scalar_size>(
                __riscv_vfwcvt_xu(src, __RISCV_FRM_RMM, vl), vl);
        else if constexpr (sizeof(dst_scalar) < sizeof(src_scalar))
        {
            auto v = __riscv_vfncvt_xu(src, __RISCV_FRM_RMM, vl);
            v = rvv_clamp_value<dst_scalar>(v, vl);
            return rvv_integral_size_cvt<dst_scalar_size>(v, vl);
        }
        else
            return __riscv_vfcvt_xu(src, __RISCV_FRM_RMM, vl);
    }
    else
    {
        if constexpr (sizeof(dst_scalar) > sizeof(src_scalar))
            return rvv_integral_size_cvt<dst_scalar_size>(
                __riscv_vfwcvt_x(src, __RISCV_FRM_RMM, vl), vl);
        else if constexpr (sizeof(dst_scalar) < sizeof(src_scalar))
        {
            auto v = __riscv_vfncvt_x(src, __RISCV_FRM_RMM, vl);
            v = rvv_clamp_value<dst_scalar>(v, vl);
            return rvv_integral_size_cvt<dst_scalar_size>(v, vl);
        }
        else
            return __riscv_vfcvt_x(src, __RISCV_FRM_RMM, vl);
    }
}

template <typename DstS, typename SrcT> inline auto rvv_cvt(SrcT src, size_t vl)
{
    using src_scalar = typename rvv_traits<SrcT>::scalar;
    using dst_scalar = DstS;
    if constexpr (std::is_integral_v<src_scalar> &&
                  std::is_integral_v<dst_scalar>)
    {
        return rvv_integral_cvt<DstS>(rvv_clamp_value<DstS>(src, vl), vl);
    }
    else if constexpr (std::is_integral_v<src_scalar> &&
                       std::is_floating_point_v<dst_scalar>)
    {
        return rvv_integral_to_fp<DstS>(src, vl);
    }
    else if constexpr (std::is_floating_point_v<src_scalar> &&
                       std::is_integral_v<dst_scalar>)
    {
        // Map nan to zero, is this necessary? (ref: gdal_priv_templates.hpp)
        auto fclass = __riscv_vfclass(src, vl);
        auto mask_nan = __riscv_vmsgeu(fclass, 0x100, vl);
        auto result = rvv_fp_to_integral<DstS>(src, vl);
        result = __riscv_vmerge(result, 0, mask_nan, vl);
        return result;
    }
    else if constexpr (std::is_floating_point_v<src_scalar> &&
                       std::is_floating_point_v<dst_scalar>)
    {
        return rvv_fp_cvt<DstS>(src, vl);
    }
    else
        return src;
}

////////////////////////////////////////////////////////////////////////////////

template <typename TIn, typename TOut> struct copy_words_lmul_hint
{
    static constexpr auto insize = sizeof(TIn);
    static constexpr auto outsize = sizeof(TOut);
    static constexpr int value = outsize == 8 * insize   ? 0
                                 : outsize == 4 * insize ? 1
                                 : outsize == 2 * insize ? 2
                                 : outsize == insize     ? 0
                                 : outsize * 2 == insize ? 0
                                                         : 3;
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

template <typename In, typename Out> struct copy_words_fn<In, Out>
{
    static auto apply(const In *__restrict in, Out *__restrict out, ptrdiff_t n)
    {
        using rvv = rvv_helper<In, copy_words_lmul_hint<In, Out>::value>;
        using scalar_out = Out;

        for (; n > 0;)
        {
            const size_t vl = rvv::setvl(n);
            auto src = rvv::le(in, vl);

            auto res = rvv_cvt<scalar_out>(src, vl);

            rvv_traits<decltype(res)>::se(out, res, vl);

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

inline void deinterleave_3byte(const uint8_t *__restrict pabySrc,
                               uint8_t *__restrict pabyDest0,
                               uint8_t *__restrict pabyDest1,
                               uint8_t *__restrict pabyDest2, size_t nIters)
{
    for (; nIters > 0;)
    {
        const size_t vl = __riscv_vsetvl_e8m2(nIters);
        const auto vx3 = __riscv_vlseg3e8_v_u8m2x3(pabySrc, vl);

        __riscv_vse8(pabyDest0, __riscv_vget_u8m2(vx3, 0), vl);
        __riscv_vse8(pabyDest1, __riscv_vget_u8m2(vx3, 1), vl);
        __riscv_vse8(pabyDest2, __riscv_vget_u8m2(vx3, 2), vl);

        nIters -= vl;

        pabySrc += 3 * vl;
        pabyDest0 += vl;
        pabyDest1 += vl;
        pabyDest2 += vl;
    }
}

inline void deinterleave_3_16b(const uint16_t *__restrict pabySrc,
                               uint16_t *__restrict pabyDest0,
                               uint16_t *__restrict pabyDest1,
                               uint16_t *__restrict pabyDest2, size_t nIters)
{
    for (; nIters > 0;)
    {
        const size_t vl = __riscv_vsetvl_e16m2(nIters);
        const auto vx3 = __riscv_vlseg3e16_v_u16m2x3(pabySrc, vl);
        __riscv_vse16(pabyDest0, __riscv_vget_u16m2(vx3, 0), vl);
        __riscv_vse16(pabyDest1, __riscv_vget_u16m2(vx3, 1), vl);
        __riscv_vse16(pabyDest2, __riscv_vget_u16m2(vx3, 2), vl);

        nIters -= vl;

        pabySrc += 3 * vl;
        pabyDest0 += vl;
        pabyDest1 += vl;
        pabyDest2 += vl;
    }
}

inline void deinterleave_4_16b(const uint16_t *__restrict pabySrc,
                               uint16_t *__restrict pabyDest0,
                               uint16_t *__restrict pabyDest1,
                               uint16_t *__restrict pabyDest2,
                               uint16_t *__restrict pabyDest3, size_t nIters)
{
    for (; nIters > 0;)
    {
        const size_t vl = __riscv_vsetvl_e16m2(nIters);
        const auto vx4 = __riscv_vlseg4e16_v_u16m2x4(pabySrc, vl);
        __riscv_vse16(pabyDest0, __riscv_vget_u16m2(vx4, 0), vl);
        __riscv_vse16(pabyDest1, __riscv_vget_u16m2(vx4, 1), vl);
        __riscv_vse16(pabyDest2, __riscv_vget_u16m2(vx4, 2), vl);
        __riscv_vse16(pabyDest3, __riscv_vget_u16m2(vx4, 3), vl);

        nIters -= vl;

        pabySrc += 4 * vl;
        pabyDest0 += vl;
        pabyDest1 += vl;
        pabyDest2 += vl;
        pabyDest3 += vl;
    }
}

inline void deinterleave_4byte(const uint8_t *__restrict pabySrc,
                               uint8_t *__restrict pabyDest0,
                               uint8_t *__restrict pabyDest1,
                               uint8_t *__restrict pabyDest2,
                               uint8_t *__restrict pabyDest3, size_t nIters)
{
#pragma GCC unroll 2
    for (; nIters > 0;)
    {
        const size_t vl = __riscv_vsetvl_e8m1(nIters);
        const auto vx4 = __riscv_vlseg4e8_v_u8m1x4(pabySrc, vl);

        __riscv_vse8(pabyDest0, __riscv_vget_u8m1(vx4, 0), vl);
        __riscv_vse8(pabyDest1, __riscv_vget_u8m1(vx4, 1), vl);
        __riscv_vse8(pabyDest2, __riscv_vget_u8m1(vx4, 2), vl);
        __riscv_vse8(pabyDest3, __riscv_vget_u8m1(vx4, 3), vl);

        nIters -= vl;

        pabySrc += 4 * vl;
        pabyDest0 += vl;
        pabyDest1 += vl;
        pabyDest2 += vl;
        pabyDest3 += vl;
    }
}

// TODO: use template metaprogramming to generate these functions
namespace detail
{
template <typename Scalar, int Logm, size_t Nfields, size_t... I>
inline void interleave_impl(const Scalar *__restrict pSrc,
                            Scalar *__restrict pDst, size_t nIters,
                            std::index_sequence<I...>)
{
#pragma GCC unroll 2
    for (size_t i = 0; i < nIters;)
    {
        using helper = gdalrvv::rvv_helper<Scalar, Logm, Nfields>;
        using helper_non_tuple = gdalrvv::rvv_helper<Scalar, Logm>;
        const size_t vl = helper_non_tuple::setvl(nIters - i);

        /*
        v0 = __riscv_vle8_v_u8m1(pSrc + 0 * nIters + i, vl);
        v1 = __riscv_vle8_v_u8m1(pSrc + 1 * nIters + i, vl);
        ...
        v = __riscv_vcreate_v_u8m1xN(v0, v1, ...);
        */

        const auto v = helper::create(
            helper_non_tuple::le(pSrc + (I * nIters) + i, vl)...);

        helper::sseg(pDst + i, v, vl);
        i += vl;
    }
}
}  // namespace detail

template <size_t Nfields, int Logm = 0, typename Scalar>
inline void interleave(const Scalar *__restrict pSrc, Scalar *__restrict pDst,
                       size_t nIters)
{
    return detail::interleave_impl<Scalar, Logm, Nfields>(
        pSrc, pDst, nIters, std::make_index_sequence<Nfields>{});
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

template <typename T, size_t SrcStride, size_t DstStride>
inline void unrolled_copy(T *__restrict dst, const T *__restrict src,
                          ptrdiff_t n)
{
    using rvv = detail::rvv_helper<T, 3>;  // only optimized for T=u8, DstS=3
    using rvv_t = typename rvv::type;
    for (; n > 0;)
    {
        const size_t vl = rvv::setvl(n);

        rvv_t v;
        if constexpr (SrcStride == 1)
            v = rvv::le(src, vl);
        else
            v = rvv::lse(src, SrcStride * sizeof(T), vl);

        if constexpr (DstStride == 1)
            detail::rvv_traits<rvv_t>::se(dst, v, vl);
        else
            detail::rvv_traits<rvv_t>::sse(dst, DstStride * sizeof(T), v, vl);

        src += vl * SrcStride;
        dst += vl * DstStride;
        n -= vl;
    }
}

////////////////////////////////////////////////////////////////////////////////

namespace detail
{

// TODO: impl other types with template metaprogramming?

inline void transpose_2d_byte_8u_8xvl(const uint8_t *__restrict src,
                                      uint8_t *__restrict dst, size_t sstep,
                                      size_t dstep, size_t vl)
{
    const auto v0 = __riscv_vle8_v_u8m1(src + (0 * sstep), vl);
    const auto v1 = __riscv_vle8_v_u8m1(src + (1 * sstep), vl);
    const auto v2 = __riscv_vle8_v_u8m1(src + (2 * sstep), vl);
    const auto v3 = __riscv_vle8_v_u8m1(src + (3 * sstep), vl);
    const auto v4 = __riscv_vle8_v_u8m1(src + (4 * sstep), vl);
    const auto v5 = __riscv_vle8_v_u8m1(src + (5 * sstep), vl);
    const auto v6 = __riscv_vle8_v_u8m1(src + (6 * sstep), vl);
    const auto v7 = __riscv_vle8_v_u8m1(src + (7 * sstep), vl);

    const auto v = __riscv_vcreate_v_u8m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    __riscv_vssseg8e8(dst, dstep, v, vl);
}

}  // namespace detail

inline void transpose_2d_byte(const uint8_t *__restrict src, uint8_t *dst,
                              size_t src_width, size_t src_height)
{
    size_t h = 0;

    for (; h + 8 <= src_height; h += 8)
    {
        const uint8_t *s = src + h * src_width;
        uint8_t *d = dst + h;
        for (size_t w = 0; w < src_width;)
        {
            const size_t vl = __riscv_vsetvl_e8m1(src_width - w);
            detail::transpose_2d_byte_8u_8xvl(s + w, d + w * src_height,
                                              src_width, src_height, vl);
            w += vl;
        }
    }

    for (; h < src_height; ++h)
    {
        const uint8_t *s = src + h * src_width;
        uint8_t *d = dst + h;

        for (size_t w = 0; w < src_width;)
        {
            const size_t vl = __riscv_vsetvl_e8m1(src_width - w);
            auto v = __riscv_vle8_v_u8m1(s + w, vl);
            __riscv_vsse8(d + w * src_height, src_height, v, vl);
            w += vl;
        }
    }
}

}  // namespace rasterio_rvv
