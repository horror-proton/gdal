#pragma once
#define RASTERIO_RVV_HPP_INCLUDED

#include <riscv_vector.h>
#include <limits>

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
