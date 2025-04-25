#pragma once
#define RASTERIO_RVV_HPP_INCLUDED

#include <riscv_vector.h>

#include <type_traits>
#include <limits>

namespace rasterio_rvv
{
namespace detail
{

template <size_t V> struct static_log2;

template <> struct static_log2<1> : std::integral_constant<int, 0>
{
};

template <> struct static_log2<2> : std::integral_constant<int, 1>
{
};

template <> struct static_log2<4> : std::integral_constant<int, 2>
{
};

template <> struct static_log2<8> : std::integral_constant<int, 3>
{
};

template <size_t V> static constexpr int static_log2_v = static_log2<V>::value;

template <size_t S>
using size_to_uint = std::conditional_t<
    (S == 1), uint8_t,
    std::conditional_t<
        (S == 2), uint16_t,
        std::conditional_t<(S == 4), uint32_t,
                           std::conditional_t<(S == 8), uint64_t, void>>>>;

////////////////////////////////////////////////////////////////////////////////

template <typename T, int LOGM> struct rvv_helper;

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
        static type lse(const T *base, ptrdiff_t bstride, size_t vl)           \
        {                                                                      \
            return __riscv_vlse##SIZE##_v_##EEW##EMUL(base, bstride, vl);      \
        }                                                                      \
        template <typename Vsrc> static type reinterpret(Vsrc src)             \
        {                                                                      \
            return __riscv_vreinterpret_##EEW##EMUL(src);                      \
        }                                                                      \
    }

RVV_HELPER(uint8_t, uint8, 8, u8, mf8, -3);
RVV_HELPER(uint8_t, uint8, 8, u8, mf4, -2);
RVV_HELPER(uint8_t, uint8, 8, u8, mf2, -1);
RVV_HELPER(uint8_t, uint8, 8, u8, m1, 0);
RVV_HELPER(uint8_t, uint8, 8, u8, m2, 1);
RVV_HELPER(uint8_t, uint8, 8, u8, m4, 2);
RVV_HELPER(uint8_t, uint8, 8, u8, m8, 3);

RVV_HELPER(int8_t, int8, 8, i8, mf8, -3);
RVV_HELPER(int8_t, int8, 8, i8, mf4, -2);
RVV_HELPER(int8_t, int8, 8, i8, mf2, -1);
RVV_HELPER(int8_t, int8, 8, i8, m1, 0);
RVV_HELPER(int8_t, int8, 8, i8, m2, 1);
RVV_HELPER(int8_t, int8, 8, i8, m4, 2);
RVV_HELPER(int8_t, int8, 8, i8, m8, 3);

RVV_HELPER(uint16_t, uint16, 16, u16, mf4, -2);
RVV_HELPER(uint16_t, uint16, 16, u16, mf2, -1);
RVV_HELPER(uint16_t, uint16, 16, u16, m1, 0);
RVV_HELPER(uint16_t, uint16, 16, u16, m2, 1);
RVV_HELPER(uint16_t, uint16, 16, u16, m4, 2);
RVV_HELPER(uint16_t, uint16, 16, u16, m8, 3);

RVV_HELPER(int16_t, int16, 16, i16, mf4, -2);
RVV_HELPER(int16_t, int16, 16, i16, mf2, -1);
RVV_HELPER(int16_t, int16, 16, i16, m1, 0);
RVV_HELPER(int16_t, int16, 16, i16, m2, 1);
RVV_HELPER(int16_t, int16, 16, i16, m4, 2);
RVV_HELPER(int16_t, int16, 16, i16, m8, 3);

RVV_HELPER(uint32_t, uint32, 32, u32, mf2, -1);
RVV_HELPER(uint32_t, uint32, 32, u32, m1, 0);
RVV_HELPER(uint32_t, uint32, 32, u32, m2, 1);
RVV_HELPER(uint32_t, uint32, 32, u32, m4, 2);
RVV_HELPER(uint32_t, uint32, 32, u32, m8, 3);

RVV_HELPER(int32_t, int32, 32, i32, mf2, -1);
RVV_HELPER(int32_t, int32, 32, i32, m1, 0);
RVV_HELPER(int32_t, int32, 32, i32, m2, 1);
RVV_HELPER(int32_t, int32, 32, i32, m4, 2);
RVV_HELPER(int32_t, int32, 32, i32, m8, 3);

RVV_HELPER(uint64_t, uint64, 64, u64, m1, 0);
RVV_HELPER(uint64_t, uint64, 64, u64, m2, 1);
RVV_HELPER(uint64_t, uint64, 64, u64, m4, 2);
RVV_HELPER(uint64_t, uint64, 64, u64, m8, 3);

RVV_HELPER(int64_t, int64, 64, i64, m1, 0);
RVV_HELPER(int64_t, int64, 64, i64, m2, 1);
RVV_HELPER(int64_t, int64, 64, i64, m4, 2);
RVV_HELPER(int64_t, int64, 64, i64, m8, 3);

RVV_HELPER(float, float32, 32, f32, mf2, -1);
RVV_HELPER(float, float32, 32, f32, m1, 0);
RVV_HELPER(float, float32, 32, f32, m2, 1);
RVV_HELPER(float, float32, 32, f32, m4, 2);
RVV_HELPER(float, float32, 32, f32, m8, 3);

RVV_HELPER(double, float64, 64, f64, m1, 0);
RVV_HELPER(double, float64, 64, f64, m2, 1);
RVV_HELPER(double, float64, 64, f64, m4, 2);
RVV_HELPER(double, float64, 64, f64, m8, 3);

#undef RVV_HELPER

////////////////////////////////////////////////////////////////////////////////

template <typename Vtype, typename = void> struct rvv_scalar
{
    using type = decltype(__riscv_vmv_x(std::declval<Vtype>()));
};

template <typename Vtype>
struct rvv_scalar<Vtype,
                  std::void_t<decltype(__riscv_vfmv_f(std::declval<Vtype>()))>>
{
    using type = decltype(__riscv_vfmv_f(std::declval<Vtype>()));
};

template <typename Vtype> using rvv_scalar_t = typename rvv_scalar<Vtype>::type;

template <typename VBool> struct rvv_bool_r;

#define RVV_BOOL_R(T, LOGV)                                                    \
    template <> struct rvv_bool_r<T>                                           \
    {                                                                          \
        static constexpr int log_value = LOGV;                                 \
        static constexpr size_t value = 1U << static_cast<size_t>(log_value);  \
    }

RVV_BOOL_R(vbool1_t, 0);
RVV_BOOL_R(vbool2_t, 1);
RVV_BOOL_R(vbool4_t, 2);
RVV_BOOL_R(vbool8_t, 3);
RVV_BOOL_R(vbool16_t, 4);
RVV_BOOL_R(vbool32_t, 5);
RVV_BOOL_R(vbool64_t, 6);

#undef RVV_BOOL_R

template <typename Vtype, typename T2>
constexpr auto rvv_eq(Vtype op1, T2 op2, size_t vl)
{
    using scalar = rvv_scalar_t<Vtype>;
    if constexpr (std::is_floating_point_v<scalar>)
        return __riscv_vmfeq(op1, op2, vl);
    else
        return __riscv_vmseq(op1, op2, vl);
}

template <typename Vtype>
static constexpr int rvv_logemul_v =
    static_log2_v<sizeof(rvv_scalar_t<Vtype>)> + 3 -
    rvv_bool_r<decltype(rvv_eq(std::declval<Vtype>(), 0, 0))>::log_value;

static_assert(rvv_logemul_v<vuint8m1_t> == 0);
static_assert(rvv_logemul_v<vuint8mf2_t> == -1);
static_assert(rvv_logemul_v<vfloat32m1_t> == 0);
static_assert(rvv_logemul_v<vfloat32m8_t> == 3);

////////////////////////////////////////////////////////////////////////////////

template <typename T> struct rvv_traits;

#define RVV_TRAITS(T, NAME, SIZE, EMUL, LOGM)                                  \
    template <> struct rvv_traits<v##NAME##EMUL##_t>                           \
    {                                                                          \
        using type = v##NAME##EMUL##_t;                                        \
        using scalar = rvv_scalar_t<type>;                                     \
        static constexpr int logm = rvv_logemul_v<type>;                       \
        static void se(scalar *base, type value, size_t vl)                    \
        {                                                                      \
            return __riscv_vse##SIZE(base, value, vl);                         \
        }                                                                      \
        static void sse(scalar *base, ptrdiff_t bstride, type value,           \
                        size_t vl)                                             \
        {                                                                      \
            return __riscv_vsse##SIZE(base, bstride, value, vl);               \
        }                                                                      \
    }

RVV_TRAITS(uint8_t, uint8, 8, mf8, -3);
RVV_TRAITS(uint8_t, uint8, 8, mf4, -2);
RVV_TRAITS(uint8_t, uint8, 8, mf2, -1);
RVV_TRAITS(uint8_t, uint8, 8, m1, 0);
RVV_TRAITS(uint8_t, uint8, 8, m2, 1);
RVV_TRAITS(uint8_t, uint8, 8, m4, 2);
RVV_TRAITS(uint8_t, uint8, 8, m8, 3);

RVV_TRAITS(uint16_t, uint16, 16, mf4, -2);
RVV_TRAITS(uint16_t, uint16, 16, mf2, -1);
RVV_TRAITS(uint16_t, uint16, 16, m1, 0);
RVV_TRAITS(uint16_t, uint16, 16, m2, 1);
RVV_TRAITS(uint16_t, uint16, 16, m4, 2);
RVV_TRAITS(uint16_t, uint16, 16, m8, 3);

RVV_TRAITS(uint32_t, uint32, 32, mf2, -1);
RVV_TRAITS(uint32_t, uint32, 32, m1, 0);
RVV_TRAITS(uint32_t, uint32, 32, m2, 1);
RVV_TRAITS(uint32_t, uint32, 32, m4, 2);
RVV_TRAITS(uint32_t, uint32, 32, m8, 3);

RVV_TRAITS(uint64_t, uint64, 64, m1, 0);
RVV_TRAITS(uint64_t, uint64, 64, m2, 1);
RVV_TRAITS(uint64_t, uint64, 64, m4, 2);
RVV_TRAITS(uint64_t, uint64, 64, m8, 3);

RVV_TRAITS(float, float32, 32, mf2, -1);
RVV_TRAITS(float, float32, 32, m1, 0);
RVV_TRAITS(float, float32, 32, m2, 1);
RVV_TRAITS(float, float32, 32, m4, 2);
RVV_TRAITS(float, float32, 32, m8, 3);

RVV_TRAITS(double, float64, 64, m1, 0);
RVV_TRAITS(double, float64, 64, m2, 1);
RVV_TRAITS(double, float64, 64, m4, 2);
RVV_TRAITS(double, float64, 64, m8, 3);

RVV_TRAITS(int8_t, int8, 8, mf8, -3);
RVV_TRAITS(int8_t, int8, 8, mf4, -2);
RVV_TRAITS(int8_t, int8, 8, mf2, -1);
RVV_TRAITS(int8_t, int8, 8, m1, 0);
RVV_TRAITS(int8_t, int8, 8, m2, 1);
RVV_TRAITS(int8_t, int8, 8, m4, 2);
RVV_TRAITS(int8_t, int8, 8, m8, 3);

RVV_TRAITS(int16_t, int16, 16, mf4, -2);
RVV_TRAITS(int16_t, int16, 16, mf2, -1);
RVV_TRAITS(int16_t, int16, 16, m1, 0);
RVV_TRAITS(int16_t, int16, 16, m2, 1);
RVV_TRAITS(int16_t, int16, 16, m4, 2);
RVV_TRAITS(int16_t, int16, 16, m8, 3);

RVV_TRAITS(int32_t, int32, 32, mf2, -1);
RVV_TRAITS(int32_t, int32, 32, m1, 0);
RVV_TRAITS(int32_t, int32, 32, m2, 1);
RVV_TRAITS(int32_t, int32, 32, m4, 2);
RVV_TRAITS(int32_t, int32, 32, m8, 3);

RVV_TRAITS(int64_t, int64, 64, m1, 0);
RVV_TRAITS(int64_t, int64, 64, m2, 1);
RVV_TRAITS(int64_t, int64, 64, m4, 2);
RVV_TRAITS(int64_t, int64, 64, m8, 3);

#undef RVV_TRAITS

////////////////////////////////////////////////////////////////////////////////

template <size_t Vf> struct vext_impl;

#define VEXT_IMPL(VF)                                                          \
    template <> struct vext_impl<VF>                                           \
    {                                                                          \
        template <typename InT> static auto apply(InT in, size_t vl)           \
        {                                                                      \
            using scalar_in = typename rvv_traits<InT>::scalar;                \
            if constexpr (std::is_signed_v<scalar_in>)                         \
                return __riscv_vsext_vf##VF(in, vl);                           \
            else                                                               \
                return __riscv_vzext_vf##VF(in, vl);                           \
        }                                                                      \
    }

VEXT_IMPL(2);
VEXT_IMPL(4);
VEXT_IMPL(8);

template <size_t DstScalarSize, typename InT>
inline auto vext(InT in, size_t vl)
{
    using scalar_in = rvv_scalar_t<InT>;
    return vext_impl<DstScalarSize / sizeof(scalar_in)>::apply(in, vl);
}

#undef VEXT_IMPL

////////////////////////////////////////////////////////////////////////////////

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
