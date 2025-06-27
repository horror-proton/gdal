#pragma once

#include <cstddef>
#include <riscv_vector.h>
#include <type_traits>

namespace gdalrvv
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

template <typename T, int LOGM, size_t NFIELD = 1> struct rvv_helper;

#define RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, NFIELD)               \
    template <> struct rvv_helper<T, LOGM, NFIELD>                             \
    {                                                                          \
        using type = v##NAME##EMUL##x##NFIELD##_t;                             \
        using non_tuple_type = v##NAME##EMUL##_t;                              \
        static constexpr size_t nfield = NFIELD;                               \
        template <typename... Args> static type create(Args... args)           \
        {                                                                      \
            return __riscv_vcreate_v_##EEW##EMUL##x##NFIELD(args...);          \
        }                                                                      \
        static void sseg(T *base, type v, size_t vl)                           \
        {                                                                      \
            __riscv_vsseg##NFIELD##e##SIZE(base, v, vl);                       \
        }                                                                      \
    }

#define RVV_HELPER_TUPLE_m8(T, NAME, SIZE, EEW, EMUL, LOGM) static_assert(true)

#define RVV_HELPER_TUPLE_m4(T, NAME, SIZE, EEW, EMUL, LOGM)                    \
    RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, 2)

#define RVV_HELPER_TUPLE_m2(T, NAME, SIZE, EEW, EMUL, LOGM)                    \
    RVV_HELPER_TUPLE_m4(T, NAME, SIZE, EEW, EMUL, LOGM);                       \
    RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, 3);                       \
    RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, 4)

#define RVV_HELPER_TUPLE_m1(T, NAME, SIZE, EEW, EMUL, LOGM)                    \
    RVV_HELPER_TUPLE_m2(T, NAME, SIZE, EEW, EMUL, LOGM);                       \
    RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, 5);                       \
    RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, 6);                       \
    RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, 7);                       \
    RVV_HELPER_TUPLE(T, NAME, SIZE, EEW, EMUL, LOGM, 8)

#define RVV_HELPER_TUPLE_mf2 RVV_HELPER_TUPLE_m1
#define RVV_HELPER_TUPLE_mf4 RVV_HELPER_TUPLE_m1
#define RVV_HELPER_TUPLE_mf8 RVV_HELPER_TUPLE_m1

#define RVV_HELPER(T, NAME, SIZE, EEW, EMUL, LOGM)                             \
    template <> struct rvv_helper<T, LOGM, 1>                                  \
    {                                                                          \
        using type = v##NAME##EMUL##_t;                                        \
        static size_t setvl(size_t n)                                          \
        {                                                                      \
            return __riscv_vsetvl_e##SIZE##EMUL(n);                            \
        }                                                                      \
        static size_t setvlmax()                                               \
        {                                                                      \
            return __riscv_vsetvlmax_e##SIZE##EMUL();                          \
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
    };                                                                         \
    RVV_HELPER_TUPLE_##EMUL(T, NAME, SIZE, EEW, EMUL, LOGM)

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

template <int Logm = 0> inline auto load_as_double(const double *ptr, size_t vl)
{
    return rvv_helper<double, Logm>::le(ptr, vl);
}

template <int Logm = 0> inline auto load_as_double(const float *ptr, size_t vl)
{
    auto a = rvv_helper<float, Logm - 1>::le(ptr, vl);
    auto result = __riscv_vfwcvt_f(a, vl);
    return result;
}

template <int Logm = 0>
inline auto load_as_double(const uint8_t *ptr, size_t vl)
{
    auto a = rvv_helper<uint8_t, Logm - 3>::le(ptr, vl);
    // auto b = __riscv_vwcvtu_x(__riscv_vwcvtu_x(a, vl), vl);
    // auto result = __riscv_vfwcvt_f(b, vl);
    auto b = __riscv_vzext_vf8(a, vl);
    auto result = __riscv_vfcvt_f(b, vl);
    return result;
}

 // TODO: use vzext/vsext?
template <int Logm = 0> inline auto load_as_double(const int8_t *ptr, size_t vl)
{
    auto a = rvv_helper<int8_t, Logm - 3>::le(ptr, vl);
    auto b = __riscv_vwcvt_x(__riscv_vwcvt_x(a, vl), vl);
    auto result = __riscv_vfwcvt_f(b, vl);
    return result;
}

template <int Logm = 0>
inline auto load_as_double(const uint16_t *ptr, size_t vl)
{
    auto a = rvv_helper<uint16_t, Logm - 2>::le(ptr, vl);
    auto b = __riscv_vwcvtu_x(a, vl);
    auto result = __riscv_vfwcvt_f(b, vl);
    return result;
}

template <int Logm = 0>
inline auto load_as_double(const int16_t *ptr, size_t vl)
{
    auto a = rvv_helper<int16_t, Logm - 2>::le(ptr, vl);
    auto b = __riscv_vwcvt_x(a, vl);
    auto result = __riscv_vfwcvt_f(b, vl);
    return result;
}

////////////////////////////////////////////////////////////////////////////////

// TODO: use vnclipu

template <typename V>
inline void store_from_double(uint8_t *base, V value, size_t vl)
{
    auto a = __riscv_vfncvt_xu(value, __RISCV_FRM_RMM, vl);
    auto b = __riscv_vncvt_x(__riscv_vncvt_x(a, vl), vl);
    rvv_traits<decltype(b)>::se(base, b, vl);
}

template <typename V>
inline void store_from_double(uint16_t *base, V value, size_t vl)
{
    auto a = __riscv_vfncvt_xu(value, __RISCV_FRM_RMM, vl);
    auto b = __riscv_vncvt_x(a, vl);
    rvv_traits<decltype(b)>::se(base, b, vl);
}

}  // namespace gdalrvv
