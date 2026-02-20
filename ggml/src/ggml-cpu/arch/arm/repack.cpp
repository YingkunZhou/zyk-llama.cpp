#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

#if defined(__aarch64__) && defined(__ARM_NEON) && (defined(__ARM_FEATURE_MATMUL_INT8) || defined(__ARM_FEATURE_DOTPROD))
#define N32B 4
static inline void decode_q4_Kx8_scales_mins(const uint8_t * scales_in,
                                             int16x8_t *     out_mins,
                                             int8_t *        out_scales) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    constexpr uint32_t kmask3 = 0x03030303;
    constexpr uint8_t  scales_size = 12;

    uint32_t sm[3];
    memcpy(sm, scales_in, scales_size);

    const uint32_t   mins_0_3 = sm[1] & kmask1;
    const uint32_t   mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);
    const uint32x2_t mins_u32 = { mins_0_3, mins_4_7 };

    *out_mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins_u32)));

    uint32_t scales_u32[2];
    scales_u32[0] = sm[0] & kmask1;
    scales_u32[1] = (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4);
    memcpy(out_scales, scales_u32, 8);
}

#include "iqk_common.h"

#if USE_IQK
struct Q4_0_R8_Dequantizer {
    Q4_0_R8_Dequantizer(const void * vx, size_t bx) : cx((const char *)vx), bx(bx) {}
    inline void new_row(int ix) { iq4 = (const block_q4_0x8 *)(cx + ix*bx); }
    inline float32x4x2_t prepare(int ib4, int k, int8x16_t * qx) const {
        auto scales16 = vld1q_f16((const float16_t *)iq4[4*ib4+k].d);
        float32x4x2_t scales = { vcvt_f32_f16(vget_low_f16(scales16)), vcvt_f32_f16(vget_high_f16(scales16)) };
        for (int j = 0; j < 4; ++j) {
            auto bits = vld1q_s8_x2(iq4[4*ib4+k].qs + 32*j);
            qx[2*j+0] = bits.val[0] << 4;
            qx[2*j+1] = bits.val[0] & 0xf0U;
            qx[2*j+8] = bits.val[1] << 4;
            qx[2*j+9] = bits.val[1] & 0xf0U;
        }
        return scales;
    }

    const char * cx;
    const size_t bx;
    const block_q4_0x8 * iq4;
};

static IQK_ALWAYS_INLINE int32x4_t interleaved_dotq(const int8x16_t * qx, const int8x16x2_t& y) {
    auto sumi = vdupq_n_s32(0);
    sumi = vdotq_laneq_s32(sumi, qx[0], y.val[0], 0);
    sumi = vdotq_laneq_s32(sumi, qx[1], y.val[1], 0);
    sumi = vdotq_laneq_s32(sumi, qx[2], y.val[0], 1);
    sumi = vdotq_laneq_s32(sumi, qx[3], y.val[1], 1);
    sumi = vdotq_laneq_s32(sumi, qx[4], y.val[0], 2);
    sumi = vdotq_laneq_s32(sumi, qx[5], y.val[1], 2);
    sumi = vdotq_laneq_s32(sumi, qx[6], y.val[0], 3);
    sumi = vdotq_laneq_s32(sumi, qx[7], y.val[1], 3);
    return sumi;
}

template <int nrc_y>
void mul_mat_qx_r8_q8_0(int n, const void * vx, size_t bx, const DataInfo* info, int nrc_x) {
    GGML_ASSERT(nrc_x%8 == 0);
    const block_q8_0_x4 * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) q8[iy] = (const block_q8_0_x4 *)info->src1_row(iy);
    Q4_0_R8_Dequantizer deq(vx, bx);
    const int nb = n / QK4_NL;
    int8x16_t qx[16];
    float32x4_t d8[nrc_y];
    const int8_t* q8qs[nrc_y];
    for (int ix = 0; ix < nrc_x; ix += 8) {
        float32x4_t acc[2*nrc_y] = {};
        deq.new_row(ix);
        for (int ib4 = 0; ib4 < nb/4; ++ib4) {
            for (int iy = 0; iy < nrc_y; ++iy) {
                d8[iy] = vcvt_f32_f16(vld1_f16((const float16_t *)(q8[iy][ib4].d)));
                q8qs[iy] = q8[iy][ib4].qs;
            }

            auto scales = deq.prepare(ib4, 0, qx);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = vld1q_s8_x2(q8qs[iy]);
                auto sumi0 = interleaved_dotq(qx+0, y);
                auto sumi1 = interleaved_dotq(qx+8, y);
                float32x4_t scale0 = vmulq_laneq_f32(scales.val[0], d8[iy], 0);
                float32x4_t scale1 = vmulq_laneq_f32(scales.val[1], d8[iy], 0);
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], scale0, vcvtq_n_f32_s32(sumi0, 4));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], scale1, vcvtq_n_f32_s32(sumi1, 4));
            }

            scales = deq.prepare(ib4, 1, qx);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = vld1q_s8_x2(q8qs[iy]+32);
                auto sumi0 = interleaved_dotq(qx+0, y);
                auto sumi1 = interleaved_dotq(qx+8, y);
                float32x4_t scale0 = vmulq_laneq_f32(scales.val[0], d8[iy], 1);
                float32x4_t scale1 = vmulq_laneq_f32(scales.val[1], d8[iy], 1);
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], scale0, vcvtq_n_f32_s32(sumi0, 4));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], scale1, vcvtq_n_f32_s32(sumi1, 4));
            }

            scales = deq.prepare(ib4, 2, qx);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = vld1q_s8_x2(q8qs[iy]+64);
                auto sumi0 = interleaved_dotq(qx+0, y);
                auto sumi1 = interleaved_dotq(qx+8, y);
                float32x4_t scale0 = vmulq_laneq_f32(scales.val[0], d8[iy], 2);
                float32x4_t scale1 = vmulq_laneq_f32(scales.val[1], d8[iy], 2);
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], scale0, vcvtq_n_f32_s32(sumi0, 4));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], scale1, vcvtq_n_f32_s32(sumi1, 4));
            }

            scales = deq.prepare(ib4, 3, qx);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto y = vld1q_s8_x2(q8qs[iy]+96);
                auto sumi0 = interleaved_dotq(qx+0, y);
                auto sumi1 = interleaved_dotq(qx+8, y);
                float32x4_t scale0 = vmulq_laneq_f32(scales.val[0], d8[iy], 3);
                float32x4_t scale1 = vmulq_laneq_f32(scales.val[1], d8[iy], 3);
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], scale0, vcvtq_n_f32_s32(sumi0, 4));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], scale1, vcvtq_n_f32_s32(sumi1, 4));
            }
        }
        for (int ib = 4*(nb/4); ib < nb; ++ib) {
            auto scales = deq.prepare(ib/4, ib%4, qx);
            for (int iy = 0; iy < nrc_y; ++iy) {
                auto qy = (const block_q8_0 *)q8[iy];
                auto y = vld1q_s8_x2(qy[ib].qs);
                auto sumi1 = interleaved_dotq(qx+0, y);
                auto sumi2 = interleaved_dotq(qx+8, y);
                auto dy = vdupq_n_f32(GGML_FP16_TO_FP32(qy[ib].d));
                float32x4_t scale0 = vmulq_f32(scales.val[0], dy);
                float32x4_t scale1 = vmulq_f32(scales.val[1], dy);
                acc[2*iy+0] = vfmaq_f32(acc[2*iy+0], scale0, vcvtq_n_f32_s32(sumi1, 4));
                acc[2*iy+1] = vfmaq_f32(acc[2*iy+1], scale1, vcvtq_n_f32_s32(sumi2, 4));
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            info->store(ix+0, iy, acc[2*iy+0]);
            info->store(ix+4, iy, acc[2*iy+1]);
        }
    }
}

std::array<mul_mat_t, IQK_MAX_NY-1> q4_0_funcs = {
    mul_mat_qx_r8_q8_0<1>,
    mul_mat_qx_r8_q8_0<2>,
    mul_mat_qx_r8_q8_0<3>,
    mul_mat_qx_r8_q8_0<4>,
    mul_mat_qx_r8_q8_0<5>,
    mul_mat_qx_r8_q8_0<6>,
    mul_mat_qx_r8_q8_0<7>};
#endif

#if USE_ZYK
static float32x4_t * thread_local_work_buffer(size_t need_elems) {
    thread_local std::vector<float32x4_t> buffer;

    if (need_elems == 0) {
        return nullptr;
    }

    if (buffer.size() < need_elems) {
        buffer.resize(need_elems);
    }

    memset(buffer.data(), 0, need_elems * sizeof(float32x4_t));

    return buffer.data();
}

/** generated by GLM-5
 * 将一个 uint32_t (包含4个 uint8_t) 转换为 float32x4_t
 * 逻辑对应 ggml_e8m0_to_fp32_half
 */
static inline float32x4_t ggml_e8m0_to_fp32_half_vec(uint32_t packed_x) {
    // 1. 数据加载与扩展
    // 将 uint32_t 加载到 64bit NEON 寄存器 (uint8x8_t)
    uint8x8_t v_u8 = vcreate_u8(packed_x);

    // 将 4 个 uint8 扩展为 4 个 uint16 (取低64位)
    uint16x4_t v_u16 = vget_low_u16(vmovl_u8(v_u8));

    // 将 4 个 uint16 扩展为 4 个 uint32
    uint32x4_t v_x = vmovl_u16(v_u16);

    // 2. 计算 Normal 路径 (x >= 2): (x - 1) << 23
    uint32x4_t v_one = vdupq_n_u32(1);
    uint32x4_t v_normal_bits = vshlq_n_u32(vsubq_u32(v_x, v_one), 23);

    // 3. 计算 Denormal 路径 (x < 2): 0x00200000 << x
    // 注意：vshlq_u32 是向量左移，移位量由第二个参数的低字节决定
    // 对于 x >= 32 的情况，结果为 0，但这在我们的逻辑中被掩码屏蔽了，所以安全
    uint32x4_t v_denorm_base = vdupq_n_u32(0x00200000);
    int32x4_t v_shift_amount = vreinterpretq_s32_u32(v_x); // 语义上是向量移位，移位量需要视为有符号数
    uint32x4_t v_denormal_bits = vshlq_u32(v_denorm_base, v_shift_amount);

    // 4. 条件选择 (x < 2)
    // 比较结果是一个掩码：若 x < 2，对应位全为 1，否则全为 0
    uint32x4_t v_mask = vcltq_u32(v_x, vdupq_n_u32(2));

    // vbslq_u32: Bitwise Select. 如果 mask 位为1选 denormal，否则选 normal
    uint32x4_t v_res_bits = vbslq_u32(v_mask, v_denormal_bits, v_normal_bits);

    // 5. 重新解释为 float
    return vreinterpretq_f32_u32(v_res_bits);
}

struct ZykQ4_0_T {
    // nx: the total rows of weight matrix (aka, colunm size)
    // ix: the start row of current process splited weight matrix
    ZykQ4_0_T(const void * vx, size_t nb, size_t nx, size_t ix) {
        q4qs = (const int8_t *)vx + sizeof(float16_t)*nb*nx + 16*nb*ix; // every block contains 16 bytes qs
        dptr = (const float16_t *)vx + nb*ix;
    }

    template <int nrc_y>
    inline void compute_block(const int8_t ** q8, float32x4_t * q8_ds, float32x4_t * accd, int nrc_x) {
        int8x16x2_t q8qs[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) {
            q8qs[iy] = vld1q_s8_x2(q8[iy]);
            q8[iy] += 0x20;
        }
        int8x16_t b0[4];
        // #pragma clang loop unroll_count(2)
        for (int base = 0; base < nrc_y * nrc_x; base += nrc_y) {
            float32x4_t q4_d = vcvt_f32_f16(vld1_f16(dptr));
            dptr += 4;
            int8x16x4_t q4 = vld1q_s8_x4(q4qs);
            q4qs += 0x40;

            b0[0] = q4.val[0] << 4;
            q4.val[0] = q4.val[0] & 0xf0U;
            b0[1] = q4.val[1] << 4;
            q4.val[1] = q4.val[1] & 0xf0U;
            b0[2] = q4.val[2] << 4;
            q4.val[2] = q4.val[2] & 0xf0U;
            b0[3] = q4.val[3] << 4;
            q4.val[3] = q4.val[3] & 0xf0U;
            #pragma clang loop unroll(full)
            for (int iy = 0; iy < nrc_y; ++iy) {
                int32x4_t accm = vdupq_n_s32(0);
                float32x4_t scale = vmulq_f32(q4_d, q8_ds[iy]);
                float32x4_t accd_val = accd[base + iy];
                accm = vdotq_laneq_s32(accm, b0[0], q8qs[iy].val[0], 0);
                accm = vdotq_laneq_s32(accm, b0[1], q8qs[iy].val[0], 1);
                accm = vdotq_laneq_s32(accm, b0[2], q8qs[iy].val[0], 2);
                accm = vdotq_laneq_s32(accm, b0[3], q8qs[iy].val[0], 3);
                accm = vdotq_laneq_s32(accm, q4.val[0], q8qs[iy].val[1], 0);
                accm = vdotq_laneq_s32(accm, q4.val[1], q8qs[iy].val[1], 1);
                accm = vdotq_laneq_s32(accm, q4.val[2], q8qs[iy].val[1], 2);
                accm = vdotq_laneq_s32(accm, q4.val[3], q8qs[iy].val[1], 3);
                accd[base + iy] = vfmaq_f32(accd_val, vcvtq_n_f32_s32(accm, 4), scale);
            }
        }
    }

    const int8_t * q4qs;
    const float16_t * dptr;
};

#if USE_Q4_0_OPT
struct ZykQ4_0_T567 {
    // nx: the total rows of weight matrix (aka, colunm size)
    // ix: the start row of current process splited weight matrix
    ZykQ4_0_T567(const void * vx, size_t nb, size_t nx, size_t ix) {
        q4qs = (const int8_t *)vx + sizeof(float16_t)*nb*nx + 16*nb*ix; // every block contains 16 bytes qs
        dptr = (const float16_t *)vx + nb*ix;
    }

    template <int nrc_y>
    inline void compute_block(const int8_t ** q8, float32x4_t * q8_ds, float32x4_t * accd, int nrc_x) {
        constexpr int nrc_yy = nrc_y - 4;
        // + 1 register
        float32x4_t q8_d = vcvt_f32_f16(vld1_f16((const float16_t *)(q8[nrc_yy])));
        // + 8 registers
        int8x16x4_t q8qs0 = vld1q_s8_x4(q8[nrc_yy] + 0x8);
        int8x16x4_t q8qs1 = vld1q_s8_x4(q8[nrc_yy] + 0x48);
        q8[nrc_yy] += sizeof(block_q8_0x4);
        int8x16x2_t q8qs[nrc_yy];
        for (int iy = 0; iy < nrc_yy; ++iy) {
            q8qs[iy] = vld1q_s8_x2(q8[iy]);
            q8[iy] += 0x20;
        }
        int32x4_t accm[nrc_yy];
        float32x4_t scale[nrc_yy]; // TODO

        for (int base = 0; base < nrc_y * nrc_x; base += nrc_y) {
            // + 4 registers
            int8x16x4_t q4 = vld1q_s8_x4(q4qs);
            q4qs += 0x40;
            // + 1 register
            int8x16_t b0 = q4.val[0] << 4;
            q4.val[0] = q4.val[0] & 0xf0U;
            // + 4 registers
            int32x4_t accm0 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 0);
            int32x4_t accm1 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 1);
            int32x4_t accm2 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 2);
            int32x4_t accm3 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs[iy].val[0], 0);
            b0 = q4.val[1] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[0], q8qs1.val[0], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[0], q8qs1.val[0], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[0], q8qs1.val[0], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[0], q8qs1.val[0], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(accm[iy], q4.val[0], q8qs[iy].val[1], 0);
            q4.val[1] = q4.val[1] & 0xf0U;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[1], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[1], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[1], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[1], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(accm[iy], b0, q8qs[iy].val[0], 1);
            b0 = q4.val[2] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[1], q8qs1.val[1], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[1], q8qs1.val[1], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[1], q8qs1.val[1], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[1], q8qs1.val[1], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(accm[iy], q4.val[1], q8qs[iy].val[1], 1);
            q4.val[2] = q4.val[2] & 0xf0U;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[2], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[2], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[2], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[2], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(accm[iy], b0, q8qs[iy].val[0], 2);
            b0 = q4.val[3] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[2], q8qs1.val[2], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[2], q8qs1.val[2], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[2], q8qs1.val[2], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[2], q8qs1.val[2], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(accm[iy], q4.val[2], q8qs[iy].val[1], 2);
            q4.val[3] = q4.val[3] & 0xf0U;
            // - 3 regs, + 1 reg
            float32x4_t q4_d = vcvt_f32_f16(vld1_f16(dptr));
            dptr += 4;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[3], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[3], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[3], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[3], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(accm[iy], b0, q8qs[iy].val[0], 3);
            // -1 reg, + 4 regs
            float32x4_t scale0 = vmulq_laneq_f32(q4_d, q8_d, 0);
            float32x4_t scale1 = vmulq_laneq_f32(q4_d, q8_d, 1);
            float32x4_t scale2 = vmulq_laneq_f32(q4_d, q8_d, 2);
            float32x4_t scale3 = vmulq_laneq_f32(q4_d, q8_d, 3);
            for (int iy = 0; iy < nrc_yy; ++iy) scale[iy] = vmulq_f32(q4_d, q8_ds[iy]);
            accm0 = vdotq_laneq_s32(accm0, q4.val[3], q8qs1.val[3], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[3], q8qs1.val[3], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[3], q8qs1.val[3], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[3], q8qs1.val[3], 3);
            for (int iy = 0; iy < nrc_yy; ++iy) accm[iy] = vdotq_laneq_s32(accm[iy], q4.val[3], q8qs[iy].val[1], 3);
            // -1 reg
            accd[base + 0] = vfmaq_f32(accd[base + 0], vcvtq_n_f32_s32(accm0, 4), scale0);
            accd[base + 1] = vfmaq_f32(accd[base + 1], vcvtq_n_f32_s32(accm1, 4), scale1);
            accd[base + 2] = vfmaq_f32(accd[base + 2], vcvtq_n_f32_s32(accm2, 4), scale2);
            accd[base + 3] = vfmaq_f32(accd[base + 3], vcvtq_n_f32_s32(accm3, 4), scale3);
            for (int iy = 0; iy < nrc_yy; ++iy) accd[base + iy + 4] = vfmaq_f32(accd[base + iy + 4], vcvtq_n_f32_s32(accm[iy], 4), scale[iy]);
        }
    }

    const int8_t * q4qs;
    const float16_t * dptr;
};

template <int nrc_y>
static void mul_mat_q4_t567_q8_0(int n, const void * vx, size_t ix, const DataInfo* info, int nrc_x) {
    GGML_ASSERT(n % QK8_0 == 0);
    constexpr int nrc_yy = nrc_y - 4;
    const size_t nb = n / QK8_0;
    const int8_t * q8[nrc_yy+1];
    for (int iy = 0; iy < nrc_yy; ++iy) {
        q8[iy] = (const int8_t *)(info->src1_row(iy+4));
    }
    q8[nrc_yy] = (const int8_t *)(info->src1_row(0));

    float d8[4*nrc_yy];
    float32x4_t q8_ds[nrc_yy];
    ZykQ4_0_T567 deq(vx, nb, info->bs, ix);
    // Initialize accumulation vector to zero
    float32x4_t * accd = thread_local_work_buffer(nrc_y * nrc_x);
    // #pragma clang loop unroll_count(4)
    for (size_t ib4 = 0; ib4 < nb/4; ++ib4) {
        for (int iy = 0; iy < nrc_yy; ++iy) {
            vst1q_f32(d8+4*iy, vcvt_f32_f16(vld1_f16((const float16_t *)(q8[iy]))));
            q8[iy] += 8;
        }
        for (int k = 0; k < 4; ++k) {
            for (int iy = 0; iy < nrc_yy; ++iy) q8_ds[iy] = vdupq_n_f32(d8[4*iy+k]);
            deq.compute_block<nrc_y>(q8, q8_ds, accd, nrc_x);
        }
    }
    for (size_t ib = 4*(nb/4); ib < nb; ++ib) {
        for (int iy = 0; iy < nrc_yy; ++iy) {
            q8_ds[iy] = vdupq_n_f32(GGML_FP16_TO_FP32(*(const ggml_half *)(q8[iy])));
            q8[iy] += 2;
        }
        deq.compute_block<nrc_y>(q8, q8_ds, accd, nrc_x);
    }
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int k = 0; k < nrc_x; ++k) {
            info->store(k * N32B, iy, accd[iy + k * nrc_y]);
        }
    }
}

static void mul_mat_q4_t4_q8_0(int n, const void * vx, size_t ix, const DataInfo* info, int nrc_x) {
    GGML_ASSERT(n % QK8_0 == 0);
    const size_t nb = n / QK8_0;
    constexpr int nrc_y = 4;
    const int8_t * q8[nrc_y/4];
    for (int iy = 0; iy < nrc_y/4; ++iy) {
        q8[iy] = (const int8_t *)(info->src1_row(iy*4));
    }
    size_t nx = info->bs;
    // every block contains 16 bytes qs
    const int8_t * q4qs = (const int8_t *)vx + sizeof(float16_t)*nb*nx + 16*nb*ix;
    const float16_t * dptr = (const float16_t *)vx + nb*ix;
    // Initialize accumulation vector to zero
    float32x4_t * accd = thread_local_work_buffer(nrc_y * nrc_x);
    // #pragma clang loop unroll_count(4)
    for (size_t i = 0; i < nb; ++i) {
        // + 1 register
        float32x4_t q8_d = vcvt_f32_f16(vld1_f16((const float16_t *)(q8[0])));
        // + 8 registers
        int8x16x4_t q8qs0 = vld1q_s8_x4(q8[0] + 0x8);
        int8x16x4_t q8qs1 = vld1q_s8_x4(q8[0] + 0x48);
        q8[0] += sizeof(block_q8_0x4);

        // #pragma clang loop unroll_count(2)
        for (int base = 0; base < nrc_y * nrc_x; base += nrc_y) {
            // + 4 registers
            int8x16x4_t q4 = vld1q_s8_x4(q4qs);
            q4qs += 0x40;
            // + 1 register
            int8x16_t b0 = q4.val[0] << 4;
            q4.val[0] = q4.val[0] & 0xf0U;
            // + 4 registers
            int32x4_t accm0 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 0);
            int32x4_t accm1 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 1);
            int32x4_t accm2 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 2);
            int32x4_t accm3 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 3);
            b0 = q4.val[1] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[0], q8qs1.val[0], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[0], q8qs1.val[0], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[0], q8qs1.val[0], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[0], q8qs1.val[0], 3);
            q4.val[1] = q4.val[1] & 0xf0U;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[1], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[1], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[1], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[1], 3);
            b0 = q4.val[2] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[1], q8qs1.val[1], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[1], q8qs1.val[1], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[1], q8qs1.val[1], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[1], q8qs1.val[1], 3);
            q4.val[2] = q4.val[2] & 0xf0U;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[2], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[2], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[2], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[2], 3);
            b0 = q4.val[3] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[2], q8qs1.val[2], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[2], q8qs1.val[2], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[2], q8qs1.val[2], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[2], q8qs1.val[2], 3);
            q4.val[3] = q4.val[3] & 0xf0U;
            // - 3 regs, + 1 reg
            float32x4_t q4_d = vcvt_f32_f16(vld1_f16(dptr));
            dptr += 4;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[3], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[3], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[3], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[3], 3);
            // -1 reg, + 4 regs
            float32x4_t scale0 = vmulq_laneq_f32(q4_d, q8_d, 0);
            float32x4_t scale1 = vmulq_laneq_f32(q4_d, q8_d, 1);
            float32x4_t scale2 = vmulq_laneq_f32(q4_d, q8_d, 2);
            float32x4_t scale3 = vmulq_laneq_f32(q4_d, q8_d, 3);
            accm0 = vdotq_laneq_s32(accm0, q4.val[3], q8qs1.val[3], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[3], q8qs1.val[3], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[3], q8qs1.val[3], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[3], q8qs1.val[3], 3);
            // -1 reg
            accd[base + 0] = vfmaq_f32(accd[base + 0], vcvtq_n_f32_s32(accm0, 4), scale0);
            accd[base + 1] = vfmaq_f32(accd[base + 1], vcvtq_n_f32_s32(accm1, 4), scale1);
            accd[base + 2] = vfmaq_f32(accd[base + 2], vcvtq_n_f32_s32(accm2, 4), scale2);
            accd[base + 3] = vfmaq_f32(accd[base + 3], vcvtq_n_f32_s32(accm3, 4), scale3);
        }
    }
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int k = 0; k < nrc_x; ++k) {
            info->store(k * N32B, iy, accd[iy + k * nrc_y]);
        }
    }
}
#endif // USE_Q4_0_OPT

static void mul_mat_q4_t8_q8_0(int n, const void * vx, size_t ix, const DataInfo* info, int nrc_x) {
    GGML_ASSERT(n % QK8_0 == 0);
    const size_t nb = n / QK8_0;
    constexpr int nrc_y = 8;
    const int8_t * q8[nrc_y/4];
    for (int iy = 0; iy < nrc_y/4; ++iy) {
        q8[iy] = (const int8_t *)(info->src1_row(iy*4));
    }
    size_t nx = info->bs;
    // every block contains 16 bytes qs
    const int8_t * q4qs = (const int8_t *)vx + sizeof(float16_t)*nb*nx + 16*nb*ix;
    const float16_t * dptr = (const float16_t *)vx + nb*ix;
    // Initialize accumulation vector to zero
    float32x4_t * accd = thread_local_work_buffer(nrc_y * nrc_x);
    // #pragma clang loop unroll_count(4)
    for (size_t i = 0; i < nb; ++i) {
        // + 2 register
        float32x4_t q8_d0 = vcvt_f32_f16(vld1_f16((const float16_t *)(q8[0])));
        float32x4_t q8_d1 = vcvt_f32_f16(vld1_f16((const float16_t *)(q8[1])));
        // + 16 registers
        int8x16x4_t q8qs0 = vld1q_s8_x4(q8[0] + 0x8);
        int8x16x4_t q8qs1 = vld1q_s8_x4(q8[0] + 0x48);
        int8x16x4_t q8qs4 = vld1q_s8_x4(q8[1] + 0x8);
        int8x16x4_t q8qs5 = vld1q_s8_x4(q8[1] + 0x48);
        q8[0] += sizeof(block_q8_0x4);
        q8[1] += sizeof(block_q8_0x4);

        for (int base = 0; base < nrc_y * nrc_x; base += nrc_y) {
            // + 4 registers
            int8x16x4_t q4 = vld1q_s8_x4(q4qs);
            // + 1 register
            int8x16_t b0 = q4.val[0] << 4;
            q4qs += 0x40;
            q4.val[0] = q4.val[0] & 0xf0U;
            // + 8 registers
            int32x4_t accm0 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 0);
            int32x4_t accm1 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 1);
            int32x4_t accm2 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 2);
            int32x4_t accm3 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs0.val[0], 3);
            int32x4_t accm4 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs4.val[0], 0);
            int32x4_t accm5 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs4.val[0], 1);
            int32x4_t accm6 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs4.val[0], 2);
            int32x4_t accm7 = vdotq_laneq_s32(vdupq_n_s32(0), b0, q8qs4.val[0], 3);
            b0 = q4.val[1] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[0], q8qs1.val[0], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[0], q8qs1.val[0], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[0], q8qs1.val[0], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[0], q8qs1.val[0], 3);
            accm4 = vdotq_laneq_s32(accm4, q4.val[0], q8qs5.val[0], 0);
            accm5 = vdotq_laneq_s32(accm5, q4.val[0], q8qs5.val[0], 1);
            accm6 = vdotq_laneq_s32(accm6, q4.val[0], q8qs5.val[0], 2);
            accm7 = vdotq_laneq_s32(accm7, q4.val[0], q8qs5.val[0], 3);
            // - 1 register
            q4.val[1] = q4.val[1] & 0xf0U;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[1], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[1], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[1], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[1], 3);
            accm4 = vdotq_laneq_s32(accm4, b0, q8qs4.val[1], 0);
            accm5 = vdotq_laneq_s32(accm5, b0, q8qs4.val[1], 1);
            accm6 = vdotq_laneq_s32(accm6, b0, q8qs4.val[1], 2);
            accm7 = vdotq_laneq_s32(accm7, b0, q8qs4.val[1], 3);
            b0 = q4.val[2] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[1], q8qs1.val[1], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[1], q8qs1.val[1], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[1], q8qs1.val[1], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[1], q8qs1.val[1], 3);
            accm4 = vdotq_laneq_s32(accm4, q4.val[1], q8qs5.val[1], 0);
            accm5 = vdotq_laneq_s32(accm5, q4.val[1], q8qs5.val[1], 1);
            accm6 = vdotq_laneq_s32(accm6, q4.val[1], q8qs5.val[1], 2);
            accm7 = vdotq_laneq_s32(accm7, q4.val[1], q8qs5.val[1], 3);
            // - 1 register
            q4.val[2] = q4.val[2] & 0xf0U;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[2], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[2], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[2], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[2], 3);
            accm4 = vdotq_laneq_s32(accm4, b0, q8qs4.val[2], 0);
            accm5 = vdotq_laneq_s32(accm5, b0, q8qs4.val[2], 1);
            accm6 = vdotq_laneq_s32(accm6, b0, q8qs4.val[2], 2);
            accm7 = vdotq_laneq_s32(accm7, b0, q8qs4.val[2], 3);
            b0 = q4.val[3] << 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[2], q8qs1.val[2], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[2], q8qs1.val[2], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[2], q8qs1.val[2], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[2], q8qs1.val[2], 3);
            accm4 = vdotq_laneq_s32(accm4, q4.val[2], q8qs5.val[2], 0);
            accm5 = vdotq_laneq_s32(accm5, q4.val[2], q8qs5.val[2], 1);
            accm6 = vdotq_laneq_s32(accm6, q4.val[2], q8qs5.val[2], 2);
            accm7 = vdotq_laneq_s32(accm7, q4.val[2], q8qs5.val[2], 3);
            // - 1 register
            q4.val[3] = q4.val[3] & 0xf0U;
            accm0 = vdotq_laneq_s32(accm0, b0, q8qs0.val[3], 0);
            accm1 = vdotq_laneq_s32(accm1, b0, q8qs0.val[3], 1);
            accm2 = vdotq_laneq_s32(accm2, b0, q8qs0.val[3], 2);
            accm3 = vdotq_laneq_s32(accm3, b0, q8qs0.val[3], 3);
            accm4 = vdotq_laneq_s32(accm4, b0, q8qs4.val[3], 0);
            accm5 = vdotq_laneq_s32(accm5, b0, q8qs4.val[3], 1);
            accm6 = vdotq_laneq_s32(accm6, b0, q8qs4.val[3], 2);
            accm7 = vdotq_laneq_s32(accm7, b0, q8qs4.val[3], 3);
            // - 1 register
            // + 1 register
            float32x4_t q4_d = vcvt_f32_f16(vld1_f16(dptr));
            dptr += 4;
            accm0 = vdotq_laneq_s32(accm0, q4.val[3], q8qs1.val[3], 0);
            accm1 = vdotq_laneq_s32(accm1, q4.val[3], q8qs1.val[3], 1);
            accm2 = vdotq_laneq_s32(accm2, q4.val[3], q8qs1.val[3], 2);
            accm3 = vdotq_laneq_s32(accm3, q4.val[3], q8qs1.val[3], 3);
            accm4 = vdotq_laneq_s32(accm4, q4.val[3], q8qs5.val[3], 0);
            accm5 = vdotq_laneq_s32(accm5, q4.val[3], q8qs5.val[3], 1);
            accm6 = vdotq_laneq_s32(accm6, q4.val[3], q8qs5.val[3], 2);
            accm7 = vdotq_laneq_s32(accm7, q4.val[3], q8qs5.val[3], 3);
            // - 1 register
            // + 2 register
            float32x4_t scale0 = vmulq_laneq_f32(q4_d, q8_d0, 0);
            float32x4_t scale1 = vmulq_laneq_f32(q4_d, q8_d0, 1);
            accd[base + 0] = vfmaq_f32(accd[base + 0], vcvtq_n_f32_s32(accm0, 4), scale0);
            scale0 = vmulq_laneq_f32(q4_d, q8_d0, 2);
            accd[base + 1] = vfmaq_f32(accd[base + 1], vcvtq_n_f32_s32(accm1, 4), scale1);
            scale1 = vmulq_laneq_f32(q4_d, q8_d0, 3);
            accd[base + 2] = vfmaq_f32(accd[base + 2], vcvtq_n_f32_s32(accm2, 4), scale0);
            scale0 = vmulq_laneq_f32(q4_d, q8_d1, 0);
            accd[base + 3] = vfmaq_f32(accd[base + 3], vcvtq_n_f32_s32(accm3, 4), scale1);
            scale1 = vmulq_laneq_f32(q4_d, q8_d1, 1);
            accd[base + 4] = vfmaq_f32(accd[base + 4], vcvtq_n_f32_s32(accm4, 4), scale0);
            scale0 = vmulq_laneq_f32(q4_d, q8_d1, 2);
            accd[base + 5] = vfmaq_f32(accd[base + 5], vcvtq_n_f32_s32(accm5, 4), scale1);
            scale1 = vmulq_laneq_f32(q4_d, q8_d1, 3);
            accd[base + 6] = vfmaq_f32(accd[base + 6], vcvtq_n_f32_s32(accm6, 4), scale0);
            accd[base + 7] = vfmaq_f32(accd[base + 7], vcvtq_n_f32_s32(accm7, 4), scale1);
        }
    }
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int k = 0; k < nrc_x; ++k) {
            info->store(k * N32B, iy, accd[iy + k * nrc_y]);
        }
    }
}

struct Zyk_MXFP4_T {
    // nx: the total rows of weight matrix (aka, colunm size)
    // ix: the start row of current process splited weight matrix
    Zyk_MXFP4_T(const void * vx, size_t nb, size_t nx, size_t ix) {
        q4qs = (const uint8_t *)vx + nb*nx + 16*nb*ix; // every block contains 16 bytes qs
        dptr = (const uint32_t *)vx + nb*ix/sizeof(uint32_t);
        values = vld1q_s8(kvalues_mxfp4);
    }

    template <int nrc_y>
    inline void compute_block(const int8_t ** q8, float32x4_t * q8_ds, float32x4_t * accd, int nrc_x) {
        int8x16x2_t q8qs[nrc_y];
        for (int iy = 0; iy < nrc_y; ++iy) {
            q8qs[iy] = vld1q_s8_x2(q8[iy]);
            q8[iy] += 0x20;
        }
        int8x16_t b0[4];
        // #pragma clang loop unroll_count(2)
        for (int base = 0; base < nrc_y * nrc_x; base += nrc_y) {
            float32x4_t q4_d = ggml_e8m0_to_fp32_half_vec(*dptr);
            dptr += 1;
            uint8x16x4_t q4 = vld1q_u8_x4(q4qs);
            q4qs += 0x40;

            b0[0]     = vqtbl1q_s8(values, q4.val[0] & 0x0F);
            q4.val[0] = vqtbl1q_s8(values, vshrq_n_u8(q4.val[0],4));
            b0[1]     = vqtbl1q_s8(values, q4.val[1] & 0x0F);
            q4.val[1] = vqtbl1q_s8(values, vshrq_n_u8(q4.val[1],4));
            b0[2]     = vqtbl1q_s8(values, q4.val[2] & 0x0F);
            q4.val[2] = vqtbl1q_s8(values, vshrq_n_u8(q4.val[2],4));
            b0[3]     = vqtbl1q_s8(values, q4.val[3] & 0x0F);
            q4.val[3] = vqtbl1q_s8(values, vshrq_n_u8(q4.val[3],4));
            #pragma clang loop unroll(full)
            for (int iy = 0; iy < nrc_y; ++iy) {
                int32x4_t accm = vdupq_n_s32(0);
                float32x4_t scale = vmulq_f32(q4_d, q8_ds[iy]);
                float32x4_t accd_val = accd[base + iy];
                accm = vdotq_laneq_s32(accm, b0[0], q8qs[iy].val[0], 0);
                accm = vdotq_laneq_s32(accm, b0[1], q8qs[iy].val[0], 1);
                accm = vdotq_laneq_s32(accm, b0[2], q8qs[iy].val[0], 2);
                accm = vdotq_laneq_s32(accm, b0[3], q8qs[iy].val[0], 3);
                accm = vdotq_laneq_s32(accm, q4.val[0], q8qs[iy].val[1], 0);
                accm = vdotq_laneq_s32(accm, q4.val[1], q8qs[iy].val[1], 1);
                accm = vdotq_laneq_s32(accm, q4.val[2], q8qs[iy].val[1], 2);
                accm = vdotq_laneq_s32(accm, q4.val[3], q8qs[iy].val[1], 3);
                accd[base + iy] = vfmaq_f32(accd_val, vcvtq_f32_s32(accm), scale);
            }
        }
    }

    const uint8_t * q4qs;
    const uint32_t * dptr;
    int8x16_t values;
};

template <typename Dequantizer, int nrc_y>
static void mul_mat_q4_t_q8_0(int n, const void * vx, size_t ix, const DataInfo* info, int nrc_x) {
    GGML_ASSERT(n % QK8_0 == 0);
    const size_t nb = n / QK8_0;
    const int8_t * q8[nrc_y];
    for (int iy = 0; iy < nrc_y; ++iy) {
        q8[iy] = (const int8_t *)(info->src1_row(iy));
    }

    float d8[4*nrc_y];
    float32x4_t q8_ds[nrc_y];
    Dequantizer deq(vx, nb, info->bs, ix);
    // Initialize accumulation vector to zero
    float32x4_t * accd = thread_local_work_buffer(nrc_y * nrc_x);
    // #pragma clang loop unroll_count(4)
    for (size_t ib4 = 0; ib4 < nb/4; ++ib4) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            vst1q_f32(d8+4*iy, vcvt_f32_f16(vld1_f16((const float16_t *)(q8[iy]))));
            q8[iy] += 8;
        }
        for (int k = 0; k < 4; ++k) {
            for (int iy = 0; iy < nrc_y; ++iy) q8_ds[iy] = vdupq_n_f32(d8[4*iy+k]);
            deq.template compute_block<nrc_y>(q8, q8_ds, accd, nrc_x);
        }
    }
    for (size_t ib = 4*(nb/4); ib < nb; ++ib) {
        for (int iy = 0; iy < nrc_y; ++iy) {
            q8_ds[iy] = vdupq_n_f32(GGML_FP16_TO_FP32(*(const ggml_half *)(q8[iy])));
            q8[iy] += 2;
        }
        deq.template compute_block<nrc_y>(q8, q8_ds, accd, nrc_x);
    }
    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int k = 0; k < nrc_x; ++k) {
            info->store(k * N32B, iy, accd[iy + k * nrc_y]);
        }
    }
}

std::array<mul_mat_t, IQK_MAX_NY-1> q4_0_t_funcs = {
    mul_mat_q4_t_q8_0<ZykQ4_0_T, 1>,
    mul_mat_q4_t_q8_0<ZykQ4_0_T, 2>,
    mul_mat_q4_t_q8_0<ZykQ4_0_T, 3>,
#if USE_Q4_0_OPT
    mul_mat_q4_t4_q8_0,
    mul_mat_q4_t567_q8_0<5>,
    mul_mat_q4_t567_q8_0<6>,
    mul_mat_q4_t567_q8_0<7>
#else
    mul_mat_q4_t_q8_0<ZykQ4_0_T, 4>,
    mul_mat_q4_t_q8_0<ZykQ4_0_T, 5>,
    mul_mat_q4_t_q8_0<ZykQ4_0_T, 6>,
    mul_mat_q4_t_q8_0<ZykQ4_0_T, 7>
#endif // USE_Q4_0_OPT
};

std::array<mul_mat_t, IQK_MAX_NY-1> mxfp4_t_funcs = {
    mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 1>,
    mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 2>,
    mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 3>,
    mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 4>,
    mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 5>,
    mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 6>,
    mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 7>
};
#endif
#endif

void ggml_quantize_mat_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 3] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 11] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 15] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x4_generic(x, vy, k);
#endif
}

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 4; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][2 * j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][2 * j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][2 * j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
        }
    }

#else
    UNUSED(nb);
    UNUSED(y);
    ggml_quantize_mat_q8_0_4x8_generic(x, vy, k);
#endif
}

void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs);
            int8x16_t b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32);
            int8x16_t b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);

            int8x16_t a0 = vld1q_s8(a_ptr->qs);
            int8x16_t a1 = vld1q_s8(a_ptr->qs + qk/2);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

            int32x4_t ret = vdupq_n_s32(0);

            ret = vdotq_laneq_s32(ret, b0 << 4, a0, 0);
            ret = vdotq_laneq_s32(ret, b1 << 4, a0, 1);
            ret = vdotq_laneq_s32(ret, b2 << 4, a0, 2);
            ret = vdotq_laneq_s32(ret, b3 << 4, a0, 3);

            ret = vdotq_laneq_s32(ret, b0 & 0xf0U, a1, 0);
            ret = vdotq_laneq_s32(ret, b1 & 0xf0U, a1, 1);
            ret = vdotq_laneq_s32(ret, b2 & 0xf0U, a1, 2);
            ret = vdotq_laneq_s32(ret, b3 & 0xf0U, a1, 3);

            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                            vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += ncols_interleaved;
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t acc = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs);
            int8x16_t b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
            int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32);
            int8x16_t b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
            float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);

            int8x16_t a0 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs);
            int8x16_t a1 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 1);
            int8x16_t a2 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 2);
            int8x16_t a3 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 3);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

            int32x4_t ret0 = vdupq_n_s32(0);
            int32x4_t ret1 = vdupq_n_s32(0);

            ret0 = vdotq_s32(ret0, b0 << 4, a0);
            ret1 = vdotq_s32(ret1, b1 << 4, a0);
            ret0 = vdotq_s32(ret0, b2 << 4, a1);
            ret1 = vdotq_s32(ret1, b3 << 4, a1);

            ret0 = vdotq_s32(ret0, b0 & 0xf0U, a2);
            ret1 = vdotq_s32(ret1, b1 & 0xf0U, a2);
            ret0 = vdotq_s32(ret0, b2 & 0xf0U, a3);
            ret1 = vdotq_s32(ret1, b3 & 0xf0U, a3);

            int32x4_t ret = vpaddq_s32(ret0, ret1);

            acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                    vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += ncols_interleaved;
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE)
    if (ggml_cpu_get_sve_cnt() == QK8_0) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;

        __asm__ __volatile__(
            "ptrue p0.b\n"
            "add %x[b_ptr], %x[b_ptr], #0x10\n"
            "1:"  // Column loop
            "add x22, %x[a_ptr], #0x2\n"
            "mov z31.b, #0x0\n"
            "mov x21, %x[nb]\n"
            "2:"  // Block loop
            "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
            "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
            "mov z28.s, #0x0\n"
            "mov z27.s, #0x0\n"
            "ld1rd { z26.d }, p0/Z, [x22]\n"
            "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
            "sub x20, x22, #0x2\n"
            "sub x21, x21, #0x1\n"
            "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
            "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
            "lsl z22.b, z30.b, #0x4\n"
            "lsl z16.b, z29.b, #0x4\n"
            "and z30.b, z30.b, #0xf0\n"
            "and z29.b, z29.b, #0xf0\n"
            "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
            "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
            "lsl z19.b, z25.b, #0x4\n"
            "and z25.b, z25.b, #0xf0\n"
            "ld1rh { z17.h }, p0/Z, [x20]\n"
            "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
            "sdot z28.s, z22.b, z26.b\n"
            "sdot z27.s, z16.b, z26.b\n"
            "lsl z16.b, z24.b, #0x4\n"
            "add x22, x22, #0x22\n"
            "and z24.b, z24.b, #0xf0\n"
            "add %x[b_ptr], %x[b_ptr], #0x90\n"
            "fcvt z17.s, p0/m, z17.h\n"
            "fcvt z18.s, p0/m, z18.h\n"
            "sdot z28.s, z19.b, z23.b\n"
            "sdot z27.s, z16.b, z23.b\n"
            "fmul z18.s, z18.s, z17.s\n"
            "sdot z28.s, z30.b, z21.b\n"
            "sdot z27.s, z29.b, z21.b\n"
            "sdot z28.s, z25.b, z20.b\n"
            "sdot z27.s, z24.b, z20.b\n"
            "uzp1 z17.s, z28.s, z27.s\n"
            "uzp2 z16.s, z28.s, z27.s\n"
            "add z17.s, z17.s, z16.s\n"
            "asr z17.s, z17.s, #0x4\n"
            "scvtf z17.s, p0/m, z17.s\n"
            "fmla z31.s, p0/M, z17.s, z18.s\n"
            "cbnz x21, 2b\n"
            "sub %x[nc], %x[nc], #0x8\n"
            "st1w { z31.s }, p0, [%x[res_ptr]]\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "cbnz %x[nc], 1b\n"
            : [b_ptr] "+&r" (b_ptr), [res_ptr] "+&r" (res_ptr), [nc] "+&r" (nc)
            : [a_ptr] "r" (a_ptr), [nb] "r" (nb)
            : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
        );
        return;
    }
#endif // #if defined(__ARM_FEATURE_SVE)

#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
    ggml_gemv_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t kvalues = vld1q_s8(kvalues_iq4nl);
    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    float * res_ptr = s;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

        float32x4_t sumf = vdupq_n_f32(0);
        for (int l = 0; l < nb; l++) {
            uint8x16_t b_0 = vld1q_u8(b_ptr[l].qs + 0);
            uint8x16_t b_1 = vld1q_u8(b_ptr[l].qs + 16);
            uint8x16_t b_2 = vld1q_u8(b_ptr[l].qs + 32);
            uint8x16_t b_3 = vld1q_u8(b_ptr[l].qs + 48);

            int8x16_t b_0_hi = vqtbl1q_s8(kvalues, b_0 >> 4);
            int8x16_t b_0_lo = vqtbl1q_s8(kvalues, b_0 & 0x0F);
            int8x16_t b_1_hi = vqtbl1q_s8(kvalues, b_1 >> 4);
            int8x16_t b_1_lo = vqtbl1q_s8(kvalues, b_1 & 0x0F);
            int8x16_t b_2_hi = vqtbl1q_s8(kvalues, b_2 >> 4);
            int8x16_t b_2_lo = vqtbl1q_s8(kvalues, b_2 & 0x0F);
            int8x16_t b_3_hi = vqtbl1q_s8(kvalues, b_3 >> 4);
            int8x16_t b_3_lo = vqtbl1q_s8(kvalues, b_3 & 0x0F);

            int8x16_t a_0 = vld1q_s8(a_ptr[l].qs + 0);
            int8x16_t a_1 = vld1q_s8(a_ptr[l].qs + 16);

            int32x4_t sumi = vdupq_n_s32(0);
            sumi = vdotq_laneq_s32(sumi, b_0_lo, a_0, 0);
            sumi = vdotq_laneq_s32(sumi, b_0_hi, a_1, 0);
            sumi = vdotq_laneq_s32(sumi, b_1_lo, a_0, 1);
            sumi = vdotq_laneq_s32(sumi, b_1_hi, a_1, 1);
            sumi = vdotq_laneq_s32(sumi, b_2_lo, a_0, 2);
            sumi = vdotq_laneq_s32(sumi, b_2_hi, a_1, 2);
            sumi = vdotq_laneq_s32(sumi, b_3_lo, a_0, 3);
            sumi = vdotq_laneq_s32(sumi, b_3_hi, a_1, 3);

            float32x4_t a_d = vcvt_f32_f16(vld1_dup_f16((const float16_t *)&a_ptr[l].d));
            float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *)b_ptr[l].d));
            float32x4_t d = a_d * b_d;

            sumf = vmlaq_f32(sumf, d, vcvtq_f32_s32(sumi));
        }

        vst1q_f32(res_ptr + x * 4, sumf);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    ggml_gemv_iq4_nl_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    col_groups = ncols_interleaved / 4; // 0123 and 4567
    const uint8x16_t m4b        = vdupq_n_u8(0x0f);

    // 1x8 tile = 2 x 4
    float32x4_t acc_f32[col_groups];

    const block_q8_K * GGML_RESTRICT q8_ptr = (const block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int i = 0; i < col_groups; i++) {
            acc_f32[i] = vdupq_n_f32(0);
        }

        for (int b = 0; b < nb; b++) {
            float32x4_t q4_d_0        = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));      // d0 d1 d2 d3
            float32x4_t q4_d_1        = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));  // d4 d5 d6 d7
            float32x4_t q8_d          = vdupq_n_f32(q8_ptr[b].d);
            float32x4_t sb_scale_0123 = vmulq_f32(q4_d_0, q8_d);
            float32x4_t sb_scale_4567 = vmulq_f32(q4_d_1, q8_d);
            float32x4_t q4_dmin_0     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));      // dmin 0..3
            float32x4_t q4_dmin_1     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));  // dmin 4..7
            float32x4_t sb_min_0123   = vmulq_f32(q4_dmin_0, q8_d);
            float32x4_t sb_min_4567   = vmulq_f32(q4_dmin_1, q8_d);

            // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567
            int32x4_t bias_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
            int32x4_t acc_lo[col_groups];
            int32x4_t acc_hi[col_groups];

            // Each bsum is 16 elements, pairwise add leaves us with the 8 bsums of the entire block
            const int16x8_t bsums = vpaddq_s16(vld1q_s16(q8_ptr[b].bsums), vld1q_s16(q8_ptr[b].bsums + 8));
            int16_t         bsums_arr[8];
            vst1q_s16(bsums_arr, bsums);
            for (int sb = 0; sb < QK_K / 64; sb++) {
                for (int i = 0; i < col_groups; i++) {
                    acc_lo[i] = vdupq_n_s32(0);
                    acc_hi[i] = vdupq_n_s32(0);
                }
                // Need scales for the low and high nibbles
                // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
                int16x8_t q4sb_mins[2];
                int16x8_t q4sb_scales[2];
                for (int i = 0; i < 2; i++) {
                    int8_t    aux_q4sb[8];
                    const int offset = sb * 24 + i * 12;
                    decode_q4_Kx8_scales_mins(&q4_ptr[b].scales[offset], &q4sb_mins[i], aux_q4sb);
                    q4sb_scales[i] = vmovl_s8(vld1_s8(aux_q4sb));
                }

                int8x16_t q8_qs[64 / 16];
                for (int i = 0; i < 64 / 16; i++) {
                    q8_qs[i] = vld1q_s8(q8_ptr[b].qs + sb * 64 + i * 16);
                }

                for (int c = 0; c < col_groups; c++) {
                    uint8x16_t q4_cols[8];
                    for (int i = 0; i < 8; i++) {
                        q4_cols[i] = vld1q_u8(q4_ptr[b].qs + sb * QK_K + i * 32 + 16 * c);
                    }

                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[0], m4b)), q8_qs[0], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[1], m4b)), q8_qs[0], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[2], m4b)), q8_qs[0], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[3], m4b)), q8_qs[0], 3);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[4], m4b)), q8_qs[1], 0);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[5], m4b)), q8_qs[1], 1);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[6], m4b)), q8_qs[1], 2);
                    acc_lo[c] = vdotq_laneq_s32(acc_lo[c], vreinterpretq_s8_u8(vandq_u8(q4_cols[7], m4b)), q8_qs[1], 3);

                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[0], 4)), q8_qs[2], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[1], 4)), q8_qs[2], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[2], 4)), q8_qs[2], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[3], 4)), q8_qs[2], 3);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[4], 4)), q8_qs[3], 0);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[5], 4)), q8_qs[3], 1);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[6], 4)), q8_qs[3], 2);
                    acc_hi[c] = vdotq_laneq_s32(acc_hi[c], vreinterpretq_s8_u8(vshrq_n_u8(q4_cols[7], 4)), q8_qs[3], 3);
                }

                // Scales
                // row c0123 blk0 and blk1
                const int16x4_t   sc_0123_lo = vget_low_s16(q4sb_scales[0]);
                const int16x4_t   sc_0123_hi = vget_low_s16(q4sb_scales[1]);
                const float32x4_t sumf_0123  = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_0123_lo), acc_lo[0]),
                                                                       vmulq_s32(vmovl_s16(sc_0123_hi), acc_hi[0])));
                acc_f32[0]                   = vfmaq_f32(acc_f32[0], sb_scale_0123, sumf_0123);
                // row c4567 blk0 and blk1
                const int16x4_t   sc_4567_lo = vget_high_s16(q4sb_scales[0]);
                const int16x4_t   sc_4567_hi = vget_high_s16(q4sb_scales[1]);
                const float32x4_t sumf_4567  = vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_4567_lo), acc_lo[1]),
                                                                       vmulq_s32(vmovl_s16(sc_4567_hi), acc_hi[1])));
                acc_f32[1]                   = vfmaq_f32(acc_f32[1], sb_scale_4567, sumf_4567);

                // Bias Correction
                const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb + 0]);
                const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);

                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
            }  // for sb

            acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0123);
            acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_4567);
        }  // for b

        int base = x * ncols_interleaved;
        vst1q_f32(s + base, acc_f32[0]);
        vst1q_f32(s + base + 4, acc_f32[1]);
    }  // for x
    return;
#endif  // #if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q4_K_8x8_q8_K(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    col_pairs = ncols_interleaved / 2;
    const uint8x16_t m4b       = vdupq_n_u8(0x0f);

    // 1x8 tile = 2 x 4
    float32x4_t acc_f32[ncols_interleaved / 4];

    const block_q8_K * GGML_RESTRICT q8_ptr = (const block_q8_K *) vy;

    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int i = 0; i < ncols_interleaved / 4; i++) {
            acc_f32[i] = vdupq_n_f32(0);
        }

        for (int b = 0; b < nb; b++) {
            float32x4_t q4_d_0     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));      // d0 d1 d2 d3
            float32x4_t q4_d_1     = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));  // d4 d5 d6 d7
            float32x4_t q8_d       = vdupq_n_f32(q8_ptr[b].d);
            float32x4_t sb_scale_0 = vmulq_f32(q4_d_0, q8_d);
            float32x4_t sb_scale_1 = vmulq_f32(q4_d_1, q8_d);
            float32x4_t q4_dmin_0  = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));      // dmin 0..3
            float32x4_t q4_dmin_1  = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));  // dmin 4..7
            float32x4_t sb_min_0   = vmulq_f32(q4_dmin_0, q8_d);
            float32x4_t sb_min_1   = vmulq_f32(q4_dmin_1, q8_d);

            // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567
            int32x4_t bias_acc[2] = { vdupq_n_s32(0), vdupq_n_s32(0) };
            // 2 sb each iteration
            int32x4_t acc_lo[col_pairs];
            int32x4_t acc_hi[col_pairs];

            // Each bsum is 16 elements, pairwise add leaves us with the 8 bsums of the entire block
            const int16x8_t bsums = vpaddq_s16(vld1q_s16(q8_ptr[b].bsums), vld1q_s16(q8_ptr[b].bsums + 8));
            int16_t         bsums_arr[8];
            vst1q_s16(bsums_arr, bsums);
            for (int sb = 0; sb < QK_K / 64; sb++) {
                for (int i = 0; i < col_pairs; i++) {
                    acc_lo[i] = vdupq_n_s32(0);
                    acc_hi[i] = vdupq_n_s32(0);
                }
                // Need scales for the low and high nibbles
                // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
                int16x8_t q4sb_mins[2];  // int16 as its needed for bias_acc later
                int16x8_t q4sb_scales[2];
                for (int i = 0; i < 2; i++) {
                    int8_t    aux_q4sb[8];
                    const int offset = sb * 24 + i * 12;
                    decode_q4_Kx8_scales_mins(&q4_ptr[b].scales[offset], &q4sb_mins[i], aux_q4sb);
                    q4sb_scales[i] = vmovl_s8(vld1_s8(aux_q4sb));
                }

                const uint8_t * q4_base = q4_ptr[b].qs + sb * QK_K;

                // Load the 64 quants from q8K duplicated to use vecdots with the interelaved columns
                // but still need the qs to use the low and hi bits from q4
                const int8_t * q8_base = q8_ptr[b].qs + sb * 64;
                int8x16_t      q8_qs[8];
                for (int i = 0; i < 8; i++) {
                    q8_qs[i] = (int8x16_t) vld1q_dup_s64((const int64_t *) (q8_base + i * 8));
                }

                // Q4s columns iterated in pairs (01, 23, 45, 67)
                for (int cp = 0; cp < col_pairs; cp++) {
                    uint8x16_t q4_qs_cp_0 = vld1q_u8(q4_base + 16 * cp);
                    uint8x16_t q4_qs_cp_1 = vld1q_u8(q4_base + 16 * cp + 64);
                    uint8x16_t q4_qs_cp_2 = vld1q_u8(q4_base + 16 * cp + 128);
                    uint8x16_t q4_qs_cp_3 = vld1q_u8(q4_base + 16 * cp + 192);

                    acc_lo[cp] =
                        ggml_vdotq_s32(acc_lo[cp], vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_0, m4b)), q8_qs[0]);  // 0 .. 7
                    acc_lo[cp] =
                        ggml_vdotq_s32(acc_lo[cp], vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_1, m4b)), q8_qs[1]);  // 8 ..15
                    acc_lo[cp] =
                        ggml_vdotq_s32(acc_lo[cp], vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_2, m4b)), q8_qs[2]);  // 16..23
                    acc_lo[cp] =
                        ggml_vdotq_s32(acc_lo[cp], vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_3, m4b)), q8_qs[3]);  // 24..31

                    acc_hi[cp] =
                        ggml_vdotq_s32(acc_hi[cp], vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_0, 4)), q8_qs[4]);  // 32..39
                    acc_hi[cp] =
                        ggml_vdotq_s32(acc_hi[cp], vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_1, 4)), q8_qs[5]);  // 40..47
                    acc_hi[cp] =
                        ggml_vdotq_s32(acc_hi[cp], vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_2, 4)), q8_qs[6]);  // 48..55
                    acc_hi[cp] =
                        ggml_vdotq_s32(acc_hi[cp], vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_3, 4)), q8_qs[7]);  // 56..63
                }

                // Iterates over a pair of column pairs (4 columns) to use a single 128 register
                // p = 0 -> 0123  p2 -> 4567
                for (int i = 0, p = 0; p < col_pairs; i++, p += 2) {
                    int16x4_t   group_scales_lo = p == 0 ? vget_low_s16(q4sb_scales[0]) : vget_high_s16(q4sb_scales[0]);
                    int16x4_t   group_scales_hi = p == 0 ? vget_low_s16(q4sb_scales[1]) : vget_high_s16(q4sb_scales[1]);
                    float32x4_t sb_scale        = p == 0 ? sb_scale_0 : sb_scale_1;

                    // 0123 or 4567
                    float32x4_t sumf_0 =
                        vcvtq_f32_s32(vmulq_s32(vmovl_s16(group_scales_lo), vpaddq_s32(acc_lo[p], acc_lo[p + 1])));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale, sumf_0);

                    float32x4_t sumf_1 =
                        vcvtq_f32_s32(vmulq_s32(vmovl_s16(group_scales_hi), vpaddq_s32(acc_hi[p], acc_hi[p + 1])));
                    acc_f32[i] = vfmaq_f32(acc_f32[i], sb_scale, sumf_1);
                }

                // Multiply Acc bsum + mins
                // Each pair of subblocks share the same bsums
                // Load scalar bsum → broadcast to a vector (vdupq_n_s16(s)).
                int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[2 * sb + 0]);
                int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[2 * sb + 1]);

                // cols 0-3 bias
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                bias_acc[0] = vmlal_s16(bias_acc[0], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));

                // cols 4-7 bias
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                bias_acc[1] = vmlal_s16(bias_acc[1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
            }  // for sb

            acc_f32[0] = vmlsq_f32(acc_f32[0], vcvtq_f32_s32(bias_acc[0]), sb_min_0);
            acc_f32[1] = vmlsq_f32(acc_f32[1], vcvtq_f32_s32(bias_acc[1]), sb_min_1);
        }  // for b

        int base = x * ncols_interleaved;
        vst1q_f32(s + base, acc_f32[0]);
        vst1q_f32(s + base + 4, acc_f32[1]);
    }  // for x
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q8_0_4x4_q8_0(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx;

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t        acc   = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) {
            int8x16x4_t b_low  = vld1q_s8_x4((const int8_t *) b_ptr->qs);
            int8x16x4_t b_high = vld1q_s8_x4((const int8_t *) b_ptr->qs + 64);
            float16x4_t bd     = vld1_f16((const __fp16 *) b_ptr->d);

            int8x16x2_t a  = vld1q_s8_x2(a_ptr->qs);
            float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

            int32x4_t ret = vdupq_n_s32(0);

            ret = vdotq_laneq_s32(ret, b_low.val[0], a.val[0], 0);
            ret = vdotq_laneq_s32(ret, b_low.val[1], a.val[0], 1);
            ret = vdotq_laneq_s32(ret, b_low.val[2], a.val[0], 2);
            ret = vdotq_laneq_s32(ret, b_low.val[3], a.val[0], 3);

            ret = vdotq_laneq_s32(ret, b_high.val[0], a.val[1], 0);
            ret = vdotq_laneq_s32(ret, b_high.val[1], a.val[1], 1);
            ret = vdotq_laneq_s32(ret, b_high.val[2], a.val[1], 2);
            ret = vdotq_laneq_s32(ret, b_high.val[3], a.val[1], 3);

            acc = vfmaq_f32(acc, vcvtq_f32_s32(ret), vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += ncols_interleaved;
    }
    return;

#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemv_q8_0_4x8_q8_0(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx;

    for (int c = 0; c < nc; c += ncols_interleaved) {
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float32x4_t        acc   = vdupq_n_f32(0);

        for (int b = 0; b < nb; b++) {
            int8x16x4_t b_low  = vld1q_s8_x4((const int8_t *) b_ptr->qs);
            int8x16x4_t b_high = vld1q_s8_x4((const int8_t *) b_ptr->qs + 64);
            float16x4_t bd     = vld1_f16((const __fp16 *) b_ptr->d);

            int8x8x4_t  a_chunks = vld1_s8_x4(a_ptr->qs);
            int8x16_t   a0       = vcombine_s8(a_chunks.val[0], a_chunks.val[0]);
            int8x16_t   a1       = vcombine_s8(a_chunks.val[1], a_chunks.val[1]);
            int8x16_t   a2       = vcombine_s8(a_chunks.val[2], a_chunks.val[2]);
            int8x16_t   a3       = vcombine_s8(a_chunks.val[3], a_chunks.val[3]);
            float16x4_t ad       = vld1_dup_f16((const __fp16 *) &a_ptr->d);

            int32x4_t ret0 = vdupq_n_s32(0);
            int32x4_t ret1 = vdupq_n_s32(0);

            // 0..7
            ret0 = vdotq_s32(ret0, b_low.val[0], a0);
            ret1 = vdotq_s32(ret1, b_low.val[1], a0);
            // 8..15
            ret0 = vdotq_s32(ret0, b_low.val[2], a1);
            ret1 = vdotq_s32(ret1, b_low.val[3], a1);
            // 16..23
            ret0 = vdotq_s32(ret0, b_high.val[0], a2);
            ret1 = vdotq_s32(ret1, b_high.val[1], a2);
            // 24..31
            ret0 = vdotq_s32(ret0, b_high.val[2], a3);
            ret1 = vdotq_s32(ret1, b_high.val[3], a3);

            int32x4_t ret = vpaddq_s32(ret0, ret1);

            acc = vfmaq_f32(acc, vcvtq_f32_s32(ret), vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
            a_ptr++;
            b_ptr++;
        }
        vst1q_f32(s, acc);
        s += ncols_interleaved;
    }
    return;

#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemv_q8_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#if 1
    const void * b_ptr = vx;
    const void * a_ptr = vy;
    float * res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:"  // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:"  // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v23.16b, #0x0\n"
        "movi v16.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v0.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v21.16b, #0x0\n"
        "movi v8.16b, #0x0\n"
        "movi v1.16b, #0x0\n"
        "3:"  // Block loop
        "ldr q3, [x28, #0x0]\n"
        "ldr q31, [x25, #0x0]\n"
        "movi v28.16b, #0x4\n"
        "movi v10.4s, #0x0\n"
        "ldr q22, [x28, #0x10]\n"
        "ldr q6, [x25, #0x10]\n"
        "movi v29.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        "ldr q27, [x28, #0x20]\n"
        "ldr q30, [x28, #0x30]\n"
        "movi v20.4s, #0x0\n"
        "movi v24.16b, #0xf0\n"
        "ldr d2, [x25, #-0x8]\n"
        "ldr d26, [x23, #-0x8]\n"
        "sshl v12.16b, v3.16b, v28.16b\n"
        "sub x20, x28, #0x8\n"
        "ldr d17, [x20, #0x0]\n"
        "and v3.16b, v3.16b, v24.16b\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
        ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
        ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
        ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
        "sshl v31.16b, v22.16b, v28.16b\n"
        "and v22.16b, v22.16b, v24.16b\n"
        "fcvtl v17.4s, v17.4h\n"
        "fcvtl v2.4s, v2.4h\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
        ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
        ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
        ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
        "sshl v6.16b, v27.16b, v28.16b\n"
        "sshl v28.16b, v30.16b, v28.16b\n"
        "and v27.16b, v27.16b, v24.16b\n"
        "and v30.16b, v30.16b, v24.16b\n"
        "ldr q24, [x25, #0x20]\n"
        ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x30]\n"
        ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
        ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
        ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x40]\n"
        ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x50]\n"
        ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
        ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
        ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
        ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x60]\n"
        ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
        ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
        ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
        ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
        "fmul v24.4s, v17.4s, v2.s[0]\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v15.4s, v10.4s, v24.4s\n"
        "ldr q24, [x23, #0x0]\n"
        "fmul v10.4s, v17.4s, v2.s[1]\n"
        "fmla v19.4s, v29.4s, v10.4s\n"
        "ldr q10, [x23, #0x10]\n"
        "fmul v29.4s, v17.4s, v2.s[2]\n"
        "fmul v2.4s, v17.4s, v2.s[3]\n"
        "fmla v18.4s, v9.4s, v29.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
        "fmla v14.4s, v20.4s, v2.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x20]\n"
        ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
        ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
        ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
        ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x30]\n"
        ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x40]\n"
        ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
        ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
        ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
        ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x50]\n"
        ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x23, #0x60]\n"
        ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
        ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
        ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
        ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
        "ldr q10, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x0]\n"
        ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
        ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
        ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
        ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
        "fmul v10.4s, v17.4s, v26.s[0]\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v11.4s, v9.4s, v10.4s\n"
        "ldr q9, [x22, #0x10]\n"
        "fmul v10.4s, v17.4s, v26.s[1]\n"
        "fmla v13.4s, v29.4s, v10.4s\n"
        "ldr d29, [x22, #-0x8]\n"
        "fmul v10.4s, v17.4s, v26.s[2]\n"
        "fmul v26.4s, v17.4s, v26.s[3]\n"
        "fcvtl v29.4s, v29.4h\n"
        "fmla v23.4s, v20.4s, v10.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v16.4s, v2.4s, v26.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v2.4s, #0x0\n"
        ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x20]\n"
        ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x30]\n"
        ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
        ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
        ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
        ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x40]\n"
        ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x50]\n"
        ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
        ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
        ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
        ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
        "ldr q24, [x22, #0x60]\n"
        ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
        "ldr q9, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
        ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
        ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
        ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
        "ldr q24, [x21, #0x0]\n"
        ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
        ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
        "fmul v9.4s, v17.4s, v29.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "fmla v25.4s, v20.4s, v9.4s\n"
        "ldr q9, [x21, #0x10]\n"
        "fmul v20.4s, v17.4s, v29.s[1]\n"
        "fmla v7.4s, v10.4s, v20.4s\n"
        "ldr d20, [x21, #-0x8]\n"
        "fmul v10.4s, v17.4s, v29.s[2]\n"
        "fmul v29.4s, v17.4s, v29.s[3]\n"
        "fcvtl v20.4s, v20.4h\n"
        "fmla v0.4s, v26.4s, v10.4s\n"
        "movi v26.4s, #0x0\n"
        "movi v10.4s, #0x0\n"
        "fmla v4.4s, v2.4s, v29.4s\n"
        "movi v2.4s, #0x0\n"
        "movi v29.4s, #0x0\n"
        ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
        ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
        ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
        ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
        "ldr q12, [x21, #0x20]\n"
        "fmul v24.4s, v17.4s, v20.s[0]\n"
        ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
        ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
        ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
        ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x30]\n"
        "fmul v31.4s, v17.4s, v20.s[1]\n"
        ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
        ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
        ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
        ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x40]\n"
        "fmul v6.4s, v17.4s, v20.s[2]\n"
        "fmul v20.4s, v17.4s, v20.s[3]\n"
        ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
        ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
        ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
        ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
        "ldr q9, [x21, #0x50]\n"
        ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
        ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
        ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
        ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
        "ldr q12, [x21, #0x60]\n"
        ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
        ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
        ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
        ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
        "ldr q17, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
        ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
        ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
        ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
        ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
        ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
        ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
        ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "scvtf v10.4s, v10.4s, #0x4\n"
        "fmla v5.4s, v26.4s, v24.4s\n"
        "scvtf v2.4s, v2.4s, #0x4\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "fmla v21.4s, v10.4s, v31.4s\n"
        "fmla v8.4s, v2.4s, v6.4s\n"
        "fmla v1.4s, v29.4s, v20.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q16, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q0, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q21, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q8, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q1, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:"  // Row loop skip
        "cbz x10, 9f\n"
        "5:"  // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:"  // Row tail: Column loop
        "movi v15.16b, #0x0\n"
        "movi v19.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v18.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "7:"  // Row tail: Block loop
        "ldr q7, [x24, #0x0]\n"
        "ldr q5, [x25, #0x0]\n"
        "movi v9.16b, #0x4\n"
        "movi v4.4s, #0x0\n"
        "ldr q3, [x24, #0x10]\n"
        "ldr q2, [x25, #0x10]\n"
        "movi v1.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q13, [x24, #0x20]\n"
        "ldr q31, [x25, #0x20]\n"
        "movi v30.4s, #0x0\n"
        "movi v29.16b, #0xf0\n"
        "ldr q28, [x24, #0x30]\n"
        "ldr q27, [x25, #0x30]\n"
        "sshl v20.16b, v7.16b, v9.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr q26, [x25, #0x40]\n"
        "ldr q25, [x25, #0x50]\n"
        "sshl v17.16b, v3.16b, v9.16b\n"
        "and v7.16b, v7.16b, v29.16b\n"
        "ldr q24, [x25, #0x60]\n"
        "ldr q16, [x25, #0x70]\n"
        "sshl v22.16b, v13.16b, v9.16b\n"
        "and v3.16b, v3.16b, v29.16b\n"
        "ldr d21, [x20, #0x0]\n"
        "ldr d12, [x25, #-0x8]\n"
        ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
        ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
        ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
        ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
        "sshl v9.16b, v28.16b, v9.16b\n"
        "subs x21, x21, #0x1\n"
        "and v13.16b, v13.16b, v29.16b\n"
        "and v28.16b, v28.16b, v29.16b\n"
        "add x25, x25, #0x88\n"
        "add x24, x24, #0x48\n"
        "fcvtl v21.4s, v21.4h\n"
        "fcvtl v12.4s, v12.4h\n"
        ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
        ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
        ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
        ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
        "fmul v11.4s, v21.4s, v12.s[0]\n"
        "fmul v23.4s, v21.4s, v12.s[1]\n"
        "fmul v17.4s, v21.4s, v12.s[2]\n"
        ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
        "fmul v6.4s, v21.4s, v12.s[3]\n"
        ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
        ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
        ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
        ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
        ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
        ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
        ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
        ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
        ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
        ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
        ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
        ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
        ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
        ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
        ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
        ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
        ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
        ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
        ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
        ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
        ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
        ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
        ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
        "scvtf v4.4s, v4.4s, #0x4\n"
        "scvtf v1.4s, v1.4s, #0x4\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "fmla v15.4s, v4.4s, v11.4s\n"
        "scvtf v30.4s, v30.4s, #0x4\n"
        "fmla v19.4s, v1.4s, v23.4s\n"
        "fmla v18.4s, v0.4s, v17.4s\n"
        "fmla v14.4s, v30.4s, v6.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q15, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q19, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q18, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q14, [x20, #0x0]\n"
        "8:"  // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:"  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
        : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
#elif 1
    const block_q4_0x4 * GGML_RESTRICT b_ptr = (const block_q4_0x4 *)vx;
    const block_q8_0x4 * GGML_RESTRICT a_ptr = (const block_q8_0x4 *)vy;
    float * GGML_RESTRICT res_ptr = s;
    size_t res_stride = bs;

    int nr_remaining = nr; // "mov x10, %x[nr]\n"
    int stride_a_block = nb * sizeof(block_q8_0x4); // "mov x9, #0x88\n", "mul x9, %x[nb], x9\n"

    while (nr_remaining >= 0x10) { // "cmp x10, #0x10\n", "blt 4f\n" | // "cmp x10, #0x10\n", "bge 1b\n"
        // "1:"  // Row loop | Main loop: Process 16 rows at a time
        // Setup row loop pointers
        // b_ptr stays at the current column block base for these 16 rows
        const int8_t *b_qs_base = (const int8_t *)(b_ptr->qs); // "add x28, %x[b_ptr], #0x8\n"
        int nc_remaining = nc; // "mov x27, %x[nc]\n"
        float *res_ptr_next_16 = res_ptr + (res_stride << 4); // "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"

        do { // "2:"  // Column loop | Column loop: Process 4 columns at a time (1 block_q4_0x4)
            // Block loop (nb): iterate over the dimension K
            int k = nb; // "mov x24, %x[nb]\n"
            // Initialize accumulators for 16 rows x 4 columns result
            float32x4_t facc0  = vdupq_n_f32(0.0f); // "movi v15.16b, #0x0\n"
            float32x4_t facc1  = vdupq_n_f32(0.0f); // "movi v19.16b, #0x0\n"
            float32x4_t facc2  = vdupq_n_f32(0.0f); // "movi v18.16b, #0x0\n"
            float32x4_t facc3  = vdupq_n_f32(0.0f); // "movi v14.16b, #0x0\n"
            float32x4_t facc4  = vdupq_n_f32(0.0f); // "movi v11.16b, #0x0\n"
            float32x4_t facc5  = vdupq_n_f32(0.0f); // "movi v13.16b, #0x0\n"
            float32x4_t facc6  = vdupq_n_f32(0.0f); // "movi v23.16b, #0x0\n"
            float32x4_t facc7  = vdupq_n_f32(0.0f); // "movi v16.16b, #0x0\n"
            float32x4_t facc8  = vdupq_n_f32(0.0f); // "movi v25.16b, #0x0\n"
            float32x4_t facc9  = vdupq_n_f32(0.0f); // "movi v7.16b, #0x0\n"
            float32x4_t facc10 = vdupq_n_f32(0.0f); // "movi v0.16b, #0x0\n"
            float32x4_t facc11 = vdupq_n_f32(0.0f); // "movi v4.16b, #0x0\n"
            float32x4_t facc12 = vdupq_n_f32(0.0f); // "movi v5.16b, #0x0\n"
            float32x4_t facc13 = vdupq_n_f32(0.0f); // "movi v21.16b, #0x0\n"
            float32x4_t facc14 = vdupq_n_f32(0.0f); // "movi v8.16b, #0x0\n"
            float32x4_t facc15 = vdupq_n_f32(0.0f); // "movi v1.16b, #0x0\n"
            // a_ptr points to the start of the row blocks. We need to handle 16 rows.
            const int8_t * a_qs_base0 = a_ptr->qs; // "add x25, %x[a_ptr], #0x8\n"
            const int8_t * a_qs_base1 = a_qs_base0 + stride_a_block; // "add x23, x25, x9\n"
            const int8_t * a_qs_base2 = a_qs_base1 + stride_a_block; // "add x22, x23, x9\n"
            const int8_t * a_qs_base3 = a_qs_base2 + stride_a_block; // "add x21, x22, x9\n"
            do { // "3:"  // Block loop
                k--; // "subs x24, x24, #0x1\n"
                // --- Load Q4_0 Block (B) ---
                // B has 4 interleaved columns in qs[0]..qs[3]. Each is 16 bytes (32 4-bit values).
                int8x16_t b_qs_0 = vld1q_s8(b_qs_base); // "ldr q3, [x28, #0x0]\n"
                int8x16_t b_qs_1 = vld1q_s8(b_qs_base + 0x10); // "ldr q22, [x28, #0x10]\n"
                int8x16_t b_qs_2 = vld1q_s8(b_qs_base + 0x20); // "ldr q27, [x28, #0x20]\n"
                int8x16_t b_qs_3 = vld1q_s8(b_qs_base + 0x30); // "ldr q30, [x28, #0x30]\n"
                // Load Q4_0 scales (d) for this block
                // x20 = x28 - 8 -> pointer to b_ptr->d
                float16x4_t b_d_f16 = vld1_f16((const float16_t *)(b_qs_base - 8)); // "sub x20, x28, #0x8\n", "ldr d17, [x20, #0x0]\n"
                float32x4_t b_d  = vcvt_f32_f16(b_d_f16);  // "fcvtl v17.4s, v17.4h\n"
                b_qs_base += sizeof(block_q4_0x4); // "add x28, x28, #0x48\n"
                // Unpack Q4_0 qs[0] (b_qs_0)
                // (low nibbles shifted)
                int8x16_t b_qs_0_low = b_qs_0 << 4; // "sshl v12.16b, v3.16b, v28.16b\n"
                int8x16_t b_qs_1_low = b_qs_1 << 4; // "sshl v31.16b, v22.16b, v28.16b\n"
                int8x16_t b_qs_2_low = b_qs_2 << 4; // "sshl v6.16b, v27.16b, v28.16b\n"
                int8x16_t b_qs_3_low = b_qs_3 << 4; // "sshl v28.16b, v30.16b, v28.16b\n"
                // (high nibbles)
                int8x16_t b_qs_0_high = b_qs_0 & 0xf0U; // "and v3.16b, v3.16b, v24.16b\n"
                int8x16_t b_qs_1_high = b_qs_1 & 0xf0U; // "and v22.16b, v22.16b, v24.16b\n"
                int8x16_t b_qs_2_high = b_qs_2 & 0xf0U; // "and v27.16b, v27.16b, v24.16b\n"
                int8x16_t b_qs_3_high = b_qs_3 & 0xf0U; // "and v30.16b, v30.16b, v24.16b\n"

                // --- Q8_0 Rows 0-3 (a_ptr0) ---
                int8x16_t a_qs_0 = vld1q_s8(a_qs_base0); // "ldr q31, [x25, #0x0]\n"
                int8x16_t a_qs_1 = vld1q_s8(a_qs_base0 + 0x10); // "ldr q6,  [x25, #0x10]\n"
                int8x16_t a_qs_2 = vld1q_s8(a_qs_base0 + 0x20); // "ldr q24, [x25, #0x20]\n"
                int8x16_t a_qs_3 = vld1q_s8(a_qs_base0 + 0x30); // "ldr q24, [x25, #0x30]\n"
                int8x16_t a_qs_4 = vld1q_s8(a_qs_base0 + 0x40); // "ldr q24, [x25, #0x40]\n"
                int8x16_t a_qs_5 = vld1q_s8(a_qs_base0 + 0x50); // "ldr q24, [x25, #0x50]\n"
                int8x16_t a_qs_6 = vld1q_s8(a_qs_base0 + 0x60); // "ldr q24, [x25, #0x60]\n"
                int8x16_t a_qs_7 = vld1q_s8(a_qs_base0 + 0x70); // "ldr q24, [x25, #0x70]\n"
                float16x4_t a0_d_f16 = vld1_f16((const float16_t *)(a_qs_base0 - 8)); // "ldr d2, [x25, #-0x8]\n"
                float32x4_t a0_d = vcvt_f32_f16(a0_d_f16); // "fcvtl v2.4s, v2.4h\n"
                a_qs_base0 += sizeof(block_q8_0x4); // "add x25, x25, #0x88\n"

                // int8x16_t v28_const = vdupq_n_s8(4); // "movi v28.16b, #0x4\n"
                // --- Compute Row Group 0 (Rows 0-3) ---
                int32x4_t iacc0 = vdupq_n_s32(0); // "movi v10.4s, #0x0\n"
                int32x4_t iacc1 = vdupq_n_s32(0); // "movi v29.4s, #0x0\n"
                int32x4_t iacc2 = vdupq_n_s32(0); // "movi v9.4s, #0x0\n"
                int32x4_t iacc3 = vdupq_n_s32(0); // "movi v20.4s, #0x0\n"

                // Dot products for qs[0] of B with qs[0..3] of A (Row 0)
                // Using b_qs_0_low
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_low, a_qs_0, 0); // sdot v10.4s, v12.16b, v31.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_low, a_qs_0, 1); // sdot v29.4s, v12.16b, v31.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_low, a_qs_0, 2); // sdot v9.4s,  v12.16b, v31.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_low, a_qs_0, 3); // sdot v20.4s, v12.16b, v31.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_low, a_qs_1, 0); // sdot v10.4s, v31.16b, v6.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_low, a_qs_1, 1); // sdot v29.4s, v31.16b, v6.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_low, a_qs_1, 2); // sdot v9.4s,  v31.16b, v6.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_low, a_qs_1, 3); // sdot v20.4s, v31.16b, v6.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_low, a_qs_2, 0); // sdot v10.4s, v6.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_low, a_qs_2, 1); // sdot v29.4s, v6.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_low, a_qs_2, 2); // sdot v9.4s,  v6.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_low, a_qs_2, 3); // sdot v20.4s, v6.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_low, a_qs_3, 0); // sdot v10.4s, v28.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_low, a_qs_3, 1); // sdot v29.4s, v28.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_low, a_qs_3, 2); // sdot v9.4s,  v28.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_low, a_qs_3, 3); // sdot v20.4s, v28.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_high, a_qs_4, 0); // sdot v10.4s, v3.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_high, a_qs_4, 1); // sdot v29.4s, v3.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_high, a_qs_4, 2); // sdot v9.4s,  v3.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_high, a_qs_4, 3); // sdot v20.4s, v3.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_high, a_qs_5, 0); // sdot v10.4s, v22.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_high, a_qs_5, 1); // sdot v29.4s, v22.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_high, a_qs_5, 2); // sdot v9.4s,  v22.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_high, a_qs_5, 3); // sdot v20.4s, v22.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_high, a_qs_6, 0); // sdot v10.4s, v27.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_high, a_qs_6, 1); // sdot v29.4s, v27.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_high, a_qs_6, 2); // sdot v9.4s,  v27.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_high, a_qs_6, 3); // sdot v20.4s, v27.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_high, a_qs_7, 0); // sdot v10.4s, v30.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_high, a_qs_7, 1); // sdot v29.4s, v30.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_high, a_qs_7, 2); // sdot v9.4s,  v30.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_high, a_qs_7, 3); // sdot v20.4s, v30.16b, v24.4b[3]\n"

                // fmul v24 (scale), scvtf, fmla
                float32x4_t scale_0 = vmulq_laneq_f32(b_d, a0_d, 0); // "fmul v24.4s, v17.4s, v2.s[0]\n"
                float32x4_t scale_1 = vmulq_laneq_f32(b_d, a0_d, 1); // "fmul v10.4s, v17.4s, v2.s[1]\n"
                float32x4_t scale_2 = vmulq_laneq_f32(b_d, a0_d, 2); // "fmul v29.4s, v17.4s, v2.s[2]\n"
                float32x4_t scale_3 = vmulq_laneq_f32(b_d, a0_d, 3); // "fmul v2.4s, v17.4s, v2.s[3]\n"
                facc0 = vfmaq_f32(facc0, vcvtq_n_f32_s32(iacc0, 4), scale_0); // "scvtf v10.4s, v10.4s, #0x4\n", "fmla v15.4s, v10.4s, v24.4s\n"
                facc1 = vfmaq_f32(facc1, vcvtq_n_f32_s32(iacc1, 4), scale_1); // "scvtf v29.4s, v29.4s, #0x4\n", "fmla v19.4s, v29.4s, v10.4s\n"
                facc2 = vfmaq_f32(facc2, vcvtq_n_f32_s32(iacc2, 4), scale_2); // "fmla v18.4s, v9.4s, v29.4s\n"
                facc3 = vfmaq_f32(facc3, vcvtq_n_f32_s32(iacc3, 4), scale_3); // "scvtf v20.4s, v20.4s, #0x4\n", "fmla v14.4s, v20.4s, v2.4s\n"

                a_qs_0 = vld1q_s8(a_qs_base1); // "ldr q24, [x23, #0x0]\n"
                a_qs_1 = vld1q_s8(a_qs_base1 + 0x10); // "ldr q10, [x23, #0x10]\n"
                a_qs_2 = vld1q_s8(a_qs_base1 + 0x20); // "ldr q24, [x23, #0x20]\n"
                a_qs_3 = vld1q_s8(a_qs_base1 + 0x30); // "ldr q10, [x23, #0x30]\n"
                a_qs_4 = vld1q_s8(a_qs_base1 + 0x40); // "ldr q24, [x23, #0x40]\n"
                a_qs_5 = vld1q_s8(a_qs_base1 + 0x50); // "ldr q10, [x23, #0x50]\n"
                a_qs_6 = vld1q_s8(a_qs_base1 + 0x60); // "ldr q24, [x23, #0x60]\n"
                a_qs_7 = vld1q_s8(a_qs_base1 + 0x70); // "ldr q10, [x23, #0x70]\n"
                float16x4_t a1_d_f16 = vld1_f16((const float16_t *)(a_qs_base1 - 8)); // "ldr d26, [x23, #-0x8]\n"
                float32x4_t a1_d = vcvt_f32_f16(a1_d_f16); // "fcvtl v26.4s, v26.4h\n"
                a_qs_base1 += sizeof(block_q8_0x4); // "add x23, x23, #0x88\n"
                // --- Compute Row Group 1 (Rows 4-7) ---
                // Accs: v9, v29, v20, v2 (Renamed to keep flow, but logically distinct)
                // Reset v9, v29, v20, v2
                iacc0 = vdupq_n_s32(0); // "movi v9.4s, #0x0\n"
                iacc1 = vdupq_n_s32(0); // "movi v29.4s, #0x0\n"
                iacc2 = vdupq_n_s32(0); // "movi v20.4s, #0x0\n"
                iacc3 = vdupq_n_s32(0); // "movi v2.4s, #0x0\n"

                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_low, a_qs_0, 0); // sdot v9.4s, v12.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_low, a_qs_0, 1); // sdot v29.4s, v12.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_low, a_qs_0, 2); // sdot v20.4s, v12.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_low, a_qs_0, 3); // sdot v2.4s, v12.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_low, a_qs_1, 0); // sdot v9.4s, v31.16b, v10.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_low, a_qs_1, 1); // sdot v29.4s, v31.16b, v10.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_low, a_qs_1, 2); // sdot v20.4s, v31.16b, v10.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_low, a_qs_1, 3); // sdot v2.4s, v31.16b, v10.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_low, a_qs_2, 0); // sdot v9.4s, v6.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_low, a_qs_2, 1); // sdot v29.4s, v6.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_low, a_qs_2, 2); // sdot v20.4s, v6.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_low, a_qs_2, 3); // sdot v2.4s, v6.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_low, a_qs_3, 0); // sdot v9.4s, v28.16b, v10.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_low, a_qs_3, 1); // sdot v29.4s, v28.16b, v10.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_low, a_qs_3, 2); // sdot v20.4s, v28.16b, v10.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_low, a_qs_3, 3); // sdot v2.4s, v28.16b, v10.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_high, a_qs_4, 0); // sdot v9.4s, v3.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_high, a_qs_4, 1); // sdot v29.4s, v3.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_high, a_qs_4, 2); // sdot v20.4s, v3.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_high, a_qs_4, 3); // sdot v2.4s, v3.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_high, a_qs_5, 0); // sdot v9.4s, v22.16b, v10.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_high, a_qs_5, 1); // sdot v29.4s, v22.16b, v10.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_high, a_qs_5, 2); // sdot v20.4s, v22.16b, v10.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_high, a_qs_5, 3); // sdot v2.4s, v22.16b, v10.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_high, a_qs_6, 0); // sdot v9.4s, v27.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_high, a_qs_6, 1); // sdot v29.4s, v27.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_high, a_qs_6, 2); // sdot v20.4s, v27.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_high, a_qs_6, 3); // sdot v2.4s, v27.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_high, a_qs_7, 0); // sdot v9.4s, v30.16b, v10.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_high, a_qs_7, 1); // sdot v29.4s, v30.16b, v10.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_high, a_qs_7, 2); // sdot v20.4s, v30.16b, v10.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_high, a_qs_7, 3); // sdot v2.4s, v30.16b, v10.4b[3]\n"

                scale_0 = vmulq_laneq_f32(b_d, a1_d, 0); // "fmul v10.4s, v17.4s, v26.s[0]\n"
                scale_1 = vmulq_laneq_f32(b_d, a1_d, 1); // "fmul v10.4s, v17.4s, v26.s[1]\n"
                scale_2 = vmulq_laneq_f32(b_d, a1_d, 2); // "fmul v10.4s, v17.4s, v26.s[2]\n"
                scale_3 = vmulq_laneq_f32(b_d, a1_d, 3); // "fmul v10.4s, v17.4s, v26.s[3]\n"
                facc4 = vfmaq_f32(facc4, vcvtq_n_f32_s32(iacc0, 4), scale_0); // "scvtf v9.4s, v9.4s, #0x4\n", "fmla v11.4s, v9.4s, v10.4s\n"
                facc5 = vfmaq_f32(facc5, vcvtq_n_f32_s32(iacc1, 4), scale_1); // "scvtf v29.4s, v29.4s, #0x4\n", "fmla v13.4s, v29.4s, v10.4s\n"
                facc6 = vfmaq_f32(facc6, vcvtq_n_f32_s32(iacc2, 4), scale_2); // "scvtf v20.4s, v20.4s, #0x4\n", "fmla v23.4s, v20.4s, v10.4s\n"
                facc7 = vfmaq_f32(facc7, vcvtq_n_f32_s32(iacc3, 4), scale_3); // "scvtf v2.4s, v2.4s, #0x4\n", "fmla v16.4s, v2.4s, v10.4s\n"

                a_qs_0 = vld1q_s8(a_qs_base2); // "ldr q24, [x22, #0x0]\n"
                a_qs_1 = vld1q_s8(a_qs_base2 + 0x10); // "ldr q9, [x22, #0x10]\n"
                a_qs_2 = vld1q_s8(a_qs_base2 + 0x20); // "ldr q24, [x22, #0x20]\n"
                a_qs_3 = vld1q_s8(a_qs_base2 + 0x30); // "ldr q9, [x22, #0x30]\n"
                a_qs_4 = vld1q_s8(a_qs_base2 + 0x40); // "ldr q9, [x22, #0x30]\n"
                a_qs_5 = vld1q_s8(a_qs_base2 + 0x50); // "ldr q9, [x22, #0x30]\n"
                a_qs_6 = vld1q_s8(a_qs_base2 + 0x60); // "ldr q24, [x22, #0x60]\n"
                a_qs_7 = vld1q_s8(a_qs_base2 + 0x70); // "ldr q9, [x22, #0x70]\n"
                float16x4_t a2_d_f16 = vld1_f16((const float16_t *)(a_qs_base2 - 8)); // "ldr d29, [x22, #-0x8]\n"
                float32x4_t a2_d = vcvt_f32_f16(a2_d_f16); // "fcvtl v29.4s, v29.4h\n"
                a_qs_base2 += sizeof(block_q8_0x4); // "add x22, x22, #0x88\n"

                iacc0 = vdupq_n_s32(0); // "movi v20.4s, #0x0\n"
                iacc1 = vdupq_n_s32(0); // "movi v10.4s, #0x0\n"
                iacc2 = vdupq_n_s32(0); // "movi v26.4s, #0x0\n"
                iacc3 = vdupq_n_s32(0); // "movi v2.4s, #0x0\n"

                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_low, a_qs_0, 0); // sdot v20.4s, v12.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_low, a_qs_0, 1); // sdot v10.4s, v12.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_low, a_qs_0, 2); // sdot v26.4s, v12.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_low, a_qs_0, 3); // sdot v2.4s, v12.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_low, a_qs_1, 0); // sdot v20.4s, v31.16b, v9.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_low, a_qs_1, 1); // sdot v10.4s, v31.16b, v9.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_low, a_qs_1, 2); // sdot v26.4s, v31.16b, v9.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_low, a_qs_1, 3); // sdot v2.4s, v31.16b, v9.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_low, a_qs_2, 0); // sdot v20.4s, v6.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_low, a_qs_2, 1); // sdot v10.4s, v6.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_low, a_qs_2, 2); // sdot v26.4s, v6.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_low, a_qs_2, 3); // sdot v2.4s, v6.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_low, a_qs_3, 0); // sdot v20.4s, v28.16b, v9.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_low, a_qs_3, 1); // sdot v10.4s, v28.16b, v9.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_low, a_qs_3, 2); // sdot v26.4s, v28.16b, v9.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_low, a_qs_3, 3); // sdot v2.4s, v28.16b, v9.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_high, a_qs_4, 0); // sdot v20.4s, v3.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_high, a_qs_4, 1); // sdot v10.4s, v3.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_high, a_qs_4, 2); // sdot v26.4s, v3.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_high, a_qs_4, 3); // sdot v2.4s, v3.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_high, a_qs_5, 0); // sdot v20.4s, v22.16b, v9.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_high, a_qs_5, 1); // sdot v10.4s, v22.16b, v9.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_high, a_qs_5, 2); // sdot v26.4s, v22.16b, v9.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_high, a_qs_5, 3); // sdot v2.4s, v22.16b, v9.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_high, a_qs_6, 0); // sdot v20.4s, v27.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_high, a_qs_6, 1); // sdot v10.4s, v27.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_high, a_qs_6, 2); // sdot v26.4s, v27.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_high, a_qs_6, 3); // sdot v2.4s, v27.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_high, a_qs_7, 0); // sdot v20.4s, v30.16b, v9.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_high, a_qs_7, 1); // sdot v10.4s, v30.16b, v9.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_high, a_qs_7, 2); // sdot v26.4s, v30.16b, v9.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_high, a_qs_7, 3); // sdot v2.4s, v30.16b, v9.4b[3]\n"

                scale_0 = vmulq_laneq_f32(b_d, a2_d, 0); // "fmul v9.4s, v17.4s, v29.s[0]\n"
                scale_1 = vmulq_laneq_f32(b_d, a2_d, 1); // "fmul v20.4s, v17.4s, v29.s[1]\n"
                scale_2 = vmulq_laneq_f32(b_d, a2_d, 2); // "fmul v10.4s, v17.4s, v29.s[2]\n"
                scale_3 = vmulq_laneq_f32(b_d, a2_d, 3); // "fmul v29.4s, v17.4s, v29.s[3]\n"
                facc8  = vfmaq_f32(facc8,  vcvtq_n_f32_s32(iacc0, 4), scale_0); // "scvtf v20.4s, v20.4s, #0x4\n", "fmla v25.4s, v20.4s, v9.4s\n"
                facc9  = vfmaq_f32(facc9,  vcvtq_n_f32_s32(iacc1, 4), scale_1); // "scvtf v10.4s, v10.4s, #0x4\n", "fmla v7.4s, v10.4s, v20.4s\n"
                facc10 = vfmaq_f32(facc10, vcvtq_n_f32_s32(iacc2, 4), scale_2); // "scvtf v26.4s, v26.4s, #0x4\n", "fmla v0.4s, v26.4s, v10.4s\n"
                facc11 = vfmaq_f32(facc11, vcvtq_n_f32_s32(iacc3, 4), scale_3); // "scvtf v2.4s, v2.4s, #0x4\n", "fmla v4.4s, v2.4s, v29.4s\n"

                a_qs_0 = vld1q_s8(a_qs_base3); // "ldr q24, [x21, #0x0]\n"
                a_qs_1 = vld1q_s8(a_qs_base3 + 0x10); // "ldr q9, [x21, #0x10]\n"
                a_qs_2 = vld1q_s8(a_qs_base3 + 0x20); // "ldr q12, [x21, #0x20]\n"
                a_qs_3 = vld1q_s8(a_qs_base3 + 0x30); // "ldr q9, [x21, #0x30]\n"
                a_qs_4 = vld1q_s8(a_qs_base3 + 0x40); // "ldr q12, [x21, #0x40]\n"
                a_qs_5 = vld1q_s8(a_qs_base3 + 0x50); // "ldr q9, [x21, #0x50]\n"
                a_qs_6 = vld1q_s8(a_qs_base3 + 0x60); //"ldr q12, [x21, #0x60]\n"
                a_qs_7 = vld1q_s8(a_qs_base3 + 0x70); // "ldr q17, [x21, #0x70]\n"
                float16x4_t a3_d_f16 = vld1_f16((const float16_t *)(a_qs_base3 - 8)); // "ldr d20, [x21, #-0x8]\n"
                float32x4_t a3_d = vcvt_f32_f16(a3_d_f16); // "fcvtl v20.4s, v20.4h\n"
                a_qs_base3 += sizeof(block_q8_0x4); // "add x21, x21, #0x88\n"

                iacc0 = vdupq_n_s32(0); // "movi v26.4s, #0x0\n"
                iacc1 = vdupq_n_s32(0); // "movi v10.4s, #0x0\n"
                iacc2 = vdupq_n_s32(0); // "movi v2.4s, #0x0\n"
                iacc3 = vdupq_n_s32(0); // "movi v29.4s, #0x0\n"

                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_low, a_qs_0, 0); // sdot v26.4s, v12.16b, v24.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_low, a_qs_0, 1); // sdot v10.4s, v12.16b, v24.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_low, a_qs_0, 2); // sdot v2.4s, v12.16b, v24.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_low, a_qs_0, 3); // sdot v29.4s, v12.16b, v24.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_low, a_qs_1, 0); // sdot v26.4s, v31.16b, v9.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_low, a_qs_1, 1); // sdot v10.4s, v31.16b, v9.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_low, a_qs_1, 2); // sdot v2.4s, v31.16b, v9.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_low, a_qs_1, 3); // sdot v29.4s, v31.16b, v9.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_low, a_qs_2, 0); // sdot v26.4s, v6.16b, v12.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_low, a_qs_2, 1); // sdot v10.4s, v6.16b, v12.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_low, a_qs_2, 2); // sdot v2.4s, v6.16b, v12.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_low, a_qs_2, 3); // sdot v29.4s, v6.16b, v12.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_low, a_qs_3, 0); // sdot v26.4s, v28.16b, v9.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_low, a_qs_3, 1); // sdot v10.4s, v28.16b, v9.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_low, a_qs_3, 2); // sdot v2.4s, v28.16b, v9.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_low, a_qs_3, 3); // sdot v29.4s, v28.16b, v9.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_high, a_qs_4, 0); // sdot v26.4s, v3.16b, v12.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_high, a_qs_4, 1); // sdot v10.4s, v3.16b, v12.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_high, a_qs_4, 2); // sdot v2.4s, v3.16b, v12.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_high, a_qs_4, 3); // sdot v29.4s, v3.16b, v12.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_high, a_qs_5, 0); // sdot v26.4s, v22.16b, v9.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_high, a_qs_5, 1); // sdot v10.4s, v22.16b, v9.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_high, a_qs_5, 2); // sdot v2.4s, v22.16b, v9.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_high, a_qs_5, 3); // sdot v29.4s, v22.16b, v9.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_high, a_qs_6, 0); // sdot v26.4s, v27.16b, v12.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_high, a_qs_6, 1); // sdot v10.4s, v27.16b, v12.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_high, a_qs_6, 2); // sdot v2.4s, v27.16b, v12.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_high, a_qs_6, 3); // sdot v29.4s, v27.16b, v12.4b[3]\n"
                iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_high, a_qs_7, 0); // sdot v26.4s, v30.16b, v17.4b[0]\n"
                iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_high, a_qs_7, 1); // sdot v10.4s, v30.16b, v17.4b[1]\n"
                iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_high, a_qs_7, 2); // sdot v2.4s, v30.16b, v17.4b[2]\n"
                iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_high, a_qs_7, 3); // sdot v29.4s, v30.16b, v17.4b[3]\n"

                scale_0 = vmulq_laneq_f32(b_d, a3_d, 0); // "fmul v24.4s, v17.4s, v20.s[0]\n"
                scale_1 = vmulq_laneq_f32(b_d, a3_d, 1); // "fmul v24.4s, v17.4s, v20.s[1]\n"
                scale_2 = vmulq_laneq_f32(b_d, a3_d, 2); // "fmul v6.4s, v17.4s, v20.s[2]\n"
                scale_3 = vmulq_laneq_f32(b_d, a3_d, 3); // "fmul v20.4s, v17.4s, v20.s[3]\n"
                facc12 = vfmaq_f32(facc12, vcvtq_n_f32_s32(iacc0, 4), scale_0); // "scvtf v26.4s, v26.4s, #0x4\n", "fmla v5.4s, v26.4s, v24.4s\n"
                facc13 = vfmaq_f32(facc13, vcvtq_n_f32_s32(iacc1, 4), scale_1); // "scvtf v10.4s, v10.4s, #0x4\n", "fmla v21.4s, v10.4s, v31.4s\n"
                facc14 = vfmaq_f32(facc14, vcvtq_n_f32_s32(iacc2, 4), scale_2); // "scvtf v2.4s, v2.4s, #0x4\n", "fmla v8.4s, v2.4s, v6.4s\n"
                facc15 = vfmaq_f32(facc15, vcvtq_n_f32_s32(iacc3, 4), scale_3); // "scvtf v29.4s, v29.4s, #0x4\n", "fmla v1.4s, v29.4s, v20.4s\n"
            } while (k > 0); // "bgt 3b\n"

            // Store results for 16 rows
            // res_ptr_local is x20 equivalent
            float *r0 = res_ptr; // "mov x20, %x[res_ptr]\n"
            nc_remaining -= 4; // "subs x27, x27, #0x4\n"
            res_ptr += 4; // "add %x[res_ptr], %x[res_ptr], #0x10\n"
            vst1q_f32(r0, facc0); // "str q15, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc1);  // "add x20, x20, %x[res_stride]\n", "str q19, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc2);  // "add x20, x20, %x[res_stride]\n", "str q18, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc3);  // "add x20, x20, %x[res_stride]\n", "str q14, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc4);  // "add x20, x20, %x[res_stride]\n", "str q11, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc5);  // "add x20, x20, %x[res_stride]\n", "str q13, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc6);  // "add x20, x20, %x[res_stride]\n", "str q23, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc7);  // "add x20, x20, %x[res_stride]\n", "str q16, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc8);  // "add x20, x20, %x[res_stride]\n", "str q25, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc9);  // "add x20, x20, %x[res_stride]\n", "str q7,  [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc10); // "add x20, x20, %x[res_stride]\n", "str q0,  [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc11); // "add x20, x20, %x[res_stride]\n", "str q4,  [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc12); // "add x20, x20, %x[res_stride]\n", "str q5,  [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc13); // "add x20, x20, %x[res_stride]\n", "str q21, [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc14); // "add x20, x20, %x[res_stride]\n", "str q8,  [x20, #0x0]\n"
            r0 += res_stride; vst1q_f32(r0, facc15); // "add x20, x20, %x[res_stride]\n", "str q1,  [x20, #0x0]\n"
        } while (nc_remaining != 0); // "bne 2b\n"

        // Advance row pointers
        nr_remaining -= 0x10; // "sub x10, x10, #0x10\n"
        res_ptr = res_ptr_next_16; // "mov %x[res_ptr], x26\n"
        // We consumed 4 blocks of rows (each block 4 rows)
        a_ptr += 4 * nb; // "mov x20, #0x4\n", "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
    }

    // "4:"  // Row loop skip
    // Tail loop for remaining rows (1 to 15)
    if (nr_remaining != 0) { // "cbz x10, 9f\n"
        // Process in chunks of 4 rows
        do { // "5:"  // Row tail: Row loop
            const int8_t *b_qs = b_ptr->qs; // "add x24, %x[b_ptr], #0x8\n"
            int nc_remaining = nc; // "mov x23, %x[nc]\n"
            float *res_ptr_next_4 = res_ptr + (res_stride << 2); // "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"

            do { // "6:"  // Row tail: Column loop
                // Initialize accumulators for 4 rows
                float32x4_t v15 = vdupq_n_f32(0.0f); // "movi v15.16b, #0x0\n"
                float32x4_t v19 = vdupq_n_f32(0.0f); // "movi v19.16b, #0x0\n"
                float32x4_t v18 = vdupq_n_f32(0.0f); // "movi v18.16b, #0x0\n"
                float32x4_t v14 = vdupq_n_f32(0.0f); // "movi v14.16b, #0x0\n"
                const int8_t *a_qs = a_ptr->qs; // "add x25, %x[a_ptr], #0x8\n"
                int k = nb; // "mov x21, %x[nb]\n"

                do { // "7:"  // Row tail: Block loop
                    k--; // "subs x21, x21, #0x1\n"
                    int8x16_t b_qs_0 = vld1q_s8(b_qs); // "ldr q7, [x24, #0x0]\n"
                    int8x16_t b_qs_1 = vld1q_s8(b_qs + 0x10); // "ldr q3, [x24, #0x10]\n"
                    int8x16_t b_qs_2 = vld1q_s8(b_qs + 0x20); // "ldr q13, [x24, #0x20]\n"
                    int8x16_t b_qs_3 = vld1q_s8(b_qs + 0x30); // "ldr q28, [x24, #0x30]\n"

                    int8x16_t a_qs_0 = vld1q_s8(a_qs); // "ldr q5, [x25, #0x0]\n"
                    int8x16_t a_qs_1 = vld1q_s8(a_qs + 0x10); // "ldr q2, [x25, #0x10]\n"
                    int8x16_t a_qs_2 = vld1q_s8(a_qs + 0x20); // "ldr q31, [x25, #0x20]\n"
                    int8x16_t a_qs_3 = vld1q_s8(a_qs + 0x30); // "ldr q27, [x25, #0x30]\n"
                    int8x16_t a_qs_4 = vld1q_s8(a_qs + 0x40); // "ldr q26, [x25, #0x40]\n"
                    int8x16_t a_qs_5 = vld1q_s8(a_qs + 0x50); // "ldr q25, [x25, #0x50]\n"
                    int8x16_t a_qs_6 = vld1q_s8(a_qs + 0x60); // "ldr q24, [x25, #0x60]\n"
                    int8x16_t a_qs_7 = vld1q_s8(a_qs + 0x70); // "ldr q16, [x25, #0x70]\n"

                    int32x4_t iacc0 = vdupq_n_s32(0); // "movi v4.4s, #0x0\n"
                    int32x4_t iacc1 = vdupq_n_s32(0); // "movi v1.4s, #0x0\n"
                    int32x4_t iacc2 = vdupq_n_s32(0); // "movi v0.4s, #0x0\n"
                    int32x4_t iacc3 = vdupq_n_s32(0); // "movi v30.4s, #0x0\n"

                    // Unpack Q4
                    int8x16_t b_qs_0_low = b_qs_0 << 4; // "sshl v20.16b, v7.16b, v9.16b\n"
                    int8x16_t b_qs_1_low = b_qs_1 << 4; // "sshl v17.16b, v3.16b, v9.16b\n"
                    int8x16_t b_qs_2_low = b_qs_2 << 4; // "sshl v22.16b, v13.16b, v9.16b\n"
                    int8x16_t b_qs_3_low = b_qs_3 << 4; // "sshl v9.16b, v28.16b, v9.16b\n"

                    int8x16_t b_qs_0_high = b_qs_0 & 0xf0U; // "and v7.16b, v7.16b, v29.16b\n"
                    int8x16_t b_qs_1_high = b_qs_1 & 0xf0U; // "and v3.16b, v3.16b, v29.16b\n"
                    int8x16_t b_qs_2_high = b_qs_2 & 0xf0U; // "and v13.16b, v13.16b, v29.16b\n"
                    int8x16_t b_qs_3_high = b_qs_3 & 0xf0U; // "and v28.16b, v28.16b, v29.16b\n"

                    float16x4_t b_d_f16 = vld1_f16((const float16_t *)(b_qs - 8)); // "sub x20, x24, #0x8\n", "ldr d21, [x20, #0x0]\n"
                    float16x4_t a_d_f16 = vld1_f16((const float16_t *)(a_qs - 8)); // "ldr d12, [x25, #-0x8]\n"
                    a_qs += sizeof(block_q8_0x4); // "add x25, x25, #0x88\n"
                    b_qs += sizeof(block_q4_0x4); // "add x24, x24, #0x48\n"

                    float32x4_t b_d = vcvt_f32_f16(b_d_f16); // "fcvtl v21.4s, v21.4h\n"
                    float32x4_t a_d = vcvt_f32_f16(a_d_f16); // "fcvtl v12.4s, v12.4h\n"

                    // Dot with low
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_low, a_qs_0, 0); // sdot v4.4s, v20.16b, v5.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_low, a_qs_0, 1); // sdot v1.4s, v20.16b, v5.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_low, a_qs_0, 2); // sdot v0.4s, v20.16b, v5.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_low, a_qs_0, 3); // sdot v30.4s, v20.16b, v5.4b[3]\n"
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_low, a_qs_1, 0); // sdot v4.4s, v17.16b, v2.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_low, a_qs_1, 1); // sdot v1.4s, v17.16b, v2.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_low, a_qs_1, 2); // sdot v0.4s, v17.16b, v2.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_low, a_qs_1, 3); // sdot v30.4s, v17.16b, v2.4b[3]\n"
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_low, a_qs_2, 0); // sdot v4.4s, v22.16b, v31.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_low, a_qs_2, 1); // sdot v1.4s, v22.16b, v31.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_low, a_qs_2, 2); // sdot v0.4s, v22.16b, v31.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_low, a_qs_2, 3); // sdot v30.4s, v22.16b, v31.4b[3]\n"
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_low, a_qs_3, 0); // sdot v4.4s, v9.16b, v27.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_low, a_qs_3, 1); // sdot v1.4s, v9.16b, v27.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_low, a_qs_3, 2); // sdot v0.4s, v9.16b, v27.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_low, a_qs_3, 3); // sdot v30.4s, v9.16b, v27.4b[3]\n"
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_0_high, a_qs_4, 0); // sdot v4.4s, v7.16b, v26.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_0_high, a_qs_4, 1); // sdot v1.4s, v7.16b, v26.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_0_high, a_qs_4, 2); // sdot v0.4s, v7.16b, v26.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_0_high, a_qs_4, 3); // sdot v30.4s, v7.16b, v26.4b[3]\n"
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_1_high, a_qs_5, 0); // sdot v4.4s, v3.16b, v25.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_1_high, a_qs_5, 1); // sdot v1.4s, v3.16b, v25.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_1_high, a_qs_5, 2); // sdot v0.4s, v3.16b, v25.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_1_high, a_qs_5, 3); // sdot v30.4s, v3.16b, v25.4b[3]\n"
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_2_high, a_qs_6, 0); // sdot v4.4s, v13.16b, v24.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_2_high, a_qs_6, 1); // sdot v1.4s, v13.16b, v24.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_2_high, a_qs_6, 2); // sdot v0.4s, v13.16b, v24.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_2_high, a_qs_6, 3); // sdot v30.4s, v13.16b, v24.4b[3]\n"
                    iacc0 = vdotq_laneq_s32(iacc0, b_qs_3_high, a_qs_7, 0); // sdot v4.4s, v28.16b, v16.4b[0]\n"
                    iacc1 = vdotq_laneq_s32(iacc1, b_qs_3_high, a_qs_7, 1); // sdot v1.4s, v28.16b, v16.4b[1]\n"
                    iacc2 = vdotq_laneq_s32(iacc2, b_qs_3_high, a_qs_7, 2); // sdot v0.4s, v28.16b, v16.4b[2]\n"
                    iacc3 = vdotq_laneq_s32(iacc3, b_qs_3_high, a_qs_7, 3); // sdot v30.4s, v28.16b, v16.4b[3]\n"

                    // Scale and Accumulate
                    // Multiply scales: b_d * a_d lanes
                    float32x4_t scale_0 = vmulq_laneq_f32(b_d, a_d, 0); // "fmul v11.4s, v21.4s, v12.s[0]\n"
                    float32x4_t scale_1 = vmulq_laneq_f32(b_d, a_d, 1); // "fmul v23.4s, v21.4s, v12.s[1]\n"
                    float32x4_t scale_2 = vmulq_laneq_f32(b_d, a_d, 2); // "fmul v17.4s, v21.4s, v12.s[2]\n"
                    float32x4_t scale_3 = vmulq_laneq_f32(b_d, a_d, 3); // "fmul v6.4s, v21.4s, v12.s[3]\n"
                    v15 = vfmaq_f32(v15, vcvtq_n_f32_s32(iacc0, 4), scale_0); // "scvtf v4.4s, v4.4s, #0x4\n", "fmla v15.4s, v4.4s, v11.4s\n"
                    v19 = vfmaq_f32(v19, vcvtq_n_f32_s32(iacc1, 4), scale_1); // "scvtf v1.4s, v1.4s, #0x4\n", "fmla v19.4s, v1.4s, v23.4s\n"
                    v18 = vfmaq_f32(v18, vcvtq_n_f32_s32(iacc2, 4), scale_2); // "scvtf v0.4s, v0.4s, #0x4\n", "fmla v18.4s, v0.4s, v17.4s\n"
                    v14 = vfmaq_f32(v14, vcvtq_n_f32_s32(iacc3, 4), scale_3); // "scvtf v30.4s, v30.4s, #0x4\n", "fmla v14.4s, v30.4s, v6.4s\n"
                } while (k > 0); // "bgt 7b\n"

                // Store partial results
                float *r = res_ptr; // "mov x20, %x[res_ptr]\n"
                vst1q_f32(r, v15); // "str q15, [x20, #0x0]\n"
                r += res_stride; vst1q_f32(r, v19); // "add x20, x20, %x[res_stride]\n", "str q19, [x20, #0x0]\n"
                r += res_stride; vst1q_f32(r, v18); // "add x20, x20, %x[res_stride]\n", "str q18, [x20, #0x0]\n"
                r += res_stride; vst1q_f32(r, v14); // "add x20, x20, %x[res_stride]\n", "str q14, [x20, #0x0]\n"
                // "8:"  // Row tail: Accumulator store skip
                nc_remaining -= 4; // "subs x23, x23, #0x4\n"
                res_ptr += 4; // "add %x[res_ptr], %x[res_ptr], #0x10\n"
            } while (nc_remaining != 0); // "bne 6b\n"

            // Tail loop row update
            nr_remaining -= 4; // "subs x10, x10, #0x4\n"
            a_ptr += nb; // "add %x[a_ptr], %x[a_ptr], x9\n"
            res_ptr = res_ptr_next_4; // "mov %x[res_ptr], x22\n"
        } while (nr_remaining > 0); // "bgt 5b\n"
    }
#else
    const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy;
    for (int r = 0; r < nr; r += 4) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;
        const block_q8_0x4 * a_base = a_ptr;
        for (int c = 0; c < nc; c += ncols_interleaved) {
            a_ptr = a_base;

            float32x4_t acc0 = vdupq_n_f32(0);
            float32x4_t acc1 = vdupq_n_f32(0);
            float32x4_t acc2 = vdupq_n_f32(0);
            float32x4_t acc3 = vdupq_n_f32(0);

            for (int b = 0; b < nb; b++) {
                float32x4_t ad = vcvt_f32_f16(vld1_f16((const __fp16 *)a_ptr->d));
                float32x4_t bd = vcvt_f32_f16(vld1_f16((const __fp16 *)b_ptr->d));

                int32x4_t ret0 = vdupq_n_s32(0);
                int32x4_t ret1 = vdupq_n_s32(0);
                int32x4_t ret2 = vdupq_n_s32(0);
                int32x4_t ret3 = vdupq_n_s32(0);

                for (int k = 0; k < 4; k++) {
                    int8x16_t b0 = vld1q_s8((const int8_t *)b_ptr->qs + 16 * k);
                    int8x16_t b1 = b0 & 0xf0U;
                    b0 <<= 4;

                    int8x16_t a0 = vld1q_s8(a_ptr->qs + 16 * k);
                    int8x16_t a1 = vld1q_s8(a_ptr->qs + 16 * k + 4 * qk/2);

                    ret0 = vdotq_laneq_s32(ret0, b0, a0, 0);
                    ret1 = vdotq_laneq_s32(ret1, b0, a0, 1);
                    ret2 = vdotq_laneq_s32(ret2, b0, a0, 2);
                    ret3 = vdotq_laneq_s32(ret3, b0, a0, 3);

                    ret0 = vdotq_laneq_s32(ret0, b1, a1, 0);
                    ret1 = vdotq_laneq_s32(ret1, b1, a1, 1);
                    ret2 = vdotq_laneq_s32(ret2, b1, a1, 2);
                    ret3 = vdotq_laneq_s32(ret3, b1, a1, 3);
                }
                acc0 = vfmaq_f32(acc0, vcvtq_n_f32_s32(ret0, 4), vmulq_laneq_f32(bd, ad, 0));
                acc1 = vfmaq_f32(acc1, vcvtq_n_f32_s32(ret1, 4), vmulq_laneq_f32(bd, ad, 1));
                acc2 = vfmaq_f32(acc2, vcvtq_n_f32_s32(ret2, 4), vmulq_laneq_f32(bd, ad, 2));
                acc3 = vfmaq_f32(acc3, vcvtq_n_f32_s32(ret3, 4), vmulq_laneq_f32(bd, ad, 3));
                a_ptr++;
                b_ptr++;
            }
            vst1q_f32(s + (r + 0) * bs + c, acc0);
            vst1q_f32(s + (r + 1) * bs + c, acc1);
            vst1q_f32(s + (r + 2) * bs + c, acc2);
            vst1q_f32(s + (r + 3) * bs + c, acc3);
        }
    }
#endif
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    ggml_gemm_q4_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const void * b_ptr = vx;
    const void * a_ptr = vy;
    float * res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
        "mov x10, %x[nr]\n"
        "mov x9, #0x88\n"
        "cmp x10, #0x10\n"
        "mul x9, %x[nb], x9\n"
        "blt 4f\n"
        "1:"  // Row loop
        "add x28, %x[b_ptr], #0x8\n"
        "mov x27, %x[nc]\n"
        "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
        "2:"  // Column loop
        "add x25, %x[a_ptr], #0x8\n"
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "mov x24, %x[nb]\n"
        "add x23, x25, x9\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "add x22, x23, x9\n"
        "movi v11.16b, #0x0\n"
        "movi v13.16b, #0x0\n"
        "add x21, x22, x9\n"
        "movi v22.16b, #0x0\n"
        "movi v23.16b, #0x0\n"
        "movi v25.16b, #0x0\n"
        "movi v5.16b, #0x0\n"
        "movi v7.16b, #0x0\n"
        "movi v4.16b, #0x0\n"
        "movi v6.16b, #0x0\n"
        "movi v30.16b, #0x0\n"
        "movi v24.16b, #0x0\n"
        "movi v14.16b, #0x0\n"
        "3:"  // Block loop
        "ldr q21, [x28, #0x0]\n"
        "ldr q16, [x28, #0x10]\n"
        "movi v1.16b, #0x4\n"
        "movi v19.4s, #0x0\n"
        "ldr q27, [x25, #0x0]\n"
        "ldr q15, [x25, #0x10]\n"
        "movi v26.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        "ldr q29, [x28, #0x20]\n"
        "ldr q3, [x28, #0x30]\n"
        "movi v17.4s, #0x0\n"
        "movi v0.16b, #0xf0\n"
        "ldr d20, [x25, #-0x8]\n"
        "ldr d9, [x23, #-0x8]\n"
        "sshl v8.16b, v21.16b, v1.16b\n"
        "sshl v31.16b, v16.16b, v1.16b\n"
        "and v21.16b, v21.16b, v0.16b\n"
        "and v16.16b, v16.16b, v0.16b\n"
        "sub x20, x28, #0x8\n"
        "subs x24, x24, #0x1\n"
        "add x28, x28, #0x48\n"
        ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
        ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
        "ldr q27, [x25, #0x20]\n"
        ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
        ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
        "sshl v15.16b, v29.16b, v1.16b\n"
        "sshl v1.16b, v3.16b, v1.16b\n"
        "and v29.16b, v29.16b, v0.16b\n"
        "and v3.16b, v3.16b, v0.16b\n"
        "ldr q0, [x25, #0x30]\n"
        "fcvtl v20.4s, v20.4h\n"
        ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
        "fcvtl v9.4s, v9.4h\n"
        ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
        "ldr q27, [x25, #0x40]\n"
        ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
        ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
        "ldr q0, [x25, #0x50]\n"
        ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
        ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
        "ldr q27, [x25, #0x60]\n"
        ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
        ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
        "ldr q0, [x25, #0x70]\n"
        "add x25, x25, #0x88\n"
        ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
        ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
        "ldr d27, [x20, #0x0]\n"
        ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
        ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
        "fcvtl v27.4s, v27.4h\n"
        "uzp1 v0.2d, v19.2d, v26.2d\n"
        "uzp2 v26.2d, v19.2d, v26.2d\n"
        "fmul v19.4s, v27.4s, v20.s[0]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v26.4s, v26.4s, #0x4\n"
        "fmla v2.4s, v0.4s, v19.4s\n"
        "ldr q19, [x23, #0x0]\n"
        "uzp1 v0.2d, v18.2d, v17.2d\n"
        "uzp2 v18.2d, v18.2d, v17.2d\n"
        "fmul v17.4s, v27.4s, v20.s[1]\n"
        "scvtf v0.4s, v0.4s, #0x4\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v10.4s, v26.4s, v17.4s\n"
        "ldr q17, [x23, #0x10]\n"
        "fmul v26.4s, v27.4s, v20.s[2]\n"
        "fmul v20.4s, v27.4s, v20.s[3]\n"
        "fmla v12.4s, v0.4s, v26.4s\n"
        "ldr d0, [x22, #-0x8]\n"
        "ldr d26, [x21, #-0x8]\n"
        "fcvtl v0.4s, v0.4h\n"
        "fmla v28.4s, v18.4s, v20.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x23, #0x20]\n"
        "fcvtl v26.4s, v26.4h\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x23, #0x40]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q19, [x23, #0x60]\n"
        ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
        ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
        "uzp1 v19.2d, v20.2d, v18.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp2 v20.2d, v20.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v9.s[0]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v11.4s, v19.4s, v18.4s\n"
        "ldr q18, [x22, #0x0]\n"
        "fmul v19.4s, v27.4s, v9.s[1]\n"
        "fmla v13.4s, v20.4s, v19.4s\n"
        "movi v19.4s, #0x0\n"
        "movi v20.4s, #0x0\n"
        ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
        ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
        "ldr q17, [x23, #0x30]\n"
        ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
        "ldr q17, [x23, #0x50]\n"
        ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
        "ldr q17, [x23, #0x70]\n"
        "add x23, x23, #0x88\n"
        ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v9.s[2]\n"
        "fmul v9.4s, v27.4s, v9.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v22.4s, v17.4s, v19.4s\n"
        "ldr q17, [x22, #0x10]\n"
        "movi v19.4s, #0x0\n"
        ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
        "fmla v23.4s, v20.4s, v9.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v9.4s, #0x0\n"
        ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
        "ldr q18, [x22, #0x20]\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
        ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
        "ldr q18, [x22, #0x40]\n"
        ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
        ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
        "ldr q18, [x22, #0x60]\n"
        ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
        ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
        "ldr q17, [x22, #0x30]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
        "ldr q17, [x22, #0x50]\n"
        ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
        ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
        "ldr q17, [x22, #0x70]\n"
        "add x22, x22, #0x88\n"
        ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
        ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
        "uzp1 v17.2d, v19.2d, v20.2d\n"
        "uzp2 v20.2d, v19.2d, v20.2d\n"
        "fmul v19.4s, v27.4s, v0.s[0]\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "fmla v25.4s, v17.4s, v19.4s\n"
        "ldr q19, [x21, #0x0]\n"
        "fmul v17.4s, v27.4s, v0.s[1]\n"
        "fmla v5.4s, v20.4s, v17.4s\n"
        "ldr q17, [x21, #0x10]\n"
        "uzp1 v20.2d, v9.2d, v18.2d\n"
        "uzp2 v9.2d, v9.2d, v18.2d\n"
        "fmul v18.4s, v27.4s, v0.s[2]\n"
        "fmul v0.4s, v27.4s, v0.s[3]\n"
        "scvtf v20.4s, v20.4s, #0x4\n"
        "scvtf v9.4s, v9.4s, #0x4\n"
        "fmla v7.4s, v20.4s, v18.4s\n"
        "movi v20.4s, #0x0\n"
        "movi v18.4s, #0x0\n"
        ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
        ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
        "ldr q19, [x21, #0x20]\n"
        "fmla v4.4s, v9.4s, v0.4s\n"
        "movi v9.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
        "fmul v8.4s, v27.4s, v26.s[0]\n"
        ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
        "ldr q17, [x21, #0x30]\n"
        ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
        "fmul v31.4s, v27.4s, v26.s[1]\n"
        ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
        "ldr q19, [x21, #0x40]\n"
        ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
        "fmul v15.4s, v27.4s, v26.s[2]\n"
        "fmul v27.4s, v27.4s, v26.s[3]\n"
        ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
        "ldr q1, [x21, #0x50]\n"
        ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
        ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
        "ldr q26, [x21, #0x60]\n"
        ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
        ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
        "ldr q21, [x21, #0x70]\n"
        "add x21, x21, #0x88\n"
        ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
        ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
        ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
        ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
        "uzp1 v29.2d, v20.2d, v18.2d\n"
        "uzp2 v21.2d, v20.2d, v18.2d\n"
        "scvtf v29.4s, v29.4s, #0x4\n"
        "uzp1 v18.2d, v9.2d, v0.2d\n"
        "uzp2 v16.2d, v9.2d, v0.2d\n"
        "scvtf v21.4s, v21.4s, #0x4\n"
        "fmla v6.4s, v29.4s, v8.4s\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v30.4s, v21.4s, v31.4s\n"
        "fmla v24.4s, v18.4s, v15.4s\n"
        "fmla v14.4s, v16.4s, v27.4s\n"
        "bgt 3b\n"
        "mov x20, %x[res_ptr]\n"
        "subs x27, x27, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q28, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q11, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q13, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q22, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q23, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q25, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q5, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q7, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q4, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q6, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q30, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q24, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "str q14, [x20, #0x0]\n"
        "bne 2b\n"
        "mov x20, #0x4\n"
        "sub x10, x10, #0x10\n"
        "cmp x10, #0x10\n"
        "mov %x[res_ptr], x26\n"
        "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
        "bge 1b\n"
        "4:"  // Row loop skip
        "cbz x10, 9f\n"
        "5:"  // Row tail: Row loop
        "add x24, %x[b_ptr], #0x8\n"
        "mov x23, %x[nc]\n"
        "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
        "6:"  // Row tail: Column loop
        "movi v2.16b, #0x0\n"
        "movi v10.16b, #0x0\n"
        "add x25, %x[a_ptr], #0x8\n"
        "mov x21, %x[nb]\n"
        "movi v12.16b, #0x0\n"
        "movi v28.16b, #0x0\n"
        "7:"  // Row tail: Block loop
        "ldr q6, [x24, #0x0]\n"
        "ldr q5, [x24, #0x10]\n"
        "movi v17.16b, #0x4\n"
        "movi v8.4s, #0x0\n"
        "ldr q4, [x25, #0x0]\n"
        "ldr q13, [x25, #0x10]\n"
        "movi v27.4s, #0x0\n"
        "movi v0.4s, #0x0\n"
        "ldr q31, [x24, #0x20]\n"
        "ldr q14, [x24, #0x30]\n"
        "movi v29.4s, #0x0\n"
        "movi v22.16b, #0xf0\n"
        "ldr q11, [x25, #0x20]\n"
        "ldr q23, [x25, #0x30]\n"
        "sshl v21.16b, v6.16b, v17.16b\n"
        "sshl v16.16b, v5.16b, v17.16b\n"
        "ldr q20, [x25, #0x40]\n"
        "ldr q26, [x25, #0x50]\n"
        "and v6.16b, v6.16b, v22.16b\n"
        "and v5.16b, v5.16b, v22.16b\n"
        "ldr q25, [x25, #0x60]\n"
        "ldr q3, [x25, #0x70]\n"
        "sshl v19.16b, v31.16b, v17.16b\n"
        "sshl v18.16b, v14.16b, v17.16b\n"
        "ldr d17, [x25, #-0x8]\n"
        ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
        ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
        "and v31.16b, v31.16b, v22.16b\n"
        ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
        ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
        "and v14.16b, v14.16b, v22.16b\n"
        "sub x20, x24, #0x8\n"
        "ldr d16, [x20, #0x0]\n"
        "subs x21, x21, #0x1\n"
        "add x25, x25, #0x88\n"
        "fcvtl v17.4s, v17.4h\n"
        "add x24, x24, #0x48\n"
        ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
        ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
        ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
        ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
        "fcvtl v16.4s, v16.4h\n"
        ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
        ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
        "fmul v23.4s, v16.4s, v17.s[0]\n"
        "fmul v21.4s, v16.4s, v17.s[1]\n"
        "fmul v1.4s, v16.4s, v17.s[2]\n"
        "fmul v20.4s, v16.4s, v17.s[3]\n"
        ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
        ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
        ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
        ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
        ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
        ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
        "uzp1 v19.2d, v8.2d, v27.2d\n"
        "uzp2 v18.2d, v8.2d, v27.2d\n"
        "scvtf v19.4s, v19.4s, #0x4\n"
        "uzp1 v17.2d, v0.2d, v29.2d\n"
        "uzp2 v16.2d, v0.2d, v29.2d\n"
        "scvtf v18.4s, v18.4s, #0x4\n"
        "fmla v2.4s, v19.4s, v23.4s\n"
        "scvtf v17.4s, v17.4s, #0x4\n"
        "scvtf v16.4s, v16.4s, #0x4\n"
        "fmla v10.4s, v18.4s, v21.4s\n"
        "fmla v12.4s, v17.4s, v1.4s\n"
        "fmla v28.4s, v16.4s, v20.4s\n"
        "bgt 7b\n"
        "mov x20, %x[res_ptr]\n"
        "cmp x10, #0x1\n"
        "str q2, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x2\n"
        "str q10, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "cmp x10, #0x3\n"
        "str q12, [x20, #0x0]\n"
        "add x20, x20, %x[res_stride]\n"
        "ble 8f\n"
        "str q28, [x20, #0x0]\n"
        "8:"  // Row tail: Accumulator store skip
        "subs x23, x23, #0x4\n"
        "add %x[res_ptr], %x[res_ptr], #0x10\n"
        "bne 6b\n"
        "subs x10, x10, #0x4\n"
        "add %x[a_ptr], %x[a_ptr], x9\n"
        "mov %x[res_ptr], x22\n"
        "bgt 5b\n"
        "9:"  // Row tail: Row loop skip
        : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
        : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    ggml_gemm_q4_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

#if USE_ZYK
void ggml_gemm_q4_0_1x4_q8_0(int n, float * GGML_RESTRICT s, size_t ix, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    UNUSED(n);
    UNUSED(s);
    UNUSED(ix);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int nrc_x = nc / (4 * N32B);
    if (nrc_x > 2048/N32B) nrc_x /= 4;
    const size_t nb = n / QK8_0;
    DataInfo info{s, (const char *)vy, (size_t) nc, nb*sizeof(block_q8_0), /*cur_y*/ 0, 1, /*row_mapping*/ nullptr, 0};
    for (int iy = 0; iy < nr/8; ++iy) {
        mul_mat_q4_t8_q8_0(n, vx, ix, &info, nrc_x);
        // mul_mat_q4_t_q8_0<ZykQ4_0_T, 8>(n, vx, ix, &info, nrc_x);
        info.cur_y += 8;
    }
    const int rem = nr % 8;
    if (rem) {
        q4_0_t_funcs[rem-1](n, vx, ix, &info, nrc_x);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
}

void ggml_gemv_q4_0_1x4_q8_0(int n, float * GGML_RESTRICT s, size_t ix, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    UNUSED(n);
    UNUSED(s);
    UNUSED(ix);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    DataInfo* info = (DataInfo *)s;
    int nrc_x = nc / N32B;
    for (int iy = 0; iy < nr/8; ++iy) {
        mul_mat_q4_t_q8_0<ZykQ4_0_T, 8>(n, vx, ix, info, nrc_x);
        info->cur_y += 8;
    }
    const int rem = nr % 8;
    if (rem) {
        q4_0_t_funcs[rem-1](n, vx, ix, info, nrc_x);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
}

void ggml_gemm_q4_0_1x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    UNUSED(n);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const int nb = n / QK8_0;
    DataInfo info{s, (const char *)vy, bs, nb*sizeof(block_q8_0), /*cur_y*/ 0, 1, /*row_mapping*/ nullptr, 0};
    for (int iy = 0; iy < nr/8; ++iy) {
        q4_0_funcs[7](n, vx, nb*sizeof(block_q4_0), info, nc);
        info.cur_y += 8;
    }
    const int rem = nr % 8;
    if (rem) {
        q4_0_funcs[rem-1](n, vx, nb*sizeof(block_q4_0), info, nc);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
}

void ggml_gemm_mxfp4_1x4_q8_0(int n, float * GGML_RESTRICT s, size_t ix, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    UNUSED(n);
    UNUSED(s);
    UNUSED(ix);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    int nrc_x = nc / (4 * N32B);
    if (nrc_x > 2048/N32B) nrc_x /= 4;
    const size_t nb = n / QK8_0;
    DataInfo info{s, (const char *)vy, (size_t) nc, nb*sizeof(block_q8_0), /*cur_y*/ 0, 1, /*row_mapping*/ nullptr, 0};
    for (int iy = 0; iy < nr/8; ++iy) {
        mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 8>(n, vx, ix, &info, nrc_x);
        info.cur_y += 8;
    }
    const int rem = nr % 8;
    if (rem) {
        mxfp4_t_funcs[rem-1](n, vx, ix, &info, nrc_x);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
}

void ggml_gemv_mxfp4_1x4_q8_0(int n, float * GGML_RESTRICT s, size_t ix, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    UNUSED(n);
    UNUSED(s);
    UNUSED(ix);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    DataInfo* info = (DataInfo *)s;
    int nrc_x = nc / N32B;
    for (int iy = 0; iy < nr/8; ++iy) {
        mul_mat_q4_t_q8_0<Zyk_MXFP4_T, 8>(n, vx, ix, info, nrc_x);
        info->cur_y += 8;
    }
    const int rem = nr % 8;
    if (rem) {
        mxfp4_t_funcs[rem-1](n, vx, ix, info, nrc_x);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
}
#endif

#if USE_IQK
void ggml_gemm_q4_0_8x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    UNUSED(n);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int nb = n / QK8_0;
    DataInfo info{s, (const char *)vy, bs, nb*sizeof(block_q8_0), /*cur_y*/ 0, 1, /*row_mapping*/ nullptr, 0};
    for (int iy = 0; iy < nr/8; ++iy) {
        mul_mat_qx_r8_q8_0<8>(n, vx, nb*sizeof(block_q4_0), &info, nc);
        info.cur_y += 8;
    }
    const int rem = nr % 8;
    if (rem) {
        q4_0_funcs[rem-1](n, vx, nb*sizeof(block_q4_0), &info, nc);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
}

void ggml_gemv_q4_0_8x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    UNUSED(n);
    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    DataInfo* info = (DataInfo *)s;
    for (int iy = 0; iy < nr/8; ++iy) {
        mul_mat_qx_r8_q8_0<8>(n, vx, bs, info, nc);
        info->cur_y += 8;
    }
    const int rem = nr % 8;
    if (rem) {
        q4_0_funcs[rem-1](n, vx, bs, info, nc);
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
}
#endif

void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (ggml_cpu_get_sve_cnt() == QK8_0) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;
        size_t res_stride = bs * sizeof(float);

        __asm__ __volatile__(
            "mov x20, #0x4\n"
            "mov x13, %x[nr]\n"
            "mov z28.s, #-0x4\n"
            "mov x12, #0x88\n"
            "ptrue p1.b\n"
            "whilelt p0.s, XZR, x20\n"
            "cmp x13, #0x10\n"
            "mul x12, %x[nb], x12\n"
            "blt 4f\n"
            "1:"  // Row loop
            "add x11, %x[b_ptr], #0x10\n"
            "mov x10, %x[nc]\n"
            "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
            "2:"  // Column loop
            "add x28, %x[a_ptr], #0x8\n"
            "mov z24.b, #0x0\n"
            "mov z15.b, #0x0\n"
            "mov x27, %x[nb]\n"
            "add x26, x28, x12\n"
            "mov z12.b, #0x0\n"
            "mov z0.b, #0x0\n"
            "add x25, x26, x12\n"
            "mov z13.b, #0x0\n"
            "mov z1.b, #0x0\n"
            "add x24, x25, x12\n"
            "mov z20.b, #0x0\n"
            "mov z25.b, #0x0\n"
            "mov z11.b, #0x0\n"
            "mov z16.b, #0x0\n"
            "mov z19.b, #0x0\n"
            "mov z26.b, #0x0\n"
            "mov z8.b, #0x0\n"
            "mov z29.b, #0x0\n"
            "mov z27.b, #0x0\n"
            "mov z10.b, #0x0\n"
            "3:"  // Block loop
            "ld1b { z30.b }, p1/Z, [x11]\n"
            "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
            "mov z18.s, #0x0\n"
            "mov z7.s, #0x0\n"
            "ld1rqb { z3.b }, p1/Z, [x28]\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
            "mov z9.s, #0x0\n"
            "mov z22.s, #0x0\n"
            "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
            "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
            "sub x20, x11, #0x10\n"
            "sub x23, x28, #0x8\n"
            "lsl z31.b, z30.b, #0x4\n"
            "lsl z6.b, z21.b, #0x4\n"
            "ld1h { z23.s }, p1/Z, [x20]\n"
            "sub x22, x26, #0x8\n"
            "and z30.b, z30.b, #0xf0\n"
            "and z21.b, z21.b, #0xf0\n"
            "sub x21, x25, #0x8\n"
            "sub x20, x24, #0x8\n"
            "lsl z14.b, z4.b, #0x4\n"
            "lsl z2.b, z17.b, #0x4\n"
            "subs x27, x27, #0x1\n"
            "add x11, x11, #0x90\n"
            ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
            ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
            "and z4.b, z4.b, #0xf0\n"
            ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
            ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
            "and z17.b, z17.b, #0xf0\n"
            "fcvt z23.s, p1/m, z23.h\n"
            ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
            ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
            ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
            ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
            "fscale z23.s, p1/m, z23.s, z28.s\n"
            ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
            ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
            ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
            ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
            "add x28, x28, #0x88\n"
            ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
            ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
            "ld1h { z3.s }, p0/Z, [x23]\n"
            ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
            ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
            "fcvt z3.s, p1/m, z3.h\n"
            "uzp1 z5.d, z18.d, z7.d\n"
            "uzp2 z18.d, z18.d, z7.d\n"
            "mov z3.q, z3.q[0]\n"
            "uzp1 z7.d, z9.d, z22.d\n"
            "uzp2 z22.d, z9.d, z22.d\n"
            "fmul z9.s, z23.s, z3.s[0]\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "scvtf z7.s, p1/m, z7.s\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z24.s, p1/M, z5.s, z9.s\n"
            "ld1rqb { z5.b }, p1/Z, [x26]\n"
            "fmul z9.s, z23.s, z3.s[1]\n"
            "fmla z15.s, p1/M, z18.s, z9.s\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
            "fmul z9.s, z23.s, z3.s[2]\n"
            "fmul z3.s, z23.s, z3.s[3]\n"
            "fmla z12.s, p1/M, z7.s, z9.s\n"
            "mov z9.s, #0x0\n"
            "ld1h { z7.s }, p0/Z, [x22]\n"
            ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
            "fmla z0.s, p1/M, z22.s, z3.s\n"
            "mov z22.s, #0x0\n"
            "ld1h { z3.s }, p0/Z, [x21]\n"
            ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
            "fcvt z7.s, p1/m, z7.h\n"
            "fcvt z3.s, p1/m, z3.h\n"
            ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
            ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
            "mov z7.q, z7.q[0]\n"
            "mov z3.q, z3.q[0]\n"
            ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
            ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
            ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
            ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
            "uzp1 z5.d, z9.d, z22.d\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "uzp2 z22.d, z9.d, z22.d\n"
            "fmul z9.s, z23.s, z7.s[0]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z13.s, p1/M, z5.s, z9.s\n"
            "ld1rqb { z9.b }, p1/Z, [x25]\n"
            "fmul z5.s, z23.s, z7.s[1]\n"
            "fmla z1.s, p1/M, z22.s, z5.s\n"
            "mov z5.s, #0x0\n"
            "mov z22.s, #0x0\n"
            ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
            ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
            ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
            ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
            ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
            ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
            "add x26, x26, #0x88\n"
            ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
            ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
            "uzp1 z18.d, z5.d, z22.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp2 z22.d, z5.d, z22.d\n"
            "fmul z5.s, z23.s, z7.s[2]\n"
            "fmul z7.s, z23.s, z7.s[3]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z20.s, p1/M, z18.s, z5.s\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
            "ld1h { z5.s }, p0/Z, [x20]\n"
            "fcvt z5.s, p1/m, z5.h\n"
            "fmla z25.s, p1/M, z22.s, z7.s\n"
            "mov z22.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
            ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
            "mov z5.q, z5.q[0]\n"
            ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
            ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
            ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
            ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
            ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
            ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
            "uzp1 z9.d, z22.d, z7.d\n"
            "scvtf z9.s, p1/m, z9.s\n"
            "uzp2 z22.d, z22.d, z7.d\n"
            "fmul z7.s, z23.s, z3.s[0]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z11.s, p1/M, z9.s, z7.s\n"
            "ld1rqb { z9.b }, p1/Z, [x24]\n"
            "fmul z7.s, z23.s, z3.s[1]\n"
            "fmla z16.s, p1/M, z22.s, z7.s\n"
            "mov z22.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
            ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
            ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
            ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
            ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
            ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
            "add x25, x25, #0x88\n"
            ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
            ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
            "uzp1 z18.d, z22.d, z7.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp2 z7.d, z22.d, z7.d\n"
            "fmul z22.s, z23.s, z3.s[2]\n"
            "fmul z3.s, z23.s, z3.s[3]\n"
            "scvtf z7.s, p1/m, z7.s\n"
            "fmla z19.s, p1/M, z18.s, z22.s\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
            "fmul z22.s, z23.s, z5.s[0]\n"
            "fmla z26.s, p1/M, z7.s, z3.s\n"
            "mov z3.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
            ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
            "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
            ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
            ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
            "mov z9.s, #0x0\n"
            ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
            "mov z31.s, #0x0\n"
            ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
            "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
            ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
            "fmul z14.s, z23.s, z5.s[1]\n"
            ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
            "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
            "fmul z2.s, z23.s, z5.s[2]\n"
            "fmul z23.s, z23.s, z5.s[3]\n"
            ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
            ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
            ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
            ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
            "add x24, x24, #0x88\n"
            ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
            ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
            ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
            ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
            "uzp1 z18.d, z3.d, z7.d\n"
            "uzp2 z5.d, z3.d, z7.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp1 z6.d, z9.d, z31.d\n"
            "uzp2 z9.d, z9.d, z31.d\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "fmla z8.s, p1/M, z18.s, z22.s\n"
            "scvtf z6.s, p1/m, z6.s\n"
            "scvtf z9.s, p1/m, z9.s\n"
            "fmla z29.s, p1/M, z5.s, z14.s\n"
            "fmla z27.s, p1/M, z6.s, z2.s\n"
            "fmla z10.s, p1/M, z9.s, z23.s\n"
            "bgt 3b\n"
            "mov x20, %x[res_ptr]\n"
            "subs x10, x10, #0x8\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "st1w { z24.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z15.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z12.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z0.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z13.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z1.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z20.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z25.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z11.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z16.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z19.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z26.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z8.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z29.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z27.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z10.s }, p1, [x20]\n"
            "bne 2b\n"
            "mov x20, #0x4\n"
            "sub x13, x13, #0x10\n"
            "cmp x13, #0x10\n"
            "mov %x[res_ptr], x9\n"
            "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
            "bge 1b\n"
            "4:"  // Row loop skip
            "cbz x13, 9f\n"
            "5:"  // Row tail: Row loop
            "add x25, %x[b_ptr], #0x10\n"
            "mov x24, %x[nc]\n"
            "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
            "6:"  // Row tail: Column loop
            "mov z24.b, #0x0\n"
            "mov z15.b, #0x0\n"
            "add x28, %x[a_ptr], #0x8\n"
            "mov x22, %x[nb]\n"
            "mov z12.b, #0x0\n"
            "mov z0.b, #0x0\n"
            "7:"  // Row tail: Block loop
            "ld1b { z3.b }, p1/Z, [x25]\n"
            "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
            "mov z2.s, #0x0\n"
            "mov z25.s, #0x0\n"
            "ld1rqb { z26.b }, p1/Z, [x28]\n"
            "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
            "mov z27.s, #0x0\n"
            "mov z19.s, #0x0\n"
            "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
            "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
            "sub x21, x25, #0x10\n"
            "sub x20, x28, #0x8\n"
            "lsl z20.b, z3.b, #0x4\n"
            "lsl z4.b, z6.b, #0x4\n"
            "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
            "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
            "and z3.b, z3.b, #0xf0\n"
            "and z6.b, z6.b, #0xf0\n"
            "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
            "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
            "lsl z8.b, z29.b, #0x4\n"
            "lsl z14.b, z16.b, #0x4\n"
            "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
            "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
            ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
            ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
            "and z29.b, z29.b, #0xf0\n"
            "ld1h { z17.s }, p1/Z, [x21]\n"
            ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
            ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
            "and z16.b, z16.b, #0xf0\n"
            "ld1h { z4.s }, p0/Z, [x20]\n"
            "subs x22, x22, #0x1\n"
            "add x28, x28, #0x88\n"
            "fcvt z17.s, p1/m, z17.h\n"
            "add x25, x25, #0x90\n"
            ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
            ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
            "fcvt z4.s, p1/m, z4.h\n"
            ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
            ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
            "fscale z17.s, p1/m, z17.s, z28.s\n"
            "mov z4.q, z4.q[0]\n"
            ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
            ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
            "fmul z23.s, z17.s, z4.s[0]\n"
            "fmul z9.s, z17.s, z4.s[1]\n"
            "fmul z21.s, z17.s, z4.s[2]\n"
            "fmul z4.s, z17.s, z4.s[3]\n"
            ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
            ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
            ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
            ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
            ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
            ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
            "uzp1 z31.d, z2.d, z25.d\n"
            "uzp2 z13.d, z2.d, z25.d\n"
            "scvtf z31.s, p1/m, z31.s\n"
            "uzp1 z17.d, z27.d, z19.d\n"
            "uzp2 z18.d, z27.d, z19.d\n"
            "scvtf z13.s, p1/m, z13.s\n"
            "fmla z24.s, p1/M, z31.s, z23.s\n"
            "scvtf z17.s, p1/m, z17.s\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "fmla z15.s, p1/M, z13.s, z9.s\n"
            "fmla z12.s, p1/M, z17.s, z21.s\n"
            "fmla z0.s, p1/M, z18.s, z4.s\n"
            "bgt 7b\n"
            "mov x20, %x[res_ptr]\n"
            "cmp x13, #0x1\n"
            "st1w { z24.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x13, #0x2\n"
            "st1w { z15.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x13, #0x3\n"
            "st1w { z12.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "st1w { z0.s }, p1, [x20]\n"
            "8:"  // Row tail: Accumulator store skip
            "subs x24, x24, #0x8\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "bne 6b\n"
            "subs x13, x13, #0x4\n"
            "add %x[a_ptr], %x[a_ptr], x12\n"
            "mov %x[res_ptr], x23\n"
            "bgt 5b\n"
            "9:"  // Row tail: Row loop skip
            : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
            : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
            : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
        );
        return;
    }
#endif // #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)

#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
    ggml_gemm_q4_0_8x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    const int8x16_t kvalues = vld1q_s8(kvalues_iq4nl);

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

            float32x4_t sumf[4];
            for (int m = 0; m < 4; m++) {
                sumf[m] = vdupq_n_f32(0);
            }

            for (int l = 0; l < nb; l++) {
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const float16_t *)a_ptr[l].d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *)b_ptr[l].d));

                int32x4_t sumi_0 = vdupq_n_s32(0);
                int32x4_t sumi_1 = vdupq_n_s32(0);
                int32x4_t sumi_2 = vdupq_n_s32(0);
                int32x4_t sumi_3 = vdupq_n_s32(0);

                for (int k = 0; k < 4; k++) {
                    int8x16_t a_0 = vld1q_s8(a_ptr[l].qs + 16 * k + 0);
                    int8x16_t a_1 = vld1q_s8(a_ptr[l].qs + 16 * k + 64);

                    uint8x16_t b = vld1q_u8(b_ptr[l].qs + 16 * k);
                    int8x16_t b_hi = vqtbl1q_s8(kvalues, b >> 4);
                    int8x16_t b_lo = vqtbl1q_s8(kvalues, b & 0xF);

                    sumi_0 = vdotq_laneq_s32(sumi_0, b_lo, a_0, 0);
                    sumi_1 = vdotq_laneq_s32(sumi_1, b_lo, a_0, 1);
                    sumi_2 = vdotq_laneq_s32(sumi_2, b_lo, a_0, 2);
                    sumi_3 = vdotq_laneq_s32(sumi_3, b_lo, a_0, 3);
                    sumi_0 = vdotq_laneq_s32(sumi_0, b_hi, a_1, 0);
                    sumi_1 = vdotq_laneq_s32(sumi_1, b_hi, a_1, 1);
                    sumi_2 = vdotq_laneq_s32(sumi_2, b_hi, a_1, 2);
                    sumi_3 = vdotq_laneq_s32(sumi_3, b_hi, a_1, 3);
                }

                sumf[0] = vmlaq_f32(sumf[0], vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(sumi_0));
                sumf[1] = vmlaq_f32(sumf[1], vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(sumi_1));
                sumf[2] = vmlaq_f32(sumf[2], vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(sumi_2));
                sumf[3] = vmlaq_f32(sumf[3], vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(sumi_3));
            }

            for (int m = 0; m < 4; m++) {
                vst1q_f32(s + (y * 4 + m) * bs + x * 4, sumf[m]);
            }
        }
    }
    return;
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    ggml_gemm_iq4_nl_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x4_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    constexpr int    q8_k_blocklen = 4;
    constexpr int    acc_size  = 2 * 4;  // 2 row pairs × 4 col pairs
    const uint8x16_t m4b       = vdupq_n_u8(0x0f);

    // 8 accumulators: 2 row pairs × 4 col pairs
    float32x4_t acc_f32[acc_size];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const block_q8_Kx4 * GGML_RESTRICT q8_ptr = (const block_q8_Kx4 *) vy + (y * nb);

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

            for (int i = 0; i < acc_size; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                // d4 0 1 2 3, 4 5 6 7
                float32x4_t q4_d_0123    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d));
                float32x4_t q4_d_4567    = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].d + 4));
                // d8 0 1 2 3
                float32x4_t q8_d_0123    = vld1q_f32(q8_ptr[b].d);
                // mins
                float32x4_t q4_dmin_0123 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin));
                float32x4_t q4_dmin_4567 = vcvt_f32_f16(vld1_f16((const __fp16 *) q4_ptr[b].dmin + 4));

                // Precomputation of scales and mins
                float32x4_t sbd_scale_0123[q8_k_blocklen];
                float32x4_t sbd_scale_4567[q8_k_blocklen];
                float32x4_t sbd_min_0123[q8_k_blocklen];
                float32x4_t sbd_min_4567[q8_k_blocklen];

                sbd_scale_0123[0] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 0);
                sbd_scale_4567[0] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 0);
                sbd_min_0123[0]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 0);
                sbd_min_4567[0]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 0);

                sbd_scale_0123[1] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 1);
                sbd_scale_4567[1] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 1);
                sbd_min_0123[1]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 1);
                sbd_min_4567[1]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 1);

                sbd_scale_0123[2] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 2);
                sbd_scale_4567[2] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 2);
                sbd_min_0123[2]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 2);
                sbd_min_4567[2]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 2);

                sbd_scale_0123[3] = vmulq_laneq_f32(q4_d_0123, q8_d_0123, 3);
                sbd_scale_4567[3] = vmulq_laneq_f32(q4_d_4567, q8_d_0123, 3);
                sbd_min_0123[3]   = vmulq_laneq_f32(q4_dmin_0123, q8_d_0123, 3);
                sbd_min_4567[3]   = vmulq_laneq_f32(q4_dmin_4567, q8_d_0123, 3);

                // Precomputation of bsums, each vpaddq calcs all the bsums for each row
                const int16x8_t bsums[q8_k_blocklen] = {
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 0), vld1q_s16(q8_ptr[b].bsums + 16 * 0 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 1), vld1q_s16(q8_ptr[b].bsums + 16 * 1 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 2), vld1q_s16(q8_ptr[b].bsums + 16 * 2 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 3), vld1q_s16(q8_ptr[b].bsums + 16 * 3 + 8)),
                };
                int16_t bsums_arr[QK_K / 64][8];
                for (int q8_row = 0; q8_row < 4; q8_row++) {
                    vst1q_s16(bsums_arr[q8_row], bsums[q8_row]);
                }

                // interleaved bias_acc: [0]->r0 0123, [1]->r1 0123, .., [4]->r0 4567, [5]->r1 4567 ..
                int32x4_t bias_acc[acc_size];
                for (int i = 0; i < acc_size; i++) {
                    bias_acc[i] = vdupq_n_s32(0);
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    // Int accumulators for qs vecdot (4 row x 2 col quartets)
                    int32x4_t acc_lo[acc_size];
                    int32x4_t acc_hi[acc_size];
                    for (int i = 0; i < acc_size; i++) {
                        acc_lo[i] = vdupq_n_s32(0);
                        acc_hi[i] = vdupq_n_s32(0);
                    }
                    // Need scales for the low and high nibbles
                    // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
                    int16x8_t q4sb_scales[2];
                    int16x8_t q4sb_mins[2];
                    for (int i = 0; i < 2; i++) {
                        int8_t    aux_q4sb[8];
                        const int offset = sb * 24 + i * 12;
                        decode_q4_Kx8_scales_mins(&q4_ptr[b].scales[offset], &q4sb_mins[i], aux_q4sb);
                        q4sb_scales[i] = vmovl_s8(vld1_s8(aux_q4sb));
                    }

                    constexpr int reads_per_sb = 8;  // 8 * 16 bytes each => 32 qs * 4 rows
                    for (int k = 0; k < reads_per_sb; k++) {
                        const int8x16_t q8_blk0 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k);
                        const int8x16_t q8_blk1 = vld1q_s8(q8_ptr[b].qs + sb * 256 + 16 * k + 128);

                        // 0..3 & 32..35
                        const uint8x16_t q4_0123 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k);
                        const uint8x16_t q4_4567 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 32 * k + 16);

                        const int8x16_t q4_0123_lo = vreinterpretq_s8_u8(vandq_u8(q4_0123, m4b));
                        const int8x16_t q4_0123_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_0123, 4));

                        acc_lo[0] = vdotq_laneq_s32(acc_lo[0], q4_0123_lo, q8_blk0, 0);  //  0..3  r0 c0123
                        acc_lo[1] = vdotq_laneq_s32(acc_lo[1], q4_0123_lo, q8_blk0, 1);  //  0..3  r1 c0123
                        acc_lo[2] = vdotq_laneq_s32(acc_lo[2], q4_0123_lo, q8_blk0, 2);  //  0..3  r2 c0123
                        acc_lo[3] = vdotq_laneq_s32(acc_lo[3], q4_0123_lo, q8_blk0, 3);  //  0..3  r3 c0123

                        acc_hi[0] = vdotq_laneq_s32(acc_hi[0], q4_0123_hi, q8_blk1, 0);  // 32..35 r0 c0123
                        acc_hi[1] = vdotq_laneq_s32(acc_hi[1], q4_0123_hi, q8_blk1, 1);  // 32..35 r1 c0123
                        acc_hi[2] = vdotq_laneq_s32(acc_hi[2], q4_0123_hi, q8_blk1, 2);  // 32..35 r2 c0123
                        acc_hi[3] = vdotq_laneq_s32(acc_hi[3], q4_0123_hi, q8_blk1, 3);  // 32..35 r3 c0123

                        const int8x16_t q4_4567_lo = vreinterpretq_s8_u8(vandq_u8(q4_4567, m4b));
                        const int8x16_t q4_4567_hi = vreinterpretq_s8_u8(vshrq_n_u8(q4_4567, 4));

                        acc_lo[4] = vdotq_laneq_s32(acc_lo[4], q4_4567_lo, q8_blk0, 0);  //  0..3  r0 c4567
                        acc_lo[5] = vdotq_laneq_s32(acc_lo[5], q4_4567_lo, q8_blk0, 1);  //  0..3  r1 c4567
                        acc_lo[6] = vdotq_laneq_s32(acc_lo[6], q4_4567_lo, q8_blk0, 2);  //  0..3  r2 c4567
                        acc_lo[7] = vdotq_laneq_s32(acc_lo[7], q4_4567_lo, q8_blk0, 3);  //  0..3  r3 c4567

                        acc_hi[4] = vdotq_laneq_s32(acc_hi[4], q4_4567_hi, q8_blk1, 0);  // 32..35 r0 c4567
                        acc_hi[5] = vdotq_laneq_s32(acc_hi[5], q4_4567_hi, q8_blk1, 1);  // 32..35 r1 c4567
                        acc_hi[6] = vdotq_laneq_s32(acc_hi[6], q4_4567_hi, q8_blk1, 2);  // 32..35 r2 c4567
                        acc_hi[7] = vdotq_laneq_s32(acc_hi[7], q4_4567_hi, q8_blk1, 3);  // 32..35 r3 c4567
                    }

                    // Scale and bias application
                    // acc is stored interleaved to match output layout
                    const int16x4_t sc_0123_lo = vget_low_s16(q4sb_scales[0]);
                    const int16x4_t sc_4567_lo = vget_high_s16(q4sb_scales[0]);
                    const int16x4_t sc_0123_hi = vget_low_s16(q4sb_scales[1]);
                    const int16x4_t sc_4567_hi = vget_high_s16(q4sb_scales[1]);
                    for (int row = 0; row < q8_k_blocklen; row++) {
                        // Bias correction
                        // row c0123 blk0 and blk1
                        const float32x4_t sumf_0123 =
                            vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_0123_lo), acc_lo[row]),
                                                    vmulq_s32(vmovl_s16(sc_0123_hi), acc_hi[row])));
                        acc_f32[2 * row] = vfmaq_f32(acc_f32[2 * row], sbd_scale_0123[row], sumf_0123);

                        // row c4567 blk0 and blk1
                        const float32x4_t sumf_4567 =
                            vcvtq_f32_s32(vaddq_s32(vmulq_s32(vmovl_s16(sc_4567_lo), acc_lo[row + 4]),
                                                    vmulq_s32(vmovl_s16(sc_4567_hi), acc_hi[row + 4])));
                        acc_f32[2 * row + 1] = vfmaq_f32(acc_f32[2 * row + 1], sbd_scale_4567[row], sumf_4567);

                        // Bias
                        const int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[sb][row * 2]);
                        const int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[sb][row * 2 + 1]);

                        // row c0123 blk0 and blk1
                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                        bias_acc[2 * row] = vmlal_s16(bias_acc[2 * row], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));

                        // row c4567 blk0 and blk1
                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                        bias_acc[2 * row + 1] =
                            vmlal_s16(bias_acc[2 * row + 1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
                    }
                }  // for sb

                for (int row = 0; row < q8_k_blocklen; row++) {
                    acc_f32[2 * row] = vmlsq_f32(acc_f32[2 * row], vcvtq_f32_s32(bias_acc[2 * row]), sbd_min_0123[row]);
                    acc_f32[2 * row + 1] =
                        vmlsq_f32(acc_f32[2 * row + 1], vcvtq_f32_s32(bias_acc[2 * row + 1]), sbd_min_4567[row]);
                }
            }  // for b

            for (int i = 0; i < q8_k_blocklen; i++) {
                int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    int col    = x * ncols_interleaved + j * 4;
                    int offset = row * bs + col;
                    vst1q_f32(s + offset, acc_f32[2 * i + j]);
                }
            }
        }  // for x
    }  // for y
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemm_q4_K_8x4_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q4_K_8x8_q8_K(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
    constexpr int qk = QK_K;
    const int     nb = n / qk;

    constexpr int ncols_interleaved = 8;
    constexpr int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    constexpr int    q8_k_blocklen = 4;
    const uint8x16_t m4b           = vdupq_n_u8(0x0f);

    // 8 accumulators: 2 row pairs × 4 col pairs
    float32x4_t acc_f32[blocklen];

    for (int y = 0; y < nr / q8_k_blocklen; y++) {
        const block_q8_Kx4 * GGML_RESTRICT q8_ptr = (const block_q8_Kx4 *) vy + (y * nb);

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * GGML_RESTRICT q4_ptr = (const block_q4_Kx8 *) vx + (x * nb);

            for (int i = 0; i < blocklen; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                // bsums pairs belongs to the same q8_k subblock
                const int16x8_t bsums[4]{
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 0), vld1q_s16(q8_ptr[b].bsums + 16 * 0 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 1), vld1q_s16(q8_ptr[b].bsums + 16 * 1 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 2), vld1q_s16(q8_ptr[b].bsums + 16 * 2 + 8)),
                    vpaddq_s16(vld1q_s16(q8_ptr[b].bsums + 16 * 3), vld1q_s16(q8_ptr[b].bsums + 16 * 3 + 8)),
                };
                int16_t bsums_arr[4][8];
                for (int q8_row = 0; q8_row < 4; q8_row++) {
                    vst1q_s16(bsums_arr[q8_row], bsums[q8_row]);
                }

                int32x4_t sb_acc[4];    // Aux accumulators to store subblock (partial) results
                int32x4_t acc[8];       // rows 01 stored in [0][1][2][3] rows 23 stored in [4][5][6][7]
                int32x4_t bias_acc[8];  // interleaved bias_acc: [0]->r0 0123, [1]->r0 4567, [2]->r1 0123 ...
                for (int i = 0; i < 8; i++) {
                    acc[i]      = vdupq_n_s32(0);
                    bias_acc[i] = vdupq_n_s32(0);
                }

                for (int sb = 0; sb < QK_K / 64; sb++) {
                    // Need scales for the low and high nibbles
                    // 2 * 12 = 24 bytes per subblock, 4 sbs -> 4 * 24 = 96 bytes total
                    int8_t    q4sb_scales[2][8];
                    int16x8_t q4sb_mins[2];  // int16 as its needed for bias_acc later
                    for (int i = 0; i < 2; i++) {
                        const int offset = sb * 24 + i * 12;
                        decode_q4_Kx8_scales_mins(&q4_ptr[b].scales[offset], &q4sb_mins[i], q4sb_scales[i]);
                    }

                    // q8_ptr[b].qs has interleaved Q8 rows (01, 23)
                    const int8_t * q8_base = q8_ptr[b].qs + sb * 256;

                    int8x16_t q8_qs_01[8];
                    int8x16_t q8_qs_23[8];

                    // Load 32-byte per row pair, 1 subblock each time
                    for (int i = 0; i < 8; i++) {
                        const int offset = i * 32;  // 16 for row 01, 16 for row 23
                        q8_qs_01[i]      = vld1q_s8(q8_base + offset);
                        q8_qs_23[i]      = vld1q_s8(q8_base + offset + 16);
                    }

                    const int8x16_t q8s[2][8] = {
                        { q8_qs_01[0], q8_qs_01[1], q8_qs_01[2], q8_qs_01[3],
                          q8_qs_01[4], q8_qs_01[5], q8_qs_01[6], q8_qs_01[7] },
                        { q8_qs_23[0], q8_qs_23[1], q8_qs_23[2], q8_qs_23[3],
                          q8_qs_23[4], q8_qs_23[5], q8_qs_23[6], q8_qs_23[7] },
                    };

                    // Q4s columns iterated in pairs (01, 23, 45, 67)
                    for (int cp = 0; cp < ncols_interleaved / 2; cp++) {
                        for (int i = 0; i < 4; i++) {
                            sb_acc[i] = vdupq_n_s32(0);
                        }

                        uint8x16_t q4_qs_cp_0 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 16 * cp + 0);    // 0 .. 7 & 32..39
                        uint8x16_t q4_qs_cp_1 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 16 * cp + 64);   // 8 ..15 & 40..47
                        uint8x16_t q4_qs_cp_2 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 16 * cp + 128);  // 16..23 & 48..55
                        uint8x16_t q4_qs_cp_3 = vld1q_u8(q4_ptr[b].qs + sb * QK_K + 16 * cp + 192);  // 24..31 & 56..63
                        const int8x16_t q4_nibbles[2][4] = {
                            {
                                vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_0, m4b)),
                                vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_1, m4b)),
                                vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_2, m4b)),
                                vreinterpretq_s8_u8(vandq_u8(q4_qs_cp_3, m4b)),
                            },
                            {
                                vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_0, 4)),
                                vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_1, 4)),
                                vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_2, 4)),
                                vreinterpretq_s8_u8(vshrq_n_u8(q4_qs_cp_3, 4)),
                            }
                        };

                        // Calculates the Qs muladd of every row pair (rp) rows 01 and 23 of q8
                        // for each of the internal 32 qs subblock (blk)
                        for (int rp = 0; rp < 2; rp++) {
                            for (int blk = 0; blk < 2; blk++) {
                                const int8x16_t * q8  = &q8s[rp][4 * blk];
                                const int8x16_t * q4  = q4_nibbles[blk];
                                int32x4_t         acc = sb_acc[2 * rp + blk];
                                // mul add for each qs in the same subblock
                                for (int qs_offset = 0; qs_offset < 4; qs_offset++) {
                                    acc = vmmlaq_s32(acc, q4[qs_offset], q8[qs_offset]);
                                }
                                sb_acc[2 * rp + blk] = acc;
                            }
                        }

                        // Scales[i] corresponds to column i
                        const int scale_offset = cp * 2;
                        for (int blk = 0; blk < 2; blk++) {
                            const int32x4_t block_scale = {
                                (int32_t) q4sb_scales[blk][scale_offset],
                                (int32_t) q4sb_scales[blk][scale_offset],
                                (int32_t) q4sb_scales[blk][scale_offset + 1],
                                (int32_t) q4sb_scales[blk][scale_offset + 1],
                            };
                            acc[cp]     = vmlaq_s32(acc[cp], sb_acc[blk], block_scale);
                            acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc[blk + 2], block_scale);
                        }
                    }

                    // Multiply Acc bsum + mins
                    for (int q8_row = 0; q8_row < 4; q8_row++) {
                        // Each pair of subblocks share the same bsums
                        // Load scalar bsum → broadcast to a vector (vdupq_n_s16(s)).
                        int16x4_t bsums_vec_lo = vdup_n_s16(bsums_arr[sb][q8_row * 2]);
                        int16x4_t bsums_vec_hi = vdup_n_s16(bsums_arr[sb][q8_row * 2 + 1]);

                        bias_acc[2 * q8_row] =
                            vmlal_s16(bias_acc[2 * q8_row], bsums_vec_lo, vget_low_s16(q4sb_mins[0]));
                        bias_acc[2 * q8_row] =
                            vmlal_s16(bias_acc[2 * q8_row], bsums_vec_hi, vget_low_s16(q4sb_mins[1]));
                        bias_acc[2 * q8_row + 1] =
                            vmlal_s16(bias_acc[2 * q8_row + 1], bsums_vec_lo, vget_high_s16(q4sb_mins[0]));
                        bias_acc[2 * q8_row + 1] =
                            vmlal_s16(bias_acc[2 * q8_row + 1], bsums_vec_hi, vget_high_s16(q4sb_mins[1]));
                    }
                }  // for sb

                // Reorder of i8mm output with bias and output layout
                for (int i = 0; i < 8; i++) {
                    int32x2x2_t aux = vzip_s32(vget_low_s32(acc[i]), vget_high_s32(acc[i]));
                    acc[i]          = vcombine_s32(aux.val[0], aux.val[1]);
                }
                int32x4_t reorder_acc[8] = {
                    vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
                    vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
                    vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
                    vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
                    vcombine_s32(vget_low_s32(acc[4]), vget_low_s32(acc[5])),
                    vcombine_s32(vget_low_s32(acc[6]), vget_low_s32(acc[7])),
                    vcombine_s32(vget_high_s32(acc[4]), vget_high_s32(acc[5])),
                    vcombine_s32(vget_high_s32(acc[6]), vget_high_s32(acc[7])),
                };

                for (int i = 0; i < q8_k_blocklen; i++) {
                    for (int j = 0; j < 2; j++) {
                        float32x4_t       q8_d    = vdupq_n_f32(q8_ptr[b].d[i]);
                        float32x4_t       q4_dmin = vcvt_f32_f16(vld1_f16((const __fp16 *) (q4_ptr[b].dmin + j * 4)));
                        const float32x4_t dmins   = vmulq_f32(q4_dmin, q8_d);

                        float32x4_t       q4_d  = vcvt_f32_f16(vld1_f16((const __fp16 *) (q4_ptr[b].d + j * 4)));
                        const float32x4_t scale = vmulq_f32(q4_d, q8_d);

                        acc_f32[2 * i + j] = vmlsq_f32(acc_f32[2 * i + j], vcvtq_f32_s32(bias_acc[2 * i + j]), dmins);
                        acc_f32[2 * i + j] =
                            vmlaq_f32(acc_f32[2 * i + j], vcvtq_f32_s32(reorder_acc[2 * i + j]), scale);
                    }
                }
            }  // for b

            // With the previous reorder, the tile is already in the correct memory layout.
            for (int i = 0; i < q8_k_blocklen; i++) {
                int row = y * q8_k_blocklen + i;
                for (int j = 0; j < 2; j++) {
                    int col    = x * ncols_interleaved + j * 4;
                    int offset = row * bs + col;
                    vst1q_f32(s + offset, acc_f32[2 * i + j]);
                }
            }
        }  // for x
    }  // for y
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    ggml_gemm_q4_K_8x8_q8_K_generic(n, s, bs, vx, vy, nr, nc);
}


void ggml_gemm_q8_0_4x4_q8_0(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 4;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q8_0x4 * b_ptr = (const block_q8_0x4 *) vx + (x * nb);

            float32x4_t sumf[4];
            for (int m = 0; m < 4; m++) {
                sumf[m] = vdupq_n_f32(0);
            }

            for (int l = 0; l < nb; l++) {
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const float16_t *) a_ptr[l].d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *) b_ptr[l].d));

                int32x4_t sumi_0 = vdupq_n_s32(0);
                int32x4_t sumi_1 = vdupq_n_s32(0);
                int32x4_t sumi_2 = vdupq_n_s32(0);
                int32x4_t sumi_3 = vdupq_n_s32(0);

                for (int k_group = 0; k_group < 8; k_group += 4) {
                    int8x16x4_t a = vld1q_s8_x4(a_ptr[l].qs + 16 * k_group);
                    int8x16x4_t b = vld1q_s8_x4(b_ptr[l].qs + 16 * k_group);

                    for (int k = 0; k < 4; k++) {
                        sumi_0 = vdotq_laneq_s32(sumi_0, b.val[k], a.val[k], 0);
                        sumi_1 = vdotq_laneq_s32(sumi_1, b.val[k], a.val[k], 1);
                        sumi_2 = vdotq_laneq_s32(sumi_2, b.val[k], a.val[k], 2);
                        sumi_3 = vdotq_laneq_s32(sumi_3, b.val[k], a.val[k], 3);
                    }
                }

                sumf[0] = vmlaq_f32(sumf[0], vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(sumi_0));
                sumf[1] = vmlaq_f32(sumf[1], vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(sumi_1));
                sumf[2] = vmlaq_f32(sumf[2], vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(sumi_2));
                sumf[3] = vmlaq_f32(sumf[3], vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(sumi_3));
            }

            for (int m = 0; m < 4; m++) {
                vst1q_f32(s + (y * 4 + m) * bs + x * 4, sumf[m]);
            }
        }
    }
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    ggml_gemm_q8_0_4x4_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}

void ggml_gemm_q8_0_4x8_q8_0(int                        n,
                             float * GGML_RESTRICT      s,
                             size_t                     bs,
                             const void * GGML_RESTRICT vx,
                             const void * GGML_RESTRICT vy,
                             int                        nr,
                             int                        nc) {
    const int qk                = QK8_0;
    const int nb                = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen          = 8;

    assert(n % qk == 0);
    assert(nr % 4 == 0);
    assert(nc % ncols_interleaved == 0);

    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    const block_q8_0x4 * b_ptr_base = (const block_q8_0x4 *) vx;

    for (int y = 0; y < nr; y += 4) {
        const block_q8_0x4 * a_ptr_base = (const block_q8_0x4 *) vy + (y / 4) * nb;

        for (int x = 0; x < nc; x += ncols_interleaved) {
            const block_q8_0x4 * b_ptr = b_ptr_base + (x / 4) * nb;
            const block_q8_0x4 * a_ptr = a_ptr_base;

            float32x4_t acc_f32[4];
            for (int i = 0; i < 4; i++) {
                acc_f32[i] = vdupq_n_f32(0);
            }

            for (int b = 0; b < nb; b++) {
                int32x4_t acc[4];
                for (int i = 0; i < 4; i++) {
                    acc[i] = vdupq_n_s32(0);
                }

                // Process 4 chunks of 8 positions each
                for (int chunk = 0; chunk < 4; chunk++) {
                    int8x16_t a01 = vld1q_s8(a_ptr->qs + chunk * 32);
                    int8x16_t a23 = vld1q_s8(a_ptr->qs + chunk * 32 + 16);
                    int8x16_t b01 = vld1q_s8(b_ptr->qs + chunk * 32);
                    int8x16_t b23 = vld1q_s8(b_ptr->qs + chunk * 32 + 16);

                    acc[0] = vmmlaq_s32(acc[0], a01, b01);
                    acc[1] = vmmlaq_s32(acc[1], a01, b23);
                    acc[2] = vmmlaq_s32(acc[2], a23, b01);
                    acc[3] = vmmlaq_s32(acc[3], a23, b23);
                }

                // Reorder outputs from 2×2 tiles to row-major
                // acc[0] = [r0c0, r0c1, r1c0, r1c1]
                // acc[1] = [r0c2, r0c3, r1c2, r1c3]
                // acc[2] = [r2c0, r2c1, r3c0, r3c1]
                // acc[3] = [r2c2, r2c3, r3c2, r3c3]
                int32x4_t row0 = vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1]));
                int32x4_t row1 = vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1]));
                int32x4_t row2 = vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3]));
                int32x4_t row3 = vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3]));

                // Scales
                float32x4_t a_d = vcvt_f32_f16(vld1_f16((const __fp16 *) a_ptr->d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const __fp16 *) b_ptr->d));

                acc_f32[0] = vfmaq_f32(acc_f32[0], vcvtq_f32_s32(row0), vmulq_laneq_f32(b_d, a_d, 0));
                acc_f32[1] = vfmaq_f32(acc_f32[1], vcvtq_f32_s32(row1), vmulq_laneq_f32(b_d, a_d, 1));
                acc_f32[2] = vfmaq_f32(acc_f32[2], vcvtq_f32_s32(row2), vmulq_laneq_f32(b_d, a_d, 2));
                acc_f32[3] = vfmaq_f32(acc_f32[3], vcvtq_f32_s32(row3), vmulq_laneq_f32(b_d, a_d, 3));

                a_ptr++;
                b_ptr++;
            }

            for (int row = 0; row < 4; row++) {
                vst1q_f32(s + (y + row) * bs + x, acc_f32[row]);
            }
        }
    }
    return;
#endif  // defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    ggml_gemm_q8_0_4x8_q8_0_generic(n, s, bs, vx, vy, nr, nc);
}
