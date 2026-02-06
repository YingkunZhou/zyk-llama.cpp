#pragma once

#include "iqk_config.h"

#include <cstdint>
#include <stddef.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define IQK_MAX_NY 8

typedef struct {
    int32_t i1;
    int32_t i2;
} mmid_row_mapping;

struct DataInfo {
    float       * s;
    const char  * cy;
    size_t        bs;
    size_t        by;
    int           cur_y = 0;
    int           ne11;
    const mmid_row_mapping * row_mapping = nullptr;
    size_t        bs2 = 0;

    inline const char * src1_row(int iy) const {
        if (!row_mapping) return cy + (cur_y + iy)*by;
        int i11 = row_mapping[cur_y + iy].i1 % ne11;
        int i12 = row_mapping[cur_y + iy].i2;
        return cy + (i11 + i12*ne11)*by;
    }

    inline void store(int ix, int iy, float result) const {
        *(dst_row(iy) + ix) = result;
    }
#ifdef __AVX__
    inline void store(int ix, int iy, __m128 result) const {
        _mm_storeu_ps(dst_row(iy) + ix, result);
    }
    inline void store(int ix, int iy, __m256 result) const {
        _mm256_storeu_ps(dst_row(iy) + ix, result);
    }
#endif
#ifdef __AVX512F__
    inline void store(int ix, int iy, __m512 result) const {
        _mm512_storeu_ps(dst_row(iy) + ix, result);
    }
#endif
#ifdef __ARM_NEON
    inline void store(int ix, int iy, float32x4_t result) const {
        vst1q_f32(dst_row(iy) + ix, result);
    }
#endif
    inline float * dst_row(int iy) const {
        if (!row_mapping) return s + (cur_y + iy)*bs;
        int i12 = row_mapping[cur_y + iy].i2;
        int i1  = row_mapping[cur_y + iy].i1;
        int i2  = i12;
        return s + i1*bs + i2*bs2;
    }
};

typedef void (*mul_mat_t)(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x);
