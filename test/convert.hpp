#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "immintrin.h"

uint32_t float_as_uint32 (float a)
{
    uint32_t r;
    memcpy (&r, &a, sizeof r);
    return r;
}

uint16_t float2half_rn (float a)
{
    uint32_t ia = float_as_uint32 (a);
    uint16_t ir;

    ir = (ia >> 16) & 0x8000;
    if ((ia & 0x7f800000) == 0x7f800000) {
        if ((ia & 0x7fffffff) == 0x7f800000) {
            ir |= 0x7c00; /* infinity */
        } else {
            ir |= 0x7e00 | ((ia >> (24 - 11)) & 0x1ff); /* NaN, quietened */
        }
    } else if ((ia & 0x7f800000) >= 0x33000000) {
        int shift = (int)((ia >> 23) & 0xff) - 127;
        if (shift > 15) {
            ir |= 0x7c00; /* infinity */
        } else {
            ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
            if (shift < -14) { /* denormal */  
                ir |= ia >> (-1 - shift);
                ia = ia << (32 - (-1 - shift));
            } else { /* normal */
                ir |= ia >> (24 - 11);
                ia = ia << (32 - (24 - 11));
                ir = ir + ((14 + shift) << 10);
            }
            /* IEEE-754 round to nearest of even */
            if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
                ir++;
            }
        }
    }
    return ir;
}

uint16_t float2half_rn_ref (float a)
{
    __m128 pa = _mm_set_ps1 (a);
    __m128i r16 = _mm_cvtps_ph (pa, _MM_FROUND_TO_NEAREST_INT);
    uint16_t res;
    memcpy (&res, &r16, sizeof res);
    return res;
}

float uint32_as_float (uint32_t a)
{
    float r;
    memcpy (&r, &a, sizeof r);
    return r;
}