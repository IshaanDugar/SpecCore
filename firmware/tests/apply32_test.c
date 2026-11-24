/* apply32 / quant_vals (fast path, ref/qref.c) must equal apply / clamp8(apply) for every input
   the model can produce: any 32-bit v, m = 0 or in [2^29, 2^30), any shift. Random + edge cases.
   PC only: python tools/sc.py native a32 firmware/tests/apply32_test.c */
#include "../../ref/qref.c"
#include <stdio.h>

static uint64_t st = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }

int main(void) {
    static const int32_t edge_v[] = {0, 1, -1, 127, -127, 2147483647, -2147483647 - 1, 4095, -4096};
    uint32_t bad = 0, n = 0, fast = 0;
    for (uint32_t i = 0; i < 20000000u; i++) {
        int32_t v = i < 9 * 64 ? edge_v[i % 9] : (int32_t)(uint32_t)rnd();
        if ((i & 3) == 1) v >>= (int)(rnd() % 31);                    /* small magnitudes too */
        uint32_t m = (i & 15) == 0 ? 0 : (uint32_t)((1u << 29) + (rnd() & ((1u << 29) - 1)));
        if ((i & 63) == 2) m = (1u << 30) - 1;
        if ((i & 63) == 3) m = 1u << 29;
        int frac = (int)(rnd() % 17);
        int32_t e = (int32_t)(rnd() % 80) - 8;                        /* shifts -8-frac .. 71 */
        int32_t sh = e - frac;
        if (sh < 1 && m) continue;                                     /* apply() needs sh >= 1 */
        n++;
        if (m && sh >= 1 && sh <= 62) fast++;          /* both fast paths */
        if (apply32(v, m, e, frac) != (int32_t)apply(v, m, e, frac)) {
            if (bad++ < 5) printf("MISMATCH v %d m %u e %d frac %d\n", v, m, e, frac);
        }
    }
    /* quant_vals on whole vectors, as quant() uses it: |v| <= a, scale 127 / a */
    int32_t vec[64]; int8_t q1[64];
    for (uint32_t t = 0; t < 200000u; t++) {
        uint32_t a = (uint32_t)(rnd() >> (33 + rnd() % 31)) | 1u;
        if (a > 2147483647u) a = 2147483647u;
        for (int i = 0; i < 64; i++) vec[i] = (int32_t)(rnd() % (2ull * a + 1)) - (int32_t)a;
        uint32_t rm; int32_t re;
        norm_scale(127, a, &rm, &re);
        quant_vals(vec, 64, rm, re, q1);
        for (int i = 0; i < 64; i++) {
            n++;
            if (q1[i] != clamp8(apply(vec[i], rm, re, 0))) {
                if (bad++ < 5) printf("QUANT MISMATCH a %u v %d\n", a, vec[i]);
            }
        }
    }
    printf("apply32 test: %u cases (%u on the fast path), %u mismatches\n", n, fast, bad);
    printf(bad ? "APPLY32 FAIL\n" : "APPLY32 PASS\n");
    return bad ? 1 : 0;
}
