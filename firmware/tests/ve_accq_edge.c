/* Verification check (PC only): ACCQ arithmetic, three independent models compared.
     A. sc_accq_      the PC model of the unit in firmware/include/sc_isa.h
     B. qref          ref/qref.c smul() + apply() + apply32() (copied below, both apply32 paths)
     C. rtl_mirror    ext/rtl/sc_vdotk.sv S2..S5 transcribed with the RTL widths (64-bit q_p,
                      32-bit q_m, 32-bit wrapping exponent, 65-bit signed q_prod/q_rnd, 6-bit shift)
   Legal domain (mantissas 0 or in [2^29, 2^30), any exponents): A == C always; A == B whenever
   qref is defined (m == 0 or sh >= 1). Wide domain (any 32-bit operands): A vs C, mismatches
   counted per class (expected only where C int64 overflows, i.e. illegal mantissas).
   python tools/sc.py native ve_accq firmware/tests/ve_accq_edge.c [-DVE_N=20000000] */
#include <stdio.h>
#include <stdlib.h>
#include "sc_isa.h"

#ifndef VE_N
#define VE_N 20000000
#endif

/* ---- B: ref/qref.c, verbatim arithmetic ---- */
static inline void smul(uint32_t am, int32_t ae, uint32_t bm, int32_t be, uint32_t *m, int32_t *e) {
    uint64_t p = (uint64_t)am * bm;
    if (!p) { *m = 0; *e = 0; return; }
    if (p >= (1ull << 59)) { *m = (uint32_t)(p >> 30); *e = ae + be - 30; }
    else { *m = (uint32_t)(p >> 29); *e = ae + be - 29; }
}
static inline int64_t apply(int64_t v, uint32_t m, int32_t e, int frac) {
    int32_t sh = e - frac;
    if (!m || sh > 62) return 0;
    return (v * (int64_t)m + (1ll << (sh - 1))) >> sh;
}
static inline int32_t apply32(int32_t v, uint32_t m, int32_t e, int frac) {
    int32_t sh = e - frac;
    if (m && sh >= 32 && sh <= 62) {
        int64_t p = (int64_t)v * (int32_t)m + (1ll << (sh - 1));
        return (int32_t)(p >> 32) >> (sh - 32);
    }
    if (m && sh >= 1 && sh <= 31) {
        uint64_t p = (uint64_t)((int64_t)v * (int32_t)m) + (1ull << (sh - 1));
        uint32_t lo = (uint32_t)p, hi = (uint32_t)(p >> 32);
        return (int32_t)((hi << (32 - sh)) | (lo >> sh));
    }
    return (int32_t)apply(v, m, e, frac);
}

/* ---- C: RTL mirror ---- */
static int last_branch59, last_sh;
static int32_t rtl_mirror(int32_t v, uint32_t sm, uint32_t se, uint32_t rm, uint32_t re) {
    uint64_t qp = (uint64_t)sm * (uint64_t)rm;                 /* S2: 64'(q_sm) * 64'(q_rm) */
    uint32_t qm, e;
    if (qp >= 0x0800000000000000ull) { qm = (uint32_t)(qp >> 30); e = se + re - 30u; }  /* q_p[61:30] */
    else { qm = (uint32_t)(qp >> 29); e = se + re - 29u; }                              /* q_p[60:29] */
    int32_t sh = (int32_t)(e - 12u);
    int zero = qp == 0 || qp < 0x20000000ull || sh > 62 || sh < 1;
    __int128 prod = (__int128)v * (__int128)qm;                /* S4: 65-bit signed, |.| < 2^63 */
    unsigned s6 = (uint32_t)sh & 63u;
    __int128 rnd = prod + ((__int128)1 << ((s6 - 1u) & 63u));   /* q_sh[5:0] - 6'd1 (6-bit wrap) */
    last_branch59 = qp >= 0x0800000000000000ull; last_sh = sh;
    return zero ? 0 : (int32_t)(uint32_t)(rnd >> s6);          /* 32'(q_rnd >>> q_sh[5:0]) */
}

static int32_t model(int l, int32_t v, uint32_t sm, int32_t se, uint32_t rm, int32_t re) {
    sc_u.acc[l] = v;
    SC_SETS(l, sm, se);
    int32_t r = SC_ACCQ(l, rm, re);
    if (sc_u.acc[l] != 0) { puts("FAIL: acc not cleared"); exit(1); }
    return r;
}

static uint64_t s = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (uint32_t)(s >> 16); }
static const int32_t VE[] = {0, 1, -1, 2, -2, 0x7FFFFFFF, (int32_t)0x80000000, (int32_t)0x80000001, 0x7FFFFFFE,
                             0x40000000, (int32_t)0xC0000000, 12345, -12345, 0xFFFF, -0x10000};
static const uint32_t ME[] = {0, 1u << 29, (1u << 29) + 1, (1u << 30) - 1, (1u << 30) - 2, 0x2D413CCDu /* ~sqrt2*2^29 */,
                              0x2D413CCCu, 0x2D413CCEu, 0x30000000u, 0x3FFF0000u};
static int32_t pick_v(void) {
    uint32_t r = rnd32() % 8;
    if (r < 3) return VE[rnd32() % (sizeof VE / sizeof *VE)];
    if (r < 5) return (int32_t)rnd32();
    if (r < 6) return (int32_t)(rnd32() % 4096) - 2048;
    return (int32_t)rnd32() >> (rnd32() % 31);
}
static uint32_t pick_m(void) {
    uint32_t r = rnd32() % 8;
    if (r < 2) return ME[rnd32() % (sizeof ME / sizeof *ME)];
    return 0x20000000u | (rnd32() & 0x1FFFFFFFu);
}

static long long n_leg, n_ref, bad_ac, bad_ab, bad_ab64, n_wide, bad_wide, bad_wide_legalm;
static long long bin_br[2], bin_sh[5], bin_mz;   /* coverage: branch 2^59, sh {<1, 1..31, 32..62, >62}, m==0 */

static void one_legal(int32_t v, uint32_t sm, int32_t se, uint32_t rm, int32_t re) {
    int l = rnd32() & 7;
    int32_t a = model(l, v, sm, se, rm, re);
    int32_t c = rtl_mirror(v, sm, (uint32_t)se, rm, (uint32_t)re);
    n_leg++;
    if (a != c) { if (bad_ac++ < 10) printf("A!=C v=%d sm=%#x se=%d rm=%#x re=%d A=%d C=%d\n", v, sm, se, rm, re, a, c); }
    uint64_t p = (uint64_t)sm * rm;
    if (!p) { bin_mz++; } else {
        bin_br[last_branch59]++;
        bin_sh[last_sh < 1 ? 0 : last_sh <= 31 ? 1 : last_sh <= 62 ? 2 : 3]++;
        if (last_sh == 1 || last_sh == 31 || last_sh == 32 || last_sh == 62) bin_sh[4]++;
    }
    /* qref is defined when m == 0 or sh >= 1, and its int32 exponent sum does not overflow */
    int64_t es = (int64_t)se + re;
    if (p && (last_sh < 1 || es > INT32_MAX || es < INT32_MIN + 64)) return;
    uint32_t m; int32_t e;
    smul(sm, se, rm, re, &m, &e);
    int32_t b = apply32(v, m, e, 12), b64 = (int32_t)apply(v, m, e, 12);
    n_ref++;
    if (a != b && bad_ab++ < 10) printf("A!=apply32 v=%d sm=%#x se=%d rm=%#x re=%d A=%d B=%d\n", v, sm, se, rm, re, a, b);
    if (a != b64 && bad_ab64++ < 10) printf("A!=apply v=%d sm=%#x se=%d rm=%#x re=%d A=%d B=%d\n", v, sm, se, rm, re, a, b64);
}

int main(void) {
    /* 1. directed boundaries: every edge v x edge mantissa pair x sh in -3..66 x both exponent splits */
    for (unsigned iv = 0; iv < sizeof VE / sizeof *VE; iv++)
        for (unsigned i = 0; i < sizeof ME / sizeof *ME; i++)
            for (unsigned j = 0; j < sizeof ME / sizeof *ME; j++)
                for (int sh = -3; sh <= 66; sh++) {
                    uint64_t p = (uint64_t)ME[i] * ME[j];
                    int tot = sh + 12 + (p >= (1ull << 59) ? 30 : 29);
                    one_legal(VE[iv], ME[i], tot / 2, ME[j], tot - tot / 2);
                    one_legal(VE[iv], ME[i], -1000, ME[j], tot + 1000);
                }
    /* 2. p right at the 2^59 normalisation boundary: rm = floor/ceil(2^59 / sm) */
    for (long k = 0; k < 2000000; k++) {
        uint32_t sm = pick_m(); if (sm == 0) sm = 1u << 29;
        uint64_t q = (1ull << 59) / sm;
        uint32_t rm = (uint32_t)q + (rnd32() % 3) - 1;
        if (rm < (1u << 29) || rm >= (1u << 30)) continue;
        int sh = (int)(rnd32() % 70) - 3;
        uint64_t p = (uint64_t)sm * rm;
        int tot = sh + 12 + (p >= (1ull << 59) ? 30 : 29);
        int32_t se = (int32_t)(rnd32() % 200) - 100;
        one_legal(pick_v(), sm, se, rm, tot - se);
    }
    /* 3. random legal inputs, sh spread over -3..66 (weighted to 1..62) */
    for (long k = 0; k < VE_N; k++) {
        uint32_t sm = pick_m(), rm = pick_m();
        uint64_t p = (uint64_t)sm * rm;
        int sh = (rnd32() % 8) ? 1 + (int)(rnd32() % 62) : (int)(rnd32() % 70) - 3;
        int tot = sh + 12 + (p >= (1ull << 59) ? 30 : 29);
        int32_t se = (rnd32() % 4) ? (int32_t)(rnd32() % 128) - 64 : (int32_t)rnd32();
        one_legal(pick_v(), sm, se, rm, (int32_t)((uint32_t)tot - (uint32_t)se));
    }
    /* 4. wide domain: arbitrary 32-bit operands (illegal mantissas, wrapping exponents): A vs RTL mirror */
    for (long k = 0; k < VE_N / 4; k++) {
        int32_t v = pick_v();
        uint32_t sm = rnd32(), rm = rnd32();
        int32_t se = (int32_t)rnd32(), re = (int32_t)rnd32();
        if (k & 1) { int sh = 1 + (int)(rnd32() % 62); uint64_t p = (uint64_t)sm * rm;
                     re = (int32_t)((uint32_t)(sh + 12 + (p >= (1ull << 59) ? 30 : 29)) - (uint32_t)se); }
        int32_t a = model(rnd32() & 7, v, sm, se, rm, re), c = rtl_mirror(v, sm, (uint32_t)se, rm, (uint32_t)re);
        n_wide++;
        if (a != c) {
            bad_wide++;
            uint64_t p = (uint64_t)sm * rm;
            uint32_t m = p >= (1ull << 59) ? (uint32_t)(p >> 30) : (uint32_t)(p >> 29);
            if (m < (1u << 31)) bad_wide_legalm++;     /* would be a real model/RTL divergence */
            if (bad_wide < 4) printf("wide A!=C v=%d sm=%#x se=%d rm=%#x re=%d m=%#x A=%d C=%d\n", v, sm, se, rm, re, m, a, c);
        }
    }
    printf("legal: %lld cases, model vs RTL mirror mismatches %lld\n", n_leg, bad_ac);
    printf("legal, qref defined: %lld cases, vs apply32 %lld, vs apply %lld\n", n_ref, bad_ab, bad_ab64);
    printf("coverage: m==0 %lld | p<2^59 %lld, p>=2^59 %lld | sh<1 %lld, 1..31 %lld, 32..62 %lld, >62 %lld, sh in {1,31,32,62} %lld\n",
           bin_mz, bin_br[0], bin_br[1], bin_sh[0], bin_sh[1], bin_sh[2], bin_sh[3], bin_sh[4]);
    printf("wide: %lld cases, model vs RTL mirror mismatches %lld (with m < 2^31: %lld)\n", n_wide, bad_wide, bad_wide_legalm);
    int ok = !bad_ac && !bad_ab && !bad_ab64 && !bad_wide_legalm;
    printf("%s\n", ok ? "VE_ACCQ PASS" : "VE_ACCQ FAIL");
    return ok ? 0 : 1;
}
