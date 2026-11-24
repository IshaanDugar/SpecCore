/* Directed tests for SETS / ACCQ (D13, ext/rtl/sc_vdotk.sv). `python tools/sc.py run accq firmware/tests/accq_test.c`
   diffs the core against the C model in sc_isa.h, and every result is also self-checked against the
   reference arithmetic below (ref/qref.c smul + apply), so a wrong C model can't hide a wrong RTL.
   Hazards (the reason ACCQ latches its operands in the first E cycle): the row scale arrives from
   a lw at distance 1 (load-use stall, then W->E forwarding) and 2, from a mul at distance 1 (M->E
   forwarding), and the ACCQ result is used by the very next instruction (forwarding out of ACCQ).
   No model image needed. */
#include "sc_io.h"
#include "sc_isa.h"

static int fails = 0, n = 0;
static void check(const char *what, int i, int32_t got, int32_t want) {
    n++;
    sc_puts(what); sc_puts(" "); sc_puti(i); sc_puts(" "); sc_puti(got);
    if (got != want) { sc_puts(" MISMATCH want "); sc_puti(want); fails++; }
    sc_puts("\n");
}

/* reference: smul then apply(v, m, e, 12), plain 64-bit C (same as ref/qref.c) */
static int32_t ref_q(int32_t v, uint32_t am, int32_t ae, uint32_t bm, int32_t be) {
    uint64_t p = (uint64_t)am * bm;
    if (!p) return 0;
    uint32_t m; int32_t e;
    if (p >= (1ull << 59)) { m = (uint32_t)(p >> 30); e = ae + be - 30; }
    else { m = (uint32_t)(p >> 29); e = ae + be - 29; }
    int32_t sh = e - 12;
    if (!m || sh > 62) return 0;
    return (int32_t)(((int64_t)v * (int64_t)m + (1ll << (sh - 1))) >> sh);
}

static uint32_t rng = 0x12345678u;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint32_t mant(void) { return 0x20000000u | (rnd() & 0x1FFFFFFFu); }

static volatile uint32_t cell[2];       /* row scale in memory, read with lw */

/* acc[0] = sum of dot4(w_i, bank0[i]) over nw words; returns that sum (plain C) */
static uint32_t bank0[8];
static int32_t load_acc(int nw) {
    int32_t s = 0;
    SC_SETK(1); SC_SETPA(0);
    for (int i = 0; i < nw; i++) {
        uint32_t w = rnd();
        SC_VDOTR(w);
        for (int b = 0; b < 4; b++) s += (int32_t)(int8_t)(w >> (8 * b)) * (int32_t)(int8_t)(bank0[i] >> (8 * b));
    }
    return s;
}

int main(void) {
    for (int i = 0; i < 8; i++) { bank0[i] = rnd(); SC_BSTORE(0, i, bank0[i]); }
    bank0[7] = 0x80808080u; SC_BSTORE(0, 7, bank0[7]);          /* -128 corner in the last word */
    for (int i = 0; i < 24; i++) {
        uint32_t am = (i % 7 == 6) ? 0 : mant(), bm = (i % 11 == 10) ? 0 : mant();
        int32_t ae = 21 + (int32_t)(rnd() % 32), be = 22 + (int32_t)(rnd() % 30);   /* shift 1..62 */
        SC_SETS(0, am, ae);
        int32_t v, want, got;

        /* (a) plain macro: whatever schedule the compiler picks */
        v = load_acc(1 + i % 8); want = ref_q(v, am, ae, bm, be);
        got = SC_ACCQ(0, bm, be);
        check("accq macro", i, got, want);

        /* (b) row scale from lw at distance 1 (load-use), (c) at distance 2 */
        cell[0] = bm; cell[1] = (uint32_t)be;
        v = load_acc(8); want = ref_q(v, am, ae, bm, be);
#ifdef SC_ON_CORE
        { uint32_t t0, t1;
          __asm__ volatile("lw %1, 0(%3)\n\tlw %2, 4(%3)\n\t.insn r 0x0B, 3, 64, %0, %1, %2"
                           : "=r"(got), "=&r"(t0), "=&r"(t1) : "r"(cell) : "memory"); }
#else
        got = SC_ACCQ(0, cell[0], (int32_t)cell[1]);
#endif
        check("accq lw d1", i, got, want);
        v = load_acc(3); want = ref_q(v, am, ae, bm, be);
#ifdef SC_ON_CORE
        { uint32_t t0, t1;
          __asm__ volatile("lw %1, 0(%3)\n\tlw %2, 4(%3)\n\taddi x0, x0, 0\n\t.insn r 0x0B, 3, 64, %0, %1, %2"
                           : "=r"(got), "=&r"(t0), "=&r"(t1) : "r"(cell) : "memory"); }
#else
        got = SC_ACCQ(0, cell[0], (int32_t)cell[1]);
#endif
        check("accq lw d2", i, got, want);

        /* (d) row mantissa from a mul at distance 1: bm = (bm / 3) * 3 + bm % 3, mul gives the top part */
        v = load_acc(5);
        { uint32_t a = bm >> 1, b = 2, bm2 = (bm >> 1) * 2;
          want = ref_q(v, am, ae, bm2, be);
#ifdef SC_ON_CORE
          uint32_t t0;
          __asm__ volatile("mul %1, %2, %3\n\t.insn r 0x0B, 3, 64, %0, %1, %4"
                           : "=r"(got), "=&r"(t0) : "r"(a), "r"(b), "r"(be));
#else
          got = SC_ACCQ(0, a * b, be);
#endif
        }
        check("accq mul d1", i, got, want);

        /* (e) ACCQ result used by the next instruction (forwarding out of the multi-cycle op) */
        v = load_acc(2); want = ref_q(v, am, ae, bm, be) + 12345;
#ifdef SC_ON_CORE
        { int32_t r;
          __asm__ volatile(".insn r 0x0B, 3, 64, %0, %2, %3\n\taddi %1, %0, 0\n\tli t6, 12345\n\tadd %1, %1, t6"
                           : "=&r"(r), "=&r"(got) : "r"(bm), "r"(be) : "t6"); }
#else
        got = SC_ACCQ(0, bm, be) + 12345;
#endif
        check("accq fwd", i, got, want);
    }
    /* Pipeline cases from the verification review (22 Nov) */
    for (int i = 0; i < 8; i++) {
        uint32_t am = mant(), bm = mant();
        int32_t ae = 21 + (int32_t)(rnd() % 32), be = 22 + (int32_t)(rnd() % 30);
        int32_t v, want, got, got2;
        /* (g) SETS immediately followed by ACCQ on the same lane */
        v = load_acc(4); want = ref_q(v, am, ae, bm, be);
#ifdef SC_ON_CORE
        __asm__ volatile(".insn r 0x0B, 1, 32, x0, %1, %2\n\t.insn r 0x0B, 3, 64, %0, %3, %4"
                         : "=r"(got) : "r"(am), "r"(ae), "r"(bm), "r"(be));
#else
        SC_SETS(0, am, ae); got = SC_ACCQ(0, bm, be);
#endif
        check("sets->accq", i, got, want);
        /* (h) VDOTR immediately followed by ACCQ (the last MAC must be in the latched acc) */
        { uint32_t w = rnd();
          v = load_acc(2);
          for (int b = 0; b < 4; b++) v += (int32_t)(int8_t)(w >> (8 * b)) * (int32_t)(int8_t)(bank0[2] >> (8 * b));
          want = ref_q(v, am, ae, bm, be);
#ifdef SC_ON_CORE
          __asm__ volatile(".insn r 0x0B, 2, 0, x0, %1, x0\n\t.insn r 0x0B, 3, 64, %0, %2, %3"
                           : "=r"(got) : "r"(w), "r"(bm), "r"(be));
#else
          SC_VDOTR(w); got = SC_ACCQ(0, bm, be);
#endif
        }
        check("vdotr->accq", i, got, want);
        /* (i) two ACCQs back to back on one lane: the second sees the cleared accumulator */
        v = load_acc(3); want = ref_q(v, am, ae, bm, be);
#ifdef SC_ON_CORE
        __asm__ volatile(".insn r 0x0B, 3, 64, %0, %2, %3\n\t.insn r 0x0B, 3, 64, %1, %2, %3"
                         : "=&r"(got), "=&r"(got2) : "r"(bm), "r"(be));
#else
        got = SC_ACCQ(0, bm, be); got2 = SC_ACCQ(0, bm, be);
#endif
        check("accq b2b first", i, got, want);
        check("accq b2b second", i, got2, 0);
    }
    /* (k) ACCQ on lane 7 while K = 1: lane 7 keeps the value it got at K = 8 */
    {
        uint32_t w = 0x7F80017Fu, b7 = 0x01FF7F80u, am = mant(), bm = mant();
        int32_t ae = 30, be = 40, v = 0;
        SC_BSTORE(7, 0, b7); SC_SETS(7, am, ae);
        /* banks are RAM without reset: give lanes 1..6 a defined word 0 too (else their acc goes X) */
        SC_BSTORE(1, 0, 0); SC_BSTORE(2, 0, 0); SC_BSTORE(3, 0, 0);
        SC_BSTORE(4, 0, 0); SC_BSTORE(5, 0, 0); SC_BSTORE(6, 0, 0);
        SC_SETK(8); SC_SETPA(0); SC_VDOTR(w);
        for (int b = 0; b < 4; b++) v += (int32_t)(int8_t)(w >> (8 * b)) * (int32_t)(int8_t)(b7 >> (8 * b));
        SC_SETK(1);
        check("accq lane7 K1", 0, SC_ACCQ(7, bm, be), ref_q(v, am, ae, bm, be));
        (void)SC_ACCRD(1); (void)SC_ACCRD(2); (void)SC_ACCRD(3); (void)SC_ACCRD(4); (void)SC_ACCRD(5); (void)SC_ACCRD(6);
        (void)SC_ACCRD(0);
    }
    /* (f) ACCQ clears the accumulator: a plain ACCRD right after reads 0 */
    (void)load_acc(4); (void)SC_ACCQ(0, 0x30000000u, 40);
    check("accq clears", 0, SC_ACCRD(0), 0);
    sc_puts(fails ? "ACCQ TEST FAIL " : "ACCQ TEST PASS "); sc_puti(n); sc_puts(" checks\n");
    sc_exit(fails ? 1 : 0);
}
