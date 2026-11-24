/* matvec: small integer kernel shaped like our int8 inference.
   int8 weights (ROWS x COLS) times an int8 vector, int32 accumulation, then requantize
   to int8 with a fixed-point multiplier (high word of a 64-bit product = mulh),
   a rounding right shift and a clamp to [-128, 127].
   Also: a second pass over uint8 activations (lbu), byte stores (sb), and div/rem.
   Same source runs on the core and on the PC: python tools/sc.py run matvec firmware/tests/matvec.c */
#include "sc_io.h"

#define ROWS 16
#define COLS 32
#define QMULT 0x4A3D70A4   /* about 0.58 in Q31; mulh gives acc * 0.29 */
#define QSHIFT 7

static int8_t  W[ROWS * COLS];   /* weights */
static int8_t  x[COLS];          /* signed input vector */
static uint8_t u[COLS];          /* unsigned activations (as after a ReLU) */
static int8_t  y[ROWS];          /* requantized W*x */
static uint8_t z[ROWS];          /* requantized W*u, clamped to [0, 255] */

volatile uint32_t seed0 = 0x1234ABCDu;   /* volatile: stops the compiler folding the whole test */

static uint32_t lcg;
static inline uint32_t rnd(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return lcg >> 16;
}

/* acc * QMULT / 2^32, then divide by 2^QSHIFT with rounding */
static inline int32_t requant(int32_t acc) {
    int32_t hi = (int32_t)(((int64_t)acc * (int64_t)QMULT) >> 32);
    return (hi + (1 << (QSHIFT - 1))) >> QSHIFT;
}

int main(void) {
    uint32_t t0 = sc_cycles();
    lcg = seed0;

    for (int i = 0; i < ROWS * COLS; i++) W[i] = (int8_t)rnd();
    for (int j = 0; j < COLS; j++) { x[j] = (int8_t)rnd(); u[j] = (uint8_t)rnd(); }

    int32_t qsum = 0, rsum = 0;
    for (int i = 0; i < ROWS; i++) {
        const int8_t *w = &W[i * COLS];
        int32_t acc = 0, accu = 0;
        for (int j = 0; j < COLS; j++) {
            acc  += (int32_t)w[j] * (int32_t)x[j];    /* lb  * lb  */
            accu += (int32_t)w[j] * (int32_t)u[j];    /* lb  * lbu */
        }
        int32_t q = requant(acc);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        y[i] = (int8_t)q;

        int32_t qu = requant(accu);
        if (qu > 255) qu = 255;
        if (qu < 0) qu = 0;
        z[i] = (uint8_t)qu;

        int32_t d = (int32_t)u[i] + 1;                /* 1..256, never zero */
        qsum += acc / d;                              /* div */
        rsum += acc % d;                              /* rem */

        sc_puts("row "); sc_putu((uint32_t)i);
        sc_puts(" acc "); sc_puti(acc);
        sc_puts(" accu "); sc_puti(accu);
        sc_puts("\n");
    }

    sc_puts("y");
    for (int i = 0; i < ROWS; i++) { sc_putc(' '); sc_puti(y[i]); }
    sc_puts("\nz");
    for (int i = 0; i < ROWS; i++) { sc_putc(' '); sc_putu(z[i]); }
    sc_puts("\n");

    uint32_t chk = 0;
    for (int i = 0; i < ROWS; i++) chk = chk * 31u + (uint32_t)(int32_t)y[i] + z[i];
    sc_puts("qsum "); sc_puti(qsum);
    sc_puts(" rsum "); sc_puti(rsum);
    sc_puts(" chk "); sc_puthex(chk);
    sc_puts("\n");

    uint32_t t1 = sc_cycles();
    sc_puts("cycles "); sc_putu(t1 - t0); sc_puts("\n");
    return 0;
}
