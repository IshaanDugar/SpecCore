/* spin: runs for a controllable long time, to measure simulator speed.
   Change SPIN_N to set the length (cycles are roughly linear in SPIN_N).
   Build + run:  python tools/sc.py fw spin firmware/tests/spin.c
                 python tools/sc.py sim spin --max 5000000 */
#include "sc_io.h"

#ifndef SPIN_N
#define SPIN_N 220000u   /* iterations of the inner work */
#endif
#define INNER 1000u

volatile uint32_t seed0 = 0xC0FFEE11u;   /* volatile: the loop cannot be folded away */

int main(void) {
    uint32_t t0 = sc_cycles();
    uint32_t v = seed0, sum = 0;
    uint32_t outer = SPIN_N / INNER;
    for (uint32_t o = 0; o < outer; o++) {
        for (uint32_t i = 0; i < INNER; i++) {
            v = v * 1664525u + 1013904223u;
            sum += (v >> 9) ^ i;
        }
        sum = (sum << 1) | (sum >> 31);
    }
    sc_puts("spin n "); sc_putu(outer * INNER);
    sc_puts(" checksum "); sc_puthex(sum);
    sc_puts("\n");
    uint32_t t1 = sc_cycles();
    sc_puts("cycles "); sc_putu(t1 - t0); sc_puts("\n");
    return 0;
}
