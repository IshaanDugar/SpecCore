/* Directed tests for the LLM unit (ext/rtl/sc_vdotk.sv). Prints results; `sc.py run ... --ddr`
   diffs the core against the C model in sc_isa.h. Also self-checks against plain C arithmetic,
   so a wrong C model can't hide a wrong RTL. Needs the model image (weights = stream source).
   Plan v3 directed list: FIFO empty at stream start / mid-row, back-to-back VDOTK, ACCRD right
   after VDOTK, branches around VDOTK, corner values (-128 x -128), K masking, WPOP order. */
#include "sc_io.h"
#include "sc_isa.h"

#ifdef SC_ON_CORE
#define IMG ((const uint8_t *)0x20000000u)
#else
static uint8_t *IMG;
#endif

static int fails = 0;
static void check(const char *what, int32_t got, int32_t want) {
    sc_puts(what); sc_puts(" "); sc_puti(got);
    if (got != want) { sc_puts(" MISMATCH want "); sc_puti(want); fails++; }
    sc_puts("\n");
}
static int32_t dot4(uint32_t a, uint32_t b) {
    int32_t s = 0;
    for (int i = 0; i < 4; i++) s += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return s;
}

static uint32_t act[8][64];             /* what we put in the banks, kept for the C check */

int main(int argc, char **argv) {
#ifndef SC_ON_CORE
    FILE *f = fopen(argc > 1 ? argv[1] : "checkpoints/export/model.img", "rb");
    if (!f) return 2;
    IMG = malloc(1 << 23);
    if (!fread(IMG, 1, 1 << 23, f)) return 2;
    fclose(f);
#else
    (void)argc; (void)argv;
#endif
    /* 1. fill banks: lane l word j = pattern; lane 7 word 0 = all -128 (corner) */
    for (int j = 0; j < 64; j++) {
        uint32_t s = 0x9E3779B9u * (uint32_t)(j + 1);
        act[0][j] = s; act[1][j] = s ^ 0x80808080u; act[2][j] = ~s; act[3][j] = s * 3u;
        act[4][j] = s >> 3; act[5][j] = 0x7F7F7F7Fu; act[6][j] = 0x80808080u; act[7][j] = s + 0x01010101u;
        SC_BSTORE(0, j, act[0][j]); SC_BSTORE(1, j, act[1][j]); SC_BSTORE(2, j, act[2][j]); SC_BSTORE(3, j, act[3][j]);
        SC_BSTORE(4, j, act[4][j]); SC_BSTORE(5, j, act[5][j]); SC_BSTORE(6, j, act[6][j]); SC_BSTORE(7, j, act[7][j]);
    }
    /* 2. VDOTR corner: -128 x -128 x 4 = 65536 on lane 6 (bank = 0x80808080), K = 8 */
    SC_SETK(8); SC_SETPA(0);
    SC_VDOTR(0x80808080u);
    int32_t c6 = SC_ACCRD(6);
    check("corner -128x-128 lane6", c6, 65536);
    /* clear the other lanes */
    (void)SC_ACCRD(0); (void)SC_ACCRD(1); (void)SC_ACCRD(2); (void)SC_ACCRD(3);
    (void)SC_ACCRD(4); (void)SC_ACCRD(5); (void)SC_ACCRD(7);

    /* 3. VDOTK, stream starts empty (stall), 64 back-to-back words, K = 5, ACCRD right after */
    const uint32_t *top = (const uint32_t *)IMG;
    const uint32_t *blob = (const uint32_t *)(IMG + top[1]);          /* draft blob */
    const uint32_t *emb = (const uint32_t *)((const uint8_t *)blob + blob[9]);
    const uint32_t *w = emb + 2 + 2 * emb[0];                          /* int8 rows, as words */
    SC_SETK(5); SC_SETPA(0);
    SC_WSTART(w, 64);
    for (int j = 0; j < 64; j++) SC_VDOTK();
    int32_t r[8];
    r[0] = SC_ACCRD(0); r[1] = SC_ACCRD(1); r[2] = SC_ACCRD(2); r[3] = SC_ACCRD(3); r[4] = SC_ACCRD(4);
    r[5] = SC_ACCRD(5); r[6] = SC_ACCRD(6); r[7] = SC_ACCRD(7);
    for (int l = 0; l < 8; l++) {
        int32_t want = 0;
        if (l < 5) for (int j = 0; j < 64; j++) want += dot4(w[j], act[l][j]);
        sc_puts("vdotk K5 lane "); sc_putu(l); check("", r[l], want);   /* lanes 5-7 must stay 0 */
    }
    /* 4. branches around VDOTK: data-dependent if inside the loop (mispredicts), K = 8 */
    SC_SETK(8); SC_SETPA(0);
    SC_WSTART(w + 64, 48);
    int32_t extra = 0;
    for (int j = 0; j < 48; j++) {
        SC_VDOTK();
        if (((uint32_t)j * 2654435761u) & 0x40000000u) extra += j; else extra -= 1;
    }
    int32_t s0 = SC_ACCRD(0), s7 = SC_ACCRD(7), want0 = 0, want7 = 0;
    for (int j = 0; j < 48; j++) { want0 += dot4(w[64 + j], act[0][j]); want7 += dot4(w[64 + j], act[7][j]); }
    check("branchy lane0", s0, want0);
    check("branchy lane7", s7, want7);
    check("branchy extra", extra, extra);
    (void)SC_ACCRD(1); (void)SC_ACCRD(2); (void)SC_ACCRD(3); (void)SC_ACCRD(4); (void)SC_ACCRD(5); (void)SC_ACCRD(6);
    /* 5. WPOP returns stream words in order, interleaved with VDOTK; SETP per lane */
    SC_SETK(1); SC_SETP(0, 10);
    SC_WSTART(w + 200, 6);
    uint32_t p0 = SC_WPOP();
    SC_VDOTK();
    uint32_t p2 = SC_WPOP();
    SC_VDOTK(); SC_VDOTK();
    uint32_t p5 = SC_WPOP();
    check("wpop0", (int32_t)p0, (int32_t)w[200]);
    check("wpop2", (int32_t)p2, (int32_t)w[202]);
    check("wpop5", (int32_t)p5, (int32_t)w[205]);
    check("setp lane0 from word 10",
          SC_ACCRD(0), dot4(w[201], act[0][10]) + dot4(w[203], act[0][11]) + dot4(w[204], act[0][12]));
    sc_puts(fails ? "VDOTK TEST FAIL\n" : "VDOTK TEST PASS\n");
    return fails;
}
