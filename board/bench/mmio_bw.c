/* PS -> PL access speed through the hello app (board/hello): Cortex-A53 load/store over M_AXI_HPM0_FPD
   (128-bit, PL clock 100 MHz) into an AXI BRAM controller (32-bit) and an AXI GPIO register.
   This is the no-DMA baseline: how fast software alone can move data into / out of the fabric.

   Build on the board:  gcc -O2 -mcpu=cortex-a53 -o mmio_bw mmio_bw.c
   Run:                 ./mmio_bw            (needs the speccore_hello app loaded + speccore-devperm)

   UIO maps are Device memory (uncached), so every load waits for the AXI round trip and stores are
   posted (they return before the write lands). Reported per access and as MB/s. */
#include <arm_neon.h>
#include <fcntl.h>
#include <glob.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static volatile void *map_uio(const char *name, size_t *size) {
    glob_t g;
    glob("/sys/class/uio/uio*", 0, NULL, &g);
    for (size_t i = 0; i < g.gl_pathc; i++) {
        char p[256], n[64] = {0}, s[64] = {0};
        snprintf(p, sizeof p, "%s/name", g.gl_pathv[i]);
        FILE *f = fopen(p, "r");
        if (!f) continue;
        if (!fgets(n, sizeof n, f)) n[0] = 0;
        fclose(f);
        n[strcspn(n, "\n")] = 0;
        if (strcmp(n, name)) continue;
        snprintf(p, sizeof p, "%s/maps/map0/size", g.gl_pathv[i]);
        f = fopen(p, "r");
        if (!f || !fgets(s, sizeof s, f)) { fprintf(stderr, "no size for %s\n", name); exit(1); }
        fclose(f);
        *size = strtoul(s, NULL, 16);
        const char *base = strrchr(g.gl_pathv[i], '/') + 1;
        snprintf(p, sizeof p, "/dev/%s", base);
        int fd = open(p, O_RDWR | O_SYNC);
        if (fd < 0) { perror(p); exit(1); }
        void *m = mmap(NULL, *size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { perror("mmap"); exit(1); }
        globfree(&g);
        return m;
    }
    fprintf(stderr, "no UIO device named %s (load speccore_hello, run speccore-devperm)\n", name);
    exit(1);
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#define REPS 200
static size_t WORDS;     /* BRAM size in 32-bit words */

static void report(const char *what, double secs, size_t accesses, size_t bytes) {
    printf("%-34s %8.1f ns/access  %8.1f MB/s\n", what, secs * 1e9 / accesses, bytes / secs / 1e6);
}

int main(void) {
    size_t bsz, isz;
    volatile uint32_t *bram = map_uio("speccore_bram", &bsz);
    volatile uint32_t *id = map_uio("speccore_id", &isz);
    WORDS = bsz / 4;
    printf("BRAM %zu B, ID %#x\n", bsz, id[0]);
    if (id[0] != 0x5C0E0001u) { puts("MMIO BENCH FAIL: wrong ID"); return 1; }
    uint32_t sink = 0;
    double t;
    int fails = 0;

    /* 1. single-register read latency (each load waits for the AXI round trip) */
    t = now();
    for (int r = 0; r < 100000; r++) sink += id[0];
    report("ID register read (32-bit)", now() - t, 100000, 400000);

    /* 2. BRAM 32-bit stores (posted) then 32-bit loads */
    t = now();
    for (int r = 0; r < REPS; r++)
        for (size_t i = 0; i < WORDS; i++) bram[i] = (uint32_t)(i * 2654435761u + r);
    report("BRAM store 32-bit", now() - t, REPS * WORDS, REPS * WORDS * 4);
    t = now();
    for (int r = 0; r < REPS; r++)
        for (size_t i = 0; i < WORDS; i++) sink += bram[i];
    report("BRAM load 32-bit", now() - t, REPS * WORDS, REPS * WORDS * 4);
    for (size_t i = 0; i < WORDS; i++) fails += bram[i] != (uint32_t)(i * 2654435761u + REPS - 1);

    /* 3. 64-bit accesses */
    volatile uint64_t *b64 = (volatile uint64_t *)bram;
    t = now();
    for (int r = 0; r < REPS; r++)
        for (size_t i = 0; i < WORDS / 2; i++) b64[i] = ((uint64_t)(i + r) << 32) | (i ^ r);
    report("BRAM store 64-bit", now() - t, REPS * WORDS / 2, REPS * WORDS * 4);
    t = now();
    for (int r = 0; r < REPS; r++)
        for (size_t i = 0; i < WORDS / 2; i++) sink += (uint32_t)b64[i];
    report("BRAM load 64-bit", now() - t, REPS * WORDS / 2, REPS * WORDS * 4);
    for (size_t i = 0; i < WORDS / 2; i++)
        fails += b64[i] != (((uint64_t)(i + REPS - 1) << 32) | (i ^ (REPS - 1)));

    /* 4. 128-bit NEON accesses (one 16-byte beat on the 128-bit port if the core issues it as one) */
    uint32_t *b = (uint32_t *)bram;
    t = now();
    for (int r = 0; r < REPS; r++)
        for (size_t i = 0; i < WORDS; i += 4) {
            uint32x4_t v = {(uint32_t)(i + r), (uint32_t)(i + 1), (uint32_t)(i + 2), (uint32_t)(r)};
            vst1q_u32(b + i, v);
            __asm__ volatile("" ::: "memory");
        }
    report("BRAM store 128-bit (NEON)", now() - t, REPS * WORDS / 4, REPS * WORDS * 4);
    uint32x4_t acc = vdupq_n_u32(0);
    t = now();
    for (int r = 0; r < REPS; r++)
        for (size_t i = 0; i < WORDS; i += 4) {
            acc = vaddq_u32(acc, vld1q_u32(b + i));
            __asm__ volatile("" ::: "memory");
        }
    report("BRAM load 128-bit (NEON)", now() - t, REPS * WORDS / 4, REPS * WORDS * 4);
    sink += vgetq_lane_u32(acc, 0);
    for (size_t i = 0; i < WORDS; i += 4)
        fails += bram[i] != (uint32_t)(i + REPS - 1) || bram[i + 3] != (uint32_t)(REPS - 1);

    /* 5. memcpy in and out (what a naive "copy weights into the fabric" would do) */
    uint8_t *buf = malloc(bsz);
    for (size_t i = 0; i < bsz; i++) buf[i] = (uint8_t)(i * 7 + 3);
    t = now();
    for (int r = 0; r < REPS; r++) memcpy((void *)bram, buf, bsz);
    report("memcpy DDR -> BRAM", now() - t, REPS, REPS * bsz);
    uint8_t *back = malloc(bsz);
    t = now();
    for (int r = 0; r < REPS; r++) memcpy(back, (void *)bram, bsz);
    report("memcpy BRAM -> DDR", now() - t, REPS, REPS * bsz);
    fails += memcmp(buf, back, bsz) != 0;

    printf("(sink %u)\n", sink);
    printf(fails ? "MMIO BENCH FAIL (%d data errors)\n" : "MMIO BENCH PASS (data checked)\n", fails);
    return fails != 0;
}
