/* D16 DMA loopback benchmark (run as root on the KV260 with the speccore_dma app loaded).
   DDR --(AXI DMA MM2S, S_AXI_HP0_FPD 128 bit)--> PL stream --(S2MM)--> DDR, PL clock 100 MHz.
   Buffers: one 4 MB buffer from the CMA dma-heap /dev/dma_heap/reserved (physically contiguous, in low
   DDR 0x37400000-0x75bfffff on this board, inside the DMA's HP0_DDR_LOW window; hugepages landed in
   high DDR, which this PS configuration does not map on HP0), split into src and dst halves.
   Physical address from /proc/self/pagemap (root); contiguity checked page by page.
   Caches: HP0 is not coherent, so the source is cleaned and the destination invalidated with
   DC CIVAC (allowed at EL0 on Linux arm64) around every transfer that is checked.
   Prints MB/s for several transfer sizes (each direction moves the same bytes at the same time)
   and checks every byte of the checked transfers.
     gcc -O2 -o dma_bw dma_bw.c && sudo ./dma_bw                                                 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-heap.h>
#include <time.h>
#include <unistd.h>

#define DMA_BASE 0xA0000000u
#define ID_BASE  0xA0010000u
#define HUGE (2u << 20)
enum { MM2S_CR = 0x00, MM2S_SR = 0x04, MM2S_SA = 0x18, MM2S_SA_MSB = 0x1C, MM2S_LEN = 0x28,
       S2MM_CR = 0x30, S2MM_SR = 0x34, S2MM_DA = 0x48, S2MM_DA_MSB = 0x4C, S2MM_LEN = 0x58 };

static volatile uint32_t *R;
static uint32_t rd(int o) { return R[o / 4]; }
static void wr(int o, uint32_t v) { R[o / 4] = v; }

static uint64_t phys_of(void *va) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    uint64_t e = 0, pg = (uint64_t)sysconf(_SC_PAGESIZE);
    if (fd < 0 || pread(fd, &e, 8, ((uint64_t)(uintptr_t)va / pg) * 8) != 8) { perror("pagemap"); exit(2); }
    close(fd);
    if (!(e >> 63) || !(e & ((1ull << 55) - 1))) { fprintf(stderr, "page not present / no PFN (root?)\n"); exit(2); }
    return (e & ((1ull << 55) - 1)) * pg + ((uintptr_t)va % pg);
}
static void flush(void *p, size_t n) {          /* clean + invalidate to the point of coherency */
    for (uintptr_t a = (uintptr_t)p & ~63ul; a < (uintptr_t)p + n; a += 64) __asm__ volatile("dc civac, %0" ::"r"(a) : "memory");
    __asm__ volatile("dsb sy" ::: "memory");
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static int xfer(uint64_t src, uint64_t dst, uint32_t n) {
    wr(S2MM_DA, (uint32_t)dst); wr(S2MM_DA_MSB, (uint32_t)(dst >> 32)); wr(S2MM_LEN, n);
    wr(MM2S_SA, (uint32_t)src); wr(MM2S_SA_MSB, (uint32_t)(src >> 32)); wr(MM2S_LEN, n);   /* starts */
    for (long spin = 0; spin < 200000000L; spin++) {
        uint32_t s = rd(S2MM_SR), m = rd(MM2S_SR);
        if ((s | m) & 0x70) { fprintf(stderr, "DMA error MM2S_SR %08x S2MM_SR %08x\n", m, s); return -1; }
        if ((s & 0x2) && (m & 0x2)) return 0;          /* both idle = done */
    }
    fprintf(stderr, "DMA timeout MM2S_SR %08x S2MM_SR %08x\n", rd(MM2S_SR), rd(S2MM_SR));
    return -1;
}

int main(void) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem (run as root)"); return 2; }
    R = mmap(0, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, DMA_BASE);
    volatile uint32_t *ID = mmap(0, 0x1000, PROT_READ, MAP_SHARED, fd, ID_BASE);
    if (R == MAP_FAILED || ID == MAP_FAILED) { perror("mmap regs"); return 2; }
    printf("ID 0x%08x (want 0x5c0e0002)\n", ID[0]);
    if (ID[0] != 0x5c0e0002u) return 3;

    int hfd = open("/dev/dma_heap/reserved", O_RDWR);
    if (hfd < 0) { perror("/dev/dma_heap/reserved"); return 2; }
    struct dma_heap_allocation_data al = {.len = 2 * HUGE, .fd_flags = O_RDWR | O_CLOEXEC};
    if (ioctl(hfd, DMA_HEAP_IOCTL_ALLOC, &al) < 0) { perror("DMA_HEAP_IOCTL_ALLOC"); return 2; }
    uint8_t *buf = mmap(0, 2 * HUGE, PROT_READ | PROT_WRITE, MAP_SHARED, (int)al.fd, 0);
    if (buf == MAP_FAILED) { perror("mmap dma-buf"); return 2; }
    memset(buf, 0, 2 * HUGE);                          /* fault every page in */
    uint8_t *src = buf, *dst = buf + HUGE;
    uint64_t ps = phys_of(src), pd = phys_of(dst);
    long pg = sysconf(_SC_PAGESIZE);
    for (long o = 0; o < 2L * HUGE; o += pg)
        if (phys_of(buf + o) != ps + (uint64_t)o) { fprintf(stderr, "buffer not contiguous at +%ld\n", o); return 5; }
    printf("src phys 0x%llx, dst phys 0x%llx\n", (unsigned long long)ps, (unsigned long long)pd);
    if (ps + 2ull * HUGE > 0x80000000ull) {   /* the DMA maps HP0_DDR_LOW (0-2 GB) only */
        fprintf(stderr, "buffer above 2 GB (DMA sees DDR_LOW only); retry or reserve hugepages early\n");
        return 5;
    }

    wr(MM2S_CR, 4); wr(S2MM_CR, 4);                     /* reset */
    for (int i = 0; i < 1000 && (rd(MM2S_CR) & 4); i++) usleep(100);
    wr(MM2S_CR, 1); wr(S2MM_CR, 1);                     /* run, no interrupts */

    int bad = 0;
    /* correctness: 3 patterns over the full 2 MB */
    for (int p = 0; p < 3; p++) {
        for (uint32_t i = 0; i < HUGE / 4; i++) ((uint32_t *)src)[i] = p == 0 ? i : p == 1 ? ~i : i * 2654435761u;
        memset(dst, 0xEE, HUGE);
        flush(src, HUGE); flush(dst, HUGE);
        if (xfer(ps, pd, HUGE)) return 4;
        flush(dst, HUGE);
        int b = 0;
        for (uint32_t i = 0; i < HUGE / 4; i++) b += ((uint32_t *)dst)[i] != ((uint32_t *)src)[i];
        printf("pattern %d: %d of %u words wrong\n", p, b, HUGE / 4);
        bad += b;
    }
    /* bandwidth: time many back-to-back transfers per size (no cache maintenance in the loop) */
    uint32_t sizes[] = {4096, 16384, 65536, 262144, 1048576, HUGE};
    for (unsigned k = 0; k < sizeof sizes / sizeof *sizes; k++) {
        uint32_t n = sizes[k];
        int reps = (int)(256u * 1024 * 1024 / n); if (reps > 20000) reps = 20000;
        double t0 = now();
        for (int r = 0; r < reps; r++) if (xfer(ps, pd, n)) return 4;
        double dt = now() - t0;
        printf("size %8u B x %5d: %8.1f MB/s per direction (%.2f us per transfer)\n",
               n, reps, (double)n * reps / dt / 1e6, dt / reps * 1e6);
    }
    printf(bad ? "DMA BENCH FAIL\n" : "DMA BENCH PASS\n");
    return bad ? 1 : 0;
}
