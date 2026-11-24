/* DDR bandwidth seen by the KV260's Cortex-A53 cores: STREAM-style kernels (copy, scale, add, triad)
   on arrays far larger than the caches, single thread and all cores (fork). Best of N repeats.
   Bytes counted the STREAM way (reads + writes the kernel does), so write-allocate traffic is not counted.
     gcc -O2 -o ddr_bw ddr_bw.c && ./ddr_bw [MB per array, default 64] [repeats, default 5]
   Prints one line per kernel per thread count, MB/s (measured). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void run(size_t n, int reps, double *best) {   /* best[k] = best seconds for kernel k */
    double *a = malloc(n * sizeof *a), *b = malloc(n * sizeof *b), *c = malloc(n * sizeof *c);
    if (!a || !b || !c) { perror("malloc"); exit(1); }
    for (size_t i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    for (int k = 0; k < 4; k++) best[k] = 1e30;
    for (int r = 0; r < reps; r++) {
        double t;
        t = now(); for (size_t i = 0; i < n; i++) c[i] = a[i];               t = now() - t; if (t < best[0]) best[0] = t;
        t = now(); for (size_t i = 0; i < n; i++) b[i] = 3.0 * c[i];         t = now() - t; if (t < best[1]) best[1] = t;
        t = now(); for (size_t i = 0; i < n; i++) c[i] = a[i] + b[i];        t = now() - t; if (t < best[2]) best[2] = t;
        t = now(); for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];  t = now() - t; if (t < best[3]) best[3] = t;
    }
    volatile double sink = a[n / 2] + b[n / 3] + c[n / 5]; (void)sink;
    free(a); free(b); free(c);
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? strtoul(argv[1], 0, 10) : 64;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    size_t n = mb * 1024 * 1024 / sizeof(double);
    static const char *nm[4] = {"copy", "scale", "add", "triad"};
    static const int words[4] = {2, 2, 3, 3};
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    printf("ddr_bw: %zu MB per array (3 arrays per process), %d repeats, best time; %d cores online\n", mb, reps, ncpu);
    for (int p = 1; p <= ncpu; p *= 2) {
        /* p processes run the same kernels at once; each reports its best; the sum is the total */
        double *res = mmap(0, sizeof(double) * 4 * p, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        for (int j = 0; j < p; j++) if (fork() == 0) { run(n, reps, res + 4 * j); _exit(0); }
        for (int j = 0; j < p; j++) wait(0);
        for (int k = 0; k < 4; k++) {
            double mbps = 0;
            for (int j = 0; j < p; j++) mbps += words[k] * n * sizeof(double) / res[4 * j + k] / 1e6;
            printf("threads %d  %-5s  %8.0f MB/s\n", p, nm[k], mbps);
        }
        munmap(res, sizeof(double) * 4 * p);
    }
    return 0;
}
