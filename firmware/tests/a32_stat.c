/* PC only: run the K-sweep with apply32 call-site counters, then print, per source line of
   ref/qref.c, how many calls had shift < 32 (slow fallback), 32..62 (fast path), other.
   python tools/sc.py native a32s firmware/tests/a32_stat.c ref/qref.c -DQR_UNIT -DRR_KSWEEP -DQR_A32STAT */
#include <stdio.h>
#include <stdint.h>
extern uint32_t a32_stat[512][3];
int rr_main(int argc, char **argv);
int main(int argc, char **argv) {
    int rc = rr_main(argc, argv);
    printf("a32 line   sh<32   fast   other\n");
    for (int i = 0; i < 512; i++)
        if (a32_stat[i][0] + a32_stat[i][1] + a32_stat[i][2])
            printf("a32 %4d %7u %7u %7u\n", i, a32_stat[i][0], a32_stat[i][1], a32_stat[i][2]);
    return rc;
}
#define main rr_main
#include "../../ref/run_ref.c"
