/* Bit-exact check of the C reference against ref/golden.h (from model/qmodel.py --golden),
   plus the cycle profile per pass. Same source on the PC (`sc.py native`) and the core (`sc.py fw`).

   Build-time switches (-D):
     RR_DRAFT=0/1   run the draft greedy check          (default 1)
     RR_TARGET=0/1  run the target greedy check         (default 1)
     RR_SPEC=0/1    run integer speculative decoding     (default 1)
     RR_NEW=n       new tokens per check (<= G_NEW)      (default G_NEW)
     RR_SMALL       use ref/golden_small.h (3-token prompt, 2 new tokens)
     RR_SPECLOG     print gamma / accepted / catch-up for every speculative round
     RR_GATE=t      D14 confidence gate (Q12 logit units, 4096 = 1.0): guess 0 is always submitted,
                    guess i >= 1 only while its own draft margin (top-1 minus top-2 logit) is > t,
                    up to RR_GMAX guesses (default 4). The pass that drafted a refused guess is
                    spent (= qmodel.qspec gate_mode "peek"). Checks against ref/golden_gate.h,
                    which must have been made with the same t and cap (G_GATE, G_GMAX).
     RR_KSWEEP      (QR_UNIT only) instead of the checks: one pass at K = 1, 2, 3, 5 for each model
                    after the prompt, buckets per K (shared vs per-lane cost)
   Lines starting with "cycles" differ between PC and core and are not compared. */
#include "qref.h"
#include "sc_io.h"
#ifdef RR_SMALL
#include "golden_small.h"     /* 3-token prompt, 2 new: short enough for xsim without VDOTK */
#define G_SPEC_PEEK 0         /* RR_GATE: with 2 new tokens a round has at most 1 guess, no gate */
#elif defined(RR_GATE)
#include "golden_gate.h"      /* same prompt as golden.h, spec rounds made with the gate */
#ifndef RR_GMAX
#define RR_GMAX 4
#endif
#if RR_GATE != G_GATE || RR_GMAX != G_GMAX
#error "RR_GATE / RR_GMAX differ from ref/golden_gate.h: rerun python model/qmodel.py --golden --gate T --gmax N"
#endif
#elif defined(RR_BSET)          /* board test set: one prompt of the 50-prompt set (tools/board_set.py) */
#include "golden_bset.h"
#elif defined(RR_SEED)           /* extra prompts: python model/qmodel.py --golden-extra SEED 8 16 golden_sSEED.h */
#if RR_SEED == 7
#include "golden_s7.h"
#elif RR_SEED == 11
#include "golden_s11.h"
#else
#error "no ref/golden_s<RR_SEED>.h"
#endif
#else
#include "golden.h"
#endif

#ifndef RR_DRAFT
#define RR_DRAFT 1
#endif
#ifndef RR_TARGET
#define RR_TARGET 1
#endif
#ifndef RR_SPEC
#define RR_SPEC 1
#endif
#ifdef QR_UNIT
#include "sc_isa.h"
#endif
#ifndef RR_NEW
#define RR_NEW G_NEW
#endif

#define CACHE_T (6 * 8 * 128 * (2 * 32 + 10))
#define CACHE_D (2 * 4 * 128 * (2 * 32 + 10))
static uint32_t cache_t[CACHE_T / 4], cache_d[CACHE_D / 4];
static qmodel_t T, D;

#ifdef SC_ON_CORE
#define IMG ((const uint8_t *)0x20000000u)
#else
static uint8_t *IMG;
static void load_img(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    IMG = malloc((size_t)n);
    if (fread(IMG, 1, (size_t)n, f) != (size_t)n) exit(2);
    fclose(f);
}
#endif

static int fails = 0;
#ifdef QR_MARGIN
static int32_t fmargin;              /* margin after the last token of the last feed() */
#endif

static void prof_line(const char *name, const qmodel_t *q, uint32_t steps) {
    static const char *nm[PR_N] = {"matvec", "attn", "normq", "other"};
    uint32_t tot = 0;
    sc_puts("cycles "); sc_puts(name); sc_puts(" per step:");
    for (int i = 0; i < PR_N; i++) {
        sc_putc(' '); sc_puts(nm[i]); sc_putc(' '); sc_putu(q->prof[i] / steps); tot += q->prof[i];
    }
    sc_puts(" total "); sc_putu(tot / steps); sc_puts(" (ctx "); sc_putu(q->n); sc_puts(")\n");
}

/* feed n tokens; returns the argmax after the last one (*lsum = its logit sum).
   QR_UNIT: up to 8 tokens per pass on the LLM unit (one weight stream for all of them). */
static int feed(qmodel_t *q, const int *toks, int n, int64_t *lsum, int *all) {
    int best = 0;
#ifdef QR_UNIT
    int b[8]; int64_t s[8];
    for (int i = 0; i < n; i += 8) {
        int k = n - i < 8 ? n - i : 8;
        qm_steps(q, toks + i, k, b, s);
        for (int j = 0; j < k; j++) if (all) all[i + j] = b[j];
        best = b[k - 1]; if (lsum) *lsum = s[k - 1];
#ifdef QR_MARGIN
        fmargin = q->margin[k - 1];
#endif
    }
#else
    for (int i = 0; i < n; i++) { best = qm_step(q, toks[i], lsum); if (all) all[i] = best; }
#ifdef QR_MARGIN
    fmargin = q->margin[0];
#endif
#endif
    return best;
}

/* plain greedy: prompt, then RR_NEW own picks; compare argmax and logit sums with the golden lists.
   "cycles plain" = from the last prompt token to the last new token (RR_NEW passes). */
static void greedy_check(const char *name, qmodel_t *q, const int16_t *gt, const int64_t *gs) {
    int64_t s; int tok = 0, bad = 0, p[G_PLEN];
    for (int i = 0; i < G_PLEN; i++) p[i] = g_prompt[i];
    q->n = 0;
    feed(q, p, G_PLEN - 1, 0, 0);
    uint32_t c0 = sc_cycles();
    tok = p[G_PLEN - 1];
    for (int k = 0; k < RR_NEW; k++) {
        for (int i = 0; i < PR_N; i++) q->prof[i] = 0;
        tok = feed(q, &tok, 1, &s, 0);
        if (tok != gt[k] || s != gs[k]) bad++;
        sc_puts(name); sc_puts(" tok "); sc_puti(tok); sc_puts(" lsum_lo ");
        sc_puthex((uint32_t)s); sc_puts(tok == gt[k] && s == gs[k] ? " ok\n" : " MISMATCH\n");
    }
    uint32_t c1 = sc_cycles();
    prof_line(name, q, 1);
    sc_puts("cycles plain "); sc_puts(name); sc_puts(" total "); sc_putu(c1 - c0); sc_puts("\n");
    fails += bad;
}

/* integer speculative decoding, same loop as qmodel.qspec. QR_UNIT: the verify pass is one
   K = g + 1 pass; draft catch-up rides in the same draft pass as the last real token. */
static void spec_check(void) {
    int seq[G_PLEN + G_NEW];
    int n = G_PLEN, end = G_PLEN + RR_NEW, rounds = 0, acc = 0, catchup = 0;
#ifdef RR_GATE
    int peek = 0;                                        /* drafted guesses the gate refused */
#endif
    for (int i = 0; i < G_PLEN; i++) seq[i] = g_prompt[i];
    T.n = 0; D.n = 0;
    feed(&T, seq, n - 1, 0, 0); feed(&D, seq, n - 1, 0, 0);
    uint32_t c0 = sc_cycles(), cd = 0, ct = 0;
    while (n < end) {
        int g = end - n - 1 < G_GAMMA ? end - n - 1 : G_GAMMA, guess[8], want[9], vin[9];
        uint32_t a0 = sc_cycles();
        int behind = n - 1 - (int)D.n;                   /* audit e3: tokens the draft has not seen */
        catchup += behind;
        int last = seq[n - 1];
        if (g > 0) {
            guess[0] = feed(&D, seq + D.n, behind + 1, 0, 0);   /* catch-up + last token, one pass */
#ifdef RR_GATE
            int gmax = g;                                /* G_GAMMA = G_GMAX in golden_gate.h */
            for (g = 1; g < gmax; g++) {                 /* D14 gate: submit guess g iff its margin > t */
                int t = feed(&D, &guess[g - 1], 1, 0, 0);
                if (fmargin <= RR_GATE) { peek++; break; }
                guess[g] = t;
            }
#else
            for (int i = 1; i < g; i++) guess[i] = feed(&D, &guess[i - 1], 1, 0, 0);
#endif
        } else if (behind > 0) {
            feed(&D, seq + D.n, behind, 0, 0);
        }
        uint32_t a1 = sc_cycles();
        uint32_t base = T.n;
        vin[0] = last;
        for (int i = 0; i < g; i++) vin[i + 1] = guess[i];
        feed(&T, vin, g + 1, 0, want);                   /* verify: K = g + 1 lanes */
        uint32_t a2 = sc_cycles();
        cd += a1 - a0; ct += a2 - a1;
        int ok = 0;
        while (ok < g && guess[ok] == want[ok]) ok++;
        for (int i = 0; i < ok; i++) seq[n++] = guess[i];
        seq[n++] = want[ok];
        qm_truncate(&T, base + 1 + ok);
        qm_truncate(&D, (uint32_t)(n - 1));
#ifdef RR_SPECLOG
        sc_puts("round gamma "); sc_putu((uint32_t)g); sc_puts(" accepted "); sc_putu((uint32_t)ok);
        sc_puts(" catchup "); sc_putu((uint32_t)behind); sc_puts("\n");
#endif
        rounds++; acc += ok;
    }
    uint32_t c1 = sc_cycles();
    int bad = 0;
    for (int k = 0; k < RR_NEW; k++) bad += seq[G_PLEN + k] != g_spec_tok[k];
    if (RR_NEW == G_NEW) bad += rounds != G_SPEC_ROUNDS || acc != G_SPEC_ACC || catchup != G_SPEC_CATCHUP;
#ifdef RR_GATE
    if (RR_NEW == G_NEW) bad += peek != G_SPEC_PEEK;
#endif
    sc_puts("spec gamma "); sc_putu(G_GAMMA);
#ifdef RR_GATE
    sc_puts(" gate "); sc_puti(RR_GATE);
#endif
    sc_puts(": rounds "); sc_putu(rounds);
    sc_puts(" accepted "); sc_putu(acc); sc_puts(" catchup "); sc_putu(catchup);
#ifdef RR_GATE
    sc_puts(" peek "); sc_putu((uint32_t)peek);
#endif
    sc_puts(bad ? " MISMATCH\n" : " ok\n");
    sc_puts("cycles spec total "); sc_putu(c1 - c0); sc_puts(" draft "); sc_putu(cd);
    sc_puts(" verify "); sc_putu(ct); sc_puts("\n");
    fails += bad;
}

#if defined(QR_UNIT) && defined(RR_KSWEEP)
/* Cost of one pass vs K (lanes), same context: prompt fed once, then for each K one pass over
   [last prompt token, golden greedy tokens...], buckets printed, cache rolled back. The argmax of
   lane l must be golden token l (checked). Splits each bucket into shared + per-lane cost. */
/* RR_KSWEEP_CTX=N (N > G_PLEN): the context before the pass is N-1 tokens, the prompt repeated
   (cost depends on the length, not the text). No golden answer then: the argmax of every lane is
   printed and `sc.py run` checks core == PC. */
#ifdef RR_KSWEEP_CTX
#define KS_CTX RR_KSWEEP_CTX
#else
#define KS_CTX G_PLEN
#endif
static void ksweep(const char *name, qmodel_t *q, const int16_t *gt) {
    static const int Ks[] = {1, 2, 3, 5};
    static const char *nm[PR_N] = {"matvec", "attn", "normq", "other"};
    static int p[KS_CTX + 8];
    int vin[8], best[8];
    for (int i = 0; i < KS_CTX + 8; i++) p[i] = g_prompt[i % G_PLEN];
    q->n = 0;
    feed(q, p, KS_CTX - 1, 0, 0);
    uint32_t n0 = q->n;
    for (unsigned i = 0; i < sizeof Ks / sizeof Ks[0]; i++) {
        int K = Ks[i], bad = 0;
        vin[0] = p[KS_CTX - 1];
        for (int l = 1; l < K; l++) vin[l] = KS_CTX == G_PLEN ? gt[l - 1] : p[KS_CTX - 1 + l];
        for (int b = 0; b < PR_N; b++) q->prof[b] = 0;
        qm_steps(q, vin, K, best, 0);
        if (KS_CTX == G_PLEN) for (int l = 0; l < K; l++) bad += best[l] != gt[l];
        sc_puts("ksweep "); sc_puts(name); sc_puts(" K "); sc_putu((uint32_t)K);
        if (KS_CTX != G_PLEN) { sc_puts(" argmax"); for (int l = 0; l < K; l++) { sc_putc(' '); sc_puti(best[l]); } }
        sc_puts(bad ? " MISMATCH\n" : " ok\n");
        uint32_t tot = 0;
        sc_puts("cycles ksweep "); sc_puts(name); sc_puts(" K "); sc_putu((uint32_t)K);
        for (int b = 0; b < PR_N; b++) {
            sc_putc(' '); sc_puts(nm[b]); sc_putc(' '); sc_putu(q->prof[b]); tot += q->prof[b];
        }
        sc_puts(" total "); sc_putu(tot); sc_puts(" (ctx before "); sc_putu(n0); sc_puts(")\n");
        qm_truncate(q, n0);
        fails += bad;
    }
}
#endif

int main(int argc, char **argv) {
#ifndef SC_ON_CORE
    load_img(argc > 1 ? argv[1] : "checkpoints/export/model.img");
#else
    (void)argc; (void)argv;
#endif
    const uint32_t *top = (const uint32_t *)IMG;
    if (qm_init(&T, IMG + top[0], (uint8_t *)cache_t) || qm_init(&D, IMG + top[1], (uint8_t *)cache_d)) {
        sc_puts("bad model image\n"); return 3;
    }
    if (qm_cache_bytes(IMG + top[0]) > sizeof cache_t || qm_cache_bytes(IMG + top[1]) > sizeof cache_d) {
        sc_puts("cache too small\n"); return 4;
    }
#if defined(QR_UNIT) && defined(RR_KSWEEP)
    ksweep("draft", &D, g_draft_tok);
    ksweep("target", &T, g_target_tok);
    sc_puts(fails ? "REF FAIL\n" : "REF PASS\n");
    return fails ? 1 : 0;
#endif
    if (RR_DRAFT) greedy_check("draft", &D, g_draft_tok, g_draft_lsum);
    if (RR_TARGET) greedy_check("target", &T, g_target_tok, g_target_lsum);
    if (RR_SPEC) spec_check();
    sc_puts(fails ? "REF FAIL\n" : "REF PASS\n");
    return fails ? 1 : 0;
}
