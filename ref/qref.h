/* C reference of the integer model (model/qmodel.py), bit-exact. Same source on the PC and the core.
   The model image comes from `python model/export.py` (checkpoints/export/model.img); its layout
   is documented in model_blob() in model/export.py. On the core the image sits at DDR_BASE. */
#ifndef QREF_H
#define QREF_H
#include <stdint.h>

#define QR_MAXL 6
#define QR_MAXC 256
#define QR_MAXH 8
#define QR_MAXCTX 128
#define QR_MAXHID 1024
#define QR_D 32                       /* head dim, both models */

/* QR_MARGIN: qm_step / qm_steps also keep the top-2 logit and leave margin[l] = top-1 minus
   top-2 logit (Q12; a tie with the argmax gives 0) of the last pass, lane l (qm_step: lane 0).
   Used by the D14 confidence gate (run_ref.c RR_GATE turns it on). Off: same code as before. */
#if defined(RR_GATE) && !defined(QR_MARGIN)
#define QR_MARGIN
#endif

typedef struct {
    uint32_t rows, cols;
    const uint32_t *m;                /* row scale mantissa, [2^29, 2^30) */
    const int32_t *e;                 /* row scale exponent: scale = m * 2^-e */
    const int8_t *w;                  /* rows x cols, row-major */
} qmat_t;

/* profile buckets (cycles on the core, clock ticks on the PC) */
enum { PR_MATVEC, PR_ATTN, PR_NORMQ, PR_OTHER, PR_N };

typedef struct {
    uint32_t vocab, C, L, H, ctx, hid, attn_m;
    int32_t attn_e;
    qmat_t emb, out, qkv[QR_MAXL], proj[QR_MAXL], fc1[QR_MAXL], fc2[QR_MAXL];
    const int16_t *cos, *sin;         /* (ctx, D/2) Q14 */
    const uint16_t *lut;              /* 256 x Q15 2^(-i/256) */
    /* KV cache: K (L, H, ctx, D); V transposed (L, H, D, ctx); one scale per (L, H, pos) */
    int8_t *k8, *v8t, *ke, *ve;
    uint32_t *km, *vm;
    uint32_t n;                       /* tokens in the cache */
    uint32_t prof[PR_N];
#ifdef QR_MARGIN
    int32_t margin[8];                /* per lane, last pass (see QR_MARGIN above) */
#endif
} qmodel_t;

/* blob = one model inside the image; cache = zeroed memory of qm_cache_bytes() */
int qm_init(qmodel_t *q, const uint8_t *blob, uint8_t *cache);
uint32_t qm_cache_bytes(const uint8_t *blob);
/* feed one token at position q->n; returns the argmax of the logits; *lsum (if not 0) gets
   the sum of all logits (checksum vs the Python model) */
int qm_step(qmodel_t *q, int tok, int64_t *lsum);
/* QR_UNIT builds: K tokens at positions n..n+K-1 in one pass on the LLM unit (K <= 8);
   best[l] = argmax after token l, lsum[l] = its logit sum (lsum may be 0) */
void qm_steps(qmodel_t *q, const int *toks, int K, int *best, int64_t *lsum);
static inline void qm_truncate(qmodel_t *q, uint32_t n) { if (n < q->n) q->n = n; }

#endif
