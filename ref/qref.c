/* Integer model, C version. Every function mirrors one in model/qmodel.py, in the same order,
   so the outputs are bit-identical (checked by ref/run_ref.c against ref/golden.h). */
#include "qref.h"
#include "sc_io.h"

#define M_LO (1u << 29)
#define M_HI (1u << 30)
#define EPS_Q24 168

/* ---------------- the number system (qmodel.py: norm_scale, smul, apply, quant) ---------------- */
static int bitlen64(uint64_t x) { return x ? 64 - __builtin_clzll(x) : 0; }

static void norm_scale(uint64_t num, uint64_t den, uint32_t *m, int32_t *e) {
    if (!num) { *m = 0; *e = 0; return; }
    int32_t ee = 29 - (bitlen64(num) - bitlen64(den));
    for (;;) {
        uint64_t q = ee >= 0 ? (num << ee) / den : num / (den << -ee);
        if (q < M_LO) ee++;
        else if (q >= M_HI) ee--;
        else { *m = (uint32_t)q; *e = ee; return; }
    }
}

static inline void smul(uint32_t am, int32_t ae, uint32_t bm, int32_t be, uint32_t *m, int32_t *e) {
    uint64_t p = (uint64_t)am * bm;
    if (!p) { *m = 0; *e = 0; return; }
    if (p >= (1ull << 59)) { *m = (uint32_t)(p >> 30); *e = ae + be - 30; }
    else { *m = (uint32_t)(p >> 29); *e = ae + be - 29; }
}

static inline int64_t apply(int64_t v, uint32_t m, int32_t e, int frac) {
    int32_t sh = e - frac;              /* > 0 whenever m != 0 (asserted in the Python model) */
    if (!m || sh > 62) return 0;
    return (v * (int64_t)m + (1ll << (sh - 1))) >> sh;
}

/* Same value as (int32_t)apply(v, m, e, frac) for any 32-bit v and m < 2^30 (always true here:
   every scale mantissa is 0 or in [2^29, 2^30)). Fast path when the shift sh is 32..62: the
   product is a 32x32->64 multiply (mul + mulh on RV32IM, not the 3-multiply 64x64 GCC emits for
   apply), and since floor(floor(x / 2^32) / 2^(sh-32)) = floor(x / 2^sh), only the high word
   is shifted (one 32-bit sra instead of a 64-bit variable shift with branches). Bit-exact. */
#ifdef QR_A32STAT                     /* PC only: how often each call site takes the fast path */
uint32_t a32_stat[512][3];             /* [source line][sh < 32, 32..62, other] */
#define apply32(v, m, e, f) apply32_(v, m, e, f, __LINE__)
static inline int32_t apply32_(int32_t v, uint32_t m, int32_t e, int frac, int line) {
    int32_t sh = e - frac;
    a32_stat[line][sh < 32 ? 0 : sh <= 62 ? 1 : 2]++;
#else
static inline int32_t apply32(int32_t v, uint32_t m, int32_t e, int frac) {
    int32_t sh = e - frac;
#endif
#ifndef QR_SLOWQ
    if (m && sh >= 32 && sh <= 62) {
        int64_t p = (int64_t)v * (int32_t)m + (1ll << (sh - 1));
        return (int32_t)(p >> 32) >> (sh - 32);
    }
    if (m && sh >= 1 && sh <= 31) {    /* low 32 bits of p >> sh, built from the two halves */
        uint64_t p = (uint64_t)((int64_t)v * (int32_t)m) + (1ull << (sh - 1));
        uint32_t lo = (uint32_t)p, hi = (uint32_t)(p >> 32);
        return (int32_t)((hi << (32 - sh)) | (lo >> sh));
    }
#endif
    return (int32_t)apply(v, m, e, frac);
}

static inline int8_t clamp8(int64_t v) { return (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }
static inline int8_t clamp8_32(int32_t v) { return (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }

/* q[i] = clamp8(apply(v[i], m, e, 0)) for the whole vector, with the fast-path test done once */
static void quant_vals(const int32_t *v, uint32_t n, uint32_t m, int32_t e, int8_t *q) {
#ifndef QR_SLOWQ
    if (m && e >= 32 && e <= 62) {
        int64_t rnd = 1ll << (e - 1);
        int32_t s2 = e - 32, mm = (int32_t)m;
        for (uint32_t i = 0; i < n; i++)
            q[i] = clamp8_32((int32_t)(((int64_t)v[i] * mm + rnd) >> 32) >> s2);
        return;
    }
#endif
    for (uint32_t i = 0; i < n; i++) q[i] = clamp8(apply(v[i], m, e, 0));
}

static uint32_t amax32(const int32_t *v, uint32_t n) {
    uint32_t a = 0;
    for (uint32_t i = 0; i < n; i++) { uint32_t t = v[i] < 0 ? (uint32_t)-v[i] : (uint32_t)v[i]; if (t > a) a = t; }
    return a;
}

static void quant(const int32_t *v, uint32_t n, int frac, int8_t *q, uint32_t *m, int32_t *e) {
    uint32_t a = amax32(v, n);
    if (!a) { for (uint32_t i = 0; i < n; i++) q[i] = 0; *m = 0; *e = 0; return; }
    uint32_t rm; int32_t re;
    norm_scale(127, a, &rm, &re);
    quant_vals(v, n, rm, re, q);
    norm_scale(a, 127, m, e);
    *e += frac;
}

static uint32_t isqrt64(uint64_t n) {       /* floor(sqrt(n)), bit by bit */
    uint64_t r = 0, bit = 1ull << 62;
    while (bit > n) bit >>= 2;
    while (bit) {
        if (n >= r + bit) { n -= r + bit; r = (r >> 1) + bit; } else r >>= 1;
        bit >>= 2;
    }
    return (uint32_t)r;
}

/* RMSNorm (gain folded into the next matrix) + per-token int8 */
static void rmsquant(const int32_t *x, uint32_t n, int8_t *q, uint32_t *m, int32_t *e) {
    uint64_t ss = 0;
    for (uint32_t i = 0; i < n; i++) ss += (uint64_t)((int64_t)x[i] * x[i]);
    uint32_t rms = isqrt64(ss / n + EPS_Q24);
    uint32_t a = amax32(x, n);
    if (!a) { for (uint32_t i = 0; i < n; i++) q[i] = 0; *m = 0; *e = 0; return; }
    uint32_t rm; int32_t re;
    norm_scale(127, a, &rm, &re);
    quant_vals(x, n, rm, re, q);
    norm_scale(a, 127ull * rms, m, e);
}

/* out[r] = Q12 of (W8 @ a8) with scale = a's scale * row scale. This is the weight stream. */
static void matvec(const qmat_t *W, const int8_t *a, uint32_t am, int32_t ae, int32_t *out) {
    const int8_t *w = W->w;
    for (uint32_t r = 0; r < W->rows; r++, w += W->cols) {
        int32_t acc = 0;
        for (uint32_t j = 0; j < W->cols; j++) acc += w[j] * a[j];
        uint32_t m; int32_t e;
        smul(am, ae, W->m[r], W->e[r], &m, &e);
        out[r] = apply32(acc, m, e, 12);
    }
}

/* ---------------- model ---------------- */
static const uint32_t *U32(const uint8_t *b, uint32_t off) { return (const uint32_t *)(b + off); }

static void mat_at(qmat_t *M, const uint8_t *b, uint32_t off) {
    const uint32_t *h = U32(b, off);
    M->rows = h[0]; M->cols = h[1];
    M->m = h + 2;
    M->e = (const int32_t *)(h + 2 + M->rows);
    M->w = (const int8_t *)(h + 2 + 2 * M->rows);
}

uint32_t qm_cache_bytes(const uint8_t *blob) {
    const uint32_t *h = U32(blob, 0);
    uint32_t lht = h[3] * h[4] * h[5];        /* L * H * ctx */
    return 2 * lht * QR_D + 2 * lht + 8 * lht;
}

int qm_init(qmodel_t *q, const uint8_t *blob, uint8_t *cache) {
    const uint32_t *h = U32(blob, 0);
    if (h[0] != 0x31514353u) return -1;
    q->vocab = h[1]; q->C = h[2]; q->L = h[3]; q->H = h[4]; q->ctx = h[5]; q->hid = h[6] * h[2];
    q->attn_m = h[7]; q->attn_e = (int32_t)h[8];
    if (q->L > QR_MAXL || q->C > QR_MAXC || q->C / q->H != QR_D || q->ctx > QR_MAXCTX) return -2;
    mat_at(&q->emb, blob, h[9]); mat_at(&q->out, blob, h[10]);
    q->cos = (const int16_t *)(blob + h[11]); q->sin = (const int16_t *)(blob + h[12]);
    q->lut = (const uint16_t *)(blob + h[13]);
    for (uint32_t i = 0; i < q->L; i++) {
        mat_at(&q->qkv[i], blob, h[16 + 4 * i]); mat_at(&q->proj[i], blob, h[17 + 4 * i]);
        mat_at(&q->fc1[i], blob, h[18 + 4 * i]); mat_at(&q->fc2[i], blob, h[19 + 4 * i]);
    }
    uint32_t lht = q->L * q->H * q->ctx;
    q->km = (uint32_t *)cache; q->vm = q->km + lht;          /* word arrays first (alignment) */
    q->k8 = (int8_t *)(q->vm + lht); q->v8t = q->k8 + lht * QR_D;
    q->ke = q->v8t + lht * QR_D; q->ve = q->ke + lht;
    q->n = 0;
    for (int i = 0; i < PR_N; i++) q->prof[i] = 0;
    return 0;
}

static void rope(const qmodel_t *q, int32_t *x, uint32_t pos) {   /* x: H x D, pairs (0,1),(2,3).. */
    const int16_t *c = q->cos + pos * (QR_D / 2), *s = q->sin + pos * (QR_D / 2);
    for (uint32_t h = 0; h < q->H; h++, x += QR_D)
        for (uint32_t i = 0; i < QR_D / 2; i++) {
            int64_t x1 = x[2 * i], x2 = x[2 * i + 1];
            x[2 * i] = (int32_t)((x1 * c[i] - x2 * s[i] + (1 << 13)) >> 14);
            x[2 * i + 1] = (int32_t)((x1 * s[i] + x2 * c[i] + (1 << 13)) >> 14);
        }
}

/* one head, query at pos over cache rows 0..pos -> y (D) Q12 */
static void attend(qmodel_t *q, uint32_t li, uint32_t h, uint32_t pos,
                   const int8_t *q8, uint32_t qm, int32_t qe, int32_t *y) {
    static int32_t s[QR_MAXCTX], wv[QR_MAXCTX];
    static int8_t w8[QR_MAXCTX];
    uint32_t base = (li * q->H + h) * q->ctx, T = pos + 1;
    const int8_t *k = q->k8 + base * QR_D;
    uint32_t c0m, cm; int32_t c0e, ce;
    smul(qm, qe, q->attn_m, q->attn_e, &c0m, &c0e);
    int32_t mx = 0;
    for (uint32_t t = 0; t < T; t++, k += QR_D) {
        int32_t dot = 0;
        for (uint32_t d = 0; d < QR_D; d++) dot += q8[d] * k[d];
        smul(c0m, c0e, q->km[base + t], (int32_t)q->ke[base + t], &cm, &ce);
        s[t] = apply32(dot, cm, ce, 12);
        if (t == 0 || s[t] > mx) mx = s[t];
    }
    uint32_t tot = 0;
    for (uint32_t t = 0; t < T; t++) {
        uint32_t d = (uint32_t)(mx - s[t]), n = d >> 12;
        uint32_t p = n >= 31 ? 0 : (uint32_t)q->lut[(d & 4095) >> 4] >> n;
        tot += p;
        wv[t] = apply32((int32_t)p, q->vm[base + t], (int32_t)q->ve[base + t], 16);
    }
    uint32_t wm, ym, rm; int32_t we, ye, re;
    quant(wv, T, 16, w8, &wm, &we);
    norm_scale(1, tot, &rm, &re);
    smul(wm, we, rm, re, &ym, &ye);
    const int8_t *vt = q->v8t + base * QR_D;              /* (D, ctx): row d = V[.., d] over t */
    for (uint32_t d = 0; d < QR_D; d++, vt += q->ctx) {
        int32_t acc = 0;
        for (uint32_t t = 0; t < T; t++) acc += w8[t] * vt[t];
        y[d] = apply32(acc, ym, ye, 12);
    }
}

#define TIC(b) do { uint32_t _t = sc_cycles(); q->prof[cur] += _t - t0; t0 = _t; cur = (b); } while (0)

int qm_step(qmodel_t *q, int tok, int64_t *lsum) {
    static int32_t x[QR_MAXC], big[3 * QR_MAXC > QR_MAXHID ? 3 * QR_MAXC : QR_MAXHID], y[QR_MAXC];
    static int8_t a[QR_MAXHID];
    uint32_t C = q->C, H = q->H, pos = q->n, m; int32_t e;
    uint32_t t0 = sc_cycles(), cur = PR_OTHER;
    for (uint32_t i = 0; i < C; i++)
        x[i] = (int32_t)apply(q->emb.w[tok * C + i], q->emb.m[tok], q->emb.e[tok], 12);
    for (uint32_t li = 0; li < q->L; li++) {
        TIC(PR_NORMQ); rmsquant(x, C, a, &m, &e);
        TIC(PR_MATVEC); matvec(&q->qkv[li], a, m, e, big);
        TIC(PR_OTHER); rope(q, big, pos); rope(q, big + C, pos);
        TIC(PR_ATTN);
        for (uint32_t h = 0; h < H; h++) {
            uint32_t base = (li * H + h) * q->ctx, km, vm, qm; int32_t ke, ve, qe;
            static int8_t q8[QR_D], v8[QR_D];
            quant(big + C + h * QR_D, QR_D, 12, q->k8 + (base + pos) * QR_D, &km, &ke);
            quant(big + 2 * C + h * QR_D, QR_D, 12, v8, &vm, &ve);
            for (uint32_t d = 0; d < QR_D; d++) q->v8t[(base * QR_D) + d * q->ctx + pos] = v8[d];
            q->km[base + pos] = km; q->ke[base + pos] = (int8_t)ke;
            q->vm[base + pos] = vm; q->ve[base + pos] = (int8_t)ve;
            quant(big + h * QR_D, QR_D, 12, q8, &qm, &qe);
            attend(q, li, h, pos, q8, qm, qe, y + h * QR_D);
        }
        TIC(PR_NORMQ); quant(y, C, 12, a, &m, &e);
        TIC(PR_MATVEC); matvec(&q->proj[li], a, m, e, big);
        TIC(PR_OTHER); for (uint32_t i = 0; i < C; i++) x[i] += big[i];
        TIC(PR_NORMQ); rmsquant(x, C, a, &m, &e);
        TIC(PR_MATVEC); matvec(&q->fc1[li], a, m, e, big);
        TIC(PR_NORMQ);
        for (uint32_t i = 0; i < q->hid; i++) if (big[i] < 0) big[i] = 0;
        quant(big, q->hid, 12, a, &m, &e);
        TIC(PR_MATVEC); matvec(&q->fc2[li], a, m, e, y);
        TIC(PR_OTHER); for (uint32_t i = 0; i < C; i++) x[i] += y[i];
    }
    TIC(PR_NORMQ); rmsquant(x, C, a, &m, &e);
    TIC(PR_MATVEC);
    /* output layer: same as matvec, but keep only the argmax (first index on ties) and the sum */
    int best = 0; int32_t bv = 0; int64_t sum = 0;
#ifdef QR_MARGIN
    int32_t b2 = INT32_MIN;                              /* second largest logit (duplicates count) */
#endif
    const int8_t *w = q->out.w;
    for (uint32_t r = 0; r < q->vocab; r++, w += C) {
        int32_t acc = 0;
        for (uint32_t j = 0; j < C; j++) acc += w[j] * a[j];
        uint32_t mm; int32_t ee;
        smul(m, e, q->out.m[r], q->out.e[r], &mm, &ee);
        int32_t lg = apply32(acc, mm, ee, 12);
        sum += lg;
#ifdef QR_MARGIN
        if (r == 0) bv = lg;
        else if (lg > bv) { b2 = bv; bv = lg; best = (int)r; }
        else if (lg > b2) b2 = lg;
#else
        if (r == 0 || lg > bv) { bv = lg; best = (int)r; }
#endif
    }
    TIC(PR_OTHER);
#ifdef QR_MARGIN
    q->margin[0] = q->vocab > 1 ? bv - b2 : 0;
#endif
    if (lsum) *lsum = sum;
    q->n = pos + 1;
    return best;
}

/* ======================================================================================
   K tokens in lockstep on the LLM unit (QR_UNIT builds): one weight stream feeds K lanes
   (VDOTK); attention reads each K/V word once with lw and feeds it to all K lanes (VDOTR).
   Token l sits at position q->n + l and sees cache rows 0..n+l only (later lanes rows are
   computed but ignored: their scores are never read and their attention weights are 0).
   Bit-exact with qm_step token by token: every per-lane number goes through the same code.
   ====================================================================================== */
#ifdef QR_UNIT
#include "sc_isa.h"

static inline void bst(int l, uint32_t a, uint32_t w) {
    switch (l) {
    case 0: SC_BSTORE(0, a, w); break; case 1: SC_BSTORE(1, a, w); break;
    case 2: SC_BSTORE(2, a, w); break; case 3: SC_BSTORE(3, a, w); break;
    case 4: SC_BSTORE(4, a, w); break; case 5: SC_BSTORE(5, a, w); break;
    case 6: SC_BSTORE(6, a, w); break; default: SC_BSTORE(7, a, w); break;
    }
}
static inline void accrd_k(int32_t *r, int K) {      /* read and clear lanes 0..K-1 */
    switch (K) {
    case 8: r[7] = SC_ACCRD(7); /* fall through */
    case 7: r[6] = SC_ACCRD(6); /* fall through */
    case 6: r[5] = SC_ACCRD(5); /* fall through */
    case 5: r[4] = SC_ACCRD(4); /* fall through */
    case 4: r[3] = SC_ACCRD(3); /* fall through */
    case 3: r[2] = SC_ACCRD(2); /* fall through */
    case 2: r[1] = SC_ACCRD(1); /* fall through */
    default: r[0] = SC_ACCRD(0);
    }
}
static void bank_fill(int l, const int8_t *v, uint32_t n, uint32_t at) {   /* n bytes -> words at word at */
#ifdef QR_ACCQ
    if (!((uintptr_t)v & 3)) {           /* little-endian: the packed word is just the aligned load */
        const uint32_t *wv = (const uint32_t *)v;
        for (uint32_t j = 0; j < n / 4; j++) bst(l, at + j, wv[j]);
        return;
    }
#endif
    for (uint32_t j = 0; j < n / 4; j++) {
        uint32_t w = (uint8_t)v[4 * j] | (uint32_t)(uint8_t)v[4 * j + 1] << 8 |
                     (uint32_t)(uint8_t)v[4 * j + 2] << 16 | (uint32_t)(uint8_t)v[4 * j + 3] << 24;
        bst(l, at + j, w);
    }
}

#define KMAX 8
static int8_t ka[KMAX][QR_MAXHID] __attribute__((aligned(4)));
static uint32_t kam[KMAX];
static int32_t kae[KMAX];

#ifdef QR_ACCQ
static inline void sets_k(const uint32_t *m, const int32_t *e, int K) {   /* lane scales 0..K-1 */
    switch (K) {
    case 8: SC_SETS(7, m[7], e[7]); /* fall through */
    case 7: SC_SETS(6, m[6], e[6]); /* fall through */
    case 6: SC_SETS(5, m[5], e[5]); /* fall through */
    case 5: SC_SETS(4, m[4], e[4]); /* fall through */
    case 4: SC_SETS(3, m[3], e[3]); /* fall through */
    case 3: SC_SETS(2, m[2], e[2]); /* fall through */
    case 2: SC_SETS(1, m[1], e[1]); /* fall through */
    default: SC_SETS(0, m[0], e[0]);
    }
}
/* r[l] = apply32(acc[l], smul(lane scale l, (m, e)), 12) for l < K, accumulators cleared (D13 ACCQ) */
static inline void accq_k(int32_t *r, int K, uint32_t m, int32_t e) {
    switch (K) {
    case 8: r[7] = SC_ACCQ(7, m, e); /* fall through */
    case 7: r[6] = SC_ACCQ(6, m, e); /* fall through */
    case 6: r[5] = SC_ACCQ(5, m, e); /* fall through */
    case 5: r[4] = SC_ACCQ(4, m, e); /* fall through */
    case 4: r[3] = SC_ACCQ(3, m, e); /* fall through */
    case 3: r[2] = SC_ACCQ(2, m, e); /* fall through */
    case 2: r[1] = SC_ACCQ(1, m, e); /* fall through */
    default: r[0] = SC_ACCQ(0, m, e);
    }
}
#define ID_M (1u << 29)                  /* (2^29, 29): smul with it is an exact identity */
#define ID_E 29
#endif

/* out[l][r] for l < K: the weight matrix streams once for all K lanes */
static void matvec_k(const qmat_t *W, int K, int32_t (*out)[QR_MAXHID]) {
    for (int l = 0; l < K; l++) bank_fill(l, ka[l], W->cols, 0);
    SC_SETK(K);
    uint32_t words = W->cols / 4;
    SC_WSTART(W->w, W->rows * words);
#ifdef QR_ACCQ
    sets_k(kam, kae, K);
#endif
    for (uint32_t r = 0; r < W->rows; r++) {
        SC_SETPA(0);
        for (uint32_t j = 0; j < words; j += 8) {
            SC_VDOTK(); SC_VDOTK(); SC_VDOTK(); SC_VDOTK();
            SC_VDOTK(); SC_VDOTK(); SC_VDOTK(); SC_VDOTK();
        }
#ifdef QR_ACCQ
        int32_t o[KMAX];
        accq_k(o, K, W->m[r], W->e[r]);
        for (int l = 0; l < K; l++) out[l][r] = o[l];
#else
        int32_t acc[KMAX];
        accrd_k(acc, K);
        for (int l = 0; l < K; l++) {                    /* per-lane readout */
            uint32_t m; int32_t e;
            smul(kam[l], kae[l], W->m[r], W->e[r], &m, &e);
            out[l][r] = apply32(acc[l], m, e, 12);
        }
#endif
    }
}

void qm_steps(qmodel_t *q, const int *toks, int K, int *best, int64_t *lsum) {
    static int32_t x[KMAX][QR_MAXC], big[KMAX][QR_MAXHID], y[KMAX][QR_MAXHID];
    static int32_t dots[KMAX][QR_MAXCTX], s[QR_MAXCTX], wv[QR_MAXCTX];
    static int8_t w8[KMAX][QR_MAXCTX + 4] __attribute__((aligned(4)));
    static int8_t q8[KMAX][QR_D] __attribute__((aligned(4)));
    static uint32_t qm[KMAX], ym[KMAX];
    static int32_t qe[KMAX], ye[KMAX];
    uint32_t C = q->C, H = q->H, n0 = q->n;
    uint32_t t0 = sc_cycles(), cur = PR_OTHER;
    for (int l = 0; l < K; l++)
        for (uint32_t i = 0; i < C; i++)
            x[l][i] = (int32_t)apply(q->emb.w[toks[l] * C + i], q->emb.m[toks[l]], q->emb.e[toks[l]], 12);
    uint32_t Tmax = n0 + K, Tw = (Tmax + 3) / 4;      /* positions any lane can see; in words */
    for (uint32_t li = 0; li < q->L; li++) {
        TIC(PR_NORMQ); for (int l = 0; l < K; l++) rmsquant(x[l], C, ka[l], &kam[l], &kae[l]);
        TIC(PR_MATVEC); matvec_k(&q->qkv[li], K, big);
        TIC(PR_OTHER); for (int l = 0; l < K; l++) { rope(q, big[l], n0 + l); rope(q, big[l] + C, n0 + l); }
        TIC(PR_ATTN);
        for (uint32_t h = 0; h < H; h++) {
            uint32_t base = (li * H + h) * q->ctx;
            for (int l = 0; l < K; l++) {                /* store k, v of every lane first */
                uint32_t pos = n0 + l, km, vm; int32_t ke, ve;
                int8_t v8[QR_D];
                quant(big[l] + C + h * QR_D, QR_D, 12, q->k8 + (base + pos) * QR_D, &km, &ke);
                quant(big[l] + 2 * C + h * QR_D, QR_D, 12, v8, &vm, &ve);
                for (uint32_t d = 0; d < QR_D; d++) q->v8t[(base * QR_D) + d * q->ctx + pos] = v8[d];
                q->km[base + pos] = km; q->ke[base + pos] = (int8_t)ke;
                q->vm[base + pos] = vm; q->ve[base + pos] = (int8_t)ve;
                quant(big[l] + h * QR_D, QR_D, 12, q8[l], &qm[l], &qe[l]);
                bank_fill(l, q8[l], QR_D, 0);            /* q of lane l at bank word 0..7 */
            }
            /* scores: one lw per K word, shared by all lanes */
            SC_SETK(K);
            const uint32_t *kw = (const uint32_t *)(q->k8 + base * QR_D);
#ifdef QR_ACCQ
            {   /* lane scale = q scale x attention constant; ACCQ gives the scaled score s[t] directly */
                uint32_t c0m[KMAX]; int32_t c0e[KMAX];
                for (int l = 0; l < K; l++) smul(qm[l], qe[l], q->attn_m, q->attn_e, &c0m[l], &c0e[l]);
                sets_k(c0m, c0e, K);
            }
#endif
            for (uint32_t t = 0; t < Tmax; t++, kw += QR_D / 4) {
                int32_t r[KMAX];
                SC_SETPA(0);
                SC_VDOTR(kw[0]); SC_VDOTR(kw[1]); SC_VDOTR(kw[2]); SC_VDOTR(kw[3]);
                SC_VDOTR(kw[4]); SC_VDOTR(kw[5]); SC_VDOTR(kw[6]); SC_VDOTR(kw[7]);
#ifdef QR_ACCQ
                accq_k(r, K, q->km[base + t], (int32_t)q->ke[base + t]);
#else
                accrd_k(r, K);
#endif
                for (int l = 0; l < K; l++) dots[l][t] = r[l];
            }
            /* per lane: softmax -> int8 weights (same arithmetic as attend()) */
            for (int l = 0; l < K; l++) {
                uint32_t T = n0 + l + 1, c0m, cm; int32_t c0e, ce, mx = 0;
                smul(qm[l], qe[l], q->attn_m, q->attn_e, &c0m, &c0e);
                for (uint32_t t = 0; t < T; t++) {
#ifdef QR_ACCQ
                    (void)cm; (void)ce; (void)c0m; (void)c0e;
                    s[t] = dots[l][t];
#else
                    smul(c0m, c0e, q->km[base + t], (int32_t)q->ke[base + t], &cm, &ce);
                    s[t] = apply32(dots[l][t], cm, ce, 12);
#endif
                    if (t == 0 || s[t] > mx) mx = s[t];
                }
                uint32_t tot = 0;
                for (uint32_t t = 0; t < T; t++) {
                    uint32_t d = (uint32_t)(mx - s[t]), nn = d >> 12;
                    uint32_t p = nn >= 31 ? 0 : (uint32_t)q->lut[(d & 4095) >> 4] >> nn;
                    tot += p;
                    wv[t] = apply32((int32_t)p, q->vm[base + t], (int32_t)q->ve[base + t], 16);
                }
                uint32_t wm, rm; int32_t we, re;
                quant(wv, T, 16, w8[l], &wm, &we);
                for (uint32_t t = T; t < Tw * 4; t++) w8[l][t] = 0;   /* positions this lane cannot see */
                norm_scale(1, tot, &rm, &re);
                smul(wm, we, rm, re, &ym[l], &ye[l]);
                bank_fill(l, w8[l], Tw * 4, 8);           /* weights of lane l at bank word 8.. */
            }
            /* values: V stored transposed, one lw per word of V^T, shared by all lanes */
            const uint32_t *vt = (const uint32_t *)(q->v8t + base * QR_D);
#ifdef QR_ACCQ
            sets_k(ym, ye, K);
#endif
            for (uint32_t d = 0; d < QR_D; d++, vt += q->ctx / 4) {
                int32_t r[KMAX];
                SC_SETPA(8);
                for (uint32_t j = 0; j < Tw; j++) SC_VDOTR(vt[j]);
#ifdef QR_ACCQ
                accq_k(r, K, ID_M, ID_E);
                for (int l = 0; l < K; l++) y[l][h * QR_D + d] = r[l];
#else
                accrd_k(r, K);
                for (int l = 0; l < K; l++) y[l][h * QR_D + d] = apply32(r[l], ym[l], ye[l], 12);
#endif
            }
        }
        TIC(PR_NORMQ); for (int l = 0; l < K; l++) quant(y[l], C, 12, ka[l], &kam[l], &kae[l]);
        TIC(PR_MATVEC); matvec_k(&q->proj[li], K, big);
        TIC(PR_OTHER); for (int l = 0; l < K; l++) for (uint32_t i = 0; i < C; i++) x[l][i] += big[l][i];
        TIC(PR_NORMQ); for (int l = 0; l < K; l++) rmsquant(x[l], C, ka[l], &kam[l], &kae[l]);
        TIC(PR_MATVEC); matvec_k(&q->fc1[li], K, big);
        TIC(PR_NORMQ);
        for (int l = 0; l < K; l++) {
            for (uint32_t i = 0; i < q->hid; i++) if (big[l][i] < 0) big[l][i] = 0;
            quant(big[l], q->hid, 12, ka[l], &kam[l], &kae[l]);
        }
        TIC(PR_MATVEC); matvec_k(&q->fc2[li], K, y);
        TIC(PR_OTHER); for (int l = 0; l < K; l++) for (uint32_t i = 0; i < C; i++) x[l][i] += y[l][i];
    }
    TIC(PR_NORMQ); for (int l = 0; l < K; l++) rmsquant(x[l], C, ka[l], &kam[l], &kae[l]);
    TIC(PR_MATVEC);
    /* output layer: stream once, argmax + sum per lane */
    int32_t bv[KMAX]; int64_t sum[KMAX];
    for (int l = 0; l < K; l++) { bank_fill(l, ka[l], C, 0); bv[l] = 0; sum[l] = 0; best[l] = 0; }
#ifdef QR_MARGIN
    int32_t b2[KMAX];
    for (int l = 0; l < K; l++) b2[l] = INT32_MIN;
#endif
    SC_SETK(K);
    SC_WSTART(q->out.w, q->vocab * (C / 4));
#ifdef QR_ACCQ
    sets_k(kam, kae, K);
#endif
    for (uint32_t r = 0; r < q->vocab; r++) {
        SC_SETPA(0);
        for (uint32_t j = 0; j < C / 4; j += 8) {
            SC_VDOTK(); SC_VDOTK(); SC_VDOTK(); SC_VDOTK();
            SC_VDOTK(); SC_VDOTK(); SC_VDOTK(); SC_VDOTK();
        }
        int32_t acc[KMAX];
#ifdef QR_ACCQ
        accq_k(acc, K, q->out.m[r], q->out.e[r]);
#else
        accrd_k(acc, K);
#endif
        for (int l = 0; l < K; l++) {
#ifdef QR_ACCQ
            int32_t lg = acc[l];
#else
            uint32_t mm; int32_t ee;
            smul(kam[l], kae[l], q->out.m[r], q->out.e[r], &mm, &ee);
            int32_t lg = apply32(acc[l], mm, ee, 12);
#endif
            sum[l] += lg;
#ifdef QR_MARGIN
            if (r == 0) bv[l] = lg;
            else if (lg > bv[l]) { b2[l] = bv[l]; bv[l] = lg; best[l] = (int)r; }
            else if (lg > b2[l]) b2[l] = lg;
#else
            if (r == 0 || lg > bv[l]) { bv[l] = lg; best[l] = (int)r; }
#endif
        }
    }
    TIC(PR_OTHER);
#ifdef QR_MARGIN
    for (int l = 0; l < K; l++) q->margin[l] = q->vocab > 1 ? bv[l] - b2[l] : 0;
#endif
    if (lsum) for (int l = 0; l < K; l++) lsum[l] = sum[l];
    q->n = n0 + K;
}
#endif
