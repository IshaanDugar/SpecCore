/* SpecCore custom instructions (custom-0 opcode 0x0B, R-type), ext/rtl/sc_vdotk.sv.
   On the core each macro is one instruction. On the PC the same macros run a C model of the
   unit (below), so `python tools/sc.py run` can diff core output against PC output.

     SC_SETK(k)          K = k active lanes (1..8)
     SC_SETP(l, p)       lane l pointer = word p of its bank          (l: constant 0..7)
     SC_SETPA(p)         every lane's pointer = p
     SC_VDOTR(w)         acc[k] += dot4(w, bank[k][ptr[k]]), ptr[k]++, for k < K   (word from a register)
     SC_VDOTK()          same, word popped from the weight stream (stalls while it is empty)
     SC_ACCRD(l)         returns acc[l] and clears it                 (l: constant)
     SC_SETS(l, m, e)    lane scale of lane l = (m, e)                 (D13; l: constant)
     SC_ACCQ(l, m, e)    returns apply(acc[l], smul(lane scale, (m, e)), 12) and clears acc[l]
                         = ref/qref.c smul() + apply32() in one instruction (multi-cycle, D13)
     SC_WPOP()           returns the next stream word
     SC_BSTORE(l, a, w)  bank[l][a] = w                               (l: constant)
     SC_WSTART(p, n)     stream n words starting at pointer p (inside the model image)
   Lane numbers in SETP/ACCRD/BSTORE are encoded in funct7, so they must be compile-time constants. */
#ifndef SC_ISA_H
#define SC_ISA_H
#include <stdint.h>

#ifdef SC_ON_CORE
#define SC_SETK(k)        __asm__ volatile(".insn r 0x0B, 0, 0, x0, %0, x0" :: "r"(k))
#define SC_SETP(l, p)     __asm__ volatile(".insn r 0x0B, 1, %1, x0, %0, x0" :: "r"(p), "i"(l))
#define SC_SETPA(p)       __asm__ volatile(".insn r 0x0B, 1, 64, x0, %0, x0" :: "r"(p))
#define SC_VDOTR(w)       __asm__ volatile(".insn r 0x0B, 2, 0, x0, %0, x0" :: "r"(w))
#define SC_VDOTK()        __asm__ volatile(".insn r 0x0B, 4, 0, x0, x0, x0")
#define SC_BSTORE(l, a, w) __asm__ volatile(".insn r 0x0B, 6, %2, x0, %0, %1" :: "r"(a), "r"(w), "i"(l))
#define SC_WSTART(p, n)   __asm__ volatile(".insn r 0x0B, 7, 0, x0, %0, %1" :: "r"(p), "r"(n))
#define SC_ACCRD(l) ({ int32_t _r; __asm__ volatile(".insn r 0x0B, 3, %1, %0, x0, x0" : "=r"(_r) : "i"(l)); _r; })
#define SC_WPOP()   ({ uint32_t _r; __asm__ volatile(".insn r 0x0B, 5, 0, %0, x0, x0" : "=r"(_r)); _r; })
#define SC_SETS(l, m, e)  __asm__ volatile(".insn r 0x0B, 1, %2, x0, %0, %1" :: "r"(m), "r"(e), "i"(32 + (l)))
#define SC_ACCQ(l, m, e) ({ int32_t _r; __asm__ volatile(".insn r 0x0B, 3, %3, %0, %1, %2" : "=r"(_r) : "r"(m), "r"(e), "i"(64 + (l))); _r; })
#else
/* ---- C model of the unit (same semantics as the RTL) ---- */
#define SC_LANES_ 8
#define SC_BANK_WORDS_ 256
typedef struct {
    uint32_t bank[SC_LANES_][SC_BANK_WORDS_], ptr[SC_LANES_], k;
    int32_t acc[SC_LANES_];
    uint32_t sm[SC_LANES_]; int32_t se[SC_LANES_];   /* lane scales (SETS) */
    const uint32_t *ws; uint32_t ws_left;
} sc_unit_t;
static sc_unit_t sc_u = {.k = 1};
static inline int32_t sc_dot4_(uint32_t a, uint32_t b) {
    int32_t s = 0;
    for (int i = 0; i < 4; i++) s += (int32_t)(int8_t)(a >> (8 * i)) * (int32_t)(int8_t)(b >> (8 * i));
    return s;
}
static inline void sc_mac_(uint32_t w) {
    for (uint32_t k = 0; k < sc_u.k; k++) {
        sc_u.acc[k] += sc_dot4_(w, sc_u.bank[k][sc_u.ptr[k]]);
        sc_u.ptr[k] = (sc_u.ptr[k] + 1) % SC_BANK_WORDS_;
    }
}
static inline uint32_t sc_pop_(void) {
    if (!sc_u.ws_left) { puts("sc_isa model: pop from an empty weight stream"); exit(9); }
    sc_u.ws_left--; return *sc_u.ws++;
}
static inline int32_t sc_accrd_l_(int l) { int32_t r = sc_u.acc[l]; sc_u.acc[l] = 0; return r; }
/* ACCQ, exactly as the RTL: p = sm*rm; normalise to m in [2^29, 2^30); sh = e - 12; round half up, >> sh */
static inline int32_t sc_accq_(int l, uint32_t rm, int32_t re) {
    int32_t v = sc_u.acc[l]; sc_u.acc[l] = 0;
    uint64_t p = (uint64_t)sc_u.sm[l] * rm;
    uint32_t m; int32_t e;
    if (p >= (1ull << 59)) { m = (uint32_t)(p >> 30); e = (int32_t)((uint32_t)sc_u.se[l] + (uint32_t)re - 30u); }
    else { m = (uint32_t)(p >> 29); e = (int32_t)((uint32_t)sc_u.se[l] + (uint32_t)re - 29u); }
    int32_t sh = (int32_t)((uint32_t)e - 12u);
    if (!m || sh > 62 || sh < 1) return 0;
    return (int32_t)(((int64_t)v * (int64_t)m + (1ll << (sh - 1))) >> sh);
}
#define SC_SETK(kk)        (sc_u.k = (kk) == 0 ? 1 : (kk) > SC_LANES_ ? SC_LANES_ : (kk))
#define SC_SETP(l, p)      (sc_u.ptr[l] = (uint32_t)(p) % SC_BANK_WORDS_)
#define SC_SETPA(p)        do { for (int _l = 0; _l < SC_LANES_; _l++) sc_u.ptr[_l] = (uint32_t)(p) % SC_BANK_WORDS_; } while (0)
#define SC_VDOTR(w)        sc_mac_((uint32_t)(w))
#define SC_VDOTK()         sc_mac_(sc_pop_())
#define SC_ACCRD(l)        sc_accrd_l_(l)
#define SC_SETS(l, m, e)   (sc_u.sm[l] = (uint32_t)(m), sc_u.se[l] = (int32_t)(e))
#define SC_ACCQ(l, m, e)   sc_accq_(l, (uint32_t)(m), (int32_t)(e))
#define SC_WPOP()          sc_pop_()
#define SC_BSTORE(l, a, w) (sc_u.bank[l][(uint32_t)(a) % SC_BANK_WORDS_] = (uint32_t)(w))
#define SC_WSTART(p, n)    (sc_u.ws = (const uint32_t *)(uintptr_t)(p), sc_u.ws_left = (n))
#endif
#endif
