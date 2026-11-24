/* Tiny I/O layer. The same C source builds for the core (UART + cycle counter MMIO)
   and for the PC (stdio + clock), so outputs can be diffed line by line.
   On the core: SC_ON_CORE is defined by `python tools/sc.py fw`. */
#ifndef SC_IO_H
#define SC_IO_H
#include <stdint.h>

#ifdef SC_ON_CORE
#define SC_MMIO(off) (*(volatile uint32_t *)(0xFFFF0000u + (off)))
static inline void sc_putc(char c) {
    while (!SC_MMIO(0x08)) {}          /* UART TX ready */
    SC_MMIO(0x0C) = (uint8_t)c;
}
static inline uint32_t sc_cycles(void) { return SC_MMIO(0xA0); }
void sc_exit(int code) __attribute__((noreturn));
#else
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static inline void sc_putc(char c) { putchar(c); }
static inline uint32_t sc_cycles(void) { return (uint32_t)clock(); }
static inline void sc_exit(int code) { fflush(stdout); exit(code); }
#endif

static inline void sc_puts(const char *s) { while (*s) sc_putc(*s++); }

static inline void sc_putu(uint32_t v) {       /* decimal (uses divu/remu on the core) */
    char buf[11]; int n = 0;
    do { buf[n++] = (char)('0' + v % 10u); v /= 10u; } while (v);
    while (n) sc_putc(buf[--n]);
}
static inline void sc_puti(int32_t v) {
    if (v < 0) { sc_putc('-'); sc_putu((uint32_t)0 - (uint32_t)v); } else sc_putu((uint32_t)v);
}
static inline void sc_puthex(uint32_t v) {
    for (int i = 28; i >= 0; i -= 4) sc_putc("0123456789abcdef"[(v >> i) & 15u]);
}
#endif
