/* Boot code: set gp and sp, zero .bss, call main, then report main's return value.
   Exit protocol (read by core/tb/tb_sc.v): write the code to SEVENSEG (0xFFFF0080),
   then send byte 0x04 (end of transmission) on the UART. */
    .section .text.start
    .globl _start
_start:
    .option push
    .option norelax
    la    gp, __global_pointer$
    .option pop
    la    sp, __stack_top
    la    t0, __bss_start
    la    t1, __bss_end
1:  bgeu  t0, t1, 2f
    sw    zero, 0(t0)
    addi  t0, t0, 4
    j     1b
2:  call  main
    .globl sc_exit
sc_exit:                      /* a0 = exit code; also callable from C */
    li    t0, 0xFFFF0000
    sw    a0, 0x80(t0)        /* SEVENSEG = exit code */
3:  lw    t1, 0x08(t0)        /* wait for UART TX ready */
    beqz  t1, 3b
    li    t1, 4
    sw    t1, 0x0C(t0)        /* UART TX = 0x04 */
4:  j     4b
