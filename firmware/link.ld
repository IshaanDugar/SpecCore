/* SpecCore firmware memory map (sim / Nexys-style Wrapper.v). KV260 sizes change in D7 later.
   IROM 0x0040_0000 (64 KB code) and DMEM 0x1000_0000 (1 MB; changed 14 Nov for model runs; DDR model image at 0x2000_0000) (rodata, data, bss, stack) as in core/rtl/nexys/Wrapper.v.
   Both are loaded by $readmemh from AA_IROM.mem / AA_DMEM.mem, so nothing is copied at boot. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY {
  IROM (rx) : ORIGIN = 0x00400000, LENGTH = 64K
  DMEM (rw) : ORIGIN = 0x10000000, LENGTH = 1M
}

SECTIONS {
  .text : {
    KEEP(*(.text.start))          /* crt0 first: the core starts at IROM base */
    *(.text .text.*)
  } > IROM

  .data : {
    *(.rodata .rodata.* .srodata .srodata.*)
    *(.data .data.*)
    . = ALIGN(4);
    __global_pointer$ = . + 0x800;
    *(.sdata .sdata.*)
    . = ALIGN(4);
  } > DMEM

  .bss (NOLOAD) : {
    __bss_start = .;
    *(.sbss .sbss.* .bss .bss.* COMMON)
    . = ALIGN(4);
    __bss_end = .;
  } > DMEM

  __stack_top = ORIGIN(DMEM) + LENGTH(DMEM);
  __heap_start = __bss_end;

  /DISCARD/ : { *(.comment) *(.riscv.attributes) *(.eh_frame*) *(.note*) }
}
