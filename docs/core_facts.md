# Core facts (read from RTL, 4 Nov 2026)

Source: `CG3207/lab4 finalised/` (identical to `integrated/`, checked file by file with `cmp`). Copied to `core/rtl`, `core/tb`, `core/sw`. `core/rtl/nexys/Wrapper.v` is the Nexys A7 top; it gets replaced on the KV260.

## Answers to the plan's open questions

| Question | Answer | Source |
| --- | --- | --- |
| MCycle cycles per multiply | Radix-4 Booth (plan said radix-2). Signed mul: 16 iterations, unsigned: 17, divide: 32 (restoring). About 17 cycles in E for `mul`. | MCycle.v `LAST_S/LAST_U/LAST_D` |
| How MCycle stalls | `Busy` holds F, D and E; a bubble goes into M (`FlushM = BusyE`). | Hazard.v |
| Data memory | 8 KB (2048 words), **asynchronous read** → distributed LUTRAM. | Wrapper.v `DMEM_DEPTH_BITS = 13`, async block |
| Instruction memory | 4 KB (1024 words), asynchronous read. | Wrapper.v `IROM_DEPTH_BITS = 12` |
| How firmware loads | `$readmemh("AA_IROM.mem")` at synthesis, from RARS dumps. Needs a rebuild to change the program. | Wrapper.v |
| Cycle counter | Yes, MMIO `0xFFFF00A0`, 32 bits, read-only, wraps after about 42 s at 100 MHz. | Wrapper.v |
| Clock reached | About 65 MHz on the Nexys A7 (student). The archived timing report only constrains the 100 MHz input clock, not the divided core clock. | student; `TOP_timing_summary_routed.rpt` |
| Simulator | Vivado xsim. Assembler: RARS. | student |
| Memory map | IROM `0x0040_0000`, DMEM `0x1001_0000`, MMIO `0xFFFF_0000`. | Wrapper.v |

## Things the plan did not know

1. **Not full RV32IM.** Supported: R/I ALU ops, `lw`, `sw`, all 6 branches, `lui`, `auipc`, `jal`, `jalr`, `mul`, `div`, `divu`.
   Missing: `lb lh lbu lhu sb sh mulh mulhsu mulhu rem remu`.
   Danger: the decoder ignores funct3 on loads and stores, so `lb` silently acts as `lw` and `sb` as `sw`. Missing M ops silently do nothing.
   GCC with `-march=rv32im` will emit these, and int8 code needs `lb`/`sb`. **Must fix before the 8 Nov C port.**
2. **Branches never squash the instruction in E.** On a mispredict, `FlushD` and `FlushE` remove the two younger instructions. The instruction in E always continues to M. The plan's claim holds.
3. **VDOTK can never sit in E while MCycle is busy**, because the busy multiply is the instruction in E. The plan's directed test "VDOTK held in E while an older MCycle multiply stalls" cannot happen. The only way VDOTK is held in E is its own FIFO-empty stall. Copy the `BusyE` path exactly: hold F/D/E, bubble into M, and update hidden state only on the cycle E advances.
4. **Async LUTRAM cannot grow to the plan's sizes.** BRAM and UltraRAM have synchronous reads. See the plan review, section 6.
5. **License.** RV.v, Wrapper.v, MCycle.v, Decoder.v and others carry: "DO NOT post it on any public repository". The plan says "Repo public" on 24 Nov. See the plan review, section 9.
6. **Team work.** `integrated/demo/PLAN.md` describes "the two team members". In applications, the core needs wording that is fair to the teammate.

## BTB

16 entries, tag + target + 2-bit counter. Enabled by DIP[15] on the Nexys; on the KV260 it becomes a control register bit.

## Update 12 Nov 2026 (session 4)
The core is now full RV32IM: lb/lh/lbu/lhu/sb/sh, mulh/mulhsu/mulhu, rem/remu added. `mul*` moved from MCycle to a 1-cycle DSP multiplier in E (D8), so the "~17 cycles per mul" above applies only to the old core; MCycle now only divides (32 cycles, measured). See LOGBOOK session 4.
