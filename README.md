# SpecCore

SpecCore runs speculative decoding of a small code language model on a RISC-V core I extended with my own LLM unit, on an AMD Kria KV260 FPGA. The unit checks several draft tokens in about the time it takes to read the model weights once, and the output matches plain greedy decoding bit for bit.

The demo page is [`docs/demo.html`](docs/demo.html). Open it in a browser to see the board results and a token-by-token replay of the draft and verify passes.

## Results on the KV260

All numbers in this table were measured on the board.

| What | Result |
| --- | --- |
| 50-prompt test set, fixed γ = 2, 16 prompt + 32 new tokens each | 1.163× speedup (sum of plain cycles over sum of speculative cycles). Per prompt it ranges from 0.86× to 1.40× with a median of 1.18×; 5 of the 50 prompts are slower than plain decoding. |
| Output | 50 of 50 prompts bit-exact against the reference, for both plain and speculative decoding |
| Board vs RTL simulation | Identical cycle counts on every program checked, including a full model run of 100,556,084 cycles |
| Cost of real DDR4 | 1% more wall time than core cycles. The weight stream is hidden by prefetch, and a 32 KB cache covers the other model reads. Counting the prompt in both runs, the real-time speedup is 1.10×. |
| Board design | 50 MHz with timing met (WNS +1.79 ns); 18,307 LUTs (15.6%), 32 URAM, 16.5 BRAM, 12 DSP |
| DMA bandwidth, DDR to PL and back, 128-bit HP0 at 100 MHz | 1,597 MB/s each way (99.8% of what the port allows), no errors |

These come from RTL simulation and implementation:

| What | Result |
| --- | --- |
| 3 prompts, 40 tokens, fixed γ = 2 | 1.215× (1.11× to 1.37× per prompt). The board reproduces these runs exactly. |
| Confidence gate, γ ≤ 4 | 1.334× on one prompt against 1.180× at fixed γ = 2. Across 50 prompts it gains about 2.5% over fixed γ = 2 (estimate). |
| `ACCQ` instruction | Each extra verify lane drops from 1.71M to 1.08M cycles; plain decoding gets 1.22× faster |
| Core and LLM unit alone, out-of-context on xck26-2LV | 122.7 MHz, 7,223 LUTs (6.2%), 12 DSPs |

## How it works

1. Two code models, Llama style, trained from scratch on Python with my own BPE tokenizer (2,048 tokens). The target has 6 layers of width 256 and 5.2M parameters. The draft has 2 layers of width 128 and is distilled from the target.
2. In speculative decoding the draft guesses γ tokens. The target checks all K = γ + 1 positions in one pass and keeps the longest prefix it agrees with.
3. The CPU is a 5-stage pipelined RV32IM core with forwarding, a hazard unit and a branch predictor.
4. The LLM unit sits on the custom-0 opcode. `VDOTK` and `VDOTR` multiply one weight word (4 × int8) against K activation banks in the same cycle. `ACCQ` is a 5-stage scaled readout that matches the integer reference exactly. The other instructions are `SETK`, `SETP`, `SETS`, `ACCRD`, `WPOP`, `BSTORE` and `WSTART`.
5. Inference uses integers only: int8 weights and activations, scales stored as a mantissa and an exponent, and integer softmax, RMSNorm and RoPE. The PyTorch integer model (`model/qmodel.py`), the C reference (`ref/qref.c`) and the core give bit-identical results.
6. On the board (`board/core/`) the core runs unchanged in the FPGA fabric. Program and data memories are on chip and clocked on the opposite edge, so the core sees single-cycle memories just as it does in simulation. The 6.9 MB model image lives in DDR4 and reaches the core through a prefetch window for the weight stream and a 32 KB cache for other reads. If a word the core needs has not arrived, its clock pauses, so the core's cycle counts match simulation exactly. The Arm cores load the program and model, start the run and read back the results.

## Verification

- LLM unit: a UVM testbench with a scoreboard and functional coverage (`ACCQ` bins at 100% on 3 seeds), SVA properties, mutation testing (one seeded bug produced 42 scoreboard errors and 7 SVA failures), and a 22.1M-case arithmetic check.
- Core: a self-checking RV32IM ISA test with 586 cases, run with the branch predictor on and off, and a constrained-random instruction test against a Python ISS. All 10 seeds pass, and it caught each of 3 forwarding and stall bugs seeded into the RTL. SVA checks live in `core/tb/tb_sc.sv`.
- Core and unit together: directed hazard tests with 154 checks.
- Full system: complete model runs compared line by line with the C reference, both in simulation and on the board.
- Board design: `board/core/tb_board.sv` simulates it against a DDR model with random latency before each bitstream build.
- Sign-off: gate-level simulation with the same cycle counts as the RTL, plus timing, utilisation, methodology, CDC and DRC reports.

## Repository layout

| Path | Contents |
| --- | --- |
| `model/` | training, tokenizer, float and integer speculative decoding (`spec.py`, `qmodel.py`), export |
| `ref/` | bit-exact C reference (`qref.c`), end-to-end driver (`run_ref.c`), golden vectors |
| `core/rtl`, `core/tb` | RV32IM core and its testbenches. The base core comes from NUS CG3207 course material and is shared with permission. |
| `ext/rtl`, `ext/tb` | LLM unit (`sc_vdotk.sv`) and its UVM testbench |
| `firmware/` | runtime, ISA macros (`include/sc_isa.h`), tests |
| `board/` | KV260 designs: `core/` runs SpecCore on the board, `dma/` measures bandwidth, `hello/` was the first bring-up |
| `tools/` | build and simulation (`sc.py`), board build and runs (`board_build.py`, `board_sim.py`, `board_set.py`), cycle model, demo generator |
| `checkpoints/export/` | exported int8 models and the model image the core reads |
| `logs/board/` | raw and summarised board runs |
| `docs/` | project summary, technical guide, lab logbook, decision log, model card, demo page |

## Reproduce

Tested on Windows with Vivado 2025.2, a RISC-V GCC toolchain and Python with PyTorch.

```sh
python tools/sc.py sim isa                                   # ISA regression on the core (xsim)
python tools/sc.py run accq firmware/tests/accq_test.c       # ACCQ hazard tests
python tools/sc.py run ref ref/qref.c ref/run_ref.c -DQR_UNIT -DQR_ACCQ --ddr --bp --max 150000000
                                                             # end-to-end speculative run in xsim (~1 h)
python tools/board_sim.py vdotk                              # board top in simulation
python tools/board_build.py build core                       # bitstream (~1 h)
python tools/board_set.py build && python tools/board_set.py run   # 50-prompt test set on the board
python tools/live_demo.py                                    # local interactive demo, http://127.0.0.1:8765
```

Every Vivado job runs under a hard memory cap (`tools/memguard.py`).

## License

Copyright © 2026 Ishaan. All rights reserved. The code is public so it can be read and reviewed.
