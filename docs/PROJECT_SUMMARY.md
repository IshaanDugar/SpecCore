# SpecCore: project summary

Last updated: 23 Nov 2026. This is the high-level picture: what the project is, what it achieved, how it was built and with which tools. Numbers are labelled **measured** (run and recorded in `docs/LOGBOOK.md`) or **estimated** (cycle model or arithmetic). Deeper material: `docs/TECHNICAL_GUIDE.md` (how everything works), `docs/LOGBOOK.md` (every step and number), `docs/review/decisions.md` (every design decision with options and reasons). The build history is kept in the appendix at the end.

## 1. One-line pitch

My own RISC-V CPU runs a small code-writing language model faster by speculating on words the way CPUs speculate on branches. A tiny draft model guesses the next few tokens, the big model checks all of them in one pass, and a custom instruction set makes that check cost about one read of the weights. The output is identical to normal decoding.

## 2. The problem and the idea

- **Generating text is memory-bound.** Each new token needs every weight read once, and each weight is used for only one multiply-add. On small hardware the multipliers wait on memory.
- **Speculative decoding** (Leviathan et al. 2023, Chen et al. 2023) turns one read of the weights into several tokens:
  - a cheap draft model guesses γ tokens;
  - the target model checks all γ+1 positions in one pass;
  - the longest correct prefix is kept.
  The result is token-for-token the same as plain greedy decoding, so the speedup costs no quality.
- **What SpecCore adds:** the algorithm runs on a CPU I extended myself. A custom "LLM unit" in its pipeline multiplies one weight word from memory against up to 8 tokens at once. The project measures, with real cycle counts, when that is actually faster and why.

## 3. Results

| What | Result | Label |
| --- | --- | --- |
| **Speedup on the KV260 board, 50-prompt test set, fixed γ2, 16 + 32 tokens** | **1.163×** (Σplain / Σspec); 50/50 bit-exact; per prompt 0.86×-1.40× (median 1.18×), 5 slower than plain | measured on hardware (core cycles; 50 MHz; model in real DDR, core paused while a word is missing; real DDR adds ~1% wall time with a 32 KB load cache, so real-time speedup = cycle speedup within 0.4%; incl. the 16-token prompt in both runs 1.10× wall) |
| Board vs RTL simulation | identical cycle counts (unit tests; full golden model run 100,556,084 cycles both) | measured |
| End-to-end speedup on the core, fixed γ2, 3 prompts, 40 tokens | **1.215×** (Σplain / Σspec); per prompt 1.11×, 1.18× and 1.37×; never slower than plain | measured, cycle-accurate RTL simulation (simulated DDR answers every request in time) |
| Same target shape trained 5× longer (val loss 1.955 → 1.821, better code), draft re-distilled | γ2 1.137× vs 1.175× for the active pair (−3%) | estimated (same accounting); cause not yet isolated (both models changed). Active results keep the original pair |
| Speedup over 50 prompts, γ2 (pre-board estimate) | 1.15× to 1.18× | estimated; superseded by the board measurement 1.163× |
| Confidence gate, γ ≤ 4 | 1.334× on 1 prompt (fixed γ2: 1.180×); about +2.5% over fixed γ2 across 50 prompts | measured / estimated |
| Output correctness | identical to plain greedy on every run: core vs C reference, 50/50 prompts in the integer model, 200/200 in float | measured |
| `ACCQ` instruction | per-token verify cost −37% (1.71M → 1.08M cycles); plain decoding 1.22× faster | measured |
| Cycles per token vs the board's ARM core | 3.22M on SpecCore vs 6.34M on a Cortex-A53 with NEON (2.0× fewer cycles; slower in wall time because of the clock) | measured |
| Clock and area on the KV260's chip | 122.7 MHz, 7,223 LUTs (6.2%), 12 DSPs, no block RAM; 0 critical warnings | measured, out-of-context place and route |
| DMA, DDR → FPGA → DDR, on the KV260 | 1,597 MB/s per direction (99.8% of the port's limit), 0 errors: about 3× what the weight stream needs | measured on the board |

**The main finding is an Amdahl's-law story.**
1. The first end-to-end run was *slower* with speculation: 0.83×. Only the weight stream is shared between the tokens in a verify pass. The per-token work (readout and rescaling, attention, norms) was paid once per token.
2. I split every pass into shared and per-token cost. The fit is linear within 1.3%.
3. I built a cycle model calibrated from those measurements. It predicts end-to-end runs within about 1%.
4. That model pointed at two levers, and both were needed:
   - moving the per-token readout into hardware (the `ACCQ` instruction);
   - a better draft model (longer training plus distillation).

   Together they took speculation from 0.83× to 1.2×.

## 4. System architecture

| Layer | What it is |
| --- | --- |
| Models | Two Llama-style decoder-only transformers (RMSNorm, RoPE, ReLU MLP, tied embeddings), trained from scratch on Python code. Own byte-level BPE tokenizer, vocab 2048, context 128. Draft: dim 128, 2 layers, 0.66M parameters, distilled from the target. Target: dim 256, 6 layers, 5.2M parameters. |
| Integer inference | int8 weights and activations. Every scale is a 30-bit mantissa plus an exponent. Integer softmax, RMSNorm and RoPE; no floating point anywhere. Python integer model = C reference = RTL, bit for bit. |
| CPU core | 5-stage pipelined RV32IM: forwarding, hazard unit, 16-entry branch target buffer with 2-bit counters, single-cycle DSP multiplier, 32-cycle divider. Started as a partial RV32I course core; extended to full RV32IM in this project. |
| LLM unit (in the E stage) | 8 lanes × 4 int8 MACs per cycle, 8 activation banks, weight FIFO. Custom-0 instructions: `SETK` (number of lanes), `SETP` (stream pointer), `SETS` (lane scale), `VDOTR`/`VDOTK` (one weight word × K lanes), `ACCRD` (read accumulator), `ACCQ` (5-stage rescaled readout), `WPOP`, `BSTORE`, `WSTART`. |
| Firmware | Bare-metal C. The same source runs on the PC and on the core, and the outputs are diffed automatically. |
| Platform | Main results: cycle-accurate simulation of the RTL in Vivado xsim. AMD Kria KV260 board: the integer model on its ARM cores, my own bitstreams (hello, DMA loopback), timing closure on its chip. |

## 5. How it was built (workflow)

1. **Model side first.** Train both models on a laptop GPU. Prove speculative decoding gives the same output as greedy. Measure how often the draft agrees with the target.
2. **Integer model, then C.**
   - Convert the models to int8 with an integer-only Python model.
   - Write a C reference that matches it bit for bit.
   - Golden files (tokens and logit sums) pin every later stage to the same numbers.
3. **Core and unit.**
   - Extend the core to RV32IM.
   - Design the LLM unit and its instructions.
   - Run the same C on the core in simulation; every run is diffed against the PC.
4. **Measure, model, decide.**
   - Cycle counters in firmware split each pass into buckets (matvec, attention, norm, other).
   - A K-sweep fits shared + per-token cost.
   - The calibrated cycle model turns measured acceptance into a speedup estimate.
   - Each hardware or model change was picked by that model, then confirmed on the core.
5. **Verify bottom-up:** block → subsystem → SoC → board (section 7).
6. **Sign-off on the FPGA flow:**
   - out-of-context place and route on the KV260 part;
   - methodology, CDC and DRC reports;
   - gate-level simulation of the synthesised netlist, which gives the same cycle counts as the RTL.
   - DFT (scan, ATPG) is for manufacturing test of custom silicon, so it does not apply to an FPGA.
7. **Record everything.**
   - LOGBOOK entry after every step.
   - HANDOFF note for the next session.
   - Decision log, with options, effects and the choice.
   - Numbers always labelled measured or estimated.

**Engineering rules that came from real problems:**
- Every Vivado job runs under a hard memory cap (a Windows Job Object through `tools/sc.py`), one job at a time. An unbounded coverage construct once made the simulator allocate about 100 GB.
- Long simulations run in a separate git worktree, so code edits can't race a running simulation.

## 6. Tools and stack

| Area | Tools |
| --- | --- |
| ML | Python 3.13, PyTorch (CUDA, RTX 5070 laptop GPU), HuggingFace `tokenizers`, NumPy |
| RTL and simulation | Verilog / SystemVerilog, Vivado 2025.2 (xsim simulation, synthesis, place and route), UVM 1.2 in xsim |
| Firmware | RISC-V GCC (`riscv64-unknown-elf`, rv32im) in WSL2 Ubuntu, own linker script and crt0; host gcc for the PC build |
| Own tools | `tools/sc.py` (build, simulate, diff, UVM, gate-level, all memory-guarded), `tools/cyclemodel.py` (calibrated cycle model), `tools/bakeoff.py` (branch-predictor-style draft-length predictors), `tools/gen_rig.py` + `tools/iss.py` (random instruction test + instruction-set simulator), `tools/synth_core.py`, `tools/chart_results.py` (results page), `tools/demo.py` (replay demo) |
| Waveform analytics | WaveQ, my own VCD → SQLite query server (MCP), used for cycle accounting and protocol checks such as "every ACCQ takes exactly 5 cycles" |
| Board | AMD Kria KV260 (Zynq UltraScale+), Ubuntu, `xmutil` app loading, device-tree overlays, UIO, AXI DMA, CMA dma-heap |
| AI assistance | Claude Code (architecture, code, runs), Codex CLI models as an independent DV engineer (UVM coverage, random-instruction test, gate-level flow) and reviewer; Undermind for literature search. Every result was run and checked in the project's own flow. |

## 7. Verification

| Level | What | Result |
| --- | --- | --- |
| Block: LLM unit | UVM (driver that behaves like the E stage and corrupts forwarded operands during stalls, monitor, scoreboard, functional coverage); 9 SVA rules; mutation testing; 22.1M-case arithmetic check | coverage of the `ACCQ` bins 100% on 3 seeds; a seeded bug was caught by both the scoreboard (42 errors) and SVA (7 failures) (measured) |
| Block: core | self-checking ISA test generated from the spec (586 cases); constrained-random instruction test vs a Python ISS; SVA | ISA PASS with the branch predictor on and off; 10 random seeds × 2,000 instructions PASS; 3 seeded forwarding and stall bugs all caught (measured) |
| Subsystem: core + unit | directed hazard tests (load-use, mul forwarding, back-to-back, lane corner cases); WaveQ on the waveform | 154 checks PASS; 154/154 `ACCQ`s take exactly 5 cycles; the operand latch is needed on the real core (measured) |
| SoC (simulation) | full model runs on the core vs the C reference, 4 prompt and mode configurations | all identical, 0 SVA failures (measured) |
| Gate level | synthesised netlist simulated with the same tests | PASS, same cycle counts as the RTL (measured) |
| Board | hello bitstream, DMA loopback | 0 errors (measured) |

## 8. Limitations (stated plainly)

- The full model runs on the core in **simulation**, not on the board. The board results are blocks (DMA, a test app) and the ARM-core comparison.
- The clock is from an out-of-context implementation. A full board design adds I/O and the memory system.
- Speedups over many prompts are **estimates** (bit-exact integer model × measured pass costs). Simulation runs at about 26K cycles/s, so each core prompt takes about an hour.
- The models are small (0.66M / 5.2M parameters) and trained on code only. The speedup depends on how often the draft agrees with the target.
- The critical path (forwarding → VDOTR dot product) could be pipelined to win back clock speed. Not done.

### 8.1 Next steps (future work; item 1 is done, the rest is not built or measured)

- **Why custom models now:** a 2,048-token vocabulary keeps the readout (the largest per-token cost) small; the sizes fit a ~26K cycles/s simulator; the integer number system is ours end to end (bit-exact Python → C → RTL); draft and target share one tokenizer by construction. Memory was never the limit: 4 GB DDR4 holds a model 100× bigger.
- **1. Full core on the KV260: DONE 24 Nov (measured, see §3).** Next on the board: higher clock (100 MHz), live token streaming, bigger models now that a run takes seconds instead of hours.
- **2. Real open model pairs** that share a tokenizer (e.g. a ~135M draft with a ~360M-1.7B target, or ~0.5B / ~1.5B coder models; check each licence), int8/int4. Costs: vocabularies of ~49K-151K make the readout much heavier; the ISA and number system need porting.
- **3. Bigger targets should favour speculation:** the shared weight stream grows with width², per-token work roughly with width, so each extra verify lane gets relatively cheaper (expectation, not measured).
- **4. A dedicated accelerator beside the core:** the RISC-V becomes the controller, a wide matrix engine on DSPs does the math (the KV260 has 1,248 DSPs; the core uses 12), with a weight port at full DMA width (measured 1.6 GB/s per direction at 128 bit × 100 MHz; VDOTK uses 4 B/cycle today). Same direction as commercial LLM inference chips; a research-scale version is realistic on this board.

## 9. Skills shown

Computer architecture (pipeline extension, custom ISA design, memory-bound analysis, roofline, Amdahl's law, a calibrated cycle model), RTL design (Verilog/SystemVerilog, multi-cycle units, stall and forwarding hazards, DSP inference), design verification (UVM with scoreboard and coverage, SVA, mutation testing, constrained-random testing against an ISS, gate-level simulation, waveform analytics with my own tool), ML systems (tokenizer, transformer training, distillation, speculative decoding, integer-only inference bit-exact across Python, C and RTL), embedded C, FPGA (Vivado, timing closure, AXI DMA on the KV260, Linux UIO), Python tooling.

## 10. Draft resume bullets (only measured claims)

- Designed a custom LLM unit for a 5-stage RISC-V pipeline (10 custom operations; one streamed weight word feeds up to 8 tokens, plus a 5-stage rescaled readout) and ran speculative decoding of a 5.2M-parameter code model on it: **1.21× faster end to end with identical output** (cycle-accurate RTL simulation, 3 prompts), up from 0.83× before the hardware/model changes.
- Measured every forward pass as shared + per-token cost (linear within 1.3%) and built a calibrated cycle model that predicts end-to-end runs within ~1%. It identified per-token readout, not memory bandwidth, as the limit, which led to the `ACCQ` instruction (−37% per-token cost).
- Verified the core and unit from block to SoC level: UVM (scoreboard, functional coverage, operand-corrupting driver), SVA, mutation testing, constrained-random instructions against a Python ISS, gate-level netlist simulation and waveform analytics with my own tool (154/154 multi-cycle instructions exactly 5 cycles).
- Built an integer-only transformer inference path that is bit-exact across Python, C on x86, the KV260's Cortex-A53 and the RISC-V core. The core + unit needs 2.0× fewer cycles per token than the A53 with NEON.
- Measured 1.6 GB/s DDR → FPGA → DDR DMA on a Kria KV260 with my own bitstream and a driver-free Linux test (99.8% of the port's limit), showing the board's memory can feed the unit's weight stream 3× over.
- Extended a course RISC-V core to full RV32IM and closed timing at 122.7 MHz on the KV260's chip (7.2K LUTs, 6.2%).

## 11. Attribution notes (for honest wording)

- The base core comes from the NUS CG3207 course project, built with a teammate; the RV32IM extension, testbench, assertions, custom ISA and everything model-side belong to this project. Course licence headers forbid a public repository; repo stays private until the professor replies (decision D11).
- Code is AI-assisted. The design decisions are mine, recorded with options and reasoning in `docs/review/decisions.md` and `docs/LOGBOOK.md`.

---

# Appendix: build history

## A1. Build history by phase

### A1.1 Model and speculative decoding (Python / PyTorch)
- Own data pipeline and BPE tokenizer (vocab 2048); 113,966,123 training tokens of Python code (measured).
- `model/model.py`: the transformer, with a self-check script covering shapes, causal masking, RoPE and a training step (8/8 pass, measured).
- Trained both models on a laptop RTX 5070 (measured): draft val loss 3.040 in 0.4 min; target val loss 1.955 in 8.3 min (about 20 tokens per parameter).
- `model/spec.py`: speculative decoding loop with a per-guess log. **Output identical to plain greedy decoding on 200/200 prompts at every γ (1, 2, 3, 4, 6, 8)** (measured).
- Measured acceptance: per-guess agreement ≈ 0.59, flat over γ; tokens per target pass 1.56 (γ1) → 1.85 (γ2) → 2.15 (γ4) → 2.28 (γ8) on 200 prompts. Matches the textbook formula (1 + α + … + α^γ) within a few percent.
- Found that later guesses in a round agree more often (slot 3: 0.82 vs slot 0: 0.57; measured, small sample), i.e. easy stretches cluster. That's the motivation for an adaptive draft-length predictor.

### A1.2 CPU core (Verilog / SystemVerilog)
- Extended the core to **full RV32IM**: byte/halfword loads and stores with byte enables and sign/zero extension, `mulh/mulhsu/mulhu` and `mul` moved to a 1-cycle DSP multiplier, `rem/remu`, RISC-V divide-by-zero semantics. Divider stays multi-cycle (32 cycles, measured).
- **Self-checking ISA test**: a Python generator computes expected results from the spec (504 op cases, 18 store, 52 load, 12 dependent chains); the core checks itself and reports `part << 16 | case` on failure. Passes with branch predictor off (10,755 cycles) and on (9,839 cycles) (measured, xsim).
- **SystemVerilog testbench with 7 SVA properties** (PC never X, legal byte enables, aligned stores, divider stall holds E and bubbles M, no X in computed register writes). 0 failures on all runs (measured).
- **Mutation check**: removing one pipeline-flush signal still let the ISA program print PASS, but the assertions fired 4,352 times. Shows that the assertions catch bugs an end-to-end test misses (measured).
- **Waveform accounting with WaveQ** (my own VCD → SQLite MCP server): mispredicts 502 → 44 with the predictor on; 458 × 2-cycle flush = 916 cycles = exactly the measured cycle difference. Divider busy 4,352 cycles = 136 divides × 32. 72 load-use stalls (measured).

### A1.3 Firmware and tooling
- Bare-metal C flow: RISC-V GCC (rv32im) → linker script + crt0 → memory images → xsim. **The same C source runs on the PC and on the core, and outputs are diffed automatically.**
- First integer kernel on the core: int8 matrix × vector with `mulh` requantisation, byte loads/stores, div/rem. **Core output identical to PC** (19 lines), 35,676 cycles, 0 assertion failures (measured).
- Simulation speed measured: ~55K simulated cycles/s plus ~4.4 s startup (measured), i.e. ~18 s per million cycles.
- `speccore.toml` single control file → generated C and Verilog headers; `tools/sc.py` runner (build, sim, PC-vs-core diff, ISA test).
- `tools/cyclemodel.py`: analytical cycle model per forward pass (weight stream, readout, attention, norms, overhead), software vs streamed attention, speedup over γ (estimates).
- `tools/bakeoff.py`: replays the logged guesses to compare draft-length predictors (fixed γ, confidence threshold, bimodal, gshare, perceptron, TAGE-lite, oracle). Borrowed directly from branch prediction.

### A1.4 Integer model, C reference, LLM unit (14-15 Nov)
- **int8 export + integer-only reference model** (`model/qmodel.py`): per-row weight scales as multiplier + shift, integer RMSNorm/softmax/RoPE, int8 KV cache. Top-1 agreement with the float model: draft 0.995, target 0.996 (measured, 1,280 teacher-forced tokens each). Integer speculative decoding identical to integer greedy on 20/20 prompts at γ 1, 2, 4 (measured).
- **C reference, bit-exact with the Python integer model** (all 16 golden tokens and logit sums, first run, measured). The same C runs on the core: the software-only draft step at ctx 4 takes 6,380,975 cycles, about 6.7 cycles per int8 MAC in plain RV32IM (measured, xsim). This is the "no new ISA" baseline.
- **LLM unit in SystemVerilog** in the E stage: 8 lanes × 4 int8 MACs, 8 activation banks, 16-word weight FIFO fed from the model image; 8 custom instructions (SETK, SETP, VDOTR, ACCRD, VDOTK, WPOP, BSTORE, WSTART). State changes only on commit, so a stall can't double-count. Directed test on the core: core output == PC C model, 0 SVA failures (measured).
- **5 unit SVA properties** plus a mutation (commit ignores the stall) → 7,003 assertion failures (measured).
- **UVM testbench for the unit** (`ext/tb/tb_vdotk_uvm.sv`): driver that behaves like the E stage, monitor, scoreboard (reference model of the unit), functional coverage (op, lane, K, op × stall), random legal programs. 10 seeds × 2,000 instructions: 2,382 scoreboard checks, 0 errors, 0 SVA failures; coverage 88-100% per seed, 100% on 3 seeds alone (measured). Mutation (off-by-one K mask, invisible to the SVA): 18 scoreboard errors, 0 SVA failures, so the scoreboard and SVA catch different bug classes (measured).
- **First end-to-end measurement on the core (xsim, one prompt, 8 tokens, output identical to plain greedy):** plain target 37.9M cycles; speculative γ2 45.9M → **0.83× (slower)**. The verify pass for 3 tokens costs 1.64× a single-token pass because only the weight stream is shared; attention, normalisation and readout are paid per token. This confirms the cycle model's Amdahl warning with real numbers and gives the baseline for the attention/model-shape fixes. Per-pass costs were ~2.5× the old analytical estimate; the model is now calibrated from measurements (4.5).
- Found and fixed with WaveQ: a driver bug (last instruction left valid on the bus, re-committed every cycle). Also found that an unbounded coverage cross (`[1:$]` bin + `with` filter) made xsim allocate memory without limit at time 0; every Vivado call now runs under a memory/time guard.

### A1.5 Measured cost split and calibrated cycle model (16 Nov)
- **K-sweep on the core** (`run_ref.c -DRR_KSWEEP`): the same pass at K = 1, 2, 3, 5 tokens, cycles per bucket (matvec, attention, norm/quant, other). Every bucket fits **cost = shared + per-token × K** within 1.3% (measured). Target pass: 2.58M shared (almost all of it the weight stream) + **2.14M per token checked**, so each extra token costs 0.45 of a full pass (measured, xsim, branch predictor off).
- **Calibrated cycle model** (`tools/cyclemodel.py --calib`): built from these measured costs plus measured acceptance. It predicts the measured end-to-end γ2 run within **0.9%** (45.47M vs 45.88M cycles). Speedup at γ1/γ2/γ4: 0.92× / 0.80× / 0.60× (measured pass costs × measured tokens per round).
- **Profiling the per-token cost:** norm/quant was 6.7× the old estimate. Cause, found in the disassembly: a 64-bit fixed-point helper compiled to ~26 RV32 instructions per element (3 multiplies + a 64-bit variable shift with branches). Rewrote it as a 32×32→64 multiply (`mul` + `mulh`) with two exact shift paths: **bit-exact** (0 mismatches over 28.8M random/edge cases; golden tokens identical on the core). Measured: norm/quant −25%, per-token cost −8%, speedup +0.01. Per-call-site counters showed why the first version gained less than expected (90% of the calls at some sites took the slow path).
- **Second end-to-end run** with the faster software and the core's branch predictor on (measured, identical output): plain 31.6M cycles (−16% vs the first run), speculative 38.0M (−17%), still **0.83×**: both get faster together, so the ratio is set by the shared/per-token split. The calibrated model predicted the speculative run within 1.2% before it ran.

### A1.6 KV260 board (16 Nov)
- Board reachable over SSH with a dedicated key, pinned host key and a narrow sudo rule (only FPGA app loading, package installs and two audited helper scripts).
- **The integer model runs bit-exact on the board's Cortex-A53** (same C source as on the RISC-V core and the PC). One A53 core at 1.33 GHz with NEON auto-vectorisation (-O3 -mcpu=cortex-a53 -funroll-loops): 4.76 ms/token for the target (210 tokens/s), i.e. 6.34M cycles/token (measured). Our core + LLM unit needs 3.98M cycles/token (measured in xsim): ~1.6× fewer cycles than a vectorised A53; the core + unit routes at 129 MHz on the KV260 chip (8 ns target met; measured, out-of-context), so in wall time it would be ~6.5× slower than the A53 (31 ms vs 4.76 ms per token, estimate from measured cycles and clock).
- **DDR bandwidth seen by the A53 cores** (STREAM-style): 4.5 GB/s copy on one core, 8.0 GB/s on four (measured). The planned weight stream needs 0.4 GB/s at 100 MHz, so the board's DDR is not the limit (estimate).
- **Own bitstream running on the board** (16 Nov): a "hello" FPGA app (8 KB AXI BRAM + ID register) built on the laptop under a hard memory cap, loaded with `xmutil loadapp` and a device-tree overlay, tested from Linux through UIO (no PYNQ): ID correct, 2,048 words × 3 patterns, 0 mismatches (measured). Debugged on the way: the KV260 firmware fixes the PS→PL bus at 128 bit, so a 32-bit build saw only every 4th word.
- **Core + LLM unit on the KV260's chip** (Vivado place-and-route, out-of-context, 16 Nov, measured): meets an 8 ns clock (129 MHz); 6,575 LUTs = 5.6% of the chip, 4 DSPs, no block RAM; the critical path is operand forwarding into the LLM unit's dot product. Getting there needed an RTL fix: the activation banks written as one 2-D array became 65,536 flip-flops and blew the synthesis memory cap; per-lane memories map to LUTRAM (behaviour and cycle counts unchanged, re-verified with UVM, the ISA test and a full golden run).
- **CPU → FPGA access speed on the board** (measured, own C benchmark through the hello app): reads ~450 ns each (9-34 MB/s), writes up to 266 MB/s. The weight stream needs ~460-520 MB/s, so on the board the weights must come through a hardware master (DMA or the core's own DDR port), not CPU copies.
- **Results page** `artifacts/charts/results.html` (generated from the measurement files): headline numbers, roofline of the LLM unit (45% → 21% of its peak as K grows, measured), speedup measured vs what-if, chip area by module.

### A1.7 Speculation pays: ACCQ instruction + better draft + DMA on the board (22 Nov)
- **ACCQ (new LLM-unit instruction):** reads a lane's accumulator already rescaled and rounded (the int32 multiplier + shift of the integer number system), in a 5-stage multi-cycle unit that holds E and latches its operands in the first cycle. Bit-exact with the integer reference. Used for matrix readout, the output layer, attention scores and the V readout. **Per-token cost of a verify pass 1.71M → 1.08M cycles (−37%); plain decoding 1.22× faster** (target pass 3.92M → 3.20M cycles; measured, xsim K-sweep).
- **Verification of ACCQ:** UVM with a driver that corrupts the forwarded operands during the hold (a missing latch gives 23 scoreboard errors, the mutation check), 9 SVA rules, a directed core test (154 checks: load-use, mul forwarding, back-to-back), an independent review + 22.1M-case arithmetic check against a C transcription of the RTL, and **WaveQ on the core waveform: 154 of 154 ACCQs take exactly 5 cycles, and the forwarded operands really do change mid-hold on the core** (so the latch is necessary). ISA regression unchanged.
- **Draft model:** the old draft saw only 13M training tokens. Training the same 128×2 shape on 228M tokens and distilling from the target (forward KL): tokens per target pass at γ2 1.85 → 2.52 on 200 prompts (float). Attribution: most of it comes from longer training (→ 2.46); distillation adds +2.4%. Smaller drafts (64×2, 128×1) lost more agreement than they saved.
- **First speedup > 1 on the core (measured, xsim): γ2 1.18×** on the golden prompt (21.90M vs 25.84M cycles, output identical, same hardware for both). **Estimate over 50 prompts with the core's exact integer arithmetic: γ2 1.15×** (γ1 1.13×, γ3 1.12×, γ4 1.04×). **Three prompts on the core (measured): 1.18×, 1.11×, 1.37× (Σplain/Σspec 1.21× over 40 tokens), none slower than plain, all identical to greedy.**
- **Adaptive draft length:** a training-free confidence gate (stop guessing when the draft's top-1/top-2 logit margin is < 1.0) is built in the integer loop and runs on the core (1.334× vs 1.180× fixed γ2 on one prompt, measured); over 50 prompts the gain is only about +2.5% (estimated, integer model × measured pass costs), so fixed γ2 stays the default. History predictors (bimodal/gshare/TAGE) lose to fixed γ2 here.
- **Clock after ACCQ:** 122.7 MHz post-route (OOC; was 129.2), 7,223 LUTs, 12 DSPs; critical path unchanged (forwarding → VDOTR dot product).
- **KV260 DMA loopback (measured on the board):** own bitstream (AXI DMA, 128-bit HP0, 100 MHz), buffers from the CMA heap: **1,597 MB/s per direction (99.8% of the 1.6 GB/s PL-side ceiling), 0 errors over 2 MB × 3 patterns.** The weight stream needs ~0.5 GB/s, so DDR can feed the unit ~3× over. Debugged on the way: hugepages sit in high DDR (not mapped on this HP0 config), and another overlay had left the PS→PL port at 32 bit.
- **Versus the board's A53 (same C, bit-exact):** our core now needs 3.22M cycles per target token vs 6.34M on an A53 with NEON (2.0× fewer cycles, measured); in wall time at 122.7 MHz it would still be ~5.5× slower (estimate).

## A2. Findings as they happened (oldest numbers are superseded by later ones)

- **Measured (22 Nov):** after moving the per-token readout into hardware (ACCQ) and training a better draft, speculation wins on the core: **1.18× measured on one prompt, ≈ 1.15× estimated over 50 prompts (γ2)**. Both levers were needed: ACCQ alone gives ≈ 1.00× (γ1) with the old draft; the better draft alone stays < 1× on the old per-token cost.

- **Measured (16 Nov):** a verify pass costs shared + 0.43-0.45 × (one pass) per extra token. Only the weight stream is shared, so with this software stack speculation is slower at every γ (best γ1 ≈ 0.93×). Software tuning of the per-token code moved it by ~0.01. Getting above 1× needs the per-token work itself cut: readout/requantisation in hardware, streamed attention, or a reshaped model. That is now a measured Amdahl argument, not an estimate.

- **Weight-stream-only estimate:** best fixed γ ≈ 2, about 1.5× speedup (estimated from measured tokens/pass and a 0.125 draft/target cost ratio).
- **Full cycle model (estimated):** once attention, per-row readout and norms are counted, they are paid K times per verify pass while only the weight stream is shared. With attention in software, speculation loses (γ2 ≈ 0.85×). With a streamed-attention instruction (O1) it is ≈ 1.18× (γ1). The ceiling for this model pair is ≈ 1.5×. This is an Amdahl's-law result, and it now drives the next hardware decisions.
- **Speedup audit (14 Nov, estimated):** speedup = T(γ) / (1 + γ·g), where g = (per-lane cost + draft pass) / plain pass. g is 0.59 today; about half of it is the draft pass, so model shape matters as much as ISA. Cheapest credible path found: a register-fed VDOTK variant ("VDOTR", ~1.17×) plus a wider retrained target (~1.3×), expected 1.2-1.3× overall. Not decided yet. Details: `docs/review/2026-11-14_speedup_audit.md`.
- **Predictor bake-off (provisional, 20-prompt log):** the draft model's own confidence carries most of the signal; history-based predictors add little; an oracle shows the ceiling (2.03 vs 1.49 tokens per cost unit for the best fixed γ). To be rerun on the 200-prompt log.


## A3. Cut or dropped (and why)

- **SDOT4** (4-way int8 dot product on registers, for the draft model with weights in on-chip memory): each use needs two loads plus the instruction, about 3-4 cycles per weight word vs 1 for VDOTK from the stream. That would push the draft cost ratio from 0.125 to about 0.44 and erase the speedup (estimated). The draft streams from DDR like the target instead, so one mechanism serves both.
- **n-gram / prompt-lookup drafter** (guess the next tokens by matching recent text, no draft model): kept only as a fallback or stretch idea, since the neural draft is the main story. It may come back (idea I3, "Token Target Buffer": a BTB-style table from recent tokens to the next token).
- HDMI output, power numbers.


## A4. Earlier resume bullets (superseded numbers kept for reference)

- Extended a 5-stage pipelined RISC-V core to full RV32IM (DSP multiplier, byte/halfword memory ops) and verified it with a spec-generated self-checking ISA test (586 cases) and 7 SystemVerilog assertions; a mutation test showed the assertions caught a pipeline bug the end-to-end test missed.
- Trained two transformer code models from scratch (0.66M / 5.2M params, own BPE tokenizer) and implemented speculative decoding with output identical to greedy on 200/200 prompts; measured 1.85-2.28 tokens per target pass.
- Built an analytical cycle model and a branch-predictor-style bake-off (bimodal, gshare, perceptron, TAGE-lite) for adaptive draft length; the model showed that non-weight work, not memory bandwidth, limits speculative speedup on a scalar core.
- Designed a custom LLM unit for the RISC-V pipeline (8 custom instructions; one streamed weight word feeds up to 8 tokens), verified with UVM (scoreboard, coverage, 10 seeds) and SVA; it made the matrix-vector work 9.3× faster than plain RV32IM C (measured, cycle-accurate simulation).
- Measured speculative decoding end to end on the core (output identical to greedy): 0.83× on one prompt. Separated every pass into shared vs per-token cost (linear within 1.3%) and built a calibrated cycle model that predicts the measured run within 0.9%, showing that per-token work (attention, requantisation, readout), not the weight stream, limits the speedup.

- Added a multi-cycle requantising readout instruction (ACCQ) to the RISC-V LLM unit, cutting per-token verify cost 37% and making speculative decoding win on the core: 1.18× end to end (measured, cycle-accurate), ≈1.15× over 50 prompts (estimate), output identical to greedy. Verified with UVM (stale-operand driver + mutation), SVA, directed hazard tests and waveform analytics (154/154 instructions exactly 5 cycles).
- Measured 1.6 GB/s DDR→FPGA→DDR DMA loopback on a Kria KV260 with my own bitstream and Linux driver-free test (99.8% of the port's ceiling), showing DDR can feed the custom unit's weight stream 3× over.
- Built an integer-only transformer inference path that is bit-exact across Python, C on x86, a Cortex-A53 (KV260 board) and my RISC-V core in cycle-accurate simulation; per token the core + custom unit needs 1.6× fewer cycles than the A53 with NEON (3.98M vs 6.34M, measured).
