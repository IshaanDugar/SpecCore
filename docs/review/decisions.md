# Decisions

All D1-D11 locked 4 Nov on the student's go ("ok good") after the "most optimal approach" summary.

Each decision: the options, what happens with each, and a table. "Rec" = my recommendation and why. All numbers are estimates.
Reasoning and sources: `2026-11-04_plan_review.md`. Mark your choice in the "Chosen" line and tell me; I log it.

---

## D1. Draft model size

- **A. Small, about 0.5-0.7M (plan).** Guesses less well. Costs about 1/8 of a target pass per guess, so speculation can win (about 1.5-2.2x).
- **B. Big, 3-5M.** Guesses better. But each guess costs 70-100% of a target pass, so the total gets slower than plain decoding. It also doesn't fit on-chip.
- **C. Tiny, about 0.4M (dim 96).** Cheapest guesses. Accepts fewer of them; good as a fallback if A turns out too slow.

| | A small | B big | C tiny |
| --- | --- | --- | --- |
| Guess quality | medium | high | lower |
| Cost per guess | ~1/8 pass | ~3/4 pass | ~1/12 pass |
| Likely speedup | 1.5-2.2x | under 1x | 1.3-2x |
| Fits on-chip | yes | no | yes |
| Extra work | none | none | none |

Rec: A. Chosen: A (4 Nov, student: "ok good")

## D2. Vocabulary size (shared tokenizer)

- **1024:** smaller output layer (cheaper passes, smaller draft). Each token covers less text, so more tokens per line.
- **2048:** middle ground.
- **4096:** more text per token, but the output layer grows to 40%+ of the draft, so guesses get pricier.

| | 1024 | 2048 | 4096 |
| --- | --- | --- | --- |
| Draft size (dim 128) | 0.52M | 0.66M | 0.92M |
| Text per token | least | middle | most |
| Cost per guess | lowest | low | higher |

**Measured 4 Nov** (byte-level BPE trained on 50 MB, tested on 2.8 MB of validation code):

| | 1024 | 2048 | 4096 |
| --- | --- | --- | --- |
| Characters per token | 2.21 | 2.54 | 2.84 |
| Target bytes read per character of code | 2.25 | **2.07** | 2.03 |
| Draft bytes read per character of code | **0.24** | 0.26 | 0.32 |
| Code that fits in a 128-token context | 283 chars | 325 chars | 364 chars |

- 2048 vs 4096: the target is almost equal per character (2% apart). The draft gets 25% more expensive at 4096.
- 1024: the cheapest draft, but the target costs 9% more per character, and the context covers less code.

Rec: 2048. Chosen: 2048 (4 Nov)

## D3. Where the draft's weights live

- **A. On-chip, read with SDOT4 (plan).** Every 4 MACs need two loads plus SDOT4: about 3-4 cycles per word. Speedup drops to about 1x.
- **B. In DDR, streamed through the weight port like the target.** 1 cycle per word. Uses hardware you build anyway. SDOT4 may not be needed at all.
- **C. On-chip, but streamed by a small on-chip reader into the weight port.** Same speed as B; more hardware; frees DDR traffic (not a bottleneck anyway).

| | A on-chip + SDOT4 | B DDR stream | C on-chip stream |
| --- | --- | --- | --- |
| Cycles per weight word | 3-4 | 1 | 1 |
| Likely speedup (α 0.7) | ~1.0x | ~1.85x | ~1.85x |
| New hardware | SDOT4 | none extra | stream reader |
| Story | "draft on-chip" | "both stream; draft is 8x smaller" | "draft on-chip, streamed" |

Rec: B. Chosen: B: both models stream from DDR; SDOT4 dropped (4 Nov)

## D4. How activations become int8

- **A. One scale per token vector, computed at run time.** Fits VDOTK: each lane (token) has its own scale. One max-find plus quantise per vector: cheap.
- **B. Group scales (every 32-64 values, like llama2.c).** More accurate. The accumulator must be read and rescaled every group, which breaks VDOTK's long sums and adds many instructions.
- **C. One fixed scale per layer, measured offline.** Cheapest at run time. Least accurate; breaks if values vary a lot between prompts.

| | A per token | B groups | C fixed |
| --- | --- | --- | --- |
| Accuracy | good (measure) | best | weakest |
| Run-time cost | low | high | lowest |
| Fits VDOTK | yes | badly | yes |

Rec: A; switch to B only if the int8 gate fails. Chosen: A (4 Nov)

## D5. Scale number format

- **A. int8 scale (as written in the plan).** Only 127 steps; scales get rounded badly, so accuracy suffers.
- **B. int32 fixed-point multiplier + shift (Jacob 2018).** Standard integer-only method. Needs `mulh`, which you add anyway.
- **C. Software float.** Simplest to write; very slow on a core with no FPU (tens of ms per token).

| | A int8 | B fixed-point | C soft float |
| --- | --- | --- | --- |
| Accuracy | poor | good | best |
| Speed on core | fast | fast with DSP multiplier | very slow |
| Needs | nothing | `mulh` | nothing |

Rec: B. Chosen: B (4 Nov)

## D6. Softmax, RMSNorm, RoPE (no FPU)

- **A. Integer versions: lookup tables plus Newton steps.** Fast and bit-exact between PC and core. More thinking up front in the C reference.
- **B. Software float.** Easy to write, matches PyTorch closely, but adds tens of ms per token.
- **C. A small hardware unit (e.g. exp).** Fastest; more RTL and tests. Overkill at this model size.

| | A integer | B soft float | C hardware |
| --- | --- | --- | --- |
| Speed | fast | slow | fastest |
| Effort | medium (C) | low | high (RTL) |
| Bit-exact PC vs core | yes | yes | needs care |

Rec: A. Chosen: A (4 Nov)

## D7. Making data memory big (BRAM/UltraRAM read one cycle late)

- **A. Send the address one stage early (from E).** Data arrives in M as today. The hazard unit doesn't change. The ALU-to-memory path may limit the clock.
- **B. Keep the address in M and add a cycle.** Easier timing. Loads take one cycle longer: a 2-cycle load-use stall and new forwarding.

| | A address from E | B extra cycle |
| --- | --- | --- |
| Hazard changes | none | stall + forwarding |
| Load speed | same | slower |
| Timing risk | higher | lower |

Rec: A; fall back to B if timing fails. Chosen: A (4 Nov)

## D8. Multiplier for normal `mul`

- **A. Keep MCycle (about 17 cycles).** No work. Scaling outputs costs about 5 ms per token (half a verify pass).
- **B. DSP multiplier, 1-2 cycles, for mul/mulh; MCycle kept for divide.** Small RTL change. Scaling drops to about 0.6 ms. The "no extensions" baseline still uses MCycle.

| | A MCycle | B DSP |
| --- | --- | --- |
| Cycles per mul | ~17 | 1-2 |
| Work | none | small + tests |
| Effect on speed | big slowdown | removes it |

Rec: B. Chosen: B (4 Nov)

## D9. Attention cost

- **A. Plain loop with SDOT4.** Simple. At K ≥ 4 it can take longer than the weight pass, which caps the speedup.
- **B. Stream the KV cache through VDOTK (K tokens share the same keys).** Attention gets the same K-fold reuse. Needs an on-chip stream reader and values stored transposed.
- **C. Decide after measuring on 8 Nov.**

Rec: C. Chosen: C, measure 8 Nov. Note: with SDOT4 dropped, attention needs a fast path anyway; see ISA options (4 Nov)

## D10. Dashboard display

- **A. HDMI (plan).** Probably needs a video pipeline in the FPGA (unverified). Days of work, no architecture value.
- **B. Browser on your laptop over Ethernet.** No extra hardware; easiest to record.
- **C. DisplayPort monitor.** Driven by the ARM side; needs a DP monitor or active adapter.

| | A HDMI | B browser | C DP |
| --- | --- | --- | --- |
| Extra work | high (likely) | low | low |
| Needs | HDMI monitor | laptop | DP monitor/adapter |
| Demo feel | board-only | laptop | board-only |

Rec: B, optional C. Chosen: B; DP optional (4 Nov)

## D11. Public repo vs course licence

- **A. Ask Dr Panicker for permission now.** One email; you can publish if he agrees.
- **B. Rewrite the derived files.** No permission needed. Costs days.
- **C. Keep it private; share on request.** No risk; less visible to admissions.

Rec: A, with C as the default until he answers. Chosen: A + C: email now, private until answered (4 Nov)

---

## D12. Draft-length predictor (12 Nov, pending)

Decides each round whether to keep guessing or stop and let the target check. The "branch" is "will my next guess be accepted?". Explained with a live perceptron demo in primer 02, step 7.

- **A. Fixed γ = 4.** Nothing to build. The baseline every other option is compared against.
- **B. Bimodal 2-bit counters** (the BranchPredictor.v scheme), indexed by the last token. About 1 h on the PC.
- **C. gshare:** last token XOR the recent accept/reject history. Learns "after a miss it stays hard". About 1 h.
- **D. Perceptron:** recent outcomes + draft confidence (gap between its top two scores), small int weights, adds only. About 2 h. Same family as AMD Zen's predictor.
- **E. TAGE:** several tagged tables with geometric history lengths. About 1 day. Little to learn from ≤8 outcomes per round.

| | A fixed | B bimodal | C gshare | D perceptron | E TAGE |
| --- | --- | --- | --- | --- | --- |
| Looks at | nothing | last token | token + history | history + confidence | many histories |
| PC effort (est.) | 0 | 1 h | 1 h | 2 h | 1 day |
| Hardware | none | counter table | + XOR | adders, int weights | tagged tables |
| Story | baseline | "my predictor, reused" | classic | "a neural net in my CPU trusts a neural net" | overkill |

Rec: run a bake-off of all five on the PC using one logged trace (about ½ day, est.), then put the winner in firmware, and in hardware only if time allows. Expected gain over the best fixed γ is modest (5-20%, est.); the bake-off chart is the main value.
Chosen: bake-off, then explain why the winner was the best choice (12 Nov, student: "we can benchmark for the BP and compare and explain our choices on which was the best optimal choice").

## D13. How to cut the per-token cost of a verify pass (16 Nov, pending, student decides after exams)

**What the measurements say (16 Nov, xsim, fast software + branch predictor on, ctx 15):** a target pass = **2.23M shared** (the weight stream, paid once) + **1.71M per token checked** (readout + bank fill 0.82M, attention 0.43M, norm/quant 0.37M, other 0.09M). End to end: plain 31.6M vs speculative γ2 38.0M → **0.83×**. Software tuning moved plain and speculative together (−17% each), so the ratio is set by the split. To win, the per-token part must shrink; the plain pass gets the same saving, so the gain is smaller than it looks. Speedups below come from `python tools/cyclemodel.py --calib build/ksweep_fast2_bp_core.txt` (estimates built on measured costs). **At ctx 64 (measured 16 Nov, `build/ksweep_ctx64_bp_core.txt`):** 2.43M shared + 2.03M per token (attention per token 0.74M vs 0.43M at ctx 15); speedup γ1 0.92× unchanged; readout+fill −80% → 0.99×, attention −80% → 0.96×, all per-lane −80% + draft −50% → 1.26×. Ranking unchanged; attention weighs more at long context.

- **A. Stop here (software only).** Nothing new to build. Result stays 0.83×; the story is "measured why speculation loses on a scalar core, predicted it within 1%".
- **B. Readout in hardware ("ACCQ-lite").** ACCRD returns the already scaled and rounded value (per-lane multiply + shift in the unit) instead of the core doing 64-bit fixed-point math per lane per row. Hits the biggest bucket (0.82M/token). Alone: γ1 ≈ 0.98× (−50%) to 1.02× (−80%). RTL: one multiplier + shifter per lane next to the accumulators (~5 h, est.).
- **C. Norm/quant in hardware.** An instruction (or small unit) that finds the vector max and requantises to int8 while writing the banks. Alone: γ1 ≈ 0.94-0.95×. Also removes part of the bank-fill cost if it writes the banks directly.
- **D. More attention in hardware.** Softmax per lane in hardware, or streaming the KV cache (O1). Alone: γ1 ≈ 0.94-0.95× at ctx 15; matters more at long context (ctx-64 measurement running).
- **E. B + C.** γ1 ≈ 1.07×, γ2 ≈ 0.99×.
- **F. B + C + D + a cheaper draft (D15).** All per-token −80% and draft −50%: **γ1 ≈ 1.27×, γ2 ≈ 1.28×**. The most work.

| | A | B | C | D | E (B+C) | F (all + D15) |
| --- | --- | --- | --- | --- | --- | --- |
| Speedup γ1 (est. on measured) | 0.83 (measured) | 0.98-1.02 | 0.94-0.95 | 0.94-0.95 | 1.07 | 1.27 |
| New RTL | none | 1 mult + shift per lane | max + requant path | softmax/stream path | both | everything |
| Effort (est.) | 0 | ~5 h | ~6 h | ~8-16 h | ~11 h | ~20 h + retrain |
| Plain decoding also faster | - | yes | yes | yes | yes | yes |
| Story | "measured Amdahl" | "biggest lever first" | | | ">1× on my CPU" | "1.3× end to end" |

Rec: **B first** (largest single bucket, smallest RTL, measurable in one K-sweep), then C if time; D only if ctx 64 shows attention dominating. Decide after exams (needs T5 memory-bound, T6 spec decoding, T8 hardware).
Chosen: **B (ACCQ), taken by the AI on the student's delegation (22 Nov), reviewed by one independent advisor agent; student to confirm or reverse.** Built: SETS + 5-stage ACCQ (`ext/rtl/sc_vdotk.sv`), used in matvec readout, output layer, attention scores and V readout (`ref/qref.c -DQR_ACCQ`). Measured (xsim K-sweep, ctx 15): per-lane cost 1.71M → 1.08M (−37%), plain target pass 3.92M → 3.20M (plain decoding 1.22× faster). Advisor note: well-tuned software alone might reach ~−55% of the readout bucket (est.), so ACCQ's margin over good software is smaller than its margin over today's code.

## D14. γ policy: fixed γ or a predictor (16 Nov, pending; supersedes the open part of D12)

**Measured evidence:** bake-off on the 200-prompt log with measured costs (`python tools/bakeoff.py --cost 0.19 --lane 0.43`): with today's costs, **plain decoding (1.00) beats every buildable policy** (best: confidence gate conf>1.5 0.965, fixed γ1 0.946); oracle 1.27. With the D13-F hardware (`--cost 0.084 --lane 0.132`): perceptron 1.306, confidence gate 1.302, **fixed γ2 1.277**, fixed γ1 1.256; oracle 1.79. History predictors (bimodal, gshare, TAGE-lite) are always worse than the draft's own confidence.

- **A. Fixed γ1.** Simplest; best fixed value today. 1.256 with fixed hardware.
- **B. Fixed γ2.** Best fixed value once the per-token cost is cut. 1.277.
- **C. Confidence gate** (keep guessing while the draft's top-1 margin > threshold). No state, a compare per guess. 1.302.
- **D. Perceptron** (history + confidence). Best measured (1.306) but only +0.004 over C; more to explain and build.
- **E. Auto off-switch** on top of any of the above: decode plainly when the measured speedup is < 1 (today's case).

| | A γ1 | B γ2 | C conf gate | D perceptron | E off-switch |
| --- | --- | --- | --- | --- | --- |
| tok/cost today | 0.946 | 0.821 | 0.965 | 0.903 | 1.000 |
| tok/cost with D13-F | 1.256 | 1.277 | 1.302 | 1.306 | same as base |
| State | none | none | none | ~3.6 Kbit | 1 flag |
| Story | | | "the draft knows when it's unsure" | "branch predictor for words" | honest engineering |

Rec: **B (fixed γ2) as the default + C as the adaptive option**, with the bake-off chart as the story (shows why history predictors lose here, unlike branches). D only if you want the branch-predictor pitch; the gain is within noise. Add E if D13 stays at A.
Chosen: **fixed γ2 as the default (AI on delegation, 22 Nov; student to confirm); confidence gate (margin > 1.0 logit, cap 4) built as the adaptive option.** Integer model, 50 prompts: γ2 is the best fixed value (est. 1.15× vs γ1 1.13×, γ3 1.12×, γ4 1.04×); measured on the core γ2: 1.18× / 1.11× / 1.37× on three prompts. Advisor disagreed with fixing γ2 before D15; after D15 (distilled draft) the calibrated model gives γ1 1.17×, γ2 1.21×, γ3 1.21×, γ4 1.16× (estimates on measured costs): γ2 and γ3 tie, γ2 needs fewer lanes. Predictor bake-off still to be rerun on the new guess logs (`logs/d15/d128x2_kd/`) with the peek cost charged.

## D15. Draft cost (16 Nov, pending; needs retraining → needs your go)

**Measured:** draft pass = 0.199 of a target pass (K=1, fast2 + BP); per round ≈ 1.5 draft passes incl. catch-up → ~0.30 of a target pass per round at γ2, about a third of the round's overhead. Agreement with the target ≈ 0.59 per guess (13 Nov). What-if: draft −50% alone → γ1 1.00×; with D13-F it is part of the 1.27×.

- **A. Keep 128 dim × 2 layers.** No retraining. Draft stays ~0.2 of a target pass.
- **B. 64 dim × 2 layers** (~0.2M params). Roughly half the draft cost (est.); agreement will drop by an unknown amount. Training ~0.4 min (est.), then α and costs re-measured.
- **C. 128 dim × 1 layer.** Halves the layer work but keeps the output layer (the 2048 × 128 readout is a big part of the draft). Agreement drop unknown.
- **D. No neural draft (n-gram / prompt lookup).** Draft cost ≈ 0, but acceptance is far lower on new code. Was cut on 4 Nov as the main path.
- **E. Distil the draft from the target** (train on the target's outputs). Same cost as A or B, higher agreement (est.). More training work.

| | A keep | B 64×2 | C 128×1 | D n-gram | E distil |
| --- | --- | --- | --- | --- | --- |
| Draft cost (est.) | 0.20 | ~0.10 | ~0.13 | ~0 | as A/B |
| Agreement | 0.59 (measured) | lower, unknown | lower, unknown | much lower | higher (est.) |
| Retrain | no | yes (minutes) | yes (minutes) | no model | yes (longer) |
| Measure after | - | α + K-sweep | α + K-sweep | α | α + K-sweep |

Rec: **train B and E-style variants quickly and measure α before choosing** (training only with your go). Keep A until then.
Chosen: **E, distilled 128×2 draft (`d128x2_kd`), taken by the AI on delegation with the student's training go (22 Nov); student to confirm.** Measured (spec.py, 200 prompts, all identical to plain): tokens/target pass γ2 1.85 → 2.52, γ4 2.15 → 3.59 at the same draft cost. **Attribution (correction 22 Nov, from the Codex review): the old draft saw 13M training tokens, the new one 228M; at equal training length (d128x2_long, no distillation) γ2 is 2.46, so most of the gain is longer training (1.85 → 2.46) and distillation adds +2.4% (2.46 → 2.52).** B (64×2) and C (128×1) lost: agreement fell more than cost (γ2 1.93 / 1.98). Old draft kept as `checkpoints/draft_v1.pt`.

## D16. What to do with the KV260 board before the freeze (16 Nov, pending)

**Measured 16 Nov (read-only SSH, dedicated key, strict host check; access per `<local path>`):** board `kria`, Ubuntu 22.04.4, kernel 5.15.0-1027-xilinx-zynqmp, 4 cores, 3.9 GB RAM (3.5 GB free), SD 25 GB free of 29 GB. FPGA manager "operating", only the stock `k26-starter-kits` app in `/lib/firmware/xilinx`. `xmutil` and `dfx-mgr-client` present (load a custom bitstream + overlay as an "app"). `/dev/udmabuf` and `/dev/uio0-3` present (DMA buffers and register access without PYNQ). **PYNQ not installed.** Laptop: Vivado 2025.2 has the `xck26` device (bitstreams possible); the KV260 board preset is not downloaded (optional). A bitstream build needs roughly 3-5 GB of laptop memory (estimate), above today's 2-3 GB cap.

- **A. No board (simulation only).** Headline stays the xsim measurement. Zero risk to the laptop; the project shows no hardware run.
- **B. DMA loopback, no PYNQ.** Small Vivado design (Zynq PS + AXI DMA + FIFO loop), load with `xmutil loadapp`, a small C program on the board moves buffers through `udmabuf`/`uio` and reports MB/s. Measures the real DDR→PL bandwidth that feeds the weight stream (the number the roofline needs). Needs a bitstream build (memory plan) and `sudo` on the board to install the app files.
- **C. Install Kria-PYNQ, then the same loopback in Python.** Easier board code and a dashboard path (D10), but a large install (internet, root, ~1 h, several GB on the SD, est.) and a changed board.
- **D. Full SpecCore on the board.** Core + LLM unit + DDR through an AXI master, the model streamed from real DDR, cycles from the on-chip counter. The strongest claim; the most work and timing risk (days, est.).

| | A sim only | B loopback, no PYNQ | C PYNQ + loopback | D full core |
| --- | --- | --- | --- | --- |
| Board changes | none | app files in /lib/firmware (sudo) | big install | app files + driver |
| Laptop memory | sims only | bitstream build 3-5 GB (est.) | same as B | larger builds |
| Measures | nothing on hardware | real DMA MB/s | real DMA MB/s | real end-to-end tokens/s |
| Effort (est.) | 0 | ~1 day | ~1 day + install | several days |
| Story | "cycle-accurate" | "measured the memory wall on hardware" | + dashboard | "runs on the board" |

Rec: **B** after exams if the laptop memory plan is agreed (that number makes the roofline real); D only if B goes quickly. The access rules in the guide stay in force (console-observed address, strict host key, no installs without approval).
**Added 16 Nov (session 10, measured):** the board flow works end to end without PYNQ (hello app: own bitstream, `xmutil loadapp`, UIO; ID + 8 KB BRAM test PASS; HPM0 must be built 128-bit). CPU access to the fabric over HPM0 (`board/bench/mmio_bw.c`): reads ~450-470 ns each (9-34 MB/s), posted writes up to 266 MB/s (128-bit NEON / memcpy). The weight stream needs ~460-520 MB/s at the measured 115.7-129 MHz core clock, so CPU copies cannot feed it: options B and D both need a PL-side master (AXI DMA or the core's own DDR read port). Core + LLM unit fits easily: 6,570 LUTs (~5.6% of the chip's 117,120), 4 DSP, 0 BRAM; routes at 129 MHz (8 ns target, OOC).
Chosen: **B, DMA loopback without PYNQ (AI on delegation, 22 Nov): DONE, measured 1,597 MB/s per direction (99.8% of the 128-bit × 100 MHz PL ceiling), 0 errors.** D (full SpecCore on the board) not attempted before the freeze.
