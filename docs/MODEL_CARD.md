# Model card: SpecCore draft and target models

Two small decoder-only transformers trained from scratch to write Python code. They exist to run on a custom RISC-V core with speculative decoding: the **draft** guesses tokens cheaply and the **target** checks them. All numbers are measured unless marked otherwise.

## 1. Models at a glance

| | Draft | Target |
| --- | --- | --- |
| Parameters | 656,000 | 5,246,208 |
| Width / layers / heads | 128 / 2 / 4 | 256 / 6 / 8 |
| Context | 128 tokens | 128 tokens |
| MLP | 4× width, ReLU | 4× width, ReLU |
| Training tokens seen | 13M (1,601 steps × 8,192) | 105M (12,808 steps × 8,192, ≈ 20 tokens/param) |
| Peak learning rate | 0.002 | 0.001 |
| Training time (RTX 5070 Laptop) | 0.4 min | 8.3 min |
| Validation loss (nats/token) | 3.040 | 1.955 |
| int8 export size (weights + row scales) | 977,584 B | 5,919,600 B |

Where the target's parameters are: MLP 3.15M, attention 1.57M, embedding 0.52M.

## 2. Architecture

- Pre-norm blocks: `x = x + attn(rmsnorm(x)); x = x + mlp(rmsnorm(x))`, final RMSNorm, then the output layer.
- Attention: fused QKV projection, rotary position embedding (RoPE) on Q and K, causal mask. No biases anywhere.
- Float model: the output layer shares weights with the token embedding (tied). The int8 export un-ties them (see §5).
- Checkpoint parameter names are listed in the `model/model.py` header and are kept fixed.

## 3. Data and tokenizer

- Source: `codeparrot/codeparrot-clean` (Python files from GitHub), streamed. **Kept only files with permissive licences** (MIT, Apache-2.0, BSD-2/3, ISC), skipped files over 64 KB. 1% held out for validation.
- Tokenizer: byte-level BPE, vocabulary 2,048, trained on the same data (`model/tokenizer/bpe2048`). Token 0 = end of text.
- 113,966,123 training tokens after tokenisation. Example: `def add(a, b):\n    return a + b\n` = 14 tokens.

## 4. Speculative decoding behaviour

| | γ = 1 | γ = 2 | γ = 4 | γ = 8 |
| --- | --- | --- | --- | --- |
| Tokens per target pass (float, 200 prompts) | 1.56 | 1.85 | 2.15 | 2.28 |
| Tokens per target pass (integer model, 20 prompts) | 1.53 | 1.86 | 2.10 | — |

- Output is **identical to plain greedy decoding** on 200/200 prompts at every γ tested (1, 2, 3, 4, 6, 8; float), and on 20/20 prompts at γ 1, 2, 4 (integer).
- Per-guess agreement (draft = target's choice) ≈ 0.58-0.59, about flat over γ. The draft's own confidence predicts acceptance strongly (0.24 below confidence 0.1, 0.97 above 3).

## 5. Integer (int8) version

Used by the C reference and the core. Integer-only arithmetic, right shifts only.

- Weights int8 with one scale per output row, stored as a 30-bit multiplier + shift.
- RMSNorm gains folded into the next matrix, so the output layer is no longer tied (+0.52 MB).
- Activations int8 with a per-token scale; residual stream int32 (Q12); K/V cache int8 with a scale per (position, head).
- Softmax in base 2 with a 256-entry Q15 table; RoPE from a Q14 table.
- **Agreement with the float model** (teacher-forced top-1, 20 prompts × 64 tokens): draft 0.995, target 0.996. Free-running greedy text is identical for 14/20 (draft) and 15/20 (target) prompts over 64 tokens; first divergence on average at token 51 / 54.
- The C reference (`ref/qref.c`) matches the Python integer model bit for bit (tokens and logit sums), on the PC and on the simulated core.

## 6. Intended use and limits

- **Intended use:** a workload for hardware research (speculative decoding on a small CPU with a custom instruction). The point is the speed and exactness of decoding, not the quality of the code.
- **Not for:** writing real code. A 5M-parameter model produces plausible-looking Python that is often wrong. It has no safety tuning, no instruction following, and it only knows Python.
- Context is 128 tokens; anything longer is cut.
- Training data is public GitHub code. It may repeat snippets from it, including licence headers and author names.
- No evaluation beyond validation loss, draft/target agreement and float/integer agreement (no pass@k or human evaluation).

## 7. Reproduce

```
python model/data.py --mb 300               # download + licence filter
python model/train_tokenizer.py --sample_mb 50
python model/prepare.py --vocab 2048
python model/train.py --model draft  --hours 1.5   # stops at its token budget (0.4 min here)
python model/train.py --model target --hours 2     # 8.3 min here; logs in checkpoints/<model>.out
python model/export.py          # int8 export + model.img
python model/qmodel.py --agree  # float vs integer agreement
```
