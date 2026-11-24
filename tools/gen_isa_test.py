"""Generate a self-checking RV32IM test for the core: build/gen/isa_test.S.

Run it through `python tools/sc.py isa` (generates, builds, simulates with BP off and on).
Python computes every expected value from the RISC-V spec; the core must match.

Parts (exit code = part_id << 16 | case number, 0 = all pass):
  1-18  register-register ops, data-driven loops: lw a, lw b, lw expected, OP, bne.
        lw -> OP is a load-use hazard; OP -> bne uses M->E forwarding.
  50    dependent chains: mul -> mulh -> div -> rem -> add back to back (forwarding into
        the DSP multiplier and into MCycle, MCycle stall).
  60    stores sb/sh/sw at every offset, read back with lw.
  61    loads lb/lbu/lh/lhu/lw at every offset, used by the very next instruction.
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "gen" / "isa_test.S"
M = 0xFFFFFFFF
N_CASES = 28


def s(x):
    return x - (1 << 32) if x & 0x80000000 else x


def div_s(a, b):
    if b == 0:
        return M
    if s(a) == -(1 << 31) and s(b) == -1:
        return a
    q = abs(s(a)) // abs(s(b))
    return (-q if (s(a) < 0) != (s(b) < 0) else q) & M


def rem_s(a, b):
    if b == 0:
        return a
    if s(a) == -(1 << 31) and s(b) == -1:
        return 0
    return (s(a) - s(b) * s(div_s(a, b))) & M


OPS = {  # name: expected(a, b)
    "add": lambda a, b: (a + b) & M,
    "sub": lambda a, b: (a - b) & M,
    "sll": lambda a, b: (a << (b & 31)) & M,
    "slt": lambda a, b: int(s(a) < s(b)),
    "sltu": lambda a, b: int(a < b),
    "xor": lambda a, b: a ^ b,
    "srl": lambda a, b: a >> (b & 31),
    "sra": lambda a, b: (s(a) >> (b & 31)) & M,
    "or": lambda a, b: a | b,
    "and": lambda a, b: a & b,
    "mul": lambda a, b: (a * b) & M,
    "mulh": lambda a, b: ((s(a) * s(b)) >> 32) & M,
    "mulhsu": lambda a, b: ((s(a) * b) >> 32) & M,
    "mulhu": lambda a, b: ((a * b) >> 32) & M,
    "div": div_s,
    "divu": lambda a, b: M if b == 0 else a // b,
    "rem": rem_s,
    "remu": lambda a, b: a if b == 0 else a % b,
}
EDGES = [0, 1, 2, 3, 31, 32, 0x7F, 0x80, 0xFF, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFF,
         0xFFFFFFFE, 0x12345678, 0xDEADBEEF]


def operands(rng):
    pairs = [(0x80000000, 0xFFFFFFFF), (0xFFFFFFFF, 0), (5, 0), (0x80000000, 0)]  # overflow, x/0
    while len(pairs) < N_CASES:
        r = rng.random()
        if r < 0.4:
            pairs.append((rng.choice(EDGES), rng.choice(EDGES)))
        elif r < 0.7:
            pairs.append((rng.getrandbits(32), rng.choice(EDGES)))
        else:
            pairs.append((rng.getrandbits(32), rng.getrandbits(32) >> rng.randrange(32)))
    return pairs


def main():
    rng = random.Random(3207)
    data, code = [], []
    A = code.append

    A("    .text\n    .globl main\nmain:")
    A("    li    s9, 0xFFFF0000          # MMIO base")

    # ---- part 60: stores at every offset, read back with lw ----
    A("    li    s2, 60\n    li    s3, 0\n    la    s0, buf")
    for _ in range(6):
        w = rng.getrandbits(32)
        for kind, offs in (("sb", range(4)), ("sh", (0, 2)), ("sw", (0,))):
            off = rng.choice(list(offs))
            v = rng.getrandbits(32)
            if kind == "sb":
                exp = (w & ~(0xFF << 8 * off) | (v & 0xFF) << 8 * off) & M
            elif kind == "sh":
                exp = (w & ~(0xFFFF << 8 * off) | (v & 0xFFFF) << 8 * off) & M
            else:
                exp = v
            A(f"    addi  s3, s3, 1\n    li    t0, {w:#x}\n    sw    t0, 0(s0)\n    li    t1, {v:#x}\n"
              f"    {kind}    t1, {off}(s0)\n    lw    t2, 0(s0)\n    li    t3, {exp:#x}\n    bne   t2, t3, fail")

    # ---- part 61: loads at every offset, result used immediately (load-use stall) ----
    A("    li    s2, 61\n    li    s3, 0")
    for _ in range(4):
        w = rng.getrandbits(32) | 0x80808080 if rng.random() < 0.5 else rng.getrandbits(32)
        A(f"    li    t0, {w:#x}\n    sw    t0, 4(s0)")
        for kind, offs in (("lb", range(4)), ("lbu", range(4)), ("lh", (0, 2)), ("lhu", (0, 2)), ("lw", (0,))):
            for off in offs:
                if kind in ("lb", "lbu"):
                    v = (w >> 8 * off) & 0xFF
                    exp = (v - 0x100) & M if kind == "lb" and v & 0x80 else v
                elif kind in ("lh", "lhu"):
                    v = (w >> 8 * off) & 0xFFFF
                    exp = (v - 0x10000) & M if kind == "lh" and v & 0x8000 else v
                else:
                    exp = w
                A(f"    addi  s3, s3, 1\n    li    t3, {exp:#x}\n    {kind}   t2, {4 + off}(s0)\n"
                  f"    bne   t2, t3, fail")

    # ---- parts 1-18: register-register ops from tables ----
    for i, (op, f) in enumerate(OPS.items(), start=1):
        rows = [(a, b, f(a, b)) for a, b in operands(rng)]
        data.append(f"tab_{op}:\n" + "\n".join(f"    .word {a:#010x}, {b:#010x}, {e:#010x}" for a, b, e in rows))
        A(f"""    # ---- {i}: {op} ----
    li    s2, {i}
    li    s3, 0
    la    s0, tab_{op}
    li    s1, {len(rows)}
1:  addi  s3, s3, 1
    lw    a0, 0(s0)
    lw    a1, 4(s0)
    lw    a2, 8(s0)
    {op:<5} a3, a0, a1
    bne   a3, a2, fail
    addi  s0, s0, 12
    addi  s1, s1, -1
    bnez  s1, 1b""")

    # ---- part 50: dependent chains, every instruction uses the previous result ----
    A("    li    s2, 50\n    li    s3, 0")
    for _ in range(12):
        x, y = rng.getrandbits(32), rng.getrandbits(32) | 1
        a2 = (x * y) & M
        a3 = ((s(a2) * s(x)) >> 32) & M
        a4 = div_s(a3, y)
        a5 = rem_s(a4, a2)
        a6 = (a5 + a3) & M
        a7 = ((a6 * y) >> 32) & M          # mulhu right after add
        A(f"""    addi  s3, s3, 1
    li    a0, {x:#x}
    li    a1, {y:#x}
    mul   a2, a0, a1
    mulh  a3, a2, a0
    div   a4, a3, a1
    rem   a5, a4, a2
    add   a6, a5, a3
    mulhu a7, a6, a1
    li    t0, {a7:#x}
    bne   a7, t0, fail""")

    A("""    la    a0, msg_pass
    call  puts
    li    a0, 0
    j     sc_exit
fail:
    slli  a0, s2, 16
    or    a0, a0, s3
    mv    s4, a0
    la    a0, msg_fail
    call  puts
    mv    a0, s4
    j     sc_exit

puts:                           # a0 = string; polls UART TX ready, sends bytes
    lbu   t0, 0(a0)
    beqz  t0, 2f
1:  lw    t1, 8(s9)
    beqz  t1, 1b
    sw    t0, 12(s9)
    addi  a0, a0, 1
    j     puts
2:  ret

    .data
msg_pass: .asciz "ISA PASS\\n"
msg_fail: .asciz "ISA FAIL\\n"
    .align 2
buf: .word 0, 0""")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(code) + "\n" + "\n".join(data) + "\n", encoding="utf-8")
    n = sum(1 for _ in OPS) * N_CASES
    print(f"wrote {OUT.relative_to(ROOT)}: {n} op cases, 18 store cases, 52 load cases, 12 chains")


if __name__ == "__main__":
    sys.exit(main())
