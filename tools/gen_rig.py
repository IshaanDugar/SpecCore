"""Constrained-random RV32IM/custom instruction test, checked by tools/iss.py.

python tools/gen_rig.py SEED [N]       # default N=2000; no compiler/simulator invoked
python tools/gen_rig.py --self-test   # five seeds plus checker fault injection, in memory
See docs/rig.md for build and simulation commands.
"""
import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from iss import ISS, MASK, Memory, signed

ROOT = Path(__file__).resolve().parents[1]
MAIN, DATA = 0x00401000, 0x10000000
DATA_WORDS, DRAIN, SNAP = 64, 256, 288
SIG, EXPECT, ACTUAL = 416, 420, 548
EDGES = [0, 1, 2, 31, 32, 127, 128, 255, 0x7FFFFFFF, 0x80000000,
         0x80000001, 0xFFFFFFFF, 0xFFFFFFFE, 0x12345678, 0xDEADBEEF]
R_OPS = {"add": (0, 0), "sub": (0, 32), "sll": (1, 0), "slt": (2, 0),
         "sltu": (3, 0), "xor": (4, 0), "srl": (5, 0), "sra": (5, 32),
         "or": (6, 0), "and": (7, 0),
         **{op: (i, 1) for i, op in enumerate(
             ("mul", "mulh", "mulhsu", "mulhu", "div", "divu", "rem", "remu"))}}
I_OPS = {"addi": 0, "slti": 2, "sltiu": 3, "xori": 4, "ori": 6, "andi": 7,
         "slli": 1, "srli": 5, "srai": 5, "lb": 0, "lh": 1, "lw": 2, "lbu": 4, "lhu": 5}
BRANCH = {"beq": 0, "bne": 1, "blt": 4, "bge": 5, "bltu": 6, "bgeu": 7}


def r_word(op, rd, a, b):
    f3, f7 = R_OPS[op]
    return f7 << 25 | b << 20 | a << 15 | f3 << 12 | rd << 7 | 0x33


def i_word(op, rd, a, imm):
    if op in ("slli", "srli", "srai"):
        assert 0 <= imm < 32
        imm |= 0x400 if op == "srai" else 0
    else:
        assert -2048 <= imm < 2048
    opcode = 0x03 if op in ("lb", "lh", "lw", "lbu", "lhu") else 0x13
    if op == "jalr":
        opcode, f3 = 0x67, 0
    else:
        f3 = I_OPS[op]
    return (imm & 4095) << 20 | a << 15 | f3 << 12 | rd << 7 | opcode


def s_word(op, value, base, imm):
    assert -2048 <= imm < 2048
    return ((imm & 0xFE0) << 20 | value << 20 | base << 15 |
            {"sb": 0, "sh": 1, "sw": 2}[op] << 12 | (imm & 31) << 7 | 0x23)


def b_word(op, a, b, offset):
    assert offset % 4 == 0 and -4096 <= offset < 4096
    return (((offset >> 12) & 1) << 31 | ((offset >> 5) & 63) << 25 |
            b << 20 | a << 15 | BRANCH[op] << 12 |
            ((offset >> 1) & 15) << 8 | ((offset >> 11) & 1) << 7 | 0x63)


def j_word(rd, offset):
    assert offset % 4 == 0 and -(1 << 20) <= offset < 1 << 20
    return (((offset >> 20) & 1) << 31 | ((offset >> 1) & 1023) << 21 |
            ((offset >> 11) & 1) << 20 | ((offset >> 12) & 255) << 12 | rd << 7 | 0x6F)


class Program:
    """Tiny encoder. Only label fixups are deferred; every emitted instruction is 4 bytes."""
    def __init__(self, base=MAIN):
        self.base, self.code, self.labels = base, [], {}

    @property
    def pc(self):
        return self.base + 4 * len(self.code)

    def label(self, name):
        assert name not in self.labels
        self.labels[name] = self.pc

    def emit(self, word, text):
        self.code.append((word, text))

    def r(self, op, rd, a, b):
        self.emit(r_word(op, rd, a, b), f"{op} x{rd},x{a},x{b}")

    def i(self, op, rd, a, imm):
        self.emit(i_word(op, rd, a, imm), f"{op} x{rd},x{a},{imm}")

    def s(self, op, value, base, imm):
        self.emit(s_word(op, value, base, imm), f"{op} x{value},{imm}(x{base})")

    def u(self, op, rd, imm):
        self.emit((imm & 0xFFFFF) << 12 | rd << 7 | (0x37 if op == "lui" else 0x17),
                  f"{op} x{rd},{imm & 0xFFFFF:#x}")

    def li(self, rd, value):
        value = signed(value)
        high = (value + 0x800) >> 12
        self.u("lui", rd, high)
        self.i("addi", rd, rd, value - (high << 12))

    def custom(self, op, lane=0, rd=0, a=0, b=0):
        f3, f7 = {"setk": (0, 0), "setp": (1, lane), "setpa": (1, 64),
                  "sets": (1, 32 + lane), "vdotr": (2, 0), "accrd": (3, lane),
                  "accq": (3, 64 + lane), "bstore": (6, lane)}[op]
        assert 0 <= lane < 8
        self.emit(f7 << 25 | b << 20 | a << 15 | f3 << 12 | rd << 7 | 0x0B,
                  f"{op} lane={lane},rd=x{rd},rs1=x{a},rs2=x{b}")

    def branch(self, op, a, b, label):
        pc = self.pc
        self.emit(lambda: b_word(op, a, b, self.labels[label] - pc), f"{op} x{a},x{b},{label}")

    def jump(self, rd, label):
        pc = self.pc
        self.emit(lambda: j_word(rd, self.labels[label] - pc), f"jal x{rd},{label}")

    def words(self, start=0):
        return [(word() if callable(word) else word) for word, _ in self.code[start:]]


def fold(words):
    value = 0x6D2B79F5
    for word in words:
        value = (((value << 5) | (value >> 27)) ^ word) & MASK
    return value


def run_body(cpu, end):
    """Check constraints on actual executed operands, independently of templates."""
    hazards, producers, previous_store = Counter(), [], None
    while cpu.pc != end:
        word, pc = cpu.mem.load(cpu.pc), cpu.pc
        op, rd, f3 = word & 127, (word >> 7) & 31, (word >> 12) & 7
        rs1, rs2, f7 = (word >> 15) & 31, (word >> 20) & 31, word >> 25
        writes = op in (0x37, 0x17, 0x6F, 0x67, 0x03, 0x13, 0x33) or (op == 0x0B and f3 == 3)
        sources = []
        if op in (0x67, 0x03, 0x13, 0x23, 0x33, 0x63):
            sources.append(rs1)
        if op in (0x23, 0x33, 0x63):
            sources.append(rs2)
        if op == 0x0B:
            if f3 != 3 or f7 & 64:
                sources.append(rs1)
            if f3 == 6 or (f3 == 1 and f7 & 32) or (f3 == 3 and f7 & 64):
                sources.append(rs2)
            if (f3 == 1 and 32 <= f7 < 40) or (f3 == 3 and 64 <= f7 < 72):
                mantissas = [cpu.reg[rs1]]
                if f3 == 3:
                    mantissas.append(cpu.sm[f7 & 7])
                    product = mantissas[0] * mantissas[1]
                    norm = 30 if product >= 1 << 59 else 29
                    shift = signed(cpu.se[f7 & 7] + signed(cpu.reg[rs2]) - norm - 12)
                    assert 1 <= shift <= 62, (pc, shift)
                assert all(m == 0 or 1 << 29 <= m < 1 << 30 for m in mantissas), (pc, mantissas)
        for distance, (producer, kind) in enumerate(producers, 1):
            if producer and producer in sources:
                hazards[f"raw_distance_{distance}"] += 1
                if distance == 1 and kind in ("load", "mul", "div"):
                    hazards[kind + "_use"] += 1
                # A nearer write to this register hides any older write.
                sources = [source for source in sources if source != producer]
        if writes:
            assert rd != 31, "x31 is reserved as the memory base"
            if rd == 0:
                hazards["x0_writes"] += 1
        address = None
        if op in (0x03, 0x23):
            imm = signed(word >> 20, 12) if op == 0x03 else signed(((word >> 25) << 5) | rd, 12)
            address = (cpu.reg[rs1] + imm) & MASK
            size = 1 << (f3 & 3)
            assert DATA <= address and address + size <= DATA + 256, (pc, address)
            if op == 0x03 and previous_store == address:
                hazards["store_load_same_address"] += 1
        previous_store = address if op == 0x23 else None
        kind = "load" if op == 0x03 else "mul" if op == 0x33 and f7 == 1 and f3 < 4 else (
            "div" if op == 0x33 and f7 == 1 else "other")
        producers = [(rd if writes else 0, kind)] + producers[:2]
        cpu.step()
        assert pc < cpu.pc <= end, "random body must advance strictly forward"
    return dict(sorted(hazards.items()))


class RandomBody:
    def __init__(self, rng):
        self.rng, self.recent = rng, [1, 2, 3]

    def dst(self):
        rd = 0 if self.rng.random() < .10 else self.rng.randint(1, 30)
        self.recent = [rd] + self.recent[:2]   # x0 too: reading x0 right after an x0 write tests the rd != 0 forwarding guards
        return rd

    def src(self):
        return self.rng.choice(self.recent) if self.rng.random() < .8 else self.rng.randrange(31)

    def alu(self, p):
        a, b = self.src(), self.src()
        rd = self.dst()
        if self.rng.random() < .55:
            p.r(self.rng.choice(list(R_OPS)), rd, a, b)
        else:
            op = self.rng.choice(list(I_OPS)[:9])
            imm = self.rng.randrange(32) if op in ("slli", "srli", "srai") else self.rng.randint(-2048, 2047)
            p.i(op, rd, a, imm)

    def block(self, pc):
        rng, p = self.rng, Program(pc)
        kind = rng.choices(["chain", "memory", "mcycle", "branch", "jump", "upper", "custom"],
                           [25, 20, 12, 12, 6, 5, 20])[0]
        if kind == "chain":
            for _ in range(rng.randint(3, 7)):
                self.alu(p)
        elif kind == "memory":
            width = rng.choice((1, 2, 4))
            off = rng.randrange(256 // width) * width
            a, b = rng.sample(range(1, 31), 2)
            p.s({1: "sb", 2: "sh", 4: "sw"}[width], self.src(), 31, off)
            p.i(rng.choice({1: ("lb", "lbu"), 2: ("lh", "lhu"), 4: ("lw",)}[width]), a, 31, off)
            p.r(rng.choice(("add", "xor", "mul")), b, a, self.src())
            self.recent = [b, a] + self.recent[:1]
        elif kind == "mcycle":
            a, b, c = rng.sample(range(1, 31), 3)
            x, y = rng.choice([(0x80000000, MASK), (MASK, 0), (7, 0),
                               (rng.getrandbits(32), rng.getrandbits(32))])
            p.li(a, x)
            p.li(b, y)
            p.r(rng.choice(list(R_OPS)[10:]), c, a, b)
            p.r(rng.choice(("add", "xor", "mul", "div", "rem")), a, c, b)
            p.r("add", b, a, c)
            self.recent = [b, a, c]
        elif kind == "branch":
            a, b = rng.sample(range(1, 31), 2)
            x, y = rng.choice([(0, 0), (0, 1), (1, 0), (0x80000000, 1), (1, MASK)])
            p.li(a, x)
            p.li(b, y)
            skip = rng.randint(1, 3)
            op = rng.choice(list(BRANCH))
            p.emit(b_word(op, a, b, 4 * (skip + 1)), f"{op} x{a},x{b},+{4*(skip+1)}")
            for _ in range(skip):
                self.alu(p)
            self.alu(p)
        elif kind == "jump":
            rd, base = rng.sample(range(1, 31), 2)
            skip = rng.randint(1, 3)
            if rng.random() < .5:
                p.emit(j_word(rd, 4 * (skip + 1)), f"jal x{rd},+{4*(skip+1)}")
            else:
                p.u("auipc", base, 0)
                p.i("addi", base, base, 4 * (skip + 3) + 1)
                p.i("jalr", rd if rng.random() < .5 else base, base, 0)
            for _ in range(skip):
                self.alu(p)
            self.alu(p)
        elif kind == "upper":
            rd = rng.randint(1, 30)
            p.u(rng.choice(("lui", "auipc")), rd, rng.getrandbits(20))
            p.r("xor", rd, rd, self.src())
            self.recent = [rd] + self.recent[:2]
        else:
            # Atomic template: every read has a BSTORE at the exact current pointer,
            # for every active lane. SETPA/SETP exercise pointer truncation and wrap.
            a, b, c, out = rng.sample(range(1, 31), 4)
            k, address = rng.randint(1, 8), rng.choice((0, 255, rng.randrange(256)))
            p.i("addi", a, 0, k)
            p.custom("setk", a=a)
            p.i("addi", b, 0, address + 256)
            p.custom("setpa", a=b)
            for lane in range(k):
                if rng.random() < .3:
                    p.custom("setp", lane, a=b)
                p.li(c, rng.getrandbits(32))
                p.custom("bstore", lane, a=b, b=c)
            p.custom("vdotr", a=c)
            # A second MAC at pointer+1 tests both accumulation and 255->0 wrap.
            if rng.random() < .45:
                p.i("addi", b, b, 1)
                for lane in range(k):
                    p.custom("bstore", lane, a=b, b=c)
                p.custom("vdotr", a=self.src())
            lane = rng.randrange(k)
            if rng.random() < .75:
                sm = rng.choice((0, 1 << 29, (1 << 30) - 1, rng.randrange(1 << 29, 1 << 30)))
                rm = rng.choice((0, 1 << 29, (1 << 30) - 1, rng.randrange(1 << 29, 1 << 30)))
                se, shift = rng.randint(12, 50), rng.choice((1, 2, 12, 29, 30, 31, 62, rng.randint(1, 62)))
                norm = 30 if sm * rm >= 1 << 59 else 29
                re = shift + 12 + norm - se
                p.li(a, sm)
                p.i("addi", b, 0, se)
                p.custom("sets", lane, a=a, b=b)
                p.li(a, rm)
                p.i("addi", b, 0, re)
                p.custom("accq", lane, rd=out, a=a, b=b)
            else:
                p.custom("accrd", lane, rd=out)
            p.r("add", out, out, c)
            self.recent = [out, b, a]
        return p


@dataclass
class Artifact:
    program: Program
    data: bytes
    manifest: dict

    def cpu(self):
        memory = Memory()
        for i, word in enumerate(self.program.words()):
            memory.store(MAIN + 4 * i, word)
        memory.put(DATA, self.data)
        return ISS(MAIN, memory)


def generate(seed, n=2000):
    if not 1 <= n <= 14000:
        raise ValueError("N must be 1..14000 (64 KiB IROM budget, including checker)")
    rng, p = random.Random(seed), Program()
    initial = [rng.getrandbits(32) for _ in range(DATA_WORDS)]
    # .balign 4096 puts main here with the current crt0/lib/link.ld. Fail clearly
    # if a future link layout changes it, rather than trusting PC-dependent results.
    p.u("auipc", 5, 0)
    p.u("lui", 6, MAIN >> 12)
    p.branch("beq", 5, 6, "layout_ok")
    p.jump(0, "layout_fail")
    p.label("layout_ok")
    for rd in range(1, 32):
        p.li(rd, DATA if rd == 31 else rng.choice(EDGES + [rng.getrandbits(32)]))
    p.label("rig_body")
    body_start = len(p.code)
    builder = RandomBody(rng)
    while len(p.code) - body_start < n:
        block = builder.block(p.pc)
        if len(block.code) > n - (len(p.code) - body_start):
            block = Program(p.pc)
            builder.alu(block)
        p.code.extend(block.code)
    p.label("rig_end")
    # Snapshot *before* using scratch registers, including x0 and the fixed x31 base.
    for reg in range(32):
        p.s("sw", reg, 31, SNAP + 4 * reg)
    for lane in range(8):
        p.custom("accrd", lane, rd=1)
        p.s("sw", 1, 31, DRAIN + 4 * lane)
    observation_end = p.pc

    memory = Memory()
    for i, word in enumerate(p.words(4), 4):
        memory.store(MAIN + 4 * i, word)
    for i, word in enumerate(initial):
        memory.store(DATA + 4 * i, word)
    cpu = ISS(MAIN + 16, memory).run(stop_pc=p.labels["rig_body"])
    cpu.counts.clear()
    hazards = run_body(cpu, p.labels["rig_end"])
    expected, coverage = list(cpu.reg), dict(sorted(cpu.counts.items()))
    executed = sum(v for k, v in coverage.items() if "." not in k)
    cpu.run(stop_pc=observation_end)
    final_words = [memory.load(DATA + off) for off in range(0, SIG, 4)]
    signature = fold(final_words)

    strings = {}
    blob = bytearray().join(word.to_bytes(4, "little") for word in initial)
    blob.extend(bytes(SIG - len(blob)))
    blob.extend(signature.to_bytes(4, "little"))
    blob.extend(b"".join(word.to_bytes(4, "little") for word in expected))
    blob.extend(bytes(4))  # actual signature diagnostic

    def message(key, text):
        strings[key] = DATA + len(blob)
        blob.extend(text.encode("ascii") + b"\0")

    message("pass", f"RIG PASS seed={seed} n={n}\n")
    message("reg", f"RIG FAIL seed={seed} register=0x")
    message("sig", f"RIG FAIL seed={seed} signature")
    message("layout", "RIG FAIL link layout (main must be 0x00401000)\n")
    message("expected", " expected=0x")
    message("actual", " actual=0x")
    message("newline", "\n")
    message("hex", "0123456789abcdef")

    def puts(key):
        p.li(10, strings[key])
        p.jump(1, "puts")

    p.i("addi", 8, 31, 0)
    p.i("addi", 9, 31, SIG)
    p.li(10, 0x6D2B79F5)
    p.label("fold_loop")
    p.i("lw", 11, 8, 0)
    p.i("slli", 12, 10, 5)
    p.i("srli", 10, 10, 27)
    p.r("or", 10, 10, 12)
    p.r("xor", 10, 10, 11)
    p.i("addi", 8, 8, 4)
    p.branch("bltu", 8, 9, "fold_loop")
    p.s("sw", 10, 31, ACTUAL)
    p.li(25, 0xFFFF0000)
    p.i("addi", 8, 31, SNAP)
    p.i("addi", 9, 31, EXPECT)
    p.i("addi", 18, 0, 0)
    p.i("addi", 19, 0, 32)
    p.label("check_regs")
    p.i("lw", 20, 8, 0)
    p.i("lw", 21, 9, 0)
    p.branch("bne", 20, 21, "reg_fail")
    p.i("addi", 8, 8, 4)
    p.i("addi", 9, 9, 4)
    p.i("addi", 18, 18, 1)
    p.branch("bltu", 18, 19, "check_regs")
    p.i("lw", 20, 31, ACTUAL)
    p.i("lw", 21, 31, SIG)
    p.branch("bne", 20, 21, "sig_fail")
    puts("pass")
    p.i("addi", 10, 0, 0)
    p.jump(0, "exit")
    p.label("reg_fail")
    p.li(22, 0x52000000)
    p.r("or", 22, 22, 18)
    puts("reg")
    p.i("addi", 10, 18, 0)
    p.jump(1, "hex")
    p.jump(0, "details")
    p.label("sig_fail")
    p.li(22, 0x52000100)
    puts("sig")
    p.label("details")
    puts("expected")
    p.i("addi", 10, 21, 0)
    p.jump(1, "hex")
    puts("actual")
    p.i("addi", 10, 20, 0)
    p.jump(1, "hex")
    puts("newline")
    p.i("addi", 10, 22, 0)
    p.jump(0, "exit")
    p.label("layout_fail")
    p.li(25, 0xFFFF0000)
    puts("layout")
    p.li(10, 0x52000200)
    p.jump(0, "exit")
    p.label("puts")
    p.i("lbu", 5, 10, 0)
    p.branch("beq", 5, 0, "puts_ret")
    p.label("puts_wait")
    p.i("lw", 6, 25, 8)
    p.branch("beq", 6, 0, "puts_wait")
    p.s("sw", 5, 25, 12)
    p.i("addi", 10, 10, 1)
    p.jump(0, "puts")
    p.label("puts_ret")
    p.i("jalr", 0, 1, 0)
    p.label("hex")
    p.i("addi", 7, 10, 0)
    p.i("addi", 28, 0, 8)
    p.li(29, strings["hex"])
    p.label("hex_loop")
    p.i("srli", 5, 7, 28)
    p.r("add", 5, 29, 5)
    p.i("lbu", 5, 5, 0)
    p.label("hex_wait")
    p.i("lw", 6, 25, 8)
    p.branch("beq", 6, 0, "hex_wait")
    p.s("sw", 5, 25, 12)
    p.i("slli", 7, 7, 4)
    p.i("addi", 28, 28, -1)
    p.branch("bne", 28, 0, "hex_loop")
    p.i("jalr", 0, 1, 0)
    p.label("exit")
    p.s("sw", 10, 25, 128)
    p.label("exit_wait")
    p.i("lw", 6, 25, 8)
    p.branch("beq", 6, 0, "exit_wait")
    p.i("addi", 6, 0, 4)
    p.s("sw", 6, 25, 12)
    p.label("exit_halt")
    p.jump(0, "exit_halt")
    assert p.pc <= 0x00410000, "program exceeds 64 KiB IROM"
    manifest = {"seed": seed, "n": n, "executed_body_instructions": executed,
                "main": f"0x{MAIN:08x}", "body": f"0x{p.labels['rig_body']:08x}",
                "end": f"0x{p.labels['rig_end']:08x}", "signature": f"0x{signature:08x}",
                "registers": {f"x{i}": f"0x{word:08x}" for i, word in enumerate(expected)},
                "final_data_words": [f"0x{memory.load(DATA + i * 4):08x}" for i in range(DATA_WORDS)],
                "final_accumulators": [f"0x{word:08x}" for word in final_words[64:72]],
                "coverage": coverage, "hazards": hazards}
    return Artifact(p, bytes(blob), manifest)


def verify(artifact, reg=None, data=False, signature=False):
    cpu = artifact.cpu()
    if reg is not None or data:
        cpu.run(stop_pc=artifact.program.labels["rig_end"])
        if reg is not None:
            cpu.reg[reg] ^= 1
        if data:
            cpu.mem.store(DATA, cpu.mem.load(DATA) ^ 1)
    if signature:
        cpu.mem.store(DATA + SIG, cpu.mem.load(DATA + SIG) ^ 1)
    cpu.run()
    expected = 0x52000000 | reg if reg is not None else 0x52000100 if data or signature else 0
    assert cpu.exit_code == expected, (cpu.exit_code, expected, bytes(cpu.uart))
    assert (b"RIG PASS" if expected == 0 else b"RIG FAIL") in cpu.uart
    if reg is not None:
        assert f"register=0x{reg:08x}".encode() in cpu.uart
    return cpu


def self_test():
    coverage, hazards = Counter(), Counter()
    for seed in range(1, 6):
        artifact = generate(seed)
        verify(artifact)
        coverage.update(artifact.manifest["coverage"])
        hazards.update(artifact.manifest["hazards"])
        if seed == 1:
            for reg in (1, 17, 30):
                verify(artifact, reg=reg)
            verify(artifact, data=True)
            verify(artifact, signature=True)
            # Ascending comparison reports the first of multiple mismatches.
            cpu = artifact.cpu().run(stop_pc=artifact.program.labels["rig_end"])
            cpu.reg[3] ^= 1
            cpu.reg[27] ^= 1
            cpu.run()
            assert cpu.exit_code == 0x52000003 and b"register=0x00000003" in cpu.uart
            # A changed text placement must terminate with a layout diagnostic.
            cpu = artifact.cpu()
            shifted = MAIN + 0x10000
            for i, word in enumerate(artifact.program.words()):
                cpu.mem.store(shifted + 4 * i, word)
            cpu.pc = shifted
            cpu.run()
            assert cpu.exit_code == 0x52000200 and b"RIG FAIL link layout" in cpu.uart
    required = (set(R_OPS) | set(I_OPS) | set(BRANCH) |
                {"lui", "auipc", "jal", "jalr", "sb", "sh", "sw",
                 "setk", "setp", "setpa", "sets", "bstore", "vdotr", "accrd", "accq"})
    required |= {op + outcome for op in BRANCH for outcome in (".taken", ".not_taken")}
    assert required <= coverage.keys(), sorted(required - coverage.keys())
    assert all(hazards[key] for key in ("raw_distance_1", "raw_distance_2", "raw_distance_3",
                                       "load_use", "mul_use", "div_use", "x0_writes",
                                       "store_load_same_address")), hazards
    for n in (1, 2, 7, 37):
        verify(generate(19, n))
    print("RIG self-test PASS (seeds 1..5, all generated opcodes/branch outcomes, register/data/signature faults)")


def write(artifact):
    p, info = artifact.program, artifact.manifest
    names = {}
    for name, pc in p.labels.items():
        names.setdefault(pc, []).append(name)
    lines = [f"# Generated by tools/gen_rig.py {info['seed']} {info['n']}",
             "# Expected registers and final data are also in rig_expected.json.",
             ".option norvc", ".option norelax", ".text", ".balign 4096", ".globl main", "main:"]
    for i, word in enumerate(p.words()):
        for name in names.get(MAIN + 4 * i, []):
            lines.append(f"rig_{name}:" if name not in ("rig_body", "rig_end") else f"{name}:")
        lines.append(f"    .word 0x{word:08x}  # {MAIN + 4*i:08x}: {p.code[i][1]}")
    lines.extend([".data", ".balign 4", "rig_data:"])
    data_labels = {DRAIN: "rig_accumulators", SNAP: "rig_register_snapshot", SIG: "rig_expected_signature",
                   EXPECT: "rig_expected_registers", ACTUAL: "rig_actual_signature"}
    for off in range(0, len(artifact.data), 4):
        if off in data_labels:
            lines.append(data_labels[off] + ":")
        value = int.from_bytes(artifact.data[off:off + 4].ljust(4, b"\0"), "little")
        comment = f" # x{(off - EXPECT)//4}" if EXPECT <= off < ACTUAL else ""
        lines.append(f"    .word 0x{value:08x}{comment}")
    out = ROOT / "build/gen"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rig.S").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "rig_expected.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"wrote build/gen/rig.S + rig_expected.json: seed={info['seed']} N={info['n']} "
          f"executed={info['executed_body_instructions']} signature={info['signature']}; ISS checker PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed", type=int, nargs="?", default=1)
    parser.add_argument("n", type=int, nargs="?", default=2000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not 1 <= args.n <= 14000:
            parser.error("N must be 1..14000")
        artifact = generate(args.seed, args.n)
        verify(artifact)
        write(artifact)


if __name__ == "__main__":
    main()
