"""Small RV32IM/custom-0 functional oracle; no pipeline or weight stream.

python tools/iss.py --self-test
python tools/iss.py --elf build/rig/rig.elf  # includes UART/exit checking
"""
import argparse
import struct
from collections import Counter
from pathlib import Path

MASK = 0xFFFFFFFF


def signed(value, bits=32):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def divrem(a, b, unsigned=False):
    if unsigned:
        return (MASK, a) if b == 0 else (a // b, a % b)
    a, b = signed(a), signed(b)
    if b == 0:
        return MASK, a & MASK
    q = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        q = -q
    return q & MASK, (a - q * b) & MASK


class Memory:
    """Sparse, flat, little-endian 32-bit address space; unwritten bytes are zero."""
    def __init__(self):
        self.bytes = {}

    def load(self, address, size=4):
        if address % size:
            raise ValueError(f"misaligned load: {address:#x}/{size}")
        return sum(self.bytes.get((address + i) & MASK, 0) << (8 * i) for i in range(size))

    def store(self, address, value, size=4):
        if address % size:
            raise ValueError(f"misaligned store: {address:#x}/{size}")
        for i in range(size):
            self.bytes[(address + i) & MASK] = (value >> (8 * i)) & 255

    def put(self, address, data):
        for i, byte in enumerate(data):
            self.bytes[(address + i) & MASK] = byte


class ISS:
    def __init__(self, pc=0, memory=None):
        self.pc = pc
        self.mem = memory if memory is not None else Memory()
        self.reg = [0] * 32
        self.k = 1
        # None catches accidental reads of uninitialised RTL RAM.
        self.bank = [[None] * 256 for _ in range(8)]
        self.ptr, self.acc, self.sm, self.se = ([0] * 8 for _ in range(4))
        self.counts = Counter()
        self.steps = 0
        self.uart = bytearray()
        self.exit_code = None
        self.halted = False

    def custom(self, f3, f7, a, b):
        lane = f7 & 7
        if f3 == 0 and f7 == 0:
            self.k = min(8, max(1, a))
            return "setk", None
        if f3 == 1:
            if 32 <= f7 < 40:
                self.sm[lane], self.se[lane] = a, signed(b)
                return "sets", None
            if f7 == 64:
                self.ptr[:] = [a % 256] * 8
                return "setpa", None
            if f7 < 8:
                self.ptr[lane] = a % 256
                return "setp", None
        if f3 == 2 and f7 == 0:
            for lane in range(self.k):
                word = self.bank[lane][self.ptr[lane]]
                if word is None:
                    raise ValueError(f"uninitialised bank[{lane}][{self.ptr[lane]}]")
                dot = sum(signed(a >> i, 8) * signed(word >> i, 8) for i in (0, 8, 16, 24))
                self.acc[lane] = (self.acc[lane] + dot) & MASK
                self.ptr[lane] = (self.ptr[lane] + 1) % 256
            return "vdotr", None
        if f3 == 3 and (f7 < 8 or 64 <= f7 < 72):
            value, self.acc[lane] = self.acc[lane], 0
            if f7 < 8:
                return "accrd", value
            p = self.sm[lane] * a
            norm = 30 if p >= 1 << 59 else 29
            m = (p >> norm) & MASK
            e = signed(self.se[lane] + signed(b) - norm)
            shift = signed(e - 12)
            result = 0 if not m or not 1 <= shift <= 62 else (
                signed(value) * m + (1 << (shift - 1))) >> shift
            return "accq", result & MASK
        if f3 == 6 and f7 < 8:
            self.bank[lane][a % 256] = b
            return "bstore", None
        raise ValueError(f"unsupported custom op f3={f3} f7={f7} (no weight stream)")

    def step(self):
        pc = self.pc
        if pc & 3:
            raise ValueError(f"misaligned PC {pc:#x}")
        insn = self.mem.load(pc)
        op, rd, f3 = insn & 127, (insn >> 7) & 31, (insn >> 12) & 7
        rs1, rs2, f7 = (insn >> 15) & 31, (insn >> 20) & 31, insn >> 25
        a, b = self.reg[rs1], self.reg[rs2]
        imm = signed(insn >> 20, 12)
        result, name, next_pc = None, None, (pc + 4) & MASK
        if op == 0x37:
            name, result = "lui", insn & 0xFFFFF000
        elif op == 0x17:
            name, result = "auipc", pc + (insn & 0xFFFFF000)
        elif op == 0x6F:
            j = ((insn >> 31) << 20 | ((insn >> 12) & 255) << 12 |
                 ((insn >> 20) & 1) << 11 | ((insn >> 21) & 1023) << 1)
            name, result, next_pc = "jal", pc + 4, (pc + signed(j, 21)) & MASK
        elif op == 0x67 and f3 == 0:
            name, result, next_pc = "jalr", pc + 4, (a + imm) & (MASK ^ 1)
        elif op == 0x63:
            offset = ((insn >> 31) << 12 | ((insn >> 7) & 1) << 11 |
                      ((insn >> 25) & 63) << 5 | ((insn >> 8) & 15) << 1)
            conditions = {0: ("beq", a == b), 1: ("bne", a != b),
                          4: ("blt", signed(a) < signed(b)), 5: ("bge", signed(a) >= signed(b)),
                          6: ("bltu", a < b), 7: ("bgeu", a >= b)}
            if f3 in conditions:
                name, taken = conditions[f3]
                self.counts[name + (".taken" if taken else ".not_taken")] += 1
                if taken:
                    next_pc = (pc + signed(offset, 13)) & MASK
        elif op == 0x03 and f3 in (0, 1, 2, 4, 5):
            name, size = {0: ("lb", 1), 1: ("lh", 2), 2: ("lw", 4),
                          4: ("lbu", 1), 5: ("lhu", 2)}[f3]
            address = (a + imm) & MASK
            value = 1 if address == 0xFFFF0008 else self.mem.load(address, size)
            result = signed(value, 8 * size) if f3 < 4 else value
        elif op == 0x23 and f3 in (0, 1, 2):
            offset = signed(((insn >> 25) << 5) | ((insn >> 7) & 31), 12)
            address, size = (a + offset) & MASK, 1 << f3
            name = ("sb", "sh", "sw")[f3]
            if address == 0xFFFF000C:
                if b & 255 == 4:
                    self.halted = True
                else:
                    self.uart.append(b & 255)
            elif address == 0xFFFF0080:
                self.exit_code = b
            else:
                self.mem.store(address, b, size)
        elif op == 0x13:
            if f3 in (0, 2, 3, 4, 6, 7):
                name, result = {0: ("addi", a + imm), 2: ("slti", int(signed(a) < imm)),
                                3: ("sltiu", int(a < (imm & MASK))), 4: ("xori", a ^ (imm & MASK)),
                                6: ("ori", a | (imm & MASK)), 7: ("andi", a & (imm & MASK))}[f3]
            elif f3 == 1 and f7 == 0:
                name, result = "slli", a << rs2
            elif f3 == 5 and f7 in (0, 32):
                name, result = ("srai", signed(a) >> rs2) if f7 else ("srli", a >> rs2)
        elif op == 0x33:
            if f7 == 1:
                q, r = divrem(a, b)
                qu, ru = divrem(a, b, True)
                name, result = [("mul", a * b), ("mulh", (signed(a) * signed(b)) >> 32),
                                ("mulhsu", (signed(a) * b) >> 32), ("mulhu", (a * b) >> 32),
                                ("div", q), ("divu", qu), ("rem", r), ("remu", ru)][f3]
            elif f7 == 0:
                name, result = [("add", a + b), ("sll", a << (b & 31)),
                                ("slt", int(signed(a) < signed(b))), ("sltu", int(a < b)),
                                ("xor", a ^ b), ("srl", a >> (b & 31)),
                                ("or", a | b), ("and", a & b)][f3]
            elif f7 == 32 and f3 in (0, 5):
                name, result = ("sub", a - b) if f3 == 0 else ("sra", signed(a) >> (b & 31))
        elif op == 0x0B:
            name, result = self.custom(f3, f7, a, b)
        elif op == 0x0F and f3 in (0, 1):
            name = "fence" if f3 == 0 else "fence.i"
        if name is None:
            raise ValueError(f"unsupported instruction {insn:08x} at {pc:08x}")
        if result is not None and rd:
            self.reg[rd] = result & MASK
        self.reg[0] = 0
        self.pc = next_pc
        self.steps += 1
        self.counts[name] += 1
        return name

    def run(self, stop_pc=None, max_steps=1_000_000):
        for _ in range(max_steps):
            if self.halted or self.pc == stop_pc:
                return self
            self.step()
        raise RuntimeError(f"ISS instruction budget exceeded at {self.pc:#x}")

    @classmethod
    def from_elf(cls, path):
        """Load little-endian RV32 ELF PT_LOAD segments (no third-party packages)."""
        data = Path(path).read_bytes()
        if data[:7] != b"\x7fELF\x01\x01\x01" or struct.unpack_from("<H", data, 18)[0] != 243:
            raise ValueError("expected little-endian RV32 ELF")
        entry, phoff = struct.unpack_from("<II", data, 24)
        phsize, phnum = struct.unpack_from("<HH", data, 42)
        iss = cls(entry)
        for i in range(phnum):
            kind, offset, address, _, size, memsize = struct.unpack_from("<6I", data, phoff + i * phsize)
            if kind == 1:
                if size > memsize or offset + size > len(data):
                    raise ValueError("invalid ELF load segment")
                iss.mem.put(address, data[offset:offset + size])
        return iss


def self_test():
    # Literal machine words; expected values below are hand computed.
    cases = [
        (0x002081B3, 0xFFFFFFFF, 2, 1),       # add x3,x1,x2
        (0x402081B3, 0, 1, MASK),            # sub
        (0x002091B3, 1, 33, 2),             # sll masks shift
        (0x4020D1B3, 0x80000000, 31, MASK),  # sra
        (0x022091B3, MASK, 2, MASK),         # mulh(-1,2)
        (0x0220A1B3, MASK, MASK, MASK),      # mulhsu(-1,UINT_MAX)
        (0x0220B1B3, MASK, MASK, 0xFFFFFFFE),
        (0x0220C1B3, 0xFFFFFFF9, 3, 0xFFFFFFFE),  # -7/3 = -2
        (0x0220E1B3, 0xFFFFFFF9, 3, MASK),       # -7%3 = -1
        (0x0220C1B3, 0x80000000, MASK, 0x80000000),
        (0x0220E1B3, 0x80000000, MASK, 0),
        (0x0220C1B3, 5, 0, MASK), (0x0220D1B3, 5, 0, MASK),
        (0x0220E1B3, 0x80000000, 0, 0x80000000),
        (0x0220F1B3, 5, 0, 5),
        (0xFFF0B193, MASK, 0, 0),            # sltiu x3,x1,-1
        (0xFFF08193, 0, 0, MASK),            # addi x3,x1,-1
        (0xFFF0A193, 0x80000000, 0, 1),      # slti x3,x1,-1
        (0xFFF0C193, 0x12345678, 0, 0xEDCBA987),  # xori -1
        (0x8000E193, 1, 0, 0xFFFFF801),      # ori -2048
        (0xFFF0F193, 0x12345678, 0, 0x12345678),  # andi -1
        (0x01F09193, 3, 0, 0x80000000),     # slli 31
        (0x01F0D193, 0x80000000, 0, 1),      # srli 31
        (0x41F0D193, 0x80000000, 0, MASK),   # srai 31
        (0x00000197, 0, 0, 0x100),          # auipc at PC 0x100
    ]
    for word, a, b, expected in cases:
        cpu = ISS(0x100)
        cpu.mem.store(cpu.pc, word)
        cpu.reg[1:3] = [a, b]
        cpu.step()
        assert cpu.reg[3] == expected, (hex(word), cpu.reg[3], expected)
    # Literal +8 branch encodings, with hand-selected true/false operands.
    for word, yes, no in (
            (0x00208463, (5, 5), (5, 6)),       # beq
            (0x00209463, (5, 6), (5, 5)),       # bne
            (0x0020C463, (MASK, 1), (1, MASK)), # blt
            (0x0020D463, (1, MASK), (MASK, 1)), # bge
            (0x0020E463, (1, MASK), (MASK, 1)), # bltu
            (0x0020F463, (MASK, 1), (1, MASK))):# bgeu
        for operands, target in ((yes, 0x108), (no, 0x104)):
            cpu = ISS(0x100)
            cpu.mem.store(cpu.pc, word)
            cpu.reg[1:3] = operands
            cpu.step()
            assert cpu.pc == target, (hex(word), operands, cpu.pc)
    cpu = ISS()
    cpu.reg[1:3] = [0x100, 0x80FF7F01]
    for word in (0x0020A023, 0x00308183):  # sw x2,0(x1); lb x3,3(x1)
        cpu.mem.store(cpu.pc, word)
        cpu.step()
    assert cpu.reg[3] == 0xFFFFFF80
    cpu.mem.store(cpu.pc, 0x00209183)      # lh x3,2(x1)
    cpu.step()
    assert cpu.reg[3] == 0xFFFF80FF
    cpu.mem.store(cpu.pc, 0x0020D183)      # lhu x3,2(x1)
    cpu.step()
    assert cpu.reg[3] == 0x80FF
    cpu.mem.store(cpu.pc, 0x0010C183)      # lbu x3,1(x1)
    cpu.step()
    assert cpu.reg[3] == 127
    cpu.mem.store(cpu.pc, 0x002080A3)      # sb x2,1(x1)
    cpu.step()
    cpu.mem.store(cpu.pc, 0x00209123)      # sh x2,2(x1)
    cpu.step()
    assert cpu.mem.load(0x100) == 0x7F010101
    cpu = ISS(0x100)
    cpu.mem.store(0x100, 0x008000EF)      # jal x1,+8
    cpu.step()
    assert (cpu.pc, cpu.reg[1]) == (0x108, 0x104)
    cpu.mem.store(0x108, 0x000080E7)      # jalr x1,x1,0 (old rs1)
    cpu.reg[1] = 0x201
    cpu.step()
    assert (cpu.pc, cpu.reg[1]) == (0x200, 0x10C)
    cpu.mem.store(0x200, 0x00100013)      # addi x0,x0,1
    cpu.step()
    assert cpu.reg[0] == 0
    # bytes [-1,2,-128,127] dot [1,-2,-1,1] = 250
    cpu.custom(6, 0, 255, 0x01FFFE01)
    cpu.custom(1, 0, 255, 0)
    cpu.custom(2, 0, 0x7F8002FF, 0)
    assert (cpu.acc[0], cpu.ptr[0]) == (250, 0)
    assert cpu.custom(3, 0, 0, 0)[1] == 250 and cpu.acc[0] == 0
    # low normalisation: m=2^29, e=42, shift=30; +/-3 round half up.
    cpu.custom(1, 32, 1 << 29, 35)
    for value, expected in ((3, 2), (-3, MASK)):
        cpu.acc[0] = value & MASK
        assert cpu.custom(3, 64, 1 << 29, 36)[1] == expected
    # low normalisation, unequal scales: 8*3/8 = 3.
    cpu.acc[0] = 8
    assert cpu.custom(3, 64, 3 << 28, 37)[1] == 3
    # high normalisation: (3*2^28)^2 >> 30 = 9*2^26; shift=30.
    cpu.custom(1, 32, 3 << 28, 35)
    cpu.acc[0] = 8
    assert cpu.custom(3, 64, 3 << 28, 37)[1] == 5
    cpu.acc[0] = MASK
    assert cpu.custom(3, 64, 0, 37)[1] == 0 and cpu.acc[0] == 0
    cpu.custom(0, 0, 0, 0)
    assert cpu.k == 1
    cpu.custom(0, 0, MASK, 0)
    assert cpu.k == 8
    cpu.custom(1, 64, 0x123, 0)
    assert cpu.ptr == [0x23] * 8
    cpu.acc[0] = 0x7FFFFFFF
    cpu.k = 1
    cpu.custom(6, 0, 0x23, 1)
    cpu.custom(2, 0, 1, 0)
    assert cpu.acc[0] == 0x80000000
    print("ISS self-test PASS (RV32IM edges, subwords, jumps, custom dot/scale/clear/wrap)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--elf", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.elf:
        cpu = ISS.from_elf(args.elf).run()
        print(cpu.uart.decode("ascii"), end="")
        print(f"ISS EXIT {cpu.exit_code} after {cpu.steps} instructions")
        if cpu.exit_code != 0:
            raise SystemExit(1)
    if not args.self_test and not args.elf:
        parser.error("select --self-test or --elf")


if __name__ == "__main__":
    main()
