# Random instruction test

From the repository root in PowerShell, generate and check a test without launching
any toolchain or simulator:

```powershell
python -B tools/iss.py --self-test
python -B tools/gen_rig.py --self-test
python -B tools/gen_rig.py 1 2000
```

The generator writes `build/gen/rig.S` and `build/gen/rig_expected.json`. It executes
both the random body and the complete checking/reporting code in the Python ISS
before writing either file. There are no Python package dependencies.

For **seeds 1 through 5**, Python-only verification is:

```powershell
foreach ($seed in 1..5) {
    python -B tools/gen_rig.py $seed 2000
    if ($LASTEXITCODE) { throw "generation failed: seed $seed" }
}
```

When simulation is authorised, the exact build/run commands for those same seeds
are below. They use the existing runner unchanged, including its memory guard.
Run serially; the next seed overwrites the small generated test/build outputs.
These simulation commands were **not run** during implementation.

```powershell
foreach ($seed in 1..5) {
    python -B tools/gen_rig.py $seed 2000
    if ($LASTEXITCODE) { throw "generation failed: seed $seed" }
    python -B tools/sc.py fw rig build/gen/rig.S
    if ($LASTEXITCODE) { throw "build failed: seed $seed" }
    python -B tools/iss.py --elf build/rig/rig.elf
    if ($LASTEXITCODE) { throw "linked ISS check failed: seed $seed" }
    python -B tools/sc.py sim rig
    if ($LASTEXITCODE) { throw "simulation tool failed: seed $seed" }
    python -B tools/sc.py sim rig --bp
    if ($LASTEXITCODE) { throw "simulation tool failed: seed $seed" }
    # Inspect both runs before continuing: RIG PASS, EXIT code 0, SVA failures: 0.
    # sc.py sim does not translate firmware failure/timeout into a failing shell status.
    foreach ($log in @("build/rig/sim.txt", "build/rig/sim_bp.txt")) {
        $result = Get-Content -Raw -LiteralPath $log
        if ($result -notmatch 'RIG PASS' -or $result -notmatch 'EXIT code 0 ' -or
            $result -notmatch 'SVA failures: 0' -or $result -match 'TIMEOUT') {
            throw "RIG failed: seed $seed ($log)"
        }
    }
}
```

`sc.py fw` invokes the existing WSL RISC-V gcc/objcopy/objdump flow. The generated
body uses `.word` with decoded instruction comments to keep the ISS input identical
to the firmware. For a disassembly that decodes these words, use objdump `-D`:

```powershell
wsl.exe -d Ubuntu-24.04 -- riscv64-unknown-elf-objdump -D -M no-aliases,numeric <local path>
```

## Scope and diagnostics

`SEED` is an integer; `N` defaults to 2000 and is limited to 1..14000 so the test
fits the current 64 KiB IROM. N counts emitted random-body instructions, including
operand setup and instructions skipped by branches. Register initialisation,
snapshot, custom accumulator drain, signature, and reporting are additional.
Small N cannot cover every instruction. The self-test checks aggregate coverage
for seeds 1..5 at N=2000 and exercises small N separately.

The body mixes all RV32I integer ALU/immediate operations, LUI/AUIPC, all six
branches, JAL/JALR (including bit-zero clearing and rd=rs1), aligned byte/half/word
loads/stores, and all eight M operations. Edge-biased inputs include signed overflow
and division/remainder by zero. Dependent sources favour the last three producers;
templates add store/load aliases, immediate load-use, multiply/divide-use, and x0
writes. All body control transfers are forward and bounded. x1..x30 are available
as random destinations, including sp/gp/ra; x31 holds the fixed data base. All 31
nonzero registers and x0 are checked. The generator validates actual executed
custom scales and memory bounds and records observed hazard/branch counts in JSON.

Custom-0 operations are SETK, SETP, SETPA, SETS, BSTORE, VDOTR, ACCRD and ACCQ.
Lane IDs are 0..7. Each VDOTR is preceded by BSTORE to every active lane's actual
read address, including a second MAC across pointer 255->0. Mantissas are zero or
in [2^29, 2^30), with ACCQ shifts 1..62, both normalisation paths, and read/clear
results immediately consumed. SETS/ACCQ setup exercises forwarding into multi-cycle
ACCQ. Hidden accumulators are drained into the signature after the random body.

Before the checker uses scratch registers, it stores x0..x31. It folds the 256-byte
data region, eight final accumulators and register snapshot with rotate-left-5/XOR.
It compares registers in ascending order, then compares the signature. UART reports
`RIG PASS seed=... n=...`, or the first mismatching register index (hex), expected
value and actual value. A signature-only failure prints expected/actual signatures.
The embedded expected-register table and JSON provide per-register diagnostics;
JSON also includes final data words and pre-drain accumulator values.

Exit codes follow the existing protocol: write SEVENSEG at 0xFFFF0080, then UART
0x04 at 0xFFFF000C, polling ready at 0xFFFF0008. Success is 0; register failure is
0x52000000 | register index; signature failure is 0x52000100; layout failure is
0x52000200. Checker/UART loops are bounded by counters or UART readiness; the
random body has no backward transfers. The ISS instruction budget catches hangs.

This is a functional final-state test. It does not model cycles, pipeline stalls,
predictor internals, interrupts, privileged/CSR/system operations, misaligned
accesses, exceptions, atomics, compressed instructions, self-modifying code, weight
streams, or exhaustive opcode/state coverage. A final signature can miss cancelling
errors or collisions; it is not a per-retired-instruction lockstep checker.

## Reference choices and validation limits

The oracle decodes raw RV32 instructions independently of the generator's encoder.
Its sparse flat memory is little endian; unwritten ordinary memory bytes read as
zero. Custom bank RAM is explicitly unknown until BSTORE and reading it raises an
error. Custom arithmetic follows `firmware/include/sc_isa.h`: signed int8 dot4,
32-bit accumulator wrap as implemented in RTL, pointer truncation modulo 256,
read-and-clear, mantissa normalisation and arithmetic round-half-up (including
negative ties). Legal mantissas keep ACCQ products within signed 64-bit range.
The C model's signed accumulator `+=` is formally undefined on overflow; the ISS
explicitly follows RTL two's-complement wrap. This is the only arithmetic reference
ambiguity encountered, and does not require changing the legal scale constraints.

`docs/core_facts.md` contains historical memory sizes and a superseded missing-ISA
list; its update records the RV32IM additions. This test follows current
`firmware/link.ld`/`sc.py`: IROM 0x00400000/64 KiB, DMEM 0x10000000/1 MiB.
The generated `.text` input is aligned to 4096 bytes, placing main at 0x00401000
after the current crt0/lib. A runtime PC guard reports layout failure if that
assumption changes. The test's `.data` starts at DMEM origin with the existing
minimal firmware link inputs. Run `iss.py --elf` after building to check the actual
linked layout and boot/checker path before RTL simulation.

Python validation covers hand-computed ISS cases, five random seeds, every generated
opcode and both outcomes of all branches across those seeds, legal executed custom
operands, and injected register/data/signature faults (including two-register first
mismatch reporting) and a relocated-main layout failure. The ISS also matched all
504 ALU/M cases from the existing directed generator; the written assembly was
parsed back and executed successfully. The generator's complete raw program is
executed in the ISS.
WSL returned E_ACCESSDENIED in the implementation session, so gcc/objdump and linked
ELF validation remain unverified. No Vivado/xsim, sc.py simulation/run, training,
or commit was performed.
