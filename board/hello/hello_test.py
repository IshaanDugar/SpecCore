"""Board test for the hello app (runs on the KV260, plain Python 3, no packages).
Finds the two UIO devices by node name, reads the constant ID through AXI GPIO, and writes / reads
back the 8 KB BRAM with three patterns. Prints HELLO TEST PASS / FAIL."""
import glob
import mmap
import os
import struct
import time


def uio(name):
    for d in glob.glob("/sys/class/uio/uio*"):
        if open(d + "/name").read().strip() == name:
            size = int(open(d + "/maps/map0/size").read(), 16)
            fd = os.open("/dev/" + os.path.basename(d), os.O_RDWR | os.O_SYNC)
            return mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
    raise SystemExit(f"no UIO device named {name} (is the app loaded? run speccore-devperm?)")


def rd(m, off):
    return struct.unpack_from("<I", m, off)[0]


def wr(m, off, v):
    struct.pack_into("<I", m, off, v)


ident, bram = uio("speccore_id"), uio("speccore_bram")
fails = 0
v = rd(ident, 0)                        # AXI GPIO channel 1 data register
print(f"ID register: {v:#010x} (want 0x5c0e0001)")
fails += v != 0x5C0E0001
words = 8192 // 4
for name, f in (("address", lambda i: i * 4), ("inverted", lambda i: ~(i * 4) & 0xFFFFFFFF),
                ("pattern", lambda i: (0xA5A5A5A5 ^ (i * 0x9E3779B1)) & 0xFFFFFFFF)):
    t0 = time.perf_counter()
    for i in range(words):
        wr(bram, i * 4, f(i))
    bad = sum(rd(bram, i * 4) != f(i) for i in range(words))
    dt = time.perf_counter() - t0
    print(f"BRAM {name:8s}: {words} words written + read back, {bad} mismatches ({dt * 1e3:.1f} ms, Python)")
    fails += bad
print("HELLO TEST PASS" if not fails else f"HELLO TEST FAIL ({fails})")
