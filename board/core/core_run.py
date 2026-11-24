"""Run one firmware build on the SpecCore core in the PL (board app speccore_core). Plain Python 3, root.

  sudo python3 core_run.py DIR [--bp] [--timeout S]
      DIR holds AA_IROM.mem, AA_DMEM.mem (sc.py fw output) and model.img (the model image)

Steps: HPM0 width register to 128 bit (AFI_FS, restored at the end) -> model image into a CMA buffer
(/dev/dma_heap/reserved, physically contiguous) -> IROM + DMEM through the AXI-Lite window -> start ->
wait for done -> console, exit code, counters. Last line: JSON with the numbers.
"""
import fcntl
import json
import mmap
import os
import struct
import sys
import time

REGS, WIN = 0xA0000000, 0x200000
AFI_FS = 0xFD615000
IROM, CON, DMEM = 0x10000, 0x20000, 0x100000
BUF_LEN = 8 * 1024 * 1024 + 65536            # 8 MB window + prefetch overshoot


def words(path):
    return [int(l, 16) for l in open(path) if l.strip()]


def phys_of(va_addr):
    with open("/proc/self/pagemap", "rb") as f:
        f.seek((va_addr // 4096) * 8)
        e = struct.unpack("<Q", f.read(8))[0]
    if not e >> 63:
        sys.exit("page not present")
    return (e & ((1 << 55) - 1)) * 4096 + va_addr % 4096


def main():
    a = sys.argv[1:]
    d = a[0]
    bp = "--bp" in a
    tmo = float(a[a.index("--timeout") + 1]) if "--timeout" in a else 600
    memfd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    afi = mmap.mmap(memfd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=AFI_FS)
    afi_old = struct.unpack_from("<I", afi, 0)[0]
    struct.pack_into("<I", afi, 0, 0x200)
    try:
        run(memfd, d, bp, tmo)
    finally:
        struct.pack_into("<I", afi, 0, afi_old)
        print(f"AFI_FS restored to 0x{afi_old:x}")


def run(memfd, d, bp, tmo):
    m = mmap.mmap(memfd, WIN, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=REGS)
    rd = lambda o: struct.unpack_from("<I", m, o)[0]

    def wr(o, v):
        struct.pack_into("<I", m, o, v)

    ident = rd(0)
    print(f"ID 0x{ident:08x} (want 0x5c0e0003)")
    if ident != 0x5C0E0003:
        sys.exit("wrong or no bitstream")
    wr(0x04, 0)                                                  # stop (abort a previous run)
    # model image -> CMA buffer
    img = open(os.path.join(d, "model.img"), "rb").read()
    hfd = os.open("/dev/dma_heap/reserved", os.O_RDWR)
    al = bytearray(struct.pack("<QIIQ", BUF_LEN, 0, os.O_RDWR | os.O_CLOEXEC, 0))
    fcntl.ioctl(hfd, 0xC0184800, al)                             # DMA_HEAP_IOCTL_ALLOC
    bfd = struct.unpack_from("<QIIQ", al)[1]
    buf = mmap.mmap(bfd, BUF_LEN, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    fcntl.ioctl(bfd, 0x40086200, struct.pack("<Q", 0 | 2))       # DMA_BUF_IOCTL_SYNC start, write
    buf[:len(img)] = img
    buf[len(img):] = bytes(BUF_LEN - len(img))
    fcntl.ioctl(bfd, 0x40086200, struct.pack("<Q", 4 | 2))       # sync end, write
    import ctypes
    va = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    pa = phys_of(va)
    for o in range(0, BUF_LEN, 4096):
        if phys_of(va + o) != pa + o:
            sys.exit(f"buffer not contiguous at +{o}")
    print(f"image {len(img):,} B at phys 0x{pa:x}")
    # program
    t0 = time.time()
    iw, dw = words(os.path.join(d, "AA_IROM.mem")), words(os.path.join(d, "AA_DMEM.mem"))
    for i in range(16384):
        wr(IROM + 4 * i, iw[i] if i < len(iw) else 0x00000013)
    for i in range(262144):
        wr(DMEM + 4 * i, dw[i] if i < len(dw) else 0)
    bad = sum(rd(IROM + 4 * i) != iw[i] for i in range(0, len(iw), 7))
    bad += sum(rd(DMEM + 4 * i) != dw[i] for i in range(0, len(dw), 3))
    print(f"loaded IROM {len(iw)} words, DMEM {len(dw)} words ({time.time() - t0:.1f} s), readback mismatches {bad}")
    if bad:
        sys.exit("load failed")
    wr(0x10, pa & 0xFFFFFFFF)
    wr(0x14, pa >> 32)
    t0 = time.time()
    wr(0x04, 3 if bp else 1)
    while not rd(0x08) & 2:
        if time.time() - t0 > tmo:
            wr(0x04, 0)
            print("TIMEOUT (core stopped)")
            break
        time.sleep(0.01)
    secs = time.time() - t0
    n = rd(0x18)
    con = "".join(chr(rd(CON + 4 * i) & 0xFF) for i in range(n))
    print(con, end="" if con.endswith("\n") else "\n")
    q = lambda o: rd(o) | rd(o + 4) << 32
    r = dict(exit=struct.unpack("<i", struct.pack("<I", rd(0x0C)))[0], done=bool(rd(0x08) & 2),
             core_cycles=q(0x20), wall_cycles=q(0x28), stream_wait=q(0x30), load_wait=q(0x38),
             load_misses=rd(0x40), stream_restarts=rd(0x44), host_seconds=round(secs, 3), bp=bp)
    print(f"EXIT {r['exit']} | core cycles {r['core_cycles']:,} | wall cycles {r['wall_cycles']:,} "
          f"(stream wait {r['stream_wait']:,}, load wait {r['load_wait']:,})")
    print("RESULT " + json.dumps(r))
    buf.close()
    os.close(bfd)
    os.close(hfd)


if __name__ == "__main__":
    main()
