import mmap, os, struct, sys
A = 0xFD615000
fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
m = mmap.mmap(fd, 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=A)
old = struct.unpack_from("<I", m, 0)[0]
if len(sys.argv) > 1:
    struct.pack_into("<I", m, 0, int(sys.argv[1], 0))
print(f"AFI_FS 0x{old:08x} -> 0x{struct.unpack_from('<I', m, 0)[0]:08x}")
