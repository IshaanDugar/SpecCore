"""Run a command with a HARD memory cap (Windows Job Object). Used by sc.py for every Vivado call.

Why: on 14/15 Nov an xsim kernel (xsimk.exe) committed ~103 GB (RAM + pagefile) and froze the
laptop with 99% disk paging. Polling the working set was not enough: commit can grow while the
working set stays small, and a poller reacts late. A Job Object makes Windows itself refuse any
allocation past the cap, for the whole process tree, including children that outlive the parent.

    rc, why, peak_mb = run_limited(cmd, cwd=..., stdout=f, mem_mb=2048, secs=600)

- The process starts SUSPENDED, joins the job, then resumes: nothing runs uncapped, not even briefly.
- JOB_MEMORY caps the total commit charge of every process in the job (cmd.exe, xsim, xsimk ...).
- KILL_ON_JOB_CLOSE: if this Python process dies (crash, Ctrl-C), Windows kills the whole tree.
- DIE_ON_UNHANDLED_EXCEPTION: an out-of-memory crash exits instead of hanging on an error dialog.
- A poller adds the wall-clock limit and stops the job once it sits at its cap.
- preflight(): refuse to start when the system has less than `need_mb` of commit headroom.

python tools/memguard.py --selftest   (uses small Python children, no simulator)
"""
import ctypes as C
import subprocess
import sys
import time
from ctypes import wintypes as W

_k32 = C.WinDLL("kernel32", use_last_error=True)
_ntdll = C.WinDLL("ntdll")

CREATE_SUSPENDED = 0x4
JobObjectExtendedLimitInformation = 9
LIMIT_PROCESS_MEMORY, LIMIT_JOB_MEMORY = 0x100, 0x200
LIMIT_DIE_ON_UNHANDLED_EXCEPTION, LIMIT_KILL_ON_JOB_CLOSE = 0x400, 0x2000


class _IO(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]


class _BASIC(C.Structure):
    _fields_ = [("PerProcessUserTimeLimit", C.c_longlong), ("PerJobUserTimeLimit", C.c_longlong),
                ("LimitFlags", W.DWORD), ("MinimumWorkingSetSize", C.c_size_t),
                ("MaximumWorkingSetSize", C.c_size_t), ("ActiveProcessLimit", W.DWORD),
                ("Affinity", C.c_size_t), ("PriorityClass", W.DWORD), ("SchedulingClass", W.DWORD)]


class _EXT(C.Structure):
    _fields_ = [("Basic", _BASIC), ("Io", _IO), ("ProcessMemoryLimit", C.c_size_t),
                ("JobMemoryLimit", C.c_size_t), ("PeakProcessMemoryUsed", C.c_size_t),
                ("PeakJobMemoryUsed", C.c_size_t)]


class _MEMSTAT(C.Structure):
    _fields_ = [("dwLength", W.DWORD), ("dwMemoryLoad", W.DWORD), ("ullTotalPhys", C.c_ulonglong),
                ("ullAvailPhys", C.c_ulonglong), ("ullTotalPageFile", C.c_ulonglong),
                ("ullAvailPageFile", C.c_ulonglong), ("ullTotalVirtual", C.c_ulonglong),
                ("ullAvailVirtual", C.c_ulonglong), ("ullAvailExtendedVirtual", C.c_ulonglong)]


_k32.CreateJobObjectW.restype = W.HANDLE
_k32.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
_k32.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
_k32.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]
_k32.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
_k32.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
_k32.CloseHandle.argtypes = [W.HANDLE]
_k32.GlobalMemoryStatusEx.argtypes = [C.POINTER(_MEMSTAT)]
_ntdll.NtResumeProcess.argtypes = [W.HANDLE]


def commit_free_mb():
    """System commit headroom (RAM + pagefile not yet promised to anyone), in MB."""
    m = _MEMSTAT(); m.dwLength = C.sizeof(_MEMSTAT)
    if not _k32.GlobalMemoryStatusEx(C.byref(m)):
        raise OSError(C.get_last_error(), "GlobalMemoryStatusEx")
    return m.ullAvailPageFile >> 20


def preflight(need_mb):
    free = commit_free_mb()
    if free < need_mb:
        sys.exit(f"MEMGUARD: only {free} MB commit headroom (< {need_mb} MB): close something first, not starting")
    return free


def _job(mem_mb):
    job = _k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(C.get_last_error(), "CreateJobObject")
    info = _EXT()
    info.Basic.LimitFlags = (LIMIT_JOB_MEMORY | LIMIT_PROCESS_MEMORY |
                             LIMIT_DIE_ON_UNHANDLED_EXCEPTION | LIMIT_KILL_ON_JOB_CLOSE)
    info.JobMemoryLimit = info.ProcessMemoryLimit = mem_mb << 20
    if not _k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, C.byref(info), C.sizeof(info)):
        raise OSError(C.get_last_error(), "SetInformationJobObject")
    return job


def _peak_mb(job):
    info = _EXT()
    _k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, C.byref(info), C.sizeof(info), None)
    return info.PeakJobMemoryUsed >> 20


def run_limited(cmd, cwd=None, stdout=None, mem_mb=2048, secs=None):
    """Returns (rc, why, peak_mb). why is None, or the reason the job was stopped (rc 99)."""
    job = _job(mem_mb)
    try:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=stdout, stderr=subprocess.STDOUT,
                             creationflags=CREATE_SUSPENDED)
        if not _k32.AssignProcessToJobObject(job, int(p._handle)):
            err = C.get_last_error()
            p.kill()
            raise OSError(err, "AssignProcessToJobObject")
        _ntdll.NtResumeProcess(int(p._handle))
        t0, why = time.time(), None
        while p.poll() is None:
            peak = _peak_mb(job)
            if peak >= mem_mb - 16:
                why = f"commit reached the {mem_mb} MB cap"
            elif secs and time.time() - t0 > secs:
                why = f"time > {secs:.0f} s"
            if why:
                _k32.TerminateJobObject(job, 99)
                p.wait()
                break
            time.sleep(0.2)
        peak = _peak_mb(job)
        if why is None and peak >= mem_mb - 16:
            why = f"commit reached the {mem_mb} MB cap (process exited)"
        return (99 if why else p.returncode), why, peak
    finally:
        _k32.TerminateJobObject(job, 99)     # nothing in the job may outlive this call
        _k32.CloseHandle(job)


def _selftest():
    py = sys.executable
    alloc = lambda mb: [py, "-c", f"b = bytearray({mb} << 20); print('allocated', len(b) >> 20, 'MB')"]
    rc, why, peak = run_limited(alloc(50), mem_mb=300)
    print(f"50 MB under a 300 MB cap:  rc {rc}, stopped: {why}, peak {peak} MB")
    ok1 = rc == 0 and why is None and peak < 300
    rc, why, peak = run_limited(alloc(2000), mem_mb=300)
    print(f"2000 MB under a 300 MB cap: rc {rc}, stopped: {why}, peak {peak} MB")
    ok2 = rc != 0 and peak <= 300
    rc, why, peak = run_limited([py, "-c", "import time; time.sleep(30)"], mem_mb=300, secs=2)
    print(f"30 s sleep under a 2 s limit: rc {rc}, stopped: {why}")
    ok3 = rc == 99 and why and "time" in why
    print(f"commit headroom now: {commit_free_mb()} MB")
    print("MEMGUARD SELFTEST", "PASS" if ok1 and ok2 and ok3 else "FAIL")
    return 0 if ok1 and ok2 and ok3 else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
