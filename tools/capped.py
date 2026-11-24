"""Run one command under memguard's hard commit cap. Usage: capped.py MB SECS LOGFILE -- cmd..."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import memguard
mb, secs, logf = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
cmd = sys.argv[sys.argv.index("--") + 1:]
memguard.preflight(mb + 4000)   # always leave >= 4 GB free for the rest of the laptop
with open(logf, "w") as f:
    rc, why, peak = memguard.run_limited(cmd, stdout=f, mem_mb=mb, secs=secs)
print(f"CAPPED DONE rc={rc} why={why} peak_commit_MB={peak}")
