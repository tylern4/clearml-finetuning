"""Print stage logs and scheduler evidence for the failed train attempts.

Runs on the clearml agent node (which mounts the stage-log directory that
login/dev nodes may not see), so we can recover the rank-level traceback
that torchrun did not capture (error_file: <N/A>) and get sacct records
from the cluster that actually owns the jobs.
"""
import glob
import os
import subprocess

LOG_DIR = os.environ.get(
    "CML_STAGE_LOG_DIR", "/pscratch/sd/t/tylern/scratch/clearml/stage_logs")
TASK_IDS = (
    "cf5df0dcc3fc4330bcd4bbc1b3f8e922",  # attempt 1, sbatch 1497128
    "91f613451d3a41259cfda1cebfdc8364",  # attempt 2, sbatch 1497131
)

print("== log dir:", LOG_DIR, "isdir:", os.path.isdir(LOG_DIR))
for alt in ("/mscratch/sd/t/tylern/scratch/clearml/stage_logs",
            "/pscratch/sd/t/tylern/scratch/clearml"):
    if os.path.isdir(alt):
        print("   alt exists:", alt,
              "->", sorted(os.listdir(alt))[:10])
if os.path.isdir(LOG_DIR):
    for p in sorted(glob.glob(os.path.join(LOG_DIR, "*"))):
        print("   ", p, os.path.getsize(p), "bytes")

for tid in TASK_IDS:
    p = os.path.join(LOG_DIR, f"{tid}_train.log")
    print(f"\n===== {p} =====")
    if not os.path.exists(p):
        print("MISSING")
        continue
    data = open(p, errors="replace").read()
    print(f"(size {len(data)} chars)")
    # index of python tracebacks so we can find the child (rank 1) one
    idx, starts = 0, []
    while True:
        i = data.find("Traceback (most recent call last)", idx)
        if i < 0:
            break
        starts.append(i)
        idx = i + 1
    print("traceback offsets:", starts)
    print("---- last 30000 chars ----")
    print(data[-30000:])

print("\n===== sacct (from inside this cluster) =====")
subprocess.run(
    ["sacct", "-j", "1497128,1497131,1497133",
     "--format=JobID,JobName%20,State%18,ExitCode,Elapsed,MaxRSS,"
     "NodeList%12,Reason%30"],
    check=False)
print("\n===== hostname =====")
subprocess.run(["hostname"], check=False)
