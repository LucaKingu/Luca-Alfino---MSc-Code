import subprocess
import os
import time
import csv
import shutil


SOLVER    = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\cmake-build-minisat-22\minisat_core.exe"
INSTANCES = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Instances\EXTRACTED2020"
FILTERED  = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Instances\FILTERED2020"
LOG_OUT   = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\filter_log_2020.csv"

TIMEOUT  = 300
MIN_TIME = 2.5
K_MAX    = 2000

os.makedirs(FILTERED, exist_ok=True)
os.makedirs(os.path.dirname(LOG_OUT), exist_ok=True)

instances = [f for f in os.listdir(INSTANCES) if f.endswith(".cnf")]
#instances = instances[:3]  # Testing purposes
print(f"Found {len(instances)} instances to filter\n")

filter_log = []

for idx, cnf in enumerate(instances):
    path        = os.path.join(INSTANCES, cnf)
    result_file = os.path.join(FILTERED, "_tmp_result.txt")
    stdout_file = os.path.join(FILTERED, "_tmp_stdout.txt")

    if os.path.exists(stdout_file):
        os.remove(stdout_file)
    if os.path.exists(result_file):
        os.remove(result_file)

    print(f"[{idx+1}/{len(instances)}] {cnf}")

    start     = time.time()
    timed_out = False

    with open(stdout_file, "w") as out_f:
        proc = subprocess.Popen(
            [SOLVER, path, result_file,
             "-policy=none",
             f"-feat-k={K_MAX}",
             "-feat-out=NUL", # Do not get features, only purpose is to filter.
             "-verb=1"],
            stdout=out_f,
            stderr=out_f
        )
        try:
            proc.wait(timeout=TIMEOUT)
            elapsed = time.time() - start
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed   = TIMEOUT
            timed_out = True

    stdout_data = ""
    try:
        with open(stdout_file, "r", errors="replace") as f:
            stdout_data = f.read()
    except:
        pass

    #print(f"  stdout length: {len(stdout_data)} chars")
    #print(f"  last 200 chars: {stdout_data[-200:]}")

    # Parse conflict count
    conflicts = 0
    for line in stdout_data.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            parts = line.split("|")
            if len(parts) >= 2:
                try:
                    val = int(parts[1].strip())
                    if val > conflicts:
                        conflicts = val
                except:
                    pass
        if "conflicts" in line and ":" in line and "/sec" in line:
            try:
                val = int(line.split(":")[1].strip().split()[0])
                if val > conflicts:
                    conflicts = val
            except:
                pass

    # Apply filters
    too_fast     = (elapsed < MIN_TIME)
    too_few_conf = (conflicts < K_MAX)
    keep         = not too_fast and not too_few_conf

    # Determine reason
    if timed_out and keep:
        reason = "kept_timeout"
    elif keep:
        reason = "kept"
    elif too_fast:
        reason = "too_fast"
    elif too_few_conf:
        reason = "too_few_conflicts"
    else:
        reason = "unknown"

    filter_log.append({
        "instance"          : cnf,
        "time_s"            : round(elapsed, 2),
        "conflicts"         : conflicts,
        "timed_out"         : timed_out,
        "too_fast"          : too_fast,
        "too_few_conflicts" : too_few_conf,
        "kept"              : keep,
        "reason"            : reason
    })

    if keep:
        shutil.copy(path, os.path.join(FILTERED, cnf))
        print(f"  KEEP  → {elapsed:.1f}s, {conflicts} conflicts {'[TIMEOUT]' if timed_out else ''}")
    else:
        print(f"  DROP  → [{reason}] {elapsed:.1f}s, {conflicts} conflicts")

# Write filter log
if filter_log:
    with open(LOG_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=filter_log[0].keys())
        writer.writeheader()
        writer.writerows(filter_log)

# Summary
kept    = sum(1 for r in filter_log if r["kept"])
dropped = len(filter_log) - kept

print(f"\n{'='*60}")
print(f"FILTERING COMPLETE")
print(f"  Total instances : {len(filter_log)}")
print(f"  Kept            : {kept}")
print(f"    - Solved      : {sum(1 for r in filter_log if r['kept'] and not r['timed_out'])}")
print(f"    - Timed out   : {sum(1 for r in filter_log if r['kept'] and r['timed_out'])}")
print(f"  Dropped         : {dropped}")
print(f"    - Timed out (too few conf) : {sum(1 for r in filter_log if r['timed_out'] and not r['kept'])}")
print(f"    - Too fast    : {sum(1 for r in filter_log if r['too_fast'])}")
print(f"    - Too few conf: {sum(1 for r in filter_log if r['too_few_conflicts'])}")
print(f"\nFilter log saved to: {LOG_OUT}")
print(f"Filtered instances in: {FILTERED}")