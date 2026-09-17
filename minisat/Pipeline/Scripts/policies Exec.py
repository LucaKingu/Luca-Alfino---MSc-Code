import subprocess
import os
import time
import csv

SOLVER       = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\cmake-build-minisat-22\minisat_core.exe"
FILTERED     = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Instances\FILTERED2020"
FEATURES_DIR = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\Features2020_postCrash_2" # COPY PASTE BOTH FILE OUTPUTS INTO ORIGINAL POST RUN
LABELS_OUT   = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\labels2020_postCrash_2.csv"

TIMEOUT = 1200
K       = 2000

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LABELS_OUT), exist_ok=True)

# Clean up leftover temp files from previous runs
print("Cleaning up temp files...")
for tmp in os.listdir(FILTERED):
    if tmp.startswith("_tmp_"):

        try:
            os.remove(os.path.join(FILTERED, tmp))
        except:
            pass

instances = [f for f in os.listdir(FILTERED) if f.endswith(".cnf")]
instances = instances[387:]  # TEMP - remove after testing
print(f"Found {len(instances)} instances to process\n")

results         = []
script_start    = time.time()
total_agg_time  = 0
total_cons_time = 0

fieldnames = ["instance", "label", "speedup",
              "time_agg", "status_agg",
              "time_cons", "status_cons",
              "features_written"]

for idx, cnf in enumerate(instances):
    path      = os.path.join(FILTERED, cnf)
    base_name = cnf.replace(".cnf", "")
    row       = {"instance": cnf}

    print(f"[{idx+1}/{len(instances)}] {cnf}")

    for policy in ["agg", "cons"]:
        result_file = os.path.join(FILTERED, f"_tmp_{policy}_result.txt")
        stdout_file = os.path.join(FILTERED, f"_tmp_{policy}_stdout.txt")

        if os.path.exists(stdout_file): os.remove(stdout_file)
        if os.path.exists(result_file): os.remove(result_file)

        start_run = time.time()
        timed_out = False
        status    = "UNKNOWN"

        with open(stdout_file, "w") as out_f:
            proc = subprocess.Popen(
                [SOLVER, path, result_file,
                 f"-policy={policy}",
                 "-verb=0"],
                stdout=out_f,
                stderr=out_f
            )
            try:
                exit_code = proc.wait(timeout=TIMEOUT)
                elapsed   = time.time() - start_run
                if exit_code == 10:
                    status = "SAT"
                elif exit_code == 20:
                    status = "UNSAT"
                else:
                    status = f"ERR_{exit_code}"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                elapsed   = TIMEOUT
                timed_out = True
                status    = "TIMEOUT"

        if policy == "agg":
            total_agg_time  += elapsed
        else:
            total_cons_time += elapsed

        row[f"time_{policy}"]    = round(elapsed, 2)
        row[f"status_{policy}"]  = status
        row[f"timeout_{policy}"] = timed_out
        print(f"  {policy.upper():<4} → {elapsed:>7.2f}s ({elapsed/60:>5.2f} min) [{status}]")

    agg_solved  = not row["timeout_agg"]
    cons_solved = not row["timeout_cons"]

    if agg_solved and cons_solved:
        row["label"]   = "AGG" if row["time_agg"] < row["time_cons"] else "CONS"
        row["speedup"] = round(
            max(row["time_agg"], row["time_cons"]) /
            max(min(row["time_agg"], row["time_cons"]), 0.001), 3)
    elif agg_solved:
        row["label"]   = "AGG"
        row["speedup"] = None
    elif cons_solved:
        row["label"]   = "CONS"
        row["speedup"] = None
    else:
        row["label"]   = "UNKNOWN"
        row["speedup"] = None

    print(f"  LABEL → {row['label']} (AGG={row['time_agg']}s, CONS={row['time_cons']}s)")

    if row["label"] != "UNKNOWN":
        feat_file   = os.path.join(FEATURES_DIR, f"{base_name}.csv")
        result_file = os.path.join(FILTERED, "_tmp_baseline_result.txt")
        stdout_file = os.path.join(FILTERED, "_tmp_baseline_stdout.txt")

        if os.path.exists(stdout_file): os.remove(stdout_file)
        if os.path.exists(result_file): os.remove(result_file)

        with open(stdout_file, "w") as out_f:
            proc = subprocess.Popen(
                [SOLVER, path, result_file,
                 "-policy=none",
                 f"-feat-k={K}",
                 f"-feat-out={feat_file}",
                 "-verb=0"],
                stdout=out_f,
                stderr=out_f
            )
            try:
                proc.wait(timeout=TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        row["features_written"] = os.path.exists(feat_file)
        print(f"  BASELINE → features={'YES' if row['features_written'] else 'NO'}")
    else:
        row["features_written"] = False
        print(f"  BASELINE → SKIPPED (UNKNOWN)")

    results.append(row)

    # Save progress after every instance
    with open(LABELS_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Clean up temp files for this instance
    for tmp in os.listdir(FILTERED):
        if tmp.startswith("_tmp_"):
            try:
                os.remove(os.path.join(FILTERED, tmp))
            except:
                pass

# Summary
script_total = time.time() - script_start
labeled      = [r for r in results if r["label"] != "UNKNOWN"]
agg_wins     = sum(1 for r in labeled if r["label"] == "AGG")
cons_wins    = sum(1 for r in labeled if r["label"] == "CONS")
no_features  = sum(1 for r in labeled if not r["features_written"])

print(f"\n{'='*65}")
print(f"PIPELINE COMPLETE")
print(f"{'='*65}")
print(f"  Total instances      : {len(results)}")
print(f"  Successfully labeled : {len(labeled)}")
print(f"    - AGG was faster   : {agg_wins}")
print(f"    - CONS was faster  : {cons_wins}")
print(f"  Unknown (both TO)    : {len(results) - len(labeled)}")
print(f"  Missing features     : {no_features}")
print(f"--- Runtimes ---")
print(f"  Total AGG time       : {total_agg_time:>8.2f}s ({total_agg_time/60:.2f} min)")
print(f"  Total CONS time      : {total_cons_time:>8.2f}s ({total_cons_time/60:.2f} min)")
print(f"  Total script time    : {script_total:>8.2f}s ({script_total/60:.2f} min)")
print(f"{'='*65}")