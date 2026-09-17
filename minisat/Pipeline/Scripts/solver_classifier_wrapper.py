
import subprocess
import os
import time
import csv
import joblib
import pandas as pd
import numpy as np




# Configuration
SOLVER        = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\cmake-build-minisat-22\minisat_core.exe"
MODEL_PATH    = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Models\lr_model_final.pkl"
TEST_CSV      = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\test_instances.csv"
FILTERED_DIRS = [
    r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Instances\FILTERED",
    r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Instances\FILTERED2021",
    r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Instances\FILTERED2020",
]
FEATURES_DIR  = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\FEATURES_COMBINED"
RESULTS_OUT   = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\runtime_results.csv"
TEMP_DIR      = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Temp"

TIMEOUT       = 1200
K             = 2000

FEATURES_LR   = [
    'mean_lbd', 'lbd_trend', 'fraction_lbd2',
    'mean_clause_len', 'mean_dec_level',
    'conflict_rate', 'progress_at_k', 'learnts_per_conflict'
]

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_OUT), exist_ok=True)



# Load model and test instances
model    = joblib.load(MODEL_PATH)
test_df  = pd.read_csv(TEST_CSV)
print(f"Model loaded.")
print(f"Test instances: {len(test_df)}")
print(f"  AGG : {(test_df['label']=='AGG').sum()}")
print(f"  CONS: {(test_df['label']=='CONS').sum()}\n")

results = []


def find_instance(cnf):
    """Search all filtered directories for the instance file."""
    for folder in FILTERED_DIRS:
        candidate = os.path.join(folder, cnf)
        if os.path.exists(candidate):
            return candidate
    return None


def run_minisat(instance_path, result_file, policy,
                feat_file=None, timeout=TIMEOUT):
    stdout_file = os.path.join(TEMP_DIR, "_stdout.txt")
    if os.path.exists(stdout_file): os.remove(stdout_file)
    if os.path.exists(result_file): os.remove(result_file)

    cmd = [SOLVER, instance_path, result_file,
           f"-policy={policy}", "-verb=0"]
    if feat_file:
        cmd += [f"-feat-k={K}", f"-feat-out={feat_file}"]

    with open(stdout_file, "w") as out_f:
        proc  = subprocess.Popen(cmd, stdout=out_f, stderr=out_f)
        start = time.time()
        try:
            exit_code = proc.wait(timeout=timeout)
            elapsed   = time.time() - start
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed   = timeout
            timed_out = True
            exit_code = None

    status = "SAT"     if exit_code == 10 else \
             "UNSAT"   if exit_code == 20 else \
             "TIMEOUT"
    return elapsed, timed_out, status


for idx, row_data in test_df.iterrows():
    cnf        = row_data['instance']
    true_label = row_data['label']
    base_name  = cnf.replace(".cnf", "")
    row        = {"instance": cnf, "true_label": true_label}

    print(f"[{idx+1}/{len(test_df)}] {cnf}")
    print(f"  True label: {true_label}")

    # Search all filtered folders
    path = find_instance(cnf)
    if path is None:
        print(f"  SKIP - instance file not found in any folder")
        row["skip_reason"] = "file_not_found"
        results.append(row)
        continue

    print(f"  Found at: {path}")



    # PHASE 1: Load pre-extracted features or re-extract
    feat_file = os.path.join(FEATURES_DIR, f"{base_name}.csv")

    if os.path.exists(feat_file):
        feat_df   = pd.read_csv(feat_file,
                                header=0, names=["feature", "value"])
        feat_dict = dict(zip(feat_df["feature"], feat_df["value"]))
        print(f"  Features loaded from existing file.")
        row["baseline_time"] = 0.0  # already extracted previously
    else:
        print(f"  Re-extracting features via baseline run...")
        tmp_feat   = os.path.join(TEMP_DIR, f"{base_name}_feat.csv")
        tmp_result = os.path.join(TEMP_DIR, "_baseline_result.txt")
        elapsed_b, _, _ = run_minisat(
            path, tmp_result, "none", feat_file=tmp_feat)

        if not os.path.exists(tmp_feat):
            print(f"  SKIP - feature extraction failed")
            row["skip_reason"] = "feature_extraction_failed"
            results.append(row)
            continue

        feat_df   = pd.read_csv(tmp_feat,
                                header=0, names=["feature", "value"])
        feat_dict = dict(zip(feat_df["feature"], feat_df["value"]))
        row["baseline_time"] = round(elapsed_b, 2)
        print(f"  Baseline run: {elapsed_b:.1f}s")




    # PHASE 2: Predict policy
    missing = [f for f in FEATURES_LR if f not in feat_dict]
    if missing:
        print(f"  SKIP - missing features: {missing}")
        row["skip_reason"] = f"missing_features: {missing}"
        results.append(row)
        continue

    feat_vector = np.array([[feat_dict[f] for f in FEATURES_LR]])
    prediction  = model.predict(feat_vector)[0]
    probability = model.predict_proba(feat_vector)[0]
    policy      = "agg" if prediction == 1 else "cons"
    confidence  = probability[prediction]
    correct     = (policy.upper() == true_label)

    row["predicted_policy"]   = policy.upper()
    row["prediction_correct"] = correct
    row["confidence"]         = round(confidence, 4)
    print(f"  Predicted : {policy.upper()} "
          f"(conf={confidence:.3f}) "
          f"{'✓ CORRECT' if correct else '✗ WRONG'}")



    # PHASE 3: Run all four configurations
    # none, agg, cons, predicted
    run_configs = ["none", "agg", "cons"]

    # Only add predicted as separate run if it differs
    # (it always differs from none, but agg/cons already covered)
    for pol in run_configs:
        result_file = os.path.join(TEMP_DIR, f"_{pol}_result.txt")
        elapsed, timed_out, status = run_minisat(
            path, result_file, pol)

        row[f"time_{pol}"]   = round(elapsed, 2)
        row[f"status_{pol}"] = status
        print(f"  {pol.upper():<12} → {elapsed:>7.1f}s [{status}]")

    # Map predicted to the corresponding already-run config
    row["time_predicted"]   = row[f"time_{policy}"]
    row["status_predicted"] = row[f"status_{policy}"]
    print(f"  PREDICTED    → {row['time_predicted']}s "
          f"[{row['status_predicted']}] "
          f"(mapped from {policy.upper()})")

    results.append(row)

    # Save progress after every instance
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        with open(RESULTS_OUT, "w", newline="") as f:
            writer = csv.DictWriter(f,
                                     fieldnames=sorted(all_keys),
                                     extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    # Cleanup temp files
    for tmp in os.listdir(TEMP_DIR):
        if tmp.startswith("_"):
            try:
                os.remove(os.path.join(TEMP_DIR, tmp))
            except:
                pass



# Summary
valid         = [r for r in results if "skip_reason" not in r]
correct_preds = sum(1 for r in valid if r.get("prediction_correct"))
skipped       = [r for r in results if "skip_reason" in r]

print(f"\n{'='*60}")
print(f"RUNTIME PREDICTION COMPLETE")
print(f"{'='*60}")
print(f"  Test instances     : {len(test_df)}")
print(f"  Valid runs         : {len(valid)}")
print(f"  Skipped            : {len(skipped)}")
print(f"  Correct predictions: {correct_preds}/{len(valid)} "
      f"({100*correct_preds/max(len(valid),1):.1f}%)")

pred_times = [r['time_predicted'] for r in valid if 'time_predicted' in r]
agg_times  = [r['time_agg']       for r in valid if 'time_agg'       in r]
cons_times = [r['time_cons']      for r in valid if 'time_cons'      in r]
none_times = [r['time_none']      for r in valid if 'time_none'      in r]

if pred_times:
    print(f"\n  Avg time - Predicted : {np.mean(pred_times):.1f}s")
    print(f"  Avg time - AGG fixed : {np.mean(agg_times):.1f}s")
    print(f"  Avg time - CONS fixed: {np.mean(cons_times):.1f}s")
    print(f"  Avg time - No policy : {np.mean(none_times):.1f}s")
    print(f"\n  Predicted vs AGG  : {np.mean(pred_times) - np.mean(agg_times):+.1f}s")
    print(f"  Predicted vs CONS : {np.mean(pred_times) - np.mean(cons_times):+.1f}s")
    print(f"  Predicted vs None : {np.mean(pred_times) - np.mean(none_times):+.1f}s")

print(f"\nResults saved to: {RESULTS_OUT}")