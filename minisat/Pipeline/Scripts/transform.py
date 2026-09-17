import pandas as pd
import os

FEATURES_DIR = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\FEATURES_COMBINED1"
LABELS_FILE  = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\LABELS_COMBINED.xlsx"
DATASET_OUT  = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\dataset.csv"

# Features to remove
REMOVE_FEATURES = ["conflict_count", "num_learnts", "avg_backtrack_depth", "reduction_count"]

# Load labels
labels_df = pd.read_excel(LABELS_FILE)
labels_df = labels_df[["instance", "label"]].copy()

rows = []
missing = []

for _, label_row in labels_df.iterrows():
    cnf       = label_row["instance"]
    base_name = cnf.replace(".cnf", "")
    feat_file = os.path.join(FEATURES_DIR, f"{base_name}.csv")

    if not os.path.exists(feat_file):
        missing.append(cnf)
        continue

    # Read feature,value CSV and pivot to single row
    feat_df   = pd.read_csv(feat_file, header=0, names=["feature", "value"])

    # Remove unwanted features
    feat_df   = feat_df[~feat_df["feature"].isin(REMOVE_FEATURES)]

    # Pivot to dict
    feat_dict = dict(zip(feat_df["feature"], feat_df["value"]))
    feat_dict["instance"] = cnf
    feat_dict["label"]    = label_row["label"]
    rows.append(feat_dict)

dataset = pd.DataFrame(rows)

# Reorder columns - instance and label at front
feature_cols = [c for c in dataset.columns if c not in ["instance", "label"]]
dataset      = dataset[["instance", "label"] + feature_cols]

print(f"Dataset shape: {dataset.shape}")
print(f"Missing instances: {len(missing)}")
print(f"\nLabel distribution:")
print(dataset["label"].value_counts())
print(f"\nFeatures ({len(feature_cols)}):")
print(feature_cols)
print(f"\nFirst few rows:")
print(dataset.head())
print(f"\nAny nulls:")
print(dataset.isnull().sum())

dataset.to_csv(DATASET_OUT, index=False)
print(f"\nDataset saved to: {DATASET_OUT}")