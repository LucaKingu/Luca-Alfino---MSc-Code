
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import joblib

from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics         import (f1_score, confusion_matrix, make_scorer)
from sklearn.dummy           import DummyClassifier
from xgboost                 import XGBClassifier




# 1. LOAD DATASET
DATASET = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\dataset.csv"
FIGURES = r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Figures"
os.makedirs(FIGURES, exist_ok=True)

df = pd.read_csv(DATASET)

# Drop zero-variance features removed after EDA
df = df.drop(columns=['restart_count', 'restart_rate'], errors='ignore')





# 2. FEATURE SETS
# LR uses reduced set - correlated duplicates removed
# Tree/MLP use full set - robust to correlation
FEATURES_LR = [
    'mean_lbd', 'lbd_trend', 'fraction_lbd2',
    'mean_clause_len', 'mean_dec_level',
    'conflict_rate', 'progress_at_k', 'learnts_per_conflict'
]

FEATURES_TREE = [
    'mean_lbd', 'variance_lbd', 'lbd_trend', 'fraction_lbd2',
    'mean_clause_len', 'mean_dec_level', 'mean_props_per_conflict',
    'conflict_rate', 'progress_at_k', 'learnts_per_conflict'
]

X_lr   = df[FEATURES_LR].values
X_tree = df[FEATURES_TREE].values
y      = (df['label'] == 'AGG').astype(int).values  # AGG=1, CONS=0

print(f"Dataset : {len(y)} instances")
print(f"Labels  : AGG={y.sum()}, CONS={len(y)-y.sum()}")
print(f"LR features   : {len(FEATURES_LR)}")
print(f"Tree features : {len(FEATURES_TREE)}\n")





# 2b. TRAIN/TEST SPLIT (85/15) FOR RUNTIME EXPERIMENT
from sklearn.model_selection import train_test_split

X_lr_train, X_lr_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_lr, y, np.arange(len(y)),
    test_size=0.15,
    stratify=y,
    random_state=42
)

X_tree_train = X_tree[idx_train]
X_tree_test  = X_tree[idx_test]

print(f"Train : {len(y_train)} instances  (AGG={y_train.sum()}, CONS={(y_train==0).sum()})")
print(f"Test  : {len(y_test)} instances  (AGG={y_test.sum()}, CONS={(y_test==0).sum()})")

# Save test instance names for runtime script
test_instances_df = df.iloc[idx_test][['instance', 'label']].copy()
test_instances_df.to_csv(
    r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Data\test_instances.csv",
    index=False)
print(f"Test instances saved.\n")





# 3. CROSS-VALIDATION SETUP
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy' : 'accuracy',
    'f1_macro' : 'f1_macro',
    'f1_agg'   : make_scorer(f1_score, pos_label=1, zero_division=0),
    'f1_cons'  : make_scorer(f1_score, pos_label=0, zero_division=0)
}






# 4. TUNE ALL MODELS VIA GRIDSEARCH
print("Tuning models via GridSearchCV...\n")

# Logistic Regression
lr_base   = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(class_weight='balanced',
                                  max_iter=1000,
                                  random_state=42))
])
lr_search = GridSearchCV(lr_base,
                          {'clf__C': [0.001, 0.01, 0.1, 1.0, 10.0]},
                          cv=cv, scoring='f1_macro', refit=True)
lr_search.fit(X_lr, y)
print(f"Best LR params   : {lr_search.best_params_}")
print(f"Best LR F1 Macro : {lr_search.best_score_:.3f}")

# Retrain best LR on training split only for runtime experiment
lr_final = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
                   C=lr_search.best_params_['clf__C'],
                   class_weight='balanced',
                   max_iter=1000,
                   random_state=42))
])
lr_final.fit(X_lr_train, y_train)

# Evaluate on held-out test set
from sklearn.metrics import classification_report
y_pred_test = lr_final.predict(X_lr_test)
print("\n=== HELD-OUT TEST SET PERFORMANCE (LR) ===")
print(classification_report(y_test, y_pred_test,
                             target_names=['CONS', 'AGG']))

# Save final model for runtime experiment
os.makedirs(r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Models", exist_ok=True)
joblib.dump(lr_final,
            r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Models\lr_model_final.pkl")
print("Final model saved → lr_model_final.pkl")

# XGBoost
xgb_base   = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    XGBClassifier(scale_pos_weight=105/52,
                              random_state=42,
                              eval_metric='logloss',
                              verbosity=0))
])
xgb_search = GridSearchCV(xgb_base, {
    'clf__max_depth'       : [2, 3, 4],
    'clf__n_estimators'    : [50, 100],
    'clf__learning_rate'   : [0.05, 0.1],
    'clf__min_child_weight': [3, 5],
    'clf__subsample'       : [0.7, 0.8],
},
cv=cv, scoring='f1_macro', refit=True, n_jobs=-1)
xgb_search.fit(X_tree, y)
print(f"\nBest XGBoost params   : {xgb_search.best_params_}")
print(f"Best XGBoost F1 Macro : {xgb_search.best_score_:.3f}")

# --- MLP ---
mlp_base   = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    MLPClassifier(random_state=42,
                              max_iter=500,
                              early_stopping=True,
                              validation_fraction=0.15))
])
mlp_search = GridSearchCV(mlp_base, {
    'clf__hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
    'clf__alpha'             : [0.001, 0.01, 0.1],
    'clf__activation'        : ['relu', 'tanh'],
},
cv=cv, scoring='f1_macro', refit=True, n_jobs=-1)
mlp_search.fit(X_tree, y)
print(f"\nBest MLP params   : {mlp_search.best_params_}")
print(f"Best MLP F1 Macro : {mlp_search.best_score_:.3f}")





# 5. DEFINE MODELS USING BEST PARAMS
models = {
    "Majority Baseline": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    DummyClassifier(strategy="most_frequent"))
        ]),
        "X": X_tree
    },
    "Logistic Regression": {
        "pipeline": lr_search.best_estimator_,
        "X"       : X_lr
    },
    "XGBoost": {
        "pipeline": xgb_search.best_estimator_,
        "X"       : X_tree
    },
    "MLP": {
        "pipeline": mlp_search.best_estimator_,
        "X"       : X_tree
    }
}

# Save best model
joblib.dump(lr_search.best_estimator_,
            r"D:\MASTER THESIS\MiniSAT 2.2\minisat\Pipeline\Models\lr_model.pkl")
print("Model saved.")




# 6. CROSS-VALIDATION EVALUATION
results = {}
print("\n" + "="*60)
print("CROSS-VALIDATION RESULTS (5-Fold Stratified)")
print("="*60)

for name, model_dict in models.items():
    pipeline = model_dict["pipeline"]
    X_use    = model_dict["X"]

    cv_out = cross_validate(pipeline, X_use, y,
                             cv=cv, scoring=scoring,
                             return_train_score=True)
    results[name] = {
        'accuracy'    : cv_out['test_accuracy'].mean(),
        'accuracy_std': cv_out['test_accuracy'].std(),
        'f1_macro'    : cv_out['test_f1_macro'].mean(),
        'f1_macro_std': cv_out['test_f1_macro'].std(),
        'f1_agg'      : cv_out['test_f1_agg'].mean(),
        'f1_cons'     : cv_out['test_f1_cons'].mean(),
        'train_f1'    : cv_out['train_f1_macro'].mean(),
    }
    r   = results[name]
    gap = r['train_f1'] - r['f1_macro']
    print(f"\n{name}")
    print(f"  Accuracy : {r['accuracy']:.3f} ± {r['accuracy_std']:.3f}")
    print(f"  F1 Macro : {r['f1_macro']:.3f} ± {r['f1_macro_std']:.3f}")
    print(f"  F1 AGG   : {r['f1_agg']:.3f}")
    print(f"  F1 CONS  : {r['f1_cons']:.3f}")
    print(f"  Train F1 : {r['train_f1']:.3f}  |  Gap: {gap:.3f}  {'OVERFIT' if gap > 0.15 else 'Not Overfit'}")






# 7. SUMMARY TABLE
results_df = pd.DataFrame(results).T[[
    'accuracy','accuracy_std','f1_macro','f1_macro_std',
    'f1_agg','f1_cons','train_f1'
]]
results_df.columns = ['Accuracy','Acc Std','F1 Macro',
                       'F1 Std','F1 AGG','F1 CONS','Train F1']
print(f"\n\n{'='*65}")
print("SUMMARY TABLE")
print('='*65)
print(results_df.round(3).to_string())
results_df.to_csv(os.path.join(FIGURES, "cv_results.csv"))






# 8. CONFUSION MATRICES
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Confusion Matrices (Full Dataset Refit)", fontsize=13)

for ax, (name, model_dict) in zip(axes, models.items()):
    pipeline = model_dict["pipeline"]
    X_use    = model_dict["X"]
    pipeline.fit(X_use, y)
    y_pred = pipeline.predict(X_use)
    cm     = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['CONS', 'AGG'],
                yticklabels=['CONS', 'AGG'])
    acc = (y == y_pred).mean()
    ax.set_title(f"{name}\nAcc={acc:.2f}", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "confusion_matrices.png"), dpi=150)
plt.show()
print("\nConfusion matrices saved.")






# 9. FEATURE IMPORTANCE - XGBoost
xgb_pipeline = models["XGBoost"]["pipeline"]
xgb_pipeline.fit(X_tree, y)
importances  = xgb_pipeline.named_steps['clf'].feature_importances_
feat_imp     = pd.Series(importances, index=FEATURES_TREE).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='steelblue')
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "feature_importance.png"), dpi=150)
plt.show()
print("Feature importance saved.")






# 10. LOGISTIC REGRESSION COEFFICIENTS
lr_pipeline = models["Logistic Regression"]["pipeline"]
lr_pipeline.fit(X_lr, y)
coefs       = lr_pipeline.named_steps['clf'].coef_[0]
coef_series = pd.Series(coefs, index=FEATURES_LR).sort_values()
colors      = ['crimson' if c < 0 else 'steelblue' for c in coef_series]

plt.figure(figsize=(8, 5))
coef_series.plot(kind='barh', color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Logistic Regression Coefficients\n(Blue = predicts AGG, Red = predicts CONS)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "lr_coefficients.png"), dpi=150)
plt.show()
print("LR coefficients saved.")

print("\nAll done. Check Figures folder for outputs.")