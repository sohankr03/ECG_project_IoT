"""
model_training.py
─────────────────────────────────────────────────────────────────────────────
Random Forest Classifier Training for ECG Anomaly Detection.

Steps:
  1. Load features_dataset.csv
  2. Apply saved StandardScaler (scaler_v1.pkl)
  3. Stratified 80/20 train-test split
  4. 5-fold cross-validation on training set
  5. Train final RandomForestClassifier (n_estimators=100)
  6. Evaluate on held-out test set
  7. Save model as ecg_rf_model_v1.pkl
  8. Print feature importances

Run:
  cd E:\Project\ECG_Project\model
  python model_training.py

Dependencies:
  - features_dataset.csv  (from data_preparation.py)
  - scaler_v1.pkl         (from data_preparation.py)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
SRC_DIR   = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
from feature_extraction import FEATURE_COLUMNS

DATASET_PATH = MODEL_DIR / "features_dataset.csv"
SCALER_PATH  = MODEL_DIR / "scaler_v1.pkl"
MODEL_PATH   = MODEL_DIR / "ecg_rf_model_v1.pkl"


# ══════════════════════════════════════════════════════════════════════════
# Main Training Pipeline
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Random Forest ECG Classifier — Training")
    print("=" * 60)

    # ── 1. Load Dataset ───────────────────────────────────────────────
    if not DATASET_PATH.exists():
        print(f"\n✗ Dataset not found: {DATASET_PATH}")
        print("  Please run data_preparation.py first.")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset: {len(df)} samples")

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    print(f"Class distribution: Normal={np.sum(y==0)}, Abnormal={np.sum(y==1)}")

    # ── 2. Load and Apply StandardScaler ─────────────────────────────
    if not SCALER_PATH.exists():
        print(f"\n✗ Scaler not found: {SCALER_PATH}")
        print("  Please run data_preparation.py first.")
        sys.exit(1)

    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)
    print(f"\n✓ StandardScaler applied ({len(FEATURE_COLUMNS)} features)")

    # ── 3. Stratified Train / Test Split (80/20) ──────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # ── 4. Random Forest Model ────────────────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,          # grow full trees (ensemble handles overfitting)
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight='balanced', # handle any class imbalance
        random_state=42,
        n_jobs=-1                 # use all CPU cores
    )

    # ── 5. Cross-Validation (5-fold, stratified) on Training Set ─────
    print("\nRunning 5-fold Cross-Validation on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    print(f"  CV F1 scores: {np.round(cv_scores, 3)}")
    print(f"  Mean F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    cv_acc = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"  Mean Accuracy (CV): {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

    # ── 6. Final Training on Full Training Set ────────────────────────
    print("\nTraining final model on full training set...")
    clf.fit(X_train, y_train)
    print("  ✓ Training complete")

    # ── 7. Evaluate on Held-Out Test Set ─────────────────────────────
    y_pred      = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "─" * 40)
    print("  Test Set Performance")
    print("─" * 40)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    # ── 8. Feature Importances ────────────────────────────────────────
    print("\nFeature Importances:")
    importances = clf.feature_importances_
    for fname, imp in sorted(zip(FEATURE_COLUMNS, importances),
                              key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"  {fname:<18} {imp:.4f}  {bar}")

    # ── 9. Save Model ─────────────────────────────────────────────────
    joblib.dump(clf, MODEL_PATH)
    print(f"\n✓ Model saved → {MODEL_PATH}")
    print(f"  Note: Always use scaler_v1.pkl to scale features before inference.")

    # ── 10. Save training metadata ────────────────────────────────────
    metadata = {
        "model_version"   : "v1",
        "n_estimators"    : 100,
        "feature_columns" : FEATURE_COLUMNS,
        "train_samples"   : int(len(X_train)),
        "test_samples"    : int(len(X_test)),
        "test_accuracy"   : float(round(acc, 4)),
        "test_f1"         : float(round(f1, 4)),
        "cv_f1_mean"      : float(round(cv_scores.mean(), 4)),
        "cv_f1_std"       : float(round(cv_scores.std(), 4)),
    }
    import json
    meta_path = MODEL_DIR / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved → {meta_path}")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Expected accuracy range: 75–90% (actual: {acc*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
