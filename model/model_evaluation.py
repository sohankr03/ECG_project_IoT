"""
model_evaluation.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive evaluation of the trained Random Forest ECG classifier.

Generates:
  - Classification report (accuracy, precision, recall, F1)
  - Confusion matrix → assets/confusion_matrix.png
  - ROC curve         → assets/roc_curve.png
  - Feature importance bar chart → assets/feature_importance.png

Run:
  cd E:\Project\ECG_Project\model
  python model_evaluation.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
MODEL_DIR  = ROOT_DIR / "model"
ASSETS_DIR = ROOT_DIR / "assets"
SRC_DIR    = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
from feature_extraction import FEATURE_COLUMNS

ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    """Plot and save a styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    classes = ['Normal', 'Abnormal']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, color='white', fontsize=12)
    ax.set_yticklabels(classes, color='white', fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] < thresh else 'black',
                    fontsize=16, fontweight='bold')

    ax.set_ylabel('Actual Label', color='white', fontsize=13)
    ax.set_xlabel('Predicted Label', color='white', fontsize=13)
    ax.set_title('ECG Classifier — Confusion Matrix', color='white', fontsize=14, pad=15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"  ✓ Confusion matrix → {save_path}")


def plot_roc_curve(y_test: np.ndarray, y_prob: np.ndarray, save_path: Path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')

    ax.plot(fpr, tpr, color='#00D4FF', lw=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#888', lw=1.5, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='white', fontsize=13)
    ax.set_ylabel('True Positive Rate', color='white', fontsize=13)
    ax.set_title('ROC Curve — ECG Anomaly Detection', color='white', fontsize=14, pad=15)
    ax.tick_params(colors='white')
    ax.legend(loc='lower right', facecolor='#1E1E2E', labelcolor='white', fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.grid(True, color='#333', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"  ✓ ROC curve → {save_path}")


def plot_feature_importance(clf, feature_names: list, save_path: Path):
    """Plot and save feature importance bar chart."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in indices]
    sorted_imps   = importances[indices]

    colors = ['#00D4FF', '#FF6B6B', '#A8FFB0', '#FFD700', '#FF9ECD', '#C3A6FF']

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0F1117')
    ax.set_facecolor('#0F1117')

    bars = ax.bar(sorted_names, sorted_imps,
                  color=colors[:len(sorted_names)], edgecolor='#333', linewidth=0.8)

    for bar, val in zip(bars, sorted_imps):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom',
                color='white', fontsize=10)

    ax.set_title('Feature Importances — Random Forest', color='white', fontsize=14, pad=15)
    ax.set_ylabel('Importance', color='white', fontsize=12)
    ax.tick_params(colors='white', axis='both')
    ax.set_xticklabels(sorted_names, rotation=20, ha='right', color='white', fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.grid(True, axis='y', color='#333', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close()
    print(f"  ✓ Feature importance → {save_path}")


def main():
    print("=" * 60)
    print("  Model Evaluation")
    print("=" * 60)

    # ── Load artifacts ────────────────────────────────────────────────
    dataset_path = MODEL_DIR / "features_dataset.csv"
    scaler_path  = MODEL_DIR / "scaler_v1.pkl"
    model_path   = MODEL_DIR / "ecg_rf_model_v1.pkl"

    for p in [dataset_path, scaler_path, model_path]:
        if not p.exists():
            print(f"\n✗ Missing: {p}")
            print("  Run data_preparation.py then model_training.py first.")
            sys.exit(1)

    df      = pd.read_csv(dataset_path)
    scaler  = joblib.load(scaler_path)
    clf     = joblib.load(model_path)

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    # Re-create same test split (same random_state=42, test_size=0.20)
    X_scaled = scaler.transform(X)
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Predictions ───────────────────────────────────────────────────
    y_pred     = clf.predict(X_test)
    y_prob     = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"]))

    # ── Plots ─────────────────────────────────────────────────────────
    print("\nGenerating evaluation plots...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm,    ASSETS_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, ASSETS_DIR / "roc_curve.png")
    plot_feature_importance(clf, FEATURE_COLUMNS, ASSETS_DIR / "feature_importance.png")

    print(f"\n  All plots saved to: {ASSETS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
