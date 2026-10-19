"""
Model Evaluation and Diagnostics
Computes ROC-AUC, PR-AUC, Brier score, Calibration Error, MAE, RMSE,
generates reliability diagrams and comparative baseline tables.
"""

from typing import Dict, Any, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        idx = np.where((y_prob >= bin_limits[i]) & (y_prob < bin_limits[i + 1]))[0]
        if len(idx) > 0:
            bin_acc = np.mean(y_true[idx])
            bin_conf = np.mean(y_prob[idx])
            ece += (len(idx) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate_system(
    predictor,
    X_test: np.ndarray,
    y_class_test: np.ndarray,
    y_reg_test: np.ndarray,
    output_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Evaluates the complete prediction system and generates visualization plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Predictions from Primary Model
    probs, preds, confs, tiers = predictor.predict_likelihood_and_confidence(X_test)
    pred_delay_hours = predictor.predict_delay_hours(X_test)
    
    # 2. Predictions from Baselines
    baseline_probs = predictor.baseline_classifier.predict_proba(X_test)[:, 1]
    baseline_preds = (baseline_probs >= 0.50).astype(int)
    baseline_delay_hours = np.maximum(predictor.baseline_regressor.predict(X_test), 0.0)
    
    # 3. Primary Classification Metrics
    roc_auc = roc_auc_score(y_class_test, probs)
    pr_auc = average_precision_score(y_class_test, probs)
    brier = brier_score_loss(y_class_test, probs)
    ece = compute_ece(y_class_test, probs)
    acc = accuracy_score(y_class_test, preds)
    prec = precision_score(y_class_test, preds, zero_division=0)
    rec = recall_score(y_class_test, preds, zero_division=0)
    f1 = f1_score(y_class_test, preds, zero_division=0)
    cm = confusion_matrix(y_class_test, preds).tolist()
    
    # Baseline Classification Metrics
    base_roc = roc_auc_score(y_class_test, baseline_probs)
    base_brier = brier_score_loss(y_class_test, baseline_probs)
    base_f1 = f1_score(y_class_test, baseline_preds, zero_division=0)
    
    # 4. Primary Regression Metrics
    mae = mean_absolute_error(y_reg_test, pred_delay_hours)
    rmse = root_mean_squared_error(y_reg_test, pred_delay_hours)
    r2 = r2_score(y_reg_test, pred_delay_hours)
    
    # Baseline Regression Metrics
    base_mae = mean_absolute_error(y_reg_test, baseline_delay_hours)
    base_rmse = root_mean_squared_error(y_reg_test, baseline_delay_hours)
    base_r2 = r2_score(y_reg_test, baseline_delay_hours)
    
    # 5. Generate and Save Plots
    # Plot 1: ROC & Precision-Recall Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    fpr, tpr, _ = roc_curve(y_class_test, probs)
    b_fpr, b_tpr, _ = roc_curve(y_class_test, baseline_probs)
    ax1.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Calibrated XGBoost (AUC = {roc_auc:.3f})")
    ax1.plot(b_fpr, b_tpr, color="#9ca3af", lw=1.8, linestyle="--", label=f"Logistic Regression (AUC = {base_roc:.3f})")
    ax1.plot([0, 1], [0, 1], color="#e5e7eb", linestyle=":")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    prec_pts, rec_pts, _ = precision_recall_curve(y_class_test, probs)
    ax2.plot(rec_pts, prec_pts, color="#10b981", lw=2.5, label=f"Calibrated XGBoost (PR-AUC = {pr_auc:.3f})")
    ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_plot_path = os.path.join(output_dir, "roc_pr_curves.png")
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    
    # Plot 2: Probability Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_class_test, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", color="#8b5cf6", lw=2, label=f"Calibrated Model (ECE = {ece:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.set_title("Probability Calibration Reliability Diagram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Fraction of Positives")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    calib_plot_path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(calib_plot_path, dpi=200)
    plt.close()
    
    # Plot 3: Delay Hours Regression Fit
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_reg_test, pred_delay_hours, alpha=0.35, color="#0ea5e9", edgecolors="none", s=25)
    max_val = max(np.max(y_reg_test), np.max(pred_delay_hours))
    ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Ideal Line (y = y_hat)")
    ax.set_title(f"Delay Duration Fit (MAE: {mae:.2f}h, R²: {r2:.3f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Actual Waiting Time (Hours)")
    ax.set_ylabel("Predicted Waiting Time (Hours)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    reg_plot_path = os.path.join(output_dir, "delay_hours_regression.png")
    plt.savefig(reg_plot_path, dpi=200)
    plt.close()
    
    metrics = {
        "classification": {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_score": round(float(brier), 4),
            "expected_calibration_error": round(float(ece), 4),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "baseline_roc_auc": round(float(base_roc), 4),
            "baseline_brier_score": round(float(base_brier), 4),
            "baseline_f1_score": round(float(base_f1), 4),
        },
        "regression": {
            "mae_hours": round(float(mae), 3),
            "rmse_hours": round(float(rmse), 3),
            "r2_score": round(float(r2), 4),
            "baseline_mae_hours": round(float(base_mae), 3),
            "baseline_rmse_hours": round(float(base_rmse), 3),
            "baseline_r2_score": round(float(base_r2), 4),
        },
        "plots": {
            "roc_pr_path": roc_plot_path,
            "calibration_path": calib_plot_path,
            "regression_path": reg_plot_path
        }
    }
    
    return metrics
