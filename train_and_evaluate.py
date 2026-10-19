"""
Master Pipeline: Training, Calibration, Evaluation, and Explainability Generation
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap
import joblib

from src.data_generator import generate_synthetic_port_dataset
from src.preprocessing import create_preprocessor
from src.model import PortCongestionPredictor
from src.explainability import MaritimeExplainer
from src.evaluate import evaluate_system


def run_pipeline(
    n_samples: int = 2500,
    data_dir: str = "data",
    model_dir: str = "models",
    artifacts_dir: str = "artifacts",
    random_seed: int = 42
):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print("=" * 70)
    print("AIQON - PORT CONGESTION & SHIPMENT DELAY PREDICTION PIPELINE")
    print("=" * 70)
    
    # 1. Dataset Generation / Loading
    dataset_csv = os.path.join(data_dir, "port_congestion_dataset.csv")
    print(f"\n[1/6] Generating synthetic AIS & port activity dataset ({n_samples} records)...")
    df = generate_synthetic_port_dataset(n_samples=n_samples, random_seed=random_seed)
    df.to_csv(dataset_csv, index=False)
    print(f"      Saved dataset to: {dataset_csv}")
    print(f"      Delay incidence rate: {df['is_congested_delayed'].mean()*100:.1f}%")
    print(f"      Mean waiting time: {df['waiting_time_hours'].mean():.2f} hours (max: {df['waiting_time_hours'].max():.2f}h)")

    # 2. Preprocessing & Feature Engineering
    print("\n[2/6] Executing maritime feature engineering and transformations...")
    feature_engineer, col_transformer = create_preprocessor()
    
    df_engineered = feature_engineer.transform(df)
    
    # Split train and test before fitting transformers to prevent data leakage
    train_df, test_df = train_test_split(
        df_engineered,
        test_size=0.20,
        random_state=random_seed,
        stratify=df_engineered["is_congested_delayed"]
    )
    
    # Fit column transformer on train
    X_train = col_transformer.fit_transform(train_df)
    X_test = col_transformer.transform(test_df)
    
    y_class_train = train_df["is_congested_delayed"].values
    y_class_test = test_df["is_congested_delayed"].values
    
    y_reg_train = train_df["waiting_time_hours"].values
    y_reg_test = test_df["waiting_time_hours"].values
    
    # Get feature names from ColumnTransformer
    cat_encoder = col_transformer.named_transformers_["cat"]
    num_cols = col_transformer.transformers[0][2]
    cat_cols_raw = col_transformer.transformers[1][2]
    cat_encoded_cols = list(cat_encoder.get_feature_names_out(cat_cols_raw))
    feature_names = list(num_cols) + cat_encoded_cols
    print(f"      Engineered {len(feature_names)} model features.")

    # 3. Model Training & Probability Calibration
    print("\n[3/6] Fitting Dual-Head Predictor (Calibrated XGBoost + Regressor + Baselines)...")
    predictor = PortCongestionPredictor(random_state=random_seed)
    predictor.fit(X_train, y_class_train, y_reg_train, feature_names=feature_names)
    print("      Model training and sigmoid calibration completed successfully.")

    # 4. Evaluation & Diagnostic Plots
    print("\n[4/6] Evaluating on held-out test set (500 samples)...")
    metrics = evaluate_system(predictor, X_test, y_class_test, y_reg_test, output_dir=artifacts_dir)
    
    metrics_path = os.path.join(artifacts_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\n" + "-" * 50)
    print("EVALUATION RESULTS (TEST SET):")
    print("-" * 50)
    print(f"  Classification (Congestion Event):")
    print(f"    - ROC-AUC:                   {metrics['classification']['roc_auc']:.4f}  (Baseline Logistic: {metrics['classification']['baseline_roc_auc']:.4f})")
    print(f"    - PR-AUC:                    {metrics['classification']['pr_auc']:.4f}")
    print(f"    - F1 Score:                  {metrics['classification']['f1_score']:.4f}  (Baseline Logistic: {metrics['classification']['baseline_f1_score']:.4f})")
    print(f"    - Brier Calibration Score:   {metrics['classification']['brier_score']:.4f}  (lower is better)")
    print(f"    - Expected Calibration Error:{metrics['classification']['expected_calibration_error']:.4f}")
    print(f"    - Accuracy:                  {metrics['classification']['accuracy']*100:.1f}%")
    print(f"  Regression (Delay Duration in Hours):")
    print(f"    - MAE:                       {metrics['regression']['mae_hours']:.2f} hrs  (Baseline Ridge: {metrics['regression']['baseline_mae_hours']:.2f} hrs)")
    print(f"    - RMSE:                      {metrics['regression']['rmse_hours']:.2f} hrs  (Baseline Ridge: {metrics['regression']['baseline_rmse_hours']:.2f} hrs)")
    print(f"    - R² Score:                  {metrics['regression']['r2_score']:.4f}  (Baseline Ridge: {metrics['regression']['baseline_r2_score']:.4f})")
    print("-" * 50)

    # 5. SHAP Explainability Engine
    print("\n[5/6] Building SHAP Explainability Engine & Global Importances...")
    explainer = MaritimeExplainer(predictor, feature_names)
    
    # Global SHAP summary plot
    shap_vals = explainer.explainer.shap_values(X_test)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_vals,
        X_test,
        feature_names=[explainer._clean_feature_name(f) for f in feature_names],
        plot_type="bar",
        max_display=12,
        show=False
    )
    plt.title("Top Global Congestion Drivers (Mean |SHAP Value|)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    shap_plot_path = os.path.join(artifacts_dir, "shap_global_summary.png")
    plt.savefig(shap_plot_path, dpi=200)
    plt.close()
    print(f"      Saved SHAP summary plot to: {shap_plot_path}")

    # 6. Save Pipeline Artifacts
    print("\n[6/6] Serializing models and pipelines...")
    predictor_path = os.path.join(model_dir, "predictor.joblib")
    preprocessor_bundle = {
        "feature_engineer": feature_engineer,
        "col_transformer": col_transformer,
        "feature_names": feature_names
    }
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    
    predictor.save(predictor_path)
    joblib.dump(preprocessor_bundle, preprocessor_path)
    print(f"      Predictor saved to: {predictor_path}")
    print(f"      Preprocessor bundle saved to: {preprocessor_path}")
    print("\nPipeline execution complete! Ready for inference and dashboard deployment.\n")


if __name__ == "__main__":
    run_pipeline()
