# ==========================================
# train.py - Loan Default Prediction
# ==========================================

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier
import shap
from src.preprocessing import feature_engineering


# ==========================================
# 1. Load Data
# ==========================================

print("Loading data...")
df = pd.read_csv("data/loan_data.csv")

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data loaded successfully ✅")


# ==========================================
# 2. Feature Engineering & Preprocessing
# ==========================================

feature_transformer = FunctionTransformer(feature_engineering)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# ==========================================
# 3. Model Comparison
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric="logloss"
    )
}

results = {}
trained_pipelines = {}

print("\n===== MODEL COMPARISON =====\n")

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("feature_engineering", feature_transformer),
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    print(f"Training {name}...")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = roc_auc
    trained_pipelines[name] = pipeline

    print(f"{name} ROC-AUC: {roc_auc:.4f}\n")


print("===== FINAL COMPARISON =====")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")


# ==========================================
# 4. Select Best Model
# ==========================================

best_model_name = max(results, key=results.get)
best_pipeline = trained_pipelines[best_model_name]

print(f"\n🏆 Best Model: {best_model_name}")


# ==========================================
# 5. Cross Validation (Only Best Model)
# ==========================================

print("\n===== CROSS-VALIDATION =====\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    best_pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("Fold ROC-AUC Scores:", cv_scores)
print("Mean ROC-AUC:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())


# ==========================================
# 6. Save Best Model
# ==========================================

os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, "models/model_v1.pkl")

print("Model saved to models/model_v1.pkl ✅")


# ==========================================
# 7. Final Evaluation
# ==========================================

print("\n===== MODEL EVALUATION =====\n")

y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 8. Threshold Analysis
# ==========================================

print("\n===== THRESHOLD ANALYSIS =====\n")

thresholds = np.arange(0.1, 0.91, 0.05)
threshold_results = []

for t in thresholds:
    y_pred_threshold = (y_prob >= t).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    threshold_results.append((t, precision, recall, f1))

    print(f"Threshold: {t:.2f} | "
          f"Precision: {precision:.3f} | "
          f"Recall: {recall:.3f} | "
          f"F1: {f1:.3f}")

best_threshold = max(threshold_results, key=lambda x: x[3])

print("\n===== BEST THRESHOLD (Based on F1) =====")
print(f"Best Threshold: {best_threshold[0]:.2f}")
print(f"Precision: {best_threshold[1]:.3f}")
print(f"Recall: {best_threshold[2]:.3f}")
print(f"F1 Score: {best_threshold[3]:.3f}")

# Save threshold
with open("models/threshold.json", "w") as f:
    json.dump({"best_threshold": float(best_threshold[0])}, f)

print("Threshold saved to models/threshold.json ✅")

print("\nTraining pipeline finished successfully 🚀")
# ==========================================
# 9. SHAP Explainability
# ==========================================

print("\n===== SHAP EXPLAINABILITY =====\n")

# SHAP works best for tree models
if "XGB" in best_model_name:

    # Transform test data
    X_test_fe = best_pipeline.named_steps["feature_engineering"].transform(X_test)
    X_test_processed = best_pipeline.named_steps["preprocessing"].transform(X_test_fe)

    # Get trained XGBoost model
    model = best_pipeline.named_steps["model"]

    # Get feature names
    feature_names = best_pipeline.named_steps["preprocessing"].get_feature_names_out()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_test_processed)

    # -------- GLOBAL IMPORTANCE --------
    shap.summary_plot(
        shap_values,
        X_test_processed,
        feature_names=feature_names,
        show=False
    )

    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    plt.close()

    print("SHAP summary plot saved to models/shap_summary.png ✅")

    # -------- SINGLE PREDICTION --------
    sample_index = 0

    shap.force_plot(
        explainer.expected_value,
        shap_values[sample_index],
        X_test_processed[sample_index],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )

    plt.title("SHAP Force Plot (Single Prediction)")
    plt.tight_layout()
    plt.savefig("models/shap_single_prediction.png")
    plt.close()

    print("Single prediction SHAP plot saved to models/shap_single_prediction.png ✅")

else:
    print("SHAP skipped (Best model is not tree-based).")
