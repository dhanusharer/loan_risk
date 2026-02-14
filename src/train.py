# ==========================================
# src/train.py
# Loan Default Prediction Training Script
# ==========================================

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from src.preprocessing import feature_engineering


# ==========================================
# 1. Load Dataset
# ==========================================

print("📂 Loading dataset...")
df = pd.read_csv("data/loan_data.csv")

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

print("✅ Dataset loaded successfully")

# ==========================================
# 2. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Data split completed")

# ==========================================
# 3. Preprocessing Pipeline
# ==========================================

feature_transformer = FunctionTransformer(feature_engineering)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ==========================================
# 4. Model Definition
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("feature_engineering", feature_transformer),
    ("preprocessing", preprocessor),
    ("model", model)
])

# ==========================================
# 5. Train Model
# ==========================================

print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# ==========================================
# 6. Evaluation
# ==========================================

y_prob = pipeline.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_prob)

print("\n📊 Model Evaluation")
print("ROC-AUC Score:", round(roc, 4))

# ==========================================
# 7. Cross Validation
# ==========================================

print("\n🔁 Performing Cross-Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("Fold ROC-AUC Scores:", cv_scores)
print("Mean CV ROC-AUC:", round(cv_scores.mean(), 4))
print("Std Dev:", round(cv_scores.std(), 4))

# ==========================================
# 8. Threshold Optimization
# ==========================================

print("\n🎯 Optimizing Threshold...")

thresholds = np.arange(0.1, 0.91, 0.05)
best_f1 = 0
best_threshold = 0.5

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("Best Threshold:", round(best_threshold, 2))
print("Best F1 Score:", round(best_f1, 4))

# ==========================================
# 9. Save Model & Threshold
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(pipeline, "models/model_v1.pkl")

with open("models/threshold.json", "w") as f:
    json.dump({"best_threshold": float(best_threshold)}, f)

print("\n✅ Model saved to models/model_v1.pkl")
print("✅ Threshold saved to models/threshold.json")

print("\n🎉 Training Completed Successfully!")
