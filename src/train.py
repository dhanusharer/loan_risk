

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.preprocessing import feature_engineering




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



feature_transformer = FunctionTransformer(feature_engineering)




numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)



from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
}

results = {}

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
    
    print(f"{name} ROC-AUC: {roc_auc:.4f}\n")


print("===== FINAL COMPARISON =====")
for model_name, score in results.items():
    print(f"{model_name}: {score:.4f}")




os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/model_v1.pkl")

print("Model saved to models/model_v1.pkl ✅")



print("\n===== MODEL EVALUATION =====\n")

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


print("\n===== THRESHOLD ANALYSIS =====\n")

thresholds = np.arange(0.1, 0.91, 0.05)

results = []

for t in thresholds:
    y_pred_threshold = (y_prob >= t).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    results.append((t, precision, recall, f1))

    print(f"Threshold: {t:.2f} | "
          f"Precision: {precision:.3f} | "
          f"Recall: {recall:.3f} | "
          f"F1: {f1:.3f}")




best = max(results, key=lambda x: x[3])  

print("\n===== BEST THRESHOLD (Based on F1) =====")
print(f"Best Threshold: {best[0]:.2f}")
print(f"Precision: {best[1]:.3f}")
print(f"Recall: {best[2]:.3f}")
print(f"F1 Score: {best[3]:.3f}")

print("\nTraining pipeline finished successfully 🚀")
