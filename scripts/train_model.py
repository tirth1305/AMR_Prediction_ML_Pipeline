# ============================================================
#  Antimicrobial Resistance Prediction - Model Training
# Author: Tirth Patel
# ============================================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# === Load Data ===
X_train = pd.read_csv("data/X_train_sel.csv")
X_test = pd.read_csv("data/X_test_sel.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

print(" Data Loaded Successfully!")
print(f"Train Samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
print(f"Test Samples: {X_test.shape[0]}")

# === Random Forest Model ===
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === STACKING ENSEMBLE ===
stack = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
y_proba_stack = stack.predict_proba(X_test)[:, 1]

print("\n Stacking Ensemble Results")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("ROC AUC:", roc_auc_score(y_test, y_proba_stack))
print("Classification Report:\n", classification_report(y_test, y_pred_stack))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_stack))
