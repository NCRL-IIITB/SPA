import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

#!/usr/bin/env python3
"""XGBoost classifier for attack detection on combined dataset."""

import matplotlib.pyplot as plt

# Load data
file_path = "Combined-Dataset.xlsx"
data = pd.read_excel(file_path)

# Features and target
X = data.drop(columns=["isAttacked"])
y = data["isAttacked"]

# Train/test split (stratified for class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# XGBoost classifier
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    random_state=42
)

# Train
print("Training XGBoost model...")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot top feature importances
xgb.plot_importance(model, importance_type="weight", max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.show()

print("\nTraining complete. Model ready.")
