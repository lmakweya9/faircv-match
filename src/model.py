import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'sample_data.csv')
PLOT_PATH = os.path.join(BASE_DIR, '..', 'data', 'data_plot.png')

# ---------------- Load Data ----------------
data = pd.read_csv(DATA_PATH)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Simulated sensitive attribute (proxy)
# Using feature2 median split
sensitive_attr = (X.iloc[:, 1] > X.iloc[:, 1].median()).astype(int)

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive_attr, test_size=0.2, random_state=42
)

# ---------------- Models ----------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42)
}

results = {}

# ---------------- Train & Evaluate ----------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = {
        "model": model,
        "preds": preds,
        "accuracy": acc
    }
    print(f"{name} Accuracy: {acc}")

# ---------------- Fairness Metric ----------------
def demographic_parity_difference(y_pred, sensitive):
    group_0_rate = y_pred[sensitive == 0].mean()
    group_1_rate = y_pred[sensitive == 1].mean()
    return abs(group_0_rate - group_1_rate)

print("\nFairness Metrics (Demographic Parity Difference):")
for name, res in results.items():
    dp = demographic_parity_difference(res["preds"], s_test)
    print(f"{name}: {dp:.3f}")

# ---------------- Visualization (Logistic Regression) ----------------
model = results["Logistic Regression"]["model"]
preds = results["Logistic Regression"]["preds"]

plt.figure(figsize=(6, 4))

plt.scatter(
    X_train.iloc[:, 0], X_train.iloc[:, 1],
    c=y_train, cmap='coolwarm', s=100, label='Train'
)

correct = preds == y_test
incorrect = ~correct

plt.scatter(
    X_test.iloc[:, 0][correct], X_test.iloc[:, 1][correct],
    c=y_test[correct], cmap='coolwarm', marker='^', s=150, label='Test Correct'
)

if incorrect.any():
    plt.scatter(
        X_test.iloc[:, 0][incorrect], X_test.iloc[:, 1][incorrect],
        c=y_test[incorrect], cmap='coolwarm', marker='x', s=150, label='Test Incorrect'
    )

plt.title("Logistic Regression: Train vs Test Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)

print(f"\nPlot saved as '{PLOT_PATH}'")
