import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Fair CV Match Model")

parser.add_argument(
    "--model",
    choices=["logistic", "tree"],
    default="logistic",
    help="Choose which ML model to use"
)

parser.add_argument(
    "--candidate",
    nargs=2,
    type=float,
    metavar=("feature1", "feature2"),
    help="Candidate feature values"
)

args = parser.parse_args()


# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "../data/sample_data.csv"
data = pd.read_csv(DATA_PATH)

X = data[["feature1", "feature2"]]
y = data["label"]
protected_attr = data["group"]  # e.g. demographic group


# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
    X, y, protected_attr, test_size=0.3, random_state=42
)


# -----------------------------
# Model Selection
# -----------------------------
if args.model == "logistic":
    model = LogisticRegression()
elif args.model == "tree":
    model = DecisionTreeClassifier(max_depth=4, random_state=42)

model.fit(X_train, y_train)


# -----------------------------
# Model Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Used: {args.model}")
print(f"Model Accuracy: {accuracy:.3f}")


# -----------------------------
# Fairness Metrics
# -----------------------------
def demographic_parity_difference(y_pred, group):
    groups = group.unique()
    rates = [y_pred[group == g].mean() for g in groups]
    return abs(rates[0] - rates[1])

def equal_opportunity_difference(y_true, y_pred, group):
    groups = group.unique()
    tpr_rates = []
    for g in groups:
        idx = (group == g) & (y_true == 1)  # Only positive true labels
        if idx.sum() == 0:
            tpr_rates.append(0)
        else:
            tpr_rates.append(y_pred[idx].mean())
    return abs(tpr_rates[0] - tpr_rates[1])


dp_diff = demographic_parity_difference(y_pred, group_test)
eod = equal_opportunity_difference(y_test, y_pred, group_test)

print(f"Demographic Parity Difference: {dp_diff:.3f}")
print(f"Equal Opportunity Difference: {eod:.3f}")


# -----------------------------
# Candidate Evaluation
# -----------------------------
if args.candidate:
    candidate_df = pd.DataFrame([args.candidate], columns=X.columns)
    match_score = model.predict_proba(candidate_df)[0][1]
    decision = "RECOMMEND" if match_score >= 0.5 else "REJECT"

    print("\n--- Candidate Evaluation ---")
    print(f"Match Score: {int(match_score * 100)}%")
    print(f"Decision: {decision}")


# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(8, 6))

correct = y_test == y_pred
incorrect = ~correct

plt.scatter(
    X_test["feature1"][correct],
    X_test["feature2"][correct],
    c=y_test[correct],
    cmap="coolwarm",
    label="Correct Predictions"
)

plt.scatter(
    X_test["feature1"][incorrect],
    X_test["feature2"][incorrect],
    c=y_test[incorrect],
    cmap="coolwarm",
    marker="x",
    s=150,
    label="Incorrect Predictions"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"Model Predictions ({args.model})")
plt.legend()

PLOT_PATH = "../data/data_plot.png"
plt.savefig(PLOT_PATH)
plt.close()

print(f"\nPlot saved to {PLOT_PATH}")
