# FairCV-Match 🧠📄

FairCV-Match is a machine learning project that demonstrates an end-to-end CV screening pipeline with an emphasis on **fairness awareness** and **model comparison**.

The project showcases how automated decision systems can be evaluated not only for performance, but also for potential bias.

> ⚠️ This project uses **synthetic data** and a **simulated sensitive attribute** for educational purposes.

---

## 🚀 Project Overview

Automated CV screening systems can unintentionally amplify bias if fairness is not considered during model evaluation.

This project demonstrates:
- Data loading and preprocessing
- Train/test splitting
- Training multiple ML models
- Model comparison
- Basic fairness metric evaluation
- Visualization of predictions

---

## 📂 Project Structure
faircv-match/
│
├── data/
│ ├── sample_data.csv
│ └── data_plot.png
│
├── src/
│ └── model.py
│
├── venv/
├── README.md
└── requirements.txt

---

## 📊 Dataset

The dataset is a synthetic CSV file with numeric features and a binary label.

Example:
```csv
feature1,feature2,label
1,2,0
3,4,1
5,6,1

All columns except the last are treated as features

The last column is treated as the label

🧠 Models Used

Two models are trained and compared:

Model	Purpose
Logistic Regression	Baseline linear classifier
Decision Tree	Non-linear comparison model

Both models are evaluated using accuracy and fairness metrics.

⚖️ Fairness Evaluation

Because no demographic data is available, the project simulates a sensitive attribute using a proxy:

feature2 is split by its median

Values are grouped into:

Group 0 (lower values)

Group 1 (higher values)

Fairness Metric Used

Demographic Parity Difference

This measures the absolute difference in positive prediction rates between groups.

Lower values indicate fairer outcomes.

📈 Visualization

The script generates a plot showing:

Training samples

Correct test predictions

Incorrect test predictions

Saved automatically to:

data/data_plot.png

▶️ How to Run
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/faircv-match.git
cd faircv-match

2️⃣ Create & activate virtual environment
python -m venv venv
source venv/Scripts/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the model
python src/model.py


Example output:

Logistic Regression Accuracy: 1.0
Decision Tree Accuracy: 0.8

Fairness Metrics (Demographic Parity Difference):
Logistic Regression: 0.000
Decision Tree: 0.200

Plot saved as 'data/data_plot.png'

🛠️ Future Improvements

Add real demographic attributes

Use additional fairness metrics (equal opportunity, equalized odds)

Hyperparameter tuning

Model explainability (SHAP / feature importance)

👩🏽‍💻 Author

Built as a portfolio and learning project focused on responsible AI and fairness-aware machine learning.


---

## 🧠 Why this is now STRONG

You can now say:
- “I compared two ML models”
- “I evaluated fairness, not just accuracy”
- “I understand demographic parity”
- “I can explain trade-offs between performance and fairness”

That’s **intermediate → advanced** level ML thinking.

---

## 🚀 Final steps (do these now)

```bash
pip freeze > requirements.txt
git add .
git commit -m "Add fairness metrics and model comparison"
git push