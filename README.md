# FairCV-Match

**FairCV-Match** is a command-line machine learning tool for CV screening that evaluates candidates against a dataset of profiles while considering both **accuracy** and **fairness**.  
This project demonstrates end-to-end ML workflow, candidate scoring, model comparison, and fairness-aware evaluation.

---

## 🚀 Features

- Train and evaluate two machine learning models:
  - **Logistic Regression**
  - **Decision Tree**
- Measure **model performance**:
  - Accuracy
- Measure **fairness metrics**:
  - **Demographic Parity Difference (DPD)** – difference in positive prediction rates across demographic groups
  - **Equal Opportunity Difference (EOD)** – difference in true positive rates across groups
- Evaluate **individual candidates** via CLI
- Generate **visualization of train/test predictions**

---

## 🛠️ Installation

1. Clone the repository:
git clone <your-repo-url>
cd faircv-match


Create and activate a virtual environment:
python -m venv venv
source venv/Scripts/activate  # Windows PowerShell / Git Bash


Install dependencies:
pip install -r requirements.txt

📂 Project Structure
faircv-match/
├─ data/
│  ├─ sample_data.csv
├─ src/
│  ├─ model.py
├─ README.md
├─ requirements.txt


sample_data.csv – example dataset of candidate features and labels

model.py – main ML pipeline

⚡ Usage
Run the default model
python src/model.py

Select a specific model
python src/model.py --model tree

Evaluate a single candidate
python src/model.py --candidate 5.5 6.2 --model logistic


Example output:

Model Used: logistic
Model Accuracy: 0.95
Demographic Parity Difference: 0.20
Equal Opportunity Difference: 0.15

--- Candidate Evaluation ---
Match Score: 87%
Decision: RECOMMEND

Output

Candidate score (0–100%)

Decision: RECOMMEND or REJECT

Fairness metrics (DPD and EOD)

Plot saved at data/data_plot.png

📊 Fairness Metrics Explained
Metric	What it Measures	Why it Matters
Demographic Parity Difference (DPD)	Difference in positive prediction rates between groups	Ensures model is not biased toward a group regardless of true qualifications
Equal Opportunity Difference (EOD)	Difference in true positive rates between groups	Ensures model is equally likely to correctly recommend qualified candidates across groups

💡 Key Takeaways
High accuracy does not guarantee fairness
FairCV-Match allows evaluation of bias alongside performance
CLI interface supports single candidate testing for demonstration

🧰 Technologies
Python 3.x
Pandas, NumPy
scikit-learn
Matplotlib

🔮 Future Improvements
Batch candidate evaluation
Fairness comparison across multiple models
Streamlit interface for interactive evaluation
Logging metrics for multiple experiments