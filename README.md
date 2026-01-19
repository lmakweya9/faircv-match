# ⚖️ FairCV-Match

**FairCV-Match** is a command-line machine learning tool designed for CV screening. It evaluates candidates against a dataset of profiles while balancing high-performance **accuracy** with algorithmic **fairness**. 

This project demonstrates a complete end-to-end ML workflow, featuring candidate scoring, model comparison, and bias-aware evaluation metrics.

---

## 🚀 Features

* **Dual Model Support:** Train and evaluate both **Logistic Regression** and **Decision Tree** models.
* **Performance Metrics:** Real-time accuracy tracking and model comparison.
* **Fairness-Aware Evaluation:** Includes built-in measurement for:
    * **Demographic Parity Difference (DPD):** Measures if the model favors one group over another.
    * **Equal Opportunity Difference (EOD):** Ensures qualified candidates are treated equally across demographics.
* **Interactive CLI:** Evaluate individual candidates on the fly via command-line arguments.
* **Data Visualization:** Automatically generates distribution plots of model predictions.

---

## 🛠️ Installation

1.  **Clone the repository:**
    git clone <your-repo-url>
    cd faircv-match

2.  **Create and activate a virtual environment:**
    # Create environment
    python -m venv venv

    # Activate (Windows PowerShell / Git Bash)
    source venv/Scripts/activate 

3.  **Install dependencies:**
    pip install -r requirements.txt

---

## 📂 Project Structure

```
faircv-match/
├── data/
│   ├── sample_data.csv    # Dataset of candidate features and labels
│   └── data_plot.png      # Auto-generated visualization of results
├── src/
│   └── model.py           # Main ML pipeline (training, testing, metrics)
├── README.md
└── requirements.txt
```

## ⚡ Usage
Run the Default Model
Trains the logistic regression model and displays performance/fairness metrics:
python src/model.py
Select a Specific Model
Switch between models using the --model flag (logistic or tree)
python src/model.py --model tree
Evaluate a Single Candidate
Predict a recommendation for a specific profile (Format: [feature1] [feature2])
python src/model.py --candidate 5.5 6.2 --model logistic
Example Output:
PlaintextModel Used: logistic
Model Accuracy: 0.95
Demographic Parity Difference: 0.20
Equal Opportunity Difference: 0.15

## 🧰 Technologies Used:
Language: Python 3.x
Data Science: Pandas, NumPy, scikit-learn
Visualization: Matplotlib

## 🔮 Future Improvements
[ ] Batch Processing: Evaluate entire folders of CSVs at once.
[ ] Streamlit UI: Build a web-based dashboard for interactive model testing.
[ ] Model Bias Mitigation: Implement "Fairlearn" techniques to actively reduce DPD and EOD scores.
