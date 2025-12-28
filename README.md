# Diabetes Severity Prediction using Decision Tree

## Project Overview

This project demonstrates a **machine learning workflow** to predict the **severity of diabetes** in patients using a **synthetic diabetes dataset**. The dataset contains patient health metrics such as age, BMI, blood pressure, and other lab values (s1–s6). The goal is to classify patients as:

* **Severe (1)**
* **Not Severe (0)**

The model used is a **Decision Tree classifier**, chosen for its interpretability and suitability for tabular healthcare data.

---

## Dataset

* **Source:** Synthetic dataset based on Scikit-learn Diabetes dataset.
* **Columns:**
  `age, bmi, bp, s1, s2, s3, s4, s5, s6, target`
* **Target:** `target` (binary, 0 = Not Severe, 1 = Severe)
* **Samples:** 442 (based on original Scikit-learn diabetes dataset size)
* **Description:**
  The dataset contains normalized numeric features representing patient health measurements. No categorical encoding is required.

---

## Project Structure

```
Diabetes-Decision-Tree-Classifier/
├─ data/
│  └─ diabetes_severity.csv          # Synthetic dataset
├─ models/
│  └─ diabetes_severity_decision_tree.pkl  # Trained model
├─ train_model.py                    # Training script
├─ gradio_app.py                     # Deployment app
├─ README.md                         # Project documentation
└─ notebooks/
   └─ 01_data_exploration.ipynb     # exploratory analysis notebook
   └─ 02_baseline_model.ipynb       # model training notebook
├─requirements.txt
├─README.md
├─results/
├─report/
├─.gradio/                          #saves output of the gradio user input and prediction 
```

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd Diabetes-Decision-Tree-Classifier
```

2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

**Dependencies include:**

* pandas
* scikit-learn
* gradio
* matplotlib (optional, for visualization)

---

## Usage

### 1. Train the Model

```bash
python train_model.py
```

* Trains a **Decision Tree classifier** using the synthetic dataset.
* Saves the trained model to `models/diabetes_severity_decision_tree.pkl`.

### 2. Run the Gradio App

```bash
python gradio_app.py
```

* Launches a web interface to input patient metrics.
* Predicts whether the patient has **severe diabetes**.
* Access locally at: `http://127.0.0.1:7860`

---

## Model Details

* **Algorithm:** Decision Tree Classifier
* **Hyperparameters:**

  * `max_depth = 5`
  * `min_samples_split = 10`
  * `min_samples_leaf = 5`
* **Evaluation Metrics:**

  * **Train Accuracy:** 0.801
  * **Test Accuracy:** 0.773
  * **F1 Score:** 0.701
* The model was trained on all features: `age, bmi, bp, s1, s2, s3, s4, s5, s6`.

---

## Notes

* The synthetic dataset has **different column names** than the original Kaggle dataset; ensure consistency when training or deploying.
* The Gradio app **expects feature names** exactly as in the CSV (`age, bmi, bp, s1, s2, s3, s4, s5, s6`).

---

## References

* Scikit-learn Diabetes dataset: [https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
* Decision Trees: [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)


