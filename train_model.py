# train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes
from sklearn.utils import Bunch
import pickle
import os

# ----------------------------
# 1. Load processed synthetic data
# ----------------------------
# Create synthetic classification dataset from sklearn diabetes data
diabetes_data = Bunch(
    data=load_diabetes().data,
    target=(load_diabetes().target > 130).astype(int),
    feature_names=load_diabetes().feature_names,
    target_names=['Not Severe (0)', 'Severe (1)']
)

# Feature matrix and target vector
X = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
y = diabetes_data.target

FEATURES = diabetes_data.feature_names

# ----------------------------
# 2. Train Decision Tree Model
# ----------------------------
# Best regularized configuration (from experiments)
max_depth = 5
min_samples_split = 10
min_samples_leaf = 5

model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)

# Train on full dataset (acceptable for deployment demo)
model.fit(X, y)

# ----------------------------
# 3. Save trained model
# ----------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_severity_decision_tree.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Trained Decision Tree model saved at: {MODEL_PATH}")