# train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import os

# ----------------------------
# 1. Load processed data
# ----------------------------
PROCESSED_DATA_PATH = "data/diabetes_processed.csv"
df = pd.read_csv(PROCESSED_DATA_PATH)

TARGET_COLUMN = "Outcome"
FEATURES = [col for col in df.columns if col != TARGET_COLUMN]

X = df[FEATURES]
y = df[TARGET_COLUMN]

# ----------------------------
# 2. Train Decision Tree Model
# ----------------------------
# Best regularized configuration
max_depth = 5
min_samples_split = 10
min_samples_leaf = 5

model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=42
)

model.fit(X, y)

# ----------------------------
# 3. Save trained model
# ----------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Trained Decision Tree saved at {MODEL_PATH}")
