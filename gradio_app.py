# diabetes_gradio_app.py
import pandas as pd
import pickle
import gradio as gr

# ----------------------------
# 1. Load trained model
# ----------------------------
MODEL_PATH = "models/diabetes_severity_decision_tree.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# 2. Feature list (Synthetic dataset)
# ----------------------------
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

# ----------------------------
# 3. Prediction function
# ----------------------------
def predict_diabetes(
    age: float,
    sex: float,
    bmi: float,
    bp: float,
    s1: float,
    s2: float,
    s3: float,
    s4: float,
    s5: float,
    s6: float
):
    input_data = pd.DataFrame([[
        age, sex, bmi, bp, s1, s2, s3, s4, s5, s6
    ]], columns=FEATURES)

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "⚠️ Prediction: Severe Diabetes"
    else:
        return "✅ Prediction: Not Severe Diabetes"

# ----------------------------
# 4. Build Gradio Interface
# ----------------------------
feature_labels = {
    "age": "Age (normalized)",
    "sex": "Sex (normalized)",
    "bmi": "Body Mass Index (normalized)",
    "bp": "Blood Pressure (normalized)",
    "s1": "Total Serum Cholesterol (s1)",
    "s2": "Low-Density Lipoproteins (s2)",
    "s3": "High-Density Lipoproteins (s3)",
    "s4": "Total Cholesterol / HDL (s4)",
    "s5": "Triglycerides (s5)",
    "s6": "Blood Sugar (s6)"
}

inputs = [
    gr.Number(label=feature_labels[col], value=0.0)
    for col in FEATURES
]

demo = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Severity Prediction (Synthetic Dataset)",
    description=(
        "This demo uses a Decision Tree model trained on the "
        "Scikit-learn synthetic diabetes severity dataset. "
        "All inputs are normalized clinical features."
    )
)

# ----------------------------
# 5. Launch Gradio app
# ----------------------------
if __name__ == "__main__":
    demo.launch()
