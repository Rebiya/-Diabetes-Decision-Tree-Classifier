# diabetes_gradio_app.py
import pandas as pd
import pickle
import gradio as gr

# ----------------------------
# 1. Load trained model
# ----------------------------
MODEL_PATH = "models/diabetes_model.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ----------------------------
# 2. Feature list
# ----------------------------
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ----------------------------
# 3. Prediction function
# ----------------------------
def predict_diabetes(
    Pregnancies: float,
    Glucose: float,
    BloodPressure: float,
    SkinThickness: float,
    Insulin: float,
    BMI: float,
    DiabetesPedigreeFunction: float,
    Age: float
):
    input_data = pd.DataFrame([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]], columns=FEATURES)

    pred = model.predict(input_data)[0]

    if pred == 1:
        return "⚠️ The model predicts DIABETES."
    else:
        return "✅ The model predicts NO DIABETES."

# ----------------------------
# 4. Build Gradio Interface
# ----------------------------
feature_labels = {
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Glucose Level",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin Level",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age"
}

inputs = [gr.Number(label=feature_labels[col], value=0) for col in FEATURES]

demo = gr.Interface(
    fn=predict_diabetes,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction",
    description="Enter patient health metrics to predict diabetes outcome."
)

# ----------------------------
# 5. Launch Gradio app
# ----------------------------
if __name__ == "__main__":
    demo.launch()
