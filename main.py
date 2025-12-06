import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Health Risk Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Health Risk Prediction Portal")

st.write("Select the health condition you want to check your risk for:")

# --------------------------
# Disease Selection
# --------------------------
disease = st.selectbox(
    "Choose a prediction:",
    ["Select...", "Lung Cancer", "Stroke", "Diabetes", "Heart Disease"]
)

# --------------------------
# Load Models
# --------------------------
models = {
    "Lung Cancer": joblib.load("LungCancer_prediction_model.pkl"),
    "Stroke": joblib.load("stroke_prediction_modell.pkl"),
    "Diabetes": joblib.load("diabetes_prediction_model.pkl"),
    "Heart Disease": joblib.load("HeartDisease_prediction_model.pkl")
}

if disease == "Stroke":
    label_encoders = joblib.load("label_encoders.pkl")

# --------------------------
# Lung Cancer Form
# --------------------------
if disease == "Lung Cancer":
    st.header("🫁 Lung Cancer Risk Prediction")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=90, step=1)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    alcohol_consuming = st.selectbox("Alcohol Consuming", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    shortness_of_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
    shallowing_difficulty = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

    def preprocess_lung():
        yes_no_map = {"No": 1, "Yes": 2}
        gender_map = {"Male": "M", "Female": "F"}
        data = {
            "GENDER": [gender_map[gender]],
            "AGE": [age],
            "SMOKING": [yes_no_map[smoking]],
            "YELLOW_FINGERS": [yes_no_map[yellow_fingers]],
            "ANXIETY": [yes_no_map[anxiety]],
            "PEER_PRESSURE": [yes_no_map[peer_pressure]],
            "CHRONIC DISEASE": [yes_no_map[chronic_disease]],
            "FATIGUE ": [yes_no_map[fatigue]],
            "ALLERGY ": [yes_no_map[allergy]],
            "WHEEZING": [yes_no_map[wheezing]],
            "ALCOHOL CONSUMING": [yes_no_map[alcohol_consuming]],
            "COUGHING": [yes_no_map[coughing]],
            "SHORTNESS OF BREATH": [yes_no_map[shortness_of_breath]],
            "SWALLOWING DIFFICULTY": [yes_no_map[shallowing_difficulty]],
            "CHEST PAIN": [yes_no_map[chest_pain]],
        }
        return pd.DataFrame(data)

    if st.button("Predict Lung Cancer Risk"):
        try:
            input_df = preprocess_lung()
            proba = models["Lung Cancer"].predict_proba(input_df)[0]
            prediction = models["Lung Cancer"].predict(input_df)[0]
            prediction = 1 if proba[1] >= 0.7 else 0

            if prediction == 1:
                st.error("⚠️ High Risk of Lung Cancer! Please consult a doctor.")
            else:
                st.success("✅ Low Risk of Lung Cancer! Keep maintaining a healthy lifestyle.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --------------------------
# Stroke Form
# --------------------------
if disease == "Stroke":
    st.header("🧠 Stroke Risk Prediction")
    gender_input = st.selectbox("gender", ["Male", "Female", "Other"])
    age_input = st.number_input("age", min_value=1, max_value=90, step=1)
    hypertension_input = st.selectbox("hypertension", ["No", "Yes"])
    heart_disease_input = st.selectbox("heart Disease", ["No", "Yes"])
    ever_married_input = st.selectbox("ever_married", ["No", "Yes"])
    work_type_input = st.selectbox("work_type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    residence_input = st.selectbox("Residence_type", ["Urban", "Rural"])
    avg_glucose_input = st.number_input("Average Glucose Level", min_value=40.0, max_value=400.0, step=0.1)
    bmi_input = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    smoking_input = st.selectbox("Smoking Status", ["Unknown", "never smoked", "formerly smoked", "smokes"])

    def preprocess_stroke():
        hypertension_map = {"No": 0, "Yes": 1}
        heart_map = {"No": 0, "Yes": 1}
        data = {
            "gender": [gender_input],
            "age": [age_input],
            "hypertension": [hypertension_map[hypertension_input]],
            "heart_disease": [heart_map[heart_disease_input]],
            "ever_married": [ever_married_input],
            "work_type": [work_type_input],
            "Residence_type": [residence_input],
            "avg_glucose_level": [avg_glucose_input],
            "bmi": [bmi_input],
            "smoking_status": [smoking_input]
        }
        df = pd.DataFrame(data)
        for col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
        return df

    if st.button("Predict Stroke Risk"):
        try:
            input_df = preprocess_stroke()
            proba = models["Stroke"].predict_proba(input_df)[0]
            if proba[1] >= 0.3:
                st.error("⚠️ High Stroke Risk")
            else:
                st.success("✅ Low Stroke Risk")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --------------------------
# Diabetes Form
# --------------------------
if disease == "Diabetes":
    st.header("🩺 Diabetes Risk Prediction")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    smoking_history = st.selectbox("Smoking History", ["never", "current", "former", "not current", "ever"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, step=0.1)
    hba1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=20.0, step=0.1)
    blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400, step=1)

    def preprocess_diabetes():
        yes_no_map = {"No": 0, "Yes": 1}
        data = {
            "gender": [gender.lower()],
            "age": [age],
            "hypertension": [yes_no_map[hypertension]],
            "heart_disease": [yes_no_map[heart_disease]],
            "smoking_history": [smoking_history],
            "bmi": [bmi],
            "HbA1c_level": [hba1c_level],
            "blood_glucose_level": [blood_glucose_level],
        }
        return pd.DataFrame(data)

    if st.button("Predict Diabetes Risk"):
        try:
            input_df = preprocess_diabetes()
            proba = models["Diabetes"].predict_proba(input_df)[0]
            prediction = models["Diabetes"].predict(input_df)[0]
            if prediction == 1:
                st.error(f"⚠️ High Risk of Diabetes! (Prob: {proba[1]:.2f})")
            else:
                st.success(f"✅ Low Risk of Diabetes! (Prob: {proba[1]:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --------------------------
# Heart Disease Form
# --------------------------
if disease == "Heart Disease":
    st.header("❤️ Heart Disease Risk Prediction")
    sex = st.selectbox("Sex", ["M", "F"])
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting BP (mm Hg)", min_value=0, max_value=300, step=1)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, step=1)
    fasting_bs = st.selectbox("Fasting BS > 120 mg/dl?", ["No", "Yes"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max HR", min_value=50, max_value=220, step=1)
    exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    def preprocess_heart():
        yes_no_map = {"No": 0, "Yes": 1}
        data = {
            "Sex": [sex],
            "Age": [age],
            "ChestPainType": [chest_pain],
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [yes_no_map.get(fasting_bs, 0)],
            "RestingECG": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope]
        }
        return pd.DataFrame(data)

    if st.button("Predict Heart Disease Risk"):
        try:
            input_df = preprocess_heart()
            proba = models["Heart Disease"].predict_proba(input_df)[0]
            prediction = models["Heart Disease"].predict(input_df)[0]
            if prediction == 1:
                st.error(f"⚠️ High Risk of Heart Disease! (Prob: {proba[1]:.2f})")
            else:
                st.success(f"✅ Low Risk of Heart Disease! (Prob: {proba[1]:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --------------------------
# Disclaimer at the bottom
# --------------------------
st.markdown("""
---
⚠️ **Disclaimer:**  
This application provides **risk predictions only** based on statistical models trained on historical data.  
The reported models are not 100% accurate, which means predictions can be incorrect.  

This tool **cannot replace professional medical advice, diagnosis, or treatment**.  
If you are experiencing symptoms or health concerns, please consult a qualified healthcare provider or doctor immediately.  
Use the results here as **informational support**, not as a medical decision-making tool.
""")
