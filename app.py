import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Load Models & Scalers
def load_models():
    diabetes_model = joblib.load('models/diabetes_model.pkl')
    heart_model = joblib.load('models/heart_model.pkl')
    scaler_diabetes = joblib.load('models/scaler_diabetes.pkl')
    scaler_heart = joblib.load('models/scaler_heart.pkl')
    return diabetes_model, heart_model, scaler_diabetes, scaler_heart

diabetes_model, heart_model, scaler_diabetes, scaler_heart = load_models()

# Header
st.markdown("""
    <h1 style='text-align:center; color:#2c3e50;'>
        🏥 Multiple Disease Prediction System
    </h1>
    <p style='text-align:center; color:gray;'>
        PES1PG25CA074 | Gowtham M | SDG 3 - Good Health and Well-Being
    </p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar - Disease Selection
st.sidebar.title("🔍 Select Disease")
disease = st.sidebar.radio("", ["🩺 Diabetes Prediction", "❤️ Heart Disease Prediction"])

# ─────────────────────────────────────────
# DIABETES PREDICTION
# ─────────────────────────────────────────
if disease == "🩺 Diabetes Prediction":
    st.subheader("🩺 Diabetes Prediction")
    st.markdown("Enter the patient details below:")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=122, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=99, value=20)

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=846, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=67.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=10, max_value=100, value=30)

    if st.button("🔍 Predict Diabetes", use_container_width=True):
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler_diabetes.transform(input_data)
        prediction = diabetes_model.predict(input_scaled)[0]
        probability = diabetes_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if prediction == 1:
                st.error("🔴 RESULT: DIABETES DETECTED")
                st.metric("Confidence", f"{probability[1]*100:.1f}%")
                st.warning("⚠️ Please consult a doctor immediately!")
            else:
                st.success("🟢 RESULT: NO DIABETES")
                st.metric("Confidence", f"{probability[0]*100:.1f}%")
                st.info("✅ You are healthy! Keep it up!")

        with col_r2:
            # Pie Chart
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(probability,
                   labels=['No Diabetes', 'Diabetes'],
                   colors=['#2ecc71', '#e74c3c'],
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Prediction Probability')
            st.pyplot(fig)

        st.markdown("---")
        # Bar Chart - Input Values
        st.subheader("📊 Your Health Profile")
        features = ['Pregnancies', 'Glucose', 'Blood Pressure',
                    'Skin Thickness', 'Insulin', 'BMI', 'DPF', 'Age']
        values = [pregnancies, glucose, blood_pressure,
                  skin_thickness, insulin, bmi, dpf, age]

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        bars = ax2.bar(features, values, color='#3498db', edgecolor='black')
        ax2.set_title('Your Input Values', fontweight='bold')
        ax2.set_ylabel('Value')
        plt.xticks(rotation=30)
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     str(round(val, 2)), ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

# ─────────────────────────────────────────
# HEART DISEASE PREDICTION
# ─────────────────────────────────────────
elif disease == "❤️ Heart Disease Prediction":
    st.subheader("❤️ Heart Disease Prediction")
    st.markdown("Enter the patient details below:")

    col1, col2 = st.columns(2)

    with col1:
        age_h = st.number_input("Age", min_value=10, max_value=100, value=45)
        sex_h = st.selectbox("Sex", options=["Male", "Female"])
        sex_val = 1 if sex_h == "Male" else 0
        cp_h = st.selectbox("Chest Pain Type", options=[
            "Typical Angina", "Atypical Angina",
            "Non-Anginal Pain", "Asymptomatic"])
        cp_val = ["Typical Angina", "Atypical Angina",
                  "Non-Anginal Pain", "Asymptomatic"].index(cp_h)
        trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs_h = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
        fbs_val = 1 if fbs_h == "Yes" else 0

    with col2:
        restecg = st.selectbox("Resting ECG Results", options=[
            "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        restecg_val = ["Normal", "ST-T Wave Abnormality",
                       "Left Ventricular Hypertrophy"].index(restecg)
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", options=["Upsloping", "Flat", "Downsloping"])
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        ca = st.number_input("Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

    if st.button("🔍 Predict Heart Disease", use_container_width=True):
        input_data = np.array([[age_h, sex_val, cp_val, trestbps, chol,
                                fbs_val, restecg_val, thalach, exang_val,
                                oldpeak, slope_val, ca, thal_val]])
        input_scaled = scaler_heart.transform(input_data)
        prediction = heart_model.predict(input_scaled)[0]
        probability = heart_model.predict_proba(input_scaled)[0]

        st.markdown("---")
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if prediction == 1:
                st.error("🔴 RESULT: HEART DISEASE DETECTED")
                st.metric("Confidence", f"{probability[1]*100:.1f}%")
                st.warning("⚠️ Please consult a cardiologist!")
            else:
                st.success("🟢 RESULT: NO HEART DISEASE")
                st.metric("Confidence", f"{probability[0]*100:.1f}%")
                st.info("✅ Your heart is healthy! Keep it up!")

        with col_r2:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(probability,
                   labels=['No Disease', 'Heart Disease'],
                   colors=['#2ecc71', '#e74c3c'],
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Prediction Probability')
            st.pyplot(fig)

        st.markdown("---")
        # Bar Chart
        st.subheader("📊 Your Heart Health Profile")
        features = ['Age', 'Resting BP', 'Cholesterol', 'Max HR', 'ST Depression']
        values = [age_h, trestbps, chol, thalach, oldpeak]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        bars = ax2.bar(features, values, color='#e74c3c', edgecolor='black')
        ax2.set_title('Your Heart Input Values', fontweight='bold')
        ax2.set_ylabel('Value')
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5,
                     str(round(val, 2)), ha='center', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("""
    <p style='text-align:center; color:gray;'>
        🏥 Multiple Disease Prediction System | 
        SDG 3 – Good Health and Well-Being |
        Gowtham M | PES1PG25CA074
    </p>
""", unsafe_allow_html=True)
