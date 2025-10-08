import streamlit as st
import pandas as pd
import joblib

# =====================
# PAGE CONFIGURATION
# =====================
st.set_page_config(page_title="🧠 Brain Tumor Prediction", page_icon="🧬", layout="centered")

# =====================
# HEADER
# =====================
st.title("🧠 Brain Tumor Classification App")
st.markdown("""
This app predicts whether a brain tumor is **Benign** or **Malignant** based on patient details.
""")

# Image
st.image("tumour.png", caption="Benign vs Malignant Tumor", use_container_width=True)

# =====================
# LOAD MODEL
# =====================
model = joblib.load("models/XGBoost.pkl")

# =====================
# USER INPUT SECTION
# =====================
st.header("🔍 Enter Patient Details")

# Input mode toggle
input_mode = st.radio("Select Input Mode:", ["Slider", "Manual Entry"], horizontal=True)
st.markdown("---")

st.subheader("🧩 Tumor and Patient Parameters")

# Numeric inputs
if input_mode == "Slider":
    age = st.slider("Age", 1, 100, 45)
    st.caption("Age of the patient (in years) — typically 1–100.")

    tumor_size = st.slider("Tumor Size (cm)", 0.1, 10.0, 3.2)
    st.caption("Approximate tumor diameter (0.1–10 cm).")

    survival_rate = st.slider("Survival Rate (%)", 0, 100, 75)
    st.caption("Predicted survival rate percentage (0–100%).")

    tumor_growth = st.slider("Tumor Growth Rate", 0.1, 5.0, 1.2)
    st.caption("Tumor growth rate index (0.1–5.0) indicating speed of growth.")
else:
    age = st.number_input("Age", 1, 100, 45)
    st.caption("Age of the patient (in years) — typically 1–100.")

    tumor_size = st.number_input("Tumor Size (cm)", 0.1, 10.0, 3.2)
    st.caption("Approximate tumor diameter (0.1–10 cm).")

    survival_rate = st.number_input("Survival Rate (%)", 0, 100, 75)
    st.caption("Predicted survival rate percentage (0–100%).")

    tumor_growth = st.number_input("Tumor Growth Rate", 0.1, 5.0, 1.2)
    st.caption("Tumor growth rate index (0.1–5.0) indicating speed of growth.")

st.markdown("---")

# Categorical inputs: Treatments
st.subheader("⚕️ Clinical and Treatment Information")
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    st.caption("Patient’s biological gender.")
with col2:
    radiation = st.selectbox("Radiation Treatment", ["Yes", "No"])
    st.caption("Has the patient undergone radiation therapy?")
with col3:
    surgery = st.selectbox("Surgery Performed", ["Yes", "No"])
    st.caption("Indicate whether surgery was performed.")

col4, col5, col6 = st.columns(3)
with col4:
    chemo = st.selectbox("Chemotherapy", ["Yes", "No"])
    st.caption("Has the patient received chemotherapy?")
with col5:
    family_history = st.selectbox("Family History", ["Yes", "No"])
    st.caption("Does the patient have a family history of brain tumors?")
with col6:
    follow_up = st.selectbox("Follow Up Required", ["Yes", "No"])
    st.caption("Is follow-up or additional monitoring required?")

st.markdown("---")

# Tumor Characteristics
st.subheader("🏥 Tumor Characteristics and Symptoms")
location = st.selectbox("Tumor Location", ["Occipital", "Parietal", "Temporal"])
st.caption("Select the anatomical location of the tumor in the brain.")

histology = st.selectbox("Tumor Histology", ["Glioblastoma", "Medulloblastoma", "Meningioma"])
st.caption("Select tumor type based on histological examination.")

stage = st.selectbox("Tumor Stage", ["I", "II", "III", "IV"])
st.caption("Select tumor stage (I–IV) indicating progression.")

# Symptoms
st.subheader("💡 Symptoms Information")
symptom1 = st.selectbox("Symptom 1", ["Nausea", "Seizures", "Vision Issues"])
st.caption("Primary symptom reported by the patient.")

symptom2 = st.selectbox("Symptom 2", ["Nausea", "Seizures", "Vision Issues"])
st.caption("Secondary symptom reported by the patient.")

symptom3 = st.selectbox("Symptom 3", ["Nausea", "Seizures", "Vision Issues"])
st.caption("Tertiary symptom reported by the patient.")

# MRI Result
mri_result = st.selectbox("MRI Result", ["Positive", "Negative"])
st.caption("MRI scan outcome: presence (Positive) or absence (Negative) of tumor.")

# =====================
# DATA PREPARATION
# =====================
data = {
    "Age": age,
    "Tumor_Size": tumor_size,
    "Survival_Rate": survival_rate / 100,
    "Tumor_Growth_Rate": tumor_growth,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Radiation_Treatment_Yes": 1 if radiation == "Yes" else 0,
    "Surgery_Performed_Yes": 1 if surgery == "Yes" else 0,
    "Chemotherapy_Yes": 1 if chemo == "Yes" else 0,
    "Family_History_Yes": 1 if family_history == "Yes" else 0,
    "Follow_Up_Required_Yes": 1 if follow_up == "Yes" else 0,
    # Location
    "Location_Occipital": 1 if location=="Occipital" else 0,
    "Location_Parietal": 1 if location=="Parietal" else 0,
    "Location_Temporal": 1 if location=="Temporal" else 0,
    # Histology
    "Histology_Glioblastoma": 1 if histology=="Glioblastoma" else 0,
    "Histology_Medulloblastoma": 1 if histology=="Medulloblastoma" else 0,
    "Histology_Meningioma": 1 if histology=="Meningioma" else 0,
    # Stage
    "Stage_II": 1 if stage=="II" else 0,
    "Stage_III": 1 if stage=="III" else 0,
    "Stage_IV": 1 if stage=="IV" else 0,
    # Symptoms
    "Symptom_1_Nausea": 1 if symptom1=="Nausea" else 0,
    "Symptom_1_Seizures": 1 if symptom1=="Seizures" else 0,
    "Symptom_1_Vision Issues": 1 if symptom1=="Vision Issues" else 0,
    "Symptom_2_Nausea": 1 if symptom2=="Nausea" else 0,
    "Symptom_2_Seizures": 1 if symptom2=="Seizures" else 0,
    "Symptom_2_Vision Issues": 1 if symptom2=="Vision Issues" else 0,
    "Symptom_3_Nausea": 1 if symptom3=="Nausea" else 0,
    "Symptom_3_Seizures": 1 if symptom3=="Seizures" else 0,
    "Symptom_3_Vision Issues": 1 if symptom3=="Vision Issues" else 0,
    # MRI Result
    "MRI_Result_Positive": 1 if mri_result=="Positive" else 0
}

input_df = pd.DataFrame([data])

expected_features = [
    'Age', 'Tumor_Size', 'Survival_Rate', 'Tumor_Growth_Rate', 'Gender_Male',
    'Location_Occipital', 'Location_Parietal', 'Location_Temporal',
    'Histology_Glioblastoma', 'Histology_Medulloblastoma', 'Histology_Meningioma',
    'Stage_II', 'Stage_III', 'Stage_IV',
    'Symptom_1_Nausea', 'Symptom_1_Seizures', 'Symptom_1_Vision Issues',
    'Symptom_2_Nausea', 'Symptom_2_Seizures', 'Symptom_2_Vision Issues',
    'Symptom_3_Nausea', 'Symptom_3_Seizures', 'Symptom_3_Vision Issues',
    'Radiation_Treatment_Yes', 'Surgery_Performed_Yes', 'Chemotherapy_Yes',
    'Family_History_Yes', 'MRI_Result_Positive', 'Follow_Up_Required_Yes'
]

for col in expected_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_features]

# =====================
# PREDICTION
# =====================
if st.button("🧩 Predict Tumor Type", key="predict_button"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.markdown(
            '<div style="background-color:#ff4d4d;padding:20px;border-radius:10px">'
            '<h3 style="color:white">🧠 Predicted Tumor Type: 🟥 Malignant</h3>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background-color:#4CAF50;padding:20px;border-radius:10px">'
            '<h3 style="color:white">🧠 Predicted Tumor Type: 🟩 Benign</h3>'
            '</div>',
            unsafe_allow_html=True
        )
        st.balloons()

    st.markdown("### Note: This prediction is based on the provided data and should be confirmed by a medical professional.")