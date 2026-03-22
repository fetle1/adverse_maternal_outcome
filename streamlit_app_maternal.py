import streamlit as st
import pandas as pd
import joblib
import pickle

# --- Page Config ---
st.set_page_config(
    page_title="Maternal Outcome Predictor",
    page_icon="🤰",
    layout="wide"
)
# --- Custom Background Color ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E6F2FF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Load Model Artifacts ---
@st.cache_resource
def load_maternal_model_artifacts():
    model = joblib.load('Model_artifacts/random_forest_model_maternal.joblib')
    with open('Model_artifacts/label_encoder_maternal.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('Model_artifacts/feature_columns_maternal.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    with open('Model_artifacts/original_categorical_data_maternal.pkl', 'rb') as f:
        original_categorical_data = pickle.load(f)
    return model, le, feature_columns, original_categorical_data

model, le, feature_columns, original_categorical_data = load_maternal_model_artifacts()

# --- Title ---
st.title("🤰 Maternal Adverse Outcome Prediction Tool")
st.markdown("This tool estimates the risk of adverse maternal outcomes using antenatal care data.")

# --- Input Sections ---
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic & Obstetric")
    residence = st.selectbox("Residence", original_categorical_data['Residence'])
    education = st.selectbox("Education Level", original_categorical_data['Education'])
    number_children = st.slider("Number of Children", 0, 10, 1)
    history_abortion = st.selectbox("History of Abortion", original_categorical_data['History_Abortion'])

with col2:
    st.subheader("Clinical & Nutritional")
    bmi = st.slider("Body Mass Index (BMI)", 15.0, 40.0, 25.0, step=0.1)
    weight_first_ANC = st.slider("Weight at First ANC Visit (kg)", 40.0, 120.0, 65.0, step=0.5)
    muac = st.slider("MUAC (cm)", 15.0, 40.0, 25.0, step=0.1)
    gdm_status = st.selectbox("Gestational Diabetes (GDM)", original_categorical_data['GDM_status'])

# --- Create DataFrame ---
input_df = pd.DataFrame({
    'BMI': [bmi],
    'weight_first_ANC': [weight_first_ANC],
    'GDM_status': [gdm_status],
    'Residence': [residence],
    'MUAC': [muac],
    'Number_Children': [number_children],
    'Education': [education],
    'History_Abortion': [history_abortion],
})

st.subheader("Input Summary")
st.dataframe(input_df)

# --- Preprocessing ---
processed_input = pd.DataFrame(0, index=[0], columns=feature_columns)

# Numerical
for col in ['BMI', 'weight_first_ANC', 'MUAC', 'Number_Children']:
    if col in processed_input.columns:
        processed_input[col] = input_df[col].values[0]

# One-hot encoding
for col in original_categorical_data.keys():
    val = input_df[col].values[0]
    dummy_col = f"{col}_{val}"
    if dummy_col in processed_input.columns:
        processed_input[dummy_col] = 1

processed_input = processed_input[feature_columns]

# --- Prediction Button ---
if st.button("🔍 Predict Outcome"):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    predicted_class = le.inverse_transform(prediction)[0]
    probability_df = pd.DataFrame(prediction_proba, columns=le.classes_)

    st.subheader("Prediction Result")

    # Highlight result
    if "adverse" in predicted_class.lower():
        st.error(f"⚠️ High Risk: {predicted_class}")
    else:
        st.success(f"✅ Low Risk: {predicted_class}")

    st.subheader("Prediction Probabilities")
    st.dataframe(probability_df)

# --- Footer ---
st.markdown("""
---
**Disclaimer:** This tool is for research and decision-support purposes only and should not replace clinical judgment.
""")
