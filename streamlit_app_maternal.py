import streamlit as st
import pandas as pd
import joblib
import pickle
import os

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

model_maternal, le_maternal, feature_columns_maternal, original_categorical_data_maternal = load_maternal_model_artifacts()

# --- Streamlit App Layout ---
st.title('Maternal Adverse Outcome Prediction App')
st.write('Enter the patient details to predict maternal adverse outcome.')

# --- User Inputs ---
st.sidebar.header('Patient Input Features')

def user_input_features_maternal():
    bmi = st.sidebar.slider('BMI', 15.0, 40.0, 25.0, step=0.1)
    weight_first_ANC = st.sidebar.slider('Weight at first ANC (kg)', 40.0, 120.0, 65.0, step=0.5)
    gdm_status = st.sidebar.selectbox('GDM Status', original_categorical_data_maternal['GDM_status'])
    residence = st.sidebar.selectbox('Residence', original_categorical_data_maternal['Residence'])
    muac = st.sidebar.slider('MUAC (cm)', 15.0, 40.0, 25.0, step=0.1)
    number_children = st.sidebar.slider('Number of Children', 0, 10, 1)
    education = st.sidebar.selectbox('Education', original_categorical_data_maternal['Education'])
    history_abortion = st.sidebar.selectbox('History of Abortion', original_categorical_data_maternal['History_Abortion'])

    data = {
        'BMI': bmi,
        'weight_first_ANC': weight_first_ANC,
        'GDM_status': gdm_status,
        'Residence': residence,
        'MUAC': muac,
        'Number_Children': number_children,
        'Education': education,
        'History_Abortion': history_abortion,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df_maternal = user_input_features_maternal()

st.subheader('User Input Features')
st.write(input_df_maternal)

# --- Data Preprocessing for Prediction ---
processed_input_maternal = pd.DataFrame(0, index=[0], columns=feature_columns_maternal)

# Fill numerical features
for col in ['BMI', 'weight_first_ANC', 'MUAC', 'Number_Children']:
    if col in processed_input_maternal.columns:
        processed_input_maternal[col] = input_df_maternal[col].values[0]

# Handle categorical features using one-hot encoding
for col in original_categorical_data_maternal.keys():
    if col in input_df_maternal.columns:
        val = input_df_maternal[col].values[0]
        # Create dummy column name, e.g., 'Residence_Urban'
        dummy_col = f"{col}_{val}"
        if dummy_col in processed_input_maternal.columns:
            processed_input_maternal[dummy_col] = 1

# Ensure the order of columns matches the training data
processed_input_maternal = processed_input_maternal[feature_columns_maternal]

# --- Prediction ---
prediction_maternal = model_maternal.predict(processed_input_maternal)
prediction_proba_maternal = model_maternal.predict_proba(processed_input_maternal)

st.subheader('Prediction')
predicted_class_maternal = le_maternal.inverse_transform(prediction_maternal)[0]
st.write(predicted_class_maternal)

st.subheader('Prediction Probability')
probability_df_maternal = pd.DataFrame(prediction_proba_maternal, columns=le_maternal.classes_)
st.write(probability_df_maternal)

st.markdown("""
---
**Note:** This app predicts the likelihood of an adverse outcome based on the provided maternal inputs.
""")
