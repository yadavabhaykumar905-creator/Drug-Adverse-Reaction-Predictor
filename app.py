import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Drug Adverse Reaction (ADR) Predictor Engine")
st.write("Patient details and Medications")

col1, col2 = st.columns(2)

with col1:
    patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=40)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    selected_condition = st.selectbox("Select Medical Condition", ["Diabetes", "Hypertension", "Heart Failure"])

with col2:
    num_medications = st.selectbox("Number of Medications", [1, 2, 3, 4], index=1)


available_drugs = ["Aspirin", "Atorvastatin", "Lisinopril", "Metformin", "Paracetamol", "Ibuprofen"]


selected_drugs = []
st.subheader("Medication Details")
for i in range(num_medications):
    drug = st.selectbox(f"Select Drug {i+1}", available_drugs, key=f"drug_{i+1}")
    selected_drugs.append(drug)

if st.button("Predict Adverse Reaction Risk"):
    
    input_data = pd.DataFrame(0, index=[0], columns=model_columns)
    
    if 'Age' in input_data.columns:
        input_data['Age'] = float(patient_age)
    if 'num_medications' in input_data.columns:
        input_data['num_medications'] = float(num_medications)
        
    gender_col = f"gender_{gender}"
    if gender_col in input_data.columns:
        input_data[gender_col] = 1.0
        
    condition_col = f"condition_{selected_condition}"
    if condition_col in input_data.columns:
        input_data[condition_col] = 1.0
        
    for idx, drug_name in enumerate(selected_drugs):
        d_col = f"drug_{idx+1}_{drug_name}"
        if d_col in input_data.columns:
            input_data[d_col] = 1.0
            
    input_data = input_data.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    if prediction[0] == 1:
        st.error(f"High Risk of Adverse Reaction Detected! (Confidence: {probability[0][1]*100:.2f}%)")
    else:
        st.success(f"Low Risk / Safe Profile. (Confidence: {probability[0][0]*100:.2f}%)")
