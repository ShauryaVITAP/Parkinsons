import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open('svm_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Parkinson's Disease Detection")

st.write("""
Please enter the information below to predict if the person has Parkinson's Disease.
""")

# Correct list of 33 feature names in order
feature_names = [
    "Age", "Gender (0=Female, 1=Male)", "Ethnicity (encoded)", "Education Level (years)",
    "BMI", "Smoking (0=No, 1=Yes)", "Alcohol Consumption (0=No, 1=Yes)",
    "Physical Activity (score)", "Diet Quality (score)", "Sleep Quality (score)",
    "Family History of Parkinson's (0=No, 1=Yes)", "Traumatic Brain Injury (0=No, 1=Yes)",
    "Hypertension (0=No, 1=Yes)", "Diabetes (0=No, 1=Yes)", "Depression (0=No, 1=Yes)",
    "Stroke (0=No, 1=Yes)", "Systolic BP", "Diastolic BP",
    "Cholesterol Total", "Cholesterol LDL", "Cholesterol HDL", "Cholesterol Triglycerides",
    "UPDRS", "MoCA", "Functional Assessment", "Tremor (0=No, 1=Yes)",
    "Rigidity (0=No, 1=Yes)", "Bradykinesia (0=No, 1=Yes)", "Postural Instability (0=No, 1=Yes)",
    "Speech Problems (0=No, 1=Yes)", "Sleep Disorders (0=No, 1=Yes)", "Constipation (0=No, 1=Yes)","Diagnosis(0=No,1=Yes)"
]

# Collect user inputs for all 33 features
input_data = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    input_data.append(value)

# Convert to numpy array
input_np = np.asarray(input_data).reshape(1, -1)

# Standardize the input
std_data = scaler.transform(input_np)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(std_data)
    if prediction[0] == 0:
        st.success("The person does NOT have Parkinson's Disease.")
    else:
        st.error("The person HAS Parkinson's Disease.")
