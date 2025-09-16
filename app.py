import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
parkinsons_data = pd.read_csv("parkinsons_disease_data.csv")

# Prepare data
X = parkinsons_data.drop(columns=['PatientID', 'DoctorInCharge'], axis=1)
y = parkinsons_data['Diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Scaler saved as scaler.pkl")

# Save the model
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as svm_model.pkl")
