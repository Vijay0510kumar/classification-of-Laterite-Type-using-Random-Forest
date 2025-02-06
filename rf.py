app_code = """
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
dataset = pd.read_csv("dataset.csv")  # Ensure you have the dataset in the same directory

# Encode the categorical Laterite Type into numerical labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset["Laterite Type Encoded"] = label_encoder.fit_transform(dataset["Laterite Type"])

# Features for training
features = ["Ds", "UCS", "IS50", "TS", "Pw", "Di", "Mc", "RQD"]
X = dataset[features]
y = dataset["Laterite Type Encoded"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediction function
def predict_laterite_type(user_input):
    user_scaled = scaler.transform([user_input])
    predicted_label = rf_model.predict(user_scaled)[0]
    predicted_laterite_type = label_encoder.inverse_transform([predicted_label])[0]
    return predicted_laterite_type

# Streamlit UI
st.title("Laterite Type Prediction")
st.write("Enter the 8 input properties:")

# Input fields for properties
user_input = []
for feature in features:
    user_input.append(st.number_input(f"{feature}: "))

# Button to make a prediction
if st.button("Predict Laterite Type"):
    result = predict_laterite_type(user_input)
    st.write(f"Predicted Laterite Type: {result}")
"""
# Save the code to an app.py file
with open('app.py', 'w') as file:
    file.write(app_code)

print("Streamlit app code written to app.py.")
