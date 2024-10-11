import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf

# Load model and transformers
model = tf.keras.models.load_model("model.h5")

with open("le.pkl", "rb") as file:
    le = pickle.load(file)

with open("ohe.pkl", "rb") as file:
    ohe = pickle.load(file)

with open("scalar.pkl", "rb") as file:
    scalar = pickle.load(file)

# Streamlit UI for inputs
st.title("CUSTOMER CHURN PREDICTION")
geography = st.selectbox("Geography", ohe.categories_[0])
gender = st.selectbox("Gender", le.classes_)
age = st.slider("Age", 18, 90)
balance = st.number_input("Balance")
credit_score = st.number_input("CreditScore")
esti_salary = st.number_input("EstimatedSalary")
tenure = st.number_input("Tenure", 0, 10)
num_of_products = st.number_input("NumOfProducts", 0, 10)
has_credit_card = st.selectbox("HasCrCard", [0, 1])
is_active_member = st.selectbox("IsActiveMember", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [le.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [esti_salary],
})

# One-hot encode Geography
geo_encoded = ohe.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=ohe.get_feature_names_out(["Geography"]))

# Concatenate the one-hot encoded Geography with other inputs
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scalar.transform(input_data)

# Make a prediction
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Display the result
if prediction_proba > 0.5:
    st.write("The person is likely to churn.")
else:
    st.write("The person is likely to stay.")
