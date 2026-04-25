import streamlit as st
import pickle
import pandas as pd

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

st.title("Customer Churn Predictor")
st.write("Enter customer details to predict if they will leave.")

# Create simple inputs (Adjust these based on your CSV columns)
age = st.number_input("Age", min_value=18, max_value=100)
freq = st.slider("Frequency of Travel", 1, 10)

if st.button("Predict"):
    # Create a dataframe for prediction matching the training columns
    # (Note: You must include all columns used in training here)
    prediction = model.predict([[age, freq]]) # Simplified example
    
    if prediction[0] == 1:
        st.error("Customer is likely to Churn!")
    else:
        st.success("Customer is likely to Stay!")
