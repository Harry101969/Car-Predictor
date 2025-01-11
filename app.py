import streamlit as st
import pandas as pd
import pickle

# Load the cleaned data and model
@st.cache_data
def load_data():
    return pd.read_csv('Cleaned_Car_data.csv')

@st.cache_resource
def load_model():
    with open('LinearRegressionModel_Final.pkl', 'rb') as file:
        return pickle.load(file)

# Load data and model
data = load_data()
model = load_model()

# Streamlit app interface
st.title("Welcome to Car Price Predictor")
st.markdown("This app predicts the price of a car you want to sell. Try filling the details below:")

# Select the company
companies = data['company'].unique()
company = st.selectbox("Select the company:", ["Select Company"] + list(companies))

# Select the model
if company != "Select Company":
    models = data[data['company'] == company]['name'].unique()
    model_name = st.selectbox("Select the model:", ["Select Model"] + list(models))
else:
    model_name = "Select Model"

# Select year of purchase
years = sorted(data['year'].unique(), reverse=True)
year = st.selectbox("Select Year of Purchase:", years)

# Select the fuel type
fuel_types = data['fuel_type'].unique()
fuel_type = st.selectbox("Select the Fuel Type:", fuel_types)

# Enter kilometers driven
kms_driven = st.text_input("Enter the Number of Kilometres that the car has travelled:", "")

# Predict price
if st.button("Predict Price"):
    # Validate inputs
    if company == "Select Company" or model_name == "Select Model" or not kms_driven.isdigit():
        st.error("Please fill all the fields correctly!")
    else:
        # Preprocess the input
        kms_driven = int(kms_driven)
        input_data = pd.DataFrame({
            'name': [model_name],
            'company': [company],
            'year': [year],
            'kms_driven': [kms_driven],
            'fuel_type': [fuel_type]
        })

        # Make prediction
        predicted_price = model.predict(input_data)[0]
        st.success(f"The predicted price of the car is: ₹{predicted_price:,.2f}")