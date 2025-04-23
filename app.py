# import streamlit as st
# import pandas as pd
# import pickle

# # Load the cleaned data and model
# @st.cache_data
# def load_data():
#     return pd.read_csv('Cleaned_Car_data.csv')

# @st.cache_resource
# def load_model():
#     with open('LinearRegressionModel_Final.pkl', 'rb') as file:
#         return pickle.load(file)

# # Load data and model
# data = load_data()
# model = load_model()
# # Streamlit app interface
# st.title("Welcome to Car Price Predictor")
# st.markdown("This app predicts the price of a car you want to sell. Try filling the details below:")

# # Select the company
# companies = data['company'].unique()
# company = st.selectbox("Select the company:", ["Select Company"] + list(companies))

# # Select the model
# if company != "Select Company":
#     models = data[data['company'] == company]['name'].unique()
#     model_name = st.selectbox("Select the model:", ["Select Model"] + list(models))
# else:
#     model_name = "Select Model"

# # Select year of purchase
# years = sorted(data['year'].unique(), reverse=True)
# year = st.selectbox("Select Year of Purchase:", years)

# # Select the fuel type
# fuel_types = data['fuel_type'].unique()
# fuel_type = st.selectbox("Select the Fuel Type:", fuel_types)

# # Enter kilometers driven
# kms_driven = st.text_input("Enter the Number of Kilometres that the car has travelled:", "")

# # Predict price
# if st.button("Predict Price"):
#     # Validate inputs
#     if company == "Select Company" or model_name == "Select Model" or not kms_driven.isdigit():
#         st.error("Please fill all the fields correctly!")
#     else:
#         # Preprocess the input
#         kms_driven = int(kms_driven)
#         input_data = pd.DataFrame({
#             'name': [model_name],
#             'company': [company],
#             'year': [year],
#             'kms_driven': [kms_driven],
#             'fuel_type': [fuel_type]
#         })

#         # Make prediction
#         predicted_price = model.predict(input_data)[0]
#         st.success(f"The predicted price of the car is: ₹{predicted_price:,.2f}")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Load the cleaned data and model
@st.cache_data
def load_data():
    return pd.read_csv('Cleaned_Car_data.csv')

@st.cache_resource
def load_model():
    with open('LinearRegressionModel_Final.pkl', 'rb') as file:
        return pickle.load(file)

# Calculate car age
def extract_car_age(year):
    return 2025 - year  # Current year minus car year

# Function to preprocess input data
def preprocess_input(company, model_name, year, kms_driven, fuel_type):
    car_age = extract_car_age(year)
    input_data = pd.DataFrame({
        'name': [model_name],
        'company': [company],
        'year': [year],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type],
        'car_age': [car_age]
    })
    return input_data

# Load data and model
try:
    data = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()

# Dashboard layout
st.title("Car Price Predictor")
st.markdown("""
This application uses machine learning to predict the price of used cars based on various features.
Fill in the details below to get an estimate for your car's market value.
""")

# Create tabs for prediction and analysis
tab1, tab2 = st.tabs(["Price Prediction", "Market Analysis"])

with tab1:
    st.header("Enter Car Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select the company
        companies = sorted(data['company'].unique())
        company = st.selectbox("Select Car Manufacturer:", ["Select Company"] + list(companies))
        
        # Select the model based on company
        if company != "Select Company":
            models = sorted(data[data['company'] == company]['name'].unique())
            model_name = st.selectbox("Select Car Model:", ["Select Model"] + list(models))
        else:
            model_name = "Select Model"
        
        # Select year of purchase
        years = sorted(data['year'].unique(), reverse=True)
        year = st.selectbox("Select Year of Manufacture:", years)
    
    with col2:
        # Select the fuel type
        fuel_types = sorted(data['fuel_type'].unique())
        fuel_type = st.selectbox("Select Fuel Type:", fuel_types)
        
        # Enter kilometers driven
        kms_driven = st.number_input("Kilometers Driven:", min_value=0, value=10000, step=1000)
        
        # Calculate car age automatically
        if year:
            car_age = 2025 - year
            st.info(f"Car Age: {car_age} years")
    
    # Predict price
    if st.button("Predict Price", type="primary"):
        # Validate inputs
        if company == "Select Company" or model_name == "Select Model":
            st.error("Please fill all the fields correctly!")
        else:
            try:
                # Preprocess the input
                input_data = preprocess_input(company, model_name, year, kms_driven, fuel_type)
                
                # Make prediction
                predicted_price = model.predict(input_data)[0]
                
                # Format the price with commas for Indian Rupees
                formatted_price = f"₹{predicted_price:,.2f}"
                
                # Display the prediction
                st.success(f"The predicted price of the car is: {formatted_price}")
                
                # Show confidence interval (approximate)
                lower_bound = predicted_price * 0.85
                upper_bound = predicted_price * 1.15
                
                st.write("#### Price Range")
                st.write(f"The price is likely to be between ₹{lower_bound:,.2f} and ₹{upper_bound:,.2f}")
                
                # Compare with similar cars
                st.write("#### Similar Cars in Market")
                similar_cars = data[
                    (data['company'] == company) & 
                    (data['fuel_type'] == fuel_type) & 
                    (abs(data['year'] - year) <= 2)
                ].sort_values('Price')
                
                if not similar_cars.empty:
                    st.write(f"Average price of similar cars: ₹{similar_cars['Price'].mean():,.2f}")
                    st.write(f"Price range: ₹{similar_cars['Price'].min():,.2f} - ₹{similar_cars['Price'].max():,.2f}")
                else:
                    st.write("No similar cars found in the database.")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")

with tab2:
    st.header("Market Analysis")
    
    # Show dataset statistics
    st.subheader("Dataset Overview")
    st.write(f"Total cars in database: {len(data)}")
    st.write(f"Number of manufacturers: {data['company'].nunique()}")
    st.write(f"Number of models: {data['name'].nunique()}")
    
    # Popular manufacturers
    st.subheader("Popular Manufacturers")
    company_counts = data['company'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 6))
    company_counts.plot(kind='bar', ax=ax)
    plt.title('Top 10 Car Manufacturers by Count')
    plt.xlabel('Manufacturer')
    plt.ylabel('Number of Cars')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Price distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.histplot(data=data, x='Price', bins=30, kde=True, ax=ax)
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price (₹)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Price by Year
    st.subheader("Price by Year")
    yearly_avg = data.groupby('year')['Price'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.lineplot(data=yearly_avg, x='year', y='Price', marker='o', ax=ax)
    plt.title('Average Car Price by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Price (₹)')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Price by Fuel Type
    st.subheader("Price by Fuel Type")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(data=data, x='fuel_type', y='Price', ax=ax)
    plt.title('Car Price Distribution by Fuel Type')
    plt.xlabel('Fuel Type')
    plt.ylabel('Price (₹)')
    plt.tight_layout()
    st.pyplot(fig)

# Add a footer
st.markdown("---")
st.markdown("Car Price Predictor - Made with Streamlit and Machine Learning")