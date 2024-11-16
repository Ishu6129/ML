import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Load the saved model
with open('smart_fridge.pkl', 'rb') as f:
    model = pickle.load(f)

# Label Encoder for product names (used during model training)
label_encoder = LabelEncoder()
label_encoder.fit(['Item A', 'Item B', 'Item C'])  # Add your unique product names here

# Streamlit app layout
st.title('Smart Fridge Product Consumption Predictor')

# Input fields for user to enter product data
product_name = st.selectbox('Select Product', ['Item A', 'Item B', 'Item C'])
manufacturing_date = st.date_input('Manufacturing Date', datetime.today())
purchase_date = st.date_input('Purchase Date', datetime.today())
expiry_date = st.date_input('Expiry Date', datetime.today())
consumption_date = st.date_input('Consumption Date', datetime.today())

# Button to predict
if st.button('Predict Time to Consumption'):
    # Feature engineering based on user input
    age = (pd.to_datetime(purchase_date) - pd.to_datetime(manufacturing_date)).days
    remaining_shelf_life = (pd.to_datetime(expiry_date) - pd.to_datetime(purchase_date)).days
    time_to_consumption = (pd.to_datetime(consumption_date) - pd.to_datetime(purchase_date)).days
    
    # Encode the product name
    product_name_encoded = label_encoder.transform([product_name])[0]
    
    # Prepare the feature vector for prediction
    X_new = np.array([[age, remaining_shelf_life, product_name_encoded]])
    
    # Make prediction
    predicted_consumption_days = model.predict(X_new)
    
    # Display the result
    st.write(f'Predicted Time to Consumption (in days): {predicted_consumption_days[0]:.0f}')
