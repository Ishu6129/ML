import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

try:
    with open('smart_fridge/smart_fridge.pkl', 'rb') as f:
        model = pickle.load(f)
        if not hasattr(model, 'predict'):
            raise ValueError("The loaded object is not a predictive model.")
except FileNotFoundError:
    st.error("Model file not found. Ensure 'smart_fridge.pkl' exists in the 'smart_fridge' directory.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

label_encoder = LabelEncoder()
label_encoder.fit(['Item A', 'Item B', 'Item C'])

st.title('Smart Fridge Product Consumption Predictor')

product_name = st.selectbox('Select Product', ['Item A', 'Item B', 'Item C'])
manufacturing_date = st.date_input('Manufacturing Date', datetime.today())
purchase_date = st.date_input('Purchase Date', datetime.today())
expiry_date = st.date_input('Expiry Date', datetime.today())
consumption_date = st.date_input('Consumption Date', datetime.today())

if st.button('Predict Time to Consumption'):
    try:
        age = (pd.to_datetime(purchase_date) - pd.to_datetime(manufacturing_date)).days
        remaining_shelf_life = (pd.to_datetime(expiry_date) - pd.to_datetime(purchase_date)).days
        time_to_consumption = (pd.to_datetime(consumption_date) - pd.to_datetime(purchase_date)).days
        product_name_encoded = label_encoder.transform([product_name])[0]
        X_new = np.array([[age, remaining_shelf_life, product_name_encoded]])
        predicted_consumption_days = model.predict(X_new)
        st.write(f'Predicted Time to Consumption (in days): {predicted_consumption_days[0]:.0f}')
    except Exception as e:
        st.error(f"Prediction failed: {e}")
