import streamlit as st
import joblib
import pandas as pd
import os

# Background image URL
background_image = "https://royalwestindies.com/wp-content/uploads/2017/11/flight-back.jpg"

# Apply background image and styling with CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url('{background_image}');
        background-size: cover;
        background-position: center;
        color: white;
    }}
    .title-text {{
        font-size: 2.5em;
        font-weight: bold;
        text-shadow: 2px 2px 5px #333;
    }}
    .subtitle-text {{
        font-size: 1.2em;
        margin-bottom: 20px;
    }}
    .input-container {{
        background: rgba(0, 0, 0, 0.6);
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<div class='title-text'>Aircraft Fuel Prediction ✈️</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Estimate the fuel consumption for a flight based on distance, duration, passenger count, and aircraft type.</div>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join("Air_Fuel_Predict", "linear_regression_model.pkl")
    return joblib.load(model_path)

model = load_model()

# Input section
st.markdown("<div class='input-container'>", unsafe_allow_html=True)

# Input field
f_dist = st.number_input("✈️ Flight Distance (km):", min_value=0.0, step=1.0, help="Enter the total distance of the flight in kilometers.")
f_type = st.selectbox("🛫 Aircraft Type:", ["Type1", "Type2", "Type3"], help="Choose the type of aircraft.")
f_duration = st.number_input("⏱ Flight Duration (hours):", min_value=0.0, step=0.1, help="Enter the total duration of the flight in hours.")
no_of_pass = st.number_input("👥 Number of Passengers:", min_value=0, step=1, help="Enter the number of passengers on the flight.")


st.markdown("</div>", unsafe_allow_html=True)


input_data = pd.DataFrame({
    'Flight_Distance': [f_dist],
    'Number_of_Passengers': [no_of_pass],
    'Flight_Duration': [f_duration],
    'Aircraft_Type_Type1': [1 if f_type == "Type1" else 0],
    'Aircraft_Type_Type2': [1 if f_type == "Type2" else 0],
    'Aircraft_Type_Type3': [1 if f_type == "Type3" else 0]
})

# Prediction button
if st.button("Predict Fuel Consumption"):
    try:
        fuel_consumption = model.predict(input_data)
        st.success(f"Estimated Fuel Consumption: {fuel_consumption[0]:.2f} units")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
