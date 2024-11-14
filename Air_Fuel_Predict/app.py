import streamlit as st
import joblib
import pandas as pd
import numpy as np

background_image = "https://royalwestindies.com/wp-content/uploads/2017/11/flight-back.jpg"

# Set the image as the background using CSS
st.markdown(f"""
<style>
.stApp {{
    background-image: url('{background_image}');
    background-size: cover;
}}
</style>
""", unsafe_allow_html=True)
st.markdown(f"""
<style>
.stApp .main {{
    text-align: center;
}}

.stApp h1 {{
    color: white;
}}
</style>
""", unsafe_allow_html=True)
#Load Model
def load_model():
    try:
        with open("linear_regression_model.pkl", 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'linear_regression_model.pkl' is in the correct directory.")
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")

model = load_model()

#Title

st.title("Aircraft Fuel Prediction")

# Input fields
f_dist=st.number_input("Figlt Distance : ")
f_type = st.selectbox("Select Type : ",["Type1", "Type2", "Type3"])
f_duration=st.number_input("Flight Duration (Hours) : ")
no_of_pass=st.number_input("Numbver of Passengers : ")

# Create input data DataFrame   
input_data=pd.DataFrame({
    'Flight_Distance':[f_dist],
    'Number_of_Passengers':[no_of_pass],
    'Flight_Duration':[f_duration],
    'Aircraft_Type_Type1':[1 if f_type=="Type1" else 0],
    'Aircraft_Type_Type2':[1 if f_type=="Type2" else 0],
    'Aircraft_Type_Type3':[1 if f_type=="Type3" else 0] 
    })

# Prediction button
if st.button("Predict"):
    fuel_consumption = model.predict(input_data)
    print(fuel_consumption)
    st.write("Predicted Fuel Consumption:", fuel_consumption[0])