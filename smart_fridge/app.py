import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('smart_fridge.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

# Initialize label encoder used during training
label_encoder = LabelEncoder()
label_encoder.fit(['Item A', 'Item B', 'Item C'])  # Use the same labels as in the training set

# Helper function to process the input data
def process_input(data):
    # Convert date strings to datetime
    manufacturing_date = datetime.strptime(data['Manufacturing Date'], '%Y-%m-%d')
    expiry_date = datetime.strptime(data['Expiry Date'], '%Y-%m-%d')
    purchase_date = datetime.strptime(data['Purchase Date'], '%Y-%m-%d')
    consumption_date = datetime.strptime(data['Consumption Date'], '%Y-%m-%d')

    # Feature Engineering
    age = (purchase_date - manufacturing_date).days
    remaining_shelf_life = (expiry_date - purchase_date).days
    time_to_consumption = (consumption_date - purchase_date).days
    
    # Encode product name
    product_name_encoded = label_encoder.transform([data['Product Name']])[0]
    
    # Prepare feature vector
    features = np.array([[age, remaining_shelf_life, product_name_encoded]])
    
    return features

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from request
    data = request.json
    
    # Process input data
    features = process_input(data)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Return the prediction as JSON
    return jsonify({'predicted_consumption_days': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
