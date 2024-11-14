WebApp : https://air-fuel-predict.streamlit.app/
<html><body>
<span style="color: #2E8B57; font-size: 2em;"><h2>Aircraft Fuel Consumption Prediction Model<h2></span>
<p style="font-size: 1.2em; color: #555;">This machine learning model estimates aircraft fuel consumption based on flight characteristics using a multiple linear regression algorithm.</p>
<span style="color: #2E8B57;">Model Overview</span>
<span style="color: #4682B4;">Model Type:</span> Linear Regression
<span style="color: #4682B4;">Dependencies:</span> Uses joblib for loading the serialized model and pandas for data processing.
<span style="color: #2E8B57;">Input Parameters</span>
The model requires the following parameters to generate fuel consumption predictions:

<span style="color: #4682B4;">Flight Distance:</span> Distance of the flight in kilometers (float).
<span style="color: #4682B4;">Flight Duration:</span> Estimated time of the flight in hours (float).
<span style="color: #4682B4;">Number of Passengers:</span> Total number of passengers on board (integer).
<span style="color: #4682B4;">Aircraft Type:</span> Categorical indicator specifying the aircraft type (options: Type1, Type2, Type3).
<span style="color: #2E8B57;">Output</span>
<span style="color: #4682B4;">Predicted Fuel Consumption:</span> The model returns an estimated value representing the fuel consumption for the specified flight parameters.
<span style="color: #2E8B57;">Model Requirements</span>
The serialized model file, linear_regression_model.pkl, must be located in the Air_Fuel_Predict directory within the project structure for predictions to function properly.

</body>
</html>
