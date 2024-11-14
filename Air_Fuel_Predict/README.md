WebApp : https://air-fuel-predict.streamlit.app/
# Aircraft Fuel Consumption Prediction Model

This machine learning model is designed to predict aircraft fuel consumption based on various flight characteristics. The model uses a multiple linear regression algorithm to provide accurate fuel consumption estimates for different flight scenarios.

---

## Model Overview

- **Model Type:** Linear Regression
- **Dependencies:** Utilizes `joblib` for loading the serialized model and `pandas` for data processing.

## Input Parameters

The model requires the following parameters to generate fuel consumption predictions:

- **Flight Distance:** Distance of the flight in kilometers (float).
- **Flight Duration:** Estimated time of the flight in hours (float).
- **Number of Passengers:** Total number of passengers on board (integer).
- **Aircraft Type:** Categorical indicator specifying the aircraft type (options: Type1, Type2, Type3).

## Output

- **Predicted Fuel Consumption:** The model returns an estimated value representing the fuel consumption for the specified flight parameters.

## Model Requirements

The serialized model file, `linear_regression_model.pkl`, must be located in the `Air_Fuel_Predict` directory within the project structure for predictions to function properly.

---

This README provides a clear overview of the model, its input requirements, and its expected output.
