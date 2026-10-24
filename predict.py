
# Step 9: Load the saved model and predict AQI

import pandas as pd
import joblib

# Load the trained model
model = joblib.load("aqi_model.pkl")

# New environmental data
input_data = pd.DataFrame([{
    "PM2.5": 45.0,
    "PM10": 80.0,
    "NO": 10.0,
    "NO2": 20.0,
    "NOx": 25.0,
    "NH3": 10.0,
    "CO": 1.0,
    "SO2": 8.0,
    "O3": 30.0,
    "Benzene": 2.0,
    "Toluene": 5.0,
    "Xylene": 1.0
}])

# Predict AQI
prediction = model.predict(input_data)

print("Predicted AQI:", round(prediction[0], 2))
