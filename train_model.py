
# Step 1: Import all necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import joblib


# Step 2: Read CSV file
data = Path("dataset/AQI.csv/city_day.csv")
df = pd.read_csv(data)


# Step 3: Performing EDA

print("Dataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical description:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Step 4: Data Cleaning

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where target AQI is missing
df = df.dropna(subset=["AQI"])

# Select input features
features = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

# Fill missing feature values using column medians
df[features] = df[features].fillna(df[features].median())

print("\nMissing values after cleaning:")
print(df[features + ["AQI"]].isnull().sum())

print("\nDataset shape after cleaning:")
print(df.shape)

# Step 5: Split features and target

# Input features
x = df[features]

# Target variable
y = df["AQI"]

print("\nFeatures shape:", x.shape)
print("Target shape:", y.shape)


# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Step 6: Train the model

model = LinearRegression()

model.fit(x_train, y_train)

print("\nModel training completed successfully!")

# Step 7: Testing and evaluation

# Predict AQI for test data
y_pred = model.predict(x_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Step 8: Save the trained model

joblib.dump(model, "aqi_model.pkl")

print("\nModel saved successfully!")

# Step 8: Save the trained model

joblib.dump(model, "aqi_model.pkl")

print("\nModel saved successfully!")
