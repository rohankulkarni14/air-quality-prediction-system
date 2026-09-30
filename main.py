import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

print("=" * 50)
print("      AIR QUALITY PREDICTION SYSTEM")
print("=" * 50)

print("\nEnter the following weather conditions:\n")

# Get user inputs
T = float(input("Average Temperature (°C): "))
TM = float(input("Maximum Temperature (°C): "))
Tm = float(input("Minimum Temperature (°C): "))
SLP = float(input("Sea Level Pressure: "))
H = float(input("Humidity (%): "))
VV = float(input("Visibility: "))
V = float(input("Wind Speed: "))
VM = float(input("Maximum Wind Speed: "))

# Create input DataFrame
input_data = pd.DataFrame({
    "T": [T],
    "TM": [TM],
    "Tm": [Tm],
    "SLP": [SLP],
    "H": [H],
    "VV": [VV],
    "V": [V],
    "VM": [VM]
})

# Make prediction
prediction = model.predict(input_data)[0]

print("\n" + "=" * 50)
print("             PREDICTION RESULT")
print("=" * 50)

print(f"\nPredicted PM2.5: {prediction:.2f} µg/m³")

# Simple interpretation
if prediction <= 50:
    category = "Good"
elif prediction <= 100:
    category = "Moderate"
elif prediction <= 150:
    category = "Unhealthy for Sensitive Groups"
elif prediction <= 200:
    category = "Unhealthy"
elif prediction <= 300:
    category = "Very Unhealthy"
else:
    category = "Hazardous"

print(f"Air Quality Category: {category}")

print("\n" + "=" * 50)
