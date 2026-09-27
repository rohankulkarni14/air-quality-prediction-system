import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor

# Load dataset
data = pd.read_csv("air_quality_dataset.csv")

# Remove missing PM2.5 values
data = data.dropna(subset=["PM 2.5"])

# Features
X = data.drop("PM 2.5", axis=1)

# Target
y = data["PM 2.5"]

# Final Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42
)

# Train on the complete cleaned dataset
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully.")
print("Saved as model.pkl")