import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("air_quality_dataset.csv")

# Remove missing PM2.5 values
data = data.dropna(subset=["PM 2.5"])

# Features and target
X = data.drop("PM 2.5", axis=1)
y = data["PM 2.5"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Cross-validation strategy
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

rf = RandomForestRegressor(
    random_state=42
)

rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

rf_grid = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

print("Tuning Random Forest...")
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

rf_predictions = best_rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRANDOM FOREST RESULTS")
print("=" * 50)
print("Best Parameters:")
print(rf_grid.best_params_)
print(f"CV R²  : {rf_grid.best_score_:.2f}")
print(f"Test MAE  : {rf_mae:.2f}")
print(f"Test RMSE : {rf_rmse:.2f}")
print(f"Test R²   : {rf_r2:.2f}")


# --------------------------------------------------
# GRADIENT BOOSTING
# --------------------------------------------------

gb = GradientBoostingRegressor(
    random_state=42
)

gb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

gb_grid = GridSearchCV(
    estimator=gb,
    param_grid=gb_params,
    cv=kf,
    scoring="r2",
    n_jobs=-1
)

print("\nTuning Gradient Boosting...")
gb_grid.fit(X_train, y_train)

best_gb = gb_grid.best_estimator_

gb_predictions = best_gb.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = mean_squared_error(y_test, gb_predictions) ** 0.5
gb_r2 = r2_score(y_test, gb_predictions)

print("\nGRADIENT BOOSTING RESULTS")
print("=" * 50)
print("Best Parameters:")
print(gb_grid.best_params_)
print(f"CV R²  : {gb_grid.best_score_:.2f}")
print(f"Test MAE  : {gb_mae:.2f}")
print(f"Test RMSE : {gb_rmse:.2f}")
print(f"Test R²   : {gb_r2:.2f}")


# --------------------------------------------------
# FINAL COMPARISON
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TUNED MODEL COMPARISON")
print("=" * 60)

print(f"\nRandom Forest")
print(f"CV R²    : {rf_grid.best_score_:.2f}")
print(f"Test R²  : {rf_r2:.2f}")
print(f"Test MAE : {rf_mae:.2f}")
print(f"Test RMSE: {rf_rmse:.2f}")

print(f"\nGradient Boosting")
print(f"CV R²    : {gb_grid.best_score_:.2f}")
print(f"Test R²  : {gb_r2:.2f}")
print(f"Test MAE : {gb_mae:.2f}")
print(f"Test RMSE: {gb_rmse:.2f}")