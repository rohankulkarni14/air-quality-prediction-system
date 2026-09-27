import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("air_quality_dataset.csv")

# Remove rows where PM2.5 is missing
data = data.dropna(subset=["PM 2.5"])

# Features
X = data.drop("PM 2.5", axis=1)

# Target
y = data["PM 2.5"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}

results = {}

# Train and evaluate
# 5-Fold Cross Validation with shuffling
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # 5-fold cross validation
    cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

    cv_mean = cv_scores.mean()

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV_R2": cv_mean
    }

# Display results
print("\nMODEL COMPARISON")
print("=" * 70)

for name, metrics in results.items():

    print(f"\n{name}")
    print(f"MAE       : {metrics['MAE']:.2f}")
    print(f"RMSE      : {metrics['RMSE']:.2f}")
    print(f"R²        : {metrics['R2']:.2f}")
    print(f"CV R²     : {metrics['CV_R2']:.2f}")

# Find best model using cross-validation R²
best_model_name = max(
    results,
    key=lambda x: results[x]["CV_R2"]
)

print("\n" + "=" * 70)
print("BEST MODEL BASED ON 5-FOLD CROSS-VALIDATION:")
print(best_model_name)