import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv("hospital_inventory_data.csv")

# Create a more realistic future demand target
np.random.seed(42)

season_multiplier = np.where(df["month"].isin([11, 12, 1, 2]), 1.25, 1.0)
flu_effect = 1 + (df["flu_cases_index"] - 1) * 0.3
random_noise = np.random.normal(1.0, 0.08, size=len(df))

df["future_month_demand"] = (
    df["daily_usage"] * 30 * season_multiplier * flu_effect * random_noise
).round(0)

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=["department", "medication_name"], drop_first=True)

# Features and target
X = df_encoded.drop(columns=["expiration_date", "future_month_demand"])
y = df_encoded["future_month_demand"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae:.2f}")

# Show sample predictions
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions.round(1)
})

print("\n=== SAMPLE PREDICTIONS ===")
print(results.head(10))