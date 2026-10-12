import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import prepare_xy


# ============================================================
# MODEL EVALUATION
# ============================================================

INPUT_FILE = "cars_features.csv"
MODEL_FILE = "car_price_model.joblib"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ------------------------------------------------------------
# 1. Ucitavanje podataka
# ------------------------------------------------------------

print("Ucitavanje podataka...")

df = pd.read_csv(INPUT_FILE)

X, y = prepare_xy(df)


# ------------------------------------------------------------
# 2. Ista train / test podela kao pri treningu
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# 3. Ucitavanje sacuvanog modela
# ------------------------------------------------------------

print("Ucitavanje modela...")

model = joblib.load(MODEL_FILE)


# ------------------------------------------------------------
# 4. Predikcije
# ------------------------------------------------------------

predictions = model.predict(
    X_test
)


# ------------------------------------------------------------
# 5. Regresione metrike
# ------------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    predictions
)


print("\nRandom Forest evaluacija:")

print(
    "MAE:",
    round(mae, 2)
)

print(
    "MSE:",
    round(mse, 2)
)

print(
    "RMSE:",
    round(rmse, 2)
)

print(
    "R2:",
    round(r2, 4)
)


# ------------------------------------------------------------
# 6. Nekoliko primera
# ------------------------------------------------------------

results = X_test.copy()

results["actual_price"] = y_test
results["predicted_price"] = predictions

results["absolute_error"] = np.abs(
    results["actual_price"]
    - results["predicted_price"]
)


print("\nPrimeri predikcija:")

print(
    results[
        [
            "make",
            "model",
            "actual_price",
            "predicted_price",
            "absolute_error"
        ]
    ]
    .head(10)
    .round(2)
    .to_string(index=False)
)