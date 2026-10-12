import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from data_preprocessing import (
    prepare_xy,
    build_preprocessor
)


# ============================================================
# MODEL TRAINING
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
# 2. Train / test podela
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(
    "Broj trening primera:",
    len(X_train)
)

print(
    "Broj test primera:",
    len(X_test)
)


# ------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------

preprocessor = build_preprocessor()


# ------------------------------------------------------------
# 4. Finalni Random Forest model
# ------------------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_depth=20,
    min_samples_leaf=2
)


# ------------------------------------------------------------
# 5. Kompletan pipeline
# ------------------------------------------------------------

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            random_forest
        )
    ]
)


# ------------------------------------------------------------
# 6. Treniranje
# ------------------------------------------------------------

print("\nTreniranje Random Forest modela...")

model_pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# 7. Cuvanje modela
# ------------------------------------------------------------

joblib.dump(
    model_pipeline,
    MODEL_FILE,
    compress=3
)


print(
    "\nModel je uspesno treniran."
)

print(
    "Model je sacuvan kao:",
    MODEL_FILE
)