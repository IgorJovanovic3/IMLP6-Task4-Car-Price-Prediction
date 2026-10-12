import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from data_preprocessing import (
    prepare_xy,
    build_preprocessor
)


# ============================================================
# MODEL COMPARISON
# ============================================================

INPUT_FILE = "cars_features.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ------------------------------------------------------------
# 1. Ucitavanje podataka
# ------------------------------------------------------------

print("Ucitavanje podataka...")

df = pd.read_csv(INPUT_FILE)

X, y = prepare_xy(df)


# ------------------------------------------------------------
# 2. Ista train / test podela za sve modele
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# 3. Modeli za poređenje
# ------------------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=RANDOM_STATE,
        max_depth=20,
        min_samples_leaf=2
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=20,
        min_samples_leaf=2
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE
    )
}


# ------------------------------------------------------------
# 4. Treniranje i evaluacija
# ------------------------------------------------------------

results = []


for model_name, model in models.items():

    print(
        f"\nTreniranje modela: {model_name}"
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

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

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })


# ------------------------------------------------------------
# 5. Tabela rezultata
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values("MAE")
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("POREDJENJE MODELA")
print("=" * 70)

print(
    results_df.round({
        "MAE": 2,
        "MSE": 2,
        "RMSE": 2,
        "R2": 4
    }).to_string(index=False)
)


# ------------------------------------------------------------
# 6. Najbolji model
# ------------------------------------------------------------

best_model = results_df.iloc[0]

print("\nNajbolji model prema MAE:")
print(
    best_model["Model"]
)

print(
    "MAE:",
    round(best_model["MAE"], 2)
)

print(
    "RMSE:",
    round(best_model["RMSE"], 2)
)

print(
    "R2:",
    round(best_model["R2"], 4)
)


# ------------------------------------------------------------
# 7. Cuvanje rezultata
# ------------------------------------------------------------

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print(
    "\nRezultati su sacuvani kao:",
    "model_comparison_results.csv"
)