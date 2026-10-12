import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# DEFINICIJA KOLONA
# ============================================================

TARGET_COLUMN = "priceUSD"

NUMERIC_FEATURES = [
    "car_age",
    "mileage(kilometers)",
    "mileage_per_year",
    "engine_volume_liters"
]

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "brand_model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():
    """
    Kreira preprocessing pipeline za numericke i
    kategorijske karakteristike.
    """

    # --------------------------------------------------------
    # Numericki pipeline
    # --------------------------------------------------------

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Kategorijski pipeline
    # --------------------------------------------------------

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Kombinovanje
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                NUMERIC_FEATURES
            ),
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor


# ============================================================
# PRIPREMA X I y
# ============================================================

def prepare_xy(df):
    """
    Razdvaja ulazne karakteristike i ciljnu promenljivu.
    """

    X = df[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# SAMOSTALNO POKRETANJE
# ============================================================

if __name__ == "__main__":

    INPUT_FILE = "cars_features.csv"

    print("Ucitavanje podataka...")

    df = pd.read_csv(INPUT_FILE)

    X, y = prepare_xy(df)

    preprocessor = build_preprocessor()

    print(
        "\nDimenzije X:",
        X.shape
    )

    print(
        "Dimenzije y:",
        y.shape
    )

    print(
        "\nNumericke karakteristike:"
    )
    print(NUMERIC_FEATURES)

    print(
        "\nKategorijske karakteristike:"
    )
    print(CATEGORICAL_FEATURES)

    print(
        "\nPreprocessing pipeline je uspesno kreiran."
    )