import pandas as pd


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def add_features(df):
    """
    Dodaje nove karakteristike korisne za predikciju cene.

    Kreirane karakteristike:
    - car_age
    - mileage_per_year
    - engine_volume_liters
    - brand_model
    """

    df_features = df.copy()

    # Referentna godina je godinu dana veca
    # od najnovije godine u datasetu.
    reference_year = (
        df_features["year"].max() + 1
    )

    # --------------------------------------------------------
    # 1. Starost automobila
    # --------------------------------------------------------

    df_features["car_age"] = (
        reference_year
        - df_features["year"]
    )

    # --------------------------------------------------------
    # 2. Prosecna kilometraza po godini
    # --------------------------------------------------------

    df_features["mileage_per_year"] = (
        df_features["mileage(kilometers)"]
        / df_features["car_age"].clip(lower=1)
    )

    # --------------------------------------------------------
    # 3. Zapremina motora u litrima
    # --------------------------------------------------------

    df_features["engine_volume_liters"] = (
        df_features["volume(cm3)"]
        / 1000
    )

    # --------------------------------------------------------
    # 4. Kombinacija marke i modela
    # --------------------------------------------------------

    df_features["brand_model"] = (
        df_features["make"].astype(str)
        + "_"
        + df_features["model"].astype(str)
    )

    return df_features


# ============================================================
# SAMOSTALNO POKRETANJE SKRIPTE
# ============================================================

if __name__ == "__main__":

    INPUT_FILE = "cars_clean.csv"
    OUTPUT_FILE = "cars_features.csv"

    print("Ucitavanje ociscenih podataka...")

    df = pd.read_csv(INPUT_FILE)

    df_features = add_features(df)

    print(
        "Broj redova:",
        len(df_features)
    )

    print(
        "Broj kolona:",
        df_features.shape[1]
    )

    print(
        "\nDodate karakteristike:"
    )

    print(
        [
            "car_age",
            "mileage_per_year",
            "engine_volume_liters",
            "brand_model"
        ]
    )

    print(
        "\nBroj jedinstvenih brand_model vrednosti:",
        df_features["brand_model"].nunique()
    )

    df_features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nPodaci sa novim karakteristikama sacuvani kao:",
        OUTPUT_FILE
    )