import numpy as np
import pandas as pd


# ============================================================
# DATA CLEANING
# ============================================================


def clean_data(df):
    """
    Osnovno ciscenje dataseta polovnih automobila.

    Koraci:
    1. Uklanjanje potpuno dupliranih redova.
    2. Ekstremne kilometraze preko 1,000,000 km
       tretiraju se kao nevalidne i postavljaju na NaN.
    3. Ekstremne zapremine motora preko 8000 cm3
       tretiraju se kao nevalidne i postavljaju na NaN.
    """

    df_clean = df.copy()

    # --------------------------------------------------------
    # 1. Uklanjanje duplikata
    # --------------------------------------------------------

    df_clean = df_clean.drop_duplicates().copy()

    # --------------------------------------------------------
    # 2. Nevalidne kilometraze
    # --------------------------------------------------------

    df_clean.loc[
        df_clean["mileage(kilometers)"] > 1_000_000,
        "mileage(kilometers)"
    ] = np.nan

    # --------------------------------------------------------
    # 3. Nevalidne zapremine motora
    # --------------------------------------------------------

    df_clean.loc[
        df_clean["volume(cm3)"] > 8000,
        "volume(cm3)"
    ] = np.nan

    return df_clean


# ============================================================
# SAMOSTALNO POKRETANJE SKRIPTE
# ============================================================

if __name__ == "__main__":

    DATA_FILE = "cars.csv"
    OUTPUT_FILE = "cars_clean.csv"

    print("Ucitavanje podataka...")

    df = pd.read_csv(DATA_FILE)

    print(
        "Broj redova pre ciscenja:",
        len(df)
    )

    df_clean = clean_data(df)

    print(
        "Broj redova posle ciscenja:",
        len(df_clean)
    )

    print("\nNedostajuce vrednosti:")
    print(
        df_clean.isnull().sum()
    )

    df_clean.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nOcisceni podaci sacuvani kao:",
        OUTPUT_FILE
    )