"""Recreate the README chart from recorded results without retraining."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    results = pd.read_csv(ROOT / "model_comparison_results.csv").sort_values("MAE")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.23, right=0.94, top=0.74, bottom=0.19)
    fig.text(0.06, 0.925, "Used-car price prediction", fontsize=22,
             weight="bold", color="#142b3d")
    fig.text(0.06, 0.855, "Mean absolute error on the shared test partition · Lower is better",
             fontsize=12, color="#526273")
    colors = ["#247a9a" if model == "Random Forest" else "#a8bac8"
              for model in results["Model"]]
    bars = ax.barh(results["Model"], results["MAE"], color=colors, height=0.58)
    ax.invert_yaxis()
    for bar, value in zip(bars, results["MAE"]):
        ax.text(value + 35, bar.get_y() + bar.get_height() / 2,
                f"${value:,.2f}", va="center", fontsize=11, color="#142b3d")
    ax.set_xlim(0, results["MAE"].max() * 1.22)
    ax.set_xlabel("Mean absolute error (USD)", labelpad=10, color="#526273")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e7edf1")
    ax.tick_params(axis="both", length=0, labelcolor="#344659")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.06, 0.055,
             "Source: model_comparison_results.csv · 80/20 split · random_state=42",
             fontsize=9, color="#647486")
    output = ROOT / "docs" / "model_comparison.png"
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output.name}: {len(results)} recorded model results")


if __name__ == "__main__":
    main()
