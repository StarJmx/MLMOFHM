from __future__ import unicode_literals

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_EXCEL = REPO_DIR / "data" / "stability_split90_10_Q_R.xlsx"


def load_test_r2(excel_path: Path, sheet_name: str) -> pd.Series:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "test_R2" not in df.columns:
        raise KeyError(f"Sheet `{sheet_name}` missing column `test_R2`. Columns: {list(df.columns)}")
    # Expect one row per seed; plotting uses the raw test_R2 values.
    return df["test_R2"].dropna()


def plot_boxplot(q_values, r_values, output_png: Path) -> None:
    mp.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
        }
    )
    mp.figure(figsize=(3.5, 2.8), dpi=600)
    data = [q_values, r_values]
    labels = ["Q", "R"]

    bp = mp.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=False,
        meanline=False,
        boxprops={"facecolor": "royalblue", "edgecolor": "royalblue", "alpha": 0.55},
        whiskerprops={"color": "royalblue"},
        capprops={"color": "royalblue"},
        medianprops={"color": "orange", "linewidth": 1.5},
    )

    # Distinguish Q vs R in legend (and visually) without changing the standard blue color.
    # (hatch patterns only affect the box patch, which we set via patch_artist=True)
    if "boxes" in bp and len(bp["boxes"]) >= 2:
        bp["boxes"][0].set_hatch("///")  # Q
        bp["boxes"][1].set_hatch("\\\\\\\\")  # R (escape for python string)

    # Light grid for readability.
    mp.grid(axis="y", linestyle="--", alpha=0.35)
    mp.ylabel("Test R$^2$")
    mp.title("Stability (90/10 splits): Test R$^2$ across seeds")

    # Annotate basic stats: mean above each box, then std just below the mean.
    y_all = np.concatenate([np.asarray(q_values), np.asarray(r_values)])
    y_min = float(np.min(y_all)) if y_all.size else 0.0
    y_max = float(np.max(y_all)) if y_all.size else 1.0
    y_range = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
    dy = y_range * 0.02  # vertical offset for the std text

    for i, series in enumerate(data, start=1):
        mean_val = float(series.mean()) if len(series) else float("nan")
        std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0

        mp.text(i, mean_val, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8)
        mp.text(i, mean_val - dy, f"±{std_val:.3f}", ha="center", va="top", fontsize=8)

    # Legend: Q/R categories + median meaning.
    handles = [
        Patch(facecolor="royalblue", edgecolor="royalblue", alpha=0.55, hatch="///", label="Q"),
        Patch(
            facecolor="royalblue",
            edgecolor="royalblue",
            alpha=0.55,
            hatch="\\\\\\\\",
            label="R",
        ),
        Line2D([0], [0], color="orange", lw=2, label="Median (50%)"),
    ]
    mp.legend(handles=handles, loc="best", frameon=False, fontsize=9)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    mp.tight_layout()
    mp.savefig(output_png, bbox_inches="tight")
    mp.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot boxplot of test_R2 from exp_stability_split_90_10.py Excel output."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_EXCEL),
        help="Path to the stability Excel file (default: stability_split90_10_Q_R.xlsx).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output png path. If omitted, save next to input with a fixed filename.",
    )
    args = parser.parse_args()

    input_excel = Path(args.input)
    if not input_excel.exists():
        raise FileNotFoundError(f"Excel not found: {input_excel}")

    q_values = load_test_r2(input_excel, "Q_seed_metrics")
    r_values = load_test_r2(input_excel, "R_seed_metrics")

    if args.output is None:
        output_png = input_excel.with_name("stability_seed_metrics_test_R2_boxplot.jpg")
    else:
        output_png = Path(args.output)

    plot_boxplot(q_values=q_values.values, r_values=r_values.values, output_png=output_png)
    print(f"Boxplot saved to: {output_png}")


if __name__ == "__main__":
    main()

