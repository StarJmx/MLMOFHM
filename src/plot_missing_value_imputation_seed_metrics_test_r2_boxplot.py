from __future__ import unicode_literals

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_EXCEL = REPO_DIR / "data" / "exp_stability_missing_value_imputation_metrics.xlsx"

DATASET_NAMES = [
    "Mean imputation",
    "RF imputation",
    "Hybrid imputation",
]


def load_test_r2(excel_path: Path, sheet_name: str) -> pd.Series:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if "test_R2" not in df.columns:
        raise KeyError(f"Sheet `{sheet_name}` missing column `test_R2`. Columns: {list(df.columns)}")
    return df["test_R2"].dropna()


def build_seed_sheet_name(dataset_name: str) -> str:
    sheet_prefix = dataset_name.replace(" ", "_")[:20]
    return f"{sheet_prefix}_seed"


def plot_boxplot(values_list, labels, output_png: Path) -> None:
    mp.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
        }
    )
    mp.figure(figsize=(4.8, 3.2), dpi=600)

    bp = mp.boxplot(
        values_list,
        labels=labels,
        patch_artist=True,
        showmeans=False,
        meanline=False,
        boxprops={"facecolor": "royalblue", "edgecolor": "royalblue", "alpha": 0.55},
        whiskerprops={"color": "royalblue"},
        capprops={"color": "royalblue"},
        medianprops={"color": "orange", "linewidth": 1.5},
    )

    hatches = ["///", "\\\\\\\\", "xx"]
    if "boxes" in bp:
        for idx, box in enumerate(bp["boxes"]):
            box.set_hatch(hatches[idx % len(hatches)])

    mp.grid(axis="y", linestyle="--", alpha=0.35)
    mp.ylabel("Test R$^2$")
    mp.title("Missing-value imputation stability: Test R$^2$ across seeds")

    y_all = np.concatenate([np.asarray(v) for v in values_list if len(v) > 0])
    y_min = float(np.min(y_all)) if y_all.size else 0.0
    y_max = float(np.max(y_all)) if y_all.size else 1.0
    y_range = (y_max - y_min) if (y_max - y_min) != 0 else 1.0
    dy = y_range * 0.02

    for i, series in enumerate(values_list, start=1):
        mean_val = float(series.mean()) if len(series) else float("nan")
        std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        mp.text(i, mean_val, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=8)
        mp.text(i, mean_val - dy, f"±{std_val:.3f}", ha="center", va="top", fontsize=8)

    handles = [
        Patch(facecolor="royalblue", edgecolor="royalblue", alpha=0.55, hatch="///", label=labels[0]),
        Patch(facecolor="royalblue", edgecolor="royalblue", alpha=0.55, hatch="\\\\\\\\", label=labels[1]),
        Patch(facecolor="royalblue", edgecolor="royalblue", alpha=0.55, hatch="xx", label=labels[2]),
        Line2D([0], [0], color="orange", lw=2, label="Median (50%)"),
    ]
    mp.legend(handles=handles, loc="best", frameon=False, fontsize=9)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    mp.tight_layout()
    mp.savefig(output_png, bbox_inches="tight")
    mp.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot boxplot of test_R2 from exp_stability_missing_value_imputation.py Excel output."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_EXCEL),
        help="Path to the stability Excel file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. If omitted, save next to input with a fixed filename.",
    )
    args = parser.parse_args()

    input_excel = Path(args.input)
    if not input_excel.exists():
        raise FileNotFoundError(f"Excel not found: {input_excel}")

    values_list = []
    for dataset_name in DATASET_NAMES:
        seed_sheet = build_seed_sheet_name(dataset_name)
        values_list.append(load_test_r2(input_excel, seed_sheet).values)

    if args.output is None:
        output_png = input_excel.with_name("missing_value_imputation_seed_metrics_test_R2_boxplot.jpg")
    else:
        output_png = Path(args.output)

    plot_boxplot(values_list=values_list, labels=DATASET_NAMES, output_png=output_png)
    print(f"Boxplot saved to: {output_png}")


if __name__ == "__main__":
    main()
