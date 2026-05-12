"""
SHAP interpretation stability analysis for this project.

Retained experiment:
1) SHAP interpretation stability
   - full-data SHAP weights vs subset SHAP weights (repeated)
   - metrics: Spearman correlation, L2 distance, Top-k overlap

Excel output format:
   - stability_runs
   - stability_summary
   - weight_stats
"""

from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore", message="ntree_limit is deprecated.*")

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


# ==============================
# Config
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "final_dataset.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "exp_stability_feature_importance_shap.xlsx"

DATASETS = [
    {
        "Dataset": "Q",
        "SheetName": "Sheet_Q",
        "ModelParams": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_child_weight": 1,
            "learning_rate": 0.1,
            "gamma": 0.5,
            "subsample": 0.9,
            "colsample_bytree": 1,
            "n_jobs": -1,
        },
    },
    {
        "Dataset": "R",
        "SheetName": "Sheet_R",
        "ModelParams": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_child_weight": 3,
            "learning_rate": 0.1,
            "gamma": 0.4,
            "subsample": 0.7,
            "colsample_bytree": 1,
        },
    },
]

GLOBAL_SEED = 42
STABILITY_REPEATS = 10
STABILITY_SUBSET_FRAC = 0.9
TOP_K = 5


# ==============================
# Common helpers
# ==============================
def normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = np.sum(weights)
    if total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    s1 = pd.Series(a).rank(method="average")
    s2 = pd.Series(b).rank(method="average")
    corr = s1.corr(s2)
    return float(corr) if corr is not None else np.nan


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    k = min(k, len(a), len(b))
    idx_a = set(np.argsort(a)[-k:])
    idx_b = set(np.argsort(b)[-k:])
    return len(idx_a.intersection(idx_b)) / k


def load_dataset(sheet_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data_df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
    feature_names = list(data_df.columns[:-1])
    data = np.array(data_df)
    x = np.array(data[:, :-1], dtype=float)
    y = np.array(data[:, -1], dtype=float)
    return x, y, feature_names


def build_xgb_model(model_params: Dict, random_state: int) -> XGBRegressor:
    params = dict(model_params)
    params["objective"] = "reg:squarederror"
    params["random_state"] = random_state
    return XGBRegressor(**params)


def compute_shap_weights(x: np.ndarray, y: np.ndarray, model_params: Dict, seed: int) -> np.ndarray:
    model = build_xgb_model(model_params, random_state=seed)
    model.fit(x, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x, check_additivity=False).values
    importance = np.mean(np.abs(shap_values), axis=0)
    return normalize_weights(importance)


# ==============================
# Experiment: SHAP interpretation stability
# ==============================
def run_shap_interpretation_stability(
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    model_params: Dict,
):
    rng = np.random.default_rng(GLOBAL_SEED)
    n_samples, n_features = x.shape
    top_k = min(TOP_K, n_features)

    w_full = compute_shap_weights(x, y, model_params, seed=GLOBAL_SEED)

    stability_rows = []
    subset_size = max(2, int(n_samples * STABILITY_SUBSET_FRAC))

    for repeat_id in range(STABILITY_REPEATS):
        subset_idx = rng.choice(n_samples, size=subset_size, replace=False)
        x_sub = x[subset_idx]
        y_sub = y[subset_idx]
        w_sub = compute_shap_weights(x_sub, y_sub, model_params, seed=GLOBAL_SEED + repeat_id + 1)

        stability_rows.append(
            {
                "Dataset": dataset_name,
                "Repeat": repeat_id + 1,
                "SubsetFrac": STABILITY_SUBSET_FRAC,
                "Spearman": spearman_corr(w_full, w_sub),
                "L2": float(np.linalg.norm(w_full - w_sub)),
                "TopKOverlap": topk_overlap(w_full, w_sub, top_k),
            }
        )

    stability_df = pd.DataFrame(stability_rows)
    stability_summary = {
        "Dataset": dataset_name,
        "SubsetFrac": STABILITY_SUBSET_FRAC,
        "Repeats": STABILITY_REPEATS,
        "Spearman_mean": float(stability_df["Spearman"].mean()),
        "Spearman_std": float(stability_df["Spearman"].std(ddof=0)),
        "L2_mean": float(stability_df["L2"].mean()),
        "L2_std": float(stability_df["L2"].std(ddof=0)),
        "TopKOverlap_mean": float(stability_df["TopKOverlap"].mean()),
        "TopKOverlap_std": float(stability_df["TopKOverlap"].std(ddof=0)),
    }
    return w_full, stability_df, pd.DataFrame([stability_summary])


def main() -> None:
    all_stability_runs = []
    all_stability_summary = []
    all_weight_stats = []

    for dataset_cfg in DATASETS:
        dataset_name = dataset_cfg["Dataset"]
        sheet_name = dataset_cfg["SheetName"]
        model_params = dataset_cfg["ModelParams"]

        print(f"Dataset: {dataset_name}")
        print("[1/1] SHAP interpretation stability")

        x, y, feature_names = load_dataset(sheet_name)
        w_full, stability_df, stability_summary_df = run_shap_interpretation_stability(
            dataset_name,
            x,
            y,
            model_params,
        )

        all_stability_runs.append(stability_df)
        all_stability_summary.append(stability_summary_df)
        all_weight_stats.append(
            pd.DataFrame(
                {
                    "Dataset": dataset_name,
                    "Feature": feature_names,
                    "FeatureIndex": list(range(len(w_full))),
                    "Feature_importance_full": w_full,
                }
            )
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        pd.concat(all_stability_runs, ignore_index=True).to_excel(
            writer, sheet_name="stability_runs", index=False
        )
        pd.concat(all_stability_summary, ignore_index=True).to_excel(
            writer, sheet_name="stability_summary", index=False
        )
        pd.concat(all_weight_stats, ignore_index=True).to_excel(
            writer, sheet_name="feature_importance_stats", index=False
        )

    print(f"\n[Saved] -> {OUTPUT_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
