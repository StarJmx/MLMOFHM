from __future__ import unicode_literals

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.metrics as sm
import sklearn.utils as su
from xgboost import XGBRegressor as xGB


# =============================================================================
# Parameters (edit here)
# =============================================================================
REPO_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_DIR / "data" / "final_dataset.xlsx"
OUTPUT_PATH = REPO_DIR / "data" / "stability_split90_10_Q_R.xlsx"

# Which targets to run: "Q", "R", or ["Q","R"]
RUN_TARGETS = ["Q", "R"]

TRAIN_RATIO = 0.9  # 90/10 split

# 10 random seeds (fixed, so results are reproducible)
SEEDS = [30, 42, 66, 99, 999, 1888, 7777, 13141, 52013, 354917]


# =============================================================================
# Target definitions (Q/R数据在文件开始标记好)
# =============================================================================
Q_SPEC = {
    "target": "Q",
    "sheet_name": "Sheet_Q",
    "model_params": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_child_weight": 1,
        "learning_rate": 0.1,
        "gamma": 0.5,
        "subsample": 0.9,
        "colsample_bytree": 1,
    },
}

R_SPEC = {
    "target": "R",
    "sheet_name": "Sheet_R",
    "model_params": {
        "n_estimators": 200,
        "max_depth": 15,
        "min_child_weight": 3,
        "learning_rate": 0.1,
        "gamma": 0.4,
        "subsample": 0.7,
        "colsample_bytree": 1,
    },
}


def load_xy(sheet_name: str):
    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)
    data = np.array(df)
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def split_dataset_90_10(x, y, *, seed: int, train_ratio: float):
    """
    90/10 拆分逻辑：
    1) 用 seed 对 (x, y) 进行 shuffle，然后按 90/10 切分
    """
    x, y = su.shuffle(x, y, random_state=seed)
    train_size = int(len(x) * train_ratio)
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    return train_x, test_x, train_y, test_y


def evaluate(train_y, train_pred_y, test_y, test_pred_y):
    return {
        "train_MAE": float(sm.mean_absolute_error(train_y, train_pred_y)),
        "train_RMSE": float(np.sqrt(sm.mean_squared_error(train_y, train_pred_y))),
        "train_R2": float(sm.r2_score(train_y, train_pred_y)),
        "test_MAE": float(sm.mean_absolute_error(test_y, test_pred_y)),
        "test_RMSE": float(np.sqrt(sm.mean_squared_error(test_y, test_pred_y))),
        "test_R2": float(sm.r2_score(test_y, test_pred_y)),
    }


def run_for_target(spec: dict, seeds: list, train_ratio: float) -> (pd.DataFrame, pd.DataFrame):
    x, y = load_xy(spec["sheet_name"])

    rows = []
    for seed in seeds:
        train_x, test_x, train_y, test_y = split_dataset_90_10(
            x,
            y,
            seed=seed,
            train_ratio=train_ratio,
        )
        model = xGB(**spec["model_params"])
        model.fit(train_x, train_y)

        train_pred_y = model.predict(train_x)
        test_pred_y = model.predict(test_x)

        metrics = evaluate(train_y, train_pred_y, test_y, test_pred_y)
        rows.append(
            {
                "target": spec["target"],
                "seed": seed,
                "n_train": int(train_y.shape[0]),
                "n_test": int(test_y.shape[0]),
                **metrics,
            }
        )

    seed_df = pd.DataFrame(rows)

    metric_cols = [c for c in seed_df.columns if c not in {"target", "seed", "n_train", "n_test"}]
    summary_df = seed_df[metric_cols].agg(["mean", "std"])
    summary_df = summary_df.T.reset_index().rename(columns={"index": "metric"})
    summary_df.insert(0, "target", spec["target"])
    summary_df = summary_df.rename(columns={"mean": "mean", "std": "std"})

    return seed_df, summary_df


def write_excel(output_path: Path, *, run_targets: list, seeds: list, train_ratio: float, seed_dfs: dict, summary_dfs: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        config_df = pd.DataFrame(
            [
                {"key": "experiment", "value": "stability_split90_10"},
                {"key": "train_ratio", "value": train_ratio},
                {"key": "targets", "value": ",".join(run_targets)},
                {"key": "n_seeds", "value": len(seeds)},
                {"key": "seeds", "value": ",".join(map(str, seeds))},
            ]
        )
        config_df.to_excel(writer, index=False, sheet_name="config")

        for t in run_targets:
            seed_dfs[t].to_excel(writer, index=False, sheet_name=f"{t}_seed_metrics")
            summary_dfs[t].to_excel(writer, index=False, sheet_name=f"{t}_summary")


def main():
    target_specs = {"Q": Q_SPEC, "R": R_SPEC}
    run_targets = RUN_TARGETS

    seed_dfs = {}
    summary_dfs = {}
    for t in run_targets:
        seed_df, summary_df = run_for_target(target_specs[t], seeds=SEEDS, train_ratio=TRAIN_RATIO)
        seed_dfs[t] = seed_df
        summary_dfs[t] = summary_df

    write_excel(
        OUTPUT_PATH,
        run_targets=run_targets,
        seeds=SEEDS,
        train_ratio=TRAIN_RATIO,
        seed_dfs=seed_dfs,
        summary_dfs=summary_dfs,
    )
    print(f"Excel results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

