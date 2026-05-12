import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor as xGB
import sklearn.utils as su
import sklearn.metrics as sm
import matplotlib.pyplot as mp
import shap
from itertools import combinations


MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_child_weight": 1,
    "learning_rate": 0.1,
    "gamma": 0.5,
    "subsample": 0.9,
    "colsample_bytree": 1,
    "n_jobs": -1,
}

dataset_names = [
    "Mean imputation",
    "RF imputation",
    "Hybrid imputation"
]


SEEDS = [30, 42, 66, 99, 999, 1888, 7777, 13141, 52013, 354917]


def build_model():
    return xGB(**MODEL_PARAMS)


def evaluate_model(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)

    pred_train_y = model.predict(train_x)
    print("Training set MAE: ", sm.mean_absolute_error(train_y, pred_train_y))
    print("Training set RMSE: ", np.sqrt(sm.mean_squared_error(train_y, pred_train_y)))
    print("Training set R2: ", sm.r2_score(train_y, pred_train_y))

    pred_test_y = model.predict(test_x)
    print("Test set MAE: ", sm.mean_absolute_error(test_y, pred_test_y))
    print("Test set RMSE: ", np.sqrt(sm.mean_squared_error(test_y, pred_test_y)))
    print("Test set R2: ", sm.r2_score(test_y, pred_test_y))

    return pred_train_y, pred_test_y


def update_plot_config(font_size):
    config = {
        "font.family": "Times New Roman",
        "font.size": font_size,
        "mathtext.fontset": "stix",
    }
    mp.rcParams.update(config)
    mp.rcParams["xtick.direction"] = "in"
    mp.rcParams["ytick.direction"] = "in"
    mp.rcParams["axes.unicode_minus"] = False
    mp.rcParams["font.sans-serif"] = ["Times New Roman"]


def plot_prediction_results(train_y, pred_train_y, test_y, pred_test_y):
    mp.figure(dataset_name, figsize=(1.8, 1.6), facecolor="white", dpi=300)
    update_plot_config(font_size=8)
    mp.title(dataset_name, fontsize=8)
    mp.xlim(0, 800)
    mp.ylim(0, 800)
    mp.grid(linestyle="")

    if len(test_y) > 0:
        r2_test = sm.r2_score(test_y, pred_test_y)
        mae_test = sm.mean_absolute_error(test_y, pred_test_y)
        rmse_test = np.sqrt(sm.mean_squared_error(test_y, pred_test_y))
        mp.text(420, 150, rf"R$^2$={r2_test:.3f}")
        mp.text(420, 90, rf"MAE={mae_test:.3f}")
        mp.text(420, 30, rf"RMSE={rmse_test:.3f}")
    mp.xlabel("Predicted Q(mg/g)")
    mp.ylabel("Actual Q(mg/g)")

    linex = np.array([0, 800])
    liney = np.array([0, 800])
    mp.plot(linex, liney, color="limegreen", linewidth=0.5, linestyle="--")

    mp.scatter(
        pred_train_y,
        train_y,
        s=1,
        c="royalblue",
        label="Train",
        alpha=0.8,
    )
    mp.scatter(
        pred_test_y,
        test_y,
        s=1,
        c="red",
        label="Test",
        alpha=0.8,
    )
    mp.legend(loc="upper left")
    mp.show()


def load_dataset(dataset):
    data_df = pd.read_excel(data_path, sheet_name=dataset)
    all_columns = data_df.columns
    feature_names = all_columns[:-1]

    print(all_columns)
    data = np.array(data_df)
    print(data.shape)

    x = data[:, :-1]
    y = data[:, -1]
    print(x.shape)
    print(y.shape)
    print(x[0], y[0])
    return x, y, feature_names, all_columns


def split_dataset(x, y):
    x, y = su.shuffle(x, y, random_state=42)
    train_size = int(len(x) * 0.9)
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    print(train_x.shape, train_y.shape)
    return train_x, test_x, train_y, test_y


def split_dataset_by_seed(x, y, seed):
    x, y = su.shuffle(x, y, random_state=seed)
    train_size = int(len(x) * 0.9)
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    return train_x, test_x, train_y, test_y


def collect_metrics(train_y, pred_train_y, test_y, pred_test_y):
    return {
        "train_MAE": float(sm.mean_absolute_error(train_y, pred_train_y)),
        "train_RMSE": float(np.sqrt(sm.mean_squared_error(train_y, pred_train_y))),
        "train_R2": float(sm.r2_score(train_y, pred_train_y)),
        "test_MAE": float(sm.mean_absolute_error(test_y, pred_test_y)),
        "test_RMSE": float(np.sqrt(sm.mean_squared_error(test_y, pred_test_y))),
        "test_R2": float(sm.r2_score(test_y, pred_test_y)),
    }


def compute_shap_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = build_model()
    model.fit(x, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(x, check_additivity=False).values
    importance = np.mean(np.abs(shap_values), axis=0)
    return normalize_weights(importance)


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


def prepare_data():
    # ==============================
    # Label encoding
    # ==============================
    # Define label encoder
    le = LabelEncoder()
    # Get the first three columns and the second-to-last column names
    first_three_cols = data.columns[:3]
    last_second_col = data.columns[-2]
    # Merge columns to encode
    columns_to_encode = list(first_three_cols) + [last_second_col]
    # Perform label encoding on these columns
    for col in columns_to_encode:
        data[col] = le.fit_transform(data[col].astype(str))
    target = data['Tim']
    features = data.iloc[:, :-1]
    print(features)
    print(target)

    # Define x and auxiliary y
    X_full, y_full = features, target
    n_samples = X_full.shape[0]
    n_features = X_full.shape[1]
    print(n_samples)
    print(n_features)
    return X_full, y_full


def impute_missing_values(X_full, y_full, strategy='rf'):
    # Count missing values
    X_missing_reg = X_full.copy()
    missing = X_missing_reg.isna().sum()
    missing = pd.DataFrame(data={'Feature': missing.index, 'MissingCount': missing.values})
    missing = missing[~missing['MissingCount'].isin([0])]
    missing['MissingRatio'] = missing['MissingCount'] / X_missing_reg.shape[0]
    X_df = X_missing_reg.isnull().sum()
    print(missing)

    if strategy == 'mean':
        X_missing_reg = pd.DataFrame(
            SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X_missing_reg),
            columns=X_full.columns,
            index=X_full.index,
        )
    elif strategy == 'rf':
        colname = X_df[~X_df.isin([0])].sort_values().index.values
        sortindex = []
        for i in colname:
            sortindex.append(X_missing_reg.columns.tolist().index(str(i)))

        for i in sortindex:
            df = X_missing_reg
            fillc = df.iloc[:, i]
            df = pd.concat([df.drop(df.columns[i], axis=1), pd.DataFrame(y_full)], axis=1)
            df_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df)

            y_train = fillc[fillc.notnull()]
            y_test = fillc[fillc.isnull()]
            x_train = df_0[y_train.index, :]
            x_test = df_0[y_test.index, :]

            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(x_train, y_train)
            y_predict = rfc.predict(x_test)

            X_missing_reg.loc[X_missing_reg.iloc[:, i].isnull(), X_missing_reg.columns[i]] = y_predict
    else:
        raise ValueError("strategy must be 'mean' or 'rf'")

    missing2 = X_missing_reg.isna().sum()
    missing2 = pd.DataFrame(data={'Column': missing2.index, 'MissingCount': missing2.values})
    missing3 = missing2[~missing2['MissingCount'].isin([0])]
    print(missing2)
    print(missing3)
    return X_missing_reg


def main():
    X_full, y_full = prepare_data()
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = base_dir / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Options: 'mean' or 'rf'
    strategy = 'rf'
    X_missing_reg = impute_missing_values(X_full, y_full, strategy=strategy)

    if strategy == 'mean':
        X_missing_reg.to_excel(output_dir / "Dataset after mean interpolation.xlsx")
    else:
        X_missing_reg.to_excel(output_dir / "Dataset after RF interpolation.xlsx")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "Sensitivity Analysis of Missing Data Imputation Methods.xlsx"
    exp_output_path = base_dir / "data" / "exp_stability_missing_value_imputation_metrics.xlsx"
    shap_output_path = base_dir / "data" / "exp_stability_missing_value_imputation_shap.xlsx"

    # exp_target: interpolation, exp_stability_imputation, exp_stability_shap
    work = "exp_stability_shap"

    if work == "exp_stability_shap":
        top_k = 10
        stability_rows = []
        weight_map = {}
        feature_names_map = {}

        for dataset_name in dataset_names:
            print(f"\n===== {dataset_name} =====")
            x, y, feature_names, all_columns = load_dataset(dataset_name)
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            w_full = compute_shap_weights(x, y)
            weight_map[dataset_name] = w_full
            feature_names_map[dataset_name] = list(feature_names)

        for left_name, right_name in combinations(dataset_names, 2):
            w_left = weight_map[left_name]
            w_right = weight_map[right_name]
            stability_rows.append(
                {
                    "Dataset": f"{right_name} vs {left_name}",
                    "Spearman": spearman_corr(w_left, w_right),
                    "TopKOverlap": topk_overlap(w_left, w_right, top_k),
                }
            )

        stability_df = pd.DataFrame(stability_rows)
        stability_summary_df = pd.DataFrame(
            [
                {
                    "ComparisonPairs": len(stability_df),
                    "Spearman_mean": float(stability_df["Spearman"].mean()),
                    "Spearman_std": float(stability_df["Spearman"].std(ddof=0)),
                    "TopKOverlap_mean": float(stability_df["TopKOverlap"].mean()),
                    "TopKOverlap_std": float(stability_df["TopKOverlap"].std(ddof=0)),
                }
            ]
        )

        weight_rows = []
        for dataset_name in dataset_names:
            w_full = weight_map[dataset_name]
            feature_names = feature_names_map[dataset_name]
            for idx, (feature_name, weight_val) in enumerate(zip(feature_names, w_full)):
                weight_rows.append(
                    {
                        "Dataset": dataset_name,
                        "Feature": feature_name,
                        "FeatureIndex": idx,
                        "Feature_importance_full": weight_val,
                    }
                )
        weight_stats_df = pd.DataFrame(weight_rows)

        shap_output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(shap_output_path) as writer:
            stability_df.to_excel(writer, sheet_name="stability_runs", index=False)
            stability_summary_df.to_excel(writer, sheet_name="stability_summary", index=False)
            weight_stats_df.to_excel(writer, sheet_name="feature_importance_stats", index=False)

        print(f"Excel results saved to: {shap_output_path}")

    if work == "exp_stability_imputation":
        seed_dfs = {}
        summary_dfs = {}

        for dataset_name in dataset_names:
            print(f"\n===== {dataset_name} =====")
            x, y, feature_names, all_columns = load_dataset(dataset_name)

            rows = []
            for seed in SEEDS:
                train_x, test_x, train_y, test_y = split_dataset_by_seed(x, y, seed)
                model = build_model()
                pred_train_y, pred_test_y = evaluate_model(model, train_x, train_y, test_x, test_y)
                metrics = collect_metrics(train_y, pred_train_y, test_y, pred_test_y)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "n_train": int(train_y.shape[0]),
                        "n_test": int(test_y.shape[0]),
                        **metrics,
                    }
                )

            seed_df = pd.DataFrame(rows)
            metric_cols = [c for c in seed_df.columns if c not in {"dataset", "seed", "n_train", "n_test"}]
            summary_df = seed_df[metric_cols].agg(["mean", "std"])
            summary_df = summary_df.T.reset_index().rename(columns={"index": "metric"})
            summary_df.insert(0, "dataset", dataset_name)

            seed_dfs[dataset_name] = seed_df
            summary_dfs[dataset_name] = summary_df

        exp_output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(exp_output_path) as writer:
            config_df = pd.DataFrame(
                [
                    {"key": "experiment", "value": "stability_missing_value_imputation"},
                    {"key": "train_ratio", "value": 0.9},
                    {"key": "datasets", "value": ",".join(dataset_names)},
                    {"key": "n_seeds", "value": len(SEEDS)},
                    {"key": "seeds", "value": ",".join(map(str, SEEDS))},
                ]
            )
            config_df.to_excel(writer, index=False, sheet_name="config")

            for dataset_name in dataset_names:
                sheet_prefix = dataset_name.replace(" ", "_")[:20]
                seed_dfs[dataset_name].to_excel(writer, index=False, sheet_name=f"{sheet_prefix}_seed")
                summary_dfs[dataset_name].to_excel(writer, index=False, sheet_name=f"{sheet_prefix}_summary")

        print(f"Excel results saved to: {exp_output_path}")

    if work == "interpolation":
        # data = pd.read_excel(data_path, sheet_name="Ready dataset")
        data = pd.read_excel(data_path, sheet_name="Hybrid imputation")
        main()
