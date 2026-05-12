from __future__ import unicode_literals

"""
Refactored version of `XGB_Q_MOFsCrPbCdAs.py`.

This script keeps the original workflow, parameters, paths, and figures,
but organizes the code into reusable functions for publication/repository use.
"""

from pathlib import Path
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as mp
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import sklearn.model_selection as ms
import sklearn.utils as su
from sklearn.inspection import plot_partial_dependence
from xgboost import XGBRegressor as xGB


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "final_dataset.xlsx"
MODEL_PATH = BASE_DIR / "pkl" / "MOFsCrPbCdAs_XGB_Q.pkl"
GRID_SEARCH_RESULTS_PATH = BASE_DIR / "data" / "grid_search_5cv_results_Q.csv"
SHEET_NAME = "Sheet_Q"

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

SHUFFLE_RANDOM_STATE = 53
TRAIN_SHUFFLE_RANDOM_STATE = 30
TRAIN_RATIO = 0.9
SAVE_MODEL = False

# Used to label multiple prediction plots generated from different test subsets.
PLOT_FIG_COUNTER = -1
PLOT_FIG_HM = ["full_test","Cr", "Pb", "Cd", "As"]
PLOT_TITLE_OVERRIDE = None


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


def load_dataset():
    data_df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
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
    x, y = su.shuffle(x, y, random_state=SHUFFLE_RANDOM_STATE)
    train_size = int(len(x) * TRAIN_RATIO)
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]
    train_x, train_y = su.shuffle(
        train_x,
        train_y,
        random_state=TRAIN_SHUFFLE_RANDOM_STATE,
    )
    print(train_x.shape, train_y.shape)
    return train_x, test_x, train_y, test_y


def build_model():
    return xGB(**MODEL_PARAMS)


def run_grid_search(model, train_x, train_y):
    full_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1, 1],
        "gamma": [0, 0.05, 0.1, 0.5],
        "min_child_weight": [1, 3, 5, 7],
        "max_depth": [3, 5, 8, 10, 15],
        "subsample": [0.9, 1],
        "colsample_bytree": [0.9, 1],
    }


    grid_search = ms.GridSearchCV(model, full_param_grid, cv=5)
    grid_search.fit(train_x, train_y)
    print(grid_search.cv_results_["params"])
    print(grid_search.cv_results_["mean_test_score"])
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)

    cv_results = grid_search.cv_results_
    fold_columns = [f"split{i}_test_score" for i in range(5)]
    result_rows = []
    for idx, params in enumerate(cv_results["params"]):
        row = {}
        row.update(params)
        for fold_col in fold_columns:
            row[fold_col] = cv_results[fold_col][idx]
        row["mean_test_score"] = cv_results["mean_test_score"][idx]
        row["std_test_score"] = cv_results["std_test_score"][idx]
        result_rows.append(row)

    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(GRID_SEARCH_RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"Grid search 5-fold results saved to: {GRID_SEARCH_RESULTS_PATH}")

    return grid_search, full_param_grid


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


def evaluate_model_split_test_by_last_feature(
    model,
    train_x,
    train_y,
    test_x,
    test_y,
):
    """
    Split (test_x, test_y) into 4 subtest sets by `test_x` last column:
      1 -> Cr, 2 -> Pb, 3 -> Cd, 4 -> As
    Train the model once (same as `evaluate_model`), then evaluate/predict
    each subtest set and plot 4 prediction figures.
    """
    global PLOT_FIG_COUNTER, PLOT_TITLE_OVERRIDE

    model.fit(train_x, train_y)

    pred_train_y = model.predict(train_x)
    print("Training set MAE: ", sm.mean_absolute_error(train_y, pred_train_y))
    print("Training set RMSE: ", np.sqrt(sm.mean_squared_error(train_y, pred_train_y)))
    print("Training set R2: ", sm.r2_score(train_y, pred_train_y))

    pred_test_y_full = model.predict(test_x)
    metal_col_int = np.round(np.asarray(test_x)[:, -1]).astype(int)

    # Plot full test set (same style as the original `evaluate_model` workflow).
    PLOT_TITLE_OVERRIDE = None
    print("Test set MAE: ", sm.mean_absolute_error(test_y, pred_test_y_full))
    print("Test set RMSE: ", np.sqrt(sm.mean_squared_error(test_y, pred_test_y_full)))
    print("Test set R2: ", sm.r2_score(test_y, pred_test_y_full))
    plot_prediction_results(train_y, pred_train_y, test_y, pred_test_y_full)

    code_to_title = {
        1: "(a)Cr",
        2: "(b)Pb",
        3: "(c)Cd",
        4: "(d)As",
    }

    # Return as 4 groups: (pred_train_y, pred_test_y_xxx)
    groups = []
    for code in [1, 2, 3, 4]:
        mask = metal_col_int == code
        test_y_sub = np.asarray(test_y)[mask]
        pred_test_y_sub = pred_test_y_full[mask]

        print(f"Sub-test set ({code_to_title.get(code, code)}):")
        if test_y_sub.size == 0:
            print("  (empty) skip metrics & plot")
            groups.append((pred_train_y, pred_test_y_sub))
            continue

        print("  MAE: ", sm.mean_absolute_error(test_y_sub, pred_test_y_sub))
        print("  RMSE: ", np.sqrt(sm.mean_squared_error(test_y_sub, pred_test_y_sub)))
        print("  R2: ", sm.r2_score(test_y_sub, pred_test_y_sub))

        PLOT_TITLE_OVERRIDE = code_to_title.get(code, f"Code{code}")
        plot_prediction_results(train_y, pred_train_y, test_y_sub, pred_test_y_sub)
        groups.append((pred_train_y, pred_test_y_sub))

    return tuple(groups)


def save_model(model):
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
        print("dump success!")


def load_model():
    with open(MODEL_PATH, "rb") as file:
        loaded_model = pickle.load(file)
        print("load success!")
    return loaded_model


def plot_prediction_results(train_y, pred_train_y, test_y, pred_test_y):
    global PLOT_FIG_COUNTER, PLOT_TITLE_OVERRIDE

    PLOT_FIG_COUNTER += 1
    fig_name = f"XGB_Q_pred_{PLOT_FIG_HM[PLOT_FIG_COUNTER]}"
    plot_title = PLOT_TITLE_OVERRIDE if PLOT_TITLE_OVERRIDE is not None else "(a)MQ"

    mp.figure(fig_name, figsize=(1.8, 1.6), facecolor="white", dpi=600)
    update_plot_config(font_size=8)
    mp.title(plot_title, fontsize=8)
    mp.xlim(0, 800)
    mp.ylim(0, 800)
    mp.grid(linestyle="")

    if len(test_y) > 0:
        r2_test = sm.r2_score(test_y, pred_test_y)
        mae_test = sm.mean_absolute_error(test_y, pred_test_y)
        rmse_test = np.sqrt(sm.mean_squared_error(test_y, pred_test_y))
        mp.text(420, 150, rf"R$^2$={r2_test:.2f}")
        mp.text(420, 90, rf"MAE={mae_test:.2f}")
        mp.text(420, 30, rf"RMSE={rmse_test:.2f}")
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


def plot_feature_importance(model, feature_names):
    cmap = cm.get_cmap("winter_r")
    rf_fi = model.feature_importances_ * 100
    print("XGB_FI:", rf_fi)

    update_plot_config(font_size=8)
    mp.figure("Feature Importance", facecolor="white", figsize=(3, 2.5), dpi=250)
    mp.title("(a)MQ")
    mp.grid(linestyle="")
    mp.xlabel("Percentage(%)", fontsize=8)
    mp.ylabel("Feature importance", fontsize=8)

    feature_x = np.arange(rf_fi.size)
    sorted_indices = rf_fi.argsort()
    rf_fi = rf_fi[sorted_indices]

    mp.barh(feature_x, rf_fi, height=0.7, color=cmap((rf_fi - 0) / (22 - 0)))
    for x_value, y_value in zip(feature_x, rf_fi):
        mp.text(y_value, x_value - 0.02, "%.2f" % y_value, va="center")

    mp.yticks(feature_x, feature_names[sorted_indices], rotation=0)
    mp.tight_layout()
    mp.show()


def reload_full_dataset():
    data_df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    all_columns = data_df.columns
    print(data_df.shape)

    data = np.array(data_df)
    x = data[:, :-1]
    y = data[:, -1]
    print(x.shape)
    print(y.shape)
    return x, y, all_columns


def plot_partial_dependence_grid(model, x, feature_names):
    update_plot_config(font_size=14)
    fig = mp.figure(figsize=(7, 5))
    grid = mp.GridSpec(2, 3, hspace=0.3, wspace=0.4)

    ax1 = fig.add_subplot(grid[0, 0])
    disp1 = plot_partial_dependence(model, x, [6], ax=ax1, feature_names=feature_names)
    axes1 = disp1.figure_.get_axes()
    for ax in axes1:
        ax.set_xlim(50, 350)
        ax.set_ylim(85, 95)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    ax2 = fig.add_subplot(grid[0, 1])
    plot_partial_dependence(model, x, [9], ax=ax2, feature_names=feature_names)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    ax3 = fig.add_subplot(grid[0, 2])
    plot_partial_dependence(model, x, [11], ax=ax3, feature_names=feature_names)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    ax4 = fig.add_subplot(grid[1, 0])
    plot_partial_dependence(model, x, [7], ax=ax4, feature_names=feature_names)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    ax5 = fig.add_subplot(grid[1, 1])
    plot_partial_dependence(model, x, [8], ax=ax5, feature_names=feature_names)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    ax6 = fig.add_subplot(grid[1, 2])
    plot_partial_dependence(model, x, [10], ax=ax6, feature_names=feature_names)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    mp.subplots_adjust(top=0.94, bottom=0.1, left=0.1, right=0.95)
    mp.tight_layout()
    mp.show()


def plot_partial_dependence_three_panels(model, x, feature_names):
    update_plot_config(font_size=14)
    fig = mp.figure(figsize=(7, 2.7))
    grid = mp.GridSpec(1, 3, hspace=0.3, wspace=0.4)

    ax1 = fig.add_subplot(grid[0, 0])
    disp1 = plot_partial_dependence(model, x, [4], ax=ax1, feature_names=feature_names)
    axes1 = disp1.figure_.get_axes()
    for ax in axes1:
        ax.set_xlim(0, 7)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    ax2 = fig.add_subplot(grid[0, 1])
    plot_partial_dependence(model, x, [3], ax=ax2, feature_names=feature_names)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    ax3 = fig.add_subplot(grid[0, 2])
    plot_partial_dependence(model, x, [5], ax=ax3, feature_names=feature_names)
    mp.xticks(fontsize=10)
    mp.yticks(fontsize=10)
    mp.locator_params(axis="x", nbins=5)

    mp.subplots_adjust(top=0.89, bottom=0.2, left=0.1, right=0.95)
    mp.tight_layout()
    mp.show()


def main():
    x, y, feature_names, all_columns = load_dataset()
    train_x, test_x, train_y, test_y = split_dataset(x, y)

    model = build_model()
    # Optional grid search block kept from the original script:
    # run_grid_search(model, train_x, train_y)

    # Keep the same workflow as the original script:
    # first train/evaluate, then load the previously saved model for plotting.
    _ = evaluate_model_split_test_by_last_feature(model, train_x, train_y, test_x, test_y)

    # Original optional model-saving block:
    # with open('pkl/MOFsCrPbCdAs_XGB_Q.pkl', 'wb') as f:
    #     pickle.dump(model, f)
    #     print('dump success!')
    if SAVE_MODEL:
        save_model(model)

    # Evaluate performance and tree-based feature importance (prediction plots already generated in sub-test set function)
    plot_feature_importance(model, feature_names)

    # Load the final saved model for global analysis
    model = load_model()

    x_full, y_full, all_columns = reload_full_dataset()
    _ = y_full

    plot_partial_dependence_grid(model, x_full, all_columns)
    plot_partial_dependence_three_panels(model, x_full, all_columns)


if __name__ == "__main__":
    main()
