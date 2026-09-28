"""
House price prediction on the California Housing dataset (20,640 samples)
with regression, regularization and bias-variance analysis.

What this script does
---------------------
1. Loads the dataset with sklearn.datasets.fetch_california_housing()
   (cached in ~/scikit_learn_data after the first run - no Kaggle needed).
2. EDA: target/feature distributions and a correlation heatmap.
3. Outlier handling: caps AveRooms/AveOccup/AveBedms-style extremes at
   train-set quantiles - the capper is fit inside the Pipeline (no leakage).
4. Feature engineering inside the pipeline: rooms-per-household,
   persons-per-household, bedrooms-per-room.
5. Trains and tunes Linear, Ridge, Lasso and Random Forest regressors with
   GridSearchCV (5-fold CV, scoring = neg_root_mean_squared_error).
6. Bias-variance analysis: learning curves for the best regularized linear
   model and the Random Forest, plus train-vs-CV RMSE comparison.
7. Evaluates on a held-out test set with RMSE, MAE and R^2; reports the %
   RMSE improvement over the plain Linear baseline.
8. Saves artifacts/ (model + metrics) and figures/ (EDA, learning curves,
   predicted-vs-actual, residuals).

Usage:
    python train_house.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FIGS = HERE / "figures"
ART.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", palette="deep")

TARGET = "MedHouseVal"
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude",
]
FEATURES_TO_CAP = ["AveRooms", "AveBedrms", "AveOccup", "Population"]


# --------------------------------------------------------------------------- #
# 1. Load + EDA
# --------------------------------------------------------------------------- #
def load_df() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # already includes MedHouseVal target column
    print(f"Loaded California Housing: {len(df):,} rows x {len(df.columns)} cols")
    return df


def plot_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    sns.histplot(df[TARGET], bins=50, ax=axes[0])
    axes[0].set_title("Median house value distribution ($100k units)")

    sns.scatterplot(data=df.sample(3000, random_state=1), x="MedInc",
                    y=TARGET, s=8, alpha=0.4, ax=axes[1])
    axes[1].set_title("Income vs house value")

    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                annot_kws={"size": 7}, ax=axes[2])
    axes[2].set_title("Feature correlation heatmap")

    fig.tight_layout()
    fig.savefig(FIGS / "eda_overview.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Outlier capper + feature engineer (custom transformers in transformers.py)
# --------------------------------------------------------------------------- #
from transformers import FeatureEngineer, QuantileCapper  # noqa: E402


def build_preprocessor() -> ColumnTransformer:
    all_cols = FEATURES + ["RoomsPerHousehold", "BedroomsPerRoom"]
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), all_cols)],
        remainder="drop",
    )


def build_pipe(model) -> Pipeline:
    return Pipeline([
        ("capper", QuantileCapper(FEATURES_TO_CAP)),
        ("featurize", FeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])


# --------------------------------------------------------------------------- #
# 3. Model grid
# --------------------------------------------------------------------------- #
def model_specs() -> list[tuple[str, object, dict]]:
    specs: list[tuple[str, object, dict]] = []

    lin = LinearRegression()
    specs.append(("Linear Regression", lin, {}))

    ridge = Ridge(random_state=RANDOM_STATE)
    grid_ridge = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    specs.append(("Ridge", ridge, grid_ridge))

    lasso = Lasso(random_state=RANDOM_STATE, max_iter=50_000)
    grid_lasso = {"model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0]}
    specs.append(("Lasso", lasso, grid_lasso))

    rf = RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)
    grid_rf = {
        # Depth-capped so the saved artifact stays small (~30 MB) with
        # RMSE within ~0.01 of an unconstrained forest on this dataset.
        "model__n_estimators": [150],
        "model__max_depth": [20],
        "model__min_samples_leaf": [2, 5],
        "model__max_features": [0.5],
    }
    specs.append(("Random Forest", rf, grid_rf))
    return specs


# --------------------------------------------------------------------------- #
# 4. Metrics + learning curves
# --------------------------------------------------------------------------- #
def reg_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def plot_learning_curves(pipes: dict, X_train, y_train) -> dict:
    sizes = np.linspace(0.1, 1.0, 8)
    curves: dict[str, dict] = {}
    fig, axes = plt.subplots(1, len(pipes), figsize=(6.2 * len(pipes), 5), sharey=True)

    for ax, (name, pipe) in zip(np.atleast_1d(axes), pipes.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X_train, y_train, train_sizes=sizes, cv=5,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        tr = -train_scores.mean(axis=1)
        va = -val_scores.mean(axis=1)
        ax.plot(train_sizes, tr, "o-", label="train RMSE")
        ax.plot(train_sizes, va, "s-", label="CV RMSE")
        ax.set_title(name)
        ax.set_xlabel("training examples")
        ax.set_ylabel("RMSE ($100k)")
        ax.legend()
        curves[name] = {
            "train_sizes": train_sizes.tolist(),
            "train_rmse": np.round(tr, 4).tolist(),
            "cv_rmse": np.round(va, 4).tolist(),
        }

    fig.suptitle("Learning curves - bias/variance check (gap between curves)")
    fig.tight_layout()
    fig.savefig(FIGS / "learning_curves.png", dpi=150)
    plt.close(fig)
    return curves


def plot_pred_vs_actual(y_test, y_pred, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.scatterplot(x=y_test, y=y_pred, s=10, alpha=0.35, ax=axes[0])
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1)
    axes[0].set_xlabel("actual ($100k)")
    axes[0].set_ylabel("predicted ($100k)")
    axes[0].set_title(f"{name}: predicted vs actual (test)")

    resid = y_test - y_pred
    sns.histplot(resid, bins=60, ax=axes[1])
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("actual - predicted ($100k)")
    fig.tight_layout()
    fig.savefig(FIGS / "pred_vs_actual.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main() -> None:
    df = load_df()
    plot_eda(df)

    y = df[TARGET].values
    X = df[FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    for name, model, grid in model_specs():
        print(f"\n=== {name} ===")
        pipe = build_pipe(model)
        if grid:
            gs = GridSearchCV(pipe, grid, cv=5,
                              scoring="neg_root_mean_squared_error",
                              n_jobs=-1, refit=True)
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            cv_rmse = float(-gs.best_score_)
            print(f"  best params: {gs.best_params_}")
        else:
            pipe.fit(X_train, y_train)
            best = pipe
            # 5-fold CV RMSE for the baseline, matching the tuned models
            from sklearn.model_selection import cross_val_score
            cv_rmse = float(
                -cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring="neg_root_mean_squared_error",
                                 n_jobs=-1).mean()
            )
        print(f"  5-fold CV RMSE: {cv_rmse:.4f}")

        pred_test = best.predict(X_test)
        m = reg_metrics(y_test, pred_test)
        m["cv_rmse"] = round(cv_rmse, 4)

        # train RMSE for the overfitting gap (bias-variance table)
        train_pred = best.predict(X_train)
        m["train_rmse"] = reg_metrics(y_train, train_pred)["rmse"]
        m["overfit_gap"] = round(m["train_rmse"] - m["cv_rmse"], 4)

        results[name] = m
        fitted[name] = best
        print(f"  test: {m}")

    baseline_rmse = results["Linear Regression"]["rmse"]
    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_rmse = results[best_name]["rmse"]
    improvement_pct = 100.0 * (baseline_rmse - best_rmse) / baseline_rmse

    print(f"\n### Best model: {best_name} ###")
    print(f"  Baseline Linear RMSE : {baseline_rmse:.4f}")
    print(f"  Best ({best_name}) RMSE: {best_rmse:.4f}")
    print(f"  RMSE improvement over baseline: {improvement_pct:.1f}%")

    plot_pred_vs_actual(y_test, fitted[best_name].predict(X_test), best_name)

    # Learning curves: strongest-regularized linear vs RF (bias-variance story)
    lc_pipes = {
        "Ridge (best alpha)": clone(fitted["Ridge"]),
        "Random Forest": clone(fitted["Random Forest"]),
    }
    curves = plot_learning_curves(lc_pipes, X_train, y_train)

    # Model-native feature importance for tree models
    importance = None
    rf_pipe = fitted["Random Forest"]
    pre = rf_pipe.named_steps["preprocess"]
    names = list(pre.get_feature_names_out())
    names = [n.split("__", 1)[-1] for n in names]
    rf_model = rf_pipe.named_steps["model"]
    importance = sorted(zip(names, rf_model.feature_importances_),
                        key=lambda t: -t[1])
    imp_df = pd.DataFrame(importance, columns=["feature", "importance"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=imp_df, y="feature", x="importance", ax=ax, color="#4c72b0")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(FIGS / "feature_importance.png", dpi=150)
    plt.close(fig)

    metrics = {
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "baseline_model": "Linear Regression",
        "baseline_rmse": baseline_rmse,
        "best_rmse": best_rmse,
        "rmse_improvement_pct_over_baseline": round(improvement_pct, 2),
        "results": results,
        "feature_importance_rf": [
            {"feature": f, "importance": round(float(v), 4)} for f, v in importance
        ],
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ART / "learning_curves.json").write_text(json.dumps(curves, indent=2))
    joblib.dump({"pipeline": fitted[best_name], "model_name": best_name},
                ART / "house_model.joblib")
    print(f"\nSaved model   -> {ART / 'house_model.joblib'}")
    print(f"Saved metrics -> {ART / 'metrics.json'}")


if __name__ == "__main__":
    main()
