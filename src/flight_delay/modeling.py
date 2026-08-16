"""Reproducible training and evaluation for flight-delay classification."""

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
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "DEP_DEL15"

CATEGORICAL_FEATURES = [
    "DEP_BLOCK",
    "CARRIER_NAME",
    "DEPARTING_AIRPORT",
    "PREVIOUS_AIRPORT",
]

NUMERIC_FEATURES = [
    "DAY_OF_WEEK",
    "DISTANCE_GROUP",
    "SEGMENT_NUMBER",
    "CONCURRENT_FLIGHTS",
    "NUMBER_OF_SEATS",
    "AVG_MONTHLY_PASS_AIRPORT",
    "AVG_MONTHLY_PASS_AIRLINE",
    "FLT_ATTENDANTS_PER_PASS",
    "GROUND_SERV_PER_PASS",
    "PLANE_AGE",
    "LATITUDE",
    "LONGITUDE",
]

EXCLUDED_FEATURES = {
    "AIRLINE_FLIGHTS_MONTH": "current-month aggregate unavailable at prediction time",
    "AIRLINE_AIRPORT_FLIGHTS_MONTH": "current-month aggregate unavailable at prediction time",
    "PRCP": "realized daily weather summary",
    "SNOW": "realized daily weather summary",
    "SNWD": "realized daily weather summary",
    "TMAX": "realized daily maximum unavailable before departure",
    "AWND": "realized daily weather summary",
}

MODEL_ORDER = [
    "Baseline",
    "Logistic Regression",
    "k-Nearest Neighbours",
    "Random Forest",
]


def load_data(path: Path) -> pd.DataFrame:
    """Load the raw data and validate the columns and binary target."""
    data = pd.read_csv(path)
    required = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET])
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[TARGET].isna().any() or set(data[TARGET].unique()) != {0, 1}:
        raise ValueError(f"{TARGET} must be a complete binary 0/1 target")
    return data


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fitted within each training fold."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def build_models() -> dict[str, BaseEstimator]:
    """Return the baseline and intentionally straightforward classifiers."""
    return {
        "Baseline": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        "k-Nearest Neighbours": KNeighborsClassifier(
            n_neighbors=21,
            n_jobs=-1,
            weights="distance",
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=5,
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def build_pipeline(model: BaseEstimator) -> Pipeline:
    """Bundle all learned preprocessing with a classifier."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Evaluate models with stratified CV on training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    rows = []
    for name in MODEL_ORDER:
        scores = cross_validate(
            build_pipeline(build_models()[name]),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        row: dict[str, str | float] = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_test_set(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | int]:
    """Calculate final holdout metrics once for the selected pipeline."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    return {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def plot_target_distribution(data: pd.DataFrame, output_path: Path) -> None:
    """Plot counts and percentages for the binary target."""
    counts = data[TARGET].value_counts().sort_index()
    labels = ["On time (<15 min)", "Delayed (>=15 min)"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#E45756"])
    ax.bar_label(
        bars,
        labels=[f"{count:,}\n({count / len(data):.1%})" for count in counts],
        padding=4,
    )
    ax.set(title="Departure-delay target distribution", ylabel="Flights")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_delay_patterns(data: pd.DataFrame, output_path: Path) -> None:
    """Plot two descriptive, non-causal delay patterns."""
    block_order = ["EARLY_MORNING", "MORNING", "MIDDAY", "AFTERNOON", "EVENING", "LATE_NIGHT"]
    by_block = data.groupby("DEP_BLOCK", observed=True)[TARGET].agg(["mean", "size"]).reindex(block_order)
    segment = data.assign(
        segment_group=np.where(data["SEGMENT_NUMBER"] >= 8, "8+", data["SEGMENT_NUMBER"].astype(str))
    )
    segment_order = [str(value) for value in range(1, 8)] + ["8+"]
    by_segment = segment.groupby("segment_group", observed=True)[TARGET].agg(["mean", "size"]).reindex(segment_order)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(by_block.index.str.replace("_", " ").str.title(), by_block["mean"], color="#4C78A8")
    axes[0].set(title="Delay rate varies by departure period", xlabel="Departure block", ylabel="Delayed flights")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(by_segment.index, by_segment["mean"], color="#F58518")
    axes[1].set(title="Later aircraft segments have higher delay rates", xlabel="Aircraft segment number", ylabel="Delayed flights")
    for ax in axes:
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        ax.set_ylim(0, max(by_block["mean"].max(), by_segment["mean"].max()) * 1.2)
    fig.suptitle("Descriptive delay patterns (associations, not causal effects)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, output_path: Path) -> None:
    """Plot cross-validated discrimination and delayed-class F1."""
    ordered = results.set_index("model").loc[MODEL_ORDER].reset_index()
    positions = np.arange(len(ordered))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, ordered["roc_auc_mean"], width, label="ROC-AUC", color="#4C78A8")
    ax.bar(positions + width / 2, ordered["f1_mean"], width, label="Delayed-class F1", color="#F58518")
    ax.set(
        title="Five-fold training-set model comparison",
        ylabel="Mean cross-validation score",
        xticks=positions,
        xticklabels=ordered["model"],
        ylim=(0, 0.8),
    )
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_holdout_evaluation(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    """Plot the final test confusion matrix and ROC curve."""
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["On time", "Delayed"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Final holdout confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, name="Selected model", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="No-skill reference")
    ax.set_title("Final holdout ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)


def calculate_feature_importance(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Measure original-column importance by holdout ROC-AUC permutation loss."""
    sample_size = min(5_000, len(X_test))
    sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
    y_sample = y_test.loc[sample.index]
    result = permutation_importance(
        pipeline,
        sample,
        y_sample,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    """Plot the ten strongest permutation importance estimates."""
    top = importance.head(10).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#54A24B")
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(
        title="Selected-model feature importance on the holdout set",
        xlabel="Mean decrease in ROC-AUC after permutation",
        ylabel="Feature",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_analysis(data_path: Path, output_dir: Path) -> dict[str, object]:
    """Run model selection, one final test evaluation, and artifact generation."""
    sns.set_theme(style="whitegrid", context="notebook")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    data = load_data(data_path)
    features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    X = data[features]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv_results = compare_models(X_train, y_train)
    selected_name = (
        cv_results.query("model != 'Baseline'").sort_values("roc_auc_mean", ascending=False).iloc[0]["model"]
    )
    selected_pipeline = build_pipeline(build_models()[selected_name])
    selected_pipeline.fit(X_train, y_train)
    test_metrics = evaluate_test_set(selected_pipeline, X_test, y_test)
    importance = calculate_feature_importance(selected_pipeline, X_test, y_test)

    cv_results.to_csv(output_dir / "cross_validation_metrics.csv", index=False, float_format="%.6f")
    importance.to_csv(output_dir / "feature_importance.csv", index=False, float_format="%.6f")
    metadata = {
        "random_state": RANDOM_STATE,
        "row_count": len(data),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_rate": y.mean(),
        "selected_model": selected_name,
        "selection_metric": "mean five-fold training ROC-AUC",
        "test_metrics": test_metrics,
        "model_features": features,
        "excluded_features": EXCLUDED_FEATURES,
    }
    (output_dir / "test_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    joblib.dump(selected_pipeline, output_dir / "model_pipeline.joblib", compress=3)

    plot_target_distribution(data, figure_dir / "target_distribution.png")
    plot_delay_patterns(data, figure_dir / "delay_patterns.png")
    plot_model_comparison(cv_results, figure_dir / "model_comparison.png")
    plot_holdout_evaluation(selected_pipeline, X_test, y_test, figure_dir)
    plot_feature_importance(importance, figure_dir / "feature_importance.png")
    return metadata
