from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "salary"
CATEGORICAL_FEATURES = [
    "job_title",
    "education_level",
    "company_size",
    "location",
    "remote_work",
]
NUMERIC_FEATURES = [
    "experience_years",
    "skills_count",
    "certifications",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES
DEFAULT_MODEL_PARAMS = {
    "depth": 6,
    "learning_rate": 0.05,
    "iterations": 500,
    "l2_leaf_reg": 3,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_state": 42,
    "verbose": False,
}


@dataclass
class TrainingMetrics:
    mae: float
    rmse: float
    r2: float


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [col for col in df.columns if col != TARGET_COLUMN]

    df_model = df.groupby(feature_cols, as_index=False)[TARGET_COLUMN].mean()
    df_model["location"] = df_model["location"].replace("Remote", "Unknown")

    if "industry" in df_model.columns:
        df_model = df_model.drop(columns=["industry"])

    for col in CATEGORICAL_FEATURES:
        df_model[col] = df_model[col].astype(str)

    return df_model


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()


def train_model(
    df: pd.DataFrame,
    model_params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[CatBoostRegressor, TrainingMetrics]:
    clean_df = prepare_training_frame(df)
    X, y = split_features_and_target(clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = CatBoostRegressor(**(model_params or DEFAULT_MODEL_PARAMS))
    model.fit(X_train, y_train, cat_features=CATEGORICAL_FEATURES)

    preds = model.predict(X_test)
    metrics = TrainingMetrics(
        mae=float(mean_absolute_error(y_test, preds)),
        rmse=float(mean_squared_error(y_test, preds) ** 0.5),
        r2=float(r2_score(y_test, preds)),
    )
    return model, metrics


def train_final_model(df: pd.DataFrame, model_params: dict | None = None) -> CatBoostRegressor:
    clean_df = prepare_training_frame(df)
    X, y = split_features_and_target(clean_df)

    model = CatBoostRegressor(**(model_params or DEFAULT_MODEL_PARAMS))
    model.fit(X, y, cat_features=CATEGORICAL_FEATURES)
    return model


def save_artifacts(
    model: CatBoostRegressor,
    output_dir: str | Path,
    metrics: TrainingMetrics | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "salary_model.cbm"
    metadata_path = output_path / "metadata.json"

    model.save_model(model_path)
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target_column": TARGET_COLUMN,
        "model_params": model.get_all_params(),
    }

    if metrics is not None:
        metadata["holdout_metrics"] = asdict(metrics)

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model(model_dir: str | Path) -> tuple[CatBoostRegressor, dict]:
    model_path = Path(model_dir) / "salary_model.cbm"
    metadata_path = Path(model_dir) / "metadata.json"

    model = CatBoostRegressor()
    model.load_model(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def build_inference_frame(payload: dict, metadata: dict) -> pd.DataFrame:
    df = pd.DataFrame([payload], columns=metadata["feature_columns"])
    for col in metadata["categorical_features"]:
        df[col] = df[col].astype(str)
    return df
