from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainArtifacts:
    model_name: str
    pipeline: Pipeline
    feature_names: List[str]
    feature_importances: Dict[str, float] = field(default_factory=dict)


def _build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def _get_model(name: str, random_state: int):
    if name == "logreg":
        return LogisticRegression(max_iter=500, class_weight="balanced", random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "hist_gbm":
        return HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=random_state,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                is_unbalance=True,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            )
        except ImportError:
            return None
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
            )
        except ImportError:
            return None
    return None


def _extract_feature_importance(pipe: Pipeline, feature_cols: List[str], numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, float]:
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]

    try:
        cat_features = []
        if categorical_cols:
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            cat_features = list(ohe.get_feature_names_out(categorical_cols))
        all_names = numeric_cols + cat_features
    except Exception:
        all_names = feature_cols

    importances = {}
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        for name, val in zip(all_names[:len(imp)], imp):
            importances[name] = float(val)
    elif hasattr(model, "coef_"):
        coef = model.coef_.flatten()
        for name, val in zip(all_names[:len(coef)], coef):
            importances[name] = float(abs(val))

    return importances


def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    random_state: int,
) -> Tuple[TrainArtifacts | None, np.ndarray | None, np.ndarray | None, str]:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    model = _get_model(model_name, random_state)
    if model is None:
        return None, None, None, f"{model_name} unavailable"

    pre = _build_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, y_train = train_df[feature_cols], train_df[target_col].astype(int)
    X_val, y_val = val_df[feature_cols], val_df[target_col].astype(int)

    pipe.fit(X_train, y_train)
    val_score = pipe.predict_proba(X_val)[:, 1]

    importances = _extract_feature_importance(pipe, feature_cols, numeric_cols, categorical_cols)

    return (
        TrainArtifacts(
            model_name=model_name,
            pipeline=pipe,
            feature_names=feature_cols,
            feature_importances=importances,
        ),
        y_val.to_numpy(),
        val_score,
        "ok",
    )


def predict_scores(artifacts: TrainArtifacts, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return artifacts.pipeline.predict_proba(df[feature_cols])[:, 1]
