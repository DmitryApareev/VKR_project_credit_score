"""
Предобработка: пропуски, кодирование категорий, масштабирование, train/test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import add_engineered_features, drop_leakage_columns


def _make_one_hot_encoder() -> OneHotEncoder:
    # в разных версиях sklearn параметр назывался по-разному
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


@dataclass
class PreprocessResult:
    """Всё, что нужно для обучения и оценки."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    preprocessor: ColumnTransformer
    raw_train: pd.DataFrame
    raw_test: pd.DataFrame


def _numeric_categorical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    target = "target"
    cols = [c for c in df.columns if c != target]
    numeric = []
    categorical = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Числовые: медиана на пропуски + стандартизация (для логистической регрессии удобно).
    Категории: отдельный импьютер + One-Hot.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )
    return pre


def preprocess_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> PreprocessResult:
    """
    Полный цикл для одной страны: фичи -> сплит -> fit на train -> transform train/test.
    """
    df = drop_leakage_columns(df)
    df = add_engineered_features(df)

    if "target" not in df.columns:
        raise ValueError("В данных должна быть колонка target (0/1).")

    X = df.drop(columns=["target"])
    y = df["target"].astype(int).values

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    numeric_features, categorical_features = _numeric_categorical_columns(
        pd.concat([X_train_df, X_test_df], axis=0)
    )

    pre = build_preprocessor(numeric_features, categorical_features)
    pre.fit(X_train_df)

    X_train = pre.transform(X_train_df)
    X_test = pre.transform(X_test_df)

    # имена признаков после OHE (для отчёта / важности — приблизительно)
    feature_names = list(
        pre.get_feature_names_out()
    )

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=pre,
        raw_train=X_train_df.assign(target=y_train),
        raw_test=X_test_df.assign(target=y_test),
    )


def save_processed(df: pd.DataFrame, path) -> None:
    """Сохранение очищенной таблицы (после FE, до сплита — как в задании processed/)."""
    df.to_csv(path, index=False, encoding="utf-8")
