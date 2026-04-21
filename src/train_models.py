"""
Обучение нескольких классических моделей и сохранение в models/ через joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Словарь моделей
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=10,
            max_samples=0.75,  # я так снижаю переобучение, когда 30k строк собраны бутстрепом из малой базы
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
        ),
    }
    return models


def train_all(
    X_train,
    y_train,
    models_dir: Path,
    random_state: int = 42,
) -> Dict[str, Any]:
    fitted: Dict[str, Any] = {}
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in get_models(random_state=random_state).items():
        model.fit(X_train, y_train)
        fitted[name] = model
        joblib.dump(model, models_dir / f"{name}.joblib")
    return fitted


def load_model(path: Path):
    return joblib.load(path)
