"""
Утилиты для диплома по кредитному скорингу: пути, папки, загрузка CSV.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Корень проекта (папка credit_scoring_diploma)."""
    return Path(__file__).resolve().parent.parent


def ensure_directories() -> None:
    """Создаю нужные подпапки, если их ещё нет."""
    root = project_root()
    for sub in [
        "data/raw",
        "data/raw/incoming",
        "data/processed",
        "models",
        "results/tables",
        "results/figures",
        "results/metrics",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def ensure_raw_datasets_exist() -> tuple[Path, Path]:
    """
    Пересобираю kazakhstan_credit.csv и russia_credit.csv из входных файлов
    """
    ensure_directories()
    from unify_external_sources import build_unified_datasets

    return build_unified_datasets(target_n=30_000, seed=42)


def load_raw_csv(path: Path):
    """Читаю CSV в pandas."""
    import pandas as pd

    return pd.read_csv(path, encoding="utf-8")
