"""
Точка входа: полный сценарий от сырых данных до таблиц и графиков в results/.

Запуск из корня проекта:
    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# чтобы импорты из папки src работали при запуске main.py
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import data_preprocessing
import evaluate_models
import feature_engineering
import train_models
import utils
import visualization


def _prepare_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка + инженерия признаков для сохранения в data/processed/."""
    df = feature_engineering.drop_leakage_columns(df)
    return feature_engineering.add_engineered_features(df)


def run_pipeline_for_country(country_code: str, raw_path: Path) -> None:
    """Обучение и оценка для одной страны; результаты кладём в results/ с суффиксом."""
    print(f"\n=== Страна: {country_code} ===")
    df = utils.load_raw_csv(raw_path)
    clean = _prepare_clean_frame(df)

    processed_path = ROOT / "data" / "processed" / (
        "kazakhstan_clean.csv" if country_code == "KZ" else "russia_clean.csv"
    )
    data_preprocessing.save_processed(clean, processed_path)
    print(f"Сохранено: {processed_path}")

    prep = data_preprocessing.preprocess_dataframe(clean, test_size=0.25, random_state=42)
    models_dir = ROOT / "models" / country_code.lower()
    fitted = train_models.train_all(prep.X_train, prep.y_train, models_dir=models_dir)

    table, detailed = evaluate_models.evaluate_all_models(fitted, prep.X_test, prep.y_test)
    table.insert(0, "country", country_code)

    tables_dir = ROOT / "results" / "tables"
    metrics_dir = ROOT / "results" / "metrics"
    fig_dir = ROOT / "results" / "figures" / country_code.lower()

    suffix = "kz" if country_code == "KZ" else "ru"
    evaluate_models.save_metrics_table(table, tables_dir / f"metrics_{suffix}.csv")
    evaluate_models.save_metrics_table(table, metrics_dir / f"metrics_{suffix}.csv")

    # распределение целевой переменной (на всей выборке после FE)
    visualization.plot_target_distribution(
        clean["target"].values,
        title=f"Распределение target ({country_code})",
        save_path=fig_dir / f"target_distribution_{suffix}.png",
    )

    visualization.plot_correlation_heatmap(
        clean,
        title=f"Корреляции (числовые признаки), {country_code}",
        save_path=fig_dir / f"correlation_heatmap_{suffix}.png",
    )

    roc_bundle = {}
    for name, res in detailed.items():
        visualization.plot_confusion_matrix(
            res["confusion_matrix"],
            title=f"Матрица ошибок — {name} ({country_code})",
            save_path=fig_dir / f"confusion_{name}_{suffix}.png",
        )
        visualization.plot_roc_curve(
            res["fpr"],
            res["tpr"],
            title=f"ROC — {name}",
            save_path=fig_dir / f"roc_{name}_{suffix}.png",
            auc_value=res["roc_auc"],
        )
        roc_bundle[name] = (res["fpr"], res["tpr"], res["roc_auc"])

    visualization.plot_multiple_roc(
        roc_bundle,
        title=f"Сравнение ROC-кривых ({country_code})",
        save_path=fig_dir / f"roc_all_models_{suffix}.png",
    )

    # важность признаков - берём Random Forest 
    rf = fitted.get("random_forest")
    if rf is not None and hasattr(rf, "feature_importances_"):
        visualization.plot_feature_importance(
            rf.feature_importances_,
            prep.feature_names,
            title=f"Важность признаков (Random Forest), {country_code}",
            save_path=fig_dir / f"feature_importance_rf_{suffix}.png",
        )

    print("Метрики:")
    print(table.to_string(index=False))


def main() -> None:
    utils.ensure_directories()
    kz_path, ru_path = utils.ensure_raw_datasets_exist()

    run_pipeline_for_country("KZ", kz_path)
    run_pipeline_for_country("RU", ru_path)

    # сводное сравнение двух стран
    t_kz = pd.read_csv(ROOT / "results" / "tables" / "metrics_kz.csv", encoding="utf-8")
    t_ru = pd.read_csv(ROOT / "results" / "tables" / "metrics_ru.csv", encoding="utf-8")
    comparison = pd.concat([t_kz, t_ru], axis=0, ignore_index=True)
    comparison_path = ROOT / "results" / "tables" / "comparison_russia_kazakhstan.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    print(f"\nСводная таблица сохранена: {comparison_path}")


if __name__ == "__main__":
    main()
