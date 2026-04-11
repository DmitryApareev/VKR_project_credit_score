"""
Графики для отчёта и ноутбуков: распределения, корреляции, ROC, важность признаков.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# настройка «аккуратного» вида графиков
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 10
sns.set_theme(style="whitegrid", context="notebook")


def plot_target_distribution(y: np.ndarray | pd.Series, title: str, save_path: Path) -> None:
    """Столбчатая диаграмма долей классов."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    s = pd.Series(y).value_counts(normalize=True).sort_index()
    fig, ax = plt.subplots()
    s.plot(kind="bar", color=["#4c72b0", "#dd8452"], ax=ax)
    ax.set_xticklabels(["Нет дефолта (0)", "Дефолт (1)"], rotation=0)
    ax.set_ylabel("Доля")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, title: str, save_path: Path) -> None:
    """Корреляции только числовых признаков (после простого отбора)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Факт")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(fpr, tpr, title: str, save_path: Path, auc_value: float | None = None) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="случайный уровень")
    if auc_value is not None:
        ax.set_title(f"{title} (AUC = {auc_value:.3f})")
    else:
        ax.set_title(title)
    ax.set_xlabel("Доля ложноположительных (FPR)")
    ax.set_ylabel("Доля истинноположительных (TPR)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray,
    names: list[str],
    title: str,
    save_path: Path,
    top_n: int = 15,
) -> None:
    """Горизонтальный barplot для важности признаков (деревья / лес)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(importances)[::-1][:top_n]
    vals = importances[order]
    labels = [names[i] if i < len(names) else f"f{i}" for i in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels[::-1], vals[::-1], color="#55a868")
    ax.set_title(title)
    ax.set_xlabel("Важность")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_multiple_roc(
    roc_data: dict[str, tuple],
    title: str,
    save_path: Path,
) -> None:
    """
    roc_data: имя модели -> (fpr, tpr, auc)
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    for name, (fpr, tpr, auc_v) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_v:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
