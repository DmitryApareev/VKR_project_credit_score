"""
Инженерия признаков: новые показатели кредитной нагрузки и уборка лишнего.
"""

from __future__ import annotations

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем признаки, которые часто используют в скоринге:
    - monthly_payment_approx: грубая оценка аннуитетного платежа (упрощённо)
    - payment_to_income: доля «условного» платежа в доходе
    - credit_burden_index: суммарная нагрузка (долг + кредит) относительно дохода
    """
    out = df.copy()

    # если debt_to_income нет или испорчен — пересчитаем из долга и дохода
    if "current_debt" in out.columns and "monthly_income" in out.columns:
        dti_calc = out["current_debt"] / (out["monthly_income"] + 1.0)
        if "debt_to_income" not in out.columns:
            out["debt_to_income"] = dti_calc
        else:
            out["debt_to_income"] = out["debt_to_income"].fillna(dti_calc)

    monthly_rate = (out["interest_rate"] / 100.0) / 12.0
    n = out["loan_term"].clip(lower=1)
    # упрощённая формула аннуитета (избегаем деления на ноль)
    r = monthly_rate.replace(0, 0.0001)
    payment = out["loan_amount"] * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    out["monthly_payment_approx"] = payment.round(2)
    out["payment_to_income"] = (out["monthly_payment_approx"] / (out["monthly_income"] + 1.0)).round(
        4
    )

    out["credit_burden_index"] = (
        out["current_debt"].fillna(0) + 0.35 * out["loan_amount"]
    ) / (out["monthly_income"] + 1.0)
    out["credit_burden_index"] = out["credit_burden_index"].round(4)

    # флаги «тяжёлых» условий (иногда помогают деревьям)
    out["has_overdue_history"] = (out["overdue_count"] > 0).astype(int)
    out["is_high_interest"] = (out["interest_rate"] > 18.0).astype(int)

    return out


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляем колонки, которые не хотим тащить в обучение (если появятся служебные).
    Сейчас просто возвращаем копию — расширять можно при появлении id и т.п.
    """
    out = df.copy()
    for col in ["customer_id", "application_id", "source_row_id"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out
