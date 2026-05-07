"""
Я привожу свои две выгрузки (РФ и КЗ) к одной таблице признаков для сравнения моделей
и увеличиваю каждую выборку до нужного N через bootstrap.

Итоговые колонки: age, gender, region, monthly_income, employment_status,
work_experience, loan_amount, loan_term, interest_rate, current_debt, debt_to_income,
number_of_loans, overdue_count, max_days_overdue, loan_purpose, target.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent

# такие имена колонок дальше ждут preprocessing и main
UNIFIED_COLUMNS = [
    "age",
    "gender",
    "region",
    "monthly_income",
    "employment_status",
    "work_experience",
    "loan_amount",
    "loan_term",
    "interest_rate",
    "current_debt",
    "debt_to_income",
    "number_of_loans",
    "overdue_count",
    "max_days_overdue",
    "loan_purpose",
    "target",
]


def _resolve_source_path(preferred_name: str, downloads_name: str) -> Path | None:
    """Сначала смотрю data/raw/incoming/, потом env, потом ~/Downloads/."""
    root = _project_root()
    local = root / "data" / "raw" / "incoming" / preferred_name
    if local.exists():
        return local

    env_key = "KZ_REGISTER_CSV" if "kazakhstan" in preferred_name.lower() else "RU_CLIENT_CSV"
    env_path = os.environ.get(env_key)
    if env_path and Path(env_path).exists():
        return Path(env_path)

    downloads = Path.home() / "Downloads" / downloads_name
    if downloads.exists():
        return downloads

    return None


def kazakhstan_register_to_unified(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Мой реестр по Казахстану: одна строка — кредит, привожу к общей схеме."""
    n = len(df)
    age = df["age"].astype(int).clip(18, 80)
    gender = df["gender"].map({"Female": "F", "Male": "M"}).fillna("F")
    region = df["region"].astype(str)

    interest_rate = df["interest_rate_pct"].astype(float).clip(5, 60).round(2)
    loan_amount = df["credit_amount_kzt"].astype(float).clip(10_000, 50_000_000).round(2)

    loan_term = rng.integers(12, 73, size=n)
    r = interest_rate / 100.0 / 12.0
    r = np.where(r < 1e-6, 1e-6, r)
    pow_term = np.power(1 + r, loan_term)
    monthly_pay = loan_amount * (r * pow_term) / (pow_term - 1)
    burden = rng.uniform(2.0, 4.8, size=n)
    monthly_income = (monthly_pay * burden).clip(80_000, 4_000_000).round(2)

    work_experience = (age - 22 - rng.integers(0, 6, size=n)).clip(0, 45).astype(int)

    emp_choices = np.array(
        ["employed", "self_employed", "unemployed", "student", "retired"]
    )
    emp_p = np.array([0.64, 0.12, 0.07, 0.05, 0.12])
    employment_status = rng.choice(emp_choices, size=n, p=emp_p)

    share_out = rng.uniform(0.1, 0.92, size=n)
    current_debt = (loan_amount * share_out).round(2)
    debt_to_income = (current_debt / (monthly_income + 1.0)).round(4)

    # Я заметил, что в моём реестре days_past_due фактически «подсказывает» npl_90plus:
    # у бездефолтных строк DPD всегда 0, у дефолтных — большой. Если подставить DPD в ту же
    # строку, что и target, модель будет жульничать. Поэтому я беру все значения DPD по выборке
    # и случайно перемешиваю их между заявками: частоты значений сохраняются, а связь с дефолтом
    # исчезает — так я убираю утечку без выдуманных «зашумлённых» прокси.
    dpd_pool = df["days_past_due"].fillna(0).astype(int).clip(0, 365).to_numpy(copy=True)
    rng.shuffle(dpd_pool)
    max_days_overdue = dpd_pool
    overdue_count = np.where(
        max_days_overdue <= 0,
        0,
        np.minimum(24, 1 + max_days_overdue // 30),
    ).astype(int)

    number_of_loans = (1 + rng.poisson(0.35, size=n)).clip(1, 15).astype(int)

    is_auto = df["loan_reason"].astype(str).eq("Auto loan")
    loan_purpose = np.where(is_auto, "auto", "consumer")

    target = df["npl_90plus"].fillna(0).astype(int).clip(0, 1)

    out = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "region": region,
            "monthly_income": monthly_income,
            "employment_status": employment_status,
            "work_experience": work_experience,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "interest_rate": interest_rate,
            "current_debt": current_debt,
            "debt_to_income": debt_to_income,
            "number_of_loans": number_of_loans,
            "overdue_count": overdue_count,
            "max_days_overdue": max_days_overdue,
            "loan_purpose": loan_purpose,
            "target": target,
        }
    )
    return out


# моя российская выгрузка кодирует тип кредита числом — маппинг в loan_purpose
_RU_TYPE_TO_PURPOSE = {
    0: "consumer",
    1: "mortgage",
    2: "auto",
    3: "consumer",
    4: "consumer",
    5: "business",
    6: "auto",
    7: "refinance",
}


def russia_bureau_to_unified(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Российский client_data: бакеты бюро. Возраста в файле нет — я оцениваю его грубо из pre_since_
    """
    n = len(df)
    ids = df["id"].astype(int).values

    base_age = 24 + df["pre_since_opened"].astype(float) + df["pre_since_confirmed"].astype(float) / 3.0
    age = np.round(base_age + rng.normal(0, 3.0, n)).astype(int).clip(22, 72)

    gender = np.where(ids % 2 == 0, "M", "F")

    regions = np.array(
        [
            "Москва",
            "Санкт-Петербург",
            "Центральный ФО",
            "Поволжье",
            "Урал",
            "Сибирь",
            "Юг России",
            "Дальний Восток",
        ]
    )
    region = regions[ids % len(regions)]

    monthly_income = (
        (df["pre_loans_next_pay_summ"].astype(float) + 0.5) * 48_000
        + df["pre_loans_credit_limit"].astype(float) * 9_500
        + rng.normal(0, 15_000, n)
    )
    monthly_income = np.clip(monthly_income, 18_000, 1_200_000).round(2)

    holder = df["enc_loans_account_holder_type"].astype(int)
    employment_status = np.full(n, "employed", dtype=object)
    employment_status[holder == 6] = "self_employed"
    employment_status[holder == 4] = "student"
    # чуть разнообразю занятость, иначе все «employed»
    flip = rng.random(n) < 0.06
    employment_status[flip] = rng.choice(
        ["unemployed", "retired", "self_employed"], size=flip.sum()
    )

    work_experience = df["pre_since_confirmed"].clip(0, 40).astype(int)

    loan_amount = (
        (df["pre_loans_credit_limit"].astype(float) + 1.0) * 105_000
        + df["pre_loans_outstanding"].astype(float) * 72_000
    )
    loan_amount = np.clip(loan_amount, 50_000, 6_000_000).round(2)

    loan_term = (df["pre_pterm"].astype(float) * 3 + df["pre_fterm"].astype(float) * 2 + 8).round().astype(
        int
    )
    loan_term = np.clip(loan_term, 6, 84)

    interest_rate = (df["pre_loans_credit_cost_rate"].astype(float) * 1.35 + 10.0).clip(9, 32).round(2)

    current_debt = (
        df["pre_loans_outstanding"].astype(float) * 68_000
        + df["pre_loans_max_overdue_sum"].astype(float) * 42_000
    )
    current_debt = np.clip(current_debt, 0, 4_000_000).round(2)

    iz530 = df["is_zero_loans530"].astype(int)
    iz3060 = df["is_zero_loans3060"].astype(int)
    iz6090 = df["is_zero_loans6090"].astype(int)
    iz90 = df["is_zero_loans90"].astype(int)
    dpd_proxy = (
        (1 - iz90) * 110
        + (1 - iz6090) * 70
        + (1 - iz3060) * 40
        + (1 - iz530) * 18
    ).astype(float)
    max_days_overdue = (dpd_proxy + df["pre_util"].astype(float) * 2.5 + rng.integers(0, 30, n)).clip(0, 365)
    max_days_overdue = max_days_overdue.astype(int)
    overdue_count = np.where(
        max_days_overdue <= 0,
        0,
        np.minimum(24, 1 + max_days_overdue // 35),
    ).astype(int)

    debt_to_income = (current_debt / (monthly_income + 1.0)).round(4)

    number_of_loans = df["rn"].clip(1, 25).astype(int)

    ctype = df["enc_loans_credit_type"].astype(int)
    loan_purpose = ctype.map(_RU_TYPE_TO_PURPOSE).fillna("consumer")

    # target: я считаю дефолтом проблемное закрытие по pclose / fclose
    target = (
        (df["pclose_flag"].astype(int) == 1) | (df["fclose_flag"].astype(int) == 1)
    ).astype(int)

    out = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "region": region,
            "monthly_income": monthly_income,
            "employment_status": employment_status,
            "work_experience": work_experience,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "interest_rate": interest_rate,
            "current_debt": current_debt,
            "debt_to_income": debt_to_income,
            "number_of_loans": number_of_loans,
            "overdue_count": overdue_count,
            "max_days_overdue": max_days_overdue,
            "loan_purpose": loan_purpose,
            "target": target,
        }
    )
    return out


def bootstrap_expand(
    df: pd.DataFrame,
    n_out: int,
    rng: np.random.Generator,
    jitter: bool = True,
    add_source_id: bool = True,
) -> pd.DataFrame:
    """Раздуваю выборку с возвращением и слегка дёргаю числа, чтобы строки не были клонами."""
    if len(df) == 0:
        raise ValueError("Пустой датафрейм — нечего расширять.")

    idx = rng.integers(0, len(df), size=n_out)
    out = df.iloc[idx].reset_index(drop=True)

    if add_source_id:
        out["source_row_id"] = idx

    if not jitter:
        return out

    num_cols = [
        "age",
        "monthly_income",
        "work_experience",
        "loan_amount",
        "loan_term",
        "interest_rate",
        "current_debt",
        "debt_to_income",
        "number_of_loans",
        "overdue_count",
        "max_days_overdue",
    ]
    for col in num_cols:
        if col not in out.columns:
            continue
        if col in ("age", "work_experience", "loan_term", "number_of_loans", "overdue_count", "max_days_overdue"):
            noise = rng.integers(-2, 3, size=len(out))
            out[col] = (out[col] + noise).clip(lower=0)
            if col == "loan_term":
                out[col] = out[col].clip(6, 84)
            if col == "age":
                out[col] = out[col].clip(18, 80)
        else:
            factor = 1.0 + rng.normal(0, 0.035, size=len(out))
            out[col] = (out[col].astype(float) * factor).round(2)

    out["debt_to_income"] = (
        out["current_debt"].astype(float) / (out["monthly_income"].astype(float) + 1.0)
    ).round(4)

    out["target"] = out["target"].astype(int).clip(0, 1)
    return out


def build_unified_datasets(
    target_n: int = 30_000,
    seed: int = 42,
    save_non_bootstrap: bool = True,
) -> tuple[Path, Path]:
    """Собираю unified и перезаписываю data/raw/kazakhstan_credit.csv и russia_credit.csv."""
    root = _project_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "incoming").mkdir(parents=True, exist_ok=True)

    kz_src = _resolve_source_path(
        "kazakhstan_credit_register_synthetic_3000.csv",
        "kazakhstan_credit_register_synthetic_3000.csv",
    )
    ru_src = _resolve_source_path("client_data.csv", "client_data.csv")

    if kz_src is None or ru_src is None:
        raise FileNotFoundError(
            "Не нашёл входные CSV: нужны kazakhstan_credit_register_synthetic_3000.csv и "
            "client_data.csv в data/raw/incoming/ или в ~/Downloads/, либо пути в "
            "KZ_REGISTER_CSV и RU_CLIENT_CSV."
        )

    rng_kz = np.random.default_rng(seed)
    rng_ru = np.random.default_rng(seed + 1)

    df_kz_raw = pd.read_csv(kz_src, encoding="utf-8")
    df_ru_raw = pd.read_csv(ru_src, encoding="utf-8")

    unified_kz = kazakhstan_register_to_unified(df_kz_raw, rng_kz)
    unified_ru = russia_bureau_to_unified(df_ru_raw, rng_ru)

    expanded_kz = bootstrap_expand(unified_kz, target_n, rng_kz, jitter=True, add_source_id=True)
    expanded_ru = bootstrap_expand(unified_ru, target_n, rng_ru, jitter=True, add_source_id=True)


    if save_non_bootstrap:
        out_kz_non_bootstrap = raw_dir / "kazakhstan_credit_non_bootstrap.csv"
        out_ru_non_bootstrap = raw_dir / "russia_credit_non_bootstrap.csv"
        unified_kz[UNIFIED_COLUMNS].to_csv(out_kz_non_bootstrap, index=False, encoding="utf-8")
        unified_ru[UNIFIED_COLUMNS].to_csv(out_ru_non_bootstrap, index=False, encoding="utf-8")

    for df in (expanded_kz, expanded_ru):
        missing = [c for c in UNIFIED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Не хватает колонок: {missing}")

    out_kz = raw_dir / "kazakhstan_credit.csv"
    out_ru = raw_dir / "russia_credit.csv"

    expanded_kz[UNIFIED_COLUMNS].to_csv(out_kz, index=False, encoding="utf-8")
    expanded_ru[UNIFIED_COLUMNS].to_csv(out_ru, index=False, encoding="utf-8")

    return out_kz, out_ru
