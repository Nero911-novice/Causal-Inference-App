from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModelDiagnostics:
    beta: Optional[float] = None
    r2: Optional[float] = None
    leverage_l: Optional[float] = None
    yoy_support_count: Optional[int] = None
    yoy_fallback_months: Optional[List[str]] = None


def train_model_1_beta(hist_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Модель 1: регрессия через начало координат:
    Target_MoM = beta * Control_MoM
    """
    train = hist_df.dropna(subset=["Target_MoM", "Control_MoM"]).copy()
    if len(train) < 2:
        raise ValueError("Недостаточно данных для Модели 1. Нужны минимум 2 наблюдения с рассчитанным MoM.")

    x = train["Control_MoM"].values.astype(float)
    y = train["Target_MoM"].values.astype(float)

    denom = np.sum(x ** 2)
    if np.isclose(denom, 0):
        raise ValueError("Невозможно оценить beta: динамика Control_Group в Historical Period имеет нулевую дисперсию.")

    beta = np.sum(x * y) / denom
    y_hat = beta * x

    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if np.isclose(sst, 0) else 1 - sse / sst

    return float(beta), float(r2) if not pd.isna(r2) else np.nan


def train_model_2_leverage(hist_df: pd.DataFrame) -> float:
    """
    Модель 2:
    L = CV_control / CV_target
    """
    if len(hist_df) < 2:
        raise ValueError("Недостаточно данных для Модели 2. Нужны минимум 2 наблюдения.")

    mean_c = hist_df["Control_Group"].mean()
    mean_t = hist_df["Target_Group"].mean()

    std_c = hist_df["Control_Group"].std(ddof=1)
    std_t = hist_df["Target_Group"].std(ddof=1)

    if np.isclose(mean_c, 0) or np.isclose(mean_t, 0):
        raise ValueError("Средние значения групп близки к нулю. Невозможно вычислить коэффициент вариации.")
    if np.isclose(std_t, 0):
        raise ValueError("Дисперсия Target_Group в Historical Period равна нулю. Невозможно вычислить рычаг стабильности.")

    cv_c = std_c / mean_c
    cv_t = std_t / mean_t

    if np.isclose(cv_t, 0):
        raise ValueError("CV целевой группы близок к нулю. Невозможно вычислить рычаг стабильности.")

    leverage_l = cv_c / cv_t
    if np.isclose(leverage_l, 0):
        raise ValueError("Рычаг стабильности L оказался близок к нулю.")

    return float(leverage_l)


def forecast_model_3_yoy(hist_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.Series, int, List[str]]:
    """
    Модель 3:
    для каждого тестового месяца берем среднее значение Target_MoM
    для такого же календарного месяца в Historical Period.
    Если наблюдений по конкретному месяцу нет, используем общий средний MoM Historical Period.
    """
    hist_mom = hist_df.dropna(subset=["Target_MoM"]).copy()
    if hist_mom.empty:
        raise ValueError("Недостаточно данных для Модели 3. В Historical Period отсутствуют рассчитанные значения Target_MoM.")

    month_means = hist_mom.groupby("Month_Num")["Target_MoM"].mean()
    overall_mean = hist_mom["Target_MoM"].mean()

    forecasts = []
    support_count = len(hist_mom)
    fallback_months = []

    for _, row in test_df.iterrows():
        m = row["Month_Num"]
        if m in month_means.index and not pd.isna(month_means.loc[m]):
            forecasts.append(month_means.loc[m])
        else:
            forecasts.append(overall_mean)
            fallback_months.append(row["Date"].strftime("%Y-%m"))

    return pd.Series(forecasts, index=test_df.index, name="Forecast_M3"), support_count, fallback_months


def build_baseline_path(
    previous_actual_value: float,
    forecast_mom: pd.Series
) -> pd.Series:
    """
    Восстанавливает контрфактуальный абсолютный ряд на тестовом периоде
    рекурсивно от последнего фактического значения до Test Period.
    """
    values = []
    prev = previous_actual_value
    for growth in forecast_mom:
        current = prev * (1 + growth)
        values.append(current)
        prev = current
    return pd.Series(values, index=forecast_mom.index)


def cumulative_growth_from_series(series: pd.Series) -> float:
    """
    Композиционная суммарная динамика за период:
    Π(1+r_t) - 1
    """
    s = series.dropna()
    if s.empty:
        return np.nan
    return float(np.prod(1 + s.values) - 1)


def run_models(df: pd.DataFrame, hist_mask: pd.Series, test_mask: pd.Series):
    hist_df = df.loc[hist_mask].copy()
    test_df = df.loc[test_mask].copy()

    # Для обучения нужны MoM, поэтому первая точка всего ряда или границы
    # могут содержать NaN — это допустимо, но модель будет работать на доступных наблюдениях.
    beta, r2 = train_model_1_beta(hist_df)
    leverage_l = train_model_2_leverage(hist_df)
    forecast_m3, yoy_support_count, fallback_months = forecast_model_3_yoy(hist_df, test_df)

    test_df["Forecast_M1_MoM"] = test_df["Control_MoM"] * beta
    test_df["Forecast_M2_MoM"] = test_df["Control_MoM"] / leverage_l
    test_df["Forecast_M3_MoM"] = forecast_m3
    test_df["Fact_MoM"] = test_df["Target_MoM"]

    # Базовый объем — последнее фактическое значение перед началом Test Period
    test_start = test_df["Date"].min()
    prev_row = df.loc[df["Date"] < test_start].sort_values("Date").iloc[-1]
    baseline_anchor_value = float(prev_row["Target_Group"])

    # Абсолютные контрфактуальные траектории
    test_df["Baseline_M1"] = build_baseline_path(baseline_anchor_value, test_df["Forecast_M1_MoM"])
    test_df["Baseline_M2"] = build_baseline_path(baseline_anchor_value, test_df["Forecast_M2_MoM"])
    test_df["Baseline_M3"] = build_baseline_path(baseline_anchor_value, test_df["Forecast_M3_MoM"])

    # Чистый эффект в п.п. и абсолютных величинах
    for model in ["M1", "M2", "M3"]:
        test_df[f"Impact_{model}_pp"] = test_df["Fact_MoM"] - test_df[f"Forecast_{model}_MoM"]
        test_df[f"Impact_{model}_abs"] = test_df["Target_Group"] - test_df[f"Baseline_{model}"]

    diagnostics = ModelDiagnostics(
        beta=beta,
        r2=r2,
        leverage_l=leverage_l,
        yoy_support_count=yoy_support_count,
        yoy_fallback_months=fallback_months,
    )

    return hist_df, test_df, diagnostics, baseline_anchor_value
