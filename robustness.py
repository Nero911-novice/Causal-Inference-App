from typing import Any

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "M1": "Модель 1: Регрессия чувствительности",
    "M2": "Модель 2: CV Leverage",
    "M3": "Модель 3: Сезонная эффективность YoY",
}


def build_scenario_result_table(
    hist_df: pd.DataFrame,
    test_df: pd.DataFrame,
    diagnostics: Any,
    scenario_name: str,
    cumulative_growth_from_series_fn,
) -> pd.DataFrame:
    """
    Возвращает единый числовой результат по сценарию:
    одна строка на каждую модель.
    """
    fact_cum = cumulative_growth_from_series_fn(test_df["Fact_MoM"])
    market_cum = cumulative_growth_from_series_fn(test_df["Control_MoM"])

    hist_start = hist_df["Date"].min()
    hist_end = hist_df["Date"].max()
    test_start = test_df["Date"].min()
    test_end = test_df["Date"].max()

    rows = []
    for model_code, model_label in MODEL_LABELS.items():
        forecast_cum = cumulative_growth_from_series_fn(test_df[f"Forecast_{model_code}_MoM"])
        effect_pp = (
            fact_cum - forecast_cum
            if pd.notna(fact_cum) and pd.notna(forecast_cum)
            else np.nan
        )
        effect_abs = test_df[f"Impact_{model_code}_abs"].sum()

        rows.append({
            "Сценарий": scenario_name,
            "Модель код": model_code,
            "Модель": model_label,
            "Historical Period": f"{hist_start:%Y-%m} → {hist_end:%Y-%m}",
            "Test Period": f"{test_start:%Y-%m} → {test_end:%Y-%m}",
            "Рыночный контекст (cum)": market_cum,
            "Прогноз без акции (cum)": forecast_cum,
            "Факт (cum)": fact_cum,
            "Чистый эффект (п.п.)": effect_pp,
            "Экономический эффект": effect_abs,
            "Beta": diagnostics.beta,
            "R²": diagnostics.r2,
            "L": diagnostics.leverage_l,
            "Ошибка": None,
        })

    return pd.DataFrame(rows)


def evaluate_scenario_by_masks(
    df: pd.DataFrame,
    hist_mask: pd.Series,
    test_mask: pd.Series,
    scenario_name: str,
    run_models_fn,
    cumulative_growth_from_series_fn,
) -> pd.DataFrame:
    """
    Унифицированный запуск сценария через существующее ядро run_models().
    Возвращает табличный результат или строку с ошибкой.
    """
    try:
        hist_df_s, test_df_s, diagnostics_s, _ = run_models_fn(df, hist_mask, test_mask)
        return build_scenario_result_table(
            hist_df=hist_df_s,
            test_df=test_df_s,
            diagnostics=diagnostics_s,
            scenario_name=scenario_name,
            cumulative_growth_from_series_fn=cumulative_growth_from_series_fn,
        )
    except Exception as e:
        return pd.DataFrame([{
            "Сценарий": scenario_name,
            "Модель код": "—",
            "Модель": "Ошибка",
            "Historical Period": "—",
            "Test Period": "—",
            "Рыночный контекст (cum)": np.nan,
            "Прогноз без акции (cum)": np.nan,
            "Факт (cum)": np.nan,
            "Чистый эффект (п.п.)": np.nan,
            "Экономический эффект": np.nan,
            "Beta": np.nan,
            "R²": np.nan,
            "L": np.nan,
            "Ошибка": str(e),
        }])


def format_robustness_table(result_df: pd.DataFrame, format_pct_fn, format_pp_fn, format_num_fn) -> pd.DataFrame:
    """
    Форматирование числовых колонок для вывода в Streamlit.
    """
    if result_df.empty:
        return result_df

    out = result_df.copy()

    for col in ["Рыночный контекст (cum)", "Прогноз без акции (cum)", "Факт (cum)"]:
        if col in out.columns:
            out[col] = out[col].apply(format_pct_fn)

    if "Чистый эффект (п.п.)" in out.columns:
        out["Чистый эффект (п.п.)"] = out["Чистый эффект (п.п.)"].apply(format_pp_fn)

    if "Экономический эффект" in out.columns:
        out["Экономический эффект"] = out["Экономический эффект"].apply(format_num_fn)

    for col in ["Beta", "R²", "L"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: "—" if pd.isna(x) else f"{x:.4f}")

    return out


def summarize_stability_against_actual(
    actual_df: pd.DataFrame,
    stress_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Сводка устойчивости:
    как меняется абсолютный эффект относительно базового сценария.
    """
    rows = []

    actual_valid = actual_df[actual_df["Ошибка"].isna()].copy()
    stress_valid = stress_df[stress_df["Ошибка"].isna()].copy()

    for model_code, model_label in MODEL_LABELS.items():
        actual_series = actual_valid.loc[
            actual_valid["Модель код"] == model_code,
            "Экономический эффект"
        ]
        stress_series = stress_valid.loc[
            stress_valid["Модель код"] == model_code,
            "Экономический эффект"
        ].dropna()

        if actual_series.empty or stress_series.empty:
            continue

        actual_effect = float(actual_series.iloc[0])

        if np.isclose(actual_effect, 0):
            same_sign_share = np.nan
        else:
            actual_sign = np.sign(actual_effect)
            same_sign_share = (
                ((np.sign(stress_series) == actual_sign) | np.isclose(stress_series, 0)).mean()
            )

        rows.append({
            "Модель": model_label,
            "Базовый эффект": actual_effect,
            "Минимум": float(stress_series.min()),
            "Медиана": float(stress_series.median()),
            "Максимум": float(stress_series.max()),
            "Диапазон": float(stress_series.max() - stress_series.min()),
            "Доля сценариев с тем же знаком": same_sign_share,
            "Количество сценариев": int(len(stress_series)),
        })

    return pd.DataFrame(rows)


def format_stability_summary_table(summary_df: pd.DataFrame, format_num_fn) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    out = summary_df.copy()
    for col in ["Базовый эффект", "Минимум", "Медиана", "Максимум", "Диапазон"]:
        out[col] = out[col].apply(format_num_fn)

    out["Доля сценариев с тем же знаком"] = out["Доля сценариев с тем же знаком"].apply(
        lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%"
    )
    return out


def summarize_placebo_results(
    actual_df: pd.DataFrame,
    placebo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Сравнение фактического эффекта с placebo-окнами.
    Это не p-value, а эмпирическая доля placebo-окон,
    где |эффект| не меньше фактического.
    """
    rows = []

    actual_valid = actual_df[actual_df["Ошибка"].isna()].copy()
    placebo_valid = placebo_df[placebo_df["Ошибка"].isna()].copy()

    for model_code, model_label in MODEL_LABELS.items():
        actual_series = actual_valid.loc[
            actual_valid["Модель код"] == model_code,
            "Экономический эффект"
        ]
        placebo_series = placebo_valid.loc[
            placebo_valid["Модель код"] == model_code,
            "Экономический эффект"
        ].dropna()

        if actual_series.empty or placebo_series.empty:
            continue

        actual_effect = float(actual_series.iloc[0])
        placebo_share_more_extreme = (
            (placebo_series.abs() >= abs(actual_effect)).mean()
        )

        rows.append({
            "Модель": model_label,
            "Фактический эффект": actual_effect,
            "Минимум placebo": float(placebo_series.min()),
            "Медиана placebo": float(placebo_series.median()),
            "Максимум placebo": float(placebo_series.max()),
            "Доля placebo-окон с |эффектом| >= фактического": placebo_share_more_extreme,
            "Количество placebo-окон": int(len(placebo_series)),
        })

    return pd.DataFrame(rows)


def format_placebo_summary_table(summary_df: pd.DataFrame, format_num_fn) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    out = summary_df.copy()
    for col in ["Фактический эффект", "Минимум placebo", "Медиана placebo", "Максимум placebo"]:
        out[col] = out[col].apply(format_num_fn)

    out["Доля placebo-окон с |эффектом| >= фактического"] = out[
        "Доля placebo-окон с |эффектом| >= фактического"
    ].apply(lambda x: "—" if pd.isna(x) else f"{x * 100:.1f}%")

    return out


def run_placebo_analysis(
    df: pd.DataFrame,
    hist_mask: pd.Series,
    test_mask: pd.Series,
    select_period_mask_fn,
    run_models_fn,
    cumulative_growth_from_series_fn,
    min_train_months: int = 6
) -> pd.DataFrame:
    """
    Placebo-анализ:
    внутри выбранного Historical Period создаются псевдо-тестовые окна
    той же длины, что и реальный Test Period.
    """
    hist_dates = list(df.loc[hist_mask, "Date"].sort_values().unique())
    test_dates = list(df.loc[test_mask, "Date"].sort_values().unique())
    placebo_len = len(test_dates)

    if len(hist_dates) < (min_train_months + placebo_len):
        return pd.DataFrame()

    results = []
    hist_start_fixed = hist_dates[0]

    for start_idx in range(min_train_months, len(hist_dates) - placebo_len + 1):
        placebo_dates = hist_dates[start_idx:start_idx + placebo_len]
        placebo_start = placebo_dates[0]
        placebo_end = placebo_dates[-1]
        placebo_hist_end = hist_dates[start_idx - 1]

        placebo_hist_mask = select_period_mask_fn(df, hist_start_fixed, placebo_hist_end)
        placebo_test_mask = df["Date"].isin(placebo_dates)

        scenario_name = f"Placebo: {placebo_start:%Y-%m} → {placebo_end:%Y-%m}"
        results.append(
            evaluate_scenario_by_masks(
                df=df,
                hist_mask=placebo_hist_mask,
                test_mask=placebo_test_mask,
                scenario_name=scenario_name,
                run_models_fn=run_models_fn,
                cumulative_growth_from_series_fn=cumulative_growth_from_series_fn,
            )
        )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def run_leave_one_year_out(
    df: pd.DataFrame,
    hist_mask: pd.Series,
    test_mask: pd.Series,
    run_models_fn,
    cumulative_growth_from_series_fn,
    min_hist_months: int = 6
) -> pd.DataFrame:
    """
    Leave-one-year-out:
    исключаем по одному календарному году из Historical Period.
    """
    hist_years = sorted(df.loc[hist_mask, "Date"].dt.year.unique())
    results = []

    for year in hist_years:
        variant_hist_mask = hist_mask & (df["Date"].dt.year != year)

        if int(variant_hist_mask.sum()) < min_hist_months:
            continue

        results.append(
            evaluate_scenario_by_masks(
                df=df,
                hist_mask=variant_hist_mask,
                test_mask=test_mask,
                scenario_name=f"Leave-one-year-out: без {year} года",
                run_models_fn=run_models_fn,
                cumulative_growth_from_series_fn=cumulative_growth_from_series_fn,
            )
        )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def run_leave_one_month_out(
    df: pd.DataFrame,
    hist_mask: pd.Series,
    test_mask: pd.Series,
    run_models_fn,
    cumulative_growth_from_series_fn,
    min_hist_months: int = 6
) -> pd.DataFrame:
    """
    Leave-one-month-out:
    исключаем по одному месяцу из Historical Period.
    """
    hist_dates = list(df.loc[hist_mask, "Date"].sort_values().unique())
    results = []

    for removed_date in hist_dates:
        variant_hist_mask = hist_mask & (df["Date"] != removed_date)

        if int(variant_hist_mask.sum()) < min_hist_months:
            continue

        results.append(
            evaluate_scenario_by_masks(
                df=df,
                hist_mask=variant_hist_mask,
                test_mask=test_mask,
                scenario_name=f"Leave-one-month-out: без {pd.Timestamp(removed_date):%Y-%m}",
                run_models_fn=run_models_fn,
                cumulative_growth_from_series_fn=cumulative_growth_from_series_fn,
            )
        )

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def run_boundary_sensitivity(
    df: pd.DataFrame,
    hist_start: pd.Timestamp,
    hist_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    select_period_mask_fn,
    run_models_fn,
    cumulative_growth_from_series_fn,
    max_shift_months: int = 2,
    min_hist_months: int = 6
) -> pd.DataFrame:
    """
    Проверка чувствительности к границам historical периода.
    Сдвигаем start/end на +/- max_shift_months по доступным точкам ряда.
    """
    pre_test_dates = list(df.loc[df["Date"] < test_start, "Date"].sort_values().unique())

    if hist_start not in pre_test_dates or hist_end not in pre_test_dates:
        return pd.DataFrame()

    base_start_idx = pre_test_dates.index(hist_start)
    base_end_idx = pre_test_dates.index(hist_end)

    test_mask_fixed = select_period_mask_fn(df, test_start, test_end)
    results = []

    for start_shift in range(-max_shift_months, max_shift_months + 1):
        for end_shift in range(-max_shift_months, max_shift_months + 1):
            new_start_idx = base_start_idx + start_shift
            new_end_idx = base_end_idx + end_shift

            if new_start_idx < 0 or new_end_idx >= len(pre_test_dates):
                continue
            if new_start_idx >= new_end_idx:
                continue
            if (new_end_idx - new_start_idx + 1) < min_hist_months:
                continue

            new_hist_start = pre_test_dates[new_start_idx]
            new_hist_end = pre_test_dates[new_end_idx]

            if new_hist_end >= test_start:
                continue

            variant_hist_mask = select_period_mask_fn(df, new_hist_start, new_hist_end)

            scenario_name = (
                f"Boundary sensitivity: start {start_shift:+d}, end {end_shift:+d}"
            )

            results.append(
                evaluate_scenario_by_masks(
                    df=df,
                    hist_mask=variant_hist_mask,
                    test_mask=test_mask_fixed,
                    scenario_name=scenario_name,
                    run_models_fn=run_models_fn,
                    cumulative_growth_from_series_fn=cumulative_growth_from_series_fn,
                )
            )

    if not results:
        return pd.DataFrame()

    result_df = pd.concat(results, ignore_index=True)

    # Удаляем полностью дублирующиеся сценарии по имени и модели
    result_df = result_df.drop_duplicates(subset=["Сценарий", "Модель код"]).reset_index(drop=True)
    return result_df
