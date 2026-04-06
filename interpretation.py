from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st


def _classify_placebo_row(row: pd.Series, format_num_fn) -> Dict[str, str]:
    model = row["Модель"]
    actual = float(row["Фактический эффект"])
    placebo_min = float(row["Минимум placebo"])
    placebo_median = float(row["Медиана placebo"])
    placebo_max = float(row["Максимум placebo"])
    share = float(row["Доля placebo-окон с |эффектом| >= фактического"])
    n = int(row["Количество placebo-окон"])

    actual_abs = abs(actual)
    placebo_max_abs = max(abs(placebo_min), abs(placebo_max))
    placebo_median_abs = abs(placebo_median)

    if np.isclose(placebo_max_abs, 0):
        ratio_to_max = np.inf if actual_abs > 0 else 1.0
    else:
        ratio_to_max = actual_abs / placebo_max_abs

    if share <= 0.05 and ratio_to_max >= 1.5:
        level = "strong"
        label = "Сильный сигнал эффекта"
    elif share <= 0.10 and ratio_to_max >= 1.15:
        level = "strong"
        label = "Сильный сигнал эффекта"
    elif share <= 0.25 and actual_abs > max(placebo_median_abs * 1.5, placebo_max_abs * 0.9):
        level = "moderate"
        label = "Умеренно сильный сигнал эффекта"
    else:
        level = "weak"
        label = "Слабый или неустойчивый сигнал эффекта"

    if n < 5 and level == "strong":
        level = "moderate"
        label = "Умеренно сильный сигнал эффекта"

    text = (
        f"**{model} — {label}.** "
        f"Фактический эффект: **{format_num_fn(actual)}**. "
        f"Максимальный по модулю placebo-эффект: **{format_num_fn(placebo_max_abs)}**. "
        f"Доля placebo-окон с сопоставимым или более сильным эффектом: **{share * 100:.1f}%** "
        f"при количестве placebo-окон **{n}**."
    )

    return {"level": level, "text": text}


def _classify_stability_row(row: pd.Series, context_label: str, format_num_fn) -> Dict[str, str]:
    model = row["Модель"]
    base_effect = float(row["Базовый эффект"])
    effect_min = float(row["Минимум"])
    effect_max = float(row["Максимум"])
    effect_range = float(row["Диапазон"])
    same_sign_share = float(row["Доля сценариев с тем же знаком"])
    n = int(row["Количество сценариев"])

    base_abs = abs(base_effect)
    rel_range = np.nan if np.isclose(base_abs, 0) else effect_range / base_abs

    if same_sign_share == 1.0 and (pd.isna(rel_range) or rel_range <= 0.10):
        level = "strong"
        label = f"Высокая устойчивость эффекта к проверке «{context_label}»"
    elif same_sign_share >= 0.90 and (pd.isna(rel_range) or rel_range <= 0.25):
        level = "moderate"
        label = f"Хорошая устойчивость эффекта к проверке «{context_label}»"
    else:
        level = "weak"
        label = f"Низкая устойчивость эффекта к проверке «{context_label}»"

    rel_range_text = "—" if pd.isna(rel_range) else f"{rel_range * 100:.1f}%"

    text = (
        f"**{model} — {label}.** "
        f"Базовый эффект: **{format_num_fn(base_effect)}**. "
        f"Диапазон по стресс-сценариям: **{format_num_fn(effect_min)} → {format_num_fn(effect_max)}** "
        f"(разброс **{format_num_fn(effect_range)}**, относительный разброс **{rel_range_text}**). "
        f"Доля сценариев с тем же знаком: **{same_sign_share * 100:.1f}%** "
        f"при количестве сценариев **{n}**."
    )

    return {"level": level, "text": text}


def _render_interpretation_box(item: Dict[str, str]) -> None:
    level = item["level"]
    text = item["text"]

    if level == "strong":
        st.success(text)
    elif level == "moderate":
        st.info(text)
    else:
        st.warning(text)


def render_placebo_interpretation(placebo_summary_df: pd.DataFrame, format_num_fn) -> None:
    if placebo_summary_df.empty:
        st.info("Интерпретация placebo-анализа недоступна: недостаточно сценариев.")
        return

    st.markdown("**Автоматическая интерпретация placebo-анализа**")
    st.caption(
        "Маркировка ниже основана на сравнении фактического эффекта с распределением placebo-окон. "
        "Это эмпирическая оценка редкости сигнала, а не классический статистический тест."
    )

    for _, row in placebo_summary_df.iterrows():
        _render_interpretation_box(_classify_placebo_row(row, format_num_fn))


def render_stability_interpretation(summary_df: pd.DataFrame, context_label: str, format_num_fn) -> None:
    if summary_df.empty:
        st.info(f"Интерпретация проверки «{context_label}» недоступна: недостаточно сценариев.")
        return

    st.markdown(f"**Автоматическая интерпретация: {context_label}**")
    st.caption(
        "Маркировка ниже основана на сохранении знака эффекта и размере разброса "
        "между стресс-сценариями. Это оценка устойчивости вывода, а не тест статистической значимости."
    )

    for _, row in summary_df.iterrows():
        _render_interpretation_box(_classify_stability_row(row, context_label, format_num_fn))
