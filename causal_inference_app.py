import io
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================
# Конфигурация Streamlit
# =========================
st.set_page_config(
    page_title="Causal Inference App | Аналитическая триангуляция",
    page_icon="📈",
    layout="wide",
)

st.title("Causal Inference App")
st.caption(
    "Оценка чистого эффекта маркетингового воздействия через три независимые модели "
    "аналитической триангуляции."
)


# =========================
# Вспомогательные функции
# =========================
@dataclass
class ModelDiagnostics:
    beta: Optional[float] = None
    r2: Optional[float] = None
    leverage_l: Optional[float] = None
    yoy_support_count: Optional[int] = None
    yoy_fallback_months: Optional[List[str]] = None


def safe_to_datetime(series: pd.Series) -> pd.Series:
    """
    Пытается преобразовать колонку к datetime.
    Поддерживает ISO-дату, day-first и формат месяц-год.
    Совместимо с новыми версиями pandas.
    """
    s = series.astype(str).str.strip()

    # 1. Базовый и наиболее частый формат: YYYY-MM-DD
    parsed = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")

    # 2. Универсальный парсинг без жесткого формата
    if parsed.isna().mean() > 0.3:
        parsed_alt = pd.to_datetime(s, errors="coerce")
        if parsed_alt.notna().sum() > parsed.notna().sum():
            parsed = parsed_alt

    # 3. Day-first формат, если пользователь загрузил даты вида 15.02.2024
    if parsed.isna().mean() > 0.3:
        parsed_alt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        if parsed_alt.notna().sum() > parsed.notna().sum():
            parsed = parsed_alt

    # 4. Формат месяц-год для уже агрегированных месячных рядов
    if parsed.isna().mean() > 0.3:
        parsed_alt = pd.to_datetime(s, errors="coerce", format="%Y-%m")
        if parsed_alt.notna().sum() > parsed.notna().sum():
            parsed = parsed_alt

    return parsed


def read_file(uploaded_file) -> pd.DataFrame:
    """
    Чтение CSV или Excel.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Поддерживаются только CSV и Excel файлы.")


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.2f}%"


def format_pp(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.2f} п.п."


def format_num(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.0f}".replace(",", " ")


def calculate_growth(series: pd.Series) -> pd.Series:
    """
    MoM: (V_t - V_t-1) / V_t-1
    """
    return series.pct_change()


def prepare_dataframe(
    raw_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    control_col: str
) -> pd.DataFrame:
    """
    Подготовка датафрейма:
    - выбор нужных колонок
    - парсинг даты
    - очистка NaN
    - агрегация по месяцу
    - расчет MoM
    """
    df = raw_df[[date_col, target_col, control_col]].copy()
    df.columns = ["Date", "Target_Group", "Control_Group"]

    df["Date"] = safe_to_datetime(df["Date"])
    df["Target_Group"] = pd.to_numeric(df["Target_Group"], errors="coerce")
    df["Control_Group"] = pd.to_numeric(df["Control_Group"], errors="coerce")

    df = df.dropna(subset=["Date", "Target_Group", "Control_Group"]).copy()

    # Приведение к месячной частоте: начало месяца
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Если в одном месяце несколько строк, агрегируем суммой
    df = (
        df.groupby("Date", as_index=False)[["Target_Group", "Control_Group"]]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Проверка на положительные значения для корректного MoM и CV
    if (df["Target_Group"] <= 0).any() or (df["Control_Group"] <= 0).any():
        raise ValueError(
            "Для расчета MoM и коэффициента вариации значения Target_Group и Control_Group должны быть строго положительными."
        )

    df["Target_MoM"] = calculate_growth(df["Target_Group"])
    df["Control_MoM"] = calculate_growth(df["Control_Group"])
    df["Month_Num"] = df["Date"].dt.month
    df["Month_Label"] = df["Date"].dt.strftime("%Y-%m")

    return df


def select_period_mask(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    return (df["Date"] >= start_date) & (df["Date"] <= end_date)


def validate_periods(
    df: pd.DataFrame,
    hist_start: pd.Timestamp,
    hist_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> Tuple[bool, str]:
    if hist_start > hist_end:
        return False, "Historical Period задан некорректно: начало позже конца."
    if test_start > test_end:
        return False, "Test Period задан некорректно: начало позже конца."
    if hist_end >= test_start:
        return False, "Historical Period должен завершаться раньше начала Test Period."
    if test_start not in set(df["Date"]):
        return False, "Начало Test Period отсутствует в ряду данных."
    if hist_start not in set(df["Date"]) or hist_end not in set(df["Date"]):
        return False, "Границы Historical Period отсутствуют в ряду данных."
    if test_end not in set(df["Date"]):
        return False, "Конец Test Period отсутствует в ряду данных."

    # Должна существовать точка до начала теста для базового объема
    previous_dates = df.loc[df["Date"] < test_start, "Date"]
    if previous_dates.empty:
        return False, "Для Test Period необходим хотя бы один месяц до его начала."
    return True, ""


def descriptive_stats(hist_df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame({
        "Показатель": [
            "Количество наблюдений",
            "Среднее Target_Group",
            "Среднее Control_Group",
            "Средний MoM Target",
            "Средний MoM Control",
            "Стандартное отклонение MoM Target",
            "Стандартное отклонение MoM Control",
            "Корреляция MoM Target vs Control",
            "Минимум Target_Group",
            "Максимум Target_Group",
            "Минимум Control_Group",
            "Максимум Control_Group",
        ],
        "Значение": [
            len(hist_df),
            hist_df["Target_Group"].mean(),
            hist_df["Control_Group"].mean(),
            hist_df["Target_MoM"].mean(),
            hist_df["Control_MoM"].mean(),
            hist_df["Target_MoM"].std(ddof=1),
            hist_df["Control_MoM"].std(ddof=1),
            hist_df[["Target_MoM", "Control_MoM"]].corr().iloc[0, 1] if len(hist_df) > 1 else np.nan,
            hist_df["Target_Group"].min(),
            hist_df["Target_Group"].max(),
            hist_df["Control_Group"].min(),
            hist_df["Control_Group"].max(),
        ]
    })

    def nice(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (int, np.integer)):
            return str(v)
        if abs(v) < 1 and v != 0:
            return f"{v:.4f}"
        if abs(v) >= 1000:
            return f"{v:,.2f}".replace(",", " ")
        return f"{v:.4f}"

    stats["Значение"] = stats["Значение"].apply(nice)
    return stats


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


def create_summary_table(test_df: pd.DataFrame, diagnostics: ModelDiagnostics) -> pd.DataFrame:
    fact_cum = cumulative_growth_from_series(test_df["Fact_MoM"])
    market_cum = cumulative_growth_from_series(test_df["Control_MoM"])

    rows = []
    for model_name, label in [
        ("M1", "Модель 1: Регрессия чувствительности"),
        ("M2", "Модель 2: CV Leverage"),
        ("M3", "Модель 3: Сезонная эффективность YoY"),
    ]:
        forecast_cum = cumulative_growth_from_series(test_df[f"Forecast_{model_name}_MoM"])
        impact_pp_cum = fact_cum - forecast_cum if pd.notna(fact_cum) and pd.notna(forecast_cum) else np.nan
        economic_effect = test_df[f"Impact_{model_name}_abs"].sum()

        rows.append({
            "Модель": label,
            "Рыночный контекст (Факт рынка)": format_pct(market_cum),
            "Прогноз падения/роста без акции": format_pct(forecast_cum),
            "Фактический результат": format_pct(fact_cum),
            "Чистый эффект (п.п.)": format_pp(impact_pp_cum),
            "Экономический эффект (в абсолютных числах)": format_num(economic_effect),
        })

    return pd.DataFrame(rows)


def create_detailed_impact_table(test_df: pd.DataFrame) -> pd.DataFrame:
    detail = pd.DataFrame({
        "Дата": test_df["Date"].dt.strftime("%Y-%m"),
        "Факт рынка (Control MoM)": test_df["Control_MoM"].apply(format_pct),
        "Факт Target (MoM)": test_df["Fact_MoM"].apply(format_pct),
        "Прогноз M1 (MoM)": test_df["Forecast_M1_MoM"].apply(format_pct),
        "Прогноз M2 (MoM)": test_df["Forecast_M2_MoM"].apply(format_pct),
        "Прогноз M3 (MoM)": test_df["Forecast_M3_MoM"].apply(format_pct),
        "Эффект M1 (п.п.)": test_df["Impact_M1_pp"].apply(format_pp),
        "Эффект M2 (п.п.)": test_df["Impact_M2_pp"].apply(format_pp),
        "Эффект M3 (п.п.)": test_df["Impact_M3_pp"].apply(format_pp),
        "Эффект M1 (абс.)": test_df["Impact_M1_abs"].apply(format_num),
        "Эффект M2 (абс.)": test_df["Impact_M2_abs"].apply(format_num),
        "Эффект M3 (абс.)": test_df["Impact_M3_abs"].apply(format_num),
    })
    return detail


def create_plot(df: pd.DataFrame, test_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Общий фактический ряд Target
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Target_Group"],
        mode="lines+markers",
        name="Target_Group (факт)",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Факт: %{y:,.0f}<extra></extra>"
    ))

    # Контрфактуальные линии только в тестовом периоде
    fig.add_trace(go.Scatter(
        x=test_df["Date"],
        y=test_df["Baseline_M1"],
        mode="lines+markers",
        name="Модель 1 baseline",
        line=dict(color="#d62728", width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%Y-%m}</b><br>M1 baseline: %{y:,.0f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=test_df["Date"],
        y=test_df["Baseline_M2"],
        mode="lines+markers",
        name="Модель 2 baseline",
        line=dict(color="#2ca02c", width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%Y-%m}</b><br>M2 baseline: %{y:,.0f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=test_df["Date"],
        y=test_df["Baseline_M3"],
        mode="lines+markers",
        name="Модель 3 baseline",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="<b>%{x|%Y-%m}</b><br>M3 baseline: %{y:,.0f}<extra></extra>"
    ))

    # Подсветка тестового окна
    fig.add_vrect(
        x0=test_df["Date"].min(),
        x1=test_df["Date"].max(),
        fillcolor="rgba(100, 149, 237, 0.12)",
        line_width=0,
        annotation_text="Test Period",
        annotation_position="top left"
    )

    fig.update_layout(
        title="Факт Target_Group и три контрфактуальных сценария",
        xaxis_title="Дата",
        yaxis_title="Абсолютное значение",
        hovermode="x unified",
        legend_title="Сценарии",
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    fig.update_xaxes(dtick="M1", tickformat="%Y-%m")
    return fig


def to_excel_bytes(summary_df: pd.DataFrame, detailed_df: pd.DataFrame, test_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        detailed_df.to_excel(writer, sheet_name="Monthly_Impact", index=False)

        export_raw = test_df.copy()
        export_raw["Date"] = export_raw["Date"].dt.strftime("%Y-%m")
        export_raw.to_excel(writer, sheet_name="Raw_Test_Calculations", index=False)
    buffer.seek(0)
    return buffer.getvalue()


# =========================
# Сайдбар
# =========================
st.sidebar.header("Загрузка данных и параметры")

uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV или Excel",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info(
        "Загрузите файл в формате CSV/XLSX/XLS. "
        "Ожидаются колонки с датой, целевой группой и контрольной группой."
    )
    st.stop()

try:
    raw_df = read_file(uploaded_file)
except Exception as e:
    st.error(f"Ошибка чтения файла: {e}")
    st.stop()

if raw_df.empty:
    st.error("Файл пустой.")
    st.stop()

all_columns = list(raw_df.columns)

st.sidebar.subheader("Выбор колонок")
date_col = st.sidebar.selectbox("Колонка даты", all_columns, index=0)
target_col = st.sidebar.selectbox("Колонка Target_Group", all_columns, index=min(1, len(all_columns) - 1))
control_col = st.sidebar.selectbox("Колонка Control_Group", all_columns, index=min(2, len(all_columns) - 1))

if len({date_col, target_col, control_col}) < 3:
    st.error("Необходимо выбрать три разные колонки: дата, целевая группа и контрольная группа.")
    st.stop()

try:
    df = prepare_dataframe(raw_df, date_col, target_col, control_col)
except Exception as e:
    st.error(f"Ошибка подготовки данных: {e}")
    with st.expander("Показать первые строки исходного файла"):
        st.dataframe(raw_df.head(20), use_container_width=True)
    st.stop()

if len(df) < 6:
    st.error("Недостаточно данных. Желательно иметь не менее 6 месяцев наблюдений.")
    st.stop()

available_dates = list(df["Date"].sort_values().unique())
date_labels = [pd.Timestamp(d).strftime("%Y-%m") for d in available_dates]
date_map = {label: pd.Timestamp(date) for label, date in zip(date_labels, available_dates)}

st.sidebar.subheader("Historical Period")
default_hist_start_idx = 0
default_hist_end_idx = max(1, len(date_labels) // 2 - 1)

hist_start_label = st.sidebar.selectbox(
    "Начало historical period",
    date_labels,
    index=default_hist_start_idx,
    key="hist_start"
)
hist_end_label = st.sidebar.selectbox(
    "Конец historical period",
    date_labels,
    index=default_hist_end_idx,
    key="hist_end"
)

st.sidebar.subheader("Test Period")
default_test_start_idx = min(default_hist_end_idx + 1, len(date_labels) - 2)
default_test_end_idx = len(date_labels) - 1

test_start_label = st.sidebar.selectbox(
    "Начало test period",
    date_labels,
    index=default_test_start_idx,
    key="test_start"
)
test_end_label = st.sidebar.selectbox(
    "Конец test period",
    date_labels,
    index=default_test_end_idx,
    key="test_end"
)

hist_start = date_map[hist_start_label]
hist_end = date_map[hist_end_label]
test_start = date_map[test_start_label]
test_end = date_map[test_end_label]

is_valid, validation_message = validate_periods(df, hist_start, hist_end, test_start, test_end)
if not is_valid:
    st.error(validation_message)
    st.stop()

hist_mask = select_period_mask(df, hist_start, hist_end)
test_mask = select_period_mask(df, test_start, test_end)

# Дополнительная защита: Test Period должен иметь рассчитанный Control_MoM
test_df_check = df.loc[test_mask]
if test_df_check["Control_MoM"].isna().any():
    st.error(
        "В Test Period присутствуют месяцы, для которых не удалось рассчитать динамику Control_Group. "
        "Проверьте непрерывность ряда и выбор периода."
    )
    st.stop()

if test_df_check["Target_MoM"].isna().any():
    st.error(
        "В Test Period присутствуют месяцы, для которых не удалось рассчитать динамику Target_Group. "
        "Проверьте непрерывность ряда и выбор периода."
    )
    st.stop()

try:
    hist_df, test_df, diagnostics, baseline_anchor_value = run_models(df, hist_mask, test_mask)
except Exception as e:
    st.error(f"Ошибка расчета моделей: {e}")
    st.stop()


# =========================
# Основной экран
# =========================
# Блок 0: ключевые метрики
st.subheader("Ключевые показатели")

summary_df = create_summary_table(test_df, diagnostics)
detailed_df = create_detailed_impact_table(test_df)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Beta (Модель 1)", f"{diagnostics.beta:.4f}" if diagnostics.beta is not None else "—")
with col2:
    st.metric("R² (Модель 1)", f"{diagnostics.r2:.4f}" if diagnostics.r2 is not None and not pd.isna(diagnostics.r2) else "—")
with col3:
    st.metric("Рычаг L (Модель 2)", f"{diagnostics.leverage_l:.4f}" if diagnostics.leverage_l is not None else "—")
with col4:
    st.metric("Базовый объём перед тестом", format_num(baseline_anchor_value))

st.subheader("Качество моделей")

quality_notes = []

# Модель 1: регрессия
if diagnostics.r2 is not None and not pd.isna(diagnostics.r2):
    if diagnostics.r2 >= 0.7:
        quality_notes.append("**Модель 1:** высокая объясняющая сила. Связь целевой группы с рынком выглядит устойчивой.")
    elif diagnostics.r2 >= 0.4:
        quality_notes.append("**Модель 1:** средняя объясняющая сила. Рыночный сигнал полезен, но не исчерпывает поведение целевой группы.")
    else:
        quality_notes.append("**Модель 1:** слабая объясняющая сила. Результат следует трактовать осторожно.")

if diagnostics.beta is not None:
    if diagnostics.beta < 0:
        quality_notes.append("**Beta < 0:** целевая группа исторически двигалась против рынка. Такая спецификация требует особенно внимательной интерпретации.")
    elif diagnostics.beta > 1.3:
        quality_notes.append("**Beta > 1:** целевая группа исторически реагировала на рынок сильнее самого рынка.")
    elif diagnostics.beta < 0.7:
        quality_notes.append("**Beta < 1:** целевая группа исторически более инертна, чем рынок.")

# Модель 2: CV Leverage
if diagnostics.leverage_l is not None and not pd.isna(diagnostics.leverage_l):
    if 0.85 <= diagnostics.leverage_l <= 1.15:
        quality_notes.append("**Модель 2:** целевая группа по общей волатильности близка к рынку. Метод дает нейтральную sanity-check оценку.")
    elif diagnostics.leverage_l < 0.85:
        quality_notes.append("**Модель 2:** целевая группа исторически более волатильна, чем рынок. Метод будет усиливать рыночные движения.")
    else:
        quality_notes.append("**Модель 2:** целевая группа исторически более стабильна, чем рынок. Метод будет сглаживать рыночные движения.")

# Модель 3: сезонность
hist_months_covered = hist_df["Month_Num"].nunique()

if hist_months_covered >= 10:
    quality_notes.append("**Модель 3:** сезонное покрытие хорошее. Историческая сезонность представлена достаточно широко.")
elif hist_months_covered >= 6:
    quality_notes.append("**Модель 3:** сезонное покрытие умеренное. Сезонную оценку желательно трактовать как ориентир, а не как единственный baseline.")
else:
    quality_notes.append("**Модель 3:** сезонное покрытие слабое. Надежность сезонной оценки ограничена.")

if diagnostics.yoy_fallback_months:
    quality_notes.append("**Модель 3:** для части месяцев не хватило сезонной истории, поэтому использовалось общее среднее исторического периода.")

for note in quality_notes:
    st.markdown(f"- {note}")

st.markdown("---")

# Блок 1: Дескриптивная статистика
st.subheader("1. Дескриптивная статистика за Historical Period")

info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.metric("Период Historical", f"{hist_start:%Y-%m} → {hist_end:%Y-%m}")
with info_col2:
    st.metric("Период Test", f"{test_start:%Y-%m} → {test_end:%Y-%m}")
with info_col3:
    st.metric("Наблюдений в Historical", str(len(hist_df)))

st.dataframe(descriptive_stats(hist_df), use_container_width=True, hide_index=True)

with st.expander("Показать подготовленный датасет"):
    preview = df.copy()
    preview["Date"] = preview["Date"].dt.strftime("%Y-%m")
    st.dataframe(preview, use_container_width=True)

if diagnostics.yoy_fallback_months:
    st.warning(
        "Для части месяцев в Модели 3 отсутствовали исторические наблюдения по тому же календарному месяцу. "
        "Использовано общее среднее Historical Period для: "
        + ", ".join(diagnostics.yoy_fallback_months)
    )

st.markdown("---")

# Блок 2: График
st.subheader("2. Интерактивный график: факт и три baseline-сценария")
fig = create_plot(df, test_df)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Блок 3: Итоговая таблица триангуляции
st.subheader("3. Итоговая таблица триангуляции")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

best_model_idx = (
    test_df[["Impact_M1_abs", "Impact_M2_abs", "Impact_M3_abs"]]
    .sum()
    .abs()
    .idxmax()
)
best_model_label = {
    "Impact_M1_abs": "Модель 1: Регрессия чувствительности",
    "Impact_M2_abs": "Модель 2: CV Leverage",
    "Impact_M3_abs": "Модель 3: Сезонная эффективность YoY"
}[best_model_idx]

best_effect_value = test_df[best_model_idx].sum()

mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.metric("Фактическая суммарная динамика Target", format_pct(cumulative_growth_from_series(test_df["Fact_MoM"])))
with mcol2:
    st.metric("Фактическая суммарная динамика рынка", format_pct(cumulative_growth_from_series(test_df["Control_MoM"])))
with mcol3:
    st.metric("Наибольший оцененный эффект", f"{best_model_label}: {format_num(best_effect_value)}")

with st.expander("Помесячная детализация расчета impact"):
    st.dataframe(detailed_df, use_container_width=True, hide_index=True)

# Краткая методологическая справка
with st.expander("Методологические пояснения"):
    st.markdown(
        """
        ### Назначение приложения

        Приложение оценивает **чистый эффект маркетингового воздействия** через построение
        **контрфактуального сценария**: каким был бы результат целевой группы, если бы акции не было.

        Основа подхода — **аналитическая триангуляция**. Это означает, что один и тот же эффект
        оценивается не одной моделью, а **тремя независимыми логиками**:

        1. через связь целевой группы с рынком;
        2. через относительную стабильность и волатильность ряда;
        3. через историческую сезонность.

        Если разные методы, опирающиеся на разные математические основания, дают близкий вывод,
        это повышает доверие к оценке эффекта.

        ---

        ### Почему расчеты ведутся не по абсолютным значениям, а по динамике

        Все модели работают не с самими объемами продаж, заказов или иной метрики, а с
        **месячной динамикой (Month-over-Month, MoM)**.

        Формула динамики:

        \\[
        R_t = \\frac{V_t - V_{t-1}}{V_{t-1}}
        \\]

        где:

        - \\(V_t\\) — значение показателя в текущем месяце,
        - \\(V_{t-1}\\) — значение показателя в предыдущем месяце.

        Такой подход нужен потому, что задача приложения — оценить не просто уровень метрики,
        а **изменение поведения ряда** на фоне рынка, сезонности и маркетингового воздействия.

        Работа с динамикой особенно полезна в условиях рыночных шоков, когда абсолютные уровни
        могут меняться из-за инфляции, общего роста бизнеса, изменения масштаба рынка или других
        внешних факторов.

        ---

        ### Структура анализа

        Анализ разделяется на два периода:

        1. **Historical Period** — исторический базовый период, на котором модели обучаются
           или извлекают закономерности.
        2. **Test Period** — период маркетингового воздействия, для которого оценивается эффект.

        Логика расчета всегда одна и та же:

        1. На историческом периоде определяется закономерность.
        2. На тестовом периоде строится **baseline** — прогноз того, что должно было бы произойти
           без акции.
        3. Фактический результат сравнивается с этим baseline.
        4. Разница интерпретируется как **чистый эффект**.

        ---

        ### Модель 1. Регрессионный анализ чувствительности (адаптация CAPM)

        #### Идея метода

        Метод исходит из предположения, что целевая группа исторически реагирует на изменения рынка
        с определенной чувствительностью.

        Здесь:

        - **Control_Group** — это внешний рыночный фон, контрольная группа или прокси рынка;
        - **Target_Group** — это сегмент, на который направлено маркетинговое воздействие.

        Модель оценивает, насколько изменение целевой группы обычно связано с изменением контрольной группы.

        #### Математическая логика

        На историческом периоде строится линейная зависимость:

        \\[
        Target\\_MoM = \\beta \\cdot Control\\_MoM
        \\]

        где:

        - \\(\\beta\\) — коэффициент чувствительности;
        - \\(R^2\\) — показатель того, насколько хорошо рынок объясняет поведение целевой группы.

        #### Как интерпретировать \\(\\beta\\)

        - **\\(\\beta = 1\\)** — целевая группа меняется примерно тем же темпом, что и рынок;
        - **\\(\\beta > 1\\)** — целевая группа реагирует сильнее рынка;
        - **\\(\\beta < 1\\)** — целевая группа более инертна и реагирует слабее рынка;
        - **\\(\\beta < 0\\)** — движение целевой группы исторически было противоположно рынку.

        #### Как интерпретировать \\(R^2\\)

        - высокое \\(R^2\\) означает, что рынок действительно хорошо объясняет поведение целевой группы;
        - низкое \\(R^2\\) означает, что связь слабая, и результаты модели нужно трактовать осторожнее.

        #### Как строится прогноз

        Для каждого месяца тестового периода приложение берет **фактическую динамику рынка**
        и умножает ее на \\(\\beta\\):

        \\[
        Forecast\\_MoM = Control\\_MoM \\cdot \\beta
        \\]

        Это и есть ожидаемая динамика целевой группы **без акции**.

        #### Когда метод особенно полезен

        Метод наиболее уместен, если:

        1. у целевой группы есть устойчивая связь с рынком;
        2. основной аналитический риск связан с внешним рыночным фоном;
        3. требуется объяснить эффект именно через отклонение от нормального рыночного поведения.

        #### Ограничения метода

        1. Предполагается, что историческая связь между целевой и контрольной группой
           сохраняется в тестовом периоде.
        2. Метод плохо отражает нелинейные режимы, пороги, насыщение и структурные переломы.
        3. Если контрольная группа сама загрязнена влиянием той же акции, baseline может быть смещен.

        ---

        ### Модель 2. Эвристический подход через коэффициент вариации (CV Leverage)

        #### Идея метода

        Этот метод не пытается напрямую оценить причинную связь с рынком через регрессию.
        Он отвечает на другой вопрос:

        **насколько целевая группа исторически более стабильна или более волатильна, чем рынок?**

        То есть метод оценивает не столько совместное движение двух рядов,
        сколько **относительную подвижность** целевого ряда.

        #### Математическая логика

        На историческом периоде рассчитывается коэффициент вариации:

        \\[
        CV = \\frac{\\sigma}{\\mu}
        \\]

        где:

        - \\(\\sigma\\) — стандартное отклонение,
        - \\(\\mu\\) — среднее значение ряда.

        Далее рассчитывается коэффициент:

        \\[
        L = \\frac{CV_{control}}{CV_{target}}
        \\]

        где:

        - \\(CV_{control}\\) — коэффициент вариации контрольной группы,
        - \\(CV_{target}\\) — коэффициент вариации целевой группы,
        - \\(L\\) — рычаг стабильности.

        #### Как интерпретировать \\(L\\)

        - **\\(L > 1\\)** — целевая группа исторически более стабильна, чем рынок;
        - **\\(L < 1\\)** — целевая группа более волатильна, чем рынок;
        - **\\(L \\approx 1\\)** — по общей степени нестабильности целевая группа близка к рынку.

        #### Как строится прогноз

        Прогнозная динамика определяется так:

        \\[
        Forecast\\_MoM = \\frac{Control\\_MoM}{L}
        \\]

        Смысл формулы следующий:

        - если целевая группа обычно более стабильна, чем рынок, ее ожидаемое движение будет сглажено;
        - если она более волатильна, чем рынок, ее ожидаемое движение будет усилено.

        #### Как трактовать метод содержательно

        Это **эвристическая sanity-check модель**. Она отвечает на вопрос:

        > если взять фактическое движение рынка и скорректировать его на историческую инертность
        > или волатильность целевой группы, какой baseline получится без акции?

        #### Когда метод полезен

        1. как дополнительная независимая проверка основного вывода;
        2. когда важно понять, является ли сегмент более нервным или более устойчивым, чем рынок;
        3. когда требуется не одна модель, а диапазон оценок.

        #### Ограничения метода

        1. Это не строгая регрессия и не полноценная причинная модель.
        2. Метод использует коэффициенты вариации по абсолютным уровням, а не моделирует
           прямую месячную связь между рядами.
        3. Он не оценивает качество связи аналогом \\(R^2\\).
        4. Его лучше использовать как **поддерживающий метод**, а не как единственную основу вывода.

        ---

        ### Модель 3. Сезонная эффективность (Year-over-Year логика)

        #### Идея метода

        Этот метод исходит из предположения, что у целевой группы есть устойчивая
        **календарная сезонность**.

        Он отвечает на вопрос:

        **какой темп изменения был бы у целевой группы в данном месяце года, если бы сработала
        обычная историческая сезонная норма?**

        #### Как строится прогноз

        Для каждого месяца тестового периода берется средний исторический MoM целевой группы
        для того же календарного месяца на историческом периоде.

        Пример логики:

        - для мая тестового периода берется средний MoM всех прошлых маев;
        - для сентября — средний MoM всех прошлых сентябрей и так далее.

        Если в историческом периоде недостаточно наблюдений по конкретному месяцу,
        приложение может использовать общий средний MoM по историческому периоду как резервный вариант.

        #### Как трактовать метод

        Это модель **календарной нормы**. Она не отвечает на вопрос, как сегмент связан с рынком.
        Она отвечает на вопрос, что для данного месяца обычно характерно с точки зрения сезонности.

        #### Когда метод особенно полезен

        1. если бизнес подвержен ярко выраженной сезонности;
        2. если необходимо избежать переоценки маркетингового эффекта в сильные сезонные месяцы;
        3. если нужно получить более консервативный baseline.

        #### Ограничения метода

        1. Метод не учитывает текущий рыночный контекст напрямую.
        2. Если исторический период короткий, сезонная оценка может быть шумной.
        3. При структурном изменении бизнеса прошлые сезонные паттерны могут уже не воспроизводиться.

        ---

        ### Как рассчитывается чистый эффект

        После того как baseline построен, приложение сравнивает **факт** и **прогноз без акции**.

        Для каждого месяца тестового периода:

        \\[
        Impact\\_{pp} = Fact\\_MoM - Forecast\\_MoM
        \\]

        где:

        - **Fact_MoM** — фактическая динамика целевой группы;
        - **Forecast_MoM** — прогнозная динамика без акции;
        - **Impact\\_{pp}** — чистый эффект в процентных пунктах.

        Далее эффект переводится в абсолютные единицы.

        В приложении baseline восстанавливается как **контрфактуальная траектория абсолютных значений**
        от последнего фактического месяца перед тестом. Поэтому экономический эффект считается как
        разница между фактическим рядом и baseline-рядом в абсолютных величинах по каждому месяцу тестового периода.

        Практически это означает:

        - если факт выше baseline, эффект положительный;
        - если факт ниже baseline, эффект отрицательный;
        - если факт близок к baseline, выраженного инкрементального эффекта не наблюдается.

        ---

        ### Как интерпретировать результаты трех моделей вместе

        Смысл триангуляции состоит не в том, чтобы выбрать одну «абсолютно правильную» модель,
        а в том, чтобы сравнить несколько независимых оценок.

        Возможны три основных сценария:

        #### 1. Все модели показывают эффект одного знака и сопоставимого масштаба
        Это наиболее сильный случай. Вывод об эффекте выглядит устойчивым.

        #### 2. Эффект положительный у всех моделей, но масштаб различается
        Это означает, что знак эффекта устойчив, но его величина зависит от того,
        какой baseline считать более реалистичным:
        - рыночный,
        - волатильностный,
        - сезонный.

        #### 3. Модели дают противоречивые выводы
        Это сигнал к дополнительной диагностике:
        - возможно, исторический период выбран неудачно;
        - возможно, контрольная группа плохо отражает рынок;
        - возможно, в данных сильный структурный разрыв;
        - возможно, сезонность и рыночный фон дают разнонаправленные сигналы.

        ---

        ### Практическая рамка интерпретации

        Для управленческого вывода модели удобно читать так:

        1. **Модель 1** — baseline через историческую чувствительность к рынку.
        2. **Модель 2** — baseline через относительную стабильность и волатильность сегмента.
        3. **Модель 3** — baseline через историческую сезонную норму.

        Поэтому итоговый результат следует трактовать как **диапазон обоснованных оценок эффекта**,
        а не обязательно как единственную цифру.

        ---

        ### Что считать сильным результатом

        Вывод об эффекте становится особенно убедительным, если одновременно выполняются условия:

        1. исторический период достаточно длинный и качественный;
        2. контрольная группа действительно отражает внешний фон;
        3. регрессионная модель имеет осмысленные параметры;
        4. сезонная модель опирается на достаточное число исторических наблюдений;
        5. итоговый знак эффекта устойчив между моделями.

        ---

        ### Ключевой смысл приложения

        Приложение не доказывает причинность в философски абсолютном смысле.
        Оно делает более практическую и аналитически ценную задачу:

        **строит несколько независимых контрфактуальных baseline-сценариев и оценивает,
        насколько фактический результат отклоняется от того, что было бы ожидаемо без воздействия.**

        Именно это и позволяет использовать его как инструмент оценки
        **инкрементального маркетингового эффекта** в условиях нестабильного рынка.
        """
    )

# Скачивание результатов
excel_bytes = to_excel_bytes(summary_df, detailed_df, test_df)
st.download_button(
    label="Скачать результаты в Excel",
    data=excel_bytes,
    file_name="causal_inference_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
