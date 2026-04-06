from typing import Tuple

import pandas as pd


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
