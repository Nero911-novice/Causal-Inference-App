import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models import ModelDiagnostics, cumulative_growth_from_series


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

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Target_Group"],
        mode="lines+markers",
        name="Target_Group (факт)",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=7),
        hovertemplate="<b>%{x|%Y-%m}</b><br>Факт: %{y:,.0f}<extra></extra>"
    ))

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
