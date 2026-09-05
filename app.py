from __future__ import annotations

from io import BytesIO
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# yfinance requiere una ubicación escribible para su caché de zonas horarias.
# Se conserva explícitamente porque evita fallos de arranque en varios equipos Windows.
yf.set_tz_cache_location("cache")
from scipy.optimize import brentq


st.set_page_config(page_title="Analizador de Portafolio", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .stApp {background:#1b2230; color:#f8fafc;}
      .block-container {max-width:1420px; padding-top:2.2rem; padding-bottom:3rem;}
      [data-testid="stMetric"] {background:linear-gradient(135deg,#253249,#202b3d); border:1px solid #3a4a63; border-radius:14px; padding:14px 16px; box-shadow:0 6px 16px rgba(0,0,0,.16);}
      [data-testid="stMetricLabel"] {color:#a9b8cc; font-size:.82rem;}
      [data-testid="stMetricValue"] {color:#f8fafc;}
      .hero {padding:1.5rem 1.8rem; border-radius:18px; color:white;
             background:linear-gradient(115deg,#0e1d33,#145b82); margin-bottom:1.4rem; border:1px solid #2b5474;}
      .hero h1 {font-size:2rem; margin:0 0 .35rem 0;}
      .hero p {margin:0; opacity:.87;}
      div[data-testid="stFileUploader"] {border:1.5px dashed #3b82a7; border-radius:14px; padding:8px; background:#202d42;}
      [data-testid="stFileUploader"] section {background:#202d42;}
      .section-label {margin:1.25rem 0 .55rem; color:#9fb0c8; font-size:.78rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase;}
      .metric-card {height:100%; min-height:104px; box-sizing:border-box; padding:14px 16px; border-radius:14px; background:linear-gradient(135deg,#253249,#202b3d); border:1px solid #3a4a63; border-top:3px solid var(--accent); box-shadow:0 6px 16px rgba(0,0,0,.16);}
      .metric-card-title {color:#a9b8cc; font-size:.78rem; font-weight:600;}
      .metric-card-value {color:var(--accent); font-size:1.38rem; font-weight:750; line-height:1.25; margin-top:.32rem;}
      .metric-card-detail {color:#8fa2bc; font-size:.72rem; margin-top:.35rem; line-height:1.25;}
    </style>
    """,
    unsafe_allow_html=True,
)

MESES = {
    "ene": "01", "feb": "02", "mar": "03", "abr": "04", "may": "05", "jun": "06",
    "jul": "07", "ago": "08", "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dic": "12",
}


def moneda(valor: float | None) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    return f"${valor:,.0f}"


def porcentaje(valor: float | None) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{valor:.2%}"


def puntos_porcentuales(valor: float | None) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{valor * 100:+.2f} pp"


def tarjeta_metricas(titulo: str, valor: str, detalle: str, color: str) -> None:
    st.markdown(
        f"""<div class="metric-card" style="--accent:{color}">
            <div class="metric-card-title">{titulo}</div>
            <div class="metric-card-value">{valor}</div>
            <div class="metric-card-detail">{detalle}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def leer_csv(archivo) -> pd.DataFrame:
    """Lee CSV con coma o punto y coma y normaliza las columnas monetarias."""
    contenido = archivo.getvalue()
    try:
        data = pd.read_csv(BytesIO(contenido), sep=None, engine="python")
    except UnicodeDecodeError:
        data = pd.read_csv(BytesIO(contenido), sep=None, engine="python", encoding="latin1")
    data.columns = data.columns.str.strip()
    requeridas = {"Fecha y hora", "Tipo de movimiento", "Estado", "Valor total"}
    faltantes = requeridas - set(data.columns)
    if faltantes:
        raise ValueError("Faltan estas columnas: " + ", ".join(sorted(faltantes)))

    fecha_texto = data["Fecha y hora"].astype(str).str.extract(r"^(.*?)(?:,|$)")[0].str.strip().str.lower()
    for mes, numero in MESES.items():
        fecha_texto = fecha_texto.str.replace(rf"\b{mes}\b", numero, regex=True)
    # Formato esperado: 2 sept 2026. También acepta fechas ISO o que pandas pueda reconocer.
    partes = fecha_texto.str.extract(r"^(\d{1,2})\s+(\d{1,2})\s+(\d{4})$")
    fecha_convertida = pd.to_datetime(partes[2] + "-" + partes[1] + "-" + partes[0], errors="coerce")
    sin_formato = fecha_convertida.isna()
    fecha_convertida.loc[sin_formato] = pd.to_datetime(fecha_texto.loc[sin_formato], errors="coerce", dayfirst=True)
    if fecha_convertida.isna().any():
        raise ValueError(f"Hay {fecha_convertida.isna().sum()} fecha(s) que no se pudieron interpretar.")

    valor = data["Valor total"].astype(str).str.strip().str.replace(r"[^0-9,.-]", "", regex=True)
    # Si hay coma y punto, se asume punto de miles y coma decimal; si solo coma, decimal.
    ambos = valor.str.contains(",", regex=False) & valor.str.contains(".", regex=False)
    valor.loc[ambos] = valor.loc[ambos].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    solo_coma = valor.str.contains(",", regex=False) & ~ambos
    valor.loc[solo_coma] = valor.loc[solo_coma].str.replace(",", ".", regex=False)
    data["Valor total"] = pd.to_numeric(valor, errors="coerce")
    if data["Valor total"].isna().any():
        raise ValueError("Hay valores monetarios inválidos en 'Valor total'.")
    data["Fecha"] = fecha_convertida
    data["Tipo de movimiento"] = data["Tipo de movimiento"].astype(str).str.strip()
    data["Estado"] = data["Estado"].astype(str).str.strip()
    return data


def calcular_xirr(fechas, flujos) -> float:
    fechas = pd.to_datetime(fechas)
    flujos = np.asarray(flujos, dtype=float)
    if not ((flujos > 0).any() and (flujos < 0).any()):
        return np.nan
    dias = np.asarray((fechas - fechas.min()).days, dtype=float)

    def npv(tasa):
        return np.sum(flujos / (1 + tasa) ** (dias / 365.0))

    # Brent es más estable que Newton cuando los flujos son irregulares.
    limites = [-0.9999, -0.99, -0.9, -0.5, -0.1, 0, 0.1, 0.5, 1, 3, 10, 100]
    valores = [npv(x) for x in limites]
    for a, b, fa, fb in zip(limites[:-1], limites[1:], valores[:-1], valores[1:]):
        if fa == 0:
            return a
        if fa * fb < 0:
            try:
                return brentq(npv, a, b, maxiter=500)
            except ValueError:
                pass
    return np.nan


@st.cache_data(ttl=3600, show_spinner=False)
def descargar_colcap(inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.Series:
    datos = yf.download("ICOLCAP.CL", start=inicio.date(), end=(fin + pd.Timedelta(days=6)).date(), progress=False, auto_adjust=True)
    if datos.empty:
        raise RuntimeError("Yahoo Finance no devolvió precios para ICOLCAP.CL.")
    cierre = datos["Close"]
    if isinstance(cierre, pd.DataFrame):
        cierre = cierre.iloc[:, 0]
    cierre.index = pd.to_datetime(cierre.index).tz_localize(None)
    return cierre


def simular_colcap(movimientos: pd.DataFrame, inicio: pd.Timestamp, fin: pd.Timestamp):
    cierre = descargar_colcap(inicio, fin)
    calendario = pd.date_range(inicio.normalize(), fin.normalize(), freq="D")
    precios = cierre.reindex(calendario).ffill().bfill()
    if precios.isna().any():
        raise RuntimeError("No fue posible alinear los precios de ICOLCAP con las fechas de movimientos.")
    aportes_cuotas = pd.Series(0.0, index=calendario)
    for _, mov in movimientos.iterrows():
        fecha = mov["Fecha"].normalize()
        signo = 1 if mov["Tipo de movimiento"] == "Depósito" else -1
        aportes_cuotas.loc[fecha] += signo * mov["Valor total"] / precios.loc[fecha]
    valor_diario = aportes_cuotas.cumsum() * precios
    return valor_diario, float(valor_diario.loc[fin.normalize()])


def calcular_reporte(datos: pd.DataFrame, valor_portafolio: float):
    movimientos = datos[datos["Estado"].str.casefold() == "aprobado"].copy()
    movimientos = movimientos[movimientos["Tipo de movimiento"].isin(["Depósito", "Retiro"])].sort_values("Fecha")
    if movimientos.empty:
        raise ValueError("No se encontraron depósitos o retiros aprobados.")
    movimientos["Flujo XIRR"] = np.where(movimientos["Tipo de movimiento"] == "Depósito", -movimientos["Valor total"], movimientos["Valor total"])
    inicio, fin = movimientos["Fecha"].min(), movimientos["Fecha"].max()
    xirr = calcular_xirr(list(movimientos["Fecha"]) + [fin], list(movimientos["Flujo XIRR"]) + [valor_portafolio])
    xirr_mensual = (1 + xirr) ** (1 / 12) - 1 if pd.notna(xirr) and xirr > -1 else np.nan
    flujo_neto = np.where(movimientos["Tipo de movimiento"] == "Depósito", movimientos["Valor total"], -movimientos["Valor total"])
    movimientos["Flujo neto"] = flujo_neto
    movimientos["Periodo fecha"] = movimientos["Fecha"].dt.to_period("M")
    mensual = movimientos.groupby("Periodo fecha", as_index=False)["Flujo neto"].sum()
    mensual["Aportes acumulados"] = mensual["Flujo neto"].cumsum()
    mensual["Periodo"] = mensual["Periodo fecha"].astype(str)
    valor_simulado, temporal = [], 0.0
    for flujo in mensual["Flujo neto"]:
        temporal = (temporal + flujo) * (1 + xirr_mensual) if pd.notna(xirr_mensual) else temporal + flujo
        valor_simulado.append(temporal)
    mensual["Valor portafolio"] = valor_simulado
    mensual.loc[mensual.index[-1], "Valor portafolio"] = valor_portafolio

    xirr_colcap = np.nan
    advertencia = None
    try:
        valor_colcap_diario, vp_colcap = simular_colcap(movimientos, inicio, fin)
        mensual["Valor COLCAP"] = [
            valor_colcap_diario[(valor_colcap_diario.index.year == p.year) & (valor_colcap_diario.index.month == p.month)].iloc[-1]
            for p in mensual["Periodo fecha"]
        ]
        mensual.loc[mensual.index[-1], "Valor COLCAP"] = vp_colcap
        xirr_colcap = calcular_xirr(list(movimientos["Fecha"]) + [fin], list(movimientos["Flujo XIRR"]) + [vp_colcap])
        mensual["Exceso absoluto"] = mensual["Valor portafolio"] - mensual["Valor COLCAP"]
        mensual["Exceso relativo"] = mensual["Valor portafolio"] / mensual["Valor COLCAP"] - 1
    except Exception as e:
        advertencia = str(e)
        mensual["Valor COLCAP"] = np.nan

    metricas = {
        "depositos": movimientos.loc[movimientos["Tipo de movimiento"] == "Depósito", "Valor total"].sum(),
        "retiros": movimientos.loc[movimientos["Tipo de movimiento"] == "Retiro", "Valor total"].sum(),
        "capital_neto": movimientos["Flujo neto"].sum(),
        "rentabilidad_pesos": valor_portafolio - movimientos["Flujo neto"].sum(),
        "xirr": xirr, "xirr_mensual": xirr_mensual, "xirr_colcap": xirr_colcap,
    }
    return mensual, metricas, advertencia, len(movimientos)


def tabla_descargable(reporte: pd.DataFrame) -> bytes:
    salida = BytesIO()
    exportar = reporte.copy()
    exportar["Periodo"] = exportar["Periodo fecha"].astype(str)
    columnas = ["Periodo", "Aportes acumulados", "Valor portafolio", "Valor COLCAP", "Exceso absoluto", "Exceso relativo"]
    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        exportar.reindex(columns=columnas).to_excel(writer, index=False, sheet_name="Comparativa")
    return salida.getvalue()


def grafica_comparativa(reporte: pd.DataFrame, metricas: dict) -> go.Figure:
    """Gráfico interactivo y legible: evolución arriba y alpha mensual abajo."""
    tiene_colcap = reporte["Valor COLCAP"].notna().any()
    filas = 2 if tiene_colcap else 1
    figura = make_subplots(
        rows=filas, cols=1, shared_xaxes=True,
        row_heights=[0.74, 0.26] if tiene_colcap else [1], vertical_spacing=0.12,
        subplot_titles=("", "Exceso mensual frente a ICOLCAP") if tiene_colcap else ("",),
    )
    meses_es = [
        "ene.", "feb.", "mar.", "abr.", "may.", "jun.",
        "jul.", "ago.", "sept.", "oct.", "nov.", "dic."
    ]

    periodos = reporte["Periodo fecha"].apply(
        lambda fecha: f"{meses_es[fecha.month - 1]} {fecha.year}"
    )
    figura.add_trace(go.Bar(
        x=periodos, y=reporte["Aportes acumulados"], name="Aportes acumulados",
        marker_color="#4e79a7", opacity=0.72,
        hovertemplate="<b>%{x}</b><br>Aportes acumulados: COP $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    figura.add_trace(go.Scatter(
        x=periodos, y=reporte["Valor portafolio"], name="Valor portafolio", mode="lines+markers",
        line=dict(color="#ff6b6b", width=3), marker=dict(size=7, color="#ff6b6b"),
        hovertemplate="<b>%{x}</b><br>Valor portafolio: COP $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    if tiene_colcap:
        figura.add_trace(go.Scatter(
            x=periodos, y=reporte["Valor COLCAP"], name="ICOLCAP (mismos flujos)", mode="lines+markers",
            line=dict(color="#35c98b", width=2.5, dash="dash"), marker=dict(size=7, symbol="square", color="#35c98b"),
            hovertemplate="<b>%{x}</b><br>Valor ICOLCAP: COP $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)
        exceso = reporte["Exceso absoluto"].fillna(0)
        porcentajes = reporte["Exceso relativo"].replace([np.inf, -np.inf], np.nan).fillna(0)
        figura.add_trace(go.Bar(
            x=periodos, y=exceso, name="Exceso absoluto",
            marker_color=np.where(exceso >= 0, "#8b5cf6", "#ef5a67"), showlegend=False,
            customdata=np.c_[porcentajes],
            hovertemplate="<b>%{x}</b><br>Variación absoluta: COP $%{y:,.0f}<br>Variación porcentual: %{customdata[0]:+.2%}<extra></extra>",
        ), row=2, col=1)
        figura.add_hline(y=0, line_color="#73849d", line_width=1, row=2, col=1)
        final_exceso = exceso.iloc[-1]
        final_porcentaje = porcentajes.iloc[-1]
        # Se ven de entrada, como en el reporte original; el panel inferior conserva
        # además la escala visual del exceso. Plotly usa <br> para saltos de línea.
        fuente_alpha = 8 if len(reporte) <= 14 else 7
        valores_superiores = np.maximum(reporte["Valor portafolio"], reporte["Valor COLCAP"].fillna(0))
        for periodo, superior, diferencia, porcentaje_exceso in zip(periodos, valores_superiores, exceso, porcentajes):
            positivo = diferencia >= 0
            color = "#c4b5fd" if positivo else "#fda4af"
            borde = "#8b5cf6" if positivo else "#ef5a67"
            figura.add_annotation(
                x=periodo, y=superior, xref="x", yref="y", yshift=15,
                text=f"Δ {'+' if positivo else ''}{diferencia:,.0f}<br>({'+' if porcentaje_exceso >= 0 else ''}{porcentaje_exceso:.1%})",
                showarrow=False, align="center", font=dict(color=color, size=fuente_alpha),
                bgcolor="rgba(27,34,48,0.88)", bordercolor=borde, borderwidth=0.7, borderpad=1.5,
            )
        maximo_superior = valores_superiores.max()
        figura.update_yaxes(range=[0, maximo_superior * 1.18], row=1, col=1)
        figura.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.9, xanchor="center", yanchor="top",
            text=f"<b>Resultado final vs. ICOLCAP</b><br>COP ${final_exceso:,.0f} · {final_porcentaje:+.2%}",
            showarrow=False, align="right", font=dict(color="#dbeafe", size=12),
            bgcolor="rgba(35,48,70,0.92)", bordercolor="#48658b", borderwidth=1, borderpad=9,
        )
        tabla_metricas = (
            "<b>MÉTRICAS Y COMPARATIVA</b><br>"
            f"<span style='color:#fcd34d'>Profit absoluto</span>&nbsp;&nbsp;<b>{moneda(metricas['rentabilidad_pesos'])}</b><br>"
            f"<span style='color:#ff8a8a'>XIRR portafolio</span>&nbsp;&nbsp;<b>{porcentaje(metricas['xirr'])}</b><br>"
            f"XIRR mensual&nbsp;&nbsp;<b>{porcentaje(metricas['xirr_mensual'])}</b><br>"
            f"<span style='color:#55e6a3'>XIRR ICOLCAP</span>&nbsp;&nbsp;<b>{porcentaje(metricas['xirr_colcap'])}</b><br>"
            f"<span style='color:#c4b5fd'>Σ diferencias</span>&nbsp;&nbsp;<b>{moneda(exceso.sum())}</b><br>"
            f"μ excesos relativos&nbsp;&nbsp;<b>{porcentaje(porcentajes.mean())}</b>"
        )
        figura.add_annotation(
            xref="paper", yref="paper", x=0.012, y=0.93, xanchor="left", yanchor="top",
            text=tabla_metricas, showarrow=False, align="left", font=dict(color="#dce7f7", size=10),
            bgcolor="rgba(22,31,46,0.94)", bordercolor="#456182", borderwidth=1, borderpad=7,
        )
    figura.update_layout(
        title=dict(text="Desempeño del portafolio frente a ICOLCAP", x=0.5, xanchor="center", y=0.99,
                   font=dict(size=19, color="#f8fafc")),
        height=680 if tiene_colcap else 510, barmode="overlay", hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#202b3d", font=dict(color="#e5eefb", family="Inter, Arial, sans-serif"),
        margin=dict(l=20, r=25, t=145, b=25), legend=dict(orientation="h", yanchor="bottom", y=0.905, xanchor="center", x=0.5),
        hoverlabel=dict(bgcolor="#101827", bordercolor="#4e79a7", font_color="#f8fafc"),
    )
    figura.update_xaxes(showgrid=False, linecolor="#40516b", tickfont=dict(color="#bac8dc"))
    figura.update_yaxes(gridcolor="#34435a", zeroline=False, tickprefix="COP $", tickformat=",.0f", tickfont=dict(color="#bac8dc"))
    return figura


st.markdown("""<div class="hero"><h1>Analizador de Portafolio</h1><p>Evalúa rentabilidad, flujos y desempeño frente al ICOLCAP con una sola carga.</p></div>""", unsafe_allow_html=True)

st.subheader("1. Carga tus movimientos")
archivo = st.file_uploader("Arrastra el archivo CSV aquí o selecciónalo", type=["csv"],
                           help="Columnas requeridas: Fecha y hora, Tipo de movimiento, Estado y Valor total.")

if not archivo:
    st.info("Carga el archivo CSV para comenzar. Luego ingresa el valor actual del portafolio y presiona **Calcular análisis**.")
    st.stop()

archivo_actual = (archivo.name, archivo.size) if archivo else None
if st.session_state.get("archivo_actual") != archivo_actual:
    st.session_state.pop("resultado", None)
    st.session_state["archivo_actual"] = archivo_actual

st.success(f"Archivo cargado: {archivo.name}")
st.subheader("2. Indica el valor actual")
with st.form("parametros_analisis", border=False):
    valor_actual = st.number_input("Valor actual del portafolio", min_value=0.0, value=10_000_000.0,
                                   step=100_000.0, format="%.2f")
    st.caption("Este valor se usa como valor final para calcular la rentabilidad y el XIRR.")
    calcular = st.form_submit_button("3. Calcular análisis", type="primary")

if calcular:
    try:
        with st.spinner("Procesando movimientos y consultando el benchmark…"):
            datos = leer_csv(archivo)
            st.session_state.resultado = calcular_reporte(datos, valor_actual)
            st.session_state.valor_actual = valor_actual
    except Exception as e:
        st.error(f"No fue posible procesar el archivo: {e}")
        st.stop()

if "resultado" not in st.session_state:
    st.info("Completa el valor actual y presiona **Calcular análisis** para ver el reporte.")
    st.stop()

reporte, metricas, advertencia, n_movimientos = st.session_state.resultado
valor_actual = st.session_state.valor_actual
if advertencia:
    st.warning(f"El análisis principal se calculó, pero no fue posible obtener el benchmark ICOLCAP. Detalle: {advertencia}")

st.caption(f"{n_movimientos} movimientos aprobados incluidos · Valor actual analizado: {moneda(valor_actual)}")
st.markdown("<div class='section-label'>Flujos y valor actual</div>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    tarjeta_metricas("Depósitos aprobados", moneda(metricas["depositos"]), "Entradas de capital", "#60a5fa")
with c2:
    tarjeta_metricas("Retiros aprobados", moneda(metricas["retiros"]), "Salidas de capital", "#fb7185")
with c3:
    tarjeta_metricas("Capital neto aportado", moneda(metricas["capital_neto"]), f"Depósitos {moneda(metricas['depositos'])} − retiros {moneda(metricas['retiros'])}", "#c4b5fd")
with c4:
    tarjeta_metricas("Valor actual del portafolio", moneda(valor_actual), "Valor final informado", "#5eead4")

st.markdown("<div class='section-label'>Rentabilidad comparada</div>", unsafe_allow_html=True)
delta_xirr = metricas["xirr"] - metricas["xirr_colcap"] if pd.notna(metricas["xirr"]) and pd.notna(metricas["xirr_colcap"]) else np.nan
c1, c2, c3, c4 = st.columns(4)
with c1:
    tarjeta_metricas("Profit absoluto", moneda(metricas["rentabilidad_pesos"]), "Valor actual − capital neto aportado", "#fcd34d")
with c2:
    tarjeta_metricas("XIRR anual · Portafolio", porcentaje(metricas["xirr"]), "Rentabilidad anual de tu cartera", "#ff8a8a")
with c3:
    tarjeta_metricas("XIRR anual · ICOLCAP", porcentaje(metricas["xirr_colcap"]), "Benchmark con los mismos flujos", "#55e6a3")
with c4:
    tarjeta_metricas("Diferencia de XIRR", puntos_porcentuales(delta_xirr), "XIRR portafolio − XIRR ICOLCAP", "#c4b5fd")

if reporte["Valor COLCAP"].notna().any():
    st.markdown("<div class='section-label'>Exceso acumulado</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        tarjeta_metricas("Σ diferencias mensuales", moneda(reporte["Exceso absoluto"].sum()), "Suma de las variaciones frente al ICOLCAP", "#c4b5fd")
    with c2:
        tarjeta_metricas("μ de excesos relativos", porcentaje(reporte["Exceso relativo"].mean()), "Promedio mensual de la variación porcentual", "#c4b5fd")

st.subheader("Evolución y comparación")
st.caption("Pasa el cursor por cualquier periodo para ver aportes, portafolio, ICOLCAP y variación absoluta y porcentual.")
st.plotly_chart(grafica_comparativa(reporte, metricas), use_container_width=True, config={"displayModeBar": False})

st.subheader("Detalle mensual")
vista = reporte.copy()
vista["Periodo"] = vista["Periodo fecha"].astype(str)
columnas = ["Periodo", "Aportes acumulados", "Valor portafolio", "Valor COLCAP", "Exceso absoluto", "Exceso relativo"]
vista = vista.reindex(columns=columnas)
st.dataframe(vista, use_container_width=True, hide_index=True, column_config={
    "Aportes acumulados": st.column_config.NumberColumn(format="$%,.0f"),
    "Valor portafolio": st.column_config.NumberColumn(format="$%,.0f"),
    "Valor COLCAP": st.column_config.NumberColumn(format="$%,.0f"),
    "Exceso absoluto": st.column_config.NumberColumn(format="$%,.0f"),
    "Exceso relativo": st.column_config.NumberColumn(format="%.2f%%"),
})
st.download_button("Descargar comparativa en Excel", data=tabla_descargable(reporte), file_name="comparativa_portafolio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
