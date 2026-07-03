# =============================================================================
# QuantαfolyΩ — pages/1_Datos.py
# Fase 1: Módulo de Datos
# =============================================================================

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import UMBRAL_DATOS_FALTANTES, get_colores, get_plotly_layout, get_plotly_config, get_legend_style
from modulos.errores import interpretar_error

st.set_page_config(page_title="Datos · QuantαfolyΩ", page_icon="📥", layout="wide")
st.title("Módulo de Datos")

# --- Verificar que el usuario configuró el portafolio ---
if "tickers_usuario" not in st.session_state or st.session_state["tickers_usuario"] is None:
    st.warning(
        "Configura el portafolio en el panel lateral de la **página de inicio** "
        "y presiona **Analizar portafolio**."
    )
    st.stop()

tickers          = st.session_state["tickers_usuario"]
ventana_anos     = st.session_state.get("ventana_anos", 5)
frecuencia       = st.session_state.get("frecuencia", "1mo")
nivel_asistente  = st.session_state.get("nivel_asistente", "basico")

st.markdown(
    f"**Portafolio:** {', '.join(tickers)} &nbsp;|&nbsp; "
    f"**Ventana:** {ventana_anos} años &nbsp;|&nbsp; "
    f"**Frecuencia:** {frecuencia}"
)
st.divider()

# --- Botón de ejecución ---
col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("▶ Ejecutar Fase 1", type="primary", width="stretch")

if ejecutar:
    from modulos.datos import pipeline_datos

    with st.spinner("Descargando y procesando datos..."):
        resultado = pipeline_datos(tickers, ventana_anos, frecuencia)

    # Mostrar logs siempre
    with st.expander("📋 Log del proceso", expanded=resultado.get("error") is not None):
        for linea in resultado.get("logs", []):
            st.text(linea)

    if resultado.get("error"):
        st.error(interpretar_error(resultado["error"], fase="datos"))
        with st.expander("🔧 Detalle técnico (para soporte)"):
            st.code(str(resultado["error"]))
        st.stop()

    # Guardar en session_state
    st.session_state["retornos"]             = resultado["retornos"]
    st.session_state["retornos_benchmark"]   = resultado["retornos_benchmark"]
    st.session_state["precios_usd"]          = resultado["precios_usd"]
    st.session_state["metadatos"]            = resultado["metadatos"]
    st.session_state["resultado_validacion"] = resultado["resultado_validacion"]
    st.session_state["divisas_activos"]      = resultado["divisas_activos"]
    st.session_state["fase_completada"]["datos"] = True

    st.success(
        f"Fase 1 completada — "
        f"{resultado['retornos'].shape[0]} periodos × "
        f"{resultado['retornos'].shape[1]} activos"
    )

    # Advertencias explícitas sobre activos eliminados o con datos parciales
    meta = resultado.get("metadatos", {})
    inv  = meta.get("tickers_invalidos", [])
    eli  = meta.get("activos_eliminados", [])
    activos_ingresados = len(tickers)
    activos_finales    = resultado['retornos'].shape[1]

    if inv:
        st.error(f"❌ Tickers no encontrados en Yahoo Finance: **{inv}**  \n"
                 "Verifica el formato (ej: AAPL para EE.UU., SAP.DE para Alemania, 7203.T para Japón).")
    if eli:
        st.warning(f"⚠️ Activos eliminados por datos insuficientes (>{int(UMBRAL_DATOS_FALTANTES*100)}% faltante): **{eli}**")
    if activos_finales < activos_ingresados - len(inv):
        st.warning(
            f"⚠️ El portafolio final tiene **{activos_finales} activos** "
            f"(ingresaste {activos_ingresados - len(inv)} válidos). "
            "Algunos activos pueden tener historia más corta que la ventana solicitada."
        )
    if not resultado["datos_suficientes"]:
        st.warning(
            "⚠️ Algunas observaciones están por debajo del mínimo recomendado para ciertos modelos. "
            "Considera ampliar la ventana temporal."
        )

# --- Mostrar resultados si ya están en session_state ---
if st.session_state.get("retornos") is not None:
    retornos   = st.session_state["retornos"]
    benchmark  = st.session_state["retornos_benchmark"]
    metadatos  = st.session_state.get("metadatos", {})
    divisas    = st.session_state.get("divisas_activos", {})

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Retornos", "📊 Estadísticas", "💱 Divisas", "🗂️ Metadata"]
    )

    with tab1:
        # Gráfico de retornos acumulados
        retornos_acum = np.exp(retornos.cumsum()) - 1
        _tema    = st.get_option("theme.base") or "light"
        _colores = get_colores(_tema)
        _seq     = _colores["graf_seq"]

        fig = go.Figure()
        for i, col in enumerate(retornos_acum.columns):
            fig.add_trace(go.Scatter(
                x=retornos_acum.index, y=retornos_acum[col],
                name=col, mode='lines',
                line=dict(color=_seq[i % len(_seq)], width=2),
            ))
        bench_acum = np.exp(benchmark.cumsum()) - 1
        fig.add_trace(go.Scatter(
            x=bench_acum.index, y=bench_acum["SP500"],
            name="S&P 500", mode='lines',
            line=dict(color=_colores["contraste"], dash='dash', width=1.5),
        ))
        fig.update_layout(
            title="Retornos acumulados (base 0)",
            yaxis_tickformat=".0%",
            height=420,
            legend=get_legend_style(_tema),
            margin=dict(l=40, r=20, t=40, b=70),
            **get_plotly_layout(_tema),
        )
        st.plotly_chart(fig, width="stretch", config=get_plotly_config())

        # --- Interpretación inline: Retornos acumulados ---
        _ret_final     = retornos_acum.iloc[-1]
        _mejor_activo  = _ret_final.idxmax()
        _peor_activo   = _ret_final.idxmin()
        _ret_bench_fin = float(bench_acum["SP500"].iloc[-1])
        _activos_sup   = [t for t in _ret_final.index if _ret_final[t] > _ret_bench_fin]

        if nivel_asistente == "basico":
            if _activos_sup:
                st.info(
                    f"**{_mejor_activo}** fue el activo con mayor retorno acumulado en el período. "
                    f"{len(_activos_sup)} de {len(_ret_final)} activos superaron al S&P 500. "
                    "Esto es contexto histórico — no garantiza el mismo resultado futuro."
                )
            else:
                st.warning(
                    f"Ningún activo superó al S&P 500 en retorno acumulado en este período. "
                    f"El mejor fue **{_mejor_activo}**, el de menor retorno fue **{_peor_activo}**."
                )
        else:
            st.info(
                f"Mejor retorno acumulado: **{_mejor_activo}** ({_ret_final[_mejor_activo]*100:.1f}%) | "
                f"Peor: **{_peor_activo}** ({_ret_final[_peor_activo]*100:.1f}%) | "
                f"S&P 500: {_ret_bench_fin*100:.1f}% | "
                f"{len(_activos_sup)}/{len(_ret_final)} activos superan al benchmark."
            )
        st.caption("Tabla de retornos logarítmicos mensuales")
        st.dataframe(retornos.round(4), width="stretch")

    with tab2:
        desc = retornos.describe().T
        desc.columns = ["N", "Media", "Std", "Mín", "Q25", "Mediana", "Q75", "Máx"]
        st.dataframe(desc.round(4), width="stretch")

        # --- Interpretación inline: Estadísticas descriptivas ---
        _activo_mas_vol   = desc["Std"].idxmax()
        _activo_menos_vol = desc["Std"].idxmin()
        _vol_max = float(desc["Std"].max())
        _vol_min = float(desc["Std"].min())

        if nivel_asistente == "basico":
            st.info(
                f"**{_activo_mas_vol}** es el activo más volátil (desviación estándar mensual: "
                f"{_vol_max*100:.2f}%). "
                f"**{_activo_menos_vol}** es el más estable ({_vol_min*100:.2f}%). "
                "Mayor volatilidad significa mayor incertidumbre en los retornos mensuales."
            )
        else:
            st.info(
                f"σ mensual máxima: **{_activo_mas_vol}** ({_vol_max*100:.2f}%) | "
                f"σ mínima: **{_activo_menos_vol}** ({_vol_min*100:.2f}%). "
                f"Ratio max/min: {_vol_max/_vol_min:.2f}x — "
                + ("dispersión alta entre activos." if _vol_max/_vol_min > 2 else "dispersión moderada entre activos.")
            )

    with tab3:
        if divisas:
            rows = []
            for ticker, divisa in divisas.items():
                rows.append({
                    "Ticker": ticker,
                    "Divisa original": divisa,
                    "Conversión": "No requerida" if divisa == "USD" else f"{divisa} → USD",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.info("Sin información de divisas disponible.")

    with tab4:
        if metadatos:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Activos", metadatos.get("n_activos", "-"))
                st.metric("Observaciones", metadatos.get("n_observaciones", "-"))
            with col2:
                st.metric("Fecha inicio", metadatos.get("fecha_inicio", "-"))
                st.metric("Fecha fin", metadatos.get("fecha_fin", "-"))
            with col3:
                st.metric("Frecuencia", metadatos.get("frecuencia", "-"))
                st.metric("Divisa base", metadatos.get("divisa_base", "-"))

            inv = metadatos.get("tickers_invalidos", [])
            eli = metadatos.get("activos_eliminados", [])
            if inv:
                st.warning(f"Tickers inválidos descartados: {inv}")
            if eli:
                st.warning(f"Activos eliminados por datos faltantes: {eli}")
        else:
            st.info("Sin metadata disponible.")

    st.divider()
    st.info("Datos listos. Continúa con **📐 2 Estadístico** en el menú lateral.")
