# =============================================================================
# QuantαfolyΩ — pages/3a_Markowitz.py
# Fase 3a: Optimización de Portafolio (Markowitz)
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from modulos.errores import interpretar_error
from config import get_colores, get_plotly_layout

st.set_page_config(page_title="Markowitz · QuantαfolyΩ", page_icon="📊", layout="wide")
st.title("📊 Fase 3a — Optimización de Portafolio (Markowitz)")

if st.session_state.get("retornos") is None:
    st.warning("Primero ejecuta la **📥 Fase 1 — Datos**.")
    st.stop()

retornos           = st.session_state["retornos"]
retornos_benchmark = st.session_state["retornos_benchmark"]
nivel_asistente    = st.session_state.get("nivel_asistente", "basico")

st.markdown(
    f"**Activos:** {', '.join(retornos.columns.tolist())} &nbsp;|&nbsp; "
    f"**Observaciones:** {len(retornos)}"
)
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("▶ Ejecutar Fase 3a", type="primary", width="stretch")

if ejecutar:
    from modulos.markowitz import pipeline_markowitz

    res_normalidad = st.session_state.get("res_norm")
    res_arch       = st.session_state.get("res_arch")
    res_mardia     = st.session_state.get("res_mardia")

    factor = st.session_state.get("metadatos", {}).get("factor_anualizacion", 12)

    with st.spinner("Optimizando portafolios..."):
        resultado = pipeline_markowitz(
            retornos, retornos_benchmark,
            nivel_asistente=nivel_asistente,
            res_normalidad=res_normalidad,
            res_arch=res_arch,
            res_mardia=res_mardia,
            factor=factor,
        )

    with st.expander("📋 Log del proceso", expanded=resultado.get("error") is not None):
        for linea in resultado.get("logs", []):
            st.text(linea)

    if resultado.get("error"):
        st.error(interpretar_error(resultado["error"], fase="markowitz"))
        with st.expander("🔧 Detalle técnico (para soporte)"):
            st.code(str(resultado["error"]))
        st.stop()

    st.session_state["mu"]           = resultado["mu"]
    st.session_state["S"]            = resultado["S"]
    st.session_state["frontera"]     = resultado["frontera"]
    st.session_state["mvp"]          = resultado["mvp"]
    st.session_state["tangente"]     = resultado["tangente"]
    st.session_state["equal_weight"] = resultado["equal_weight"]
    st.session_state["metricas_3a"]  = resultado["metricas_3a"]
    st.session_state["pesos_df"]     = resultado["pesos_df"]
    st.session_state["res_3a"]       = resultado["res_3a"]
    st.session_state["fig_frontera"] = resultado["fig_frontera"]
    st.session_state["narrativa_3a"] = resultado["narrativa_3a"]
    st.session_state["fase_completada"]["markowitz"] = True
    st.success("✅ Fase 3a completada.")

# --- Mostrar resultados ---
if st.session_state.get("frontera") is not None:
    fig_frontera = st.session_state["fig_frontera"]
    metricas_3a  = st.session_state["metricas_3a"]
    pesos_df     = st.session_state["pesos_df"]
    res_3a       = st.session_state["res_3a"]
    narrativa    = st.session_state.get("narrativa_3a", "")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Frontera eficiente", "📋 Métricas", "⚖️ Pesos", "🤖 Asistente"
    ])

    with tab1:
        _tema    = st.get_option("theme.base") or "light"
        _colores = get_colores(_tema)
        fig_frontera.update_layout(**get_plotly_layout(_tema))

        # CORRECCIÓN (auditoría P1.2→P3.1): los nombres reales de las trazas
        # incluyen el Sharpe (ej. "Tangente (Sharpe=0.842)"), así que el selector
        # exacto de antes nunca encontraba nada y este bloque no hacía nada.
        # Se recolorea por prefijo y se cubren todas las trazas relevantes.
        for _trace in fig_frontera.data:
            _nombre = getattr(_trace, "name", "") or ""
            if _nombre.startswith("Tangente"):
                _trace.update(marker=dict(color=_colores["acento_principal"],
                                           size=_trace.marker.size))
            elif _nombre.startswith("MVP"):
                _trace.update(marker=dict(color=_colores["acento_secundario"],
                                           size=_trace.marker.size))
            elif _nombre.startswith("1/N"):
                _trace.update(marker=dict(color=_colores["texto_secundario"],
                                           size=_trace.marker.size))
            elif _nombre == "Activos individuales":
                _trace.update(marker=dict(color=_colores["highlight"]))
            elif _nombre == "Capital Market Line (CML)":
                _trace.update(line=dict(color=_colores["acento_secundario"],
                                         dash="dash", width=1.5))
            elif _nombre.startswith("Rf ="):
                _trace.update(marker=dict(color=_colores["texto_principal"]))
            # "Frontera eficiente" usa escala continua por Sharpe — se deja intacta.
        st.plotly_chart(fig_frontera, width="stretch")

        col1, col2, col3 = st.columns(3)
        mvp      = st.session_state["mvp"]
        tangente = st.session_state["tangente"]
        ew       = st.session_state["equal_weight"]
        rf       = res_3a.get("rf_anual", 0)
        with col1:
            st.metric("MVP — Volatilidad", f"{mvp['volatilidad']*100:.2f}%")
            st.metric("MVP — Sharpe", f"{mvp['sharpe']:.3f}")
        with col2:
            st.metric("Tangente — Retorno", f"{tangente['retorno']*100:.2f}%")
            st.metric("Tangente — Sharpe", f"{tangente['sharpe']:.3f}")
        with col3:
            st.metric("Tasa libre de riesgo (rf)", f"{rf*100:.2f}%")
            cond = res_3a.get("num_condicion_sigma", 0)
            st.metric("N° condición Σ", f"{cond:.1f}",
                      delta="OK" if cond < 1000 else "ALTA",
                      delta_color="normal" if cond < 1000 else "inverse")

        # --- Interpretación inline: Frontera eficiente ---
        _sharpe_tang  = tangente["sharpe"]
        _sharpe_bench = metricas_3a[metricas_3a["nombre"] == "S&P 500 (benchmark)"]["sharpe"].values
        _sharpe_bench = float(_sharpe_bench[0]) if len(_sharpe_bench) > 0 else 0.0
        _sharpe_ew    = ew["sharpe"]

        if nivel_asistente == "basico":
            if _sharpe_tang > _sharpe_bench:
                st.success(
                    f"✅ El portafolio tangente (Sharpe {_sharpe_tang:.2f}) supera al S&P 500 "
                    f"(Sharpe {_sharpe_bench:.2f}) — obtienes más retorno por unidad de riesgo "
                    "que simplemente comprar el índice."
                )
            else:
                st.warning(
                    f"⚠️ El portafolio tangente (Sharpe {_sharpe_tang:.2f}) no supera al S&P 500 "
                    f"(Sharpe {_sharpe_bench:.2f}) en este período. "
                    "Considera revisar la selección de activos."
                )
        else:
            if _sharpe_tang > _sharpe_bench:
                st.success(
                    f"✅ Sharpe tangente {_sharpe_tang:.3f} > benchmark {_sharpe_bench:.3f} "
                    f"(+{_sharpe_tang - _sharpe_bench:.3f}). "
                    "La optimización media-varianza genera alpha positivo sobre el índice en este período."
                )
            else:
                st.warning(
                    f"⚠️ Sharpe tangente {_sharpe_tang:.3f} ≤ benchmark {_sharpe_bench:.3f}. "
                    "Error de estimación de Michaud (1989): μ muestral tiene alta varianza — "
                    "la optimización puede amplificar ruido estadístico."
                )

    with tab2:
        cols_mostrar = ['nombre', 'retorno_anual', 'volatilidad',
                        'sharpe', 'sortino', 'max_drawdown', 'calmar', 'omega', 'concentracion']
        cols_disp = [c for c in cols_mostrar if c in metricas_3a.columns]
        _df_met = metricas_3a[cols_disp].copy()
        # Reemplazar strings no numéricos ('—', 'N/A', etc.) por NaN en columnas numéricas
        for _c in cols_disp:
            if _c != 'nombre':
                _df_met[_c] = pd.to_numeric(_df_met[_c], errors='coerce')
        st.dataframe(
            _df_met.set_index('nombre').round(4),
            width="stretch",
        )
        st.caption("Todas las métricas están anualizadas.")

        # --- Interpretación inline: Métricas ---
        _sharpe_tang = tangente["sharpe"]
        _sharpe_ew   = ew["sharpe"]
        _mdd_row     = metricas_3a[metricas_3a["nombre"] == "Tangente"]["max_drawdown"].values
        _mdd         = float(_mdd_row[0]) if len(_mdd_row) > 0 else None

        if nivel_asistente == "basico":
            if _sharpe_tang < _sharpe_ew:
                st.warning(
                    f"⚠️ El portafolio igual-peso (1/N) supera al Tangente en Sharpe "
                    f"({_sharpe_ew:.2f} vs {_sharpe_tang:.2f}). "
                    "Con muestras históricas cortas, diversificar en partes iguales puede ser más robusto "
                    "que la optimización."
                )
            else:
                st.success(
                    f"✅ El portafolio Tangente supera al igual-peso en Sharpe "
                    f"({_sharpe_tang:.2f} vs {_sharpe_ew:.2f}). "
                    "La optimización agrega valor real sobre una distribución simple."
                )
            if _mdd is not None and abs(_mdd) > 0.30:
                st.warning(
                    f"⚠️ El portafolio tangente tuvo una caída máxima de **{abs(_mdd)*100:.1f}%** "
                    "en el período analizado. Evalúa si esa pérdida potencial es tolerable."
                )
        else:
            if _sharpe_tang < _sharpe_ew:
                st.warning(
                    f"⚠️ 1/N supera al Tangente en Sharpe ({_sharpe_ew:.3f} vs {_sharpe_tang:.3f}). "
                    "Error de estimación de Michaud (1989): la media muestral como estimador de μ "
                    "tiene alta varianza — la optimización amplifica ruido. "
                    "Alternativas: MVP (no requiere μ), Black-Litterman, Robust Optimization."
                )
            else:
                st.success(
                    f"✅ Tangente supera 1/N en Sharpe ({_sharpe_tang:.3f} vs {_sharpe_ew:.3f}). "
                    "La estructura de covarianza estimada aporta valor al proceso de optimización."
                )

    with tab3:
        st.dataframe(pesos_df.round(4), width="stretch")

        fig_pesos = go.Figure()
        _seq = _colores["graf_seq"]
        for i, ticker in enumerate(pesos_df.index):
            fig_pesos.add_trace(go.Bar(
                name=ticker,
                x=pesos_df.columns.tolist(),
                y=(pesos_df.loc[ticker] * 100).tolist(),
                marker_color=_seq[i % len(_seq)],
            ))
        fig_pesos.update_layout(
            barmode='stack', height=350,
            yaxis_title='Peso (%)', xaxis_title='Portafolio',
            legend=dict(orientation="h", y=-0.2, x=0),
            margin=dict(l=40, r=20, t=20, b=60),
            **get_plotly_layout(_tema),
        )
        st.plotly_chart(fig_pesos, width="stretch")

        # --- Interpretación inline: Concentración de pesos ---
        _pesos_tang = pesos_df["Tangente"]
        _activo_max = _pesos_tang.idxmax()
        _peso_max   = float(_pesos_tang.max())
        _n_activos  = int((_pesos_tang > 0.05).sum())  # activos con peso > 5%

        if nivel_asistente == "basico":
            if _peso_max > 0.50:
                st.warning(
                    f"⚠️ **{_activo_max}** concentra el **{_peso_max*100:.1f}%** del portafolio tangente. "
                    "Una concentración tan alta aumenta el riesgo específico — "
                    "si ese activo cae fuerte, el portafolio lo siente directamente."
                )
            else:
                st.success(
                    f"✅ El portafolio está bien distribuido — ningún activo supera el 50%. "
                    f"El mayor peso es **{_activo_max}** con {_peso_max*100:.1f}%. "
                    f"{_n_activos} activos tienen participación significativa (>5%)."
                )
        else:
            if _peso_max > 0.50:
                st.warning(
                    f"⚠️ Concentración elevada: **{_activo_max}** = {_peso_max*100:.1f}% en el Tangente. "
                    f"HHI normalizado elevado — diversificación efectiva reducida. "
                    "Considerar restricción de peso máximo (PESO_MAXIMO_ACTIVO en config.py)."
                )
            else:
                st.success(
                    f"✅ Concentración moderada: máximo {_peso_max*100:.1f}% en {_activo_max}. "
                    f"{_n_activos} activos con peso > 5% — diversificación efectiva razonable."
                )

    with tab4:
        if narrativa:
            st.markdown(narrativa)

    st.divider()
    st.info("✅ Markowitz listo. Continúa con **📐 3b Factores** en el menú lateral.")
