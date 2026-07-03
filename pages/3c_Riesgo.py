# =============================================================================
# QuantαfolyΩ — pages/3c_Riesgo.py
# Fase 3c: Métricas de Riesgo
# =============================================================================

import streamlit as st
import pandas as pd
from modulos.errores import interpretar_error
from config import get_colores, get_plotly_layout, get_plotly_config, get_legend_style

st.set_page_config(page_title="Riesgo · QuantαfolyΩ", page_icon="⚠️", layout="wide")
st.title("Métricas de Riesgo")

if st.session_state.get("retornos") is None:
    st.warning("Primero ejecuta la **📥 Fase 1 — Datos**.")
    st.stop()
if st.session_state.get("res_3a") is None:
    st.warning("Primero ejecuta la **📊 Fase 3a — Markowitz**.")
    st.stop()

retornos           = st.session_state["retornos"]
retornos_benchmark = st.session_state["retornos_benchmark"]
pesos_df           = st.session_state["pesos_df"]
rf_anual           = st.session_state["res_3a"]["rf_anual"]
res_capm           = st.session_state.get("res_capm")
nivel_asistente    = st.session_state.get("nivel_asistente", "basico")

st.markdown(
    f"**Portafolio:** {', '.join(retornos.columns.tolist())} &nbsp;|&nbsp; "
    f"**rf anual:** {rf_anual*100:.2f}%"
)
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("▶ Ejecutar Fase 3c", type="primary", width="stretch")

if ejecutar:
    from modulos.riesgo import pipeline_riesgo

    factor = st.session_state.get("metadatos", {}).get("factor_anualizacion", 12)
    with st.spinner("Calculando VaR, backtesting y métricas..."):
        resultado = pipeline_riesgo(
            retornos, retornos_benchmark,
            pesos_df=pesos_df,
            res_capm=res_capm,
            rf_anual=rf_anual,
            nivel_asistente=nivel_asistente,
            factor=factor,
        )

    with st.expander("📋 Log del proceso", expanded=resultado.get("error") is not None):
        for linea in resultado.get("logs", []):
            st.text(linea)

    if resultado.get("error"):
        st.error(interpretar_error(resultado["error"], fase="riesgo"))
        with st.expander("🔧 Detalle técnico (para soporte)"):
            st.code(str(resultado["error"]))
        st.stop()

    st.session_state["res_var"]        = resultado["res_var"]
    st.session_state["_resultados_var"] = resultado["resultados_var"]
    st.session_state["metricas_3c"]    = resultado["metricas_3c"]
    st.session_state["res_bt"]         = resultado["res_bt"]
    st.session_state["_resultados_bt"] = resultado["resultados_bt"]
    st.session_state["res_estres"]     = resultado["res_estres"]
    st.session_state["fig_riesgo"]     = resultado["fig_riesgo"]
    st.session_state["narrativa_3c"]   = resultado["narrativa_3c"]
    st.session_state["fase_completada"]["riesgo"] = True
    st.success("Fase 3c completada.")

# --- Resultados ---
if st.session_state.get("res_var") is not None:
    res_var      = st.session_state["res_var"]
    res_var_dict = st.session_state.get("_resultados_var", {})
    metricas_3c  = st.session_state["metricas_3c"]
    res_bt       = st.session_state["res_bt"]
    res_estres   = st.session_state["res_estres"]
    fig_riesgo   = st.session_state["fig_riesgo"]
    narrativa    = st.session_state.get("narrativa_3c", "")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Gráfico", "📉 VaR y CVaR", "🧪 Backtesting",
        "📋 Métricas", "🤖 Asistente"
    ])

    with tab1:
        _tema    = st.get_option("theme.base") or "light"
        _colores = get_colores(_tema)
        fig_riesgo.update_layout(
            legend=get_legend_style(_tema),
            **get_plotly_layout(_tema),
        )
        for trace in fig_riesgo.data:
            name     = getattr(trace, "name", "") or ""
            tipo     = trace.type
            name_low = name.lower()

            if tipo == "scatter":
                if "cvar" in name_low:
                    trace.update(line=dict(color=_colores["alerta"], width=2))
                elif "var" in name_low:
                    trace.update(line=dict(color=_colores["acento_principal"], width=2))
                elif "portafolio" in name_low or "tangente" in name_low:
                    trace.update(line=dict(color=_colores["acento_principal"], width=2))
                elif "s&p" in name_low or "benchmark" in name_low:
                    trace.update(line=dict(color=_colores["contraste"], width=1.5))
                else:
                    trace.update(line=dict(color=_colores["texto_secundario"], width=1.5))
            elif tipo in ("bar", "histogram"):
                if "cvar" in name_low:
                    trace.update(marker_color=_colores["alerta"])
                elif "var" in name_low:
                    trace.update(marker_color=_colores["acento_principal"])
                else:
                    # Para barras sin nombre claro — alternar entre acento y contraste
                    _idx = list(fig_riesgo.data).index(trace)
                    _pal = [_colores["acento_principal"], _colores["alerta"],
                            _colores["acento_secundario"], _colores["contraste"]]
                    trace.update(marker_color=_pal[_idx % len(_pal)])
        st.plotly_chart(fig_riesgo, width="stretch", config=get_plotly_config())

    with tab2:
        st.markdown("**VaR y CVaR por nivel de confianza y método**")
        st.dataframe(res_var.set_index('Nivel').round(4), width="stretch")

        # Métricas rápidas
        if res_var_dict:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("VaR Hist 95%",
                          f"{res_var_dict['95%']['var_hist']*100:.2f}%")
            with col2:
                st.metric("CVaR Hist 95%",
                          f"{res_var_dict['95%']['cvar_hist']*100:.2f}%")
            with col3:
                st.metric("VaR Hist 99%",
                          f"{res_var_dict['99%']['var_hist']*100:.2f}%")
            with col4:
                st.metric("CVaR Hist 99%",
                          f"{res_var_dict['99%']['cvar_hist']*100:.2f}%")

            # --- Interpretación inline: VaR y CVaR ---
            _var_95  = res_var_dict['95%']['var_hist']
            _cvar_95 = res_var_dict['95%']['cvar_hist']
            _var_99  = res_var_dict['99%']['var_hist']
            _diff_metodos = abs(_var_95 - res_var_dict['95%'].get('var_param', _var_95))

            if nivel_asistente == "basico":
                st.info(
                    f"En el 95% de los meses, el portafolio no pierde más del **{_var_95*100:.2f}%**. "
                    f"En el peor 5% de los meses, la pérdida promedio sería de **{_cvar_95*100:.2f}%** "
                    f"(CVaR). Eso es lo que puedes esperar en un mes muy malo."
                )
                if _cvar_95 > 0.10:
                    st.warning(
                        f"⚠️ Un CVaR de {_cvar_95*100:.1f}% mensual es elevado. "
                        "En los meses más adversos, el portafolio puede perder una parte significativa."
                    )
            else:
                st.info(
                    f"VaR histórico 95%: {_var_95*100:.2f}% | CVaR 95%: {_cvar_95*100:.2f}% | "
                    f"VaR 99%: {_var_99*100:.2f}% (mensual, portafolio tangente)."
                )
                if _diff_metodos > 0.01:
                    st.warning(
                        f"⚠️ VaR hist vs paramétrico difieren en {_diff_metodos*100:.2f}pp — "
                        "desviación de normalidad confirmada. VaR histórico es la métrica primaria."
                    )

    with tab3:
        st.markdown("**Backtesting del VaR (Kupiec + Christoffersen)**")
        cols_bt = ['Nivel', 'n', 'n_viol', 'p_hat', 'p_teorica',
                   'p_kupiec', 'kupiec_ok', 'p_christ', 'christ_ok']
        cols_disp = [c for c in cols_bt if c in res_bt.columns]
        st.dataframe(res_bt[cols_disp].set_index('Nivel'), width="stretch")

        # --- Interpretación inline: Backtesting ---
        _kupiec_ok_todos = bool(res_bt['kupiec_ok'].all()) if 'kupiec_ok' in res_bt.columns else True
        _kupiec_falla    = res_bt[res_bt['kupiec_ok'] == False]['Nivel'].tolist() if 'kupiec_ok' in res_bt.columns else []

        if nivel_asistente == "basico":
            if _kupiec_ok_todos:
                st.success(
                    "El modelo de riesgo estuvo bien calibrado — "
                    "las pérdidas extremas ocurrieron con la frecuencia esperada. "
                    "Puedes confiar en el VaR como referencia para este portafolio."
                )
            else:
                st.error(
                    f"❌ El VaR al nivel **{_kupiec_falla}** no estuvo bien calibrado — "
                    "hubo más pérdidas extremas de las esperadas. "
                    "El modelo subestimó el riesgo real en este período."
                )
        else:
            if _kupiec_ok_todos:
                st.success(
                    "Kupiec POF no rechaza H0 en ningún nivel de confianza. "
                    "Frecuencia de violaciones consistente con la probabilidad teórica."
                )
            else:
                st.error(
                    f"❌ Kupiec rechaza calibración del VaR al nivel **{_kupiec_falla}**. "
                    "Frecuencia empírica de violaciones incompatible con la teórica — "
                    "VaR histórico requiere recalibración o ventana más larga."
                )

        res_bt_dict = st.session_state.get("_resultados_bt", {})
        if res_bt_dict:
            bt_95 = res_bt_dict.get('95%', {})
            if bt_95.get('nota_christ'):
                st.info(f"ℹ️ Christoffersen 95%: {bt_95['nota_christ']}")
            if bt_95.get('nota_kupiec'):
                st.info(f"ℹ️ Kupiec 95%: {bt_95['nota_kupiec']}")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Métricas de desempeño (portafolio tangente)**")
            metricas_df = pd.DataFrame([metricas_3c]).T
            metricas_df.columns = ['Valor']
            st.dataframe(metricas_df.round(4), width="stretch")

            # --- Interpretación inline: Métricas de desempeño ---
            _sharpe  = metricas_3c.get('sharpe')
            _sortino = metricas_3c.get('sortino')
            _mdd     = metricas_3c.get('max_drawdown')

            if _sharpe is not None:
                if nivel_asistente == "basico":
                    if float(_sharpe) > 0.5:
                        st.success(
                            f"Sharpe de **{float(_sharpe):.2f}** — buena compensación por el riesgo asumido. "
                            "Por encima de 0.5 se considera un desempeño sólido."
                        )
                    elif float(_sharpe) > 0:
                        st.warning(
                            f"🟡 Sharpe de **{float(_sharpe):.2f}** — el portafolio genera retorno positivo "
                            "sobre la tasa libre, pero la compensación por riesgo es modesta."
                        )
                    else:
                        st.error(
                            f"❌ Sharpe negativo ({float(_sharpe):.2f}) — el portafolio no compensó "
                            "el riesgo asumido en este período."
                        )
                else:
                    if float(_sharpe) > 1.0:
                        st.success(f"Sharpe {float(_sharpe):.3f} — excepcional (> 1.0). "
                                   "Consistente con alpha significativo sobre el benchmark.")
                    elif float(_sharpe) > 0.5:
                        st.success(f"Sharpe {float(_sharpe):.3f} — sólido (0.5–1.0). "
                                   "Desempeño ajustado por riesgo por encima de la media histórica del mercado.")
                    elif float(_sharpe) > 0:
                        st.warning(f"🟡 Sharpe {float(_sharpe):.3f} — positivo pero moderado. "
                                   "Retorno sobre rf existe pero con baja eficiencia relativa.")
                    else:
                        st.error(f"❌ Sharpe {float(_sharpe):.3f} — negativo. "
                                 "El portafolio no compensó el riesgo en este período muestral.")

        with col2:
            if not res_estres.empty:
                st.markdown("**Análisis de estrés histórico**")
                st.dataframe(res_estres.set_index('Escenario').round(4),
                             width="stretch")

                # --- Interpretación inline: Estrés ---
                _peor_escenario = res_estres.loc[res_estres['Ret_acum'].idxmin()]
                _peor_nombre    = _peor_escenario['Escenario']
                _peor_ret       = float(_peor_escenario['Ret_acum'])
                if nivel_asistente == "basico":
                    st.info(
                        f"El peor escenario histórico fue **{_peor_nombre}** con una caída acumulada "
                        f"de **{_peor_ret*100:.1f}%**. Así se habría comportado este portafolio en esa crisis."
                    )
                else:
                    _peor_dd = float(_peor_escenario['Max_DD'])
                    st.info(
                        f"Escenario más adverso: **{_peor_nombre}** — "
                        f"Ret. acum. {_peor_ret*100:.2f}% | Max DD {_peor_dd*100:.2f}%. "
                        "Análisis ex-post con pesos actuales aplicados al período histórico."
                    )
            else:
                st.info("Sin datos de estrés disponibles "
                        "(el período analizado no cubre las crisis históricas).")

    with tab5:
        if narrativa:
            st.markdown(narrativa)

    st.divider()
    st.info("Riesgo listo. Continúa con **4 Verificación** en el menú lateral.")
