# =============================================================================
# QuantαfolyΩ — pages/3b_Factores.py
# Fase 3b: Modelos de Factores
# =============================================================================

import streamlit as st
from modulos.errores import interpretar_error
from config import get_colores, get_plotly_layout

st.set_page_config(page_title="Factores · QuantαfolyΩ", page_icon="📐", layout="wide")
st.title("📐 Fase 3b — Modelos de Factores")

if st.session_state.get("retornos") is None:
    st.warning("Primero ejecuta la **📥 Fase 1 — Datos**.")
    st.stop()
if st.session_state.get("res_3a") is None:
    st.warning("Primero ejecuta la **📊 Fase 3a — Markowitz**.")
    st.stop()

retornos           = st.session_state["retornos"]
retornos_benchmark = st.session_state["retornos_benchmark"]
rf_anual           = st.session_state["res_3a"]["rf_anual"]
frecuencia         = st.session_state.get("frecuencia", "1mo")
nivel_asistente    = st.session_state.get("nivel_asistente", "basico")

st.markdown(
    f"**Activos:** {', '.join(retornos.columns.tolist())} &nbsp;|&nbsp; "
    f"**rf anual:** {rf_anual*100:.2f}%"
)
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("▶ Ejecutar Fase 3b", type="primary", width="stretch")

if ejecutar:
    from modulos.factores import pipeline_factores

    factor = st.session_state.get("metadatos", {}).get("factor_anualizacion", 12)
    with st.spinner("Estimando CAPM, FF3 y APT... (puede tardar ~1 minuto)"):
        resultado = pipeline_factores(
            retornos, retornos_benchmark,
            rf_anual=rf_anual,
            frecuencia=frecuencia,
            nivel_asistente=nivel_asistente,
            factor=factor,
        )

    with st.expander("📋 Log del proceso", expanded=resultado.get("error") is not None):
        for linea in resultado.get("logs", []):
            st.text(linea)

    if resultado.get("error"):
        st.error(interpretar_error(resultado["error"], fase="factores"))
        with st.expander("🔧 Detalle técnico (para soporte)"):
            st.code(str(resultado["error"]))
        st.stop()

    st.session_state["res_capm"]     = resultado["res_capm"]
    st.session_state["res_ff3"]      = resultado["res_ff3"]
    st.session_state["res_apt"]      = resultado["res_apt"]
    st.session_state["tabla_comp"]   = resultado["tabla_comp"]
    st.session_state["loadings_pca"] = resultado["loadings_pca"]
    st.session_state["fig_betas"]    = resultado["fig_betas"]
    st.session_state["narrativa_3b"] = resultado["narrativa_3b"]
    st.session_state["fase_completada"]["factores"] = True
    st.success("✅ Fase 3b completada.")

# --- Resultados ---
if st.session_state.get("res_capm") is not None:
    res_capm     = st.session_state["res_capm"]
    res_ff3      = st.session_state["res_ff3"]
    res_apt      = st.session_state["res_apt"]
    tabla_comp   = st.session_state["tabla_comp"]
    loadings_pca = st.session_state["loadings_pca"]
    fig_betas    = st.session_state["fig_betas"]
    narrativa    = st.session_state.get("narrativa_3b", "")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Gráfico", "📋 CAPM", "📋 FF3", "📋 APT", "🔄 Comparativa", "🤖 Asistente"
    ])

    with tab1:
        _tema    = st.get_option("theme.base") or "light"
        _colores = get_colores(_tema)
        fig_betas.update_layout(**get_plotly_layout(_tema))
        _seq = _colores["graf_seq"]
        for i, trace in enumerate(fig_betas.data):
            trace.update(marker=dict(color=_seq[i % len(_seq)]))
        st.plotly_chart(fig_betas, width="stretch")

    with tab2:
        st.markdown("**CAPM — Sharpe (1964), Lintner (1965)**")
        cols_capm = ['Ticker', 'Alpha_anual', 'Beta', 'R2', 'p_alpha', 'p_beta',
                     'Perfil_beta', 'BLUE', 'Semaforo']
        cols_disp = [c for c in cols_capm if c in res_capm.columns]
        st.dataframe(res_capm[cols_disp].set_index('Ticker'), width="stretch")

        # --- Interpretación inline: CAPM ---
        _betas        = res_capm['Beta'].values
        _beta_port    = float((st.session_state.get('pesos_df', {}).get('Tangente', 1) * res_capm.set_index('Ticker')['Beta']).sum()) if st.session_state.get('pesos_df') is not None else float(_betas.mean())
        _alpha_sig    = res_capm[res_capm['Alpha_sig'] == True]['Ticker'].tolist() if 'Alpha_sig' in res_capm.columns else []
        _blue_falla   = res_capm[res_capm['BLUE'] == 'VIOLA SUPUESTOS']['Ticker'].tolist() if 'BLUE' in res_capm.columns else []

        if nivel_asistente == "basico":
            if _beta_port > 1.1:
                st.warning(
                    f"⚠️ El portafolio tiene una beta promedio de **{_beta_port:.2f}** — "
                    "amplifica los movimientos del mercado. En una caída del 10% del S&P 500, "
                    f"este portafolio caería aproximadamente {_beta_port*10:.1f}%."
                )
            elif _beta_port < 0.9:
                st.info(
                    f"🛡️ El portafolio tiene una beta promedio de **{_beta_port:.2f}** — "
                    "más defensivo que el mercado. Sube menos en los rallies, pero cae menos en las crisis."
                )
            else:
                st.info(
                    f"↔️ Beta del portafolio: **{_beta_port:.2f}** — sigue al mercado de cerca. "
                    "El comportamiento esperado es similar al S&P 500."
                )
            if _alpha_sig:
                st.success(f"✅ **{_alpha_sig}** tienen alpha significativo — "
                           "generan retorno más allá de lo que explica el riesgo de mercado.")
        else:
            st.info(
                f"Beta del portafolio tangente (ponderado): **{_beta_port:.3f}**. "
                + ("Beta > 1: portafolio agresivo." if _beta_port > 1 else "Beta < 1: portafolio defensivo.")
            )
            if _alpha_sig:
                st.success(f"✅ Alpha de Jensen significativo en **{_alpha_sig}** (p < 0.05). "
                           "Retorno anormal positivo no explicado por el factor de mercado.")
            if _blue_falla:
                st.warning(f"⚠️ **{_blue_falla}**: residuos CAPM violan supuestos Gauss-Markov. "
                           "Errores estándar sesgados — inferencia sobre alpha/beta con precaución.")

    with tab3:
        if res_ff3.empty:
            st.warning("FF3 no disponible (problema de descarga).")
        else:
            st.markdown("**Fama-French 3 Factores — Fama & French (1993)**")
            st.dataframe(res_ff3.set_index('Ticker'), width="stretch")

            # --- Interpretación inline: FF3 vs CAPM ---
            _r2_capm_vals = res_capm.set_index('Ticker')['R2'] if 'R2' in res_capm.columns else None
            _r2_ff3_vals  = res_ff3.set_index('Ticker')['R2']  if 'R2' in res_ff3.columns else None
            if _r2_capm_vals is not None and _r2_ff3_vals is not None:
                _mejora_media = float((_r2_ff3_vals - _r2_capm_vals.reindex(_r2_ff3_vals.index)).mean())
                if nivel_asistente == "basico":
                    if _mejora_media > 0.05:
                        st.info(
                            f"Agregar tamaño (SMB) y valor (HML) al modelo mejora la explicación "
                            f"del comportamiento de los activos en **{_mejora_media*100:.1f}pp** en promedio. "
                            "El mercado no es el único factor que importa."
                        )
                    else:
                        st.info(
                            "El modelo de 3 factores mejora marginalmente al CAPM en este portafolio. "
                            "El factor de mercado captura la mayor parte del comportamiento."
                        )
                else:
                    st.info(
                        f"Mejora media de R² CAPM → FF3: **+{_mejora_media*100:.2f}pp**. "
                        + ("Factores SMB/HML aportan poder explicativo relevante." if _mejora_media > 0.05
                           else "Factores SMB/HML marginales — modelo unifactorial adecuado para este universo.")
                    )

    with tab4:
        if res_apt.empty:
            st.warning("APT no disponible.")
        else:
            st.markdown("**APT con factores macro — Ross (1976), Chen et al. (1986)**")
            st.dataframe(res_apt.set_index('Ticker'), width="stretch")
            if loadings_pca is not None:
                with st.expander("Cargas factoriales PCA"):
                    st.dataframe(loadings_pca.round(3), width="stretch")

    with tab5:
        st.markdown("**Comparativa α / R² por modelo**")
        st.dataframe(tabla_comp, width="stretch")

        # --- Interpretación inline: Comparativa de modelos ---
        if not tabla_comp.empty and 'R2_CAPM' in tabla_comp.columns:
            _col_r2 = [c for c in tabla_comp.columns if 'R2' in c or 'r2' in c.lower()]
            if len(_col_r2) >= 2:
                _mejor_modelo = tabla_comp[_col_r2].mean().idxmax()
                if nivel_asistente == "basico":
                    st.info(
                        f"El modelo **{_mejor_modelo}** explica mejor el comportamiento de los activos "
                        "en promedio. Mayor R² significa que el modelo captura más de lo que mueve a cada activo."
                    )
                else:
                    st.info(
                        f"Mayor R² promedio: **{_mejor_modelo}**. "
                        "R² ajustado creciente CAPM→FF3→APT es el patrón esperado — "
                        "cada modelo adicional agrega factores con poder explicativo marginal."
                    )

    with tab6:
        if narrativa:
            st.markdown(narrativa)

    st.divider()
    st.info("✅ Factores listos. Continúa con **⚠️ 3c Riesgo** en el menú lateral.")
