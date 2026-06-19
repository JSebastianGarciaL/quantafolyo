# =============================================================================
# QuantαfolyΩ — pages/2_Estadistico.py
# Fase 2: Análisis Estadístico Preliminar
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from modulos.errores import interpretar_error
from config import get_colores, get_plotly_layout

st.set_page_config(page_title="Estadístico · QuantαfolyΩ", page_icon="📐", layout="wide")
st.title("📐 Fase 2 — Análisis Estadístico Preliminar")

if st.session_state.get("retornos") is None:
    st.warning("Primero ejecuta la **📥 Fase 1 — Datos**.")
    st.stop()

retornos          = st.session_state["retornos"]
nivel_asistente   = st.session_state.get("nivel_asistente", "basico")

st.markdown(
    f"**Activos:** {', '.join(retornos.columns.tolist())} &nbsp;|&nbsp; "
    f"**Observaciones:** {len(retornos)} &nbsp;|&nbsp; "
    f"**Nivel asistente:** {nivel_asistente}"
)
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("▶ Ejecutar Fase 2", type="primary", width="stretch")

if ejecutar:
    from modulos.estadistico import pipeline_estadistico

    with st.spinner("Ejecutando pruebas estadísticas..."):
        resultado = pipeline_estadistico(retornos, nivel_asistente)

    with st.expander("📋 Log del proceso", expanded=resultado.get("error") is not None):
        for linea in resultado.get("logs", []):
            st.text(linea)

    if resultado.get("error"):
        st.error(interpretar_error(resultado["error"], fase="estadistico"))
        with st.expander("🔧 Detalle técnico (para soporte)"):
            st.code(str(resultado["error"]))
        st.stop()

    st.session_state["res_estac"]    = resultado["res_estac"]
    st.session_state["res_pp"]       = resultado["res_pp"]
    st.session_state["res_norm"]     = resultado["res_norm"]
    st.session_state["res_spearman"] = resultado["res_spearman"]
    st.session_state["res_autocorr"] = resultado["res_autocorr"]
    st.session_state["res_arch"]     = resultado["res_arch"]
    st.session_state["res_mardia"]   = resultado["res_mardia"]
    st.session_state["res_hme"]      = resultado["res_hme"]
    st.session_state["semaforo_f2"]  = resultado["semaforo_f2"]
    st.session_state["narrativa_f2"] = resultado["narrativa_f2"]
    st.session_state["fase_completada"]["estadistico"] = True
    st.success("✅ Fase 2 completada.")

# --- Mostrar resultados ---
if st.session_state.get("semaforo_f2") is not None:
    semaforo    = st.session_state["semaforo_f2"]
    res_estac   = st.session_state["res_estac"]
    res_pp      = st.session_state["res_pp"]
    res_norm    = st.session_state["res_norm"]
    res_spearman = st.session_state["res_spearman"]
    res_autocorr = st.session_state["res_autocorr"]
    res_arch    = st.session_state["res_arch"]
    res_mardia  = st.session_state["res_mardia"]
    res_hme     = st.session_state["res_hme"]
    narrativa   = st.session_state.get("narrativa_f2", "")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🚦 Semáforo", "📉 Estacionariedad", "🔔 Normalidad",
        "🔗 Correlaciones", "🔁 Autocorrelación", "⚡ ARCH",
        "🌐 Multivariante", "🤖 Asistente"
    ])

    with tab1:
        st.markdown("### Diagnóstico consolidado")
        st.dataframe(semaforo.set_index("Prueba"), width="stretch")
        st.caption("🟢 Supuesto cumplido · 🟡 Advertencia · 🔴 Violación")

    with tab2:
        st.markdown("#### ADF + KPSS")
        st.dataframe(res_estac.set_index("Ticker"), width="stretch")

        # --- Interpretación inline: Estacionariedad ---
        _no_estac = res_estac[res_estac["Semaforo"] == "ROJO"]["Ticker"].tolist()
        _inconc   = res_estac[res_estac["Semaforo"] == "AMARILLO"]["Ticker"].tolist()
        if not _no_estac and not _inconc:
            if nivel_asistente == "basico":
                st.success("✅ Todos los activos tienen retornos estables en el tiempo. "
                           "Los modelos estadísticos que vienen pueden aplicarse con confianza.")
            else:
                st.success("✅ Estacionariedad confirmada en todos los activos (ADF + KPSS consistentes). "
                           "Supuesto de covarianza-estacionariedad satisfecho para MCO y Markowitz.")
        elif _no_estac:
            if nivel_asistente == "basico":
                st.warning(f"⚠️ **{_no_estac}** muestran retornos inestables en el tiempo. "
                           "Esto puede afectar la fiabilidad de los modelos — los resultados son orientativos.")
            else:
                st.warning(f"⚠️ **{_no_estac}** no rechazan raíz unitaria (ADF) y KPSS rechaza estacionariedad. "
                           "MCO sobre series no estacionarias produce regresiones espurias (Granger & Newbold, 1974).")
        else:
            if nivel_asistente == "basico":
                st.warning(f"🟡 **{_inconc}** tienen resultados mixtos — ADF y KPSS no coinciden. "
                           "Puede haber un cambio estructural en el período. Los modelos aplican con precaución.")
            else:
                st.warning(f"🟡 **{_inconc}**: ADF y KPSS contradictorios — posible quiebre estructural "
                           "o heterocedasticidad no modelada. Considerar prueba de Zivot-Andrews.")

        st.markdown("#### Phillips-Perron")
        st.dataframe(res_pp.set_index("Ticker"), width="stretch")

        # --- Interpretación inline: Phillips-Perron vs ADF ---
        _pp_incons = res_pp[res_pp["Consistencia"] == "INCONSISTENTES — revisar"]["Ticker"].tolist()
        if _pp_incons:
            if nivel_asistente == "basico":
                st.warning(f"🟡 **{_pp_incons}**: el test PP y el ADF no llegan a la misma conclusión. "
                           "Con muestras pequeñas esto es esperable — prevalece el diagnóstico ADF+KPSS.")
            else:
                st.warning(f"🟡 **{_pp_incons}**: ADF y PP inconsistentes. PP es robusto a heterocedasticidad "
                           "en errores (corrección no paramétrica de Newey-West). "
                           "Divergencia sugiere estructura de error compleja — considerar KPSS como desempate.")
        else:
            st.success("✅ ADF y Phillips-Perron coinciden en todos los activos.")

    with tab3:
        st.dataframe(res_norm.set_index("Ticker"), width="stretch")

        # --- Interpretación inline: Normalidad ---
        _no_norm  = res_norm[res_norm["Semaforo"] == "ROJO"]["Ticker"].tolist()
        _mix_norm = res_norm[res_norm["Semaforo"] == "AMARILLO"]["Ticker"].tolist()
        if not _no_norm and not _mix_norm:
            if nivel_asistente == "basico":
                st.success("✅ Los retornos siguen una distribución normal. "
                           "Los modelos de Markowitz y VaR paramétrico aplican sin restricciones.")
            else:
                st.success("✅ No se rechaza normalidad (JB + SW). "
                           "Supuesto distribucional de Markowitz (1952) satisfecho. "
                           "VaR paramétrico es una aproximación válida.")
        elif _no_norm:
            if nivel_asistente == "basico":
                st.warning(f"⚠️ **{_no_norm}** no siguen distribución normal — "
                           "los eventos extremos ocurren con más frecuencia de lo esperado. "
                           "Por eso el análisis incluye CVaR histórico como métrica de riesgo principal.")
            else:
                st.warning(f"⚠️ **{_no_norm}** rechazan normalidad (JB + SW). "
                           "Colas pesadas y/o asimetría detectadas. "
                           "VaR paramétrico subestima riesgo de cola — usar VaR histórico como referencia. "
                           "Frontera eficiente es aproximación de primer orden.")
        else:
            if nivel_asistente == "basico":
                st.warning(f"🟡 **{_mix_norm}** muestran evidencia mixta de normalidad. "
                           "JB y Shapiro-Wilk no coinciden — los modelos aplican con precaución.")
            else:
                st.warning(f"🟡 **{_mix_norm}**: JB y SW divergen. "
                           "Con n > 50, SW pierde poder estadístico — JB es la prueba de referencia.")

        retornos = st.session_state["retornos"]
        n_cols   = len(retornos.columns)
        _tema    = st.get_option("theme.base") or "light"
        _colores = get_colores(_tema)
        _seq     = _colores["graf_seq"]

        fig = make_subplots(rows=1, cols=n_cols,
                            subplot_titles=list(retornos.columns))
        for i, col in enumerate(retornos.columns, 1):
            fig.add_trace(go.Histogram(
                x=retornos[col].dropna(), name=col,
                nbinsx=20, showlegend=False,
                marker_color=_seq[(i-1) % len(_seq)],
                opacity=0.85,
            ), row=1, col=i)
        fig.update_layout(
            title="Distribución de retornos",
            height=350,
            margin=dict(l=40, r=20, t=40, b=20),
            **get_plotly_layout(_tema),
        )
        st.plotly_chart(fig, width="stretch")

    with tab4:
        pearson  = res_spearman["pearson"]
        spearman = res_spearman["spearman"]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Correlación de Pearson**")
            st.dataframe(pearson.round(4), width="stretch")
        with col2:
            st.markdown("**Correlación de Spearman**")
            st.dataframe(spearman.round(4), width="stretch")

        # --- Interpretación inline: Correlaciones ---
        pares = res_spearman.get("pares_problematicos", [])
        if pares:
            if nivel_asistente == "basico":
                st.warning(f"⚠️ **{len(pares)} par(es)** tienen una relación más compleja que una línea recta. "
                           "Markowitz usa correlaciones lineales — puede perder dependencias reales entre estos activos. "
                           "Los pesos óptimos son válidos como aproximación.")
            else:
                st.warning(f"⚠️ **{len(pares)} par(es)** con |Δ Pearson−Spearman| > 0.10 — dependencia no lineal. "
                           "La matriz de covarianza de Markowitz puede ser una aproximación imprecisa. "
                           "DCC-GARCH modelaría la dependencia dinámica con mayor precisión.")
        else:
            if nivel_asistente == "basico":
                st.success("✅ Las relaciones entre activos son lineales. "
                           "Markowitz captura bien la dependencia real — los pesos óptimos son confiables.")
            else:
                st.success("✅ Pearson y Spearman son consistentes en todos los pares (|Δ| ≤ 0.10). "
                           "No hay evidencia de dependencia no lineal relevante. "
                           "La estructura de covarianza es una representación adecuada.")

    with tab5:
        st.dataframe(res_autocorr.set_index("Ticker"), width="stretch")

        # --- Interpretación inline: Autocorrelación ---
        _col_rechaza = next((c for c in res_autocorr.columns if "rechaza" in c.lower()), None)
        _con_autocorr = res_autocorr[res_autocorr[_col_rechaza] == True]["Ticker"].tolist() if _col_rechaza else []
        if _con_autocorr:
            if nivel_asistente == "basico":
                st.warning(f"⚠️ **{_con_autocorr}**: los retornos pasados tienen cierto poder predictivo sobre los futuros. "
                           "Esto puede indicar ineficiencias de mercado o simplemente un período atípico.")
            else:
                st.warning(f"⚠️ **{_con_autocorr}**: Ljung-Box rechaza ausencia de autocorrelación. "
                           "Residuos serialmente correlacionados violan supuesto de independencia de MCO. "
                           "Errores estándar de CAPM/FF3 pueden estar sesgados — usar HAC (Newey-West).")
        else:
            if nivel_asistente == "basico":
                st.success("✅ Los retornos no muestran patrones predecibles en el tiempo. "
                           "Consistente con mercados eficientes en forma débil.")
            else:
                st.success("✅ Ljung-Box no rechaza independencia serial en ningún activo. "
                           "Supuesto de no autocorrelación de MCO satisfecho.")

    with tab6:
        st.dataframe(res_arch.set_index("Ticker"), width="stretch")

        # --- Interpretación inline: ARCH ---
        _col_arch = next((c for c in res_arch.columns if "rechaza" in c.lower()), None)
        _con_arch = res_arch[res_arch[_col_arch] == True]["Ticker"].tolist() if _col_arch else []
        if _con_arch:
            if nivel_asistente == "basico":
                st.warning(f"⚠️ **{_con_arch}** tienen volatilidad variable en el tiempo — "
                           "hay períodos de calma y períodos de turbulencia que se agrupan. "
                           "El VaR paramétrico puede subestimar el riesgo en momentos de alta volatilidad.")
            else:
                st.warning(f"⚠️ **{_con_arch}**: ARCH-LM rechaza homocedasticidad. "
                           "Volatilidad condicional heterocedástica — la varianza de Markowitz es un promedio "
                           "que subestima el riesgo en regímenes de alta volatilidad. "
                           "VaR histórico es la métrica primaria; GARCH modelaría la dinámica (Capa 2).")
        else:
            if nivel_asistente == "basico":
                st.success("✅ La volatilidad es relativamente estable en el tiempo. "
                           "El VaR paramétrico es una aproximación razonable del riesgo real.")
            else:
                st.success("✅ ARCH-LM no rechaza homocedasticidad. "
                           "Varianza condicional constante — supuesto de homocedasticidad de MCO satisfecho.")

    with tab7:
        if res_mardia:
            import pandas as pd
            datos_mardia = {k: v for k, v in res_mardia.items()
                            if k not in ('diagnostico', 'semaforo', 'implicacion')}
            st.dataframe(
                pd.DataFrame([datos_mardia]).round(4),
                width="stretch",
            )
            diag = res_mardia.get('diagnostico', '')
            impl = res_mardia.get('implicacion', '')
            color_fn = {"VERDE": st.success, "AMARILLO": st.warning, "ROJO": st.error}
            color_fn.get(res_mardia.get('semaforo', 'AMARILLO'), st.info)(
                f"**{diag}** — {impl}"
            )

    with tab8:
        if narrativa:
            st.markdown(narrativa)

    st.divider()
    st.info("✅ Estadística lista. Continúa con **📊 3a Markowitz** en el menú lateral.")
