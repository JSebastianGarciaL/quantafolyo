# =============================================================================
# QuantαfolyΩ — pages/4_Verificacion.py
# Verificación Consolidada Fase 3
# =============================================================================

import streamlit as st
import pandas as pd
from modulos.errores import interpretar_error

st.set_page_config(page_title="Verificación · QuantαfolyΩ", page_icon="✅", layout="wide")
st.title("✅ Verificación Consolidada — Fase 3")

# Verificar prerrequisitos
faltan = []
if st.session_state.get("retornos") is None:       faltan.append("Fase 1 — Datos")
if st.session_state.get("res_3a") is None:          faltan.append("Fase 3a — Markowitz")
if st.session_state.get("res_capm") is None:        faltan.append("Fase 3b — Factores")
if st.session_state.get("res_bt") is None:          faltan.append("Fase 3c — Riesgo")

if faltan:
    st.warning(f"Primero ejecuta: {', '.join(faltan)}")
    st.stop()

retornos           = st.session_state["retornos"]
retornos_benchmark = st.session_state["retornos_benchmark"]
capm_df            = st.session_state["res_capm"]
ff3_df             = st.session_state.get("res_ff3", pd.DataFrame())
apt_df             = st.session_state.get("res_apt", pd.DataFrame())
res_bt             = st.session_state["res_bt"]
metricas_3a        = st.session_state["metricas_3a"]
rf_anual           = st.session_state["res_3a"]["rf_anual"]
nivel_asistente    = st.session_state.get("nivel_asistente", "basico")

st.markdown(
    f"**Activos:** {', '.join(retornos.columns.tolist())} &nbsp;|&nbsp; "
    f"**Pruebas:** Spanning · Chow · Consistencia modelos"
)
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("▶ Ejecutar Verificación", type="primary", width="stretch")

if ejecutar:
    from modulos.verificacion import pipeline_verificacion

    with st.spinner("Ejecutando pruebas de verificación..."):
        factor = st.session_state.get("metadatos", {}).get("factor_anualizacion", 12)
        resultado = pipeline_verificacion(
            retornos, retornos_benchmark,
            capm_df=capm_df,
            ff3_df=ff3_df,
            apt_df=apt_df,
            res_bt=res_bt,
            metricas_3a=metricas_3a,
            rf_anual=rf_anual,
            nivel_asistente=nivel_asistente,
            factor=factor,
        )

    with st.expander("📋 Log del proceso", expanded=resultado.get("error") is not None):
        for linea in resultado.get("logs", []):
            st.text(linea)

    if resultado.get("error"):
        st.error(interpretar_error(resultado["error"], fase="verificacion"))
        with st.expander("🔧 Detalle técnico (para soporte)"):
            st.code(str(resultado["error"]))
        st.stop()

    st.session_state["res_spanning"]    = resultado["res_spanning"]
    st.session_state["res_chow"]        = resultado["res_chow"]
    st.session_state["res_consistencia"] = resultado["res_consistencia"]
    st.session_state["semaforo_f3"]     = resultado["semaforo_f3"]
    st.session_state["narrativa_ver"]   = resultado["narrativa_ver"]
    st.session_state["fase_completada"]["verificacion"] = True
    st.success("✅ Verificación completada.")

# --- Resultados ---
if st.session_state.get("semaforo_f3") is not None:
    semaforo_f3    = st.session_state["semaforo_f3"]
    res_spanning   = st.session_state["res_spanning"]
    res_chow       = st.session_state["res_chow"]
    res_consistencia = st.session_state["res_consistencia"]
    narrativa      = st.session_state.get("narrativa_ver", "")
    problemas      = res_consistencia['problema'].tolist() if not res_consistencia.empty else []

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚦 Semáforo", "🌐 Spanning", "📉 Chow",
        "🔄 Consistencia", "🤖 Asistente"
    ])

    with tab1:
        st.markdown("### Diagnóstico consolidado Fase 3")
        st.dataframe(semaforo_f3.set_index("Prueba"), width="stretch")
        st.caption("🟢 Validado · 🟡 Advertencia · 🔴 Violación")

        # Diagnóstico rápido
        n_expande   = int(res_spanning['rechaza'].sum())
        n_inestable = int((res_chow['semaforo'] == 'ROJO').sum())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Activos que expanden", f"{n_expande}/{len(res_spanning)}")
        with col2:
            st.metric("Betas inestables", n_inestable,
                      delta_color="inverse" if n_inestable > 0 else "normal")
        with col3:
            st.metric("Inconsistencias", len(problemas),
                      delta_color="inverse" if problemas else "normal")

    with tab2:
        st.markdown("**Spanning — Huberman & Kandel (1987)**")
        st.caption("H0: α=0 Y β=1. Rechazar → el activo expande el conjunto de oportunidades.")
        st.dataframe(res_spanning.round(4), width="stretch")

        # --- Interpretación inline: Spanning ---
        _n_expande = int(res_spanning['rechaza'].sum())
        _n_total   = len(res_spanning)
        _expanden  = res_spanning[res_spanning['rechaza']].index.tolist()
        _no_expanden = res_spanning[~res_spanning['rechaza']].index.tolist()

        if nivel_asistente == "basico":
            if _n_expande == _n_total:
                st.success(
                    f"✅ Los **{_n_total} activos** aportan oportunidades que no están en el S&P 500. "
                    "Construir este portafolio tiene valor real — no es simplemente duplicar el índice."
                )
            elif _n_expande >= _n_total // 2:
                st.success(
                    f"✅ **{_n_expande} de {_n_total} activos** ({', '.join(_expanden)}) "
                    "expanden las oportunidades más allá del S&P 500. "
                    "El portafolio tiene valor de diversificación demostrable."
                )
            else:
                st.warning(
                    f"⚠️ Solo **{_n_expande} de {_n_total} activos** expanden el conjunto de oportunidades. "
                    f"{', '.join(_no_expanden)} se comportan tan similar al índice que podrías reemplazarlos "
                    "con el S&P 500 directamente."
                )
        else:
            if _n_expande >= _n_total // 2:
                st.success(
                    f"✅ {_n_expande}/{_n_total} activos rechazan H0 de spanning (α=0 ∩ β=1). "
                    "El conjunto de oportunidades de inversión es estadísticamente más amplio que el benchmark."
                )
            else:
                st.warning(
                    f"⚠️ {_n_expande}/{_n_total} activos rechazan spanning. "
                    f"**{', '.join(_no_expanden)}** no aportan diversificación incremental sobre el S&P 500 "
                    "(Huberman & Kandel, 1987). Considera ampliar el universo de activos."
                )

    with tab3:
        st.markdown("**Estabilidad de betas — Chow (1960)**")
        st.caption("H0: beta estable entre primera y segunda mitad del período.")
        st.dataframe(res_chow.round(4), width="stretch")

        # --- Interpretación inline: Chow ---
        _inestables = res_chow[res_chow['semaforo'] == 'ROJO'].index.tolist()
        _cambio_mod = res_chow[res_chow['semaforo'] == 'AMARILLO'].index.tolist()
        _estables   = res_chow[res_chow['semaforo'] == 'VERDE'].index.tolist()

        if nivel_asistente == "basico":
            if not _inestables and not _cambio_mod:
                st.success(
                    "✅ La sensibilidad al mercado fue estable a lo largo de todo el período. "
                    "Las betas calculadas son representativas del comportamiento esperado del portafolio."
                )
            else:
                if _inestables:
                    st.error(
                        f"❌ **{_inestables}** cambiaron su comportamiento frente al mercado de forma significativa. "
                        "La beta histórica puede no predecir bien su comportamiento futuro — "
                        "úsala con precaución."
                    )
                if _cambio_mod:
                    st.warning(
                        f"🟡 **{_cambio_mod}** muestran un cambio moderado en su relación con el mercado. "
                        "Posiblemente el perfil de riesgo del activo cambió en la segunda mitad del período."
                    )
        else:
            if not _inestables and not _cambio_mod:
                st.success(
                    "✅ Prueba de Chow no rechaza estabilidad de parámetros en ningún activo. "
                    "Supuesto de coeficientes constantes de MCO satisfecho en el período analizado."
                )
            else:
                if _inestables:
                    st.error(
                        f"❌ **{_inestables}**: Chow rechaza estabilidad (quiebre estructural). "
                        "β₁ ≠ β₂ estadísticamente — el beta de la regresión completa es un promedio "
                        "de dos regímenes distintos. Interpretar con extrema precaución."
                    )
                if _cambio_mod:
                    st.warning(
                        f"🟡 **{_cambio_mod}**: Δβ > 0.30 pero Chow no rechaza formalmente. "
                        "Cambio económicamente significativo aunque no estadístico — "
                        "posible quiebre suave o cambio gradual de régimen."
                    )

    with tab4:
        st.markdown("**Consistencia entre modelos (CAPM → FF3 → APT)**")
        if not problemas:
            st.success("✅ Todos los modelos son consistentes entre sí.")
        else:
            for p in problemas:
                st.warning(f"⚠️ {p}")

    with tab5:
        if narrativa:
            st.markdown(narrativa)

    st.divider()
    st.info("✅ Verificación lista. Continúa con **📄 5 Reporte** en el menú lateral.")
