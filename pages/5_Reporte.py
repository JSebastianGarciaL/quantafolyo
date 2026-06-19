# =============================================================================
# QuantαfolyΩ — pages/5_Reporte.py
# Fase 4: Asistente Explicativo + Reporte PDF
# =============================================================================

import streamlit as st
import pandas as pd
from modulos.errores import interpretar_error
from config import get_colores, get_plotly_layout

st.set_page_config(page_title="Reporte · QuantαfolyΩ", page_icon="📄", layout="wide")
st.title("📄 Fase 4 — Asistente y Reporte PDF")

# Verificar prerrequisitos mínimos
faltan = []
if st.session_state.get("retornos")   is None: faltan.append("Fase 1 — Datos")
if st.session_state.get("res_3a")     is None: faltan.append("Fase 3a — Markowitz")
if st.session_state.get("res_capm")   is None: faltan.append("Fase 3b — Factores")
if st.session_state.get("res_var")    is None: faltan.append("Fase 3c — Riesgo")
if st.session_state.get("res_spanning") is None: faltan.append("Verificación")

if faltan:
    st.warning(f"Primero ejecuta: {', '.join(faltan)}")
    st.stop()

# Recoger todos los datos de session_state
retornos           = st.session_state["retornos"]
retornos_benchmark = st.session_state["retornos_benchmark"]
pesos_df           = st.session_state["pesos_df"]
res_3a             = st.session_state["res_3a"]
metricas_3a        = st.session_state["metricas_3a"]
res_capm           = st.session_state["res_capm"]
res_ff3            = st.session_state.get("res_ff3", pd.DataFrame())
res_apt            = st.session_state.get("res_apt", pd.DataFrame())
res_var_dict       = st.session_state.get("_resultados_var", {})
metricas_3c        = st.session_state.get("metricas_3c", {})
res_bt             = st.session_state["res_bt"]
res_estres         = st.session_state.get("res_estres", pd.DataFrame())
res_spanning       = st.session_state["res_spanning"]
res_chow           = st.session_state["res_chow"]
res_estac          = st.session_state.get("res_estac", pd.DataFrame())
res_norm           = st.session_state.get("res_norm",  pd.DataFrame())
res_arch           = st.session_state.get("res_arch",  pd.DataFrame())
res_mardia         = st.session_state.get("res_mardia", {})
nivel_asistente    = st.session_state.get("nivel_asistente", "basico")

st.markdown(
    f"**Activos:** {', '.join(retornos.columns.tolist())} &nbsp;|&nbsp; "
    f"**Nivel:** {nivel_asistente}"
)
st.divider()

col_btn, _ = st.columns([1, 3])
with col_btn:
    ejecutar = st.button("▶ Generar Reporte", type="primary", width="stretch")

if ejecutar:
    from modulos.reporte import pipeline_reporte

    factor = st.session_state.get("metadatos", {}).get("factor_anualizacion", 12)
    with st.spinner("Generando narrativa, gráfico y PDF..."):
        resultado = pipeline_reporte(
            retornos=retornos,
            retornos_benchmark=retornos_benchmark,
            pesos_df=pesos_df,
            res_3a=res_3a,
            metricas_3a=metricas_3a,
            res_capm=res_capm,
            res_ff3=res_ff3,
            res_apt=res_apt,
            res_var_dict=res_var_dict,
            metricas_3c=metricas_3c,
            res_bt=res_bt,
            res_estres=res_estres,
            res_spanning=res_spanning,
            res_chow=res_chow,
            res_estac=res_estac,
            res_norm=res_norm,
            res_arch=res_arch,
            res_mardia=res_mardia,
            nivel_asistente=nivel_asistente,
            factor=factor,
        )

    with st.expander("📋 Log del proceso", expanded=resultado.get("error") is not None):
        for linea in resultado.get("logs", []):
            st.text(linea)

    if resultado.get("error"):
        st.error(interpretar_error(resultado["error"], fase="reporte"))
        with st.expander("🔧 Detalle técnico (para soporte)"):
            st.code(str(resultado["error"]))
        st.stop()

    st.session_state["narrativa"]   = resultado["narrativa"]
    st.session_state["fig_resumen"] = resultado["fig_resumen"]
    st.session_state["pdf_bytes"]   = resultado["pdf_bytes"]
    st.session_state["fase_completada"]["asistente"] = True
    st.success("✅ Reporte generado.")

# --- Mostrar resultados ---
if st.session_state.get("narrativa") is not None:
    narrativa   = st.session_state["narrativa"]
    fig_resumen = st.session_state["fig_resumen"]
    pdf_bytes   = st.session_state["pdf_bytes"]

    # =========================================================================
    # RESUMEN EJECUTIVO GLOBAL (P1.3)
    # =========================================================================

    # --- Extraer valores clave ---
    _res_3a      = st.session_state.get("res_3a", {})
    _metricas_3a = st.session_state.get("metricas_3a", pd.DataFrame())
    _metricas_3c = st.session_state.get("metricas_3c", {})
    _res_var_dict = st.session_state.get("_resultados_var", {})
    _res_capm    = st.session_state.get("res_capm", pd.DataFrame())
    _pesos_df    = st.session_state.get("pesos_df", pd.DataFrame())
    _res_spanning = st.session_state.get("res_spanning", pd.DataFrame())
    _res_chow    = st.session_state.get("res_chow", pd.DataFrame())
    _res_bt      = st.session_state.get("res_bt", pd.DataFrame())
    _res_norm    = st.session_state.get("res_norm", pd.DataFrame())

    # Portafolio tangente
    _tang      = _res_3a.get("tangente", {})
    _ret_tang  = float(_tang.get("retorno", 0))
    _vol_tang  = float(_tang.get("volatilidad", 0))
    _sh_tang   = float(_tang.get("sharpe", 0))
    _sh_bench  = float(_res_3a.get("benchmark", {}).get("sharpe", 0))

    # Riesgo
    _var_95   = float(_res_var_dict.get("95%", {}).get("var_hist", 0))
    _cvar_95  = float(_res_var_dict.get("95%", {}).get("cvar_hist", 0))
    _mdd_row  = (_metricas_3a[_metricas_3a["nombre"] == "Tangente"]["max_drawdown"].values
                 if not _metricas_3a.empty and "nombre" in _metricas_3a.columns else [])
    _mdd      = float(_mdd_row[0]) if len(_mdd_row) > 0 else float(_metricas_3c.get("max_drawdown", 0) or 0)

    # Factores
    _beta_port = 1.0
    if not _res_capm.empty and not _pesos_df.empty and "Beta" in _res_capm.columns:
        try:
            _betas_idx = _res_capm.set_index("Ticker")["Beta"]
            _beta_port = float((_pesos_df["Tangente"] * _betas_idx).sum())
        except Exception:
            pass
    _alpha_sig = (_res_capm[_res_capm["Alpha_sig"] == True]["Ticker"].tolist()
                  if not _res_capm.empty and "Alpha_sig" in _res_capm.columns else [])

    # Validación
    _n_expande  = int(_res_spanning["rechaza"].sum()) if not _res_spanning.empty and "rechaza" in _res_spanning.columns else 0
    _n_total_sp = len(_res_spanning) if not _res_spanning.empty else 0
    _n_inestab  = int((_res_chow["semaforo"] == "ROJO").sum()) if not _res_chow.empty and "semaforo" in _res_chow.columns else 0
    _kupiec_ok  = bool(_res_bt["kupiec_ok"].all()) if not _res_bt.empty and "kupiec_ok" in _res_bt.columns else True
    _no_norm    = (_res_norm[_res_norm["Semaforo"] == "ROJO"]["Ticker"].tolist()
                   if not _res_norm.empty and "Semaforo" in _res_norm.columns else [])

    # --- Semáforo global — usando la función compartida (única fuente de verdad) ---
    # CORRECCIÓN (auditoría P1.2→P3.1): antes se calculaba aquí con criterios
    # distintos a los de pipeline_reporte, pudiendo dar veredictos diferentes
    # en pantalla vs en el PDF. Ahora ambos usan calcular_semaforo_global().
    from modulos.reporte import calcular_semaforo_global

    _n_no_norm = len(_no_norm)
    _sem = calcular_semaforo_global(
        sharpe_port=_sh_tang,
        sharpe_bench=_sh_bench,
        kupiec_ok=_kupiec_ok,
        n_chow_inestables=_n_inestab,
        n_no_normales=_n_no_norm,
    )
    _alertas_raw   = _sem["alertas"]    # lista de claves internas
    _positivos_raw = _sem["positivos"]
    _n_alertas     = len(_alertas_raw)

    # Textos legibles para mostrar en pantalla
    _TEXTOS_ALERTA = {
        "sharpe":     "Sharpe por debajo del benchmark",
        "kupiec":     "VaR no calibrado (Kupiec)",
        "chow":       f"{_n_inestab} beta(s) inestable(s) (Chow)",
        "normalidad": f"no normalidad en {_no_norm}",
    }
    _alertas   = [_TEXTOS_ALERTA.get(k, k) for k in _alertas_raw]
    _positivos = _positivos_raw

    # Añadir puntos adicionales al resumen que no forman parte del semáforo formal
    if _n_expande >= _n_total_sp // 2 and _n_total_sp > 0:
        _positivos = _positivos + [f"{_n_expande}/{_n_total_sp} activos expanden oportunidades"]
    if abs(_mdd) < 0.20:
        _positivos = _positivos + ["caída máxima controlada (<20%)"]

    if _n_alertas == 0:
        _estado_global = ("✅", "success", "Portafolio bien construido", "green")
    elif _n_alertas <= 2:
        _estado_global = ("⚠️", "warning", "Portafolio con advertencias menores", "orange")
    else:
        _estado_global = ("❌", "error", "Portafolio requiere revisión", "red")

    # --- Renderizar Resumen Ejecutivo ---
    st.markdown("## 📋 Resumen Ejecutivo")

    # Semáforo global
    _icono, _tipo_msg, _label_global, _ = _estado_global
    getattr(st, _tipo_msg)(f"{_icono} **{_label_global}**")

    st.markdown("---")

    # Cuatro columnas de métricas clave
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("#### 📈 Portafolio")
        st.metric("Retorno esperado anual", f"{_ret_tang*100:.1f}%")
        st.metric("Volatilidad anual",      f"{_vol_tang*100:.1f}%")
        _delta_sh = _sh_tang - _sh_bench
        st.metric("Sharpe vs S&P 500",
                  f"{_sh_tang:.2f}",
                  delta=f"{_delta_sh:+.2f} vs benchmark",
                  delta_color="normal" if _delta_sh >= 0 else "inverse")

    with col2:
        st.markdown("#### ⚠️ Riesgo")
        st.metric("VaR histórico 95% (mensual)", f"{_var_95*100:.2f}%")
        st.metric("CVaR histórico 95% (mensual)", f"{_cvar_95*100:.2f}%")
        st.metric("Caída máxima (Max DD)",
                  f"{abs(_mdd)*100:.1f}%",
                  delta="Controlada" if abs(_mdd) < 0.20 else "Elevada",
                  delta_color="normal" if abs(_mdd) < 0.20 else "inverse")

    with col3:
        st.markdown("#### 📐 Factores")
        _perfil_beta = ("Agresivo" if _beta_port > 1.1
                        else "Defensivo" if _beta_port < 0.9
                        else "Neutral")
        st.metric("Beta del portafolio",
                  f"{_beta_port:.2f}",
                  delta=_perfil_beta)
        st.metric("Alpha significativo",
                  f"{len(_alpha_sig)} activo(s)" if _alpha_sig else "Ninguno",
                  delta="Retorno extra detectado" if _alpha_sig else "Sin alpha",
                  delta_color="normal" if _alpha_sig else "off")
        st.metric("Activos sin normalidad", f"{len(_no_norm)}",
                  delta="Usar CVaR histórico" if _no_norm else "Normalidad OK",
                  delta_color="inverse" if _no_norm else "normal")

    with col4:
        st.markdown("#### ✅ Validación")
        st.metric("Expanden vs S&P 500",
                  f"{_n_expande}/{_n_total_sp}",
                  delta="Valor de diversificación" if _n_expande >= _n_total_sp // 2 else "Diversificación débil",
                  delta_color="normal" if _n_expande >= _n_total_sp // 2 else "inverse")
        st.metric("Betas inestables (Chow)",
                  f"{_n_inestab}",
                  delta="Betas estables ✅" if _n_inestab == 0 else "Revisar betas ⚠️",
                  delta_color="normal" if _n_inestab == 0 else "inverse")
        st.metric("VaR calibrado (Kupiec)",
                  "Sí ✅" if _kupiec_ok else "No ❌",
                  delta="Modelo confiable" if _kupiec_ok else "Subestima riesgo",
                  delta_color="normal" if _kupiec_ok else "inverse")

    st.markdown("---")

    # Conclusión narrativa adaptada al nivel
    if nivel_asistente == "basico":
        if _n_alertas == 0:
            st.success(
                f"**¿Vale la pena este portafolio?** Sí — el análisis completo respalda su construcción. "
                f"Genera un retorno esperado del **{_ret_tang*100:.1f}% anual** con una volatilidad del "
                f"**{_vol_tang*100:.1f}%**, superando al S&P 500 en eficiencia (Sharpe {_sh_tang:.2f} vs {_sh_bench:.2f}). "
                f"El modelo de riesgo está bien calibrado y {_n_expande} de {_n_total_sp} activos "
                "aportan diversificación real más allá del índice."
            )
        elif _n_alertas <= 2:
            st.warning(
                f"**¿Vale la pena este portafolio?** Con matices. "
                f"Retorno esperado del **{_ret_tang*100:.1f}% anual** con volatilidad del **{_vol_tang*100:.1f}%**. "
                f"Hay {_n_alertas} punto(s) de atención: {', '.join(_alertas)}. "
                "Los resultados son válidos pero conviene tener estas limitaciones presentes al tomar decisiones."
            )
        else:
            st.error(
                f"**¿Vale la pena este portafolio?** Requiere revisión. "
                f"El análisis detectó {_n_alertas} advertencias relevantes: {', '.join(_alertas)}. "
                "Considera ajustar la selección de activos o ampliar la ventana temporal antes de usar estos resultados."
            )
    else:
        _linea_alpha = (f"Alpha de Jensen significativo en {_alpha_sig}. " if _alpha_sig else "Sin alpha significativo. ")
        _linea_norm  = (f"No normalidad en {_no_norm} — VaR histórico es la métrica primaria. " if _no_norm else "")
        if _n_alertas == 0:
            st.success(
                f"**Diagnóstico:** Portafolio estadísticamente robusto. "
                f"Sharpe tangente {_sh_tang:.3f} > benchmark {_sh_bench:.3f}. "
                f"Beta ponderado {_beta_port:.2f} ({_perfil_beta.lower()}). "
                f"{_linea_alpha}{_linea_norm}"
                f"Spanning: {_n_expande}/{_n_total_sp} activos expanden el conjunto de oportunidades. "
                "Chow y Kupiec validados."
            )
        elif _n_alertas <= 2:
            st.warning(
                f"**Diagnóstico:** Portafolio válido con advertencias. "
                f"Sharpe {_sh_tang:.3f} {'>' if _sh_tang > _sh_bench else '<'} benchmark {_sh_bench:.3f}. "
                f"{_linea_alpha}{_linea_norm}"
                f"Advertencias: {'; '.join(_alertas)}. "
                "Resultados interpretables con las limitaciones señaladas."
            )
        else:
            st.error(
                f"**Diagnóstico:** Múltiples violaciones detectadas. "
                f"Alertas: {'; '.join(_alertas)}. "
                "Supuestos estadísticos comprometidos — inferencia con alta precaución."
            )

    st.divider()

    # =========================================================================
    # FIN RESUMEN EJECUTIVO
    # =========================================================================

    tab1, tab2, tab3 = st.tabs(["📊 Gráfico resumen", "📝 Narrativa", "📄 Descargar PDF"])

    with tab1:
        _tema    = st.get_option("theme.base") or "light"
        _colores = get_colores(_tema)
        fig_resumen.update_layout(**get_plotly_layout(_tema))
        _seq = _colores["graf_seq"]
        for i, trace in enumerate(fig_resumen.data):
            if hasattr(trace, "line"):
                trace.update(line=dict(color=_seq[i % len(_seq)], width=2))
            elif hasattr(trace, "marker"):
                trace.update(marker=dict(color=_seq[i % len(_seq)]))
        st.plotly_chart(fig_resumen, width="stretch")

    with tab2:
        st.text(narrativa)

    with tab3:
        st.markdown("### Reporte PDF listo para descargar")
        st.markdown(
            "El PDF contiene portada, narrativa completa del análisis "
            "y referencias bibliográficas."
        )
        nombre_archivo = (
            f"QuantafolyO_{'_'.join(retornos.columns.tolist())}"
            f"_{nivel_asistente}.pdf"
        )
        st.download_button(
            label="⬇️ Descargar PDF",
            data=pdf_bytes,
            file_name=nombre_archivo,
            mime="application/pdf",
            type="primary",
            width="stretch",
        )
        st.caption(f"Archivo: {nombre_archivo} · {len(pdf_bytes)//1024} KB")

    st.divider()
    st.success("🎉 ¡Análisis completo! Todas las fases ejecutadas correctamente.")
