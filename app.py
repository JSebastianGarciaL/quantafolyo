# =============================================================================
# QuantαfolyΩ — app.py
# Punto de entrada de la aplicación Streamlit.
# Gestiona la navegación multipágina y el estado global de la sesión.
# =============================================================================

import streamlit as st
from config import MIN_ACTIVOS, MAX_ACTIVOS, TICKER_BENCHMARK, DIVISA_BASE

# --- Configuración de página ---
st.set_page_config(
    page_title="QuantαfolyΩ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Tipografía y estilos globales ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.qaf-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.qaf-logo .alpha { color: #D76F02; }
.qaf-logo .omega { color: #985D73; }
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Inicialización de session_state ---
# Todos los resultados intermedios viven aquí.
# Ningún módulo escribe a disco — todo pasa por session_state.
_KEYS_RESULTADOS = [
    # Fase 1 — Datos
    "retornos", "retornos_benchmark", "precios_usd", "metadatos",
    "resultado_validacion", "divisas_activos",
    # Fase 2 — Estadístico
    "res_estac", "res_pp", "res_norm", "res_spearman",
    "res_autocorr", "res_arch", "res_mardia", "res_hme",
    "semaforo_f2", "narrativa_f2",
    # Fase 3a — Markowitz
    "mu", "S", "frontera", "mvp", "tangente", "equal_weight",
    "metricas_3a", "pesos_df", "res_3a", "fig_frontera", "narrativa_3a",
    # Fase 3b — Factores
    "res_capm", "res_ff3", "res_apt",
    "tabla_comp", "loadings_pca", "fig_betas", "narrativa_3b",
    # Fase 3c — Riesgo
    "res_var", "_resultados_var", "metricas_3c", "res_bt",
    "_resultados_bt", "res_estres", "fig_riesgo", "narrativa_3c",
    # Verificación Fase 3
    "res_spanning", "res_chow", "res_consistencia", "semaforo_f3", "narrativa_ver",
    # Fase 4 — Asistente y reporte
    "narrativa", "fig_resumen", "pdf_bytes",
]

for key in _KEYS_RESULTADOS:
    if key not in st.session_state:
        st.session_state[key] = None

# --- Estado del pipeline ---
if "pipeline_ejecutado" not in st.session_state:
    st.session_state["pipeline_ejecutado"] = False
if "fase_completada" not in st.session_state:
    st.session_state["fase_completada"] = {
        "datos": False, "estadistico": False, "markowitz": False,
        "factores": False, "riesgo": False, "verificacion": False,
        "asistente": False,
    }

# Parámetros de usuario — inicializados con valores por defecto
# para que las páginas no lancen KeyError antes del primer click
if "tickers_usuario" not in st.session_state:
    st.session_state["tickers_usuario"] = None
if "ventana_anos" not in st.session_state:
    st.session_state["ventana_anos"] = 5
if "frecuencia" not in st.session_state:
    st.session_state["frecuencia"] = "1mo"
if "nivel_asistente" not in st.session_state:
    st.session_state["nivel_asistente"] = "basico"

# --- Página de inicio ---
st.markdown("""
<div class="qaf-logo">
    Quant<span class="alpha">α</span>foly<span class="omega">Ω</span>
</div>
<p style="font-family:'Space Grotesk',sans-serif; color:#576071; font-size:1rem; margin-top:4px; margin-bottom:0;">
    Análisis cuantitativo de portafolios de inversión
</p>
""", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        ### ¿Cómo funciona?

        1. **Ingresa tus tickers** en el panel lateral izquierdo
        2. Selecciona la **ventana temporal** y la **frecuencia**
        3. Presiona **Analizar portafolio**
        4. Navega por los resultados usando el menú de la izquierda

        ### ¿Qué analiza?

        - **Datos:** descarga, conversión a USD, retornos logarítmicos
        - **Estadística:** estacionariedad, normalidad, autocorrelación, efectos ARCH, HME
        - **Markowitz:** frontera eficiente, portafolio tangente, MVP, equal-weight
        - **Modelos de factores:** CAPM, Fama-French 3F, APT con factores macro
        - **Riesgo:** VaR, CVaR, backtesting Kupiec-Christoffersen, análisis de estrés
        - **Verificación:** prueba de spanning, estabilidad de betas (Chow)
        - **Reporte:** narrativa completa + PDF descargable
        """
    )

with col2:
    st.markdown("### Estado del análisis")
    fases = st.session_state["fase_completada"]
    iconos = {True: "✅", False: "⬜"}
    st.markdown(f"{iconos[fases['datos']]} Fase 1 — Datos")
    st.markdown(f"{iconos[fases['estadistico']]} Fase 2 — Estadística")
    st.markdown(f"{iconos[fases['markowitz']]} Fase 3a — Markowitz")
    st.markdown(f"{iconos[fases['factores']]} Fase 3b — Factores")
    st.markdown(f"{iconos[fases['riesgo']]} Fase 3c — Riesgo")
    st.markdown(f"{iconos[fases['verificacion']]} Verificación")
    st.markdown(f"{iconos[fases['asistente']]} Asistente / Reporte")

    if st.session_state["pipeline_ejecutado"]:
        st.success("Análisis completado.")
    else:
        st.info("Configura el portafolio en el panel lateral y presiona **Analizar**.")

# --- Sidebar: inputs del usuario ---
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem; font-weight:600; margin-bottom:8px;">
        ⚙️ Quant<span style="color:#D76F02">α</span>foly<span style="color:#985D73">Ω</span>
    </div>
    """, unsafe_allow_html=True)

    tickers_input = st.text_area(
        "Tickers (uno por línea o separados por coma)",
        value="AAPL\nMSFT\nAMZN\nGOOGL",
        height=120,
        help="Ejemplos: AAPL (EE.UU.), EC (Ecopetrol, ADR NYSE), SAP.DE (Alemania), 7203.T (Japón)",
    )

    ventana_anos = st.slider(
        "Ventana temporal (años)",
        min_value=1, max_value=10, value=5,
        help="Período histórico de análisis. Mínimo recomendado: 5 años.",
    )

    frecuencia = st.selectbox(
        "Frecuencia de retornos",
        options=["1mo", "1wk"],
        format_func=lambda x: {"1mo": "Mensual (recomendada)", "1wk": "Semanal"}[x],
        index=0,
        help="Mensual es el estándar en la literatura académica de finanzas.",
    )

    nivel_asistente = st.selectbox(
        "Nivel del asistente explicativo",
        options=["basico", "tecnico"],
        format_func=lambda x: {"basico": "Básico (lenguaje accesible)", "tecnico": "Técnico (términos académicos)"}[x],
        index=0,
    )

    st.divider()

    analizar = st.button(
        "🚀 Analizar portafolio",
        type="primary",
        width="stretch",
    )

    if analizar:
        # Parsear tickers
        raw = tickers_input.replace(",", "\n").replace(";", "\n")
        tickers = [t.strip().upper() for t in raw.splitlines() if t.strip()]

        if len(tickers) < MIN_ACTIVOS:
            st.error(f"Se necesitan al menos {MIN_ACTIVOS} tickers.")
        elif len(tickers) > MAX_ACTIVOS:
            st.error(f"Máximo {MAX_ACTIVOS} tickers permitidos.")
        else:
            # Guardar parámetros en session_state
            st.session_state["tickers_usuario"]   = tickers
            st.session_state["ventana_anos"]       = ventana_anos
            st.session_state["frecuencia"]         = frecuencia
            st.session_state["nivel_asistente"]    = nivel_asistente

            # Resetear resultados anteriores
            for key in _KEYS_RESULTADOS:
                st.session_state[key] = None
            for fase in st.session_state["fase_completada"]:
                st.session_state["fase_completada"][fase] = False
            st.session_state["pipeline_ejecutado"] = False

            st.success(f"Portafolio configurado: {tickers}")
            st.info("Ve a **📥 1 Datos** en el menú para iniciar el análisis.")

    st.divider()
    st.caption(
        f"QuantαfolyΩ · Datos: Yahoo Finance · "
        f"Benchmark: {TICKER_BENCHMARK} · Divisa base: {DIVISA_BASE}"
    )
