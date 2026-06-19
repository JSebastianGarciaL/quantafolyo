# =============================================================================
# QuantαfolyΩ — modulos/errores.py
# Interpreta errores técnicos y los convierte en mensajes amigables.
# Uso: from modulos.errores import interpretar_error
#      mensaje = interpretar_error(error_str, fase="datos")
#      st.error(mensaje)
# =============================================================================

# Mapa de patrones → (título, causa probable, acción sugerida)
_PATRONES = [

    # --- Conectividad ---
    (["connectionerror", "connection error", "timeout", "timed out",
      "network", "urlopen", "remotedisconnected", "ssl"],
     "No se pudo conectar a internet",
     "La descarga de datos requiere conexión activa.",
     "Verifica tu conexión y vuelve a intentarlo."),

    # --- Yahoo Finance: ticker inválido ---
    (["no data found", "no timezone found", "possibly delisted",
      "ticker", "symbol", "yfinance", "not found", "invalid"],
     "Ticker no reconocido por Yahoo Finance",
     "El símbolo puede estar mal escrito, ser de un mercado diferente "
     "o haber sido retirado de cotización.",
     "Verifica el formato: AAPL (EE.UU.), SAP.DE (Alemania), 7203.T (Japón), "
     "EC (Ecopetrol, ADR NYSE)."),

    # --- Datos insuficientes ---
    (["insufficient", "not enough", "too few", "minimum", "mínimo",
      "observaciones", "shape[0]", "index 0 is out of bounds"],
     "Datos insuficientes para el análisis",
     "La ventana temporal es demasiado corta o el activo tiene poca historia disponible.",
     "Amplía la ventana temporal (mínimo recomendado: 5 años) "
     "o reemplaza el activo por uno con más historia."),

    # --- Datos faltantes / NaN ---
    (["nan", "na values", "missing", "dropna", "all-nan", "faltantes"],
     "Demasiados datos faltantes en uno o más activos",
     "Algunos activos tienen períodos sin cotización "
     "(suspensión, baja liquidez o inicio reciente).",
     "Reduce la ventana temporal o elimina los activos con más datos faltantes."),

    # --- Matriz singular / no invertible ---
    (["singular", "singular matrix", "not invertible", "linalg",
      "linalgError", "positive definite", "not positive"],
     "Matriz de covarianza no válida para optimización",
     "Dos o más activos están altamente correlacionados "
     "o tienen retornos idénticos en el período.",
     "Elimina activos muy similares entre sí (ej: dos ETFs del mismo índice) "
     "o amplía la ventana temporal."),

    # --- Optimización infactible ---
    (["infeasible", "optimal", "solver", "cvxpy", "problem",
      "unbounded", "status: infeasible"],
     "La optimización de portafolio no encontró solución",
     "Las restricciones de pesos son incompatibles con los datos "
     "o la matriz de covarianza es inestable.",
     "Intenta con menos activos, amplía la ventana temporal "
     "o revisa config.py (PESO_MINIMO_ACTIVO, PESO_MAXIMO_ACTIVO)."),

    # --- Ken French / FF3 ---
    (["french", "fama", "zip", "famafrench", "ken french",
      "http error", "403", "404"],
     "No se pudo descargar el modelo Fama-French",
     "El servidor de Ken French Library no respondió o cambió su estructura.",
     "El análisis continuará usando proxies ETF (SPY/IWM/IVE/IVW) "
     "como aproximación de los factores SMB y HML."),

    # --- APT / PCA ---
    (["pca", "components", "varianza", "variance", "apt",
      "n_components", "factores macro"],
     "El modelo APT no encontró factores significativos",
     "Los factores macro candidatos no tienen poder explicativo suficiente "
     "en este período o hay multicolinealidad severa.",
     "Prueba con una ventana temporal diferente o un conjunto distinto de activos."),

    # --- Regresión / MCO ---
    (["regression", "regresion", "ols", "mco", "converge",
      "perfect multicollinearity", "rank"],
     "Error en la estimación del modelo de regresión",
     "Multicolinealidad perfecta o datos insuficientes para MCO.",
     "Verifica que los activos tengan varianza positiva "
     "y que la ventana temporal sea suficiente (mínimo 50 observaciones)."),

    # --- Backtesting / VaR ---
    (["backtesting", "violations", "kupiec", "christoffersen",
      "var", "cvar", "quantile"],
     "Error en el cálculo de VaR o backtesting",
     "Pocas observaciones para estimar cuantiles con fiabilidad.",
     "Amplía la ventana temporal. Se recomiendan al menos 60 observaciones "
     "para backtesting mensual."),

    # --- Tasa libre de riesgo ---
    (["irx", "risk.free", "rf", "tasa libre", "treasury", "^irx"],
     "No se pudo obtener la tasa libre de riesgo",
     "Yahoo Finance no devolvió datos del T-Bill (^IRX) para el período.",
     "El análisis usará rf = 0% como fallback conservador. "
     "Los ratios de Sharpe pueden estar ligeramente sobreestimados."),

    # --- Memoria / tamaño ---
    (["memory", "memoryerror", "out of memory", "ram"],
     "Memoria insuficiente para completar el análisis",
     "Monte Carlo o la frontera eficiente requieren demasiada memoria "
     "con este número de activos.",
     "Reduce el número de activos (máximo recomendado: 15) "
     "o reinicia la aplicación."),

    # --- Errores de tipo / formato ---
    (["typeerror", "valueerror", "keyerror", "attributeerror",
      "indexerror", "assertionerror"],
     "Error interno de formato de datos",
     "Un resultado intermedio tiene un formato inesperado. "
     "Puede ocurrir si se cambió la frecuencia o los tickers entre ejecuciones.",
     "Vuelve a la página de inicio, presiona **Analizar portafolio** de nuevo "
     "y re-ejecuta todas las fases en orden."),
]

# Mensaje de fallback para errores no reconocidos
_FALLBACK = (
    "Error inesperado en el análisis",
    "Ocurrió un problema no identificado.",
    "Vuelve a la página de inicio, presiona **Analizar portafolio** "
    "y re-ejecuta todas las fases en orden. "
    "Si el problema persiste, revisa el log del proceso para más detalles.",
)

# Contexto adicional por fase
_CONTEXTO_FASE = {
    "datos":        "**Fase 1 — Datos:**",
    "estadistico":  "**Fase 2 — Estadístico:**",
    "markowitz":    "**Fase 3a — Markowitz:**",
    "factores":     "**Fase 3b — Factores:**",
    "riesgo":       "**Fase 3c — Riesgo:**",
    "verificacion": "**Verificación:**",
    "reporte":      "**Reporte:**",
}


def interpretar_error(error_str: str, fase: str = "") -> str:
    """
    Convierte un mensaje de error técnico en un mensaje amigable.

    Parámetros
    ----------
    error_str : str
        El texto del error capturado (resultado['error']).
    fase : str
        Identificador de la fase donde ocurrió el error.
        Valores válidos: 'datos', 'estadistico', 'markowitz',
        'factores', 'riesgo', 'verificacion', 'reporte'.

    Retorna
    -------
    str
        Mensaje formateado en Markdown con título, causa y acción.
    """
    if not error_str:
        error_str = ""

    error_lower = str(error_str).lower()
    titulo, causa, accion = _FALLBACK

    for patrones, t, c, a in _PATRONES:
        if any(p in error_lower for p in patrones):
            titulo, causa, accion = t, c, a
            break

    prefijo = _CONTEXTO_FASE.get(fase, "")
    separador = " " if prefijo else ""

    mensaje = (
        f"{prefijo}{separador}**{titulo}**\n\n"
        f"🔍 **Causa probable:** {causa}\n\n"
        f"✅ **Qué hacer:** {accion}"
    )

    return mensaje


def interpretar_error_con_detalle(error_str: str, fase: str = "") -> tuple[str, str]:
    """
    Igual que interpretar_error pero retorna (mensaje_amigable, detalle_tecnico)
    para mostrar el detalle en un expander opcional.

    Retorna
    -------
    tuple[str, str]
        (mensaje_amigable, error_original)
    """
    return interpretar_error(error_str, fase), str(error_str)
