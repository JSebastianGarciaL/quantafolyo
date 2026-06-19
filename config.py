# =============================================================================
# QuantαfolyΩ — config.py
# Fuente única de verdad para todos los parámetros globales.
# Todos los módulos importan desde aquí. Ningún valor se hardcodea en otro archivo.
# =============================================================================

# --- Estadístico ---
ALPHA = 0.05                        # Nivel de significancia estándar (convención académica)

# --- Divisa y mercado ---
DIVISA_BASE          = "USD"        # Todos los retornos se expresan en USD
TICKER_BENCHMARK     = "^GSPC"      # S&P 500 — benchmark global más usado
TICKER_TASA_LR       = "^IRX"       # T-Bill 3 meses — proxy tasa libre de riesgo CAPM

# --- Frecuencia y ventana ---
FRECUENCIA_DEFAULT      = "1mo"     # Mensual — estándar en literatura académica
VENTANA_DEFAULT_ANOS    = 5         # Ventana de 5 años por defecto

# Factor de anualización por frecuencia
# Se usa FACTOR_ANUALIZACION como valor por defecto (mensual).
# pipeline_datos() calcula el factor correcto y lo guarda en metadatos['factor_anualizacion']
# Los módulos deben leer ese valor desde metadatos, no usar esta constante directamente.
FACTOR_ANUALIZACION     = 12        # Mensual → anual
FACTORES_ANUALIZACION   = {
    "1mo": 12,   # mensual
    "1wk": 52,   # semanal
    "1d":  252,  # diaria
}

# --- Portafolio ---
MIN_ACTIVOS             = 2
MAX_ACTIVOS             = 20
PERMITIR_VENTAS_CORTO     = False   # Restricción de no short-selling (más realista)
PUNTOS_FRONTERA_EFICIENTE = 100     # Resolución de la frontera eficiente
PESO_MINIMO_ACTIVO        = 0.0     # Peso mínimo por activo (0 = sin límite inferior)
PESO_MAXIMO_ACTIVO        = 1.0     # Peso máximo por activo (1 = sin límite superior)
TICKER_TASA_LIBRE_RIESGO  = "^IRX"  # Alias explícito para descarga de rf en Markowitz

# --- Calidad de datos ---
UMBRAL_DATOS_FALTANTES  = 0.05      # Máximo 5% de NaN por activo
UMBRAL_OUTLIER_ZSCORE   = 3.0       # |Z| > 3 se reporta como outlier

# --- Observaciones mínimas por modelo ---
MIN_OBSERVACIONES_MARKOWITZ = 30    # Mínimo para estimación estable de Σ
MIN_OBSERVACIONES_CAPM      = 50    # Grados de libertad suficientes para MCO
MIN_OBSERVACIONES_APT       = 50    # Idem para regresión multivariada
MIN_OBSERVACIONES_GARCH     = 100   # GARCH necesita muestra grande (Capa 2)

# --- APT: Factores macro candidatos ---
# Chen, Roll & Ross (1986) — factores macroeconómicos que sistemáticamente
# afectan los retornos del mercado. Todos descargables desde Yahoo Finance.
FACTORES_MACRO_CANDIDATOS = {
    "^TNX":    "Tasa bono 10 años (EE.UU.)",
    "^IRX":    "Tasa T-Bill 3 meses",
    "CL=F":    "Petróleo crudo WTI (futuros)",
    "GC=F":    "Oro (futuros)",
    "DX-Y.NYB":"Índice dólar DXY",
    "^VIX":    "Índice de volatilidad VIX",
    "TIP":     "ETF bonos ligados a inflación (TIPS)",
    "HYG":     "ETF bonos high yield (spread crédito)",
    "XLB":     "ETF sector materiales",
    "XLU":     "ETF sector utilities",
}

# --- Modelo APT ---
VIF_UMBRAL               = 10.0    # Umbral de varianza inflada (multicolinealidad)
APT_VARIANZA_MINIMA_PCA  = 0.80    # % varianza acumulada mínima que deben explicar los PCs
APT_MIN_FACTORES         = 1       # Mínimo de componentes PCA a retener
APT_MAX_FACTORES         = 5       # Máximo de factores en el modelo APT por activo

# --- VaR ---
NIVEL_CONFIANZA_VAR          = 0.95   # Nivel de confianza estándar (95%)
NIVEL_CONFIANZA_VAR_ESTRICTO = 0.99   # Nivel estricto (99%)
N_SIMULACIONES_MC            = 10000  # Simulaciones Monte Carlo para VaR

# --- Sistema de color QuantαfolyΩ ---
# Dos paletas completas: una para modo claro (cálida) y otra para modo oscuro (fría).
# Las páginas deben leer get_colores() para obtener la paleta activa según el tema.
# Nunca hardcodear colores en módulos — siempre importar desde aquí.

COLORES_CLARO = {
    # Fondos
    "fondo_principal":      "#FDF6EE",   # Crema cálido — fondo de página
    "fondo_card":           "#F7E1D3",   # Terracota claro — cards y paneles

    # Acentos principales
    "acento_principal":     "#D76F02",   # Naranja quemado — portafolio tangente, CTA
    "acento_secundario":    "#DB9435",   # Ámbar — portafolio MVP
    "highlight":            "#ECD577",   # Amarillo suave — portafolio 1/N, líneas secundarias
    "contraste":            "#9FD0D6",   # Teal claro — benchmark S&P 500

    # Semántica
    "positivo":             "#4A7C59",   # Verde salvia — bueno, éxito
    "alerta":               "#985D73",   # Rosa malva — advertencia, alerta cálida
    "negativo":             "#B33000",   # Rojo oscuro cálido — error

    # Texto
    "texto_principal":      "#3B200B",   # Café oscuro — texto principal
    "texto_secundario":     "#576071",   # Gris azulado — texto secundario, labels
    "texto_muted":          "#AFA0A0",   # Gris rosado — texto desactivado

    # Gráficos Plotly — secuencia para múltiples activos
    "graf_seq": [
        "#D76F02",   # naranja quemado
        "#DB9435",   # ámbar
        "#985D73",   # rosa malva
        "#9FD0D6",   # teal
        "#ECD577",   # amarillo
        "#4A7C59",   # verde salvia
        "#AC6646",   # terracota
        "#576071",   # gris azulado
    ],
}

COLORES_OSCURO = {
    # Fondos
    "fondo_principal":      "#0B161E",   # Azul noche — fondo de página
    "fondo_card":           "#162e1a",   # Verde musgo oscuro — cards y paneles

    # Acentos principales
    "acento_principal":     "#97b261",   # Verde lima — portafolio tangente, CTA
    "acento_secundario":    "#739842",   # Verde medio — portafolio MVP
    "highlight":            "#c5d7d7",   # Teal claro — portafolio 1/N, líneas secundarias
    "contraste":            "#A6B9BE",   # Gris teal — benchmark S&P 500

    # Semántica
    "positivo":             "#437a38",   # Verde bosque — bueno, éxito
    "alerta":               "#CA6378",   # Rosa — advertencia (valor original aprobado en P2.2)
    "negativo":             "#E93C35",   # Rojo — error

    # Texto
    "texto_principal":      "#D8D2C6",   # Blanco cálido — texto principal
    "texto_secundario":     "#A6B9BE",   # Gris teal — texto secundario, labels
    "texto_muted":          "#51625C",   # Verde grisáceo — texto desactivado

    # Gráficos Plotly — secuencia para múltiples activos
    "graf_seq": [
        "#97b261",   # verde lima
        "#739842",   # verde medio
        "#CA6378",   # rosa oscuro
        "#c5d7d7",   # teal claro
        "#DEBAB1",   # salmón cálido
        "#A6B9BE",   # gris teal
        "#437a38",   # verde bosque
        "#51625C",   # verde grisáceo
    ],
}

# Alias para compatibilidad con módulos existentes (usan COLORES directamente).
# Apunta al modo claro por defecto — las páginas que detectan tema usan get_colores().
COLORES = {
    "portafolio_optimo":    COLORES_CLARO["acento_principal"],
    "minima_varianza":      COLORES_CLARO["acento_secundario"],
    "activos_individuales": COLORES_CLARO["highlight"],
    "benchmark":            COLORES_CLARO["contraste"],
}


def get_colores(tema: str = "light") -> dict:
    """
    Retorna la paleta de colores según el tema activo.

    Uso en páginas:
        import streamlit as st
        from config import get_colores
        colores = get_colores(st.get_option("theme.base") or "light")

    Parámetros
    ----------
    tema : str
        "light" o "dark". Cualquier otro valor retorna la paleta clara.

    Retorna
    -------
    dict con todas las claves de color para la paleta seleccionada.
    """
    return COLORES_OSCURO if tema == "dark" else COLORES_CLARO


def get_plotly_layout(tema: str = "light") -> dict:
    """
    Retorna un diccionario de layout Plotly consistente con el tema activo.

    Uso en páginas:
        from config import get_plotly_layout
        fig.update_layout(**get_plotly_layout(st.get_option("theme.base") or "light"))

    Parámetros
    ----------
    tema : str
        "light" o "dark".

    Retorna
    -------
    dict con configuración de layout Plotly lista para usar con update_layout().
    """
    c = get_colores(tema)

    grilla  = "rgba(87,96,113,0.15)" if tema == "light" else "rgba(81,98,92,0.20)"
    eje     = c["texto_secundario"]
    fuente  = c["texto_principal"]

    return dict(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(
            family = "Space Grotesk, Inter, sans-serif",
            color  = fuente,
            size   = 12,
        ),
        xaxis = dict(
            gridcolor    = grilla,
            linecolor    = grilla,
            tickcolor    = eje,
            tickfont     = dict(color=eje, size=11),
            title_font   = dict(color=eje, size=12),
            showgrid     = True,
            zeroline     = False,
        ),
        yaxis = dict(
            gridcolor    = grilla,
            linecolor    = grilla,
            tickcolor    = eje,
            tickfont     = dict(color=eje, size=11),
            title_font   = dict(color=eje, size=12),
            showgrid     = True,
            zeroline     = False,
        ),
        hoverlabel = dict(
            bgcolor     = c["fondo_card"],
            font_size   = 12,
            font_family = "Space Grotesk, Inter, sans-serif",
        ),
    )
