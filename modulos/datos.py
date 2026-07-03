# =============================================================================
# QuantαfolyΩ — modulos/datos.py
# Fase 1: Módulo de Datos
#
# Funciones migradas desde el notebook PortfolioLab_V0_7_0.ipynb (celdas 5–18).
# REGLA DE MIGRACIÓN: la lógica matemática y financiera no se modificó.
# Cambios realizados:
#   - print() → entradas en la lista `logs` que la función retorna
#   - RUTA_PROYECTO / Drive / google.colab → eliminados
#   - Variables globales de config.py → recibidas como parámetros o importadas
#   - savefig/plt → eliminados (el gráfico de verificación vive en la página)
#   - pipeline_datos() envuelve todo el flujo en una sola llamada
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import (
    DIVISA_BASE,
    TICKER_BENCHMARK,
    TICKER_TASA_LR,
    UMBRAL_DATOS_FALTANTES,
    UMBRAL_OUTLIER_ZSCORE,
    MIN_ACTIVOS,
    MAX_ACTIVOS,
    MIN_OBSERVACIONES_MARKOWITZ,
    MIN_OBSERVACIONES_CAPM,
    MIN_OBSERVACIONES_APT,
    MIN_OBSERVACIONES_GARCH,
    FACTORES_ANUALIZACION,
)


# =============================================================================
# FUNCIÓN 1 — Validación de tickers
# =============================================================================

def validar_tickers(lista_tickers: list) -> tuple[dict, list]:
    """
    Valida que los tickers existen en Yahoo Finance y extrae su información básica.

    Parámetros
    ----------
    lista_tickers : list
        Lista de strings con los tickers a validar.

    Retorna
    -------
    resultado : dict
        {'validos': dict {ticker: info}, 'invalidos': list}
    logs : list[str]
        Mensajes informativos para renderizar en la UI.
    """
    validos   = {}
    invalidos = []
    logs      = []

    logs.append("Validando tickers...")

    for ticker in lista_tickers:
        ticker = ticker.strip().upper()
        try:
            info = yf.Ticker(ticker).info

            # Yahoo Finance devuelve dict vacío o sin precio cuando el ticker no existe
            precio  = info.get('regularMarketPrice') or info.get('currentPrice')
            nombre  = info.get('longName') or info.get('shortName') or 'Sin nombre'
            # CORRECCIÓN v0.2.1: leer 'currency' directamente del objeto info de Yahoo Finance
            divisa  = info.get('currency', 'USD')
            mercado = info.get('exchange', 'N/D')

            if precio is None and nombre == 'Sin nombre':
                raise ValueError("Ticker no encontrado")

            validos[ticker] = {
                'nombre':   nombre,
                'currency': divisa,   # guardado como 'currency' (nombre original del campo)
                'mercado':  mercado,
                'info':     info,
            }
            logs.append(f"{ticker:<15} {nombre:<40} {divisa:<8} {mercado}")

        except Exception:
            invalidos.append(ticker)
            logs.append(f"❌ {ticker:<15} No encontrado en Yahoo Finance")

    logs.append(f"\nResultado: {len(validos)} válidos, {len(invalidos)} inválidos.")

    if invalidos:
        logs.append(f"Tickers no encontrados: {invalidos}")
        logs.append("Verifica el formato: AAPL (EE.UU.), EC (Ecopetrol, ADR NYSE), "
                    "SAP.DE (Alemania), 7203.T (Japón)")

    resultado = {'validos': validos, 'invalidos': invalidos}
    return resultado, logs


# =============================================================================
# FUNCIÓN 2 — Descarga de precios
# =============================================================================

def descargar_precios(tickers: list, fecha_inicio: datetime,
                      fecha_fin: datetime, frecuencia: str) -> tuple[pd.DataFrame, list]:
    """
    Descarga precios de cierre ajustados desde Yahoo Finance.
    Compatible con yfinance >= 0.2.x (maneja MultiIndex).

    Retorna
    -------
    precios : pd.DataFrame
        Columnas = tickers, índice = fechas.
    logs : list[str]
    """
    todos_precios    = {}
    descarga_fallida = []
    logs             = []

    logs.append("Descargando precios...")

    for ticker in tickers:
        try:
            datos = yf.download(
                ticker,
                start=fecha_inicio,
                end=fecha_fin,
                interval=frecuencia,
                auto_adjust=True,
                progress=False,
            )

            if datos.empty:
                raise ValueError("Sin datos")

            # Compatibilidad con MultiIndex de yfinance >= 0.2.x
            if isinstance(datos.columns, pd.MultiIndex):
                serie = datos[('Close', ticker)]
            else:
                serie = datos['Close']

            # Asegurarse de que es una Serie unidimensional
            if isinstance(serie, pd.DataFrame):
                serie = serie.squeeze()

            todos_precios[ticker] = serie
            logs.append(f"  {ticker:<20} {len(serie.dropna())} observaciones descargadas")

        except Exception as e:
            descarga_fallida.append(ticker)
            logs.append(f"  {ticker:<20} ERROR: {str(e)}")

    if descarga_fallida:
        logs.append(f"Fallaron: {descarga_fallida}")

    precios = pd.DataFrame(todos_precios)
    precios.index = pd.to_datetime(precios.index)
    precios.sort_index(inplace=True)

    return precios, logs


# =============================================================================
# FUNCIÓN 3 — Detección de divisas
# =============================================================================

def detectar_divisas(resultado_validacion: dict) -> tuple[dict, list]:
    """
    Extrae la divisa de cada activo desde la información guardada en validar_tickers.
    Lee el campo 'currency' (nombre original del campo en Yahoo Finance).

    CORRECCIÓN v0.2.1: se usa 'currency' en lugar de 'divisa' para ser
    consistente con el nombre original del campo en la API de Yahoo Finance.

    Retorna
    -------
    divisas : dict {ticker: str}
    logs : list[str]
    """
    divisas = {}
    logs    = []

    for ticker, info in resultado_validacion['validos'].items():
        divisa = info.get('currency', 'USD')
        if divisa is None or divisa == '':
            divisa = 'USD'
            logs.append(f"  ADVERTENCIA: {ticker} sin divisa detectada. Asumiendo USD.")
        divisas[ticker] = divisa

    logs.append("Divisas detectadas:")
    for ticker, divisa in divisas.items():
        estado = "OK (ya en USD)" if divisa == DIVISA_BASE else "Requiere conversión a USD"
        logs.append(f"  {ticker:<20} {divisa:<8} {estado}")

    return divisas, logs


# =============================================================================
# FUNCIÓN 4 — Conversión a USD
# =============================================================================

def convertir_a_usd(precios: pd.DataFrame, divisas: dict,
                    fecha_inicio: datetime, fecha_fin: datetime,
                    frecuencia: str) -> tuple[pd.DataFrame, list]:
    """
    Convierte todos los precios a USD usando tipos de cambio históricos de Yahoo Finance.

    Para cada activo en divisa distinta a USD, descarga el par de tipo de cambio
    correcto según la convención real de Yahoo Finance e invierte si corresponde
    para obtener siempre divisa→USD, luego multiplica precio_local × tipo_de_cambio.

    Formato Yahoo Finance: EURUSD=X, GBPUSD=X (cotizadas contra USD) vs.
    USDCOP=X, USDJPY=X (cotizadas con USD como base — se invierten internamente).

    Retorna
    -------
    precios_usd : pd.DataFrame
    logs : list[str]
    """
    precios_usd         = precios.copy()
    divisas_a_convertir = {t: d for t, d in divisas.items() if d != DIVISA_BASE}
    logs                = []

    if not divisas_a_convertir:
        logs.append("Todos los activos ya están en USD. No se requiere conversión.")
        return precios_usd, logs

    pares_necesarios = set(divisas_a_convertir.values())
    logs.append(f"Descargando tipos de cambio históricos para: {pares_necesarios}")

    # CORRECCIÓN (auditoría P1.2→P3.1): convención real de Yahoo Finance.
    # EUR, GBP, AUD, NZD se cotizan como {DIVISA}USD=X (cuántos USD vale 1 unidad).
    # El resto de divisas se cotiza como USD{DIVISA}=X (cuántas unidades vale 1 USD)
    # y hay que invertir el tipo de cambio descargado.
    # IMPORTANTE: verificar con datos reales antes de confiar ciegamente en esto —
    # no se pudo probar en vivo durante la auditoría por restricciones de red.
    MONEDAS_COTIZADAS_CONTRA_USD = {"EUR", "GBP", "AUD", "NZD"}

    tipos_cambio = {}
    for divisa in pares_necesarios:
        invertir = divisa not in MONEDAS_COTIZADAS_CONTRA_USD
        par = f"USD{divisa}=X" if invertir else f"{divisa}USD=X"
        try:
            tc, _ = descargar_precios([par], fecha_inicio, fecha_fin, frecuencia)
            if not tc.empty:
                serie_tc = tc.iloc[:, 0]
                if invertir:
                    serie_tc = 1 / serie_tc
                tipos_cambio[divisa] = serie_tc
                logs.append(f"  {par:<15} descargado correctamente"
                            + (" (invertido a divisa→USD)" if invertir else ""))
            else:
                logs.append(f"  {par:<15} SIN DATOS — activo quedará en divisa original")
                logs.append(f"  ⚠️ Sin conversión, los retornos de activos en {divisa} "
                            "no son comparables con los demás.")
        except Exception as e:
            logs.append(f"  {par:<15} ERROR: {e}")
            logs.append(f"  ⚠️ No se pudo convertir activos en {divisa} a USD.")

    logs.append("Aplicando conversión:")
    for ticker, divisa in divisas_a_convertir.items():
        if divisa in tipos_cambio:
            tc_serie = tipos_cambio[divisa]
            precios_ticker, tc_alineado = precios_usd[ticker].align(tc_serie, join='inner')
            precios_usd.loc[precios_ticker.index, ticker] = (
                precios_ticker * tc_alineado
            ).values
            logs.append(f"  {ticker:<20} {divisa} → USD: conversión aplicada")
        else:
            logs.append(f"  {ticker:<20} {divisa} → Sin tipo de cambio disponible.")
            logs.append(f"  ⚠️ {ticker} mantiene precios en {divisa}. "
                        "Sus retornos NO son comparables con activos en USD.")

    return precios_usd, logs


# =============================================================================
# FUNCIÓN 5 — Retornos logarítmicos
# =============================================================================

def calcular_retornos_log(precios: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Calcula retornos logarítmicos a partir de precios ajustados.
    r_t = ln(P_t) - ln(P_{t-1})
    La primera observación siempre es NaN y se elimina.

    Base teórica: Campbell, Lo & MacKinlay (1997) — aditividad temporal de
    retornos logarítmicos permite anualizar multiplicando por el factor.

    Retorna
    -------
    retornos : pd.DataFrame
    logs : list[str]
    """
    logs     = []
    # Eliminar columnas completamente vacías antes de calcular retornos
    cols_antes = list(precios.columns)
    precios    = precios.dropna(axis=1, how='all')
    cols_vacias = [c for c in cols_antes if c not in precios.columns]
    if cols_vacias:
        logs.append(f"⚠️ Activos sin datos descartados antes de calcular retornos: {cols_vacias}")

    retornos = np.log(precios / precios.shift(1))
    retornos = retornos.dropna(how='all')

    # Reportar activos que perdieron muchas observaciones (inicio tardío)
    n_total = len(retornos)
    for col in retornos.columns:
        n_validos = retornos[col].count()
        pct_faltante = 1 - n_validos / n_total
        if pct_faltante > 0.20:  # más del 20% de NaN = activo con historia corta
            logs.append(f"⚠️ {col}: solo {n_validos}/{n_total} observaciones válidas "
                        f"({pct_faltante*100:.0f}% faltante) — activo con historia más corta que la ventana solicitada")

    logs.append(f"Retornos logarítmicos calculados.")
    logs.append(f"Dimensiones: {retornos.shape[0]} periodos × {retornos.shape[1]} activos")

    return retornos, logs


# =============================================================================
# FUNCIÓN 6 — Validación de datos faltantes
# =============================================================================

def validar_datos_faltantes(retornos: pd.DataFrame,
                             umbral: float) -> tuple[pd.DataFrame, list, list]:
    """
    Detecta activos con datos faltantes excesivos.
    Los datos NO se modifican — se reportan para decisión del usuario.

    Retorna
    -------
    reporte : pd.DataFrame
    activos_problematicos : list
    logs : list[str]
    """
    total_obs = len(retornos)
    reporte   = pd.DataFrame({
        'total_obs':       total_obs,
        'datos_validos':   retornos.count(),
        'datos_faltantes': retornos.isnull().sum(),
        'pct_faltantes':   (retornos.isnull().sum() / total_obs).round(4),
    })

    logs                 = []
    activos_problematicos = []

    logs.append("Reporte de datos faltantes:")
    for ticker, row in reporte.iterrows():
        estado = "OK" if row['pct_faltantes'] <= umbral else "⚠️ ADVERTENCIA"
        if row['pct_faltantes'] > umbral:
            activos_problematicos.append(ticker)
        logs.append(
            f"  {ticker:<20} {row['datos_validos']:>6} válidos  "
            f"{row['pct_faltantes']*100:>5.1f}% faltante  {estado}"
        )

    if activos_problematicos:
        logs.append(f"\n⚠️ Activos con datos insuficientes: {activos_problematicos}")
        logs.append(f"Tienen más del {umbral*100:.0f}% de datos faltantes.")
        logs.append("Se eliminarán del análisis automáticamente.")

    return reporte, activos_problematicos, logs


# =============================================================================
# FUNCIÓN 7 — Detección de outliers
# =============================================================================

def detectar_outliers(retornos: pd.DataFrame,
                      umbral_z: float) -> tuple[pd.DataFrame, list]:
    """
    Detecta retornos extremos usando Z-score.
    |z| > umbral_z se considera outlier.
    NO se eliminan automáticamente — se reportan para contexto del usuario.

    Los outliers pueden ser reales (crisis, dividendos extraordinarios)
    o errores de datos. El contexto histórico determina cuál es cuál.

    Retorna
    -------
    outliers_mask : pd.DataFrame (booleano)
    logs : list[str]
    """
    z_scores      = (retornos - retornos.mean()) / retornos.std()
    outliers_mask = z_scores.abs() > umbral_z
    n_outliers    = outliers_mask.sum()
    logs          = []

    logs.append(f"Detección de outliers (|Z| > {umbral_z}):")

    hay_outliers = False
    for ticker in retornos.columns:
        if n_outliers[ticker] > 0:
            hay_outliers   = True
            fechas_outlier = retornos.index[outliers_mask[ticker]]
            valores        = retornos[ticker][outliers_mask[ticker]]
            logs.append(f"  {ticker}: {n_outliers[ticker]} outlier(s)")
            for fecha, valor in zip(fechas_outlier, valores):
                z = z_scores[ticker][fecha]
                logs.append(f"    {fecha.strftime('%Y-%m')}: retorno={valor:.4f}, Z={z:.2f}")

    if not hay_outliers:
        logs.append("  No se detectaron outliers significativos.")
    else:
        logs.append("NOTA: Los outliers no se eliminaron automáticamente.")
        logs.append("Verifique si corresponden a eventos reales (crisis, splits, etc.)")

    return outliers_mask, logs


# =============================================================================
# FUNCIÓN 8 — Verificación de observaciones mínimas
# =============================================================================

def verificar_observaciones_minimas(retornos: pd.DataFrame) -> tuple[bool, list]:
    """
    Verifica que el dataset tiene suficientes observaciones para cada modelo.
    Reporta cuáles modelos son aplicables con los datos actuales.

    Retorna
    -------
    todos_ok : bool
    logs : list[str]
    """
    n          = len(retornos)
    requisitos = {
        'Markowitz (MVP)':        MIN_OBSERVACIONES_MARKOWITZ,
        'CAPM':                   MIN_OBSERVACIONES_CAPM,
        'Fama-French 3 factores': MIN_OBSERVACIONES_CAPM,
        'APT con factores macro': MIN_OBSERVACIONES_APT,
        'GARCH (Capa 2)':         MIN_OBSERVACIONES_GARCH,
    }

    logs     = []
    todos_ok = True

    logs.append(f"Observaciones disponibles: {n}\n")
    for modelo, minimo in requisitos.items():
        estado = "✅ APLICABLE" if n >= minimo else "⚠️ INSUFICIENTE"
        if n < minimo:
            todos_ok = False
        logs.append(f"  {modelo:<35} mín={minimo:>4}  actual={n:>4}  {estado}")

    if not todos_ok:
        logs.append("\nSugerencia: ampliar la ventana temporal para cumplir todos los requisitos.")

    return todos_ok, logs


# =============================================================================
# FUNCIÓN PIPELINE — Envuelve todo el flujo de la Fase 1
# =============================================================================

def pipeline_datos(tickers: list, ventana_anos: int, frecuencia: str) -> dict:
    """
    Ejecuta el flujo completo de la Fase 1:
      1. Validar tickers
      2. Calcular fechas
      3. Descargar precios
      4. Detectar divisas
      5. Convertir a USD
      6. Calcular retornos logarítmicos (activos + benchmark)
      7. Validar datos faltantes y limpiar
      8. Detectar outliers
      9. Verificar observaciones mínimas
      10. Construir metadatos

    Parámetros
    ----------
    tickers : list[str]
        Lista de tickers del portafolio (ya validados en el sidebar de app.py).
    ventana_anos : int
        Años de historia a descargar.
    frecuencia : str
        Intervalo de Yahoo Finance, e.g. '1mo', '1wk'.

    Retorna
    -------
    dict con claves:
        'retornos'              : pd.DataFrame — retornos log mensuales en USD
        'retornos_benchmark'    : pd.DataFrame — retornos log del S&P 500
        'precios_usd'           : pd.DataFrame — precios históricos en USD
        'metadatos'             : dict — info del dataset
        'resultado_validacion'  : dict — tickers válidos/inválidos
        'divisas_activos'       : dict — divisa de cada activo
        'outliers_mask'         : pd.DataFrame — máscara de outliers
        'reporte_faltantes'     : pd.DataFrame
        'datos_suficientes'     : bool
        'logs'                  : list[str] — todos los mensajes del pipeline
        'error'                 : str | None — si ocurrió un error fatal
    """
    logs_totales = []

    def _log(msgs):
        if isinstance(msgs, list):
            logs_totales.extend(msgs)
        else:
            logs_totales.append(msgs)

    try:
        # --- 1. Fechas ---
        fecha_fin    = datetime.today()
        fecha_inicio = fecha_fin - timedelta(days=365 * ventana_anos)
        _log(f"Período: {fecha_inicio.strftime('%Y-%m-%d')} → {fecha_fin.strftime('%Y-%m-%d')}")

        # --- 2. Validar tickers ---
        _log("=" * 50)
        _log("PASO 1 — Validación de tickers")
        _log("=" * 50)
        resultado_validacion, lg = validar_tickers(tickers)
        _log(lg)
        tickers_validos = list(resultado_validacion['validos'].keys())

        if len(tickers_validos) < MIN_ACTIVOS:
            return {
                'error': f"Solo {len(tickers_validos)} tickers válidos. "
                         f"Se necesitan al menos {MIN_ACTIVOS}.",
                'logs': logs_totales,
            }

        # --- 3. Descargar precios activos ---
        _log("=" * 50)
        _log("PASO 2 — Descarga de precios")
        _log("=" * 50)
        precios_raw, lg = descargar_precios(tickers_validos, fecha_inicio, fecha_fin, frecuencia)
        _log(lg)

        if precios_raw.empty:
            return {'error': "No se descargaron datos de precios.", 'logs': logs_totales}

        # --- 4. Descargar benchmark ---
        benchmark_raw, lg = descargar_precios(
            [TICKER_BENCHMARK], fecha_inicio, fecha_fin, frecuencia
        )
        _log(lg)
        benchmark_raw.columns = ['SP500']

        # --- 5. Detectar divisas ---
        _log("=" * 50)
        _log("PASO 3 — Detección y conversión de divisas")
        _log("=" * 50)
        divisas_activos, lg = detectar_divisas(resultado_validacion)
        _log(lg)

        # --- 6. Convertir a USD ---
        precios_usd, lg = convertir_a_usd(
            precios_raw, divisas_activos, fecha_inicio, fecha_fin, frecuencia
        )
        _log(lg)

        # --- 7. Calcular retornos ---
        _log("=" * 50)
        _log("PASO 4 — Retornos logarítmicos")
        _log("=" * 50)
        retornos, lg = calcular_retornos_log(precios_usd)
        _log(lg)

        # Verificar que queden suficientes activos con datos después del cálculo de retornos
        if retornos.shape[1] < MIN_ACTIVOS:
            return {
                'error': (
                    f"Solo {retornos.shape[1]} activo con datos válidos tras la conversión de divisas. "
                    f"Se necesitan al menos {MIN_ACTIVOS}.\n"
                    f"Verifica que los activos tengan historia suficiente en la ventana seleccionada."
                ),
                'logs': logs_totales,
            }

        retornos_benchmark, lg = calcular_retornos_log(benchmark_raw)
        _log(lg)
        retornos_benchmark.columns = ['SP500']

        # --- 8. Validar y limpiar ---
        _log("=" * 50)
        _log("PASO 5 — Calidad de datos")
        _log("=" * 50)
        reporte_faltantes, activos_problematicos, lg = validar_datos_faltantes(
            retornos, UMBRAL_DATOS_FALTANTES
        )
        _log(lg)

        if activos_problematicos:
            retornos = retornos.drop(columns=activos_problematicos)
            _log(f"Activos eliminados por datos insuficientes: {activos_problematicos}")
            _log(f"Portafolio final: {list(retornos.columns)}")

        # Verificar que queden suficientes activos después de la limpieza
        if retornos.shape[1] < MIN_ACTIVOS:
            return {
                'error': (
                    f"Después de eliminar activos con datos insuficientes, "
                    f"solo queda {retornos.shape[1]} activo. "
                    f"Se necesitan al menos {MIN_ACTIVOS} para el análisis.\n\n"
                    f"Activos eliminados: {activos_problematicos}\n"
                    f"Soluciones posibles:\n"
                    f"  • Agrega más activos al portafolio\n"
                    f"  • Reduce la ventana temporal\n"
                    f"  • Usa activos con mayor historia disponible"
                ),
                'logs': logs_totales,
            }

        # --- 9. Outliers ---
        outliers_mask, lg = detectar_outliers(retornos, UMBRAL_OUTLIER_ZSCORE)
        _log(lg)

        # --- 10. Observaciones mínimas ---
        datos_suficientes, lg = verificar_observaciones_minimas(retornos)
        _log(lg)

        # --- 11. Metadatos ---
        metadatos = {
            'fecha_generacion':     datetime.today().strftime('%Y-%m-%d %H:%M'),
            'activos':              list(retornos.columns),
            'n_activos':            int(retornos.shape[1]),
            'n_observaciones':      int(retornos.shape[0]),
            'fecha_inicio':         str(retornos.index[0].date()),
            'fecha_fin':            str(retornos.index[-1].date()),
            'frecuencia':           frecuencia,
            'ventana_anos':         ventana_anos,
            'divisa_base':          DIVISA_BASE,
            'benchmark':            TICKER_BENCHMARK,
            'tickers_invalidos':    resultado_validacion['invalidos'],
            'activos_eliminados':   activos_problematicos,
            'datos_suficientes':    datos_suficientes,
            'factor_anualizacion':  FACTORES_ANUALIZACION.get(frecuencia, 12),
            'fase':                 'Fase 1 completada — v0.7.0 → Streamlit',
        }

        _log("=" * 50)
        _log("✅ FASE 1 COMPLETADA")
        _log(f"Activos: {list(retornos.columns)}")
        _log(f"Observaciones: {retornos.shape[0]} periodos")
        _log("=" * 50)

        return {
            'retornos':             retornos,
            'retornos_benchmark':   retornos_benchmark,
            'precios_usd':          precios_usd,
            'metadatos':            metadatos,
            'resultado_validacion': resultado_validacion,
            'divisas_activos':      divisas_activos,
            'outliers_mask':        outliers_mask,
            'reporte_faltantes':    reporte_faltantes,
            'datos_suficientes':    datos_suficientes,
            'logs':                 logs_totales,
            'error':                None,
        }

    except Exception as e:
        import traceback
        _log(f"ERROR FATAL en pipeline_datos: {e}")
        _log(traceback.format_exc())
        return {'error': str(e), 'logs': logs_totales}
