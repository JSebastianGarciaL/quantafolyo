# =============================================================================
# QuantαfolyΩ — modulos/estadistico.py
# Fase 2: Análisis Estadístico Preliminar
#
# Funciones migradas desde PortfolioLab_V0_7_0.ipynb (celdas 22–31).
# REGLA DE MIGRACIÓN: lógica estadística sin cambios.
# Cambios:
#   - print() → entradas en lista `logs`
#   - asistente_fase2 retorna str en lugar de imprimir
#   - tabla_semaforo retorna DataFrame (emojis incluidos en columna)
#   - pipeline_estadistico() envuelve todo el flujo
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from arch.unitroot import PhillipsPerron

from config import ALPHA


# =============================================================================
# FUNCIÓN 1 — Estacionariedad (ADF + KPSS)
# =============================================================================

def prueba_estacionariedad(retornos: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, list]:
    """
    ADF + KPSS por activo con diagnóstico conjunto.

    ADF (Dickey & Fuller, 1979):   H0 = raíz unitaria (NO estacionaria)
    KPSS (Kwiatkowski et al., 1992): H0 = ES estacionaria

    Lógica de decisión conjunta:
      ADF rechaza  + KPSS no rechaza → ESTACIONARIA
      ADF rechaza  + KPSS rechaza    → POSIBLE QUIEBRE ESTRUCTURAL
      ADF no rechaza + KPSS no rechaza → NO CONCLUYENTE
      ADF no rechaza + KPSS rechaza  → NO ESTACIONARIA
    """
    resultados = []
    logs       = ["PRUEBA 1 — ESTACIONARIEDAD (ADF + KPSS)",
                  f"α = {alpha} | H0 ADF: raíz unitaria | H0 KPSS: estacionaria"]

    for ticker in retornos.columns:
        serie = retornos[ticker].dropna()

        adf_stat, adf_p, _, _, _, _ = adfuller(serie, regression='c', autolag='AIC')
        adf_rechaza = adf_p < alpha

        kpss_stat, kpss_p, _, _ = kpss(serie, regression='c', nlags='auto')
        kpss_rechaza = kpss_p < alpha

        if adf_rechaza and not kpss_rechaza:
            diagnostico = "ESTACIONARIA"
            color       = "VERDE"
        elif adf_rechaza and kpss_rechaza:
            diagnostico = "POSIBLE QUIEBRE ESTRUCTURAL"
            color       = "AMARILLO"
        elif not adf_rechaza and not kpss_rechaza:
            diagnostico = "NO CONCLUYENTE"
            color       = "AMARILLO"
        else:
            diagnostico = "NO ESTACIONARIA"
            color       = "ROJO"

        resultados.append({
            'Ticker':       ticker,
            'ADF_stat':     round(adf_stat, 4),
            'ADF_pvalue':   round(adf_p, 4),
            'ADF_rechaza':  adf_rechaza,
            'KPSS_stat':    round(kpss_stat, 4),
            'KPSS_pvalue':  round(kpss_p, 4),
            'KPSS_rechaza': kpss_rechaza,
            'Diagnostico':  diagnostico,
            'Semaforo':     color,
        })

        logs.append(
            f"  {ticker}: ADF p={adf_p:.4f} ({'rechaza' if adf_rechaza else 'no rechaza'}) | "
            f"KPSS p={kpss_p:.4f} ({'rechaza' if kpss_rechaza else 'no rechaza'}) → {diagnostico}"
        )

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 2 — Phillips-Perron
# =============================================================================

def prueba_phillips_perron(retornos: pd.DataFrame,
                            res_estac: pd.DataFrame,
                            alpha: float) -> tuple[pd.DataFrame, list]:
    """
    Complemento robusto al ADF. Corrige heterocedasticidad y autocorrelación
    en los errores de forma no paramétrica.
    Phillips & Perron (1988). H0: raíz unitaria.
    """
    resultados = []
    logs       = ["PRUEBA 1B — PHILLIPS-PERRON (1988)",
                  f"α = {alpha} | H0: raíz unitaria"]

    for ticker in retornos.columns:
        serie = retornos[ticker].dropna()

        pp       = PhillipsPerron(serie, trend='c')
        pp_stat  = pp.stat
        pp_pval  = pp.pvalue
        rechaza  = pp_pval < alpha

        diagnostico = "ESTACIONARIA"    if rechaza else "NO ESTACIONARIA"
        color       = "VERDE"           if rechaza else "ROJO"

        # Consistencia con ADF
        adf_row  = res_estac[res_estac['Ticker'] == ticker]
        adf_diag = adf_row.iloc[0]['Diagnostico'] if not adf_row.empty else "N/D"

        if 'ESTACIONARIA' in adf_diag and diagnostico == 'ESTACIONARIA':
            consistencia = "CONSISTENTES"
        elif 'ESTACIONARIA' not in adf_diag and diagnostico != 'ESTACIONARIA':
            consistencia = "CONSISTENTES"
        else:
            consistencia = "INCONSISTENTES — revisar"

        resultados.append({
            'Ticker':       ticker,
            'PP_stat':      round(pp_stat, 4),
            'PP_pvalue':    round(pp_pval, 4),
            'Rechaza_H0':   rechaza,
            'Diagnostico':  diagnostico,
            'Semaforo':     color,
            'Consistencia': consistencia,
        })

        logs.append(
            f"  {ticker}: PP p={pp_pval:.4f} ({'rechaza' if rechaza else 'no rechaza'}) "
            f"→ {diagnostico} | ADF vs PP: {consistencia}"
        )

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 3 — Normalidad univariante
# =============================================================================

def prueba_normalidad(retornos: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, list]:
    """
    Jarque-Bera (1980) + Shapiro-Wilk (1965).
    H0: distribución normal.
    """
    resultados = []
    logs       = ["PRUEBA 2 — NORMALIDAD UNIVARIANTE (JB + SW)",
                  f"α = {alpha} | H0: distribución normal"]

    n_obs = len(retornos)
    if n_obs > 50:
        logs.append(f"  NOTA: n={n_obs} > 50 — Shapiro-Wilk pierde poder estadístico. "
                    "Jarque-Bera es la prueba de referencia con esta muestra.")

    for ticker in retornos.columns:
        serie = retornos[ticker].dropna()

        jb_stat, jb_p = stats.jarque_bera(serie)
        jb_rechaza    = jb_p < alpha

        sw_stat, sw_p = stats.shapiro(serie)
        sw_rechaza    = sw_p < alpha

        asimetria = stats.skew(serie)
        curtosis  = stats.kurtosis(serie)

        if not jb_rechaza and not sw_rechaza:
            diagnostico = "NORMAL"
            color       = "VERDE"
        elif jb_rechaza and sw_rechaza:
            diagnostico = "NO NORMAL"
            color       = "ROJO"
        else:
            diagnostico = "EVIDENCIA MIXTA"
            color       = "AMARILLO"

        resultados.append({
            'Ticker':      ticker,
            'JB_stat':     round(jb_stat, 4),
            'JB_pvalue':   round(jb_p, 4),
            'JB_rechaza':  jb_rechaza,
            'SW_stat':     round(sw_stat, 4),
            'SW_pvalue':   round(sw_p, 4),
            'SW_rechaza':  sw_rechaza,
            'Asimetria':   round(asimetria, 4),
            'Curtosis':    round(curtosis, 4),
            'Diagnostico': diagnostico,
            'Semaforo':    color,
        })

        logs.append(
            f"  {ticker}: JB p={jb_p:.4f} | SW p={sw_p:.4f} | "
            f"asimetría={asimetria:.3f} | curtosis={curtosis:.3f} → {diagnostico}"
        )

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 4 — Correlación Pearson vs Spearman
# =============================================================================

def prueba_correlacion_spearman(retornos: pd.DataFrame,
                                 alpha: float) -> tuple[dict, list]:
    """
    Detecta dependencia no lineal comparando Pearson vs Spearman.
    Umbral de alerta: diferencia > 0.10.
    """
    tickers    = list(retornos.columns)
    n          = len(tickers)
    pearson    = retornos.corr(method='pearson')
    spearman   = retornos.corr(method='spearman')
    diferencia = (spearman - pearson).abs()

    UMBRAL_DIFF       = 0.10
    pares_problematicos = []
    logs              = ["PRUEBA 2B — CORRELACIÓN PEARSON vs SPEARMAN",
                         "Umbral de alerta: |diferencia| > 0.10"]

    for i in range(n):
        for j in range(i+1, n):
            t1, t2 = tickers[i], tickers[j]
            p_val  = pearson.loc[t1, t2]
            s_val  = spearman.loc[t1, t2]
            diff   = abs(s_val - p_val)

            if diff > UMBRAL_DIFF:
                pares_problematicos.append((t1, t2, p_val, s_val, diff))
                logs.append(f"  ⚠️ {t1}—{t2}: Pearson={p_val:.4f} | "
                            f"Spearman={s_val:.4f} | |Δ|={diff:.4f}")

    if not pares_problematicos:
        color_global = "VERDE"
        logs.append("  Diagnóstico: CONSISTENTE — dependencia lineal en todos los pares.")
    else:
        color_global = "AMARILLO"
        logs.append(f"  Diagnóstico: {len(pares_problematicos)} par(es) con dependencia no lineal.")

    return {
        'pearson':             pearson,
        'spearman':            spearman,
        'diferencia':          diferencia,
        'pares_problematicos': pares_problematicos,
        'semaforo':            color_global,
    }, logs


# =============================================================================
# FUNCIÓN 5 — Autocorrelación (Ljung-Box)
# =============================================================================

def prueba_autocorrelacion(retornos: pd.DataFrame, alpha: float,
                            lags: list = None) -> tuple[pd.DataFrame, list]:
    """
    Ljung-Box (1978) en retornos y retornos².
    H0: no hay autocorrelación hasta rezago k.
    """
    if lags is None:
        lags = [1, 5, 10]

    resultados = []
    logs       = [f"PRUEBA 3 — AUTOCORRELACIÓN (LJUNG-BOX) | rezagos={lags}",
                  f"α = {alpha} | H0: sin autocorrelación"]

    for ticker in retornos.columns:
        serie = retornos[ticker].dropna()

        lb_ret  = acorr_ljungbox(serie,    lags=lags, return_df=True)
        lb_ret2 = acorr_ljungbox(serie**2, lags=lags, return_df=True)

        hay_autocorr     = any(lb_ret['lb_pvalue']  < alpha)
        hay_autocorr_vol = any(lb_ret2['lb_pvalue'] < alpha)

        diag_ret  = "AUTOCORRELACIÓN EN RETORNOS" if hay_autocorr     else "SIN AUTOCORRELACIÓN"
        diag_vol  = "CLUSTERING DE VOLATILIDAD"   if hay_autocorr_vol else "SIN CLUSTERING"
        color_ret = "ROJO"     if hay_autocorr     else "VERDE"
        color_vol = "AMARILLO" if hay_autocorr_vol else "VERDE"

        resultados.append({
            'Ticker':            ticker,
            'Autocorr_retornos': hay_autocorr,
            'Diag_retornos':     diag_ret,
            'Semaforo_retornos': color_ret,
            'Autocorr_vol':      hay_autocorr_vol,
            'Diag_vol':          diag_vol,
            'Semaforo_vol':      color_vol,
        })

        logs.append(f"  {ticker}: retornos → {diag_ret} | volatilidad → {diag_vol}")

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 6 — Efectos ARCH (Engle, 1982)
# =============================================================================

def prueba_arch(retornos: pd.DataFrame, alpha: float,
                lags: int = 5) -> tuple[pd.DataFrame, list]:
    """
    ARCH-LM de Engle (1982). H0: no hay efectos ARCH (varianza constante).
    Rechazar → volatilidad time-varying → candidato para GARCH en Capa 2.
    """
    resultados = []
    logs       = [f"PRUEBA 4 — EFECTOS ARCH (ENGLE, 1982) | rezagos={lags}",
                  f"α = {alpha} | H0: sin efectos ARCH"]

    for ticker in retornos.columns:
        serie = retornos[ticker].dropna()

        lm_stat, lm_p, _, _ = het_arch(serie, nlags=lags)
        rechaza = lm_p < alpha

        diagnostico = "EFECTOS ARCH PRESENTES" if rechaza else "SIN EFECTOS ARCH"
        color       = "AMARILLO"               if rechaza else "VERDE"

        resultados.append({
            'Ticker':      ticker,
            'LM_stat':     round(lm_stat, 4),
            'LM_pvalue':   round(lm_p, 4),
            'Rechaza_H0':  rechaza,
            'Diagnostico': diagnostico,
            'Semaforo':    color,
        })

        logs.append(
            f"  {ticker}: LM p={lm_p:.4f} ({'rechaza' if rechaza else 'no rechaza'}) "
            f"→ {diagnostico}"
        )

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 7 — Normalidad multivariante (Mardia + Henze-Zirkler)
# =============================================================================

def prueba_normalidad_multivariante(retornos: pd.DataFrame,
                                     alpha: float) -> tuple[dict, list]:
    """
    Test de Mardia (1970) + Henze-Zirkler vía pingouin.
    CORRECCIÓN v0.3.1: implementación con pingouin evita overflow con n grande.
    H0: distribución normal multivariante.
    """
    logs = ["PRUEBA 5 — NORMALIDAD MULTIVARIANTE (MARDIA + HENZE-ZIRKLER)",
            f"α = {alpha} | H0: normal multivariante"]

    X    = retornos.dropna().values
    n, p = X.shape
    logs.append(f"  Dimensiones: {n} observaciones × {p} activos")

    try:
        import pingouin as pg

        hz_result = pg.multivariate_normality(X, alpha=alpha)
        hz_stat   = hz_result.hz
        hz_pval   = hz_result.pval
        hz_normal = hz_result.normal

        logs.append(f"  Henze-Zirkler: HZ={hz_stat:.4f} | p={hz_pval:.4f} | "
                    f"{'Normal' if hz_normal else 'No normal'}")

        mu    = X.mean(axis=0)
        S     = np.cov(X.T)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logs.append("  ERROR: Matriz de covarianza singular.")
            return {}, logs

        diff    = X - mu
        md_diag = np.einsum('ij,jk,ik->i', diff, S_inv, diff)

        if n <= 500:
            diff_sinv = diff @ S_inv
            G         = diff_sinv @ diff.T
            b1p       = np.sum(G**3) / n**2
            k_asim    = (n * b1p) / 6
            df_asim   = p * (p + 1) * (p + 2) / 6
            p_asim    = 1 - stats.chi2.cdf(k_asim, df=df_asim)
            rechaza_asim = p_asim < alpha
            logs.append(f"  Mardia asimetría (b1p): χ²={k_asim:.4f} | p={p_asim:.4f}")
        else:
            logs.append(f"  n={n} > 500: asimetría de Mardia omitida, usar Henze-Zirkler.")
            rechaza_asim = not hz_normal
            p_asim       = hz_pval
            b1p = k_asim = None

        b2p          = np.mean(md_diag**2)
        k_kurt       = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
        p_kurt       = 2 * (1 - stats.norm.cdf(abs(k_kurt)))
        rechaza_kurt = p_kurt < alpha
        logs.append(f"  Mardia curtosis (b2p): Z={k_kurt:.4f} | p={p_kurt:.4f}")

        if not rechaza_asim and not rechaza_kurt and hz_normal:
            diagnostico = "NORMAL MULTIVARIANTE"
            color       = "VERDE"
            implicacion = "Markowitz es estadísticamente apropiado."
        elif rechaza_asim and rechaza_kurt and not hz_normal:
            diagnostico = "NO NORMAL MULTIVARIANTE"
            color       = "ROJO"
            implicacion = "Markowitz puede subestimar el riesgo. CVaR es más confiable."
        else:
            diagnostico = "EVIDENCIA MIXTA"
            color       = "AMARILLO"
            implicacion = "Proceder con precaución. Complementar con CVaR."

        logs.append(f"  Diagnóstico: {diagnostico} | {implicacion}")

        return {
            'b1p': float(b1p) if b1p is not None else None,
            'k_asim': float(k_asim) if k_asim is not None else None,
            'p_asim': float(p_asim),
            'b2p': float(b2p),
            'k_kurt': float(k_kurt),
            'p_kurt': float(p_kurt),
            'hz_stat': float(hz_stat),
            'hz_pval': float(hz_pval),
            'rechaza_asim': bool(rechaza_asim),
            'rechaza_kurt': bool(rechaza_kurt),
            'diagnostico': diagnostico,
            'semaforo': color,
            'implicacion': implicacion,
        }, logs

    except ImportError:
        logs.append("  pingouin no disponible. Usando implementación manual vectorizada.")

    # --- Fallback manual vectorizado ---
    mu    = X.mean(axis=0)
    S     = np.cov(X.T)
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        logs.append("  ERROR: Matriz de covarianza singular.")
        return {}, logs

    diff    = X - mu
    md_diag = np.einsum('ij,jk,ik->i', diff, S_inv, diff)

    if n <= 500:
        diff_sinv = diff @ S_inv
        G         = diff_sinv @ diff.T
        b1p       = np.sum(G**3) / n**2
        k_asim    = (n * b1p) / 6
        df_asim   = p * (p + 1) * (p + 2) / 6
        p_asim    = 1 - stats.chi2.cdf(k_asim, df=df_asim)
        rechaza_asim = p_asim < alpha
    else:
        b1p = k_asim = None
        p_asim       = 0.5
        rechaza_asim = False

    b2p          = np.mean(md_diag**2)
    k_kurt       = (b2p - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    p_kurt       = 2 * (1 - stats.norm.cdf(abs(k_kurt)))
    rechaza_kurt = p_kurt < alpha

    if not rechaza_asim and not rechaza_kurt:
        diagnostico = "NORMAL MULTIVARIANTE"
        color       = "VERDE"
        implicacion = "Markowitz es estadísticamente apropiado."
    elif rechaza_asim and rechaza_kurt:
        diagnostico = "NO NORMAL MULTIVARIANTE"
        color       = "ROJO"
        implicacion = "Markowitz puede subestimar el riesgo. Usar CVaR como referencia."
    else:
        diagnostico = "EVIDENCIA MIXTA"
        color       = "AMARILLO"
        implicacion = "Proceder con precaución. Complementar con CVaR."

    logs.append(f"  Diagnóstico: {diagnostico}")

    return {
        'b1p': float(b1p) if b1p is not None else None,
        'k_asim': float(k_asim) if k_asim is not None else None,
        'p_asim': float(p_asim),
        'b2p': float(b2p),
        'k_kurt': float(k_kurt),
        'p_kurt': float(p_kurt),
        'rechaza_asim': bool(rechaza_asim),
        'rechaza_kurt': bool(rechaza_kurt),
        'diagnostico': diagnostico,
        'semaforo': color,
        'implicacion': implicacion,
    }, logs


# =============================================================================
# FUNCIÓN 8 — HME (Lo & MacKinlay, 1988)
# =============================================================================

def prueba_hme(retornos: pd.DataFrame, alpha: float,
               periodos: list = None) -> tuple[pd.DataFrame, list]:
    """
    Test de Ratio de Varianza overlapping (Lo & MacKinlay, 1988).
    H0: caminata aleatoria (HME débil, Fama 1970).
    CORRECCIÓN: períodos limitados a [2,4,8] con datos mensuales.
    """
    n_total = len(retornos)
    if periodos is None:
        periodos_candidatos = [2, 4, 8, 16]
        periodos = [k for k in periodos_candidatos if n_total / k >= 10]
        if not periodos:
            periodos = [2]

    resultados = []
    logs       = [f"PRUEBA 6 — HME DÉBIL (LO & MACKINLAY, 1988) | períodos={periodos}",
                  f"α = {alpha} | H0: caminata aleatoria"]

    for ticker in retornos.columns:
        serie = retornos[ticker].dropna().values
        n     = len(serie)
        mu_c  = np.mean(serie)
        var1  = np.sum((serie - mu_c)**2) / (n - 1)

        rechazos = []
        for k in periodos:
            if n < k * 2:
                continue

            ret_k = np.array([np.sum(serie[i:i+k]) for i in range(n - k + 1)])
            m     = k * (n - k + 1) * (1 - k / n)
            var_k = np.sum((ret_k - k * mu_c)**2) / m
            vr    = var_k / var1

            theta = 0.0
            denom = (np.sum((serie - mu_c)**2))**2  # constante para todo el loop en j
            for j in range(1, k):
                num = np.sum(((serie[j:] - mu_c)**2) * ((serie[:-j] - mu_c)**2))
                delta_j = num / denom
                theta += ((2 * (k - j)) / k)**2 * delta_j

            if theta > 0:
                z_stat = (vr - 1) / np.sqrt(theta / n)
            else:
                z_stat = (vr - 1) * np.sqrt(n * k / (2 * (k - 1)))

            p_valor = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            rechaza = p_valor < alpha
            rechazos.append(rechaza)

        if not rechazos:
            resultados.append({
                'Ticker': ticker, 'N_rechazos': 0,
                'Diagnostico': 'NO EVALUADO', 'Semaforo': 'AMARILLO',
            })
            logs.append(f"  {ticker}: NO EVALUADO — muestra insuficiente")
            continue

        n_rechazos = sum(rechazos)
        if n_rechazos == 0:
            diagnostico = "HME DÉBIL NO RECHAZADA"
            color       = "VERDE"
        elif n_rechazos <= len(rechazos) // 2:
            diagnostico = "EVIDENCIA MIXTA"
            color       = "AMARILLO"
        else:
            diagnostico = "HME DÉBIL RECHAZADA"
            color       = "ROJO"

        resultados.append({
            'Ticker':      ticker,
            'N_rechazos':  n_rechazos,
            'Diagnostico': diagnostico,
            'Semaforo':    color,
        })
        logs.append(f"  {ticker}: {n_rechazos}/{len(rechazos)} períodos rechazan H0 → {diagnostico}")

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 9 — Tabla semáforo consolidada
# =============================================================================

def tabla_semaforo(res_estac, res_pp, res_norm, res_autocorr,
                   res_arch, res_mardia, res_hme,
                   res_spearman) -> pd.DataFrame:
    """
    Tabla consolidada con semáforo por prueba y activo.
    Retorna DataFrame con columnas: Prueba + un ticker por columna.
    La columna 'Semaforo' muestra el emoji correspondiente.
    """
    colores_emoji = {'VERDE': '🟢', 'AMARILLO': '🟡', 'ROJO': '🔴'}
    tickers = res_estac['Ticker'].tolist()

    pruebas = [
        ('Estacionariedad (ADF+KPSS)',       res_estac,    'Semaforo'),
        ('Estacionariedad (Phillips-Perron)', res_pp,       'Semaforo'),
        ('Normalidad univariante',            res_norm,     'Semaforo'),
        ('Autocorrelación (Ljung-Box)',       res_autocorr, 'Semaforo_retornos'),
        ('Clustering volatilidad',            res_autocorr, 'Semaforo_vol'),
        ('Efectos ARCH',                      res_arch,     'Semaforo'),
        ('HME débil (Lo-MacKinlay)',          res_hme,      'Semaforo'),
    ]

    tabla = []
    for nombre, df, col in pruebas:
        fila = {'Prueba': nombre}
        for ticker in tickers:
            row = df[df['Ticker'] == ticker]
            color = row.iloc[0][col] if not row.empty else 'AMARILLO'
            fila[ticker] = colores_emoji.get(color, '⚪') + ' ' + color
        tabla.append(fila)

    # Mardia — fila global
    color_mardia = res_mardia.get('semaforo', 'AMARILLO')
    fila_mardia  = {'Prueba': 'Normalidad multivariante (Mardia)'}
    for ticker in tickers:
        fila_mardia[ticker] = colores_emoji.get(color_mardia, '⚪') + ' ' + color_mardia
    tabla.append(fila_mardia)

    # Spearman — fila global
    color_sp = res_spearman.get('semaforo', 'VERDE')
    fila_sp  = {'Prueba': 'Correlación Pearson vs Spearman'}
    for ticker in tickers:
        fila_sp[ticker] = colores_emoji.get(color_sp, '⚪') + ' ' + color_sp
    tabla.append(fila_sp)

    return pd.DataFrame(tabla)


# =============================================================================
# FUNCIÓN 10 — Asistente Fase 2
# =============================================================================

def asistente_fase2(res_estac, res_norm, res_arch, res_mardia,
                    res_hme, res_spearman, nivel='basico') -> str:
    """
    Interpreta los resultados del análisis estadístico preliminar.
    Sigue el contrato: qué ocurrió → por qué importa → valoración → implicación → acción.
    """
    lineas = []

    def add(texto=""):
        lineas.append(texto)

    add("## Análisis Estadístico Preliminar")
    add()

    # --- Estacionariedad ---
    no_estac  = res_estac[res_estac['Semaforo'] == 'ROJO']['Ticker'].tolist()
    adv_estac = res_estac[res_estac['Semaforo'] == 'AMARILLO']['Ticker'].tolist()

    if nivel == 'basico':
        add("### Comportamiento de los retornos en el tiempo")
        if not no_estac and not adv_estac:
            add("✅ **Los retornos de todos los activos son estables en el tiempo.**")
            add("Esto significa que su comportamiento promedio y su variabilidad no cambian "
                "drásticamente de un período a otro — condición necesaria para que los modelos "
                "que usaremos más adelante den resultados confiables.")
            add("**Acción:** ninguna. El análisis puede continuar normalmente.")
        elif no_estac:
            add(f"⚠️ **Los retornos de `{no_estac}` muestran comportamiento inestable.**")
            add("Su promedio o variabilidad cambia a lo largo del tiempo, lo que puede afectar "
                "la confiabilidad de las estimaciones. Esto suele ocurrir con activos que "
                "atravesaron cambios estructurales importantes en el período analizado.")
            add("**Acción:** los resultados de estos activos deben interpretarse con precaución. "
                "Considera ampliar la ventana temporal o investigar si hubo eventos relevantes.")
    else:
        add("### Estacionariedad (ADF + KPSS + Phillips-Perron)")
        if not no_estac:
            add("✅ **Todas las series son I(0).**")
            add("ADF y PP rechazan raíz unitaria. KPSS no rechaza estacionariedad. "
                "Diagnóstico consistente entre los tres tests — base estadística sólida para MCO.")
        else:
            add(f"⚠️ **`{no_estac}` presentan raíz unitaria (evidencia I(1)).**")
            add("Usar estas series en niveles en modelos de regresión produce regresiones espurias. "
                "Las series deberían diferenciarse antes de entrar a los modelos de factores. "
                "Sin embargo, dado que trabajamos con retornos logarítmicos (ya una diferencia "
                "del log-precio), este resultado podría indicar un quiebre estructural en el período.")
        if adv_estac:
            add(f"🟡 **`{adv_estac}` muestran evidencia mixta (ADF y KPSS inconsistentes).** "
                "Posible quiebre estructural en el período analizado.")

    add()

    # --- Normalidad ---
    no_norm   = res_norm[res_norm['Semaforo'] == 'ROJO']['Ticker'].tolist()
    kurt_alta = res_norm[res_norm['Curtosis'] > 1]['Ticker'].tolist() if 'Curtosis' in res_norm.columns else []

    if nivel == 'basico':
        add("### Distribución de los retornos")
        if not no_norm:
            add("✅ **Los retornos siguen aproximadamente una distribución normal.**")
            add("Esto valida el uso de Markowitz y el CAPM sin restricciones adicionales. "
                "Los modelos asumen que los retornos se distribuyen de forma simétrica alrededor "
                "de su media — y los datos son consistentes con ese supuesto.")
            add("**Acción:** ninguna. El análisis continúa con sus supuestos intactos.")
        else:
            add(f"⚠️ **Los retornos de `{no_norm}` no siguen distribución normal.**")
            add("Esto es la norma, no la excepción, en finanzas reales: los mercados tienen "
                "más días extremos de los que la distribución normal predice. "
                "Markowitz y el CAPM funcionan como buenas aproximaciones, pero el VaR "
                "paramétrico subestimará el riesgo real en esos activos.")
            add("**Acción:** en la Fase 3c, usar el VaR histórico y el CVaR como métricas "
                "de riesgo primarias en lugar del VaR paramétrico.")
    else:
        add("### Normalidad univariante (Jarque-Bera + Shapiro-Wilk)")
        if not no_norm:
            add("✅ **No se rechaza H0 de normalidad en ningún activo.**")
            add("Asimetría y curtosis en exceso dentro de rangos aceptables. "
                "Los supuestos de Gauss-Markov aplicables.")
        else:
            add(f"⚠️ **`{no_norm}` rechazan H0 de normalidad.**")
            if kurt_alta:
                add(f"Leptocurtosis en `{kurt_alta}` — colas más pesadas que la normal. "
                    "Implica mayor frecuencia de retornos extremos de lo que la distribución "
                    "normal predice. El VaR paramétrico subestima el riesgo de cola — "
                    "CVaR histórico es la métrica adecuada.")
            add("Los estimadores MCO siguen siendo insesgados (Gauss-Markov no requiere normalidad), "
                "pero los intervalos de confianza basados en t-Student pierden exactitud.")

    add()

    # --- ARCH ---
    col_rechaza = next((c for c in res_arch.columns if 'rechaza' in c.lower()), 'Rechaza_H0')
    con_arch = res_arch[res_arch[col_rechaza] == True]['Ticker'].tolist() if col_rechaza in res_arch.columns else []

    if nivel == 'basico':
        add("### Estabilidad de la volatilidad")
        if con_arch:
            add(f"🟡 **La volatilidad de `{con_arch}` no es constante en el tiempo.**")
            add("Hay períodos de alta volatilidad seguidos de períodos de baja volatilidad — "
                "los mercados tienden a agrupar los movimientos grandes. "
                "Markowitz asume volatilidad constante, lo que puede subestimar el riesgo "
                "en períodos de turbulencia.")
            add("**Acción:** los resultados de riesgo (VaR, CVaR) son más conservadores "
                "de lo que Markowitz sugiere. Presta especial atención al análisis de estrés.")
        else:
            add("✅ **La volatilidad es aproximadamente constante en todos los activos.**")
            add("No hay evidencia de que los movimientos grandes se agrupen en el tiempo, "
                "lo que valida el uso de la desviación estándar como medida de riesgo.")
            add("**Acción:** ninguna. Markowitz aplica bien.")
    else:
        add("### Efectos ARCH (Engle, 1982)")
        if con_arch:
            add(f"🟡 **`{con_arch}` presentan heterocedasticidad condicional.**")
            add("Varianza condicional time-varying — evidencia de clustering de volatilidad. "
                "Implicación: los pesos óptimos de Markowitz son sensibles al régimen de "
                "volatilidad vigente. GARCH(1,1) modelaría esta dinámica (Capa 2).")
        else:
            add("✅ **Sin efectos ARCH en frecuencia mensual.**")
            add("Puede ser consistente con Drost & Nijman (1993): la agregación temporal "
                "tiende a suavizar los efectos ARCH presentes en frecuencias más altas.")

    add()

    # --- Mardia ---
    diag_mardia = res_mardia.get('diagnostico', 'NO EVALUADO') if res_mardia else 'NO EVALUADO'

    if nivel == 'basico':
        add("### El portafolio como conjunto")
        if 'NO NORMAL' in diag_mardia:
            add("⚠️ **El portafolio en su conjunto no sigue distribución normal multivariante.**")
            add("Esto refuerza lo anterior: las relaciones entre activos durante eventos extremos "
                "son más complejas de lo que Markowitz captura. "
                "En una crisis, los activos que parecían independientes tienden a caer juntos — "
                "justo cuando más se necesita la diversificación.")
            add("**Acción:** complementar Markowitz con el CVaR y el análisis de estrés histórico.")
        elif 'NORMAL' in diag_mardia:
            add("✅ **El portafolio cumple los supuestos de normalidad multivariante.**")
            add("La optimización de Markowitz aplica con sus supuestos estadísticos intactos.")
    else:
        add("### Normalidad multivariante (Mardia, 1970 + Henze-Zirkler)")
        if 'NO NORMAL' in diag_mardia:
            add("⚠️ **Se rechaza H0 de normalidad multivariante.**")
            add("Distribución conjunta con colas pesadas y/o asimetría multivariante. "
                "Las correlaciones en escenarios extremos son superiores a las estimadas "
                "en condiciones normales (Longin & Solnik, 2001). "
                "La frontera eficiente de Markowitz subestima el riesgo de portafolio "
                "en escenarios de estrés.")
        elif 'NORMAL' in diag_mardia:
            add("✅ **No se rechaza normalidad multivariante.**")
            add("La estructura de covarianzas de Markowitz captura adecuadamente "
                "la dependencia entre activos.")

    add()

    # --- Spearman ---
    pares_prob = res_spearman.get('pares_problematicos', []) if res_spearman else []

    if nivel == 'basico':
        add("### Relaciones entre activos")
        if pares_prob:
            add(f"🟡 **En {len(pares_prob)} par(es), la dependencia entre activos es no lineal.**")
            add("Markowitz mide la relación entre activos con correlaciones lineales — "
                "pero algunos pares se mueven juntos de formas más complejas que una línea recta. "
                "Los pesos óptimos son válidos como aproximación, pero puede haber dependencias "
                "que el modelo no captura.")
            add("**Acción:** ninguna inmediata. Tener en cuenta al interpretar los pesos.")
        else:
            add("✅ **Las relaciones entre activos son aproximadamente lineales.**")
            add("La correlación de Pearson captura bien la dependencia real — "
                "Markowitz no pierde información relevante al usar la covarianza.")
    else:
        add("### Correlación Pearson vs Spearman")
        if pares_prob:
            add("🟡 **Dependencia no lineal detectada en los siguientes pares:**")
            for t1, t2, p, s, d in pares_prob:
                dir_texto = "subestima" if s > p else "sobreestima"
                add(f"- `{t1}—{t2}`: Pearson={p:.4f}, Spearman={s:.4f}, |Δ|={d:.4f} "
                    f"→ Pearson {dir_texto} la dependencia real.")
            add("La matriz de covarianza de Markowitz puede ser una aproximación imprecisa "
                "para estos pares. DCC-GARCH modelaría la dependencia dinámica (Capa 2).")
        else:
            add("✅ **Correlaciones de Pearson y Spearman son consistentes en todos los pares.**")
            add("No hay evidencia de dependencia no lineal relevante.")

    add()

    # --- HME ---
    hme_rechazada = res_hme[res_hme['Semaforo'] == 'ROJO']['Ticker'].tolist() if not res_hme.empty else []

    if nivel == 'basico':
        add("### ¿Se puede predecir el comportamiento futuro con datos pasados?")
        if hme_rechazada:
            add(f"🟡 **En `{hme_rechazada}`, los retornos pasados tienen cierto poder predictivo.**")
            add("Los mercados perfectamente eficientes no permiten predicciones basadas en "
                "precios históricos. Que esto se rechace puede indicar ineficiencias reales "
                "o simplemente que el período analizado tuvo patrones atípicos.")
            add("**Acción:** no actuar sobre esto directamente. Es contexto para interpretar "
                "los modelos de factores con mayor atención.")
        else:
            add("✅ **Los retornos son consistentes con mercados eficientes en forma débil.**")
            add("No hay evidencia de que los precios pasados predigan los futuros — "
                "los modelos de factores parten de una base estadísticamente sólida.")
    else:
        add("### Hipótesis de Mercado Eficiente débil (Lo & MacKinlay, 1988)")
        if hme_rechazada:
            add(f"🟡 **`{hme_rechazada}` rechazan la hipótesis de caminata aleatoria.**")
            add("El ratio de varianza overlapping es significativamente distinto de 1 "
                "en al menos un horizonte temporal. Posibles causas: autocorrelación positiva "
                "de corto plazo (momentum), efectos de microestructura, o iliquidez.")
        else:
            add("✅ **No se rechaza la caminata aleatoria en ningún activo.**")
            add("Evidencia consistente con HME débil — los retornos pasados no contienen "
                "información predictiva explotable.")

    add()
    add("---")
    if nivel == 'basico':
        add("*Cuando un supuesto no se cumple, el análisis no se cancela — se matiza. "
            "Saber dónde están las limitaciones es lo que hace riguroso un análisis.*")
    else:
        add("*Las violaciones de supuestos identificadas aquí se propagan a los módulos "
            "siguientes como contexto para calibrar la interpretación de los resultados.*")

    return "\n".join(lineas)


# =============================================================================
# FUNCIÓN PIPELINE — Envuelve todo el flujo de la Fase 2
# =============================================================================

def pipeline_estadistico(retornos: pd.DataFrame,
                          nivel_asistente: str = 'basico') -> dict:
    """
    Ejecuta el flujo completo de la Fase 2.

    Parámetros
    ----------
    retornos : pd.DataFrame
        Retornos logarítmicos de la Fase 1 (st.session_state['retornos']).
    nivel_asistente : str
        'basico' o 'tecnico'.

    Retorna
    -------
    dict con claves:
        'res_estac', 'res_pp', 'res_norm', 'res_spearman',
        'res_autocorr', 'res_arch', 'res_mardia', 'res_hme',
        'semaforo_f2', 'narrativa_f2',
        'logs', 'error'
    """
    logs_totales = []

    def _log(msgs):
        if isinstance(msgs, list):
            logs_totales.extend(msgs)
        else:
            logs_totales.append(msgs)

    try:
        res_estac,    lg = prueba_estacionariedad(retornos, ALPHA)
        _log(lg)

        res_pp,       lg = prueba_phillips_perron(retornos, res_estac, ALPHA)
        _log(lg)

        res_norm,     lg = prueba_normalidad(retornos, ALPHA)
        _log(lg)

        res_spearman, lg = prueba_correlacion_spearman(retornos, ALPHA)
        _log(lg)

        res_autocorr, lg = prueba_autocorrelacion(retornos, ALPHA)
        _log(lg)

        res_arch,     lg = prueba_arch(retornos, ALPHA)
        _log(lg)

        res_mardia,   lg = prueba_normalidad_multivariante(retornos, ALPHA)
        _log(lg)

        res_hme,      lg = prueba_hme(retornos, ALPHA)
        _log(lg)

        semaforo = tabla_semaforo(
            res_estac, res_pp, res_norm, res_autocorr,
            res_arch, res_mardia, res_hme, res_spearman,
        )

        narrativa = asistente_fase2(
            res_estac, res_norm, res_arch, res_mardia,
            res_hme, res_spearman, nivel=nivel_asistente,
        )

        _log("✅ FASE 2 COMPLETADA")

        return {
            'res_estac':    res_estac,
            'res_pp':       res_pp,
            'res_norm':     res_norm,
            'res_spearman': res_spearman,
            'res_autocorr': res_autocorr,
            'res_arch':     res_arch,
            'res_mardia':   res_mardia,
            'res_hme':      res_hme,
            'semaforo_f2':  semaforo,
            'narrativa_f2': narrativa,
            'logs':         logs_totales,
            'error':        None,
        }

    except Exception as e:
        import traceback
        _log(f"ERROR FATAL en pipeline_estadistico: {e}")
        _log(traceback.format_exc())
        return {'error': str(e), 'logs': logs_totales}
