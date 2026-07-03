# =============================================================================
# QuantαfolyΩ — modulos/factores.py
# Fase 3b: Modelos de Factores (CAPM, FF3, APT)
#
# Funciones migradas desde PortfolioLab_V0_7_0.ipynb (celdas 46–56).
# REGLA DE MIGRACIÓN: lógica econométrica sin cambios.
# Cambios:
#   - print() → entradas en lista `logs`
#   - graficar_betas() reescrita en Plotly (retorna fig)
#   - asistente_fase3b() retorna str markdown
#   - descargar_factores_macro() recibe retornos.index como parámetro
#     (en lugar de usar la variable global `retornos`)
#   - Drive / Colab eliminados
#   - pipeline_factores() envuelve todo el flujo
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import io
import zipfile
import requests
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    ALPHA,
    FACTOR_ANUALIZACION,
    TICKER_BENCHMARK,
    FRECUENCIA_DEFAULT,
    MIN_OBSERVACIONES_CAPM,
    VIF_UMBRAL,
    APT_VARIANZA_MINIMA_PCA,
    APT_MIN_FACTORES,
    APT_MAX_FACTORES,
    FACTORES_MACRO_CANDIDATOS,
    COLORES,
)


# =============================================================================
# FUNCIÓN 1 — Pruebas de residuos (BLUE)
# =============================================================================

def pruebas_residuos(residuos: pd.Series, regresores_modelo: pd.DataFrame,
                     alpha: float, nombre_modelo: str,
                     nombre_activo: str) -> dict:
    """
    Pruebas Gauss-Markov sobre residuos de una regresión MCO.
    CORRECCIÓN v0.5.1: Breusch-Pagan usa los regresores del modelo correcto.

    Breusch & Pagan (1979): H0 = homocedasticidad
    Breusch & Godfrey (1978): H0 = sin autocorrelación serial
    Jarque & Bera (1980): H0 = normalidad de residuos
    """
    res = residuos.dropna()
    reg = regresores_modelo.reindex(res.index).dropna()
    res = res.reindex(reg.index)

    resultados = {}

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(res)
    resultados['JB_stat']    = round(jb_stat, 4)
    resultados['JB_pvalue']  = round(jb_p, 4)
    resultados['JB_rechaza'] = jb_p < alpha

    # Breusch-Pagan — CORRECCIÓN: usa regresores del modelo
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(res, reg)
        resultados['BP_stat']    = round(bp_stat, 4)
        resultados['BP_pvalue']  = round(bp_p, 4)
        resultados['BP_rechaza'] = bp_p < alpha
    except Exception:
        resultados['BP_stat']    = None
        resultados['BP_pvalue']  = None
        resultados['BP_rechaza'] = None

    # Breusch-Godfrey
    try:
        bg_stat, bg_p, _, _ = acorr_breusch_godfrey(
            sm.OLS(res, reg).fit(), nlags=4
        )
        resultados['BG_stat']    = round(bg_stat, 4)
        resultados['BG_pvalue']  = round(bg_p, 4)
        resultados['BG_rechaza'] = bg_p < alpha
    except Exception:
        resultados['BG_stat']    = None
        resultados['BG_pvalue']  = None
        resultados['BG_rechaza'] = None

    violaciones = sum([
        bool(resultados['JB_rechaza']),
        bool(resultados['BP_rechaza']) if resultados['BP_rechaza'] is not None else False,
        bool(resultados['BG_rechaza']) if resultados['BG_rechaza'] is not None else False,
    ])

    if violaciones == 0:
        resultados['BLUE']     = 'CUMPLE'
        resultados['Semaforo'] = 'VERDE'
    elif violaciones == 1:
        resultados['BLUE']     = 'ADVERTENCIA'
        resultados['Semaforo'] = 'AMARILLO'
    else:
        resultados['BLUE']     = 'VIOLA SUPUESTOS'
        resultados['Semaforo'] = 'ROJO'

    return resultados


# =============================================================================
# FUNCIÓN 2 — CAPM
# =============================================================================

def estimar_capm(retornos: pd.DataFrame,
                 retornos_benchmark: pd.DataFrame,
                 rf_mes: float, alpha: float) -> tuple[pd.DataFrame, list]:
    """
    CAPM: (rᵢ - rf) = α + β(rm - rf) + ε
    Sharpe (1964), Lintner (1965), Mossin (1966).
    """
    # A1: acceso robusto al benchmark por nombre de columna
    col_bench = 'SP500' if 'SP500' in retornos_benchmark.columns else retornos_benchmark.columns[0]
    bench   = retornos_benchmark[col_bench].reindex(retornos.index).dropna()
    exc_mkt = bench - rf_mes
    logs    = ["BLOQUE 1 — CAPM"]
    resultados = []

    for ticker in retornos.columns:
        serie = retornos[ticker].copy()
        idx   = serie.index.intersection(exc_mkt.index)
        y     = (serie.loc[idx] - rf_mes)
        X_df  = pd.DataFrame({'exc_mkt': exc_mkt.loc[idx]})
        X     = sm.add_constant(X_df)

        modelo   = sm.OLS(y, X).fit()
        residuos = modelo.resid
        pruebas  = pruebas_residuos(residuos, X, alpha, 'CAPM', ticker)

        alpha_j    = float(modelo.params.get('const', modelo.params.iloc[0]))
        beta_val   = float(modelo.params.get('exc_mkt', modelo.params.iloc[1]))
        alpha_anual = alpha_j * FACTOR_ANUALIZACION  # anualizado por convención estándar (×factor), no definición original de Jensen (1968)
        r2         = modelo.rsquared
        t_alpha    = float(modelo.tvalues.iloc[0])
        p_alpha    = float(modelo.pvalues.iloc[0])
        t_beta     = float(modelo.tvalues.iloc[1])
        p_beta     = float(modelo.pvalues.iloc[1])

        perfil = ("Agresivo — amplifica movimientos del mercado" if beta_val > 1.1
                  else "Defensivo — amortigua movimientos del mercado" if beta_val < 0.9
                  else "Neutral — sigue al mercado")

        logs.append(f"  {ticker}: α={alpha_anual:.4f} (p={p_alpha:.4f}) | "
                    f"β={beta_val:.4f} (p={p_beta:.4f}) | R²={r2:.4f} | "
                    f"BLUE={pruebas['BLUE']} | {perfil}")

        resultados.append({
            'Ticker':      ticker,
            'Alpha_anual': round(alpha_anual, 4),
            'Alpha_mens':  round(alpha_j, 6),
            'Beta':        round(beta_val, 4),
            'R2':          round(r2, 4),
            't_alpha':     round(t_alpha, 4),
            'p_alpha':     round(p_alpha, 4),
            't_beta':      round(t_beta, 4),
            'p_beta':      round(p_beta, 4),
            'Alpha_sig':   p_alpha < alpha,
            'Beta_sig':    p_beta < alpha,
            'Perfil_beta': perfil,
            'JB_pvalue':   pruebas['JB_pvalue'],
            'BP_pvalue':   pruebas['BP_pvalue'],
            'BG_pvalue':   pruebas['BG_pvalue'],
            'BLUE':        pruebas['BLUE'],
            'Semaforo':    pruebas['Semaforo'],
        })

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 3 — Descarga FF3
# =============================================================================

def descargar_ff3_v2(fecha_inicio: str, fecha_fin: str) -> tuple[pd.DataFrame, list]:
    """
    Descarga factores Fama-French 3 (Mkt-RF, SMB, HML) con fallback.

    Método 1: ZIP directo desde Ken French con parsing robusto
    Método 2: proxies ETF desde Yahoo Finance (aproximación)
    Método 3: degradación controlada

    CORRECCIÓN: índice normalizado a primer día del mes para alineación.
    """
    logs = ["DESCARGA FF3 — Fama & French (1993)"]

    def _normalizar_ff3(df):
        df.index = df.index.to_period('M').to_timestamp()
        df.index = df.index + pd.offsets.MonthBegin(0)
        df = df.loc[fecha_inicio:fecha_fin]
        return df

    # ------------------------------------------------------------------
    # Método 1: ZIP directo desde Ken French
    # ------------------------------------------------------------------
    try:
        logs.append("  Método 1: descarga directa ZIP Ken French...")
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36'),
            'Accept': 'application/zip,application/octet-stream,*/*',
            'Referer': ('https://mba.tuck.dartmouth.edu/pages/faculty/'
                        'ken.french/data_library.html'),
        }
        url  = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
                "ftp/F-F_Research_Data_Factors_CSV.zip")
        resp = requests.get(url, headers=headers, timeout=15)

        if resp.status_code != 200:
            raise ValueError(f"HTTP {resp.status_code}")

        zf        = zipfile.ZipFile(io.BytesIO(resp.content))
        nombre    = [n for n in zf.namelist() if n.upper().endswith('.CSV')][0]
        contenido = zf.read(nombre).decode('utf-8', errors='replace')
        lineas    = contenido.split('\n')

        # Encontrar inicio: líneas donde el primer campo es exactamente 6 dígitos
        inicio = None
        for i, l in enumerate(lineas):
            primer = l.strip().split(',')[0].strip()
            if primer.isdigit() and len(primer) == 6:
                inicio = i
                break

        if inicio is None:
            raise ValueError("No se encontró sección mensual (fechas YYYYMM)")

        # Encontrar fin: primera línea donde el primer campo es exactamente 4 dígitos
        # (inicio de sección anual). Ignorar líneas de texto.
        fin = len(lineas)
        for i in range(inicio + 1, len(lineas)):
            primer = lineas[i].strip().split(',')[0].strip()
            if primer.isdigit() and len(primer) == 4:
                fin = i
                break

        # Parsear solo el bloque mensual
        filas = []
        for l in lineas[inicio:fin]:
            partes = [p.strip() for p in l.split(',')]
            if len(partes) >= 5 and partes[0].isdigit() and len(partes[0]) == 6:
                try:
                    filas.append({
                        'Date':   partes[0],
                        'Mkt_RF': float(partes[1]) / 100,
                        'SMB':    float(partes[2]) / 100,
                        'HML':    float(partes[3]) / 100,
                        'RF':     float(partes[4]) / 100,
                    })
                except ValueError:
                    continue  # saltar líneas con valores no numéricos

        if not filas:
            raise ValueError("No se pudieron parsear datos mensuales del CSV")

        df = pd.DataFrame(filas)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m')
        df = df.set_index('Date')
        df.index = df.index + pd.offsets.MonthEnd(0)
        df = _normalizar_ff3(df)
        logs.append(f"  {len(df)} períodos via ZIP Ken French")
        return df, logs

    except Exception as e:
        logs.append(f"  Método 1 falló: {type(e).__name__}: {str(e)[:100]}")

    # ------------------------------------------------------------------
    # Método 2: proxies ETF desde Yahoo Finance
    # ------------------------------------------------------------------
    try:
        logs.append("  Método 2: proxies ETF Yahoo Finance (SPY/IWM/IVE/IVW)...")
        import yfinance as yf

        precios_p = {}
        for ticker in ['SPY', 'IWM', 'IVE', 'IVW']:
            datos = yf.download(ticker, start=fecha_inicio, end=fecha_fin,
                                interval='1mo', auto_adjust=True, progress=False)
            if datos.empty:
                raise ValueError(f"Sin datos para {ticker}")
            if isinstance(datos.columns, pd.MultiIndex):
                serie = datos[('Close', ticker)]
            else:
                serie = datos['Close']
            precios_p[ticker] = serie.squeeze()

        rf_raw = yf.download('^IRX', start=fecha_inicio, end=fecha_fin,
                             interval='1mo', auto_adjust=True, progress=False)
        if isinstance(rf_raw.columns, pd.MultiIndex):
            rf_serie = rf_raw[('Close', '^IRX')]
        else:
            rf_serie = rf_raw['Close']
        rf_mens = (rf_serie.squeeze() / 100 / 12).ffill()  # fix: .ffill() no method=

        ret = {t: np.log(s / s.shift(1)).dropna() for t, s in precios_p.items()}

        idx = ret['SPY'].index
        for t in ['IWM', 'IVE', 'IVW']:
            idx = idx.intersection(ret[t].index)
        rf_a = rf_mens.reindex(idx).ffill().fillna(rf_mens.mean())

        ff3_proxy = pd.DataFrame({
            'Mkt_RF': ret['SPY'].reindex(idx) - rf_a,
            'SMB':    ret['IWM'].reindex(idx) - ret['SPY'].reindex(idx),
            'HML':    ret['IVE'].reindex(idx) - ret['IVW'].reindex(idx),
            'RF':     rf_a,
        }, index=idx)

        ff3_proxy = _normalizar_ff3(ff3_proxy)
        logs.append(f"  {len(ff3_proxy)} períodos via proxies ETF")
        logs.append("  NOTA: aproximación con ETFs — SMB/HML pueden diferir de factores oficiales.")
        return ff3_proxy, logs

    except Exception as e:
        logs.append(f"  Método 2 falló: {type(e).__name__}: {str(e)[:100]}")

    # ------------------------------------------------------------------
    # Método 3: degradación controlada
    # ------------------------------------------------------------------
    logs.append("  ❌ FF3 no disponible — métodos fallaron.")
    logs.append("  CAPM y APT continúan normalmente.")
    return pd.DataFrame(), logs


# =============================================================================
# FUNCIÓN 4 — Fama-French 3 Factores
# =============================================================================

def estimar_ff3(retornos: pd.DataFrame, factores_ff3: pd.DataFrame,
                alpha: float) -> tuple[pd.DataFrame, list]:
    """
    FF3: (rᵢ - rf) = α + β₁(Mkt-RF) + β₂SMB + β₃HML + ε
    CORRECCIÓN: índice de ff3 normalizado al primer día del mes.
    """
    logs = ["BLOQUE 2 — FAMA-FRENCH 3 FACTORES"]

    if factores_ff3.empty:
        logs.append("  FF3 no disponible. Omitiendo modelo.")
        return pd.DataFrame(), logs

    # Normalizar al primer día del mes para alineación con retornos
    ff3 = factores_ff3.copy()
    ff3.index = ff3.index.to_period('M').to_timestamp()
    ff3.index = ff3.index + pd.offsets.MonthBegin(0)

    interseccion = retornos.index.intersection(ff3.index)
    logs.append(f"  Observaciones alineadas: {len(interseccion)}")

    if len(interseccion) < MIN_OBSERVACIONES_CAPM:
        logs.append(f"  ERROR: insuficientes obs ({len(interseccion)} < {MIN_OBSERVACIONES_CAPM})")
        return pd.DataFrame(), logs

    resultados = []

    for ticker in retornos.columns:
        serie = retornos[ticker].copy()
        rf_ff = ff3['RF']
        idx   = serie.index.intersection(ff3.index)
        y     = serie.loc[idx] - rf_ff.loc[idx]

        X_df  = pd.DataFrame({
            'Mkt_RF': ff3.loc[idx, 'Mkt_RF'],
            'SMB':    ff3.loc[idx, 'SMB'],
            'HML':    ff3.loc[idx, 'HML'],
        })
        X = sm.add_constant(X_df)

        modelo   = sm.OLS(y, X).fit()
        residuos = modelo.resid
        pruebas  = pruebas_residuos(residuos, X, alpha, 'FF3', ticker)

        alpha_j = float(modelo.params.get('const',   modelo.params.iloc[0]))
        b_mkt   = float(modelo.params.get('Mkt_RF',  modelo.params.iloc[1]))
        b_smb   = float(modelo.params.get('SMB',     modelo.params.iloc[2]))
        b_hml   = float(modelo.params.get('HML',     modelo.params.iloc[3]))
        r2      = modelo.rsquared
        r2_adj  = modelo.rsquared_adj
        alpha_anual = alpha_j * FACTOR_ANUALIZACION  # anualizado por convención estándar (×factor), no definición original de Jensen (1968)

        estilo_smb = "Small-cap" if b_smb > 0 else "Large-cap"
        estilo_hml = "Valor"     if b_hml > 0 else "Crecimiento"

        logs.append(f"  {ticker}: α={alpha_anual:.4f} | β_Mkt={b_mkt:.4f} | "
                    f"β_SMB={b_smb:.4f}({estilo_smb}) | β_HML={b_hml:.4f}({estilo_hml}) | "
                    f"R²={r2:.4f} | BLUE={pruebas['BLUE']}")

        resultados.append({
            'Ticker':      ticker,
            'Alpha_anual': round(alpha_anual, 4),
            'p_alpha':     round(float(modelo.pvalues.iloc[0]), 4),
            'Beta_Mkt':    round(b_mkt, 4),
            'Beta_SMB':    round(b_smb, 4),
            'Beta_HML':    round(b_hml, 4),
            'Estilo_SMB':  estilo_smb,
            'Estilo_HML':  estilo_hml,
            'R2':          round(r2, 4),
            'R2_adj':      round(r2_adj, 4),
            'BLUE':        pruebas['BLUE'],
            'Semaforo':    pruebas['Semaforo'],
            'BP_pvalue':   pruebas['BP_pvalue'],
            'BG_pvalue':   pruebas['BG_pvalue'],
        })

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 5 — Descarga factores macro (APT)
# =============================================================================

def descargar_factores_macro(tickers_macro: dict,
                              fecha_inicio: str, fecha_fin: str,
                              frecuencia: str,
                              indice_retornos: pd.DatetimeIndex) -> tuple[pd.DataFrame, list]:
    """
    Descarga los factores macro candidatos desde Yahoo Finance.
    Calcula retornos logarítmicos de cada factor.
    """
    import yfinance as yf
    logs   = [f"DESCARGA FACTORES MACRO — {list(tickers_macro.keys())}"]
    series = {}

    for ticker, nombre in tickers_macro.items():
        try:
            datos = yf.download(
                ticker, start=fecha_inicio, end=fecha_fin,
                interval=frecuencia, auto_adjust=True, progress=False,
            )
            if datos.empty:
                logs.append(f"  {ticker:<20} SIN DATOS — omitido")
                continue

            if isinstance(datos.columns, pd.MultiIndex):
                serie = datos[('Close', ticker)]
            else:
                serie = datos['Close']
            if isinstance(serie, pd.DataFrame):
                serie = serie.squeeze()

            retorno_factor = np.log(serie / serie.shift(1)).dropna()
            series[ticker] = retorno_factor
            logs.append(f"  {ticker:<20} {nombre:<35} {len(retorno_factor)} obs")

        except Exception as e:
            logs.append(f"  {ticker:<20} ERROR: {e}")

    if not series:
        raise ValueError("No se descargó ningún factor macro.")

    df = pd.DataFrame(series)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().reindex(indice_retornos).dropna(how='all')
    logs.append(f"  Observaciones comunes con portafolio: {len(df)}")

    return df, logs


# =============================================================================
# FUNCIÓN 6 — VIF
# =============================================================================

def calcular_vif(X: pd.DataFrame) -> pd.Series:
    """VIF para cada regresor. VIF > VIF_UMBRAL indica multicolinealidad.
    CORRECCIÓN (auditoría P1.2→P3.1): se agrega columna constante antes de
    calcular, como exige la documentación de statsmodels para que el VIF
    sea el centrado/estándar (sin esto, el resultado es el VIF "no centrado").
    """
    X_const = sm.add_constant(X)
    X_vals  = X_const.values
    vifs = [variance_inflation_factor(X_vals, i)
            for i in range(1, X_vals.shape[1])]  # se omite la columna de la constante
    return pd.Series(vifs, index=X.columns)


# =============================================================================
# FUNCIÓN 7 — Selección de factores APT (PCA + stepwise AIC)
# =============================================================================

def seleccionar_factores_apt(retornos: pd.DataFrame,
                              factores_macro: pd.DataFrame,
                              alpha_stat: float) -> tuple[dict, pd.DataFrame, pd.DataFrame, list]:
    """
    PCA + Forward Selection por AIC por activo.
    CORRECCIÓN v0.5.1: modelo base = constante sola (sin vector de ceros).
    """
    logs = [f"SELECCIÓN APT — PCA + Forward Selection AIC",
            f"Varianza mínima PCA: {APT_VARIANZA_MINIMA_PCA*100:.0f}% | VIF máx: {VIF_UMBRAL}",
            "NOTA: los coeficientes del modelo son sobre componentes principales (PCs), "
            "no sobre los factores macro originales. Ver tabla de loadings para interpretar "
            "qué factores económicos representa cada PC."]

    idx_comun = retornos.index.intersection(factores_macro.index)
    F         = factores_macro.loc[idx_comun].dropna(axis=1, how='any')

    if F.shape[1] < 2:
        raise ValueError("Se necesitan ≥2 factores candidatos para PCA.")

    scaler    = StandardScaler()
    F_scaled  = scaler.fit_transform(F)
    pca       = PCA()
    F_pca_all = pca.fit_transform(F_scaled)

    varianza_acum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = int(np.searchsorted(varianza_acum, APT_VARIANZA_MINIMA_PCA) + 1)
    n_comp = max(APT_MIN_FACTORES, min(n_comp, APT_MAX_FACTORES))

    F_pca = pd.DataFrame(
        F_pca_all[:, :n_comp],
        index=idx_comun,
        columns=[f'PC{i+1}' for i in range(n_comp)]
    )
    logs.append(f"  PCA: {n_comp} componentes → {varianza_acum[n_comp-1]*100:.1f}% varianza")

    loadings = pd.DataFrame(
        pca.components_[:n_comp, :],
        columns=F.columns,
        index=[f'PC{i+1}' for i in range(n_comp)]
    )

    seleccion_por_activo = {}

    for ticker in retornos.columns:
        y         = retornos[ticker].reindex(idx_comun).dropna()
        idx_valid = y.index

        factores_disponibles   = list(F_pca.columns)
        factores_seleccionados = []

        # CORRECCIÓN: base = constante sola
        X_base = sm.add_constant(
            pd.Series(np.ones(len(idx_valid)), index=idx_valid, name='const'),
            has_constant='add'
        )
        try:
            aic_actual = sm.OLS(y, X_base).fit().aic
        except Exception:
            aic_actual = np.inf

        for _ in range(APT_MAX_FACTORES):
            mejor_aic    = aic_actual
            mejor_factor = None

            for factor in factores_disponibles:
                candidatos = factores_seleccionados + [factor]
                X_cand     = sm.add_constant(F_pca.loc[idx_valid, candidatos])
                try:
                    aic_cand = sm.OLS(y, X_cand).fit().aic
                    if aic_cand < mejor_aic:
                        mejor_aic    = aic_cand
                        mejor_factor = factor
                except Exception:
                    continue

            if mejor_factor is None:
                break

            factores_seleccionados.append(mejor_factor)
            factores_disponibles.remove(mejor_factor)
            aic_actual = mejor_aic

        # Verificación VIF
        if len(factores_seleccionados) > 1:
            X_final = F_pca.loc[idx_valid, factores_seleccionados]
            vif     = calcular_vif(X_final)
            ok      = vif[vif <= VIF_UMBRAL].index.tolist()
            if len(ok) < len(factores_seleccionados):
                eliminados = set(factores_seleccionados) - set(ok)
                logs.append(f"  {ticker}: VIF>{VIF_UMBRAL} en {eliminados} → eliminados")
                factores_seleccionados = ok

        if not factores_seleccionados:
            factores_seleccionados = ['PC1']

        seleccion_por_activo[ticker] = {
            'factores': factores_seleccionados,
            'X_pca':    F_pca.loc[idx_valid, factores_seleccionados],
        }
        logs.append(f"  {ticker}: factores seleccionados = {factores_seleccionados}")

    return seleccion_por_activo, F_pca, loadings, logs


# =============================================================================
# FUNCIÓN 8 — Estimación APT
# =============================================================================

def estimar_apt(retornos: pd.DataFrame, seleccion: dict,
                alpha: float) -> tuple[pd.DataFrame, list]:
    """
    APT: rᵢ = α + Σ βⱼFⱼ + ε (factores = PCs seleccionados por AIC).
    Ross (1976), Chen, Roll & Ross (1986).
    """
    logs       = ["BLOQUE 3 — APT CON FACTORES MACRO"]
    resultados = []

    for ticker in retornos.columns:
        info     = seleccion[ticker]
        factores = info['factores']
        X_pca    = info['X_pca']

        y         = retornos[ticker].reindex(X_pca.index).dropna()
        idx_valid = y.index
        X_df      = X_pca.loc[idx_valid]
        X         = sm.add_constant(X_df)

        modelo   = sm.OLS(y, X).fit()
        residuos = modelo.resid
        pruebas  = pruebas_residuos(residuos, X, alpha, 'APT', ticker)

        alpha_j  = float(modelo.params.iloc[0])
        betas    = modelo.params.iloc[1:].to_dict()
        r2       = modelo.rsquared
        r2_adj   = modelo.rsquared_adj
        alpha_anual = alpha_j * FACTOR_ANUALIZACION  # anualizado por convención estándar (×factor), no definición original de Jensen (1968)

        logs.append(f"  {ticker}: α={alpha_anual:.4f} | factores={factores} | "
                    f"R²={r2:.4f} | BLUE={pruebas['BLUE']}")

        fila = {
            'Ticker':      ticker,
            'Alpha_anual': round(alpha_anual, 4),
            'p_alpha':     round(float(modelo.pvalues.iloc[0]), 4),
            'Factores':    str(factores),
            'n_factores':  len(factores),
            'R2':          round(r2, 4),
            'R2_adj':      round(r2_adj, 4),
            'AIC':         round(float(modelo.aic), 4),
            'BIC':         round(float(modelo.bic), 4),
            'BLUE':        pruebas['BLUE'],
            'Semaforo':    pruebas['Semaforo'],
        }
        for factor, beta in betas.items():
            fila[f'Beta_{factor}'] = round(float(beta), 4)
        resultados.append(fila)

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 9 — Tabla comparativa
# =============================================================================

def tabla_comparativa(res_capm: pd.DataFrame, res_ff3: pd.DataFrame,
                       res_apt: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Tabla CAPM vs FF3 vs APT por ticker."""
    logs  = ["TABLA COMPARATIVA — CAPM vs FF3 vs APT"]
    filas = []

    for ticker in res_capm['Ticker'].tolist():
        row_capm = res_capm[res_capm['Ticker'] == ticker].iloc[0]
        filas.append({'Ticker': ticker, 'Modelo': 'CAPM',
                      'Alpha_anual': row_capm['Alpha_anual'],
                      'R2': row_capm['R2'], 'BLUE': row_capm['BLUE']})

        if not res_ff3.empty and ticker in res_ff3['Ticker'].values:
            row_ff3 = res_ff3[res_ff3['Ticker'] == ticker].iloc[0]
            filas.append({'Ticker': ticker, 'Modelo': 'FF3',
                          'Alpha_anual': row_ff3['Alpha_anual'],
                          'R2': row_ff3['R2'], 'BLUE': row_ff3['BLUE']})

        if ticker in res_apt['Ticker'].values:
            row_apt = res_apt[res_apt['Ticker'] == ticker].iloc[0]
            filas.append({'Ticker': ticker, 'Modelo': 'APT',
                          'Alpha_anual': row_apt['Alpha_anual'],
                          'R2': row_apt['R2'], 'BLUE': row_apt['BLUE']})

        r2_vals = [(m, f['R2']) for m, f in zip(
            ['CAPM', 'FF3', 'APT'],
            [row_capm,
             res_ff3[res_ff3['Ticker'] == ticker].iloc[0] if not res_ff3.empty and ticker in res_ff3['Ticker'].values else {'R2': 0},
             res_apt[res_apt['Ticker'] == ticker].iloc[0] if ticker in res_apt['Ticker'].values else {'R2': 0}]
        )]
        mejor = max(r2_vals, key=lambda x: x[1])
        logs.append(f"  {ticker}: mejor modelo = {mejor[0]} (R²={mejor[1]:.4f})")

    return pd.DataFrame(filas), logs


# =============================================================================
# FUNCIÓN 10 — Gráfico Plotly (reemplaza graficar_betas matplotlib)
# =============================================================================

def graficar_betas_plotly(res_capm: pd.DataFrame, res_ff3: pd.DataFrame,
                           res_apt: pd.DataFrame) -> go.Figure:
    """
    Panel izquierdo: Beta de mercado CAPM vs FF3.
    Panel derecho: Alpha anual CAPM vs FF3 vs APT.
    """
    tickers = res_capm['Ticker'].tolist()
    x       = list(range(len(tickers)))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Beta de mercado: CAPM vs FF3",
                        "Alpha de Jensen por modelo (anual)"),
    )

    # Beta de mercado
    # CORRECCIÓN (auditoría P1.2→P3.1): búsqueda robusta de la columna de beta
    # de mercado en vez de asumir el nombre exacto 'Beta_Mkt'.
    def _col_beta_mkt(df):
        return next((c for c in df.columns if 'beta' in c.lower() and 'mkt' in c.lower()), None)

    betas_capm   = res_capm['Beta'].tolist()
    col_beta_ff3 = _col_beta_mkt(res_ff3) if not res_ff3.empty else None
    betas_ff3    = (res_ff3[col_beta_ff3].tolist() if col_beta_ff3
                    else [0] * len(tickers))

    fig.add_trace(go.Bar(
        x=tickers, y=betas_capm, name='CAPM β mercado',
        marker_color=COLORES['portafolio_optimo'], opacity=0.85,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=tickers, y=betas_ff3, name='FF3 β mercado',
        marker_color=COLORES['minima_varianza'], opacity=0.85,
    ), row=1, col=1)
    fig.add_hline(y=1, line_dash='dash', line_color='black',
                  line_width=1, row=1, col=1)

    # Alpha por modelo
    alphas_capm = res_capm.set_index('Ticker')['Alpha_anual'].reindex(tickers).tolist()
    alphas_ff3  = (res_ff3.set_index('Ticker')['Alpha_anual'].reindex(tickers).tolist()
                   if not res_ff3.empty else [0] * len(tickers))
    alphas_apt  = (res_apt.set_index('Ticker')['Alpha_anual'].reindex(tickers).fillna(0).tolist()
                   if not res_apt.empty else [0] * len(tickers))

    fig.add_trace(go.Bar(x=tickers, y=alphas_capm, name='α CAPM',
                         marker_color=COLORES['portafolio_optimo'], opacity=0.85),
                  row=1, col=2)
    fig.add_trace(go.Bar(x=tickers, y=alphas_ff3, name='α FF3',
                         marker_color=COLORES['minima_varianza'], opacity=0.85),
                  row=1, col=2)
    fig.add_trace(go.Bar(x=tickers, y=alphas_apt, name='α APT',
                         marker_color=COLORES['benchmark'], opacity=0.85),
                  row=1, col=2)
    fig.add_hline(y=0, line_dash='dash', line_color='black',
                  line_width=1, row=1, col=2)

    fig.update_layout(
        barmode='group',
        height=440,
        legend=dict(orientation='h', y=-0.18),
        yaxis=dict(title='Beta'),
        yaxis2=dict(title='Alpha anual'),
    )
    return fig


# =============================================================================
# FUNCIÓN 11 — Asistente Fase 3b
# =============================================================================

def asistente_fase3b(res_capm: pd.DataFrame, res_ff3: pd.DataFrame,
                      res_apt: pd.DataFrame, nivel: str = 'basico') -> str:
    """
    Interpreta los resultados de los modelos de factores.
    Sigue el contrato: qué ocurrió → por qué importa → valoración → implicación → acción.
    """
    lineas = []
    def add(t=""): lineas.append(t)

    tickers    = res_capm['Ticker'].tolist() if 'Ticker' in res_capm.columns else list(res_capm.index)
    tiene_ff3  = not res_ff3.empty
    tiene_apt  = not res_apt.empty

    add("## Modelos de Factores de Riesgo")
    add()

    if nivel == 'basico':
        add("### ¿Qué explica el retorno de cada activo?")
        add("Tres modelos intentan responder esa pregunta, cada uno con más detalle que el anterior:")
        add("- **CAPM:** el mercado en su conjunto explica el retorno del activo")
        add("- **Fama-French:** el mercado + el tamaño de la empresa + si es de valor o crecimiento")
        add("- **APT:** factores macroeconómicos identificados automáticamente con los datos")
        add()

        add("### ¿Qué tan sensible es cada activo al mercado?")
        for _, row in res_capm.iterrows():
            ticker = row['Ticker'] if 'Ticker' in row else row.name
            beta   = row['Beta']
            r2     = row['R2']
            if beta > 1.1:
                desc = f"amplifica los movimientos del mercado — cuando el índice sube 1%, este tiende a subir más (y a caer más también)"
            elif beta < 0.9:
                desc = f"amortigua los movimientos del mercado — sube menos cuando el mercado sube, pero cae menos también"
            else:
                desc = f"sigue al mercado de cerca"
            add(f"**{ticker}** (β={beta:.2f}): {desc}. "
                f"El mercado explica el **{r2*100:.0f}%** de sus movimientos.")
        add()

        no_blue = res_capm[res_capm['BLUE'] != 'CUMPLE']['Ticker'].tolist() if 'BLUE' in res_capm.columns else []
        if no_blue:
            add(f"⚠️ En `{no_blue}` la regresión CAPM no cumple todos sus supuestos estadísticos. "
                "Los coeficientes son orientativos — los márgenes de error pueden estar subestimados.")
            add()

        if tiene_ff3:
            add("### ¿Explican más factores el comportamiento de los activos?")
            for _, row in res_ff3.iterrows():
                ticker  = row['Ticker'] if 'Ticker' in row else row.name
                r2_ff3  = row['R2']
                r2_capm = res_capm[res_capm['Ticker'] == ticker]['R2'].values[0] if 'Ticker' in res_capm.columns else res_capm.loc[ticker, 'R2']
                mejora  = r2_ff3 - r2_capm
                estilo_smb = row.get('Estilo_SMB', '')
                estilo_hml = row.get('Estilo_HML', '')
                add(f"**{ticker}:** Agregar tamaño y estilo sube el poder explicativo del "
                    f"{r2_capm*100:.0f}% al {r2_ff3*100:.0f}% (+{mejora*100:.0f}pp). "
                    f"Perfil: {estilo_smb}, {estilo_hml}.")
            add()

        add("**Acción:** usa los perfiles beta para entender cómo se comportará el portafolio "
            "en una caída del mercado — activos con beta alto caerán más.")

    else:
        add("### Comparación R² ajustado por modelo")
        add("| Ticker | R²adj CAPM | R²adj FF3 | R²adj APT | Mejor |")
        add("|---|---|---|---|---|")
        for ticker in tickers:
            def get_r2(df, t):
                if df.empty: return 0.0
                col = 'R2_adj' if 'R2_adj' in df.columns else 'R2'
                idx_col = 'Ticker' if 'Ticker' in df.columns else None
                if idx_col:
                    row = df[df[idx_col] == t]
                    return float(row[col].values[0]) if not row.empty else 0.0
                return float(df.loc[t, col]) if t in df.index else 0.0

            r2_c = get_r2(res_capm, ticker)
            r2_f = get_r2(res_ff3, ticker)  if tiene_ff3 else 0.0
            r2_a = get_r2(res_apt, ticker)  if tiene_apt else 0.0
            mejor = max([('CAPM', r2_c), ('FF3', r2_f), ('APT', r2_a)], key=lambda x: x[1])
            ff3_str = f"{r2_f:.3f}" if tiene_ff3 else "N/D"
            apt_str = f"{r2_a:.3f}" if tiene_apt else "N/D"
            add(f"| {ticker} | {r2_c:.3f} | {ff3_str} | {apt_str} | **{mejor[0]}** |")

        add()
        add("### Alphas de Jensen (anualizado por convención estándar)")
        add("| Ticker | α CAPM | p-valor | α FF3 | α APT | Significativo |")
        add("|---|---|---|---|---|---|")
        for ticker in tickers:
            def get_val(df, t, col, default='—'):
                if df.empty: return default
                idx_col = 'Ticker' if 'Ticker' in df.columns else None
                if idx_col:
                    row = df[df[idx_col] == t]
                    return f"{float(row[col].values[0]):.4f}" if not row.empty and col in row.columns else default
                return f"{float(df.loc[t, col]):.4f}" if t in df.index and col in df.columns else default

            a_capm = get_val(res_capm, ticker, 'Alpha_anual')
            p_capm = get_val(res_capm, ticker, 'p_alpha')
            a_ff3  = get_val(res_ff3, ticker, 'Alpha_anual') if tiene_ff3 else 'N/D'
            a_apt  = get_val(res_apt, ticker, 'Alpha_anual') if tiene_apt else 'N/D'
            try:
                sig = "✅ Sí" if float(p_capm) < 0.05 else "No"
            except (ValueError, TypeError):
                sig = "—"
            add(f"| {ticker} | {a_capm} | {p_capm} | {a_ff3} | {a_apt} | {sig} |")

        add()
        add("### Supuestos de residuos (BLUE)")
        for modelo, df in [('CAPM', res_capm), ('FF3', res_ff3), ('APT', res_apt)]:
            if df is None or df.empty or 'BLUE' not in df.columns: continue
            idx_col = 'Ticker' if 'Ticker' in df.columns else None
            violan  = df[df['BLUE'] == 'VIOLA SUPUESTOS']['Ticker'].tolist() if idx_col else []
            advert  = df[df['BLUE'] == 'ADVERTENCIA']['Ticker'].tolist()    if idx_col else []
            if violan:
                add(f"- **{modelo}** `{violan}`: viola supuestos Gauss-Markov — estimadores MCO no son BLUE.")
            if advert:
                add(f"- **{modelo}** `{advert}`: advertencia en 1 supuesto.")
            if not violan and not advert:
                add(f"- **{modelo}**: todos los residuos cumplen supuestos BLUE.")

    add()
    add("---")
    if nivel == 'basico':
        add("*La beta mide sensibilidad histórica — no garantiza el comportamiento futuro. "
            "Úsala para entender el perfil de riesgo, no para predecir.*")
    else:
        add("*Próximo paso: Fase 3c — Métricas de riesgo (VaR, CVaR, backtesting).*")

    return "\n".join(lineas)


# =============================================================================
# PIPELINE — Fase 3b completa
# =============================================================================

def pipeline_factores(retornos: pd.DataFrame,
                       retornos_benchmark: pd.DataFrame,
                       rf_anual: float,
                       frecuencia: str = FRECUENCIA_DEFAULT,
                       nivel_asistente: str = 'basico',
                       factor: int = FACTOR_ANUALIZACION) -> dict:
    """
    Ejecuta CAPM, FF3 y APT para todos los activos.

    Parámetros
    ----------
    retornos / retornos_benchmark : de session_state
    rf_anual : de session_state['res_3a']['rf_anual']
    frecuencia : de session_state['frecuencia']

    Retorna
    -------
    dict con: res_capm, res_ff3, res_apt, tabla_comp, loadings_pca,
              fig_betas, narrativa_3b, logs, error
    """
    logs_totales = []

    def _log(msgs):
        if isinstance(msgs, list):
            logs_totales.extend(msgs)
        else:
            logs_totales.append(msgs)

    try:
        rf_mes = rf_anual / factor
        fecha_inicio = retornos.index[0].strftime('%Y-%m-%d')
        fecha_fin    = retornos.index[-1].strftime('%Y-%m-%d')

        # CAPM
        res_capm, lg = estimar_capm(retornos, retornos_benchmark, rf_mes, ALPHA)
        _log(lg)

        # FF3
        factores_ff3, lg = descargar_ff3_v2(fecha_inicio, fecha_fin)
        _log(lg)
        res_ff3, lg = estimar_ff3(retornos, factores_ff3, ALPHA)
        _log(lg)

        # APT
        factores_macro, lg = descargar_factores_macro(
            FACTORES_MACRO_CANDIDATOS, fecha_inicio, fecha_fin,
            frecuencia, retornos.index,
        )
        _log(lg)

        seleccion, F_pca, loadings_pca, lg = seleccionar_factores_apt(
            retornos, factores_macro, ALPHA
        )
        _log(lg)

        res_apt, lg = estimar_apt(retornos, seleccion, ALPHA)
        _log(lg)

        # Tabla comparativa
        tabla_comp, lg = tabla_comparativa(res_capm, res_ff3, res_apt)
        _log(lg)

        # Gráfico
        fig_betas = graficar_betas_plotly(res_capm, res_ff3, res_apt)

        # Asistente
        narrativa = asistente_fase3b(res_capm, res_ff3, res_apt, nivel_asistente)

        _log("✅ FASE 3b COMPLETADA")

        return {
            'res_capm':     res_capm,
            'res_ff3':      res_ff3,
            'res_apt':      res_apt,
            'tabla_comp':   tabla_comp,
            'loadings_pca': loadings_pca,
            'fig_betas':    fig_betas,
            'narrativa_3b': narrativa,
            'logs':         logs_totales,
            'error':        None,
        }

    except Exception as e:
        import traceback
        _log(f"ERROR FATAL en pipeline_factores: {e}")
        _log(traceback.format_exc())
        return {'error': str(e), 'logs': logs_totales}
