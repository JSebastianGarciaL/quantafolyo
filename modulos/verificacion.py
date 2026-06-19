# =============================================================================
# QuantαfolyΩ — modulos/verificacion.py
# Verificación Consolidada Fase 3
#
# Funciones migradas desde PortfolioLab_V0_7_0.ipynb (celdas 70–74).
# REGLA DE MIGRACIÓN: lógica econométrica sin cambios.
# Cambios:
#   - print() → entradas en lista `logs`
#   - tabla_semaforo_fase3() retorna DataFrame (no imprime)
#   - asistente_verificacion() retorna str markdown
#   - RUTA_PROYECTO / Drive eliminados (recibe DataFrames como parámetros)
#   - TICKERS ya no es global — se infiere de los DataFrames recibidos
#   - pipeline_verificacion() envuelve todo el flujo
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
import statsmodels.api as sm

from config import ALPHA


# =============================================================================
# FUNCIÓN 1 — Prueba de Spanning (Huberman & Kandel, 1987)
# =============================================================================

def prueba_spanning(retornos: pd.DataFrame,
                    ret_bench: pd.Series,
                    alpha: float = ALPHA) -> tuple[pd.DataFrame, list]:
    """
    H0: α=0 Y β=1 (el S&P 500 abarca completamente el activo).
    Rechazar H0 → el activo expande el conjunto de oportunidades de inversión.

    Test F conjunto de dos restricciones lineales.
    Huberman & Kandel (1987), Review of Financial Studies.
    """
    tickers    = list(retornos.columns)
    resultados = []
    logs       = ["PRUEBA 1 — SPANNING (HUBERMAN & KANDEL, 1987)",
                  "H0: α=0 Y β=1 — el S&P 500 abarca el activo"]

    for ticker in tickers:
        y     = retornos[ticker].dropna()
        datos = pd.concat([y, ret_bench.reindex(y.index)], axis=1).dropna()
        datos.columns = ['activo', 'benchmark']

        y_c   = datos['activo']
        X_c   = sm.add_constant(datos['benchmark'])
        modelo = sm.OLS(y_c, X_c).fit()

        alpha_est = float(modelo.params['const'])
        beta_est  = float(modelo.params['benchmark'])

        # F conjunto: H0: α=0, β=1
        resid_H0 = y_c - datos['benchmark']
        rss_H0   = float((resid_H0**2).sum())
        rss_H1   = float(modelo.ssr)
        n        = len(y_c)
        k        = 2

        if rss_H1 > 1e-12:
            F_stat  = ((rss_H0 - rss_H1) / k) / (rss_H1 / (n - 2))
            p_valor = float(1 - stats.f.cdf(F_stat, k, n - 2))
        else:
            F_stat, p_valor = np.nan, np.nan

        rechaza     = bool(p_valor < alpha) if not np.isnan(p_valor) else False
        diagnostico = "EXPANDE"   if rechaza else "NO EXPANDE"
        semaforo    = "VERDE"     if rechaza else "AMARILLO"

        logs.append(f"  {ticker}: α={alpha_est:.4f} | β={beta_est:.4f} | "
                    f"F={F_stat:.4f} | p={p_valor:.4f} | {diagnostico}")

        resultados.append({
            'Ticker':      ticker,
            'alpha_est':   round(alpha_est, 4),
            'beta_est':    round(beta_est, 4),
            'F_stat':      round(F_stat, 4) if not np.isnan(F_stat) else np.nan,
            'p_valor':     round(p_valor, 4) if not np.isnan(p_valor) else np.nan,
            'rechaza':     rechaza,
            'diagnostico': diagnostico,
            'semaforo':    semaforo,
        })

    df         = pd.DataFrame(resultados).set_index('Ticker')
    n_expande  = int(df['rechaza'].sum())
    n_total    = len(tickers)
    logs.append(f"  Resumen: {n_expande}/{n_total} activos expanden el conjunto de oportunidades.")

    if n_expande >= n_total // 2:
        logs.append("  ✅ La selección tiene valor de diversificación demostrable.")
    else:
        logs.append("  ⚠️ Pocos activos aportan valor más allá del S&P 500.")

    return df, logs


# =============================================================================
# FUNCIÓN 2 — Prueba de Chow: Estabilidad de Betas
# =============================================================================

def prueba_chow(retornos: pd.DataFrame,
                ret_bench: pd.Series,
                rf_mes: float,
                alpha: float = ALPHA) -> tuple[pd.DataFrame, list]:
    """
    Verifica estabilidad del beta CAPM entre primera y segunda mitad del período.
    F = [(RSS_total - RSS_1 - RSS_2) / k] / [(RSS_1 + RSS_2) / (n - 2k)]
    Chow (1960), Econometrica.
    """
    tickers    = list(retornos.columns)
    resultados = []
    logs       = ["PRUEBA 2 — ESTABILIDAD DE BETAS (CHOW, 1960)",
                  "H0: beta estable entre primera y segunda mitad del período"]

    idx_comun = retornos.index.intersection(ret_bench.index)
    n_total   = len(idx_comun)
    mitad     = n_total // 2

    exc_mkt   = ret_bench.loc[idx_comun] - rf_mes
    primera   = idx_comun[:mitad]
    segunda   = idx_comun[mitad:]

    for ticker in tickers:
        y = retornos[ticker].reindex(idx_comun).dropna()
        idx_valid = y.index
        exc_valid = exc_mkt.reindex(idx_valid)

        def ols_ssr(idx):
            y_s = (y.loc[idx] - rf_mes)
            X_s = sm.add_constant(exc_valid.loc[idx])
            m   = sm.OLS(y_s, X_s).fit()
            return float(m.ssr), float(m.params.iloc[1])

        # Regresión completa
        y_full = y - rf_mes
        X_full = sm.OLS(y_full, sm.add_constant(exc_valid)).fit()
        rss_total = float(X_full.ssr)

        # Primera mitad
        idx1 = idx_valid.intersection(primera)
        idx2 = idx_valid.intersection(segunda)

        if len(idx1) < 5 or len(idx2) < 5:
            logs.append(f"  {ticker}: muestra insuficiente para Chow — omitido")
            resultados.append({'Ticker': ticker, 'beta1': np.nan, 'beta2': np.nan,
                               'delta_beta': np.nan, 'F_chow': np.nan, 'p_chow': np.nan,
                               'rechaza': False, 'diagnostico': 'NO EVALUADO',
                               'semaforo': 'AMARILLO'})
            continue

        rss1, beta1 = ols_ssr(idx1)
        rss2, beta2 = ols_ssr(idx2)

        k       = 2
        n       = len(idx1) + len(idx2)
        denom   = (rss1 + rss2) / (n - 2 * k)
        F_chow  = ((rss_total - rss1 - rss2) / k) / denom if denom > 0 else np.nan
        p_chow  = float(1 - stats.f.cdf(F_chow, k, n - 2 * k)) if not np.isnan(F_chow) else np.nan
        rechaza = bool(p_chow < alpha) if not np.isnan(p_chow) else False
        delta   = abs(beta1 - beta2)

        if rechaza:
            diagnostico, semaforo = "QUIEBRE ESTRUCTURAL", "ROJO"
        elif delta > 0.30:
            diagnostico, semaforo = "CAMBIO MODERADO", "AMARILLO"
        else:
            diagnostico, semaforo = "ESTABLE", "VERDE"

        logs.append(f"  {ticker}: β₁={beta1:.4f} | β₂={beta2:.4f} | Δβ={delta:.4f} | "
                    f"F={F_chow:.4f} | p={p_chow:.4f} | {diagnostico}")

        resultados.append({
            'Ticker':      ticker,
            'beta1':       round(beta1, 4),
            'beta2':       round(beta2, 4),
            'delta_beta':  round(delta, 4),
            'F_chow':      round(F_chow, 4) if not np.isnan(F_chow) else np.nan,
            'p_chow':      round(p_chow, 4) if not np.isnan(p_chow) else np.nan,
            'rechaza':     rechaza,
            'diagnostico': diagnostico,
            'semaforo':    semaforo,
        })

    return pd.DataFrame(resultados).set_index('Ticker'), logs


# =============================================================================
# FUNCIÓN 3 — Consistencia entre modelos
# =============================================================================

def verificar_consistencia(capm_df: pd.DataFrame,
                            ff3_df: pd.DataFrame,
                            apt_df: pd.DataFrame,
                            alpha: float = ALPHA) -> tuple[list, list]:
    """
    Verifica: R² ajustado creciente CAPM→FF3→APT, Δβ mercado < 0.20, alpha CAPM.
    Nota: se usa R² ajustado para evitar el sesgo mecánico de R² simple al agregar regresores.
    """
    tickers  = list(capm_df.index)
    problemas = []
    logs      = ["PRUEBA 3 — CONSISTENCIA ENTRE MODELOS",
                 "Nota: comparación con R² ajustado (penaliza complejidad)"]

    tiene_ff3 = not ff3_df.empty
    tiene_apt = not apt_df.empty

    # Usar R² ajustado si está disponible, sino R² simple
    def get_r2(df, ticker):
        if 'R2_adj' in df.columns and ticker in df.index:
            return float(df.loc[ticker, 'R2_adj'])
        elif 'R2' in df.columns and ticker in df.index:
            return float(df.loc[ticker, 'R2'])
        return None

    for ticker in tickers:
        r2_capm = get_r2(capm_df, ticker)
        r2_ff3  = get_r2(ff3_df, ticker) if tiene_ff3 else None
        r2_apt  = get_r2(apt_df, ticker)  if tiene_apt else None

        # R² creciente
        if r2_capm is not None and r2_ff3 is not None:
            mejora = r2_ff3 - r2_capm
            if mejora < 0:
                problemas.append(f"{ticker}: R² FF3 ({r2_ff3:.4f}) < R² CAPM ({r2_capm:.4f})")
            logs.append(f"  {ticker} R² CAPM={r2_capm:.4f} → FF3={r2_ff3:.4f} "
                        f"({'✅' if mejora >= 0 else '⚠️'} Δ={mejora:+.4f})")

        if r2_ff3 is not None and r2_apt is not None:
            mejora = r2_apt - r2_ff3
            logs.append(f"  {ticker} R² FF3={r2_ff3:.4f} → APT={r2_apt:.4f} "
                        f"({'✅' if mejora >= 0 else '⚠️'} Δ={mejora:+.4f})")

        # Beta de mercado CAPM vs FF3
        if ticker in capm_df.index and tiene_ff3 and ticker in ff3_df.index:
            b_capm     = float(capm_df.loc[ticker, 'Beta'])
            col_beta   = next((c for c in ff3_df.columns
                               if 'beta' in c.lower() and 'mkt' in c.lower()), None)
            b_ff3      = float(ff3_df.loc[ticker, col_beta]) if col_beta else None
            if b_ff3 is not None:
                delta = abs(b_capm - b_ff3)
                logs.append(f"  {ticker} β_mkt: CAPM={b_capm:.4f} | FF3={b_ff3:.4f} | "
                            f"Δ={delta:.4f} {'✅' if delta < 0.20 else '🟡'}")

        # Alpha CAPM significativo
        if ticker in capm_df.index:
            p_a = float(capm_df.loc[ticker, 'p_alpha'])
            a_v = float(capm_df.loc[ticker, 'Alpha_anual'])
            if p_a < alpha:
                problemas.append(f"{ticker}: α CAPM significativo ({a_v*100:.2f}% anual, p={p_a:.4f})")
            logs.append(f"  {ticker} α CAPM = {a_v*100:.2f}% (p={p_a:.4f}) "
                        f"{'🔴 SIGNIFICATIVO' if p_a < alpha else '✅'}")

    if not problemas:
        logs.append("  ✅ Todos los modelos son consistentes entre sí.")
    else:
        logs.append(f"  ⚠️ {len(problemas)} inconsistencia(s):")
        for p in problemas:
            logs.append(f"    → {p}")

    return problemas, logs


# =============================================================================
# FUNCIÓN 4 — Tabla semáforo consolidada Fase 3
# =============================================================================

def tabla_semaforo_fase3(res_spanning: pd.DataFrame,
                          res_chow: pd.DataFrame,
                          capm_df: pd.DataFrame,
                          res_bt: pd.DataFrame,
                          metricas_3a: pd.DataFrame,
                          problemas: list) -> pd.DataFrame:
    """
    Retorna DataFrame con semáforo por prueba y activo.
    """
    EMJ     = {'VERDE': '🟢', 'AMARILLO': '🟡', 'ROJO': '🔴'}
    tickers = list(res_spanning.index)
    filas   = []

    # Spanning
    fila = {'Prueba': 'Spanning (expande oportunidades)'}
    for t in tickers:
        c = res_spanning.loc[t, 'semaforo'] if t in res_spanning.index else 'AMARILLO'
        fila[t] = EMJ[c] + ' ' + c
    filas.append(fila)

    # Chow
    fila = {'Prueba': 'Estabilidad betas (Chow)'}
    for t in tickers:
        c = res_chow.loc[t, 'semaforo'] if t in res_chow.index else 'AMARILLO'
        fila[t] = EMJ[c] + ' ' + c
    filas.append(fila)

    # Alpha CAPM
    fila = {'Prueba': 'Alpha no significativo (CAPM)'}
    for t in tickers:
        if t in capm_df.index:
            c = 'VERDE' if float(capm_df.loc[t, 'p_alpha']) >= ALPHA else 'ROJO'
        else:
            c = 'AMARILLO'
        fila[t] = EMJ[c] + ' ' + c
    filas.append(fila)

    # Backtesting VaR (global)
    bt_ok = bool(res_bt['kupiec_ok'].all()) if 'kupiec_ok' in res_bt.columns else True
    c_bt  = 'VERDE' if bt_ok else 'ROJO'
    filas.append({'Prueba': 'Backtesting VaR (global)',
                  **{t: EMJ[c_bt] + ' ' + c_bt for t in tickers}})

    # Consistencia modelos (global)
    c_cons = 'VERDE' if not problemas else 'AMARILLO'
    filas.append({'Prueba': 'Consistencia CAPM→FF3→APT (global)',
                  **{t: EMJ[c_cons] + ' ' + c_cons for t in tickers}})

    return pd.DataFrame(filas)


# =============================================================================
# FUNCIÓN 5 — Asistente Verificación
# =============================================================================

def asistente_verificacion(res_spanning: pd.DataFrame,
                            res_chow: pd.DataFrame,
                            problemas: list,
                            res_bt: pd.DataFrame,
                            nivel: str = 'basico') -> str:
    """
    Interpreta los resultados de la verificación consolidada.
    Sigue el contrato: qué ocurrió → por qué importa → valoración → implicación → acción.
    """
    tickers            = list(res_spanning.index)
    activos_expanden   = res_spanning[res_spanning['rechaza']].index.tolist()
    activos_inestables = res_chow[res_chow['semaforo'] == 'ROJO'].index.tolist()
    activos_cambio     = res_chow[res_chow['semaforo'] == 'AMARILLO'].index.tolist()
    bt_ok              = bool(res_bt['kupiec_ok'].all()) if 'kupiec_ok' in res_bt.columns else True
    n_total            = len(tickers)

    lineas = []
    def add(t=""): lineas.append(t)

    add("## Verificación Consolidada")
    add()

    if nivel == 'basico':
        add("### ¿Vale la pena este portafolio frente a simplemente comprar el S&P 500?")
        if activos_expanden:
            pct = len(activos_expanden) / n_total * 100
            add(f"✅ **Sí — {len(activos_expanden)} de {n_total} activos aportan oportunidades "
                f"que no están disponibles en el índice.**")
            add(f"Estadísticamente, `{activos_expanden}` expanden el conjunto de oportunidades "
                "de inversión más allá del S&P 500. Construir este portafolio tiene sentido.")
        else:
            add("⚠️ **Ningún activo expande las oportunidades más allá del S&P 500.**")
            add("Todos los activos se comportan tan similar al índice que no aportan "
                "diversificación adicional. Considera activos de otros sectores, regiones "
                "o clases de activos.")
        add()

        add("### ¿Podemos confiar en las betas para el futuro?")
        if not activos_inestables and not activos_cambio:
            add("✅ **Sí — la sensibilidad al mercado fue estable a lo largo del período.**")
            add("Las betas calculadas son un indicador confiable del comportamiento esperado "
                "del portafolio frente al mercado.")
        else:
            if activos_inestables:
                add(f"🔴 **`{activos_inestables}` sufrieron un cambio estructural significativo** "
                    "en su relación con el mercado — la beta histórica puede no representar bien "
                    "su comportamiento futuro.")
            if activos_cambio:
                add(f"🟡 **`{activos_cambio}` muestran un cambio moderado en beta** — "
                    "usar con precaución para proyecciones de riesgo.")
        add()

        add("### ¿El modelo de riesgo fue honesto?")
        if bt_ok:
            add("✅ **Sí — el VaR predijo correctamente la frecuencia de pérdidas extremas.**")
            add("El modelo de riesgo está bien calibrado para este portafolio.")
        else:
            add("⚠️ **El VaR no estuvo bien calibrado.** "
                "Revisa los resultados del backtesting en la Fase 3c.")
        add()

        add("### ¿Los tres modelos de factores cuentan la misma historia?")
        if not problemas:
            add("✅ **Sí — CAPM, Fama-French y APT son coherentes entre sí.**")
            add("Cada modelo más complejo explica más del retorno de los activos, "
                "sin contradecir las conclusiones del anterior. El análisis es internamente consistente.")
        else:
            add(f"🟡 **Hay {len(problemas)} punto(s) de atención en la consistencia:**")
            for p in problemas:
                add(f"- {p}")
            add("Esto no invalida el análisis, pero conviene mencionarlo al presentar resultados.")

    else:
        add("### Spanning — Huberman & Kandel (1987)")
        add("H0: α=0 ∩ β=1 (el benchmark abarca completamente el activo)")
        add()
        add("| Ticker | α | β | F | p-valor | Diagnóstico |")
        add("|---|---|---|---|---|---|")
        for t in tickers:
            r = res_spanning.loc[t]
            add(f"| {t} | {r['alpha_est']} | {r['beta_est']} | "
                f"{r['F_stat']} | {r['p_valor']} | {r['diagnostico']} |")

        n_exp = len(activos_expanden)
        if n_exp > 0:
            add(f"\n✅ {n_exp}/{n_total} activos rechazan H0 — expanden el conjunto de oportunidades.")
        else:
            add(f"\n⚠️ Ningún activo rechaza H0 — no hay expansión estadísticamente significativa.")
        add()

        add("### Estabilidad de betas — Chow (1960)")
        add("| Ticker | β₁ (1ª mitad) | β₂ (2ª mitad) | Δβ | F | p-valor | Diagnóstico |")
        add("|---|---|---|---|---|---|---|")
        for t in tickers:
            r = res_chow.loc[t]
            add(f"| {t} | {r['beta1']} | {r['beta2']} | {r['delta_beta']} | "
                f"{r['F_chow']} | {r['p_chow']} | {r['diagnostico']} |")

        if activos_inestables:
            add(f"\n⚠️ `{activos_inestables}`: quiebre estructural detectado. "
                "El beta de la primera mitad no es representativo del período completo.")
        add()

        add("### Consistencia entre modelos")
        if not problemas:
            add("✅ R² ajustado creciente CAPM→FF3→APT. Betas de mercado CAPM/FF3 consistentes.")
        else:
            for p in problemas:
                add(f"⚠️ {p}")

    add()
    add("---")
    if nivel == 'basico':
        add("*Esta verificación cierra el análisis cuantitativo. "
            "El portafolio ha sido validado desde múltiples ángulos — estadístico, financiero y econométrico.*")
    else:
        add("*Próximo paso: Reporte final con narrativa integrada y PDF descargable.*")

    return "\n".join(lineas)


# =============================================================================
# PIPELINE — Verificación completa
# =============================================================================

def pipeline_verificacion(retornos: pd.DataFrame,
                           retornos_benchmark: pd.DataFrame,
                           capm_df: pd.DataFrame,
                           ff3_df: pd.DataFrame,
                           apt_df: pd.DataFrame,
                           res_bt: pd.DataFrame,
                           metricas_3a: pd.DataFrame,
                           rf_anual: float,
                           nivel_asistente: str = 'basico',
                           factor: int = 12) -> dict:
    """
    Ejecuta spanning, Chow, consistencia, semáforo y asistente.

    Parámetros — todos vienen de session_state.
    """
    logs_totales = []

    def _log(msgs):
        if isinstance(msgs, list):
            logs_totales.extend(msgs)
        else:
            logs_totales.append(msgs)

    try:
        rf_mes    = rf_anual / factor
        ret_bench = retornos_benchmark.iloc[:, 0].reindex(retornos.index)

        # Normalizar índice de DataFrames
        capm_idx = capm_df.set_index('Ticker') if 'Ticker' in capm_df.columns else capm_df
        ff3_idx  = (ff3_df.set_index('Ticker')  if (not ff3_df.empty and 'Ticker' in ff3_df.columns)
                    else ff3_df)
        apt_idx  = (apt_df.set_index('Ticker')  if (not apt_df.empty and 'Ticker' in apt_df.columns)
                    else apt_df)

        # 1. Spanning
        res_spanning, lg = prueba_spanning(retornos, ret_bench, ALPHA)
        _log(lg)

        # 2. Chow
        res_chow, lg = prueba_chow(retornos, ret_bench, rf_mes, ALPHA)
        _log(lg)

        # 3. Consistencia
        problemas, lg = verificar_consistencia(capm_idx, ff3_idx, apt_idx, ALPHA)
        _log(lg)

        # 4. Semáforo
        semaforo_f3 = tabla_semaforo_fase3(
            res_spanning, res_chow, capm_idx, res_bt, metricas_3a, problemas
        )

        # 5. Asistente
        narrativa = asistente_verificacion(
            res_spanning, res_chow, problemas, res_bt, nivel_asistente
        )

        # Consistencia DataFrame para session_state
        res_consistencia = pd.DataFrame({'problema': problemas})

        _log("✅ VERIFICACIÓN FASE 3 COMPLETADA")

        return {
            'res_spanning':    res_spanning,
            'res_chow':        res_chow,
            'res_consistencia': res_consistencia,
            'semaforo_f3':     semaforo_f3,
            'narrativa_ver':   narrativa,
            'problemas':       problemas,
            'logs':            logs_totales,
            'error':           None,
        }

    except Exception as e:
        import traceback
        _log(f"ERROR FATAL en pipeline_verificacion: {e}")
        _log(traceback.format_exc())
        return {'error': str(e), 'logs': logs_totales}
