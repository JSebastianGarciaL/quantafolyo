# =============================================================================
# QuantαfolyΩ — modulos/reporte.py
# Fase 4: Asistente Explicativo Integrado + Reporte PDF ejecutivo
#
# Funciones migradas desde PortfolioLab_V0_7_0.ipynb (celdas 78–80).
# REGLA DE MIGRACIÓN: lógica sin cambios.
# Cambios:
#   - asistente_completo() recibe todos los DataFrames como parámetros
#     (en el notebook leía los CSV desde Drive)
#   - grafico_resumen_plotly() reescrito en Plotly (retorna fig)
#   - generar_pdf() retorna bytes en lugar de guardar a Drive
#     (la página usa st.download_button con esos bytes)
#   - TICKERS / N_ACTIVOS / RF_ANUAL → inferidos de los datos recibidos
#   - RUTA_PROYECTO / Drive / files.download eliminados
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import io
from datetime import datetime
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from config import FACTOR_ANUALIZACION, COLORES


# =============================================================================
# FUNCIÓN 1 — Asistente completo (narrativa de principio a fin)
# =============================================================================

def asistente_completo(retornos: pd.DataFrame,
                        res_3a: dict,
                        pesos_df: pd.DataFrame,
                        metricas_3a: pd.DataFrame,
                        res_capm: pd.DataFrame,
                        res_ff3: pd.DataFrame,
                        res_apt: pd.DataFrame,
                        res_var_dict: dict,
                        metricas_3c: dict,
                        res_bt: pd.DataFrame,
                        res_estres: pd.DataFrame,
                        res_spanning: pd.DataFrame,
                        res_chow: pd.DataFrame,
                        res_estac: pd.DataFrame,
                        res_norm: pd.DataFrame,
                        res_arch: pd.DataFrame,
                        res_mardia: dict,
                        nivel: str = 'basico') -> str:
    """
    Genera narrativa completa del análisis de principio a fin.
    Retorna string con formato texto plano (secciones delimitadas).
    El PDF lo convierte a párrafos formateados.
    """
    tickers   = list(retornos.columns)
    n_activos = len(tickers)
    rf_anual  = res_3a.get('rf_anual', 0.04)
    lineas    = []
    sep       = "=" * 70

    def add(texto=""):
        lineas.append(texto)

    # Normalizar índices de factores
    def norm_idx(df):
        if df is None or df.empty:
            return pd.DataFrame()
        return df.set_index('Ticker') if 'Ticker' in df.columns else df.copy()

    capm_local = norm_idx(res_capm)
    ff3_local  = norm_idx(res_ff3)
    apt_local  = norm_idx(res_apt)

    add(sep)
    add("QuantαfolyΩ — REPORTE DE ANÁLISIS DE PORTAFOLIO")
    add(f"Fecha: {datetime.today().strftime('%d/%m/%Y')}")
    add(f"Nivel: {'Básico' if nivel == 'basico' else 'Técnico'}")
    add(sep)

    # Advertencia de muestra pequeña — visible desde el inicio del reporte
    n_obs = len(retornos)
    if n_obs < 36:
        add()
        add("⚠️ ADVERTENCIA DE MUESTRA PEQUEÑA")
        add(f"Este análisis usa {n_obs} observaciones mensuales.")
        add("Los modelos estadísticos requieren al menos 36-60 observaciones para tener")
        add("poder estadístico confiable. Los resultados son orientativos.")
        add("Recomendación: ampliar la ventana temporal a 3-5 años.")

    # -------------------------------------------------------------------------
    # SECCIÓN 1 — El portafolio analizado
    # -------------------------------------------------------------------------
    add()
    add("SECCIÓN 1 — EL PORTAFOLIO ANALIZADO")
    add("-" * 40)

    periodo_inicio = retornos.index[0].strftime('%B %Y')
    periodo_fin    = retornos.index[-1].strftime('%B %Y')
    n_obs          = len(retornos)

    if nivel == 'basico':
        add(f"Se analizaron {n_activos} activos financieros durante el período")
        add(f"{periodo_inicio} — {periodo_fin} ({n_obs} observaciones mensuales).")
        add()
        add(f"Activos incluidos: {', '.join(tickers)}")
        add()
        add("Todos los precios fueron convertidos a dólares estadounidenses (USD)")
        add("para garantizar comparabilidad entre activos de distintos mercados.")
    else:
        add(f"Universo: {n_activos} activos | Frecuencia: mensual | "
            f"n = {n_obs} | Divisa base: USD")
        add(f"Período: {periodo_inicio} — {periodo_fin}")
        add(f"Tickers: {', '.join(tickers)}")
        add("Retornos: logarítmicos r_t = ln(P_t/P_{t-1}) — "
            "aditividad temporal (Campbell et al., 1997)")

    # -------------------------------------------------------------------------
    # SECCIÓN 2 — Validación estadística
    # -------------------------------------------------------------------------
    add()
    add()
    add("SECCIÓN 2 — VALIDACIÓN ESTADÍSTICA DE LOS DATOS")
    add("-" * 40)

    if nivel == 'basico':
        add("Antes de construir cualquier modelo, verificamos que los datos")
        add("cumplen los requisitos estadísticos necesarios.")
        add()

        if not res_estac.empty:
            todos_estac = (res_estac['Semaforo'] == 'VERDE').all()
            if todos_estac:
                add("✅ ESTACIONARIEDAD: Los retornos de todos los activos son")
                add("   estables en el tiempo — condición necesaria para los modelos.")
            else:
                prob = res_estac[res_estac['Semaforo'] != 'VERDE']['Ticker'].tolist()
                add(f"⚠️  ESTACIONARIEDAD: {prob} muestran comportamiento inestable.")

        if not res_norm.empty:
            no_norm  = res_norm[res_norm['Semaforo'] == 'ROJO']['Ticker'].tolist()
            mixtos   = res_norm[res_norm['Semaforo'] == 'AMARILLO']['Ticker'].tolist()
            prob_n   = no_norm + mixtos
            if not prob_n:
                add("✅ NORMALIDAD: Los retornos siguen distribución normal.")
            else:
                add(f"⚠️  NORMALIDAD: {prob_n} no siguen distribución normal.")
                add("   Los eventos extremos son más frecuentes de lo esperado.")
                add("   Por eso complementamos Markowitz con CVaR.")

        if not res_arch.empty:
            col_r    = next((c for c in res_arch.columns if 'rechaza' in c.lower()), None)
            con_arch = res_arch[res_arch[col_r] == True]['Ticker'].tolist() if col_r else []
            if not con_arch:
                add("✅ VOLATILIDAD: Constante en el tiempo para todos los activos.")
            else:
                add(f"⚠️  VOLATILIDAD VARIABLE: {con_arch} muestran clustering.")

        diag_mardia = res_mardia.get('diagnostico', '') if res_mardia else ''
        if 'NORMAL' in diag_mardia and 'NO' not in diag_mardia:
            add("✅ NORMALIDAD CONJUNTA: El portafolio cumple los supuestos de Markowitz.")
        else:
            add("⚠️  NORMALIDAD CONJUNTA: El portafolio presenta desviaciones.")
            add("   Se recomienda interpretar el VaR paramétrico con precaución.")
    else:
        add("Pruebas aplicadas sobre retornos logarítmicos mensuales:")
        add()
        if not res_estac.empty:
            for _, row in res_estac.iterrows():
                add(f"  {row['Ticker']}: ADF p={float(row.get('ADF_pvalue', 0)):.4f} | "
                    f"KPSS p={float(row.get('KPSS_pvalue', 0)):.4f} | "
                    f"{row.get('Diagnostico', '?')}")
        if not res_norm.empty:
            add()
            for _, row in res_norm.iterrows():
                add(f"  {row['Ticker']}: JB p={float(row.get('JB_pvalue', 0)):.4f} | "
                    f"SW p={float(row.get('SW_pvalue', 0)):.4f} | "
                    f"Kurt={float(row.get('Curtosis', 0)):.4f} | "
                    f"{row.get('Diagnostico', '?')}")
        if res_mardia:
            add(f"  Mardia: {res_mardia.get('diagnostico', 'N/D')} | "
                f"p_asim={res_mardia.get('p_asim', 0):.4f} | "
                f"p_kurt={res_mardia.get('p_kurt', 0):.4f}")

    # -------------------------------------------------------------------------
    # SECCIÓN 3 — Markowitz
    # -------------------------------------------------------------------------
    add()
    add()
    add("SECCIÓN 3 — OPTIMIZACIÓN DE PORTAFOLIO (MARKOWITZ, 1952)")
    add("-" * 40)

    mvp_ret  = res_3a.get('mvp',      {}).get('retorno',     0)
    mvp_vol  = res_3a.get('mvp',      {}).get('volatilidad', 0)
    mvp_sr   = res_3a.get('mvp',      {}).get('sharpe',      0)
    tang_ret = res_3a.get('tangente', {}).get('retorno',     0)
    tang_vol = res_3a.get('tangente', {}).get('volatilidad', 0)
    tang_sr  = res_3a.get('tangente', {}).get('sharpe',      0)

    if nivel == 'basico':
        add("Se construyeron dos portafolios óptimos:")
        add()
        add("PORTAFOLIO DE MÍNIMA VARIANZA (MVP)")
        add(f"  Retorno esperado: {mvp_ret*100:.2f}% anual")
        add(f"  Volatilidad:      {mvp_vol*100:.2f}% anual")
        add(f"  Sharpe Ratio:     {mvp_sr:.4f}")
        if not pesos_df.empty and 'MVP' in pesos_df.columns:
            add("  Composición:")
            for t in tickers:
                if t in pesos_df.index:
                    w = float(pesos_df.loc[t, 'MVP'])
                    if w > 0.01:
                        add(f"    {t:<20}: {w*100:.1f}%  {'█' * int(w * 30)}")
        add()
        add("PORTAFOLIO TANGENTE (MÁXIMO SHARPE RATIO)")
        add(f"  Retorno esperado: {tang_ret*100:.2f}% anual")
        add(f"  Volatilidad:      {tang_vol*100:.2f}% anual")
        add(f"  Sharpe Ratio:     {tang_sr:.4f}")
        if not pesos_df.empty and 'Tangente' in pesos_df.columns:
            add("  Composición:")
            for t in tickers:
                if t in pesos_df.index:
                    w = float(pesos_df.loc[t, 'Tangente'])
                    if w > 0.01:
                        add(f"    {t:<20}: {w*100:.1f}%  {'█' * int(w * 30)}")
    else:
        add("Problema: min w'Σw  s.t.  w'μ = μ_p,  Σwᵢ = 1,  wᵢ ≥ 0")
        add()
        add(f"MVP:      μ={mvp_ret*100:.2f}%  σ={mvp_vol*100:.2f}%  SR={mvp_sr:.4f}")
        add(f"Tangente: μ={tang_ret*100:.2f}%  σ={tang_vol*100:.2f}%  SR={tang_sr:.4f}")
        add()
        add("Nota: pesos sensibles a la estimación de μ (Michaud, 1989).")
        add("Limitación: correlaciones aumentan en crisis (Longin & Solnik, 2001).")

    # -------------------------------------------------------------------------
    # SECCIÓN 4 — Modelos de factores
    # -------------------------------------------------------------------------
    add()
    add()
    add("SECCIÓN 4 — MODELOS DE FACTORES DE RIESGO")
    add("-" * 40)

    if nivel == 'basico':
        add("Se estimaron tres modelos para entender qué impulsa el retorno:")
        add()
        add("CAPM — Un factor (el mercado):")
        for t in tickers:
            if t in capm_local.index:
                beta = float(capm_local.loc[t, 'Beta'])
                r2   = float(capm_local.loc[t, 'R2'])
                tipo = "agresivo" if beta > 1.2 else ("defensivo" if beta < 0.8 else "neutral")
                add(f"  {t}: β={beta:.2f} ({tipo}) | R²={r2:.2f}")
        add()
        add("FAMA-FRENCH 3 FACTORES — Mercado + Tamaño + Valor:")
        for t in tickers:
            if t in ff3_local.index:
                r2    = float(ff3_local.loc[t, 'R2'])
                r2c   = float(capm_local.loc[t, 'R2']) if t in capm_local.index else 0.0
                mejora = r2 - r2c
                add(f"  {t}: R²={r2:.2f} (mejora +{mejora:.2f} sobre CAPM)")
        add()
        add("APT — Factores macroeconómicos seleccionados automáticamente:")
        for t in tickers:
            if t in apt_local.index:
                r2       = float(apt_local.loc[t, 'R2'])
                factores = apt_local.loc[t].get('Factores', '?')
                add(f"  {t}: R²={r2:.2f} | Factores: {factores}")
    else:
        add("CAPM (Sharpe, 1964; Lintner, 1965): R_i - Rf = alpha + beta(Rm-Rf) + e")
        for t in tickers:
            if t in capm_local.index:
                add(f"  {t}: alpha={float(capm_local.loc[t, 'Alpha_anual'])*100:.2f}% "
                    f"(p={float(capm_local.loc[t, 'p_alpha']):.4f}) | "
                    f"beta={float(capm_local.loc[t, 'Beta']):.4f} | "
                    f"R²={float(capm_local.loc[t, 'R2']):.4f} | "
                    f"BLUE={capm_local.loc[t, 'BLUE']}")
        add()
        add("FF3 (Fama & French, 1993):")
        for t in tickers:
            if t in ff3_local.index:
                col_mkt = next((c for c in ff3_local.columns
                                if 'beta' in c.lower() and 'mkt' in c.lower()), None)
                b_mkt   = float(ff3_local.loc[t, col_mkt]) if col_mkt else 0.0
                b_smb   = float(ff3_local.loc[t, 'Beta_SMB']) if 'Beta_SMB' in ff3_local.columns else 0.0
                b_hml   = float(ff3_local.loc[t, 'Beta_HML']) if 'Beta_HML' in ff3_local.columns else 0.0
                add(f"  {t}: beta_mkt={b_mkt:.4f} | beta_smb={b_smb:.4f} | "
                    f"beta_hml={b_hml:.4f} | R²={float(ff3_local.loc[t,'R2']):.4f}")

    # -------------------------------------------------------------------------
    # SECCIÓN 5 — Riesgo
    # -------------------------------------------------------------------------
    add()
    add()
    add("SECCIÓN 5 — MÉTRICAS DE RIESGO")
    add("-" * 40)

    var_95  = res_var_dict.get('95%', {}).get('var_hist', 0)
    cvar_95 = res_var_dict.get('95%', {}).get('cvar_hist', 0)
    bt_95   = {}
    if not res_bt.empty and len(res_bt) > 0:
        bt_row = res_bt[res_bt['Nivel'] == '95%'].iloc[0] if 'Nivel' in res_bt.columns else res_bt.iloc[0]
        bt_95  = bt_row.to_dict()

    if nivel == 'basico':
        add(f"VALUE AT RISK (VaR 95%): {var_95*100:.2f}% mensual")
        add(f"  En el 95% de los meses, la pérdida no supera este valor.")
        add()
        add(f"EXPECTED SHORTFALL (CVaR 95%): {cvar_95*100:.2f}% mensual")
        add(f"  En el 5% de los peores meses, la pérdida promedio es este valor.")
        add()
        if metricas_3c:
            add(f"MÉTRICAS DE DESEMPEÑO (Portafolio Tangente):")
            add(f"  Sharpe Ratio:    {metricas_3c.get('sharpe', '—')}")
            add(f"  Sortino Ratio:   {metricas_3c.get('sortino', '—')}")
            add(f"  Calmar Ratio:    {metricas_3c.get('calmar', '—')}")
            add(f"  Omega Ratio:     {metricas_3c.get('omega', '—')}")
            add(f"  Max Drawdown:    {float(metricas_3c.get('max_drawdown', 0))*100:.2f}%")
    else:
        add("VaR y CVaR (portafolio tangente, mensual):")
        for etiq in ['95%', '99%']:
            v = res_var_dict.get(etiq, {})
            add(f"  {etiq}: VaR hist={v.get('var_hist',0):.4f} | "
                f"VaR param={v.get('var_param',0):.4f} | "
                f"VaR MC={v.get('var_mc',0):.4f}")
            add(f"         CVaR hist={v.get('cvar_hist',0):.4f} | "
                f"CVaR param={v.get('cvar_param',0):.4f}")
        add()
        if metricas_3c:
            add(f"Sharpe={metricas_3c.get('sharpe','—')} | "
                f"Sortino={metricas_3c.get('sortino','—')} | "
                f"Treynor={metricas_3c.get('treynor','—')} | "
                f"IR={metricas_3c.get('ir','—')}")

    if not res_estres.empty:
        add()
        add("ANÁLISIS DE ESTRÉS HISTÓRICO:")
        for _, row in res_estres.iterrows():
            add(f"  {row['Escenario']}: ret={row['Ret_acum']*100:.1f}% | "
                f"MDD={row['Max_DD']*100:.1f}%")

    # -------------------------------------------------------------------------
    # SECCIÓN 6 — Verificación
    # -------------------------------------------------------------------------
    add()
    add()
    add("SECCIÓN 6 — VERIFICACIÓN CONSOLIDADA")
    add("-" * 40)

    if not res_spanning.empty:
        n_exp   = int(res_spanning['rechaza'].sum())
        n_total = len(res_spanning)
        if nivel == 'basico':
            add("DIVERSIFICACIÓN (Prueba de Spanning):")
            if n_exp >= n_total // 2:
                add(f"  ✅ {n_exp}/{n_total} activos expanden el conjunto de oportunidades.")
            else:
                add(f"  ⚠️  Solo {n_exp}/{n_total} activos aportan valor más allá del S&P 500.")
        else:
            add(f"Spanning (Huberman & Kandel, 1987): {n_exp}/{n_total} activos expanden.")

    if not res_chow.empty:
        n_inest = int((res_chow['semaforo'] == 'ROJO').sum())
        if nivel == 'basico':
            add()
            add("ESTABILIDAD DE BETAS (Prueba de Chow):")
            if n_inest == 0:
                add("  ✅ Betas estables — confiables para proyecciones futuras.")
            else:
                add(f"  ⚠️  {n_inest} activo(s) con quiebre estructural en beta.")
        else:
            add(f"Chow (1960): {n_inest} activos con quiebre estructural.")

    # -------------------------------------------------------------------------
    # SECCIÓN 7 — Conclusiones
    # -------------------------------------------------------------------------
    add()
    add()
    add("SECCIÓN 7 — CONCLUSIONES")
    add("-" * 40)

    if nivel == 'basico':
        add("El análisis completo muestra que:")
        add()
        add(f"1. Los {n_activos} activos seleccionados forman un portafolio")
        add("   estadísticamente válido y bien diversificado.")
        add()
        add(f"2. El portafolio tangente (máximo Sharpe Ratio) ofrece")
        add(f"   {tang_ret*100:.2f}% de retorno esperado anual con "
            f"{tang_vol*100:.2f}% de volatilidad.")
        add()
        add("3. Los modelos de factores confirman que el retorno de cada")
        add("   activo puede explicarse por factores económicos identificables.")
        add()
        add("LIMITACIONES:")
        add("  • Los resultados son históricos — el futuro puede diferir.")
        add("  • Las correlaciones pueden aumentar en períodos de crisis.")
        add("  • Con muestras cortas (~60 obs) los estimadores tienen mayor incertidumbre.")
    else:
        add("1. Estacionariedad I(0) confirmada (ADF + KPSS + Phillips-Perron).")
        add("   Normalidad parcialmente rechazada — CVaR es métrica primaria.")
        add()
        add("2. Markowitz: solución viable, Σ definida positiva.")
        add("   Sensibilidad a μ documentada (Michaud, 1989).")
        add()
        add("3. CAPM explica varianza moderada. FF3 mejora R² consistentemente.")
        add("   APT con factores macro complementa el análisis.")
        add()
        add("4. VaR histórico preferible al paramétrico.")
        add("5. Spanning confirma valor de diversificación.")

    add()
    add(sep)
    add("Fin del reporte — QuantαfolyΩ")
    add(sep)

    return '\n'.join(lineas)


# =============================================================================
# FUNCIÓN 2 — Gráfico resumen en Plotly (reemplaza grafico_resumen matplotlib)
# =============================================================================

def grafico_resumen_plotly(retornos: pd.DataFrame,
                            retornos_benchmark: pd.DataFrame,
                            pesos_df: pd.DataFrame,
                            res_capm: pd.DataFrame,
                            res_ff3: pd.DataFrame,
                            res_apt: pd.DataFrame,
                            metricas_3a: pd.DataFrame,
                            metricas_3c: dict,
                            rf_anual: float,
                            factor: int = FACTOR_ANUALIZACION) -> go.Figure:
    """
    4 paneles interactivos:
    1. Retornos acumulados — portafolios vs benchmark
    2. Composición MVP vs Tangente
    3. R² por modelo y activo
    4. Métricas de desempeño comparadas
    """
    tickers   = list(retornos.columns)
    n_activos = len(tickers)
    rf_mes    = rf_anual / factor

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Retorno acumulado (base 1)",
            "Composición MVP vs Tangente (%)",
            "R² por modelo y activo",
            "Métricas de desempeño",
        ),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    bench = retornos_benchmark.iloc[:, 0].reindex(retornos.index).dropna()

    # --- Panel 1: Retornos acumulados ---
    colores_port = {
        'MVP':         COLORES['minima_varianza'],
        'Tangente':    COLORES['portafolio_optimo'],
        'EqualWeight': 'gray',
    }
    labels_port = {'MVP': 'MVP', 'Tangente': 'Tangente', 'EqualWeight': '1/N'}

    for col, label in labels_port.items():
        if not pesos_df.empty and col in pesos_df.columns:
            w    = pesos_df[col].values
            ret  = (retornos @ w).dropna()
            acum = np.exp(ret.cumsum())
            fig.add_trace(go.Scatter(
                x=acum.index, y=acum, mode='lines',
                name=label, line=dict(color=colores_port[col], width=2),
            ), row=1, col=1)

    acum_b = np.exp(bench.cumsum())
    fig.add_trace(go.Scatter(
        x=acum_b.index, y=acum_b, mode='lines', name='S&P 500',
        line=dict(color=COLORES['benchmark'], width=2, dash='dash'),
    ), row=1, col=1)

    # --- Panel 2: Composición barras agrupadas ---
    palette = px.colors.qualitative.Set2
    for i, ticker in enumerate(tickers):
        vals = []
        cols_port = [c for c in ['MVP', 'Tangente'] if c in pesos_df.columns]
        for col in cols_port:
            w = float(pesos_df.loc[ticker, col]) if ticker in pesos_df.index else 0.0
            vals.append(w * 100)
        fig.add_trace(go.Bar(
            x=cols_port, y=vals, name=ticker,
            marker_color=palette[i % len(palette)],
            showlegend=(i < n_activos),
        ), row=1, col=2)

    # --- Panel 3: R² por modelo ---
    def norm_idx(df):
        if df is None or df.empty:
            return pd.DataFrame()
        return df.set_index('Ticker') if 'Ticker' in df.columns else df.copy()

    capm_g = norm_idx(res_capm)
    ff3_g  = norm_idx(res_ff3)
    apt_g  = norm_idx(res_apt)

    modelos_r2 = []
    if not capm_g.empty:
        modelos_r2.append(('CAPM', [float(capm_g.loc[t, 'R2']) if t in capm_g.index else 0 for t in tickers], '#2E86AB'))
    if not ff3_g.empty:
        modelos_r2.append(('FF3',  [float(ff3_g.loc[t, 'R2'])  if t in ff3_g.index  else 0 for t in tickers], '#F18F01'))
    if not apt_g.empty:
        modelos_r2.append(('APT',  [float(apt_g.loc[t, 'R2'])  if t in apt_g.index  else 0 for t in tickers], '#A23B72'))

    for nombre, vals, color in modelos_r2:
        fig.add_trace(go.Bar(
            x=tickers, y=vals, name=nombre,
            marker_color=color, opacity=0.85,
        ), row=2, col=1)

    # --- Panel 4: Métricas de desempeño (barras horizontales) ---
    metricas_mostrar = ['sharpe', 'sortino', 'calmar', 'omega']
    labels_met       = ['Sharpe', 'Sortino', 'Calmar', 'Omega']

    def metricas_port(col):
        if pesos_df.empty or col not in pesos_df.columns:
            return {m: 0.0 for m in metricas_mostrar}
        w    = pesos_df[col].values
        ret  = (retornos @ w).dropna()
        mu_a = float(ret.mean() * factor)
        sig  = float(ret.std() * np.sqrt(factor))
        sr   = (mu_a - rf_anual) / sig if sig > 0 else 0.0
        desvios = (ret - rf_mes).clip(upper=0)
        ds   = float(np.sqrt((desvios**2).mean()) * np.sqrt(factor)) if (desvios < 0).any() else sig
        so   = (mu_a - rf_anual) / ds if ds > 0 else 0.0
        p    = np.exp(ret.cumsum())
        dd   = float(((p - p.cummax()) / p.cummax()).min())
        cal  = min(mu_a / abs(dd) if dd != 0 else 0.0, 5.0)
        gan  = (ret - rf_mes).clip(lower=0).sum()
        per  = (rf_mes - ret).clip(lower=0).sum()
        om   = min(float(gan / per) if per > 0 else 5.0, 5.0)
        return {'sharpe': sr, 'sortino': so, 'calmar': cal, 'omega': om}

    bench_ret = bench
    mu_b      = float(bench_ret.mean() * factor)
    sig_b     = float(bench_ret.std() * np.sqrt(factor))
    sr_b      = (mu_b - rf_anual) / sig_b if sig_b > 0 else 0.0
    desvios_b = (bench_ret - rf_mes).clip(upper=0)
    ds_b      = float(np.sqrt((desvios_b**2).mean()) * np.sqrt(factor)) if (desvios_b < 0).any() else sig_b
    so_b      = (mu_b - rf_anual) / ds_b if ds_b > 0 else 0.0
    p_b       = np.exp(bench_ret.cumsum())
    dd_b      = float(((p_b - p_b.cummax()) / p_b.cummax()).min())
    cal_b     = min(mu_b / abs(dd_b) if dd_b != 0 else 0.0, 5.0)
    gan_b     = (bench_ret - rf_mes).clip(lower=0).sum()
    per_b     = (rf_mes - bench_ret).clip(lower=0).sum()
    om_b      = min(float(gan_b / per_b) if per_b > 0 else 5.0, 5.0)

    tang_metricas = metricas_3c if metricas_3c else {}
    ports_p4 = {
        'MVP':      metricas_port('MVP'),
        'Tangente': {m: min(float(tang_metricas.get(m, 0) or 0), 5.0) for m in metricas_mostrar},
        '1/N':      metricas_port('EqualWeight'),
        'S&P 500':  {'sharpe': sr_b, 'sortino': so_b, 'calmar': cal_b, 'omega': om_b},
    }
    colores_p4 = [COLORES['minima_varianza'], COLORES['portafolio_optimo'],
                  'gray', COLORES['benchmark']]

    for (nombre, vals), color in zip(ports_p4.items(), colores_p4):
        fig.add_trace(go.Bar(
            x=[min(float(vals.get(m, 0) or 0), 5.0) for m in metricas_mostrar],
            y=labels_met, orientation='h',
            name=nombre, marker_color=color, opacity=0.85,
            showlegend=True,
        ), row=2, col=2)

    fig.update_layout(
        height=620,
        barmode='stack',
        title_text='QuantαfolyΩ — Resumen del Análisis',
        legend=dict(orientation='h', y=-0.12),
    )
    fig.update_yaxes(title_text='Valor (base=1)', row=1, col=1)
    fig.update_yaxes(title_text='Peso (%)',       row=1, col=2)
    fig.update_yaxes(title_text='R²',             row=2, col=1)

    return fig


# =============================================================================
# FUNCIÓN 3 — Generación del PDF
# =============================================================================

# =============================================================================
# FUNCIÓN 3 — Generación del PDF ejecutivo
# =============================================================================

def generar_pdf(reporte_texto: str,
                tickers: list,
                fecha_inicio: str,
                fecha_fin: str,
                nivel: str = 'basico',
                fig_retornos=None,
                fig_pesos=None,
                fig_betas_pdf=None,
                metricas_resumen: dict = None,
                semaforo_global: str = "VERDE",
                conclusion: str = "",
                metricas_tabla: pd.DataFrame = None,
                semaforo_f2: pd.DataFrame = None,
                semaforo_f3: pd.DataFrame = None) -> bytes:
    """
    Genera el PDF ejecutivo del reporte y retorna los bytes.

    Páginas:
        1 — Portada con branding QuantαfolyΩ
        2 — Resumen ejecutivo (métricas clave + semáforo + conclusión)
        3 — Gráfico de retornos acumulados (kaleido) + tabla de métricas
        4 — Narrativa completa del asistente
        5 — Anexo técnico (semáforos + referencias)
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     PageBreak, HRFlowable, Table, TableStyle,
                                     Image as RLImage, KeepTogether)
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

    # --- Colores de la paleta QuantαfolyΩ ---
    C_NARANJA  = colors.HexColor('#D76F02')
    C_MALVA    = colors.HexColor('#985D73')
    C_CAFE     = colors.HexColor('#3B200B')
    C_GRIS_AZ  = colors.HexColor('#576071')
    C_CREMA    = colors.HexColor('#FDF6EE')
    C_TERRAC   = colors.HexColor('#F7E1D3')
    C_VERDE    = colors.HexColor('#4A7C59')
    C_AMBAR    = colors.HexColor('#DB9435')
    C_ROJO     = colors.HexColor('#B33000')
    C_BLANCO   = colors.white

    W, H = A4

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        title=f"QuantafolyO — {', '.join(tickers)}",
        author="QuantafolyO",
    )

    # --- Estilos ---
    styles = getSampleStyleSheet()

    def estilo(name, parent='Normal', **kwargs):
        return ParagraphStyle(name, parent=styles[parent], **kwargs)

    est_logo      = estilo('logo',      fontSize=28, fontName='Helvetica-Bold',
                           textColor=C_CAFE, alignment=TA_CENTER, spaceAfter=4)
    est_tagline   = estilo('tagline',   fontSize=10, textColor=C_GRIS_AZ,
                           alignment=TA_CENTER, spaceAfter=8)
    est_cover_inf = estilo('cov_inf',   fontSize=10, textColor=C_GRIS_AZ,
                           alignment=TA_CENTER, spaceAfter=4)
    est_h1        = estilo('h1',        fontSize=13, fontName='Helvetica-Bold',
                           textColor=C_NARANJA, spaceBefore=14, spaceAfter=5)
    est_h2        = estilo('h2',        fontSize=10, fontName='Helvetica-Bold',
                           textColor=C_CAFE, spaceBefore=8, spaceAfter=3)
    est_normal    = estilo('normal',    fontSize=9,  textColor=C_CAFE,
                           leading=14, spaceAfter=3, alignment=TA_JUSTIFY)
    est_mono      = estilo('mono',      fontSize=8,  textColor=C_GRIS_AZ,
                           leading=12, spaceAfter=2, fontName='Courier')
    est_centrado  = estilo('centrado',  fontSize=9,  textColor=C_GRIS_AZ,
                           alignment=TA_CENTER, spaceAfter=4)
    est_caption   = estilo('caption',   fontSize=8,  textColor=C_GRIS_AZ,
                           alignment=TA_CENTER, spaceAfter=6, spaceBefore=2)
    est_concl     = estilo('concl',     fontSize=9,  textColor=C_CAFE,
                           leading=15, spaceAfter=4, alignment=TA_JUSTIFY,
                           leftIndent=10, rightIndent=10)

    def limpiar(t):
        s = str(t)
        # Primero proteger el nombre del proyecto
        s = s.replace('QuantαfolyΩ', 'QuantafolyO_PROTECTED')
        # Limpiar emojis y símbolos no soportados
        s = (s.replace('✅','[OK]').replace('⚠️','[!]').replace('❌','[X]')
              .replace('🟢','[V]').replace('🟡','[A]').replace('🔴','[R]')
              .replace('📈','').replace('📉','').replace('📊','').replace('📐','')
              .replace('🛡️','')
              .replace('β','beta').replace('α','alpha').replace('Σ','Sigma')
              .replace('μ','mu').replace('σ','sigma').replace('Ω','Omega')
              .replace('→','->').replace('≥','>=').replace('≤','<=')
              .replace('—','-').replace('\u2019',"'").replace('\u2018',"'")
              .replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))
        # Restaurar nombre del proyecto
        s = s.replace('QuantafolyO_PROTECTED', 'Quant\u03b1foly\u03a9')
        return s

    def hr(color=C_NARANJA, width="100%", thickness=0.8):
        return HRFlowable(width=width, thickness=thickness,
                          color=color, hAlign='LEFT', spaceAfter=6)

    story = []

    # =========================================================================
    # PÁGINA 1 — PORTADA
    # =========================================================================
    story.append(Spacer(1, 3.5*cm))

    # Barra de color superior
    story.append(HRFlowable(width="100%", thickness=4,
                             color=C_NARANJA, hAlign='CENTER'))
    story.append(Spacer(1, 0.6*cm))

    # Logo — generado como imagen con kaleido para soportar Unicode + colores
    try:
        import plotly.graph_objects as go_logo
        fig_logo = go_logo.Figure()
        fig_logo.add_annotation(
            text="Quant<span style='color:#D76F02'>\u03b1</span>foly<span style='color:#985D73'>\u03a9</span>",
            x=0.5, y=0.6, xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=52, color='#3B200B', family='Arial Black'),
            xanchor='center', yanchor='middle',
        )
        fig_logo.add_annotation(
            text="An\u00e1lisis cuantitativo de portafolios de inversi\u00f3n",
            x=0.5, y=0.15, xref='paper', yref='paper',
            showarrow=False,
            font=dict(size=14, color='#576071', family='Arial'),
            xanchor='center', yanchor='middle',
        )
        fig_logo.update_layout(
            paper_bgcolor='white', plot_bgcolor='white',
            margin=dict(l=0, r=0, t=0, b=0),
            height=120, width=500,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        _logo_bytes = fig_logo.to_image(format='png', width=500, height=120, scale=2)
        story.append(RLImage(io.BytesIO(_logo_bytes),
                             width=doc.width*0.65, height=doc.width*0.65*120/500,
                             hAlign='CENTER'))
    except Exception:
        # Fallback texto simple si kaleido falla
        story.append(Paragraph("QuantafolyO", estilo('logo_fb',
                     fontSize=30, fontName='Helvetica-Bold',
                     textColor=C_NARANJA, alignment=TA_CENTER, spaceAfter=4)))
        story.append(Paragraph("An\u00e1lisis cuantitativo de portafolios de inversi\u00f3n",
                                est_tagline))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width="60%", thickness=1,
                             color=C_MALVA, hAlign='CENTER'))
    story.append(Spacer(1, 1.2*cm))

    # Datos del portafolio
    story.append(Paragraph(f"<b>Portafolio analizado</b>", est_centrado))
    story.append(Paragraph(f"{', '.join(tickers)}", estilo('tickers_port',
                 fontSize=12, fontName='Helvetica-Bold', textColor=C_NARANJA,
                 alignment=TA_CENTER, spaceAfter=6)))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Per\u00edodo: {fecha_inicio} \u2014 {fecha_fin}", est_cover_inf))
    story.append(Paragraph(f"Fecha de generaci\u00f3n: {datetime.today().strftime('%d de %B de %Y')}", est_cover_inf))
    story.append(Paragraph(f"Nivel del an\u00e1lisis: {'B\u00e1sico' if nivel == 'basico' else 'T\u00e9cnico'}", est_cover_inf))
    story.append(Spacer(1, 2*cm))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=C_MALVA, hAlign='CENTER'))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Datos: Yahoo Finance  \u00b7  Benchmark: S&amp;P 500  \u00b7  Divisa base: USD", est_centrado))
    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 2 — RESUMEN EJECUTIVO
    # =========================================================================
    story.append(Paragraph("Resumen Ejecutivo", est_h1))
    story.append(hr())

    # Semáforo global
    color_sem = {'VERDE': C_VERDE, 'AMARILLO': C_AMBAR, 'ROJO': C_ROJO}.get(semaforo_global, C_AMBAR)
    label_sem = {
        'VERDE':   '[OK] Portafolio bien construido',
        'AMARILLO':'[!] Portafolio con advertencias menores',
        'ROJO':    '[X] Portafolio requiere revision',
    }.get(semaforo_global, '[!] Portafolio con advertencias')

    sem_data = [[Paragraph(f"<b>{limpiar(label_sem)}</b>",
                 estilo('sem_txt', fontSize=10, fontName='Helvetica-Bold',
                        textColor=C_BLANCO, alignment=TA_CENTER))]]
    sem_table = Table(sem_data, colWidths=[doc.width])
    sem_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), color_sem),
        ('ROUNDEDCORNERS', [4]),
        ('TOPPADDING',    (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(sem_table)
    story.append(Spacer(1, 0.4*cm))

    # Conclusión narrativa
    if conclusion:
        story.append(Paragraph(limpiar(conclusion), est_concl))
        story.append(Spacer(1, 0.3*cm))

    # Tabla de métricas clave
    if metricas_resumen:
        story.append(Paragraph("M\u00e9tricas clave del portafolio tangente", est_h2))

        mr = metricas_resumen
        met_data = [
            [Paragraph('<b>M\u00e9trica</b>',   estilo('mh', fontSize=9, fontName='Helvetica-Bold', textColor=C_BLANCO)),
             Paragraph('<b>Valor</b>',            estilo('mh', fontSize=9, fontName='Helvetica-Bold', textColor=C_BLANCO)),
             Paragraph('<b>Referencia</b>',       estilo('mh', fontSize=9, fontName='Helvetica-Bold', textColor=C_BLANCO)),
             Paragraph('<b>Diagn\u00f3stico</b>', estilo('mh', fontSize=9, fontName='Helvetica-Bold', textColor=C_BLANCO))],
            ["Retorno esperado anual",
             f"{mr.get('retorno', 0)*100:.1f}%",
             f"S&P 500: {mr.get('retorno_bench', 0)*100:.1f}%",
             "[OK]" if mr.get('retorno', 0) > mr.get('retorno_bench', 0) else "[!]"],
            ["Volatilidad anual",
             f"{mr.get('volatilidad', 0)*100:.1f}%",
             f"S&P 500: {mr.get('vol_bench', 0)*100:.1f}%",
             "[OK]" if mr.get('volatilidad', 0) <= mr.get('vol_bench', 0)*1.2 else "[!]"],
            ["Sharpe ratio",
             f"{mr.get('sharpe', 0):.3f}",
             f"Benchmark: {mr.get('sharpe_bench', 0):.3f}",
             "[OK]" if mr.get('sharpe', 0) > mr.get('sharpe_bench', 0) else "[!]"],
            ["VaR hist\u00f3rico 95% (mensual)",
             f"{mr.get('var_95', 0)*100:.2f}%",
             "Peor mes esperado",
             ""],
            ["CVaR hist\u00f3rico 95% (mensual)",
             f"{mr.get('cvar_95', 0)*100:.2f}%",
             "Promedio peor 5%",
             "[!]" if abs(mr.get('cvar_95', 0)) > 0.10 else "[OK]"],
            ["Beta del portafolio",
             f"{mr.get('beta', 1):.2f}",
             "1.0 = neutral mercado",
             "Agresivo" if mr.get('beta', 1) > 1.1 else ("Defensivo" if mr.get('beta', 1) < 0.9 else "Neutral")],
            ["VaR calibrado (Kupiec)",
             "Si" if mr.get('kupiec_ok', True) else "No",
             "Prueba de backtesting",
             "[OK]" if mr.get('kupiec_ok', True) else "[X]"],
        ]

        col_w = [doc.width*0.38, doc.width*0.15, doc.width*0.27, doc.width*0.20]
        met_table = Table(met_data, colWidths=col_w, repeatRows=1)
        ts = TableStyle([
            ('BACKGROUND',   (0,0), (-1,0), C_NARANJA),
            ('TEXTCOLOR',    (0,0), (-1,0), C_BLANCO),
            ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',     (0,0), (-1,-1), 9),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_CREMA, C_BLANCO]),
            ('GRID',         (0,0), (-1,-1), 0.3, colors.HexColor('#AC664640')),
            ('TOPPADDING',   (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
            ('LEFTPADDING',  (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('ALIGN',        (1,0), (-1,-1), 'CENTER'),
        ])
        # Color diagnóstico
        for i, row in enumerate(met_data[1:], 1):
            diag = str(row[3]) if len(row) > 3 else ''
            if '[OK]' in diag:
                ts.add('TEXTCOLOR', (3,i), (3,i), C_VERDE)
                ts.add('FONTNAME',  (3,i), (3,i), 'Helvetica-Bold')
            elif '[X]' in diag or '[!]' in diag:
                c = C_ROJO if '[X]' in diag else C_AMBAR
                ts.add('TEXTCOLOR', (3,i), (3,i), c)
                ts.add('FONTNAME',  (3,i), (3,i), 'Helvetica-Bold')
        met_table.setStyle(ts)
        story.append(met_table)

    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 3 — DESEMPEÑO DEL PORTAFOLIO
    # =========================================================================
    story.append(Paragraph("Desempe\u00f1o del Portafolio", est_h1))
    story.append(hr())

    # Tabla PRIMERO — más importante, arriba de la página
    if metricas_tabla is not None and not metricas_tabla.empty:
        story.append(Paragraph("M\u00e9tricas comparativas por portafolio", est_h2))
        cols_show = ['nombre','retorno_anual','volatilidad','sharpe','sortino','max_drawdown','calmar']
        cols_disp = [c for c in cols_show if c in metricas_tabla.columns]
        headers_map = {
            'nombre':'Portafolio','retorno_anual':'Retorno','volatilidad':'Volatilidad',
            'sharpe':'Sharpe','sortino':'Sortino','max_drawdown':'Max DD','calmar':'Calmar'
        }
        tbl_data = [[Paragraph(f'<b>{headers_map.get(c,c)}</b>',
                     estilo(f'th_{c}', fontSize=8, fontName='Helvetica-Bold',
                            textColor=C_BLANCO)) for c in cols_disp]]
        for _, row in metricas_tabla[cols_disp].iterrows():
            fila = []
            for c in cols_disp:
                v = row[c]
                if c == 'nombre':
                    fila.append(str(v))
                else:
                    try:
                        vf = float(v)
                        if c in ('retorno_anual','volatilidad','max_drawdown'):
                            fila.append(f"{vf*100:.1f}%")
                        else:
                            fila.append(f"{vf:.3f}")
                    except (ValueError, TypeError):
                        fila.append('-')
            tbl_data.append(fila)
        n_cols_t = len(cols_disp)
        col_w_t  = [doc.width*0.28] + [doc.width*0.72/(n_cols_t-1)]*(n_cols_t-1)
        tbl = Table(tbl_data, colWidths=col_w_t, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), C_NARANJA),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [C_CREMA, C_BLANCO]),
            ('FONTSIZE',      (0,0), (-1,-1), 8),
            ('GRID',          (0,0), (-1,-1), 0.3, colors.HexColor('#AC664640')),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING',   (0,0), (-1,-1), 5),
            ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

    # Gráfico retornos acumulados
    if fig_retornos is not None:
        try:
            img_bytes = fig_retornos.to_image(format="png", width=700, height=240, scale=2)
            story.append(RLImage(io.BytesIO(img_bytes),
                                 width=doc.width, height=doc.width*240/700))
            story.append(Paragraph("Retornos acumulados \u2014 activos vs S&amp;P 500", est_caption))
            story.append(Spacer(1, 0.3*cm))
        except Exception as e:
            story.append(Paragraph(f"[Gr\u00e1fico retornos no disponible]", est_caption))

    # Gráfico pesos del portafolio
    if fig_pesos is not None:
        try:
            img_bytes = fig_pesos.to_image(format="png", width=700, height=200, scale=2)
            story.append(RLImage(io.BytesIO(img_bytes),
                                 width=doc.width, height=doc.width*200/700))
            story.append(Paragraph("Composici\u00f3n de los portafolios \u2014 pesos por activo (%)", est_caption))
            story.append(Spacer(1, 0.3*cm))
        except Exception as e:
            story.append(Paragraph(f"[Gr\u00e1fico pesos no disponible]", est_caption))

    # Gráfico betas
    if fig_betas_pdf is not None:
        try:
            img_bytes = fig_betas_pdf.to_image(format="png", width=700, height=200, scale=2)
            story.append(RLImage(io.BytesIO(img_bytes),
                                 width=doc.width, height=doc.width*200/700))
            story.append(Paragraph("Beta por activo \u2014 sensibilidad al mercado (CAPM)", est_caption))
        except Exception as e:
            story.append(Paragraph(f"[Gr\u00e1fico betas no disponible]", est_caption))

    story.append(PageBreak())

    # =========================================================================
    # PÁGINA 4+ — NARRATIVA COMPLETA (flujo continuo, sin PageBreak por sección)
    # =========================================================================
    encabezados  = ['SECCIÓN 1','SECCIÓN 2','SECCIÓN 3','SECCIÓN 4',
                    'SECCIÓN 5','SECCIÓN 6','SECCIÓN 7']
    subsecciones = ['CAPM','FAMA','APT','VALUE AT RISK','MVP',
                    'PORTAFOLIO','MÉTRICAS','DIVERSIFICACIÓN',
                    'ESTABILIDAD','BACKTESTING','LIMITACIONES']

    primera_seccion = True
    for linea in reporte_texto.split('\n'):
        linea = linea.rstrip()
        if not linea or linea.startswith('=') or linea.startswith('-'*10):
            continue
        if any(linea.startswith(h) for h in encabezados):
            if primera_seccion:
                primera_seccion = False
            else:
                # Separador visual entre secciones sin PageBreak
                story.append(Spacer(1, 0.5*cm))
                story.append(HRFlowable(width="100%", thickness=0.4,
                                         color=C_TERRAC, hAlign='LEFT'))
                story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(limpiar(linea), est_h1))
            story.append(hr())
            continue
        if any(linea.strip().startswith(s) for s in subsecciones):
            try:
                story.append(Spacer(1, 0.2*cm))
                story.append(Paragraph(f"<b>{limpiar(linea.strip())}</b>", est_h2))
            except Exception:
                pass
            continue
        linea_l = limpiar(linea)
        if linea_l.strip():
            if (linea_l.startswith('  ') and
                    any(c in linea_l for c in ['=','|','%','beta','alpha'])):
                try:
                    story.append(Paragraph(linea_l, est_mono))
                except Exception:
                    pass
            else:
                try:
                    story.append(Paragraph(linea_l.strip(), est_normal))
                except Exception:
                    pass

    # =========================================================================
    # PÁGINA 5 — ANEXO TÉCNICO
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("Anexo T\u00e9cnico", est_h1))
    story.append(hr())

    # Semáforo Fase 2
    if semaforo_f2 is not None and not semaforo_f2.empty:
        story.append(Paragraph("Diagn\u00f3stico estad\u00edstico — Estacionariedad (Fase 2)", est_h2))
        # Usar las columnas disponibles — res_estac tiene Ticker + columnas de resultados
        sem_cols = [c for c in semaforo_f2.columns if c in
                    ['Ticker','ADF_p','ADF_resultado','KPSS_p','KPSS_resultado','Semaforo','Conclusion']]
        if not sem_cols:
            sem_cols = list(semaforo_f2.columns)[:5]  # fallback: primeras 5 columnas
        if sem_cols:
            sem2_data = [[Paragraph(f'<b>{c}</b>',
                          estilo(f'sh2_{c}', fontSize=7, fontName='Helvetica-Bold',
                                 textColor=C_BLANCO)) for c in sem_cols]]
            for _, row in semaforo_f2[sem_cols].head(15).iterrows():
                sem2_data.append([limpiar(str(row[c]))[:40] for c in sem_cols])
            col_w_s = [doc.width/len(sem_cols)]*len(sem_cols)
            s2t = Table(sem2_data, colWidths=col_w_s, repeatRows=1)
            s2t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,0), C_GRIS_AZ),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_CREMA, C_BLANCO]),
                ('FONTSIZE',      (0,0),(-1,-1), 7),
                ('GRID',          (0,0),(-1,-1), 0.3, colors.HexColor('#57607150')),
                ('TOPPADDING',    (0,0),(-1,-1), 3),
                ('BOTTOMPADDING', (0,0),(-1,-1), 3),
                ('LEFTPADDING',   (0,0),(-1,-1), 4),
            ]))
            story.append(s2t)
            story.append(Spacer(1, 0.3*cm))

    # Semáforo Fase 3
    if semaforo_f3 is not None and not semaforo_f3.empty:
        story.append(Paragraph("Diagn\u00f3stico verificaci\u00f3n \u2014 Spanning (Fase 3)", est_h2))
        sem3_cols = [c for c in semaforo_f3.columns
                     if c in ['Ticker','rechaza','p_valor','alpha','beta','semaforo','Conclusion']]
        if not sem3_cols:
            sem3_cols = list(semaforo_f3.columns)[:5]
        if sem3_cols:
            sem3_data = [[Paragraph(f'<b>{c}</b>',
                          estilo(f'sh3_{c}', fontSize=7, fontName='Helvetica-Bold',
                                 textColor=C_BLANCO)) for c in sem3_cols]]
            for _, row in semaforo_f3[sem3_cols].head(15).iterrows():
                sem3_data.append([limpiar(str(row[c]))[:40] for c in sem3_cols])
            col_w_s3 = [doc.width/len(sem3_cols)]*len(sem3_cols)
            s3t = Table(sem3_data, colWidths=col_w_s3, repeatRows=1)
            s3t.setStyle(TableStyle([
                ('BACKGROUND',    (0,0),(-1,0), C_GRIS_AZ),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[C_CREMA, C_BLANCO]),
                ('FONTSIZE',      (0,0),(-1,-1), 7),
                ('GRID',          (0,0),(-1,-1), 0.3, colors.HexColor('#57607150')),
                ('TOPPADDING',    (0,0),(-1,-1), 3),
                ('BOTTOMPADDING', (0,0),(-1,-1), 3),
                ('LEFTPADDING',   (0,0),(-1,-1), 4),
            ]))
            story.append(s3t)
            story.append(Spacer(1, 0.3*cm))

    # Referencias
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Referencias bibliogr\u00e1ficas", est_h2))
    story.append(hr(color=C_GRIS_AZ, thickness=0.5))
    refs = [
        "Markowitz, H. (1952). Portfolio Selection. <i>Journal of Finance</i>, 7(1), 77-91.",
        "Sharpe, W. F. (1964). Capital Asset Prices. <i>Journal of Finance</i>, 19(3), 425-442.",
        "Lintner, J. (1965). The Valuation of Risk Assets. <i>Review of Economics and Statistics</i>, 47(1), 13-37.",
        "Ross, S. A. (1976). The Arbitrage Theory of Capital Asset Pricing. <i>Journal of Economic Theory</i>, 13(3), 341-360.",
        "Fama, E. F. &amp; French, K. R. (1993). Common Risk Factors in the Returns on Stocks and Bonds. <i>Journal of Financial Economics</i>, 33(1), 3-56.",
        "Chen, N., Roll, R. &amp; Ross, S. A. (1986). Economic Forces and the Stock Market. <i>Journal of Business</i>, 59(3), 383-403.",
        "Artzner, P. et al. (1999). Coherent Measures of Risk. <i>Mathematical Finance</i>, 9(3), 203-228.",
        "Kupiec, P. H. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. <i>Journal of Derivatives</i>, 3(2), 73-84.",
        "Christoffersen, P. F. (1998). Evaluating Interval Forecasts. <i>International Economic Review</i>, 39(4), 841-862.",
        "Huberman, G. &amp; Kandel, S. (1987). Mean-Variance Spanning. <i>Journal of Finance</i>, 42(4), 873-888.",
        "Chow, G. C. (1960). Tests of Equality Between Sets of Coefficients. <i>Econometrica</i>, 28(3), 591-605.",
        "Michaud, R. O. (1989). The Markowitz Optimization Enigma. <i>Financial Analysts Journal</i>, 45(1), 31-42.",
    ]
    for ref in refs:
        story.append(Paragraph(ref, estilo('ref', fontSize=8, textColor=C_CAFE,
                                            leading=12, spaceAfter=4,
                                            leftIndent=12, firstLineIndent=-12)))

    story.append(Spacer(1, 0.6*cm))
    story.append(hr(color=C_MALVA, thickness=1))
    story.append(Paragraph("Quant\u03b1foly\u03a9  \u00b7  Datos: Yahoo Finance  \u00b7  Benchmark: S&amp;P 500",
                            est_centrado))

    doc.build(story)
    return buffer.getvalue()


# =============================================================================
# SEMÁFORO GLOBAL — única fuente de verdad
# CORRECCIÓN (auditoría P1.2→P3.1): antes se calculaba por separado en el PDF
# (pipeline_reporte) y en la página 5_Reporte.py, con criterios distintos —
# podían mostrar un veredicto diferente para el mismo análisis. Ahora ambos
# llaman a esta misma función.
# =============================================================================

def calcular_semaforo_global(sharpe_port: float, sharpe_bench: float,
                              kupiec_ok: bool, n_chow_inestables: int,
                              n_no_normales: int = 0) -> dict:
    """
    Criterios consolidados para el semáforo ejecutivo global (4 criterios reales).
    Retorna dict con 'semaforo', 'alertas' (lista de claves) y 'positivos'.
    """
    alertas   = []
    positivos = []

    if sharpe_port > sharpe_bench:
        positivos.append("Sharpe del portafolio supera al benchmark")
    else:
        alertas.append("sharpe")

    if kupiec_ok:
        positivos.append("VaR bien calibrado (Kupiec)")
    else:
        alertas.append("kupiec")

    if n_chow_inestables > 0:
        alertas.append("chow")
    else:
        positivos.append("Betas estables en el período (Chow)")

    if n_no_normales > 0:
        alertas.append("normalidad")
    else:
        positivos.append("Normalidad razonable en los activos")

    n = len(alertas)
    semaforo = "VERDE" if n == 0 else ("AMARILLO" if n <= 2 else "ROJO")

    return {"semaforo": semaforo, "alertas": alertas, "positivos": positivos}


# =============================================================================
# PIPELINE — Fase 4 completa
# =============================================================================

def pipeline_reporte(retornos: pd.DataFrame,
                     retornos_benchmark: pd.DataFrame,
                     pesos_df: pd.DataFrame,
                     res_3a: dict,
                     metricas_3a: pd.DataFrame,
                     res_capm: pd.DataFrame,
                     res_ff3: pd.DataFrame,
                     res_apt: pd.DataFrame,
                     res_var_dict: dict,
                     metricas_3c: dict,
                     res_bt: pd.DataFrame,
                     res_estres: pd.DataFrame,
                     res_spanning: pd.DataFrame,
                     res_chow: pd.DataFrame,
                     res_estac: pd.DataFrame,
                     res_norm: pd.DataFrame,
                     res_arch: pd.DataFrame,
                     res_mardia: dict,
                     nivel_asistente: str = 'basico',
                     factor: int = FACTOR_ANUALIZACION) -> dict:
    """
    Genera narrativa completa, gráfico resumen y PDF.

    Retorna dict con: narrativa, fig_resumen, pdf_bytes, logs, error
    """
    logs = []

    def _log(m):
        logs.append(m) if isinstance(m, str) else logs.extend(m)

    try:
        _log("Generando narrativa completa...")
        narrativa = asistente_completo(
            retornos=retornos,
            res_3a=res_3a,
            pesos_df=pesos_df,
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
            nivel=nivel_asistente,
        )
        _log("  OK — narrativa generada")

        _log("Generando gráfico resumen Plotly...")
        fig_resumen = grafico_resumen_plotly(
            retornos=retornos,
            retornos_benchmark=retornos_benchmark,
            pesos_df=pesos_df,
            res_capm=res_capm,
            res_ff3=res_ff3,
            res_apt=res_apt,
            metricas_3a=metricas_3a,
            metricas_3c=metricas_3c,
            rf_anual=res_3a.get('rf_anual', 0.04),
            factor=factor,
        )
        _log("  OK — gráfico generado")

        _log("Generando PDF...")
        tickers      = list(retornos.columns)
        fecha_inicio = retornos.index[0].strftime('%B %Y')
        fecha_fin    = retornos.index[-1].strftime('%B %Y')

        # --- Construir métricas resumen para página 2 ---
        _tang     = res_3a.get('tangente', {})
        _bench    = res_3a.get('benchmark', {})
        _var_dict = res_var_dict or {}
        _capm_df  = res_capm if res_capm is not None and not res_capm.empty else pd.DataFrame()
        _pesos    = pesos_df if pesos_df is not None and not pesos_df.empty else pd.DataFrame()

        _beta_port = 1.0
        if not _capm_df.empty and not _pesos.empty and 'Beta' in _capm_df.columns:
            try:
                _betas = _capm_df.set_index('Ticker')['Beta']
                _beta_port = float((_pesos['Tangente'] * _betas).sum())
            except Exception:
                pass

        _kupiec_ok = True
        if res_bt is not None and not res_bt.empty and 'kupiec_ok' in res_bt.columns:
            _kupiec_ok = bool(res_bt['kupiec_ok'].all())

        metricas_resumen = {
            'retorno':       float(_tang.get('retorno', 0) or 0),
            'volatilidad':   float(_tang.get('volatilidad', 0) or 0),
            'sharpe':        float(_tang.get('sharpe', 0) or 0),
            'retorno_bench': float(_bench.get('retorno', 0) or 0),
            'vol_bench':     float(_bench.get('volatilidad', 0) or 0),
            'sharpe_bench':  float(_bench.get('sharpe', 0) or 0),
            'var_95':        float(_var_dict.get('95%', {}).get('var_hist', 0) or 0),
            'cvar_95':       float(_var_dict.get('95%', {}).get('cvar_hist', 0) or 0),
            'beta':          _beta_port,
            'kupiec_ok':     _kupiec_ok,
        }

        # --- Semáforo global (única fuente de verdad, ver calcular_semaforo_global) ---
        _n_inestab = 0
        if res_chow is not None and not res_chow.empty and 'semaforo' in res_chow.columns:
            _n_inestab = int((res_chow['semaforo'] == 'ROJO').sum())

        _n_no_norm = 0
        if res_norm is not None and not res_norm.empty and 'Semaforo' in res_norm.columns:
            _n_no_norm = int((res_norm['Semaforo'] == 'ROJO').sum())

        _sem = calcular_semaforo_global(
            sharpe_port=metricas_resumen['sharpe'],
            sharpe_bench=metricas_resumen['sharpe_bench'],
            kupiec_ok=_kupiec_ok,
            n_chow_inestables=_n_inestab,
            n_no_normales=_n_no_norm,
        )
        semaforo_global = _sem['semaforo']
        _alertas        = _sem['alertas']

        # --- Conclusión narrativa corta ---
        if nivel_asistente == 'basico':
            if semaforo_global == 'VERDE':
                conclusion = (
                    f"El portafolio esta bien construido. Genera un retorno esperado del "
                    f"{metricas_resumen['retorno']*100:.1f}% anual con una volatilidad del "
                    f"{metricas_resumen['volatilidad']*100:.1f}%, superando al S&P 500 en "
                    f"eficiencia (Sharpe {metricas_resumen['sharpe']:.2f} vs "
                    f"{metricas_resumen['sharpe_bench']:.2f}). El modelo de riesgo esta bien calibrado."
                )
            elif semaforo_global == 'AMARILLO':
                conclusion = (
                    f"El portafolio es valido con matices. Retorno esperado del "
                    f"{metricas_resumen['retorno']*100:.1f}% anual con volatilidad del "
                    f"{metricas_resumen['volatilidad']*100:.1f}%. "
                    f"Hay {len(_alertas)} punto(s) de atencion que conviene considerar al tomar decisiones."
                )
            else:
                conclusion = (
                    f"El portafolio requiere revision. El analisis detecto {len(_alertas)} advertencias "
                    f"relevantes. Considera ajustar la seleccion de activos o ampliar la ventana temporal."
                )
        else:
            conclusion = (
                f"Sharpe tangente {metricas_resumen['sharpe']:.3f} vs benchmark "
                f"{metricas_resumen['sharpe_bench']:.3f}. "
                f"Beta ponderado {_beta_port:.2f}. "
                f"Kupiec: {'validado' if _kupiec_ok else 'no validado'}. "
                f"Alertas activas: {len(_alertas)}."
            )

        # --- Gráfico de retornos para kaleido ---
        try:
            import plotly.graph_objects as go_pdf
            retornos_acum = np.exp(retornos.cumsum()) - 1
            bench_acum    = np.exp(retornos_benchmark.cumsum()) - 1
            seq = ['#D76F02','#DB9435','#985D73','#9FD0D6','#ECD577','#4A7C59','#AC6646','#576071']
            _layout_pdf = dict(
                paper_bgcolor='white', plot_bgcolor='white',
                font=dict(family='Helvetica', color='#3B200B', size=10),
                legend=dict(orientation='h', y=-0.28, x=0, font=dict(size=9)),
                margin=dict(l=40, r=20, t=20, b=65),
                xaxis=dict(gridcolor='#EEE8E0', showgrid=True),
                yaxis=dict(gridcolor='#EEE8E0', showgrid=True),
            )

            # Fig 1 — Retornos acumulados
            fig_ret = go_pdf.Figure()
            for i, col in enumerate(retornos_acum.columns):
                fig_ret.add_trace(go_pdf.Scatter(
                    x=retornos_acum.index, y=retornos_acum[col],
                    name=col, mode='lines',
                    line=dict(color=seq[i % len(seq)], width=2),
                ))
            fig_ret.add_trace(go_pdf.Scatter(
                x=bench_acum.index, y=bench_acum['SP500'],
                name='S&P 500', mode='lines',
                line=dict(color='#9FD0D6', dash='dash', width=1.5),
            ))
            fig_ret.update_layout(yaxis_tickformat='.0%', **_layout_pdf)
        except Exception:
            fig_ret = None

        # --- Gráfico de pesos para kaleido ---
        fig_pesos_pdf = None
        try:
            if pesos_df is not None and not pesos_df.empty:
                fig_pesos_pdf = go_pdf.Figure()
                for i, ticker in enumerate(pesos_df.index):
                    fig_pesos_pdf.add_trace(go_pdf.Bar(
                        name=ticker,
                        x=pesos_df.columns.tolist(),
                        y=(pesos_df.loc[ticker] * 100).tolist(),
                        marker_color=seq[i % len(seq)],
                    ))
                fig_pesos_pdf.update_layout(
                    barmode='stack',
                    yaxis_title='Peso (%)',
                    **_layout_pdf,
                )
        except Exception:
            fig_pesos_pdf = None

        # --- Gráfico de betas para kaleido ---
        fig_betas_pdf = None
        try:
            if res_capm is not None and not res_capm.empty and 'Beta' in res_capm.columns:
                _tickers_b = res_capm['Ticker'].tolist()
                _betas_b   = res_capm['Beta'].tolist()
                # Colores bien diferenciados: agresivo=naranja, defensivo=verde, neutral=teal
                _colores_b = []
                for b in _betas_b:
                    if b > 1.1:
                        _colores_b.append('#D76F02')   # naranja — agresivo
                    elif b < 0.9:
                        _colores_b.append('#4A7C59')   # verde — defensivo
                    else:
                        _colores_b.append('#9FD0D6')   # teal — neutral
                fig_betas_pdf = go_pdf.Figure()
                fig_betas_pdf.add_trace(go_pdf.Bar(
                    x=_tickers_b, y=_betas_b,
                    marker_color=_colores_b,
                    text=[f"{b:.2f}" for b in _betas_b],
                    textposition='outside',
                    textfont=dict(color='#3B200B', size=10),
                    showlegend=False,
                ))
                # Línea de referencia beta=1
                fig_betas_pdf.add_shape(type='line',
                    x0=-0.5, x1=len(_tickers_b)-0.5, y0=1, y1=1,
                    line=dict(color='#576071', dash='dash', width=1.5))
                # Anotación de referencia
                fig_betas_pdf.add_annotation(
                    x=len(_tickers_b)-0.5, y=1,
                    text="  mercado", showarrow=False,
                    font=dict(color='#576071', size=9), xanchor='left')
                # Leyenda manual de colores
                for color, label in [('#D76F02','Agresivo (β>1.1)'),
                                      ('#9FD0D6','Neutral (0.9-1.1)'),
                                      ('#4A7C59','Defensivo (β<0.9)')]:
                    fig_betas_pdf.add_trace(go_pdf.Bar(
                        x=[None], y=[None],
                        marker_color=color, name=label, showlegend=True,
                    ))
                fig_betas_pdf.update_layout(
                    yaxis_title='Beta',
                    yaxis=dict(range=[0, max(_betas_b)*1.25]),
                    **_layout_pdf,
                )
        except Exception:
            fig_betas_pdf = None

        pdf_bytes = generar_pdf(
            reporte_texto    = narrativa,
            tickers          = tickers,
            fecha_inicio     = fecha_inicio,
            fecha_fin        = fecha_fin,
            nivel            = nivel_asistente,
            fig_retornos     = fig_ret,
            fig_pesos        = fig_pesos_pdf,
            fig_betas_pdf    = fig_betas_pdf,
            metricas_resumen = metricas_resumen,
            semaforo_global  = semaforo_global,
            conclusion       = conclusion,
            metricas_tabla   = metricas_3a,
            semaforo_f2      = res_estac,
            semaforo_f3      = res_spanning,
        )
        _log(f"  OK — PDF generado ({len(pdf_bytes)//1024} KB)")

        _log("✅ FASE 4 COMPLETADA")

        return {
            'narrativa':   narrativa,
            'fig_resumen': fig_resumen,
            'pdf_bytes':   pdf_bytes,
            'logs':        logs,
            'error':       None,
        }

    except Exception as e:
        import traceback
        _log(f"ERROR FATAL en pipeline_reporte: {e}")
        _log(traceback.format_exc())
        return {'error': str(e), 'logs': logs}
