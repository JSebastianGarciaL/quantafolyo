# =============================================================================
# QuantαfolyΩ — modulos/markowitz.py
# Fase 3a: Optimización de Portafolio (Markowitz)
#
# Funciones migradas desde PortfolioLab_V0_7_0.ipynb (celdas 35–43).
# REGLA DE MIGRACIÓN: lógica matemática sin cambios.
# Cambios:
#   - print() → entradas en lista `logs`
#   - graficar_frontera() reescrita en Plotly (retorna fig en lugar de plt.show())
#   - asistente_fase3a() retorna str markdown en lugar de imprimir
#   - RUTA_PROYECTO / Drive eliminados
#   - Variables globales → importadas explícitamente desde config.py
#   - pipeline_markowitz() envuelve todo el flujo
# =============================================================================

import pandas as pd
import numpy as np
import cvxpy as cp
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    FACTOR_ANUALIZACION,
    PERMITIR_VENTAS_CORTO,
    PESO_MINIMO_ACTIVO,
    PESO_MAXIMO_ACTIVO,
    PUNTOS_FRONTERA_EFICIENTE,
    TICKER_TASA_LIBRE_RIESGO,
    COLORES,
)


# =============================================================================
# FUNCIÓN AUXILIAR — Descargar tasa libre de riesgo
# =============================================================================

def _obtener_rf() -> tuple[float, list]:
    """Descarga la tasa libre de riesgo actual (T-Bill 3M) desde Yahoo Finance."""
    logs = []
    try:
        datos = yf.download(TICKER_TASA_LIBRE_RIESGO,
                            period='1mo', progress=False, auto_adjust=True)
        if isinstance(datos.columns, pd.MultiIndex):
            datos = datos['Close']
        rf_anual = float(datos.iloc[-1].values[0]) / 100
        logs.append(f"Tasa libre de riesgo (T-Bill 3M): {rf_anual*100:.2f}% anual")
    except Exception:
        rf_anual = 0.04
        logs.append("AVISO: no se pudo descargar T-Bill. Usando rf=4% por defecto.")
    return rf_anual, logs


# =============================================================================
# FUNCIÓN 1 — Estimación de parámetros: μ y Σ
# =============================================================================

def estimar_parametros(retornos: pd.DataFrame,
                        factor: int = FACTOR_ANUALIZACION) -> tuple[dict, list]:
    """
    Estima μ (retornos esperados) y Σ (covarianza), ambos anualizados.
    Valida definida positiva y número de condición.
    Aplica regularización Tikhonov si Σ no es definida positiva.
    """
    tickers    = list(retornos.columns)
    logs       = ["PASO 1 — PARÁMETROS DE ENTRADA (anualizados)"]

    mu_mensual = retornos.mean()
    S_mensual  = retornos.cov()

    mu = mu_mensual * factor
    S  = S_mensual  * factor

    eigenvalores  = np.linalg.eigvalsh(S.values)
    es_def_pos    = bool(np.all(eigenvalores > 0))
    num_condicion = float(np.linalg.cond(S.values))

    if not es_def_pos:
        epsilon      = abs(eigenvalores.min()) + 1e-8
        S_arr        = S.values + epsilon * np.eye(len(tickers))
        S            = pd.DataFrame(S_arr, index=tickers, columns=tickers)
        logs.append(f"ADVERTENCIA: Σ no era definida positiva. "
                    f"Regularización Tikhonov aplicada (ε={epsilon:.2e}).")
        eigenvalores  = np.linalg.eigvalsh(S.values)
        es_def_pos    = bool(np.all(eigenvalores > 0))
        num_condicion = float(np.linalg.cond(S.values))

    logs.append(f"Definida positiva:   {'SÍ' if es_def_pos else 'NO'}")
    logs.append(f"Número de condición: {num_condicion:.2f} "
                f"({'OK' if num_condicion < 1000 else 'ALTA — resultados sensibles'})")
    logs.append(f"Eigenvalores: min={eigenvalores.min():.6f} / max={eigenvalores.max():.6f}")

    if num_condicion > 1000:
        logs.append("NOTA: Número de condición alto (Michaud, 1989). "
                    "Interpretar pesos con precaución.")

    for t, v in mu.items():
        logs.append(f"  μ {t:<20} {v:>8.4f}  ({v*100:.2f}% anual)")

    return {
        'mu':            mu,
        'S':             S,
        'eigenvalores':  eigenvalores,
        'es_def_pos':    es_def_pos,
        'num_condicion': num_condicion,
    }, logs


# =============================================================================
# FUNCIÓN 2 — Portafolio de Mínima Varianza (MVP)
# =============================================================================

def optimizar_mvp(mu: pd.Series, S: pd.DataFrame,
                   rf: float) -> tuple[dict, list]:
    """
    MVP via cvxpy CLARABEL.
    min  w'Σw   s.t. Σwᵢ=1, wᵢ≥0 (sin short-selling)
    Base: Markowitz (1952). Sin restricción de short-selling (wᵢ≥0).
    """
    tickers = list(mu.index)
    n       = len(mu)
    S_mat   = S.values
    w       = cp.Variable(n)
    logs    = ["PASO 2 — PORTAFOLIO DE MÍNIMA VARIANZA (MVP)"]

    objetivo      = cp.Minimize(cp.quad_form(w, S_mat))
    restricciones = [cp.sum(w) == 1]
    if not PERMITIR_VENTAS_CORTO:
        restricciones.append(w >= PESO_MINIMO_ACTIVO)
    if PESO_MAXIMO_ACTIVO < 1.0:
        restricciones.append(w <= PESO_MAXIMO_ACTIVO)

    problema = cp.Problem(objetivo, restricciones)
    problema.solve(solver=cp.CLARABEL)

    if problema.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"MVP no convergió. Estado: {problema.status}")

    pesos       = pd.Series(w.value, index=tickers).round(6)
    retorno     = float(mu.values @ pesos.values)
    varianza    = float(pesos.values @ S_mat @ pesos.values)
    volatilidad = np.sqrt(varianza)
    sharpe      = (retorno - rf) / volatilidad if volatilidad > 0 else 0.0

    logs.append(f"Solver: CLARABEL | Estado: {problema.status}")
    logs.append(f"Retorno esperado:  {retorno*100:.2f}% anual")
    logs.append(f"Volatilidad:       {volatilidad*100:.2f}% anual")
    logs.append(f"Ratio de Sharpe:   {sharpe:.4f}")
    for t, w_val in pesos.items():
        logs.append(f"  {t:<20} {w_val*100:>5.1f}%")

    return {
        'pesos':       pesos,
        'retorno':     retorno,
        'volatilidad': volatilidad,
        'sharpe':      sharpe,
        'rf_anual':    rf,
        'varianza':    varianza,
    }, logs


# =============================================================================
# FUNCIÓN 3 — Portafolio Tangente (Máximo Sharpe)
# =============================================================================

def optimizar_tangente(mu: pd.Series, S: pd.DataFrame,
                        rf: float) -> tuple[dict, list]:
    """
    Portafolio tangente via formulación convexificada.
    Base: Sharpe (1966), Lintner (1965).
    """
    tickers = list(mu.index)
    n       = len(mu)
    S_mat   = S.values
    mu_exc  = (mu - rf).values
    y       = cp.Variable(n)
    logs    = ["PASO 3 — PORTAFOLIO TANGENTE (MÁXIMO SHARPE)"]

    objetivo      = cp.Minimize(cp.quad_form(y, S_mat))
    restricciones = [mu_exc @ y == 1]
    if not PERMITIR_VENTAS_CORTO:
        restricciones.append(y >= 0)
    if PESO_MAXIMO_ACTIVO < 1.0:
        restricciones.append(y <= PESO_MAXIMO_ACTIVO * cp.sum(y))

    problema = cp.Problem(objetivo, restricciones)
    problema.solve(solver=cp.CLARABEL)

    if problema.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError(f"Tangente no convergió. Estado: {problema.status}")

    y_val       = y.value
    pesos       = pd.Series(y_val / y_val.sum(), index=tickers).round(6)
    retorno     = float(mu.values @ pesos.values)
    varianza    = float(pesos.values @ S_mat @ pesos.values)
    volatilidad = np.sqrt(varianza)
    sharpe      = (retorno - rf) / volatilidad

    logs.append(f"Solver: CLARABEL | Estado: {problema.status}")
    logs.append(f"Retorno esperado:  {retorno*100:.2f}% anual")
    logs.append(f"Volatilidad:       {volatilidad*100:.2f}% anual")
    logs.append(f"Ratio de Sharpe:   {sharpe:.4f}")
    for t, w_val in pesos.items():
        logs.append(f"  {t:<20} {w_val*100:>5.1f}%")

    return {
        'pesos':       pesos,
        'retorno':     retorno,
        'volatilidad': volatilidad,
        'sharpe':      sharpe,
    }, logs


# =============================================================================
# FUNCIÓN 4 — Portafolio Equal Weight (1/N)
# =============================================================================

def portafolio_equal_weight(mu: pd.Series, S: pd.DataFrame,
                             rf: float) -> tuple[dict, list]:
    """
    Benchmark 1/N. DeMiguel, Garlappi & Uppal (2009).
    """
    tickers     = list(mu.index)
    n           = len(mu)
    S_mat       = S.values
    pesos       = pd.Series(np.ones(n) / n, index=tickers)
    retorno     = float(mu.values @ pesos.values)
    varianza    = float(pesos.values @ S_mat @ pesos.values)
    volatilidad = np.sqrt(varianza)
    sharpe      = (retorno - rf) / volatilidad
    logs        = [f"PASO 4 — EQUAL WEIGHT (1/N={1/n:.4f} = {100/n:.1f}% por activo)",
                   f"Retorno: {retorno*100:.2f}% | Vol: {volatilidad*100:.2f}% | Sharpe: {sharpe:.4f}"]

    return {
        'pesos':       pesos,
        'retorno':     retorno,
        'volatilidad': volatilidad,
        'sharpe':      sharpe,
    }, logs


# =============================================================================
# FUNCIÓN 5 — Frontera eficiente
# =============================================================================

def calcular_frontera(mu: pd.Series, S: pd.DataFrame,
                       mvp_ret: float, rf: float,
                       n_puntos: int = PUNTOS_FRONTERA_EFICIENTE) -> tuple[pd.DataFrame, list]:
    """
    Calcula n_puntos de la frontera eficiente.
    CORRECCIÓN v0.4.1: rango robusto con retornos negativos (ajuste de signo
    explícito, no usa np.clip).
    """
    n       = len(mu)
    tickers = list(mu.index)
    S_mat   = S.values
    mu_vals = mu.values
    logs    = [f"PASO 5 — FRONTERA EFICIENTE ({n_puntos} puntos)"]

    ret_min = mvp_ret * (1.001 if mvp_ret > 0 else 0.999)
    ret_max = float(mu.max()) * (0.999 if float(mu.max()) > 0 else 1.001)

    if ret_min >= ret_max:
        mu_sorted = np.sort(mu_vals)
        ret_min   = float(mu_sorted[0])
        ret_max   = float(mu_sorted[-1])
        margen    = max(abs(ret_max - ret_min) * 0.001, 1e-6)
        ret_min  += margen
        ret_max  -= margen
        logs.append(f"AVISO: rango ajustado automáticamente [{ret_min:.4f}, {ret_max:.4f}]")

    retornos_objetivo = np.linspace(ret_min, ret_max, n_puntos)
    resultados        = []
    fallos            = 0

    for ret_obj in retornos_objetivo:
        w             = cp.Variable(n)
        objetivo      = cp.Minimize(cp.quad_form(w, S_mat))
        restricciones = [cp.sum(w) == 1, mu_vals @ w == ret_obj]
        if not PERMITIR_VENTAS_CORTO:
            restricciones.append(w >= 0)
        if PESO_MAXIMO_ACTIVO < 1.0:
            restricciones.append(w <= PESO_MAXIMO_ACTIVO)

        problema = cp.Problem(objetivo, restricciones)
        problema.solve(solver=cp.CLARABEL, verbose=False)

        if problema.status in ['optimal', 'optimal_inaccurate']:
            vol = np.sqrt(problema.value)
            if vol > 0:
                resultados.append({
                    'retorno':     ret_obj,
                    'volatilidad': vol,
                    'sharpe':      (ret_obj - rf) / vol,
                })
        else:
            fallos += 1

    df = pd.DataFrame(resultados)
    logs.append(f"{len(df)} puntos calculados | {fallos} no convergieron")
    logs.append(f"Rango vol: {df['volatilidad'].min():.4f} — {df['volatilidad'].max():.4f}")

    return df, logs


# =============================================================================
# FUNCIÓN 6 — Gráfico Plotly (reemplaza graficar_frontera con matplotlib)
# =============================================================================

def graficar_frontera_plotly(frontera: pd.DataFrame, mvp: dict, tangente: dict,
                              equal_weight: dict, mu: pd.Series, S: pd.DataFrame,
                              rf: float) -> go.Figure:
    """
    Gráfico interactivo de la frontera eficiente con Plotly.
    Retorna fig — la página llama st.plotly_chart(fig).

    Panel izquierdo: frontera + CML + portafolios + activos individuales.
    Panel derecho: composición de portafolios (barras apiladas).
    """
    tickers = list(mu.index)
    S_mat   = S.values

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Frontera Eficiente de Markowitz + CML",
                        "Composición de Portafolios"),
        column_widths=[0.6, 0.4],
    )

    # --- Panel izquierdo ---

    # Frontera eficiente (coloreada por Sharpe)
    fig.add_trace(go.Scatter(
        x=frontera['volatilidad'] * 100,
        y=frontera['retorno'] * 100,
        mode='markers',
        marker=dict(
            color=frontera['sharpe'],
            colorscale='RdYlGn',
            size=5,
            colorbar=dict(title="Sharpe", x=0.58),
            showscale=True,
        ),
        name='Frontera eficiente',
        hovertemplate='Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<br>Sharpe: %{marker.color:.3f}',
    ), row=1, col=1)

    # CML — limitada al rango visible del gráfico
    vol_ind     = np.sqrt(np.diag(S_mat)) * 100
    _vol_max_vis = max(frontera['volatilidad'].max() * 100,
                       vol_ind.max()) * 1.05  # 5% de margen visual
    vol_cml = np.linspace(0, _vol_max_vis, 100)
    ret_cml = rf * 100 + tangente['sharpe'] * vol_cml
    fig.add_trace(go.Scatter(
        x=vol_cml, y=ret_cml,
        mode='lines',
        line=dict(color='navy', dash='dash', width=1.5),
        name='Capital Market Line (CML)',
    ), row=1, col=1)

    # MVP
    fig.add_trace(go.Scatter(
        x=[mvp['volatilidad'] * 100], y=[mvp['retorno'] * 100],
        mode='markers',
        marker=dict(color=COLORES['minima_varianza'], size=14, symbol='diamond'),
        name=f"MVP (Sharpe={mvp['sharpe']:.3f})",
        hovertemplate=f"MVP<br>Vol: {mvp['volatilidad']*100:.2f}%<br>"
                      f"Ret: {mvp['retorno']*100:.2f}%<br>Sharpe: {mvp['sharpe']:.3f}",
    ), row=1, col=1)

    # Tangente
    fig.add_trace(go.Scatter(
        x=[tangente['volatilidad'] * 100], y=[tangente['retorno'] * 100],
        mode='markers',
        marker=dict(color=COLORES['portafolio_optimo'], size=16, symbol='star'),
        name=f"Tangente (Sharpe={tangente['sharpe']:.3f})",
        hovertemplate=f"Tangente<br>Vol: {tangente['volatilidad']*100:.2f}%<br>"
                      f"Ret: {tangente['retorno']*100:.2f}%<br>Sharpe: {tangente['sharpe']:.3f}",
    ), row=1, col=1)

    # Equal weight
    fig.add_trace(go.Scatter(
        x=[equal_weight['volatilidad'] * 100], y=[equal_weight['retorno'] * 100],
        mode='markers',
        marker=dict(color='gray', size=12, symbol='square'),
        name=f"1/N (Sharpe={equal_weight['sharpe']:.3f})",
    ), row=1, col=1)

    # Activos individuales
    vol_ind = np.sqrt(np.diag(S_mat)) * 100
    ret_ind = mu.values * 100
    fig.add_trace(go.Scatter(
        x=vol_ind, y=ret_ind,
        mode='markers+text',
        marker=dict(color=COLORES['activos_individuales'], size=9),
        text=tickers,
        textposition='top right',
        textfont=dict(size=9),
        name='Activos individuales',
        hovertemplate='%{text}<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%',
    ), row=1, col=1)

    # Tasa libre de riesgo
    fig.add_trace(go.Scatter(
        x=[0], y=[rf * 100],
        mode='markers',
        marker=dict(color='black', size=10, symbol='triangle-up'),
        name=f"Rf = {rf*100:.2f}%",
    ), row=1, col=1)

    # --- Panel derecho: composición ---
    portafolios = {
        'MVP': mvp['pesos'],
        'Tangente': tangente['pesos'],
        '1/N': equal_weight['pesos'],
    }

    import plotly.express as px
    palette = px.colors.qualitative.Set2
    for i, ticker in enumerate(tickers):
        vals = [portafolios[p].get(ticker, 0) * 100 for p in portafolios]
        fig.add_trace(go.Bar(
            x=list(portafolios.keys()),
            y=vals,
            name=ticker,
            marker_color=palette[i % len(palette)],
            hovertemplate=f'{ticker}: %{{y:.1f}}%',
        ), row=1, col=2)

    fig.update_layout(
        barmode='stack',
        height=520,
        legend=dict(orientation='h', y=-0.18, x=0),
        xaxis=dict(
            title='Volatilidad anual (%)',
            range=[0, _vol_max_vis * 1.05],
        ),
        yaxis=dict(title='Retorno esperado anual (%)'),
        xaxis2=dict(title='Portafolio'),
        yaxis2=dict(title='Peso (%)', range=[0, 105]),
    )

    return fig


# =============================================================================
# FUNCIÓN 7 — Métricas consolidadas
# =============================================================================

def calcular_metricas(pesos: pd.Series, retornos: pd.DataFrame,
                       mu: pd.Series, S: pd.DataFrame,
                       rf: float, nombre: str,
                       factor: int = FACTOR_ANUALIZACION) -> dict:
    """
    Calcula métricas de desempeño anualizadas para un portafolio.
    """
    S_mat    = S.values
    ret_port = (retornos @ pesos).dropna()

    ret_anual   = float(mu.values @ pesos.values)
    var_anual   = float(pesos.values @ S_mat @ pesos.values)
    vol_anual   = np.sqrt(var_anual)
    sharpe      = (ret_anual - rf) / vol_anual if vol_anual > 0 else 0.0

    rf_periodo   = rf / factor
    desvios      = (ret_port - rf_periodo).clip(upper=0)
    downside_std = (np.sqrt((desvios**2).mean()) * np.sqrt(factor)
                    if (desvios < 0).any() else vol_anual)
    sortino      = (ret_anual - rf) / downside_std if downside_std > 0 else 0.0

    precio_idx  = np.exp(ret_port.cumsum())
    max_acum    = precio_idx.cummax()
    drawdown    = (precio_idx - max_acum) / max_acum
    max_drawdown = float(drawdown.min())
    calmar       = ret_anual / abs(max_drawdown) if max_drawdown != 0 else 0.0

    hhi      = float((pesos**2).sum())
    n        = len(pesos)
    hhi_norm = (hhi - 1/n) / (1 - 1/n) if n > 1 else 0.0

    umbral   = rf / factor
    ganancias = ret_port[ret_port > umbral] - umbral
    perdidas  = umbral - ret_port[ret_port <= umbral]
    omega     = (ganancias.sum() / perdidas.sum()
                 if perdidas.sum() > 0 else np.inf)

    return {
        'nombre':        nombre,
        'retorno_anual': round(ret_anual, 4),
        'volatilidad':   round(vol_anual, 4),
        'sharpe':        round(sharpe, 4),
        'sortino':       round(sortino, 4),
        'max_drawdown':  round(max_drawdown, 4),
        'calmar':        round(calmar, 4),
        'omega':         round(omega, 4) if omega != np.inf else 999.0,
        'hhi':           round(hhi, 4),
        'hhi_norm':      round(hhi_norm, 4),
        'concentracion': 'Alta' if hhi_norm > 0.5 else ('Media' if hhi_norm > 0.2 else 'Baja'),
    }


# =============================================================================
# FUNCIÓN 8 — Asistente Fase 3a
# =============================================================================

def asistente_fase3a(mvp: dict, tangente: dict, equal_weight: dict,
                      metricas_mvp: dict, metricas_tang: dict,
                      metricas_ew: dict, metricas_bench: dict,
                      params: dict,
                      res_normalidad: pd.DataFrame = None,
                      res_arch: pd.DataFrame = None,
                      res_mardia: dict = None,
                      nivel: str = 'basico') -> str:
    """
    Interpreta los resultados de la optimización de Markowitz.
    Sigue el contrato: qué ocurrió → por qué importa → valoración → implicación → acción.
    """
    if res_normalidad is None: res_normalidad = pd.DataFrame()
    if res_arch is None:       res_arch       = pd.DataFrame()
    if res_mardia is None:     res_mardia     = {}

    lineas       = []
    sharpe_tang  = metricas_tang['sharpe']
    sharpe_bench = metricas_bench['sharpe']
    sharpe_ew    = metricas_ew['sharpe']
    ret_tang     = tangente['retorno']
    vol_tang     = tangente['volatilidad']
    mdd_tang     = metricas_tang['max_drawdown']

    def add(t=""): lineas.append(t)

    add("## Optimización de Portafolio (Markowitz)")
    add()

    if nivel == 'basico':
        add("### ¿Qué encontramos?")
        add("Construimos el portafolio que mejor equilibra retorno y riesgo con los activos que elegiste. "
            "Hay dos versiones: el que **minimiza el riesgo** (MVP) y el que **maximiza el retorno "
            "por unidad de riesgo** (Portafolio Tangente). Te explicamos el Tangente porque "
            "es el más relevante para un inversor típico.")
        add()

        add("### ¿Qué tan bueno es el portafolio?")
        if sharpe_tang > sharpe_bench:
            add(f"✅ **El portafolio supera al S&P 500 en eficiencia.**")
            add(f"Por cada punto de riesgo asumido, el portafolio genera **{sharpe_tang:.2f}** "
                f"unidades de retorno extra sobre la tasa libre de riesgo — "
                f"vs {sharpe_bench:.2f} del S&P 500. "
                f"Dicho de otro modo: el portafolio da más retorno por el mismo nivel de riesgo que el índice.")
        else:
            add(f"🟡 **El portafolio no supera al S&P 500 en eficiencia (Sharpe {sharpe_tang:.2f} vs {sharpe_bench:.2f}).**")
            add("Esto no lo descarta. El portafolio puede tener menor volatilidad absoluta, "
                "mejor comportamiento en caídas, o mayor diversificación que simplemente comprar el índice.")

        if sharpe_tang < sharpe_ew:
            add(f"\n⚠️ **El portafolio equitativo (1/N) tiene mejor eficiencia que el optimizado "
                f"({sharpe_ew:.2f} vs {sharpe_tang:.2f}).**")
            add("Esto ocurre porque la optimización de Markowitz es sensible a los errores en la "
                "estimación de retornos esperados — pequeños errores se amplifican en los pesos. "
                "En la práctica, repartir igual entre activos a veces funciona mejor que optimizar.")

        add()
        add("### ¿Cuánto puede perder?")
        add(f"El portafolio tuvo una caída máxima de **{mdd_tang*100:.1f}%** en el período analizado — "
            f"la mayor pérdida acumulada desde un máximo hasta el siguiente mínimo. "
            f"El retorno esperado anual es **{ret_tang*100:.1f}%** con una volatilidad de **{vol_tang*100:.1f}%**.")
        add()
        add("**Acción:** revisa la composición del portafolio en la tabla de pesos. "
            "Si algún activo concentra más del 50%, considera si eso refleja tu intención.")

    else:
        add("### Portafolio Tangente vs Benchmarks")
        add("| Portafolio | Retorno | Volatilidad | Sharpe | Sortino | Max DD | Calmar |")
        add("|---|---|---|---|---|---|---|")
        for m in [metricas_tang, metricas_mvp, metricas_ew, metricas_bench]:
            sortino = m.get('sortino', '—')
            calmar  = m.get('calmar', '—')
            ret_m   = m.get('retorno_anual', '—')
            vol_m   = m.get('volatilidad', '—')
            add(f"| {m['nombre']} | {ret_m} | {vol_m} | {m['sharpe']} | "
                f"{sortino} | {m['max_drawdown']} | {calmar} |")

        add()
        cond = params.get('num_condicion', 0)
        if cond > 100:
            add(f"⚠️ **Número de condición de Σ: {cond:.1f}** — matriz de covarianza mal condicionada. "
                "Los pesos son sensibles a perturbaciones en los inputs. "
                "Regularización Tikhonov aplicada si fue necesario.")

        if sharpe_tang < sharpe_ew:
            add(f"\n⚠️ **1/N supera al Tangente en Sharpe ({sharpe_ew:.3f} vs {sharpe_tang:.3f}).**")
            add("Error de estimación de Michaud (1989): la media muestral como estimador de μ "
                "tiene alta varianza. La optimización amplifica estos errores. "
                "Alternativas: MVP (no requiere estimación de μ), Black-Litterman (Capa 2), "
                "o Robust Optimization.")

        if not res_normalidad.empty:
            no_norm = res_normalidad[res_normalidad['Semaforo'] == 'ROJO']['Ticker'].tolist()
            if no_norm:
                add(f"\n⚠️ **Supuesto de normalidad violado en `{no_norm}` (Fase 2).** "
                    "La frontera eficiente es una aproximación — CVaR es la métrica de riesgo primaria.")

        if 'NO NORMAL' in res_mardia.get('diagnostico', ''):
            add("\n⚠️ **Normalidad multivariante rechazada (Fase 2).** "
                "En escenarios de estrés, correlaciones aumentan (Longin & Solnik, 2001) — "
                "la diversificación se debilita exactamente cuando más se necesita.")

    add()
    add("---")
    if nivel == 'basico':
        add("*Los pesos aquí calculados son el punto de partida óptimo según los datos históricos. "
            "No garantizan el mismo desempeño en el futuro — úsalos como referencia, no como receta.*")
    else:
        add("*Próximo paso: Fase 3b — Modelos de Factores para descomponer las fuentes de retorno.*")

    return "\n".join(lineas)


# =============================================================================
# FUNCIÓN PIPELINE — Envuelve todo el flujo de la Fase 3a
# =============================================================================

def pipeline_markowitz(retornos: pd.DataFrame,
                        retornos_benchmark: pd.DataFrame,
                        nivel_asistente: str = 'basico',
                        res_normalidad: pd.DataFrame = None,
                        res_arch: pd.DataFrame = None,
                        res_mardia: dict = None,
                        factor: int = FACTOR_ANUALIZACION) -> dict:
    """
    Ejecuta el flujo completo de la Fase 3a.

    Parámetros
    ----------
    retornos             : pd.DataFrame — de session_state['retornos']
    retornos_benchmark   : pd.DataFrame — de session_state['retornos_benchmark']
    nivel_asistente      : str — 'basico' o 'tecnico'
    res_normalidad/arch/mardia : contexto de Fase 2 (opcionales)

    Retorna
    -------
    dict con: mu, S, frontera, mvp, tangente, equal_weight,
              metricas_3a, pesos_df, res_3a, fig_frontera,
              narrativa_3a, logs, error
    """
    logs_totales = []

    def _log(msgs):
        if isinstance(msgs, list):
            logs_totales.extend(msgs)
        else:
            logs_totales.append(msgs)

    try:
        # 1. Tasa libre de riesgo
        rf, lg = _obtener_rf()
        _log(lg)

        # 2. Parámetros
        params, lg = estimar_parametros(retornos, factor=factor)
        _log(lg)
        mu = params['mu']
        S  = params['S']

        # 3. Portafolios
        mvp,          lg = optimizar_mvp(mu, S, rf)
        _log(lg)
        tangente,     lg = optimizar_tangente(mu, S, rf)
        _log(lg)
        equal_weight, lg = portafolio_equal_weight(mu, S, rf)
        _log(lg)

        # 4. Frontera
        frontera, lg = calcular_frontera(mu, S, mvp['retorno'], rf)
        _log(lg)

        # 5. Métricas
        metricas_mvp  = calcular_metricas(mvp['pesos'],         retornos, mu, S, rf, 'MVP')
        metricas_tang = calcular_metricas(tangente['pesos'],     retornos, mu, S, rf, 'Tangente')
        metricas_ew   = calcular_metricas(equal_weight['pesos'], retornos, mu, S, rf, '1/N Equal Weight')

        ret_bench   = retornos_benchmark.iloc[:, 0].dropna()
        ret_bench_a = float(ret_bench.mean() * factor)
        vol_bench_a = float(ret_bench.std() * np.sqrt(factor))
        sharpe_b    = (ret_bench_a - rf) / vol_bench_a
        precio_b    = np.exp(ret_bench.cumsum())
        dd_b        = (precio_b - precio_b.cummax()) / precio_b.cummax()
        metricas_bench = {
            'nombre':        'S&P 500 (benchmark)',
            'retorno_anual': round(ret_bench_a, 4),
            'volatilidad':   round(vol_bench_a, 4),
            'sharpe':        round(sharpe_b, 4),
            'sortino':       '—',
            'max_drawdown':  round(float(dd_b.min()), 4),
            'calmar':        '—', 'omega': '—',
            'hhi': '—', 'hhi_norm': '—', 'concentracion': '—',
        }

        metricas_3a = pd.DataFrame([metricas_mvp, metricas_tang,
                                     metricas_ew, metricas_bench])

        pesos_df = pd.DataFrame({
            'MVP':         mvp['pesos'],
            'Tangente':    tangente['pesos'],
            'EqualWeight': equal_weight['pesos'],
        })

        res_3a = {
            'rf_anual':               rf,
            'num_condicion_sigma':    params['num_condicion'],
            'sigma_definida_positiva': params['es_def_pos'],
            'mvp':         {'retorno': mvp['retorno'], 'volatilidad': mvp['volatilidad'],
                            'sharpe': mvp['sharpe']},
            'tangente':    {'retorno': tangente['retorno'], 'volatilidad': tangente['volatilidad'],
                            'sharpe': tangente['sharpe']},
            'equal_weight':{'retorno': equal_weight['retorno'], 'volatilidad': equal_weight['volatilidad'],
                            'sharpe': equal_weight['sharpe']},
            'benchmark':   {'retorno': metricas_bench['retorno_anual'],
                            'volatilidad': metricas_bench['volatilidad'],
                            'sharpe': metricas_bench['sharpe']},
        }

        # 6. Gráfico Plotly
        fig_frontera = graficar_frontera_plotly(
            frontera, mvp, tangente, equal_weight, mu, S, rf
        )

        # 7. Asistente
        narrativa = asistente_fase3a(
            mvp, tangente, equal_weight,
            metricas_mvp, metricas_tang, metricas_ew, metricas_bench,
            params, res_normalidad, res_arch, res_mardia,
            nivel=nivel_asistente,
        )

        _log("✅ FASE 3a COMPLETADA")

        return {
            'mu':            mu,
            'S':             S,
            'frontera':      frontera,
            'mvp':           mvp,
            'tangente':      tangente,
            'equal_weight':  equal_weight,
            'metricas_3a':   metricas_3a,
            'pesos_df':      pesos_df,
            'res_3a':        res_3a,
            'fig_frontera':  fig_frontera,
            'narrativa_3a':  narrativa,
            'logs':          logs_totales,
            'error':         None,
        }

    except Exception as e:
        import traceback
        _log(f"ERROR FATAL en pipeline_markowitz: {e}")
        _log(traceback.format_exc())
        return {'error': str(e), 'logs': logs_totales}
