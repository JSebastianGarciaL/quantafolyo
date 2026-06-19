# =============================================================================
# QuantαfolyΩ — modulos/riesgo.py
# Fase 3c: Métricas de Riesgo
#
# Funciones migradas desde PortfolioLab_V0_7_0.ipynb (celdas 59–66).
# REGLA DE MIGRACIÓN: lógica matemática sin cambios.
# Cambios:
#   - print() → entradas en lista `logs`
#   - graficar_riesgo() reescrita en Plotly (retorna fig)
#   - asistente_fase3c() retorna str markdown
#   - ESCENARIOS_ESTRES se mantiene en este archivo (datos fijos, no parámetro configurable)
#   - RUTA_PROYECTO / Drive eliminados
#   - pipeline_riesgo() envuelve todo el flujo
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    ALPHA,
    FACTOR_ANUALIZACION,
    NIVEL_CONFIANZA_VAR,
    NIVEL_CONFIANZA_VAR_ESTRICTO,
    N_SIMULACIONES_MC,
    COLORES,
)

# Escenarios de estrés histórico (definidos aquí por ser datos fijos, no parámetros configurables)
ESCENARIOS_ESTRES = {
    'Crisis Financiera 2008': ('2008-01-01', '2009-06-30'),
    'COVID-19 2020':          ('2020-01-01', '2020-06-30'),
    'Bear Market 2022':       ('2022-01-01', '2022-12-31'),
    'Crisis bancaria 2023':   ('2023-01-01', '2023-06-30'),
}


# =============================================================================
# FUNCIÓN 1 — VaR (tres métodos)
# =============================================================================

def calcular_var(ret: pd.Series, nivel: float,
                 n_simulaciones: int = N_SIMULACIONES_MC) -> dict:
    """
    VaR por tres métodos. Retorna VaR como número positivo (pérdida).

    1. Histórico: cuantil empírico — sin supuestos distribucionales
    2. Paramétrico: normalidad — fórmula cerrada
    3. Monte Carlo: simulación bajo normalidad con μ y σ muestrales

    Base: RiskMetrics (J.P. Morgan, 1994)
    """
    alpha_var = 1 - nivel
    mu        = ret.mean()
    sigma     = ret.std()

    # 1. Histórico
    var_hist  = float(-np.percentile(ret, alpha_var * 100))

    # 2. Paramétrico
    # VaR = -(μ + σ × z_α), z_α = norm.ppf(alpha_var) < 0
    z_alpha   = norm.ppf(alpha_var)
    var_param = float(-(mu + sigma * z_alpha))

    # 3. Monte Carlo
    np.random.seed(42)
    sim    = np.random.normal(mu, sigma, n_simulaciones)
    var_mc = float(-np.percentile(sim, alpha_var * 100))

    return {
        'nivel':     nivel,
        'alpha':     alpha_var,
        'mu':        mu,
        'sigma':     sigma,
        'var_hist':  var_hist,
        'var_param': var_param,
        'var_mc':    var_mc,
    }


# =============================================================================
# FUNCIÓN 2 — CVaR / Expected Shortfall
# =============================================================================

def calcular_cvar(ret: pd.Series, nivel: float,
                  n_simulaciones: int = N_SIMULACIONES_MC) -> dict:
    """
    CVaR por tres métodos. Retorna CVaR como número positivo.

    CORRECCIÓN v0.6.2: CVaR paramétrico — fórmula y signo verificados.
    Fórmula: CVaR = -(μ - σ × φ(z_α) / α)
    donde φ es la densidad normal (siempre positiva).

    Base: Artzner et al. (1999)
    """
    alpha_var = 1 - nivel
    mu        = ret.mean()
    sigma     = ret.std()

    # 1. Histórico
    umbral_hist = np.percentile(ret, alpha_var * 100)
    cola_hist   = ret[ret <= umbral_hist]
    cvar_hist   = float(-cola_hist.mean()) if len(cola_hist) > 0 else 0.0

    # 2. Paramétrico
    z_alpha    = norm.ppf(alpha_var)
    densidad   = norm.pdf(z_alpha)           # > 0 siempre
    cvar_param = float(-(mu - sigma * densidad / alpha_var))

    # 3. Monte Carlo
    np.random.seed(42)
    sim        = np.random.normal(mu, sigma, n_simulaciones)
    umbral_mc  = np.percentile(sim, alpha_var * 100)
    cola_mc    = sim[sim <= umbral_mc]
    cvar_mc    = float(-cola_mc.mean()) if len(cola_mc) > 0 else 0.0

    return {
        'nivel':      nivel,
        'cvar_hist':  cvar_hist,
        'cvar_param': cvar_param,
        'cvar_mc':    cvar_mc,
    }


# =============================================================================
# FUNCIÓN 3 — Backtesting (Kupiec + Christoffersen)
# =============================================================================

def backtesting_var(ret: pd.Series, nivel: float,
                    alpha_test: float) -> dict:
    """
    Backtesting del VaR.
    Kupiec (1995) — prueba POF: frecuencia de violaciones.
    Christoffersen (1998) — independencia de violaciones.

    CORRECCIÓN v0.6.1: n11=0 reporta "SIN PODER ESTADÍSTICO"
    en lugar de aprobar por default.
    """
    alpha_var   = 1 - nivel
    n           = len(ret)
    var_fijo    = -np.percentile(ret, alpha_var * 100)
    violaciones = (ret < -var_fijo).astype(int)
    n_viol      = int(violaciones.sum())
    p_hat       = n_viol / n
    p_teorica   = alpha_var

    # Kupiec
    if n_viol == 0:
        lr_kupiec   = 0.0
        p_kupiec    = 1.0
        kupiec_ok   = True
        nota_kupiec = "Sin violaciones — VaR conservador o muestra pequeña"
    elif n_viol == n:
        lr_kupiec   = np.inf
        p_kupiec    = 0.0
        kupiec_ok   = False
        nota_kupiec = "Todas las obs son violaciones — VaR inútil"
    else:
        lr_kupiec = -2 * (
            n_viol * np.log(p_teorica / p_hat) +
            (n - n_viol) * np.log((1 - p_teorica) / (1 - p_hat))
        )
        p_kupiec    = float(1 - stats.chi2.cdf(lr_kupiec, df=1))
        kupiec_ok   = p_kupiec > alpha_test
        nota_kupiec = ""

    # Christoffersen
    v   = violaciones.values
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    if n11 == 0 and n10 == 0:
        lr_christ   = None
        p_christ    = None
        christ_ok   = None
        nota_christ = ("SIN PODER ESTADÍSTICO — no hay violaciones consecutivas. "
                       "Con n≈60 obs y α=5% es esperable. No interpretar como validación.")
    elif 0 < p01 < 1 and 0 < p11 < 1:
        p_hat_c = (n01 + n11) / (n - 1)
        try:
            lr_christ = -2 * (
                (n00 + n10) * np.log(1 - p_hat_c) +
                (n01 + n11) * np.log(p_hat_c) -
                (n00 * np.log(1 - p01) + n01 * np.log(p01) +
                 n10 * np.log(1 - p11) + n11 * np.log(p11))
            )
            p_christ    = float(1 - stats.chi2.cdf(lr_christ, df=1))
            christ_ok   = p_christ > alpha_test
            nota_christ = ""
        except Exception:
            lr_christ   = None
            p_christ    = None
            christ_ok   = None
            nota_christ = "Error numérico en el cálculo."
    else:
        lr_christ   = None
        p_christ    = None
        christ_ok   = None
        nota_christ = "SIN PODER ESTADÍSTICO — probabilidades de transición en los límites."

    return {
        'n':           n,
        'n_viol':      n_viol,
        'p_hat':       round(p_hat, 4),
        'p_teorica':   round(p_teorica, 4),
        'var_fijo':    round(var_fijo, 4),
        'lr_kupiec':   round(lr_kupiec, 4) if lr_kupiec != np.inf else np.inf,
        'p_kupiec':    round(p_kupiec, 4),
        'kupiec_ok':   kupiec_ok,
        'nota_kupiec': nota_kupiec,
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
        'p01':         round(p01, 4),
        'p11':         round(p11, 4),
        'lr_christ':   round(lr_christ, 4) if lr_christ is not None else None,
        'p_christ':    round(p_christ, 4)  if p_christ  is not None else None,
        'christ_ok':   christ_ok,
        'nota_christ': nota_christ,
    }


# =============================================================================
# FUNCIÓN 4 — Métricas avanzadas
# =============================================================================

def calcular_metricas_avanzadas(ret_log: pd.Series,
                                 ret_bench_log: pd.Series,
                                 beta_port: float,
                                 rf_mes: float,
                                 rf_anual: float,
                                 factor: int = FACTOR_ANUALIZACION) -> dict:
    """
    Treynor, Sortino, Calmar, Omega, Ratio de Información.
    CORRECCIÓN v0.6.1: IR calculado en retornos simples (no logarítmicos).
    """
    ret_anual = float(ret_log.mean() * factor)
    vol_anual = float(ret_log.std() * np.sqrt(factor))

    treynor = (ret_anual - rf_anual) / beta_port if beta_port != 0 else None
    sharpe  = (ret_anual - rf_anual) / vol_anual if vol_anual > 0 else None

    desvios  = (ret_log - rf_mes).clip(upper=0)
    down_std = (float(np.sqrt((desvios**2).mean()) * np.sqrt(factor))
                if (desvios < 0).any() else vol_anual)
    sortino  = (ret_anual - rf_anual) / down_std if down_std > 0 else None

    precio_idx = np.exp(ret_log.cumsum())
    drawdown   = (precio_idx - precio_idx.cummax()) / precio_idx.cummax()
    max_dd     = float(drawdown.min())
    calmar     = ret_anual / abs(max_dd) if max_dd != 0 else None

    umbral_mes = rf_mes
    ganancias  = (ret_log - umbral_mes).clip(lower=0).sum()
    perdidas   = (umbral_mes - ret_log).clip(lower=0).sum()
    omega      = float(ganancias / perdidas) if perdidas > 0 else np.inf

    # IR en retornos simples (CORRECCIÓN)
    ret_simple_port  = np.exp(ret_log) - 1
    ret_simple_bench = np.exp(ret_bench_log.reindex(ret_log.index).dropna()) - 1
    idx_comun        = ret_simple_port.index.intersection(ret_simple_bench.index)
    ret_activo       = ret_simple_port.loc[idx_comun] - ret_simple_bench.loc[idx_comun]
    tracking_err     = float(ret_activo.std() * np.sqrt(factor))
    ret_activo_a     = float(ret_activo.mean() * factor)
    ir               = ret_activo_a / tracking_err if tracking_err > 0 else None

    return {
        'ret_anual':    round(ret_anual, 4),
        'vol_anual':    round(vol_anual, 4),
        'sharpe':       round(sharpe, 4)  if sharpe  is not None else None,
        'treynor':      round(treynor, 4) if treynor is not None else None,
        'sortino':      round(sortino, 4) if sortino is not None else None,
        'max_drawdown': round(max_dd, 4),
        'calmar':       round(calmar, 4)  if calmar  is not None else None,
        'omega':        round(omega, 4)   if omega   != np.inf   else 999.0,
        'ir':           round(ir, 4)      if ir      is not None else None,
        'tracking_err': round(tracking_err, 4),
        'ret_activo_a': round(ret_activo_a, 4),
    }


# =============================================================================
# FUNCIÓN 5 — Análisis de estrés histórico
# =============================================================================

def analisis_estres(retornos: pd.DataFrame, pesos: pd.Series,
                    escenarios: dict,
                    factor: int = FACTOR_ANUALIZACION) -> tuple[pd.DataFrame, list]:
    """
    Aplica pesos actuales a períodos de crisis históricos.
    """
    logs       = ["ANÁLISIS DE ESTRÉS HISTÓRICO",
                  "NOTA: análisis retrospectivo 'what-if' — aplica los pesos actuales "
                  "a períodos de crisis pasados. No implica que el portafolio existía en esas fechas."]
    resultados = []

    for nombre, (fecha_ini, fecha_fin) in escenarios.items():
        try:
            ret_periodo  = retornos.loc[fecha_ini:fecha_fin]
            if ret_periodo.empty:
                logs.append(f"  {nombre}: SIN DATOS en {fecha_ini}—{fecha_fin}")
                continue

            tickers_disp = [t for t in pesos.index if t in ret_periodo.columns]
            if not tickers_disp:
                logs.append(f"  {nombre}: ningún activo tiene datos")
                continue

            pesos_norm = pesos[tickers_disp] / pesos[tickers_disp].sum()
            ret_port_e = ret_periodo[tickers_disp] @ pesos_norm

            ret_acum = float(np.exp(ret_port_e.sum()) - 1)
            vol      = float(ret_port_e.std() * np.sqrt(factor))
            precio_e = np.exp(ret_port_e.cumsum())
            mdd      = float(((precio_e - precio_e.cummax()) / precio_e.cummax()).min())
            n_meses  = len(ret_port_e)

            logs.append(f"  {nombre} ({n_meses} meses): ret={ret_acum*100:.2f}% | "
                        f"MDD={mdd*100:.2f}% | vol={vol*100:.2f}%")

            resultados.append({
                'Escenario':    nombre,
                'Fecha_inicio': fecha_ini,
                'Fecha_fin':    fecha_fin,
                'N_meses':      n_meses,
                'Ret_acum':     round(ret_acum, 4),
                'Max_DD':       round(mdd, 4),
                'Vol_anual':    round(vol, 4),
            })

        except Exception as e:
            logs.append(f"  {nombre}: ERROR — {e}")

    return pd.DataFrame(resultados), logs


# =============================================================================
# FUNCIÓN 6 — Gráfico Plotly (reemplaza graficar_riesgo matplotlib)
# =============================================================================

def graficar_riesgo_plotly(ret_port: pd.Series, ret_bench: pd.Series,
                            resultados_var: dict,
                            resultados_estres: pd.DataFrame) -> go.Figure:
    """
    4 paneles interactivos:
    1. Distribución de retornos vs normal con líneas VaR
    2. Drawdown histórico
    3. VaR vs CVaR comparativo por método
    4. Retornos acumulados portafolio vs benchmark
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Distribución de retornos vs Normal teórica",
            "Drawdown histórico",
            "VaR vs CVaR por método (95%)",
            "Retornos acumulados: Portafolio vs S&P 500",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.10,
    )

    # --- Panel 1: Histograma + normal ---
    ret_vals  = ret_port.dropna().values
    mu_r, sig_r = ret_vals.mean(), ret_vals.std()
    x_rng = np.linspace(ret_vals.min(), ret_vals.max(), 200)

    fig.add_trace(go.Histogram(
        x=ret_vals, histnorm='probability density',
        name='Retornos reales', opacity=0.65,
        marker_color=COLORES['portafolio_optimo'],
        nbinsx=20,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_rng, y=norm.pdf(x_rng, mu_r, sig_r),
        mode='lines', name='Normal teórica',
        line=dict(color='red', width=2),
    ), row=1, col=1)

    var_95 = resultados_var['95%']['var_hist']
    var_99 = resultados_var['99%']['var_hist']
    fig.add_vline(x=-var_95, line_dash='dash', line_color='orange',
                  annotation_text=f"VaR 95%={var_95:.3f}", row=1, col=1)
    fig.add_vline(x=-var_99, line_dash='dash', line_color='red',
                  annotation_text=f"VaR 99%={var_99:.3f}", row=1, col=1)

    # --- Panel 2: Drawdown ---
    precio_idx = np.exp(ret_port.dropna().cumsum())
    drawdown   = (precio_idx - precio_idx.cummax()) / precio_idx.cummax() * 100

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        mode='lines', fill='tozeroy',
        name='Drawdown (%)',
        line=dict(color=COLORES['benchmark'], width=1),
        fillcolor='rgba(158,158,158,0.3)',
    ), row=1, col=2)

    # --- Panel 3: VaR vs CVaR ---
    metodos  = ['Histórico', 'Paramétrico', 'Monte Carlo']
    var_95v  = [resultados_var['95%']['var_hist'],
                resultados_var['95%']['var_param'],
                resultados_var['95%']['var_mc']]
    cvar_95v = [resultados_var['95%']['cvar_hist'],
                resultados_var['95%']['cvar_param'],
                resultados_var['95%']['cvar_mc']]

    fig.add_trace(go.Bar(x=metodos, y=var_95v, name='VaR 95%',
                         marker_color=COLORES['portafolio_optimo'], opacity=0.85),
                  row=2, col=1)
    fig.add_trace(go.Bar(x=metodos, y=cvar_95v, name='CVaR 95%',
                         marker_color=COLORES['minima_varianza'], opacity=0.85),
                  row=2, col=1)

    # --- Panel 4: Retornos acumulados ---
    idx_comun = ret_port.index.intersection(ret_bench.index)
    precio_p  = np.exp(ret_port.loc[idx_comun].dropna().cumsum())
    precio_b  = np.exp(ret_bench.loc[idx_comun].dropna().cumsum())

    fig.add_trace(go.Scatter(
        x=precio_p.index, y=precio_p,
        mode='lines', name='Portafolio tangente',
        line=dict(color=COLORES['portafolio_optimo'], width=2),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=precio_b.index, y=precio_b,
        mode='lines', name='S&P 500',
        line=dict(color=COLORES['benchmark'], width=2, dash='dash'),
    ), row=2, col=2)

    fig.update_layout(
        height=600,
        barmode='group',
        legend=dict(orientation='h', y=-0.12),
        title_text='Análisis de Riesgo — Portafolio Tangente',
    )
    fig.update_yaxes(title_text='Drawdown (%)', row=1, col=2)
    fig.update_yaxes(title_text='Pérdida mensual', row=2, col=1)
    fig.update_yaxes(title_text='Valor (base=1)', row=2, col=2)

    return fig


# =============================================================================
# FUNCIÓN 7 — Asistente Fase 3c
# =============================================================================

def asistente_fase3c(resultados_var: dict, resultados_bt: dict,
                      metricas: dict, resultados_estres: pd.DataFrame,
                      nivel: str = 'basico') -> str:
    """
    Interpreta las métricas de riesgo.
    Sigue el contrato: qué ocurrió → por qué importa → valoración → implicación → acción.
    """
    lineas  = []
    def add(t=""): lineas.append(t)

    var_95  = resultados_var.get('95%', {}).get('var_hist', 0)
    cvar_95 = resultados_var.get('95%', {}).get('cvar_hist', 0)
    var_99  = resultados_var.get('99%', {}).get('var_hist', 0)
    cvar_99 = resultados_var.get('99%', {}).get('cvar_hist', 0)
    bt_95   = resultados_bt.get('95%', {})
    sharpe  = metricas.get('sharpe', 0) or 0
    mdd     = metricas.get('max_drawdown', 0) or 0

    add("## Métricas de Riesgo")
    add()

    if nivel == 'basico':
        add("### ¿Cuánto puede perder el portafolio en un mes malo?")
        add(f"El **VaR al 95%** es **{var_95*100:.2f}%**: en el 95% de los meses, "
            f"el portafolio no pierde más que eso.")
        add(f"El **CVaR al 95%** es **{cvar_95*100:.2f}%**: en el 5% de los meses más malos, "
            f"la pérdida promedio llega hasta ese nivel.")
        add()
        add("La diferencia entre VaR y CVaR importa: el VaR te dice dónde está el límite del 95% "
            "de los días normales. El CVaR te dice qué tan malos son el 5% restante — "
            "los días que realmente duelen.")
        add()

        add("### ¿El modelo de riesgo fue preciso?")
        kupiec_ok = bt_95.get('kupiec_ok', True)
        n_viol    = bt_95.get('n_viol', 0)
        n_obs     = bt_95.get('n', 0)
        nota      = bt_95.get('nota_christ', '')

        if kupiec_ok:
            add(f"✅ **El VaR estuvo bien calibrado.** "
                f"Hubo {n_viol} meses con pérdidas superiores al VaR de {n_obs} observados — "
                f"consistente con lo esperado al 95% de confianza.")
        else:
            if n_obs > 0 and n_viol > n_obs * 0.05:
                add(f"⚠️ **El VaR subestimó el riesgo real.** "
                    f"Hubo {n_viol} violaciones vs las ~{n_obs*0.05:.1f} esperadas. "
                    "El modelo es más optimista de lo que los datos indican.")
            else:
                add(f"🟡 **El VaR fue demasiado conservador** — pocas violaciones observadas. "
                    "Esto puede indicar un período excepcionalmente bueno.")

        if nota and 'SIN PODER' in nota:
            add(f"*Nota: con {n_obs} observaciones, el test de independencia no tiene "
                "poder estadístico suficiente. Es normal con muestras mensuales cortas.*")
        add()

        if not resultados_estres.empty:
            add("### ¿Cómo habría resistido el portafolio en crisis históricas?")
            for _, row in resultados_estres.iterrows():
                ret_e = row['Ret_acum'] * 100
                mdd_e = row['Max_DD'] * 100
                icono = "🔴" if ret_e < -20 else ("🟡" if ret_e < -10 else "🟢")
                add(f"{icono} **{row['Escenario']}:** retorno acumulado {ret_e:.1f}%, "
                    f"caída máxima {mdd_e:.1f}%")
            add()
            add("*Este análisis aplica los pesos actuales a períodos pasados — "
                "es un ejercicio 'qué hubiera pasado', no una predicción.*")
        add()
        add("**Acción:** si el CVaR supera tu tolerancia al riesgo, considera reducir "
            "la exposición a los activos con mayor beta o aumentar la diversificación.")

    else:
        add("### VaR y CVaR — Portafolio Tangente (mensual)")
        add("| Nivel | VaR Hist | VaR Param | VaR MC | CVaR Hist | CVaR Param | CVaR MC |")
        add("|---|---|---|---|---|---|---|")
        for etiq, res in resultados_var.items():
            add(f"| {etiq} | {res.get('var_hist',0):.4f} | {res.get('var_param',0):.4f} | "
                f"{res.get('var_mc',0):.4f} | {res.get('cvar_hist',0):.4f} | "
                f"{res.get('cvar_param',0):.4f} | {res.get('cvar_mc',0):.4f} |")

        diff = abs(var_95 - resultados_var.get('95%', {}).get('var_param', var_95))
        if diff > 0.01:
            add(f"\n⚠️ **VaR hist vs paramétrico difieren en {diff:.4f}** — "
                "desviación de normalidad confirmada. VaR histórico es la métrica primaria.")

        add()
        add("### Backtesting (Kupiec + Christoffersen)")
        for etiq, bt in resultados_bt.items():
            p_k     = bt.get('p_kupiec')
            ok_k    = bt.get('kupiec_ok')
            p_c     = bt.get('p_christ')
            ok_c    = bt.get('christ_ok')
            nota_c  = bt.get('nota_christ', '')
            add(f"**{etiq}:** Kupiec p={p_k:.4f} {'✅' if ok_k else '❌'} | "
                + (f"Christoffersen p={p_c:.4f} {'✅' if ok_c else '❌'}" if p_c is not None
                   else f"Christoffersen: {nota_c[:60]}"))

        add()
        add("### Ratios de desempeño (portafolio tangente, anualizados)")
        add("| Ratio | Valor | Qué mide |")
        add("|---|---|---|")
        add(f"| Sharpe  | {metricas.get('sharpe','—')} | Retorno por unidad de riesgo total (σ) |")
        add(f"| Treynor | {metricas.get('treynor','—')} | Retorno por unidad de riesgo sistemático (β) |")
        add(f"| Sortino | {metricas.get('sortino','—')} | Retorno por unidad de riesgo a la baja |")
        add(f"| Calmar  | {metricas.get('calmar','—')} | Retorno anual / Max Drawdown |")
        add(f"| Omega   | {metricas.get('omega','—')} | Σganancias / Σpérdidas sobre Rf |")
        ir     = metricas.get('ir')
        t_err  = metricas.get('tracking_err')
        if ir is not None:
            ir_desc = "excepcional (>1)" if (ir or 0) > 1 else ("bueno (0.5-1)" if (ir or 0) > 0.5 else "subperforma benchmark")
            add(f"| IR      | {ir} | Retorno activo / Tracking error — {ir_desc} (Grinold & Kahn, 2000) |")

        if not resultados_estres.empty:
            add()
            add("### Estrés histórico")
            add("| Escenario | Ret. acumulado | Max Drawdown | Vol. anual |")
            add("|---|---|---|---|")
            for _, row in resultados_estres.iterrows():
                add(f"| {row['Escenario']} | {row['Ret_acum']*100:.2f}% | "
                    f"{row['Max_DD']*100:.2f}% | {row['Vol_anual']*100:.2f}% |")

    add()
    add("---")
    if nivel == 'basico':
        add("*El riesgo no es malo — es el precio del retorno. Lo que importa es conocerlo y decidir si es aceptable.*")
    else:
        add("*Próximo paso: Verificación consolidada — spanning, Chow y consistencia entre modelos.*")

    return "\n".join(lineas)


# =============================================================================
# PIPELINE — Fase 3c completa
# =============================================================================

def pipeline_riesgo(retornos: pd.DataFrame,
                    retornos_benchmark: pd.DataFrame,
                    pesos_df: pd.DataFrame,
                    res_capm: pd.DataFrame,
                    rf_anual: float,
                    nivel_asistente: str = 'basico',
                    factor: int = FACTOR_ANUALIZACION) -> dict:
    """
    Ejecuta VaR, CVaR, backtesting, métricas avanzadas, estrés.

    Parámetros
    ----------
    pesos_df   : de session_state['pesos_df'] — columnas MVP, Tangente, EqualWeight
    res_capm   : de session_state['res_capm'] — necesario para beta del portafolio
    rf_anual   : de session_state['res_3a']['rf_anual']

    Retorna
    -------
    dict con: res_var, metricas_3c, res_bt, res_estres,
              fig_riesgo, narrativa_3c, logs, error
    """
    logs_totales = []

    def _log(msgs):
        if isinstance(msgs, list):
            logs_totales.extend(msgs)
        else:
            logs_totales.append(msgs)

    try:
        rf_mes  = rf_anual / factor

        pesos_tang = pesos_df['Tangente']
        ret_port   = (retornos @ pesos_tang).dropna()
        ret_bench  = retornos_benchmark.iloc[:, 0].reindex(ret_port.index).dropna()
        ret_port   = ret_port.reindex(ret_bench.index)

        # VaR y CVaR para ambos niveles
        _log("BLOQUE 1 — VaR y CVaR")
        resultados_var = {}
        var_rows       = []

        for nivel, etiq in [(NIVEL_CONFIANZA_VAR, '95%'),
                             (NIVEL_CONFIANZA_VAR_ESTRICTO, '99%')]:
            var_res  = calcular_var(ret_port, nivel)
            cvar_res = calcular_cvar(ret_port, nivel)
            resultados_var[etiq] = {**var_res, **cvar_res}

            _log(f"  {etiq}: VaR_hist={var_res['var_hist']:.4f} | "
                 f"CVaR_hist={cvar_res['cvar_hist']:.4f}")

            var_rows.append({
                'Nivel':      etiq,
                'VaR_hist':   var_res['var_hist'],
                'VaR_param':  var_res['var_param'],
                'VaR_mc':     var_res['var_mc'],
                'CVaR_hist':  cvar_res['cvar_hist'],
                'CVaR_param': cvar_res['cvar_param'],
                'CVaR_mc':    cvar_res['cvar_mc'],
            })

        res_var = pd.DataFrame(var_rows)

        # Backtesting
        _log("BLOQUE 2 — Backtesting")
        resultados_bt = {}
        bt_rows       = []

        for nivel, etiq in [(NIVEL_CONFIANZA_VAR, '95%'),
                             (NIVEL_CONFIANZA_VAR_ESTRICTO, '99%')]:
            bt = backtesting_var(ret_port, nivel, ALPHA)
            resultados_bt[etiq] = bt
            _log(f"  {etiq}: {bt['n_viol']} violaciones | "
                 f"Kupiec={'OK' if bt['kupiec_ok'] else 'FALLA'}")
            if bt['nota_christ']:
                _log(f"  Christoffersen: {bt['nota_christ']}")
            bt_rows.append({'Nivel': etiq, **{k: v for k, v in bt.items()
                                               if k not in ['nota_kupiec', 'nota_christ']}})

        res_bt = pd.DataFrame(bt_rows)

        # Beta del portafolio (ponderado por pesos tangente)
        if res_capm is not None and not res_capm.empty:
            betas_capm = res_capm.set_index('Ticker')['Beta']
            beta_port  = float((pesos_tang * betas_capm).sum())
        else:
            beta_port = 1.0
            _log("AVISO: beta del CAPM no disponible — usando beta=1.0")

        # Métricas avanzadas
        _log("BLOQUE 3 — Métricas avanzadas")
        metricas_3c = calcular_metricas_avanzadas(
            ret_port, ret_bench, beta_port, rf_mes, rf_anual, factor
        )
        _log(f"  Sharpe={metricas_3c['sharpe']} | Sortino={metricas_3c['sortino']} | "
             f"MaxDD={metricas_3c['max_drawdown']} | IR={metricas_3c['ir']}")

        # Estrés
        res_estres, lg = analisis_estres(retornos, pesos_tang, ESCENARIOS_ESTRES, factor)
        _log(lg)

        # Gráfico
        fig_riesgo = graficar_riesgo_plotly(ret_port, ret_bench,
                                             resultados_var, res_estres)

        # Asistente
        narrativa = asistente_fase3c(resultados_var, resultados_bt,
                                      metricas_3c, res_estres, nivel_asistente)

        _log("✅ FASE 3c COMPLETADA")

        return {
            'res_var':      res_var,
            'resultados_var': resultados_var,
            'metricas_3c':  metricas_3c,
            'res_bt':       res_bt,
            'resultados_bt': resultados_bt,
            'res_estres':   res_estres,
            'fig_riesgo':   fig_riesgo,
            'narrativa_3c': narrativa,
            'logs':         logs_totales,
            'error':        None,
        }

    except Exception as e:
        import traceback
        _log(f"ERROR FATAL en pipeline_riesgo: {e}")
        _log(traceback.format_exc())
        return {'error': str(e), 'logs': logs_totales}
