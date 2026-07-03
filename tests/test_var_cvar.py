"""
Verificación matemática de VaR y CVaR (relacionado con C1)
=============================================================

C1 fue un bug crítico donde el CVaR paramétrico tenía el signo del término
sigma invertido (`-(mu + sigma*densidad/alpha)` en vez de
`-(mu - sigma*densidad/alpha)`). Este script no depende de datos de mercado
reales — genera retornos sintéticos de una distribución NORMAL conocida y
verifica dos propiedades que son verdad SIEMPRE, sin importar los datos:

  1. CVaR ≥ VaR, en los tres métodos (histórico, paramétrico, Monte Carlo).
     Esto es una propiedad matemática de la Expected Shortfall: el promedio
     de las pérdidas más allá del VaR nunca puede ser menor que el VaR
     mismo. Si esto falla, es señal casi segura de un error de signo.

  2. Los tres métodos (histórico / paramétrico / Monte Carlo) deben
     converger entre sí dentro de una tolerancia razonable, porque los
     datos sintéticos SÍ son normales por construcción — es el escenario
     ideal donde el método paramétrico (que asume normalidad) debe
     coincidir con los otros dos.

Cómo correrlo
-------------
    cd portfoliolab_streamlit
    python tests/test_var_cvar.py

No requiere conexión a internet — todo es sintético y determinístico
(semilla fija).
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modulos.riesgo import calcular_var, calcular_cvar

TOLERANCIA_ENTRE_METODOS = 0.30   # 30% de diferencia relativa máxima aceptada
NIVELES_A_PROBAR         = [0.95, 0.99]


def linea(char="-", n=78):
    print(char * n)


def generar_retornos_normales(n=2000, mu=0.0005, sigma=0.02, semilla=123):
    rng = np.random.default_rng(semilla)
    return pd.Series(rng.normal(mu, sigma, n))


def main():
    print()
    linea("=")
    print("VERIFICACIÓN MATEMÁTICA — VaR y CVaR (relacionado con C1)")
    linea("=")

    ret = generar_retornos_normales()
    print(f"Muestra sintética: {len(ret)} observaciones ~ N(mu={ret.mean():.5f}, sigma={ret.std():.5f})")
    print()

    fallas = []

    for nivel in NIVELES_A_PROBAR:
        print(f"Nivel de confianza: {nivel*100:.0f}%")
        linea()

        var_res  = calcular_var(ret, nivel)
        cvar_res = calcular_cvar(ret, nivel)

        filas = [
            ("Histórico",    var_res["var_hist"],  cvar_res["cvar_hist"]),
            ("Paramétrico",  var_res["var_param"], cvar_res["cvar_param"]),
            ("Monte Carlo",  var_res["var_mc"],    cvar_res["cvar_mc"]),
        ]

        print(f"  {'Método':<14}{'VaR':>10}{'CVaR':>10}{'CVaR>=VaR':>14}")
        for nombre, v, cv in filas:
            ok_orden = cv >= v - 1e-9   # margen numérico
            estado   = "✅ sí" if ok_orden else "❌ NO — revisar signo"
            print(f"  {nombre:<14}{v:>10.5f}{cv:>10.5f}{estado:>18}")
            if not ok_orden:
                fallas.append(f"{nombre} @ {nivel*100:.0f}%: CVaR ({cv:.5f}) < VaR ({v:.5f})")

        # Coherencia entre métodos (todos deberían rondar el mismo valor
        # porque los datos SÍ son normales por construcción)
        vars_  = [v for _, v, _ in filas]
        cvars_ = [cv for _, _, cv in filas]

        def dispersión_relativa(vals):
            vals = np.array(vals)
            return (vals.max() - vals.min()) / abs(vals.mean()) if vals.mean() != 0 else 0

        disp_var  = dispersión_relativa(vars_)
        disp_cvar = dispersión_relativa(cvars_)

        print()
        print(f"  Dispersión relativa entre métodos — VaR: {disp_var:.1%}  |  CVaR: {disp_cvar:.1%}")
        if disp_var > TOLERANCIA_ENTRE_METODOS:
            fallas.append(f"VaR @ {nivel*100:.0f}%: métodos divergen más de {TOLERANCIA_ENTRE_METODOS:.0%}")
        if disp_cvar > TOLERANCIA_ENTRE_METODOS:
            fallas.append(f"CVaR @ {nivel*100:.0f}%: métodos divergen más de {TOLERANCIA_ENTRE_METODOS:.0%}")

        print()

    linea("=")
    if fallas:
        print("❌ FALLAS ENCONTRADAS:")
        for f in fallas:
            print(f"   - {f}")
        print()
        print("Si ves 'CVaR < VaR', es casi con certeza un error de signo como C1.")
        sys.exit(1)
    else:
        print("✅ Todo consistente: CVaR ≥ VaR en los tres métodos y niveles,")
        print("   y los tres métodos convergen entre sí como se espera con")
        print("   datos normales sintéticos.")
    linea("=")


if __name__ == "__main__":
    main()
