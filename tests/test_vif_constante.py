"""
Verificación de VIF con columna constante (M11)
=================================================

M11 fue una corrección donde `calcular_vif()` no agregaba una columna
constante antes de calcular el Variance Inflation Factor, lo cual produce
el VIF "no centrado" (uncentered) en vez del VIF estándar — un error común
y bien documentado en la literatura de econometría (ver documentación de
statsmodels: `variance_inflation_factor` asume que X ya incluye una
constante).

Este script construye regresores sintéticos con colinealidad CONOCIDA
(una variable que es casi combinación lineal de otras dos) y verifica:

  1. La variable colineal tiene un VIF sensiblemente más alto que las
     variables independientes — la propiedad que el VIF existe para medir.
  2. El VIF calculado CON constante (correcto) difiere del VIF calculado
     SIN constante (el bug de M11) — confirmando que la corrección
     realmente cambia el resultado y no es un no-op.

Cómo correrlo
-------------
    cd portfoliolab_streamlit
    python tests/test_vif_constante.py

No requiere conexión a internet — todo es sintético y determinístico.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from modulos.factores import calcular_vif


def linea(char="-", n=78):
    print(char * n)


def generar_regresores_colineales(n=500, semilla=99):
    rng = np.random.default_rng(semilla)
    # Media NO-cero a propósito — como factores de mercado reales
    # (ej. Mkt-RF, SMB, HML tienen medias distintas de cero). Con media
    # cero la diferencia centrado/no-centrado se vuelve casi invisible;
    # con media distinta de cero, el efecto de M11 se ve claramente.
    x1 = rng.normal(0.8, 1, n)
    x2 = rng.normal(-0.5, 1, n)
    # x3 es casi x1 + x2 + ruido pequeño -> fuerte colinealidad
    x3 = x1 + x2 + rng.normal(0, 0.05, n)
    return pd.DataFrame({"factor_1": x1, "factor_2": x2, "factor_3_colineal": x3})


def vif_sin_constante(X: pd.DataFrame) -> pd.Series:
    """Reproduce el bug de M11 a propósito: VIF sin agregar constante."""
    X_vals = X.values
    vifs = [variance_inflation_factor(X_vals, i) for i in range(X_vals.shape[1])]
    return pd.Series(vifs, index=X.columns)


def main():
    print()
    linea("=")
    print("VERIFICACIÓN — VIF con columna constante (M11)")
    linea("=")

    X = generar_regresores_colineales()
    print(f"Regresores sintéticos: {list(X.columns)} ({len(X)} observaciones)")
    print("'factor_3_colineal' se construyó como factor_1 + factor_2 + ruido")
    print("pequeño — debería mostrar un VIF muy alto.")
    print()

    vif_correcto = calcular_vif(X)          # con constante (código real del proyecto)
    vif_bug      = vif_sin_constante(X)     # sin constante (el bug de M11)

    print(f"{'Factor':<20}{'VIF (con constante)':>22}{'VIF (sin constante — bug M11)':>32}")
    linea()
    for col in X.columns:
        print(f"{col:<20}{vif_correcto[col]:>22.3f}{vif_bug[col]:>32.3f}")
    print()

    fallas = []

    # Chequeo 1: la variable colineal debe destacar claramente
    umbral_alerta = 10  # umbral clásico de la literatura (VIF > 10 = colinealidad severa)
    if vif_correcto["factor_3_colineal"] <= umbral_alerta:
        fallas.append(
            f"factor_3_colineal debería tener VIF > {umbral_alerta} "
            f"(dio {vif_correcto['factor_3_colineal']:.2f}) — el VIF no está "
            "detectando la colinealidad conocida."
        )
    else:
        print(f"✅ factor_3_colineal tiene VIF = {vif_correcto['factor_3_colineal']:.1f} "
              f"(> {umbral_alerta}), correctamente señalado como colineal.")

    # Chequeo 2: con y sin constante deben diferir (confirma que la
    # corrección de M11 realmente tiene efecto, no es un no-op)
    diferencias = (vif_correcto - vif_bug).abs()
    if diferencias.max() < 1e-6:
        fallas.append(
            "El VIF con y sin constante dio IDÉNTICO para todos los factores. "
            "Se esperaba una diferencia — revisa si calcular_vif() todavía "
            "usa sm.add_constant() internamente."
        )
    else:
        print(f"✅ El VIF con constante difiere del VIF sin constante "
              f"(diferencia máxima: {diferencias.max():.3f}), confirmando "
              "que la corrección de M11 está activa.")

    print()
    linea("=")
    if fallas:
        print("❌ FALLAS ENCONTRADAS:")
        for f in fallas:
            print(f"   - {f}")
        sys.exit(1)
    else:
        print("✅ calcular_vif() detecta colinealidad conocida y usa la")
        print("   corrección de M11 (columna constante) correctamente.")
    linea("=")


if __name__ == "__main__":
    main()
