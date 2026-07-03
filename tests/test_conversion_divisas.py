"""
Verificación en vivo de la conversión de divisas (C4)
======================================================

Este script corre el pipeline REAL de conversión de divisas del proyecto
(modulos/datos.py: validar_tickers → descargar_precios → detectar_divisas →
convertir_a_usd) contra activos reales no denominados en USD, y muestra
precio local vs. precio convertido para que puedas verificar a simple
vista que el orden de magnitud tiene sentido.

Cubre los DOS casos de la lógica de conversión:
  - Divisas cotizadas CONTRA USD (EUR, GBP, AUD, NZD) → par {DIVISA}USD=X
  - Resto de divisas (JPY, CHF, etc.) → par USD{DIVISA}=X, invertido

Cómo correrlo
-------------
    cd portfoliolab_streamlit
    python tests/test_conversion_divisas.py

Requiere conexión a internet (Yahoo Finance) y las dependencias del
proyecto ya instaladas (requirements.txt).

Qué mirar en el resultado
--------------------------
Para cada activo se imprime:
  - Precio en divisa local (el más reciente del rango descargado)
  - Tipo de cambio aplicado y su dirección (invertido o no)
  - Precio resultante en USD
  - Un chequeo automático de orden de magnitud (no reemplaza tu propio
    juicio: compara el precio USD contra una cotización real del mismo
    activo en Google Finance / Yahoo Finance / tu bróker el mismo día)

Si algo sale con un orden de magnitud absurdo (ej. una acción de ~80 CHF
que termina en ~8000 USD o en ~0.008 USD), es señal de que el par o la
dirección de inversión está mal para esa divisa.
"""

import sys
import os
from datetime import datetime, timedelta

# El script vive en tests/, un nivel debajo de la raíz del proyecto —
# hay que subir un nivel para encontrar el paquete modulos/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modulos.datos import (
    validar_tickers,
    descargar_precios,
    detectar_divisas,
    convertir_a_usd,
    DIVISA_BASE,
)

# --- Activos de prueba: cubren ambas ramas de la lógica de conversión ---
# EUR y AUD -> cotizadas CONTRA USD (rama sin inversión)
# JPY y CHF -> cotizadas CON USD como base (rama con inversión)
#
# NOTA: se evitan tickers .L (Londres) a propósito — muchas acciones del
# LSE cotizan en PENIQUES (GBX), no en libras (GBP), lo cual rompería la
# comparación de orden de magnitud por una razón AJENA a esta lógica de
# conversión. Si más adelante quieres probar GBP, usa un ticker que SÍ
# cotice en GBP reales, no en GBX.
ACTIVOS_PRUEBA = {
    "SAP.DE":  "EUR — Alemania (rama SIN inversión: EURUSD=X)",
    "BHP.AX":  "AUD — Australia (rama SIN inversión: AUDUSD=X)",
    "7203.T":  "JPY — Japón (rama CON inversión: USDJPY=X invertido)",
    "NESN.SW": "CHF — Suiza (rama CON inversión: USDCHF=X invertido)",
}

VENTANA_DIAS = 10
FRECUENCIA   = "1d"


def linea(char="-", n=78):
    print(char * n)


def main():
    fecha_fin    = datetime.today()
    fecha_inicio = fecha_fin - timedelta(days=VENTANA_DIAS)
    tickers      = list(ACTIVOS_PRUEBA.keys())

    print()
    linea("=")
    print("VERIFICACIÓN EN VIVO — CONVERSIÓN DE DIVISAS (C4)")
    linea("=")
    print(f"Ventana: {fecha_inicio.date()} → {fecha_fin.date()}  |  Frecuencia: {FRECUENCIA}")
    print()

    # --- Paso 1: validar tickers (obtiene la divisa real desde Yahoo) ---
    print("Paso 1/4 — Validando tickers y detectando divisas reales...")
    resultado_val, logs_val = validar_tickers(tickers)
    for l in logs_val:
        print(f"  {l}")

    if resultado_val["invalidos"]:
        print()
        print(f"⚠️  No se pudieron validar: {resultado_val['invalidos']}")
        print("    (puede ser el ticker, o un problema temporal de red — reintenta)")

    validos = list(resultado_val["validos"].keys())
    if not validos:
        print()
        print("❌ Ningún ticker de prueba se pudo validar. Revisa tu conexión a internet.")
        sys.exit(1)

    # --- Paso 2: descargar precios en divisa local ---
    print()
    print("Paso 2/4 — Descargando precios en divisa local...")
    precios_local, logs_desc = descargar_precios(validos, fecha_inicio, fecha_fin, FRECUENCIA)
    for l in logs_desc:
        print(f"  {l}")

    # --- Paso 3: detectar divisas (lógica real del proyecto) ---
    print()
    print("Paso 3/4 — Detectando divisas por activo...")
    divisas, logs_div = detectar_divisas(resultado_val)
    for l in logs_div:
        print(f"  {l}")

    # --- Paso 4: convertir a USD (la función que estamos verificando) ---
    print()
    print("Paso 4/4 — Convirtiendo a USD (convertir_a_usd — lógica C4)...")
    precios_usd, logs_conv = convertir_a_usd(precios_local, divisas, fecha_inicio, fecha_fin, FRECUENCIA)
    for l in logs_conv:
        print(f"  {l}")

    # --- Resultado comparativo ---
    print()
    linea("=")
    print("RESULTADO COMPARATIVO — precio local vs. precio convertido a USD")
    linea("=")
    print(f"{'Ticker':<10}{'Divisa':<8}{'Precio local':>16}{'Precio USD':>16}{'Ratio':>12}  Veredicto")
    linea()

    hubo_sospechoso = False

    for t in validos:
        if t not in precios_local.columns or t not in precios_usd.columns:
            print(f"{t:<10}  (sin datos suficientes para comparar)")
            continue

        serie_local = precios_local[t].dropna()
        serie_usd   = precios_usd[t].dropna()
        if serie_local.empty or serie_usd.empty:
            print(f"{t:<10}  (serie vacía tras limpieza)")
            continue

        p_local = serie_local.iloc[-1]
        p_usd   = serie_usd.iloc[-1]
        divisa  = divisas.get(t, "?")
        ratio   = p_usd / p_local if p_local else float("nan")

        # Heurística de sanidad: si ya estaba en USD, ratio debe ser 1.0.
        # Si se convirtió, un tipo de cambio real está casi siempre entre
        # 0.001 y 200 (cubre desde JPY/COP hasta GBP/EUR/KWD). Fuera de eso,
        # algo probablemente está mal (par equivocado o inversión al revés).
        if divisa == DIVISA_BASE:
            sospechoso = abs(ratio - 1.0) > 1e-6
        else:
            sospechoso = not (0.001 <= ratio <= 200)

        veredicto = "⚠️  REVISAR — orden de magnitud raro" if sospechoso else "✅ razonable"
        if sospechoso:
            hubo_sospechoso = True

        print(f"{t:<10}{divisa:<8}{p_local:>16,.2f}{p_usd:>16,.2f}{ratio:>12,.5f}  {veredicto}")

    linea()
    print()
    if hubo_sospechoso:
        print("⚠️  Al menos un activo dio un ratio fuera de rango esperado.")
        print("    Antes de hacer deploy, revisa manualmente ese caso: compara el")
        print("    precio USD mostrado contra una cotización real del mismo día")
        print("    (Google Finance, Yahoo Finance web, o tu bróker).")
    else:
        print("✅ Todos los activos de prueba dieron un orden de magnitud razonable.")
        print("   Como último paso, compara al menos UN valor contra una fuente")
        print("   externa (ej. Google Finance) para confirmar que no solo el orden")
        print("   de magnitud es correcto, sino el valor exacto.")
    print()
    print("Recuerda: este script prueba activos que SÍ tienen datos de tipo de")
    print("cambio en Yahoo Finance. Si en tu app real usas un ticker distinto,")
    print("verifica también ESE caso específico antes del deploy.")
    linea("=")


if __name__ == "__main__":
    main()
