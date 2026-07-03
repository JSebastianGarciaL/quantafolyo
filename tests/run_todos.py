"""
Corre todos los tests LOCALES (sin conexión a internet) en secuencia.

El test de conversión de divisas (test_conversion_divisas.py) NO se incluye
aquí porque necesita red real hacia Yahoo Finance — córrelo aparte:

    python tests/test_conversion_divisas.py

Cómo correr este runner
------------------------
    cd portfoliolab_streamlit
    python tests/run_todos.py
"""

import subprocess
import sys
import os

CARPETA = os.path.dirname(os.path.abspath(__file__))

TESTS_LOCALES = [
    "test_var_cvar.py",
    "test_retornos_acumulados.py",
    "test_vif_constante.py",
]


def main():
    resultados = {}

    for nombre in TESTS_LOCALES:
        ruta = os.path.join(CARPETA, nombre)
        print("\n" + "#" * 78)
        print(f"# Corriendo: {nombre}")
        print("#" * 78)
        proceso = subprocess.run([sys.executable, ruta])
        resultados[nombre] = (proceso.returncode == 0)

    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    for nombre, ok in resultados.items():
        print(f"  {'✅ PASÓ' if ok else '❌ FALLÓ':<10} {nombre}")

    print()
    print("Recuerda correr también, por separado (necesita internet):")
    print("  python tests/test_conversion_divisas.py")
    print("=" * 78)

    if not all(resultados.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
