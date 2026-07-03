# Tests de verificación — QuantαfolyΩ

Scripts de verificación matemática y funcional del proyecto. **No son tests
unitarios exhaustivos** (no hay cobertura de cada función línea por línea);
son verificaciones dirigidas a las correcciones más críticas de las
auditorías previas — las que, si se revirtieran por accidente en un cambio
futuro, producirían resultados numéricos silenciosamente incorrectos sin
que la app truene ni avise.

## Filosofía

Cada test verifica una **propiedad matemática que debe ser verdad siempre**,
no un valor específico hardcodeado. Por ejemplo: "CVaR ≥ VaR" es verdad para
cualquier distribución razonable, así que si un test la viola, es señal casi
segura de un bug de signo — no depende de que los datos de mercado sean
exactamente los mismos que cuando se escribió el test.

Todos los tests, salvo el de divisas, usan **datos sintéticos generados con
semilla fija** (deterministas, reproducibles, sin depender de que Yahoo
Finance esté disponible o de que el mercado esté abierto).

## Cómo correrlos

```bash
cd portfoliolab_streamlit

# Todos los tests locales (no requieren internet) en un solo comando:
python tests/run_todos.py

# El de divisas aparte, porque SÍ requiere internet real:
python tests/test_conversion_divisas.py
```

## Qué verifica cada uno

| Script | Corrección que protege | Requiere internet |
|---|---|---|
| `test_conversion_divisas.py` | **C4** — dirección de conversión de divisas (activos no-USD) | ✅ Sí |
| `test_var_cvar.py` | **C1** — signo del CVaR paramétrico; además valida que CVaR ≥ VaR siempre y que los 3 métodos (histórico/paramétrico/Monte Carlo) convergen entre sí | ❌ No |
| `test_retornos_acumulados.py` | **C3** — reconstrucción de índices de precio desde retornos logarítmicos (`exp(cumsum(r))`, no `(1+r).cumprod()`) | ❌ No |
| `test_vif_constante.py` | **M11** — VIF calculado con columna constante (VIF centrado vs. no-centrado) | ❌ No |

## Si un test falla

Cada script imprime qué específicamente falló y por qué, y termina con
código de salida distinto de 0 (útil si algún día se conecta a CI/CD).
El mensaje de cada test explica a qué corrección histórica corresponde el
fallo — revisa esa sección en `Contexto_Desarrollo_PostAuditoria.md` /
la auditoría de cierre para recordar el detalle exacto del bug original.

## Agregar un test nuevo

Si en el futuro se corrige otro bug matemático, vale la pena escribir un
test del mismo estilo: datos sintéticos con una propiedad conocida,
assert/print claro de ✅/❌, y agregarlo a `TESTS_LOCALES` en `run_todos.py`
si no requiere red.
