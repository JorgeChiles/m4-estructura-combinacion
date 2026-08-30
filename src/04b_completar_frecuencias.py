#!/usr/bin/env python3
"""
Completa forecasting_errors_partial.xlsx con las tres frecuencias que faltan.

El archivo actual tiene 95.000 series (Yearly, Quarterly, Monthly, poblacion
COMPLETA de esas frecuencias). Faltan Weekly (359), Daily (4.227) y Hourly
(414) = 5.000, y por eso se llama "partial".

Usa las MISMAS funciones que 04_forecasting_errors.py para que los errores
sean comparables: naive, seasonal naive, ARIMA(1,1,1) y SARIMA con la
estacionalidad de cada frecuencia, evaluados por SMAPE.

Paralelizado: SARIMA con m=52 sobre series semanales de 2000+ puntos toma
~150 s por serie, lo que serian ~15 h en secuencial. Con un Pool baja a ~2 h.

Salida: forecasting_errors_completo.xlsx  (no pisa el parcial)

Uso:  python3 04b_completar_frecuencias.py [n_workers]
"""
import os
import sys
import time
import warnings
from pathlib import Path

# Limitar hilos ANTES de importar numpy: cada worker debe usar 1 core, si no
# los procesos se pelean entre si y el Pool rinde peor que el secuencial.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

warnings.filterwarnings("ignore")

from rutas import BASE_DIR, DATA_DIR, entrada, salida  # noqa: E402

# =========================================================
# RUTAS
# =========================================================

PARCIAL = entrada("forecasting_errors_partial.xlsx")
SALIDA = salida("forecasting_errors_completo.xlsx")

# =========================================================
# CONFIGURACION
# =========================================================

FALTANTES = {
    "Weekly": dict(train="Weekly-train.csv", test="Weekly-test.csv", h=13, m=52),
    "Daily":  dict(train="Daily-train.csv",  test="Daily-test.csv",  h=14, m=7),
    "Hourly": dict(train="Hourly-train.csv", test="Hourly-test.csv", h=48, m=24),
}


# =========================================================
# FUNCIONES DE PRONOSTICO
# =========================================================

def load_m4(path):
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]
    id_col = df.columns[0]
    return {str(r[id_col]): (pd.to_numeric(r.iloc[1:], errors="coerce")
                             .dropna().astype(float).values)
            for _, r in df.iterrows()}


def safe_smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def forecast_naive(train, h):
    train = np.asarray(train, float)
    return np.full(h, np.nan) if len(train) == 0 else np.repeat(train[-1], h)


def forecast_seasonal_naive(train, h, m):
    train = np.asarray(train, float)
    if m is None or len(train) < m:
        return np.full(h, np.nan)
    return np.tile(train[-m:], int(np.ceil(h / m)))[:h]


def forecast_arima(train, h):
    try:
        return np.asarray(SM_ARIMA(np.asarray(train, float),
                                   order=(1, 1, 1)).fit().forecast(steps=h), float)
    except Exception:
        return np.full(h, np.nan)


def forecast_sarima(train, h, m):
    if m is None or len(train) < 2 * m:
        return np.full(h, np.nan)
    try:
        return np.asarray(SM_ARIMA(np.asarray(train, float), order=(1, 1, 1),
                                   seasonal_order=(1, 1, 1, m)).fit().forecast(steps=h), float)
    except Exception:
        return np.full(h, np.nan)


# =========================================================
# EVALUACION DE UNA SERIE
# =========================================================

def evaluar(args):
    """Una serie -> una fila. Definida a nivel de modulo para que sea picklable."""
    sid, train, test, nombre, h, m = args
    fila = {"serie": sid, "category": nombre}
    for etq, pred, aplica in (
        ("naive",  forecast_naive(train, h), True),
        ("snaive", forecast_seasonal_naive(train, h, m), m is not None),
        ("arima",  forecast_arima(train, h), True),
        ("sarima", forecast_sarima(train, h, m), m is not None),
    ):
        if not aplica:
            fila[f"error_{etq}_smape"] = np.nan
            fila[f"status_{etq}"] = "no_aplica"
            continue
        p = np.asarray(pred, float)[:len(test)]
        ok = len(p) == len(test) and np.all(np.isfinite(p))
        fila[f"error_{etq}_smape"] = safe_smape(test, p) if ok else np.nan
        fila[f"status_{etq}"] = "ok" if ok else "error"
    return fila


# =========================================================
# PROCESAMIENTO POR FRECUENCIA
# =========================================================

def procesar(nombre, cfg, pool):
    tr = load_m4(DATA_DIR / cfg["train"])
    te = load_m4(DATA_DIR / cfg["test"])
    h, m = cfg["h"], cfg["m"]
    tareas = [(sid, tr[sid], te[sid][:h], nombre, h, m)
              for sid in tr if sid in te and len(te[sid]) > 0]
    print(f"── {nombre}: {len(tareas)} series (h={h}, m={m})", flush=True)
    filas = []
    t0 = time.time()
    for i, fila in enumerate(pool.imap_unordered(evaluar, tareas, chunksize=1), 1):
        filas.append(fila)
        if i % 50 == 0 or i == len(tareas):
            tr_ = time.time() - t0
            eta = tr_ / i * (len(tareas) - i) / 60
            print(f"   {i}/{len(tareas)}  {tr_/60:.1f} min  ETA {eta:.0f} min", flush=True)
    return pd.DataFrame(filas)


# =========================================================
# PRINCIPAL
# =========================================================

def main():
    import multiprocessing as mp
    # Ojo con sys.argv: en un kernel de Jupyter vale
    #   ['ipykernel_launcher.py', '-f', '/ruta/kernel.json']
    # asi que sys.argv[1] es '-f' y int() revienta. Se acepta solo si es un
    # numero; si no, se usa el valor por defecto.
    n_work = max(1, (os.cpu_count() or 4) - 2)
    if len(sys.argv) > 1 and str(sys.argv[1]).isdigit():
        n_work = int(sys.argv[1])

    print(f"parcial actual: {PARCIAL.name}")
    prev = pd.read_excel(PARCIAL)
    print(f"  {len(prev)} filas  |  {sorted(prev.category.unique())}")

    # El 04 escribe un checkpoint por categoria en el mismo archivo. Si esa
    # corrida llego hasta el final, el "parcial" ya trae las seis frecuencias
    # y volver a calcularlas las DUPLICARIA. Se procesa solo lo que falta.
    ya = set(prev.category.unique())
    pendientes = [n for n in ("Daily", "Hourly", "Weekly") if n not in ya]
    if not pendientes:
        print("\nel parcial ya tiene las seis frecuencias: no hay nada que calcular")
        prev.to_excel(SALIDA, index=False)
        print(f"{SALIDA.name}: {len(prev)} filas (copiado del parcial)")
        return 0
    print(f"faltan: {pendientes}")
    print(f"workers: {n_work}\n")

    t0 = time.time()
    nuevos = []
    with mp.Pool(n_work) as pool:
        # de la mas rapida a la mas lenta: da resultados utiles antes
        for nombre in pendientes:
            nuevos.append(procesar(nombre, FALTANTES[nombre], pool))

    df = pd.concat([prev] + nuevos, ignore_index=True)
    df.to_excel(SALIDA, index=False)

    print(f"\n{SALIDA.name}: {len(df)} filas  ({(time.time()-t0)/60:.0f} min)")
    print(df.groupby("category").agg(
        n=("serie", "count"),
        naive_ok=("status_naive", lambda s: (s == "ok").sum()),
        arima_ok=("status_arima", lambda s: (s == "ok").sum()),
        sarima_ok=("status_sarima", lambda s: (s == "ok").sum()),
    ).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
