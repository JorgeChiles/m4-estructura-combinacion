#!/usr/bin/env python3
"""
Metricas OFICIALES de la competencia M4: sMAPE, MASE y OWA.

Por que hace falta
------------------
El script 04 guarda un sMAPE en fraccion (0.0398) y usa estacionalidades que
no son las de M4 (Weekly 52, Daily 7). M4 mide distinto y ese es el numero
comparable con la literatura:

  sMAPE = (200/h) * sum |A - F| / (|A| + |F|)              -> en PORCENTAJE

  MASE  = mean|A - F| / [ (1/(n-m)) * sum |Y_t - Y_{t-m}| ]
          el denominador es el error del naive estacional DENTRO de la
          muestra de entrenamiento: escala la serie y hace comparables
          frecuencias distintas.

  OWA   = 0.5 * ( sMAPE_modelo/sMAPE_Naive2 + MASE_modelo/MASE_Naive2 )

OWA es la metrica con la que se decidio el ranking de M4. Por construccion
Naive2 vale exactamente 1.000: por debajo de 1 se le gana al benchmark.

Naive2
------
No es el naive comun. Es: probar estacionalidad, desestacionalizar si da
significativa, aplicar naive sobre la serie ajustada y volver a aplicar los
indices estacionales al pronostico. Es el denominador de OWA, asi que hay
que reproducirlo con cuidado o todos los OWA quedan corridos.

Estacionalidad segun M4 (la del archivo m4_info.csv, no la "natural"):
    Yearly 1 | Quarterly 4 | Monthly 12 | Weekly 1 | Daily 1 | Hourly 24
M4 trata Weekly y Daily como NO estacionales. El script 04 usaba 52 y 7:
por eso su snaive no es comparable con la competencia.

Salidas
-------
    results/m4_metrics_por_serie.xlsx   una fila por serie
    results/m4_owa_resumen.xlsx         sMAPE / MASE / OWA por frecuencia

Uso:
    python3 src/06_m4_metrics.py                # naive, snaive, naive2
    python3 src/06_m4_metrics.py --con-arima    # agrega ARIMA y SARIMA (lento)
    python3 src/06_m4_metrics.py --workers 8
"""
import os
import sys
import time
import warnings

# Limitar hilos ANTES de importar numpy: cada worker usa 1 core, si no los
# procesos se pelean entre si y el Pool rinde peor que el secuencial.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# El filterwarnings() de mas abajo solo silencia al proceso padre: los
# workers del Pool no lo heredan y statsmodels inunda el log con
# ConvergenceWarning (una por serie que no converge, decenas de miles).
# La variable de entorno SI la heredan los hijos.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rutas import DATA_DIR, salida  # noqa: E402


# =========================================================
# CONFIGURACION
# =========================================================

# h y m exactamente como los define M4 (m4_info.csv)
M4_CONFIG = {
    "Yearly":    dict(train="Yearly-train.csv",    test="Yearly-test.csv",    h=6,  m=1),
    "Quarterly": dict(train="Quarterly-train.csv", test="Quarterly-test.csv", h=8,  m=4),
    "Monthly":   dict(train="Monthly-train.csv",   test="Monthly-test.csv",   h=18, m=12),
    "Weekly":    dict(train="Weekly-train.csv",    test="Weekly-test.csv",    h=13, m=1),
    "Daily":     dict(train="Daily-train.csv",     test="Daily-test.csv",     h=14, m=1),
    "Hourly":    dict(train="Hourly-train.csv",    test="Hourly-test.csv",    h=48, m=24),
}

# Tambien por variable de entorno, para que el notebook pueda pedirlo sin
# argumentos de linea de comandos. Sin esto, correr el notebook regeneraria
# m4_metrics_por_serie.xlsx SIN las columnas de ARIMA y SARIMA, pisando el
# resultado de la corrida larga.
CON_ARIMA = "--con-arima" in sys.argv or bool(os.environ.get("CON_ARIMA"))

# El codigo oficial de M4 tiene una particularidad en el test de
# estacionalidad: el primer termino de la suma NO va al cuadrado.
#   s = acf(ts, 1)                        <- sin cuadrado
#   for i in range(2, ppy): s += acf(ts, i) ** 2
# Es distinto de la formula de libro (Box-Jenkins), que eleva todos.
# Se deja True para REPRODUCIR M4; en False usa la formula estandar.
M4_EXACTO = True


# =========================================================
# METRICAS
# =========================================================

def smape(real, pred):
    """sMAPE de M4, en porcentaje (0-200)."""
    real = np.asarray(real, float)
    pred = np.asarray(pred, float)
    den = np.abs(real) + np.abs(pred)
    mask = den != 0
    if mask.sum() == 0:
        return np.nan
    return float(200 * np.mean(np.abs(real[mask] - pred[mask]) / den[mask]))


def mase(train, real, pred, m):
    """MASE: error absoluto medio escalado por el naive estacional in-sample."""
    train = np.asarray(train, float)
    if len(train) <= m:
        return np.nan
    escala = np.mean(np.abs(train[m:] - train[:-m]))
    if not np.isfinite(escala) or escala == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(real, float) - np.asarray(pred, float))) / escala)


# =========================================================
# NAIVE2: TEST DE ESTACIONALIDAD Y DESESTACIONALIZACION
# =========================================================

def acf_lag(x, lag):
    """Autocorrelacion muestral (estimador sesgado, el que usa M4)."""
    n = len(x)
    if lag >= n:
        return 0.0
    mu = x.mean()
    den = np.sum((x - mu) ** 2)
    if den == 0:
        return 0.0
    return float(np.sum((x[lag:] - mu) * (x[:n - lag] - mu)) / den)


def test_estacionalidad(x, m):
    """Test de M4: |acf(m)| por encima de la banda del 90% => estacional."""
    if m <= 1 or len(x) < 3 * m:
        return False
    s = acf_lag(x, 1) if M4_EXACTO else acf_lag(x, 1) ** 2
    for i in range(2, m):
        s += acf_lag(x, i) ** 2
    limite = 1.645 * np.sqrt((1 + 2 * s) / len(x))
    return bool(abs(acf_lag(x, m)) > limite)


def indices_estacionales(x, m):
    """Indices estacionales multiplicativos por descomposicion clasica.
    Devuelve m unos si la serie no pasa el test o no admite el modelo
    multiplicativo (algun valor <= 0)."""
    if m <= 1 or not test_estacionalidad(x, m) or np.any(x <= 0):
        return np.ones(m)
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        dec = seasonal_decompose(x, model="multiplicative", period=m,
                                 extrapolate_trend="freq")
        si = np.asarray(dec.seasonal[:m], float)
        if not np.all(np.isfinite(si)) or np.any(si <= 0):
            return np.ones(m)
        return si * m / si.sum()          # normalizados: suman m
    except Exception:
        return np.ones(m)


def forecast_naive2(train, h, m):
    """Naive sobre la serie desestacionalizada, re-estacionalizado."""
    train = np.asarray(train, float)
    si = indices_estacionales(train, m)
    if np.allclose(si, 1.0):
        return np.repeat(train[-1], h)
    ciclo = np.tile(si, int(np.ceil(len(train) / m)) + 1)[:len(train)]
    ajustada = train / ciclo
    fut = np.tile(si, int(np.ceil((len(train) + h) / m)) + 1)[len(train):len(train) + h]
    return np.repeat(ajustada[-1], h) * fut


# =========================================================
# PRONOSTICOS DE REFERENCIA
# =========================================================

def forecast_naive(train, h):
    return np.repeat(np.asarray(train, float)[-1], h)


def forecast_snaive(train, h, m):
    train = np.asarray(train, float)
    if m <= 1 or len(train) < m:
        return forecast_naive(train, h)
    return np.tile(train[-m:], int(np.ceil(h / m)))[:h]


def forecast_arima(train, h):
    from statsmodels.tsa.arima.model import ARIMA
    try:
        return np.asarray(ARIMA(np.asarray(train, float),
                                order=(1, 1, 1)).fit().forecast(steps=h), float)
    except Exception:
        return np.full(h, np.nan)


def forecast_sarima(train, h, m):
    from statsmodels.tsa.arima.model import ARIMA
    if m <= 1 or len(train) < 2 * m:
        return np.full(h, np.nan)
    try:
        return np.asarray(ARIMA(np.asarray(train, float), order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, m)).fit().forecast(steps=h), float)
    except Exception:
        return np.full(h, np.nan)


# =========================================================
# EVALUACION DE UNA SERIE
# =========================================================

def evaluar(args):
    """Una serie -> una fila con sMAPE y MASE de cada metodo."""
    sid, train, test, categoria, h, m, con_arima = args
    real = np.asarray(test, float)[:h]

    metodos = {
        "naive":  forecast_naive(train, h),
        "snaive": forecast_snaive(train, h, m),
        "naive2": forecast_naive2(train, h, m),
    }
    if con_arima:
        metodos["arima"] = forecast_arima(train, h)
        metodos["sarima"] = forecast_sarima(train, h, m)

    fila = {"serie": sid, "category": categoria, "h": h, "m": m,
            "n_train": len(train),
            "estacional": test_estacionalidad(np.asarray(train, float), m)}

    for nombre, pred in metodos.items():
        pred = np.asarray(pred, float)[:len(real)]
        ok = len(pred) == len(real) and np.all(np.isfinite(pred))
        fila[f"smape_{nombre}"] = smape(real, pred) if ok else np.nan
        fila[f"mase_{nombre}"] = mase(train, real, pred, m) if ok else np.nan
    return fila


# =========================================================
# CARGA DE DATOS
# =========================================================

def load_m4(path):
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]
    id_col = df.columns[0]
    return {str(r[id_col]): pd.to_numeric(r.iloc[1:], errors="coerce")
            .dropna().astype(float).values
            for _, r in df.iterrows()}


# =========================================================
# AGREGACION: OWA
# =========================================================

def resumir(df):
    """OWA se calcula sobre los PROMEDIOS, no serie por serie: es el
    cociente de las medias, como en la competencia.

    Se agregan medianas y un conteo de divergencias porque el promedio de
    MASE es fragil: sMAPE esta ACOTADO en 200, MASE NO tiene cota. Un ajuste
    que diverge queda topeado en el sMAPE pero se dispara sin limite en el
    MASE, y con eso solo arrastra el promedio de las 100.000 series. Pasa
    con SARIMA de orden fijo: en Hourly UNA serie (H366, MASE 127.224)
    explica todo el promedio, mientras su mediana es la mejor de todos los
    metodos. Las columnas *_mediana NO son la metrica oficial de M4; estan
    para poder ver cuando la media esta contaminada."""
    metodos = sorted({c.split("_", 1)[1] for c in df.columns if c.startswith("smape_")})
    filas = []
    for cat, g in list(df.groupby("category")) + [("TODAS", df)]:
        base_s = g["smape_naive2"].mean()
        base_m = g["mase_naive2"].mean()
        med_s = g["smape_naive2"].median()
        med_m = g["mase_naive2"].median()
        for met in metodos:
            col_s, col_m = g[f"smape_{met}"], g[f"mase_{met}"]
            s, ma = col_s.mean(), col_m.mean()          # .mean() ignora NaN
            filas.append({
                "category": cat, "metodo": met, "n": len(g),
                # cuantas series pudo pronosticar el metodo: ARIMA no
                # converge en todas y SARIMA no aplica con m=1. Sin este
                # conteo, un promedio sobre pocas series se lee igual que
                # uno sobre todas.
                "n_validas": int(col_s.notna().sum()),
                "cobertura_%": round(100 * col_s.notna().mean(), 1),
                "sMAPE": s, "MASE": ma,
                "OWA": 0.5 * (s / base_s + ma / base_m)
                if base_s and base_m else np.nan,
                "sMAPE_mediana": col_s.median(), "MASE_mediana": col_m.median(),
                "OWA_mediana": 0.5 * (col_s.median() / med_s + col_m.median() / med_m)
                if med_s and med_m else np.nan,
                # MASE > 100 marca casos extremos. Puede ser un ajuste que
                # DIVERGIO (SARIMA de orden fijo lo hace) o una serie con
                # denominador casi nulo, que infla el MASE de CUALQUIER
                # metodo. El naive no puede divergir: si aparece marcado, es
                # lo segundo. Se distinguen mirando si el naive tambien cae.
                "mase_extremos": int((col_m > 100).sum()),
            })
    return pd.DataFrame(filas)


# =========================================================
# VALIDACION CONTRA LOS VALORES PUBLICADOS
# =========================================================

# Naive2 segun Makridakis, Spiliotis & Assimakopoulos (2020), "The M4
# Competition: 100,000 time series and 61 forecasting methods",
# International Journal of Forecasting 36(1). Es el benchmark oficial:
# si nuestra implementacion reproduce estos numeros, el denominador de OWA
# es el correcto y los OWA que salgan son comparables con la literatura.
NAIVE2_PUBLICADO = {
    "Yearly":    (16.342, 3.974),
    "Quarterly": (11.012, 1.371),
    "Monthly":   (14.427, 1.063),
    "Weekly":    (9.161,  2.777),
    "Daily":     (3.045,  3.278),
    "Hourly":    (18.383, 2.395),
    "TODAS":     (13.564, 1.912),
}


def validar(res):
    """Compara nuestro Naive2 contra el publicado. Yearly, Weekly y Daily
    tienen m=1: ahi Naive2 == Naive y el numero tiene que dar EXACTO. En las
    estacionales puede haber decimas de diferencia porque la descomposicion
    de statsmodels no es byte a byte la del codigo original de M4."""
    n2 = res[res.metodo == "naive2"].set_index("category")
    filas = []
    for cat, (s_pub, m_pub) in NAIVE2_PUBLICADO.items():
        if cat not in n2.index:
            continue
        s, m = n2.loc[cat, "sMAPE"], n2.loc[cat, "MASE"]
        filas.append({"category": cat,
                      "sMAPE_calculado": s, "sMAPE_M4": s_pub, "dif_sMAPE": s - s_pub,
                      "MASE_calculado": m, "MASE_M4": m_pub, "dif_MASE": m - m_pub})
    return pd.DataFrame(filas)


# =========================================================
# PRINCIPAL
# =========================================================

def informar(df, t0=None):
    """Escribe el resumen y la validacion, e imprime las tablas."""
    res = resumir(df)
    res.to_excel(salida("m4_owa_resumen.xlsx"), index=False)

    cab = f" ({(time.time()-t0)/60:.1f} min, {len(df)} series)" if t0 else f" ({len(df)} series)"
    print(f"\n═══ RESUMEN{cab} ═══")
    print(res.pivot(index="category", columns="metodo",
                    values=["sMAPE", "MASE", "OWA"]).round(3).to_string())
    print("\nNaive2 tiene OWA = 1.000 por definicion: es el denominador.")

    div = res[res.mase_extremos > 0]
    if len(div):
        print("\n⚠ MEDIAS CONTAMINADAS POR CASOS EXTREMOS (MASE > 100)")
        print(div[["category", "metodo", "mase_extremos", "MASE",
                   "MASE_mediana", "OWA", "OWA_mediana"]].round(3).to_string(index=False))
        print("  sMAPE topea en 200 y MASE no tiene cota: un caso extremo queda")
        print("  disimulado en el sMAPE y arrastra el promedio del MASE.")
        print("  Si el naive tambien aparece, es denominador chico (la serie);")
        print("  si solo aparece SARIMA, es el ajuste que diverge (el modelo).")
        print("  OWA_mediana es diagnostico: NO es comparable con los OWA")
        print("  publicados de M4, que se calculan sobre medias.")

    val = validar(res)
    val.to_excel(salida("m4_validacion_naive2.xlsx"), index=False)
    print("\n═══ VALIDACION: nuestro Naive2 vs el publicado por M4 ═══")
    print(val.round(3).to_string(index=False))
    print(f"\ndesvio maximo: sMAPE {val.dif_sMAPE.abs().max():.3f}  "
          f"MASE {val.dif_MASE.abs().max():.3f}")
    print("\nresults/m4_owa_resumen.xlsx\nresults/m4_validacion_naive2.xlsx")


def main():
    import multiprocessing as mp

    # Recalcula el resumen desde el archivo por serie ya calculado, sin
    # rehacer los pronosticos (que con --con-arima son ~36 min).
    if "--solo-resumen" in sys.argv:
        ruta = salida("m4_metrics_por_serie.xlsx")
        print(f"recalculando el resumen desde {ruta.name}")
        informar(pd.read_excel(ruta))
        return 0
    if "--workers" in sys.argv:
        n_work = int(sys.argv[sys.argv.index("--workers") + 1])
    else:
        n_work = max(1, (os.cpu_count() or 4) - 2)

    print(f"metricas oficiales M4  |  workers: {n_work}  |  "
          f"ARIMA/SARIMA: {'si' if CON_ARIMA else 'no'}\n", flush=True)

    t0 = time.time()
    filas = []
    with mp.Pool(n_work) as pool:
        for cat, cfg in M4_CONFIG.items():
            tr = load_m4(DATA_DIR / cfg["train"])
            te = load_m4(DATA_DIR / cfg["test"])
            h, m = cfg["h"], cfg["m"]
            tareas = [(sid, tr[sid], te[sid][:h], cat, h, m, CON_ARIMA)
                      for sid in tr if sid in te and len(te[sid]) > 0]
            print(f"── {cat}: {len(tareas)} series (h={h}, m={m})", flush=True)
            t1 = time.time()
            paso = 250 if CON_ARIMA else 5000
            for i, fila in enumerate(pool.imap_unordered(evaluar, tareas, chunksize=25), 1):
                filas.append(fila)
                if i % paso == 0 or i == len(tareas):
                    # ojo: no llamar 'tr' a esto, que pisa el dict de series
                    transcurrido = time.time() - t1
                    eta = transcurrido / i * (len(tareas) - i) / 60
                    print(f"   {i}/{len(tareas)}  {transcurrido/60:.1f} min  "
                          f"ETA {eta:.0f} min", flush=True)
            # Con ARIMA esto son horas: se vuelca lo hecho al terminar cada
            # frecuencia para no perder todo si el proceso muere.
            pd.DataFrame(filas).to_csv(salida("m4_metrics_parcial.csv"), index=False)

    df = pd.DataFrame(filas).sort_values(["category", "serie"]).reset_index(drop=True)
    df.to_excel(salida("m4_metrics_por_serie.xlsx"), index=False)

    informar(df, t0)
    print("results/m4_metrics_por_serie.xlsx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
