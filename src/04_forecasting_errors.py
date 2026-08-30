# =========================================================
# 04_forecasting_errors.py
# Calcula errores de forecasting:
# Naive, Seasonal Naive, ARIMA(1,1,1), SARIMA(1,1,1)(1,1,1,s)
# =========================================================

import os
import warnings
from pathlib import Path

# Limitar los hilos de BLAS ANTES de importar numpy. Sin esto, SARIMAX con
# m=52 (Weekly) abre un hilo por core y los hilos se pelean por matrices
# chicas: MEDIDO, una serie semanal pasa de ~37 s a mas de 39 MINUTOS.
# Sobre las 294 series semanales que admiten SARIMA, es la diferencia entre
# 3 horas y ocho dias. Los scripts 01, 02 y 06 ya lo hacian; este no.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


# =========================================================
# RUTAS
# =========================================================

from rutas import BASE_DIR, DATA_DIR, entrada, salida  # noqa: E402

archivo_features = entrada("df_features_complexity.xlsx")


# =========================================================
# CONFIGURACIÓN
# =========================================================

RUN_ARIMA = True
RUN_SARIMA = True

# Series por categoría. None = todas (la corrida en serio, ~1 dia de maquina).
# Se puede acotar sin tocar el archivo:  MAX_SERIES=500 python3 src/04_...py
# El notebook usa esa variable para correr sobre muestra.
MAX_SERIES_PER_CATEGORY = (int(os.environ["MAX_SERIES"])
                           if os.environ.get("MAX_SERIES") else None)

# Una corrida de MUESTRA no puede pisar los resultados de la corrida completa:
# forecasting_errors_partial.xlsx es de donde salen el 04b y el 05, y
# reemplazarlo por 3.000 filas de muestra romperia toda la cadena aguas abajo.
# Por eso las salidas llevan sufijo cuando MAX_SERIES_PER_CATEGORY no es None.
SUF = "" if MAX_SERIES_PER_CATEGORY is None else f"_MUESTRA{MAX_SERIES_PER_CATEGORY}"
if SUF:
    print(f"⚠ CORRIDA DE MUESTRA: {MAX_SERIES_PER_CATEGORY} series por categoria.")
    print(f"  Las salidas llevan el sufijo {SUF} y NO reemplazan a las completas.")

archivo_salida = salida(f"df_features_with_errors{SUF}.xlsx")

M4_CONFIG = {
    "Yearly": {
        "train_file": DATA_DIR / "Yearly-train.csv",
        "test_file": DATA_DIR / "Yearly-test.csv",
        "h": 6,
        "season_length": None,
    },
    "Quarterly": {
        "train_file": DATA_DIR / "Quarterly-train.csv",
        "test_file": DATA_DIR / "Quarterly-test.csv",
        "h": 8,
        "season_length": 4,
    },
    "Monthly": {
        "train_file": DATA_DIR / "Monthly-train.csv",
        "test_file": DATA_DIR / "Monthly-test.csv",
        "h": 18,
        "season_length": 12,
    },
    "Weekly": {
        "train_file": DATA_DIR / "Weekly-train.csv",
        "test_file": DATA_DIR / "Weekly-test.csv",
        "h": 13,
        "season_length": 52,
    },
    "Daily": {
        "train_file": DATA_DIR / "Daily-train.csv",
        "test_file": DATA_DIR / "Daily-test.csv",
        "h": 14,
        "season_length": 7,
    },
    "Hourly": {
        "train_file": DATA_DIR / "Hourly-train.csv",
        "test_file": DATA_DIR / "Hourly-test.csv",
        "h": 48,
        "season_length": 24,
    },
}


# =========================================================
# FUNCIONES
# =========================================================

def load_m4(file_path):
    df = pd.read_csv(file_path)

    if df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]

    id_col = df.columns[0]

    series = {}

    for _, row in df.iterrows():
        serie_id = str(row[id_col])
        values = (
            pd.to_numeric(row.iloc[1:], errors="coerce")
            .dropna()
            .astype(float)
            .values
        )
        series[serie_id] = values

    return series


def safe_smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask])


def forecast_naive(train, h):
    train = np.asarray(train, dtype=float)

    if len(train) == 0:
        return np.full(h, np.nan)

    return np.repeat(train[-1], h)


def forecast_seasonal_naive(train, h, season_length):
    train = np.asarray(train, dtype=float)

    if season_length is None:
        return np.full(h, np.nan)

    if len(train) < season_length:
        return np.full(h, np.nan)

    last_season = train[-season_length:]
    reps = int(np.ceil(h / season_length))

    return np.tile(last_season, reps)[:h]


def forecast_arima(train, h):
    try:
        model = ARIMA(train, order=(1, 1, 1))
        fit = model.fit()
        pred = fit.forecast(steps=h)
        return np.asarray(pred)[:h]
    except Exception:
        return np.full(h, np.nan)


def forecast_sarima(train, h, season_length):
    if season_length is None:
        return np.full(h, np.nan)

    if len(train) < 2 * season_length:
        return np.full(h, np.nan)

    try:
        model = SARIMAX(
            train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, season_length),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        pred = fit.forecast(steps=h)
        return np.asarray(pred)[:h]
    except Exception:
        return np.full(h, np.nan)


# =========================================================
# CARGA DE FEATURES
# =========================================================

df_features = pd.read_excel(archivo_features)

print("Shape df_features:", df_features.shape)


# =========================================================
# LOOP PRINCIPAL
# =========================================================

resultados = []

for category, cfg in M4_CONFIG.items():
    print(f"\nProcesando categoría: {category}")

    train_series = load_m4(cfg["train_file"])
    test_series = load_m4(cfg["test_file"])

    h = cfg["h"]
    season_length = cfg["season_length"]

    df_cat = df_features[df_features["category"] == category].copy()

    if MAX_SERIES_PER_CATEGORY is not None:
        df_cat = df_cat.head(MAX_SERIES_PER_CATEGORY)

    print("Series a procesar:", len(df_cat))

    for i, (_, row) in enumerate(df_cat.iterrows(), start=1):
        serie_id = str(row["serie"])

        train = train_series.get(serie_id)
        test = test_series.get(serie_id)

        resultado = {
            "serie": serie_id,
            "category": category,
            "error_naive_smape": np.nan,
            "error_snaive_smape": np.nan,
            "error_arima_smape": np.nan,
            "error_sarima_smape": np.nan,
            "status_naive": "error",
            "status_snaive": "error",
            "status_arima": "error",
            "status_sarima": "error",
        }

        if train is None or test is None:
            resultado.update({
                "status_naive": "serie_no_encontrada",
                "status_snaive": "serie_no_encontrada",
                "status_arima": "serie_no_encontrada",
                "status_sarima": "serie_no_encontrada",
            })
            resultados.append(resultado)
            continue

        if len(train) < 2 or len(test) < h:
            resultado.update({
                "status_naive": "serie_corta",
                "status_snaive": "serie_corta",
                "status_arima": "serie_corta",
                "status_sarima": "serie_corta",
            })
            resultados.append(resultado)
            continue

        y_true = np.asarray(test[:h], dtype=float)

        # -------------------------
        # Naive
        # -------------------------
        y_naive = forecast_naive(train, h)
        resultado["error_naive_smape"] = safe_smape(y_true, y_naive)
        resultado["status_naive"] = "ok"

        # -------------------------
        # Seasonal Naive
        # -------------------------
        if season_length is not None and len(train) >= season_length:
            y_snaive = forecast_seasonal_naive(train, h, season_length)
            resultado["error_snaive_smape"] = safe_smape(y_true, y_snaive)
            resultado["status_snaive"] = "ok"
        else:
            resultado["status_snaive"] = "no_aplica"

        # -------------------------
        # ARIMA
        # -------------------------
        if RUN_ARIMA:
            y_arima = forecast_arima(train, h)
            if np.isnan(y_arima).all():
                resultado["status_arima"] = "error"
            else:
                resultado["error_arima_smape"] = safe_smape(y_true, y_arima)
                resultado["status_arima"] = "ok"
        else:
            resultado["status_arima"] = "no_ejecutado"

        # -------------------------
        # SARIMA
        # -------------------------
        if RUN_SARIMA and season_length is not None and len(train) >= 2 * season_length:
            y_sarima = forecast_sarima(train, h, season_length)
            if np.isnan(y_sarima).all():
                resultado["status_sarima"] = "error"
            else:
                resultado["error_sarima_smape"] = safe_smape(y_true, y_sarima)
                resultado["status_sarima"] = "ok"
        else:
            resultado["status_sarima"] = "no_aplica"

        resultados.append(resultado)

        if i % 1000 == 0:
            print(f"{category}: {i}/{len(df_cat)} series procesadas")

    # Checkpoint por categoría
    df_partial = pd.DataFrame(resultados)
    df_partial.to_excel(salida(f"forecasting_errors_partial{SUF}.xlsx"), index=False)
    print(f"Checkpoint guardado después de {category}")


# =========================================================
# MERGE FINAL
# =========================================================

df_errors = pd.DataFrame(resultados)

df_final = df_features.merge(
    df_errors,
    on=["serie", "category"],
    how="left"
)


# =========================================================
# GUARDADO
# =========================================================

df_errors.to_excel(salida(f"forecasting_errors_only{SUF}.xlsx"), index=False)
df_final.to_excel(archivo_salida, index=False)

print("\nArchivos guardados:")
print("-", salida(f"forecasting_errors_only{SUF}.xlsx"))
print("-", archivo_salida)

print("\nStatus Naive:")
print(df_final["status_naive"].value_counts(dropna=False))

print("\nStatus SNaive:")
print(df_final["status_snaive"].value_counts(dropna=False))

print("\nStatus ARIMA:")
print(df_final["status_arima"].value_counts(dropna=False))

print("\nStatus SARIMA:")
print(df_final["status_sarima"].value_counts(dropna=False))

print("\nsMAPE válidos:")
print("Naive :", df_final["error_naive_smape"].notna().sum())
print("SNaive:", df_final["error_snaive_smape"].notna().sum())
print("ARIMA :", df_final["error_arima_smape"].notna().sum())
print("SARIMA:", df_final["error_sarima_smape"].notna().sum())