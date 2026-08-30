# =========================================================
# 03_compute_entropy.py
# Calcula entropía de permutación, sample entropy y LZ
# =========================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
from math import factorial
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from rutas import BASE_DIR, DATA_DIR, entrada, salida  # noqa: E402

# =========================================================
# RUTAS Y CONFIGURACION
# =========================================================

archivo_features = entrada("df_features_complexity.xlsx")
archivo_salida = salida("df_entropy.xlsx")

# Estaba en 1000, que da 6.000 series (1000 x 6 frecuencias). Pero el
# df_entropy.xlsx guardado tiene 99.935 filas: se genero con None. Se deja en
# None para que el script reproduzca el resultado que usa el script 05.
MAX_SERIES_PER_CATEGORY_ENT = None

M4_EVAL_CONFIG = {
    "Yearly":    {"train_file": DATA_DIR / "Yearly-train.csv"},
    "Quarterly": {"train_file": DATA_DIR / "Quarterly-train.csv"},
    "Monthly":   {"train_file": DATA_DIR / "Monthly-train.csv"},
    "Weekly":    {"train_file": DATA_DIR / "Weekly-train.csv"},
    "Daily":     {"train_file": DATA_DIR / "Daily-train.csv"},
    "Hourly":    {"train_file": DATA_DIR / "Hourly-train.csv"},
}


# =========================================================
# ENTROPIA DE PERMUTACION
# =========================================================

def permutation_entropy(x, m=3, tau=1, normalize=True):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < (m - 1) * tau + 1:
        return np.nan

    patterns = []
    for i in range(n - (m - 1) * tau):
        window = x[i:i + m * tau:tau]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)

    counts = Counter(patterns)
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    probs = probs[probs > 0]

    pe = -np.sum(probs * np.log(probs))

    if normalize:
        pe /= np.log(factorial(m))

    return max(pe, 0)


# =========================================================
# SAMPLE ENTROPY
# =========================================================

def sample_entropy(x, m=2, r=None):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < m + 2:
        return np.nan

    if r is None:
        r = 0.2 * np.std(x)

    if r == 0:
        return 0.0

    def _count_matches(mm):
        count = 0
        total = 0

        for i in range(n - mm):
            template = x[i:i + mm]

            for j in range(i + 1, n - mm + 1):
                window = x[j:j + mm]

                if np.max(np.abs(template - window)) <= r:
                    count += 1

                total += 1

        return count, total

    count_m, total_m = _count_matches(m)
    count_m1, total_m1 = _count_matches(m + 1)

    if total_m == 0 or total_m1 == 0:
        return np.nan

    if count_m == 0 or count_m1 == 0:
        return np.nan

    phi_m = count_m / total_m
    phi_m1 = count_m1 / total_m1

    return -np.log(phi_m1 / phi_m)


# =========================================================
# COMPLEJIDAD LEMPEL-ZIV
# =========================================================

def lz_complexity_binary(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n < 2:
        return np.nan

    median = np.median(x)
    s = "".join("1" if v > median else "0" for v in x)

    i, l, k = 0, 1, 1
    c = 1

    while True:
        if l + k > n or i + k > n:
            c += 1
            break

        if s[i:i + k] == s[l:l + k]:
            k += 1
        else:
            i += 1

            if i == l:
                c += 1
                l += k

                if l >= n:
                    break

                i = 0
                k = 1

    return c / (n / np.log2(n)) if n > 1 else np.nan


# =========================================================
# CARGA DE SERIES DE M4
# =========================================================

def load_m4_train_series(train_file):
    df_train = pd.read_csv(train_file)



    if df_train.columns[0].startswith("Unnamed"):
        df_train = df_train.iloc[:, 1:]

    id_col = df_train.columns[0]
    series_dict = {}

    for _, row in df_train.iterrows():
        serie_id = str(row[id_col])

        values = (
            pd.to_numeric(row.iloc[1:], errors="coerce")
            .dropna()
            .astype(float)
            .values
        )

        series_dict[serie_id] = values

    return series_dict


# =========================================================
# CARGA DE FEATURES
# =========================================================

df_features = pd.read_excel(archivo_features)

print("Shape df_features:", df_features.shape)

# =========================================================
# LOOP PRINCIPAL
# =========================================================

resultados_entropy = []

for category_name, cfg in M4_EVAL_CONFIG.items():
    print(f"\nProcesando categoría: {category_name}")

    series_dict = load_m4_train_series(cfg["train_file"])

    df_cat = df_features[df_features["category"] == category_name].copy()

    if MAX_SERIES_PER_CATEGORY_ENT is not None:
        df_cat = df_cat.head(MAX_SERIES_PER_CATEGORY_ENT)

    print("Series a procesar:", len(df_cat))

    for _, row in df_cat.iterrows():
        serie_id = str(row["serie"])
        x = series_dict.get(serie_id, None)

        if x is None or len(x) < 10:
            resultados_entropy.append({
                "serie": serie_id,
                "category": category_name,
                "perm_entropy": np.nan,
                "sample_entropy": np.nan,
                "lz_complexity": np.nan
            })
            continue

        resultados_entropy.append({
            "serie": serie_id,
            "category": category_name,
            "perm_entropy": permutation_entropy(x, m=3, tau=1, normalize=True),
            "sample_entropy": sample_entropy(x, m=2),
            "lz_complexity": lz_complexity_binary(x)
        })

# =========================================================
# GUARDADO
# =========================================================

df_entropy = pd.DataFrame(resultados_entropy)

print("\nShape df_entropy:", df_entropy.shape)
print(df_entropy.head())

df_entropy.to_excel(archivo_salida, index=False)

print(f"\nArchivo guardado: {archivo_salida}")