# =========================================================
# 04_entropy_validation.py
# Valida índice de complejidad contra entropías clásicas
# =========================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from rutas import entrada, salida  # noqa: E402

# =========================================================
# RUTAS
# =========================================================

archivo_features = entrada("df_features_complexity.xlsx")
archivo_entropy = entrada("df_entropy.xlsx")

# =========================================================
# CARGA Y UNION DE DATOS
# =========================================================

df_features = pd.read_excel(archivo_features)
df_entropy = pd.read_excel(archivo_entropy)

df_features_entropy = df_features.merge(
    df_entropy,
    on=["serie", "category"],
    how="left"
)

# Los errores de pronostico NO estan en df_features_complexity: los produce
# el script 04 en un archivo aparte. Sin este merge, error_naive_smape no
# existe y el script falla. Se prefiere el completo (100.000 series) y se
# cae al parcial (95.000) si el completo todavia no se genero.
_errores = None
for _cand in ("forecasting_errors_completo.xlsx", "forecasting_errors_partial.xlsx"):
    _ruta = entrada(_cand)
    if _ruta.exists():
        _errores = pd.read_excel(_ruta)
        print(f"errores de pronostico: {_cand}  ({len(_errores)} filas)")
        break
if _errores is None:
    raise FileNotFoundError("Falta forecasting_errors_completo.xlsx (corre 04b primero)")

_cols_err = ["serie", "category"] + [c for c in _errores.columns if c.startswith("error_")]
df_features_entropy = df_features_entropy.merge(
    _errores[_cols_err], on=["serie", "category"], how="left"
)
print(f"union final: {len(df_features_entropy)} filas, "
      f"{df_features_entropy['error_naive_smape'].notna().sum()} con error de pronostico")

# =========================================================
# VARIABLES A EVALUAR
# =========================================================

cols_eval = [
    "complexity_index",
    "perm_entropy",
    "sample_entropy",
    "lz_complexity",
    "error_naive_smape"
]

print("\nPrimeras filas:")
print(df_features_entropy[[
    "serie", "category", "complexity_index",
    "perm_entropy", "sample_entropy", "lz_complexity",
    "error_naive_smape"
]].head())

# =========================================================
# CORRELACION GLOBAL
# =========================================================

corr_global = df_features_entropy[cols_eval].corr(numeric_only=True)

print("\n===== CORRELACIÓN GLOBAL =====")
print(corr_global)

corr_global.to_excel(salida("corr_global_entropy.xlsx"))

# =========================================================
# CORRELACION POR FRECUENCIA
# =========================================================

resultados_freq = []

for cat in df_features_entropy["category"].dropna().unique():
    sub = df_features_entropy[df_features_entropy["category"] == cat].copy()

    corr = sub[cols_eval].corr(numeric_only=True)

    print(f"\n===== {cat} =====")
    print(corr)

    resultados_freq.append({
        "category": cat,
        "corr_complexity_error": corr.loc["complexity_index", "error_naive_smape"],
        "corr_perm_entropy_error": corr.loc["perm_entropy", "error_naive_smape"],
        "corr_sample_entropy_error": corr.loc["sample_entropy", "error_naive_smape"],
        "corr_lz_error": corr.loc["lz_complexity", "error_naive_smape"],
    })

df_corr_freq = pd.DataFrame(resultados_freq)
df_corr_freq.to_excel(salida("corr_by_frequency_entropy.xlsx"), index=False)

# =========================================================
# REGRESIONES
# =========================================================

df_reg = df_features_entropy.dropna(subset=cols_eval).copy()

y = df_reg["error_naive_smape"].values

X1 = df_reg[["complexity_index"]].values
m1 = LinearRegression().fit(X1, y)
r2_1 = r2_score(y, m1.predict(X1))

X2 = df_reg[["perm_entropy", "sample_entropy", "lz_complexity"]].values
m2 = LinearRegression().fit(X2, y)
r2_2 = r2_score(y, m2.predict(X2))

X3 = df_reg[[
    "complexity_index",
    "perm_entropy",
    "sample_entropy",
    "lz_complexity"
]].values
m3 = LinearRegression().fit(X3, y)
r2_3 = r2_score(y, m3.predict(X3))

res_reg = pd.DataFrame({
    "modelo": [
        "solo_complexity_index",
        "solo_entropias",
        "complexity_index_mas_entropias"
    ],
    "r2": [r2_1, r2_2, r2_3]
})

coef_m3 = pd.DataFrame({
    "variable": [
        "complexity_index",
        "perm_entropy",
        "sample_entropy",
        "lz_complexity"
    ],
    "coeficiente": m3.coef_
})

print("\n===== R2 REGRESIONES =====")
print(res_reg)

print("\n===== COEFICIENTES MODELO 3 =====")
print(coef_m3)

res_reg.to_excel(salida("r2_entropy_regressions.xlsx"), index=False)
coef_m3.to_excel(salida("coef_entropy_regression.xlsx"), index=False)

print("\nArchivos guardados:")
print("- corr_global_entropy.xlsx")
print("- corr_by_frequency_entropy.xlsx")
print("- r2_entropy_regressions.xlsx")
print("- coef_entropy_regression.xlsx")