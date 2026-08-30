# =========================================================
# 01_pca_complexity.py
# PCA + índice de complejidad estructural
# Compatible con Jupyter y VS Code
# =========================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


# =========================================================
# CONTROL DE HILOS
# =========================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =========================================================
# RUTAS COMPATIBLES CON JUPYTER Y VS CODE
# =========================================================

from rutas import BASE_DIR, RES_DIR, entrada, salida  # noqa: E402

# features_m4_all_v2.xlsx es una ENTRADA original (la produce el notebook
# CLASIFICACION_V3_All), no una salida del pipeline: se queda en la raiz.
archivo_features = entrada("features_m4_all_v2.xlsx")

archivo_salida = salida("df_features_complexity.xlsx")
archivo_loadings = salida("pc1_loadings.xlsx")

figura_loadings = salida("pc1_loadings.png")
figura_hist = salida("hist_complexity.png")

archivo_x_pca = salida("X_pca.npy")
archivo_x_scaled = salida("X_scaled.npy")


# =========================================================
# FEATURES
# =========================================================

FEATURES_CLUSTER_V2 = [
    "z_skew", "z_kurtosis", "z_entropy", "z_spectral_entropy",
    "z_outlier_ratio", "z_turning_points_ratio", "z_hurst",
    "z_acf1", "z_acf6", "z_acf_freq", "z_acf_decay",
    "z_trend_linearity_r2", "z_curvature_gain", "z_trend_slope",
    "trend_strength", "seasonal_strength",
    "dominant_frequency", "dominant_energy_ratio",
    "adf_pvalue", "kpss_pvalue", "stationarity_conflict",
    "diff_var_ratio", "change_points_per_length",
    "diff_skew", "diff_kurtosis", "diff_entropy",
    "diff_turning_points_ratio", "robust_entropy", "robust_outlier_ratio",
]


# =========================================================
# CONFIGURACIÓN
# =========================================================

RANDOM_STATE = 42
PCA_VARIANCE = 0.90


# =========================================================
# CARGA
# =========================================================

df = pd.read_excel(archivo_features)

print("Shape original:", df.shape)


# =========================================================
# SELECCIÓN Y LIMPIEZA DE FEATURES
# =========================================================

features_disponibles = [c for c in FEATURES_CLUSTER_V2 if c in df.columns]

X_df = df[features_disponibles].copy()
X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.dropna(axis=1, how="all")

selector_var = VarianceThreshold(threshold=0.0)
selector_var.fit(X_df.fillna(X_df.median(numeric_only=True)))

cols_var = X_df.columns[selector_var.get_support()]
X_df = X_df[cols_var].copy()

print("Features usadas:", len(X_df.columns))
print("Columnas finales:", list(X_df.columns))


# =========================================================
# IMPUTACIÓN
# =========================================================

imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X_df)


# =========================================================
# ESTANDARIZACIÓN
# =========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)


# =========================================================
# PCA
# =========================================================

pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

print("Dimensión original:", X_scaled.shape[1])
print("Dimensión PCA:", X_pca.shape[1])
print("Varianza explicada acumulada:", pca.explained_variance_ratio_.sum())


# =========================================================
# ÍNDICE DE COMPLEJIDAD
# =========================================================

df["pc1"] = X_pca[:, 0]

# PC1 positivo = estructura temporal explotable.
# Complejidad = inverso de esa estructura.
df["complexity_index"] = -df["pc1"]

df["complexity_level"] = pd.qcut(
    df["complexity_index"],
    q=4,
    labels=["low", "mid_low", "mid_high", "high"]
)

print("\nConteo por nivel de complejidad:")
print(df["complexity_level"].value_counts().sort_index())


# =========================================================
# LOADINGS PC1
# =========================================================

loadings_pc1 = pd.Series(
    pca.components_[0],
    index=X_df.columns,
    name="loading_pc1"
)

loadings_pc1_sorted = loadings_pc1.reindex(
    loadings_pc1.abs().sort_values(ascending=False).index
)

tabla_loadings_pc1 = pd.DataFrame({
    "feature": loadings_pc1_sorted.index,
    "loading_pc1": loadings_pc1_sorted.values,
    "abs_loading": loadings_pc1_sorted.abs().values,
    "direccion": ["positiva" if x > 0 else "negativa" for x in loadings_pc1_sorted.values]
})

print("\nTop 15 loadings PC1:")
print(tabla_loadings_pc1.head(15))


# =========================================================
# FIGURA LOADINGS
# =========================================================

top_n = 12
loadings_top = loadings_pc1_sorted.head(top_n)

plt.figure(figsize=(8, 6))
loadings_top.sort_values().plot(kind="barh")
plt.title("Loadings de la primera componente principal (PC1)")
plt.xlabel("Peso (loading)")
plt.ylabel("Feature")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figura_loadings, dpi=300)
plt.show()


# =========================================================
# HISTOGRAMA COMPLEJIDAD
# =========================================================

plt.figure(figsize=(8, 5))
df["complexity_index"].hist(bins=50)
plt.title("Distribución del índice de complejidad")
plt.xlabel("Complexity Index")
plt.ylabel("Frecuencia")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figura_hist, dpi=300)
plt.show()


# =========================================================
# GUARDADO
# =========================================================

df.to_excel(archivo_salida, index=False)
tabla_loadings_pc1.to_excel(archivo_loadings, index=False)

np.save(archivo_x_pca, X_pca)
np.save(archivo_x_scaled, X_scaled)

print("\nArchivos guardados:")
print("-", archivo_salida)
print("-", archivo_loadings)
print("-", figura_loadings)
print("-", figura_hist)
print("-", archivo_x_pca)
print("-", archivo_x_scaled)