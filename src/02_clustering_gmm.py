# =========================================================
# 02_clustering_gmm.py
# Evaluación de clustering y selección GMM
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture


# =========================================================
# CONTROL DE HILOS / THREADS
# =========================================================

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =========================================================
# ARCHIVOS DE ENTRADA Y SALIDA
# =========================================================

from rutas import entrada, salida  # noqa: E402

# Antes eran nombres relativos sueltos: el script solo funcionaba si se lo
# corria parado en la raiz del proyecto. Ahora la ruta no depende del cwd.
archivo_features = entrada("df_features_complexity.xlsx")
archivo_x_pca = entrada("X_pca.npy")

archivo_metricas_k = salida("kmeans_metrics.xlsx")
archivo_gmm = salida("gmm_bic_aic.xlsx")
archivo_clustered = salida("df_features_clustered.xlsx")

figura_k = salida("k_silhouette_davies.png")
figura_gmm = salida("gmm_bic_aic.png")
figura_clusters = salida("clusters_pca.png")


# =========================================================
# CONFIGURACIÓN
# =========================================================

RANDOM_STATE = 42
N_SAMPLE = 30000
K_RANGE = range(2, 10)
GMM_RANGE = range(2, 11)
BATCH_SIZE = 2048


# =========================================================
# CARGA DE DATOS
# =========================================================

df = pd.read_excel(archivo_features)
X_pca = np.load(archivo_x_pca)

print("Shape df:", df.shape)
print("Shape X_pca:", X_pca.shape)


# =========================================================
# MUESTRA PARA EVALUACIÓN
# =========================================================

rng = np.random.default_rng(RANDOM_STATE)

sample_size = min(N_SAMPLE, len(df))
sample_idx = rng.choice(len(df), size=sample_size, replace=False)

X_pca_sample = X_pca[sample_idx]

print("Muestra para selección de K:", X_pca_sample.shape)


# =========================================================
# EVALUACIÓN DE KMEANS
# =========================================================

resultados_k = []

for k in K_RANGE:
    print(f"Evaluando MiniBatchKMeans con K={k}")

    modelo_tmp = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=BATCH_SIZE,
        n_init=10
    )

    labels_tmp = modelo_tmp.fit_predict(X_pca_sample)

    silhouette = silhouette_score(X_pca_sample, labels_tmp)
    davies_bouldin = davies_bouldin_score(X_pca_sample, labels_tmp)

    resultados_k.append({
        "k": k,
        "metodo": "MiniBatchKMeans",
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin
    })


res_k = pd.DataFrame(resultados_k)

# Tabla ordenada por K para graficar correctamente
res_k_plot = res_k.sort_values("k").copy()

# Tabla ordenada por calidad para seleccionar mejor K
res_k_quality = res_k.sort_values(
    by=["silhouette", "davies_bouldin"],
    ascending=[False, True]
).copy()

best_k = int(res_k_quality.iloc[0]["k"])

print("\nResultados por K:")
print(res_k_plot)

print("\nMejor K según métricas internas:", best_k)


# =========================================================
# GRÁFICA SILHOUETTE Y DAVIES-BOULDIN
# =========================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(res_k_plot["k"], res_k_plot["silhouette"], marker="o")
plt.title("Silhouette vs K")
plt.xlabel("K")
plt.ylabel("Silhouette")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(res_k_plot["k"], res_k_plot["davies_bouldin"], marker="o")
plt.title("Davies-Bouldin vs K")
plt.xlabel("K")
plt.ylabel("Davies-Bouldin")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figura_k, dpi=300)
plt.close()


# =========================================================
# ENTRENAMIENTO FINAL KMEANS
# =========================================================

modelo_final = MiniBatchKMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    batch_size=BATCH_SIZE,
    n_init=20
)

labels_final = modelo_final.fit_predict(X_pca)

df["cluster"] = labels_final

tam_clusters = df["cluster"].value_counts().sort_index()
prop_clusters = df["cluster"].value_counts(normalize=True).sort_index()

print("\nTamaño de clusters:")
print(tam_clusters)

print("\nProporción de clusters:")
print(prop_clusters)


# =========================================================
# FIGURA PCA CON CLUSTERS
# =========================================================

plot_n = min(10000, len(df))
plot_idx = rng.choice(len(df), size=plot_n, replace=False)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca[plot_idx, 0],
    X_pca[plot_idx, 1],
    c=df.iloc[plot_idx]["cluster"],
    s=6,
    alpha=0.7
)

plt.title("Clusters globales M4 sobre PCA (muestra)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figura_clusters, dpi=300)
plt.close()


# =========================================================
# GMM: AIC Y BIC
# =========================================================

resultados_gmm = []

for k in GMM_RANGE:
    print(f"Evaluando GMM con K={k}")

    gmm = GaussianMixture(
        n_components=k,
        random_state=RANDOM_STATE,
        covariance_type="full"
    )

    gmm.fit(X_pca_sample)

    resultados_gmm.append({
        "k": k,
        "aic": gmm.aic(X_pca_sample),
        "bic": gmm.bic(X_pca_sample)
    })


res_gmm = pd.DataFrame(resultados_gmm)

print("\nResultados GMM:")
print(res_gmm)


# =========================================================
# GRÁFICA GMM AIC/BIC
# =========================================================

plt.figure(figsize=(8, 4))

plt.plot(res_gmm["k"], res_gmm["bic"], marker="o", label="BIC")
plt.plot(res_gmm["k"], res_gmm["aic"], marker="o", label="AIC")

plt.title("GMM Model Selection")
plt.xlabel("Número de componentes")
plt.ylabel("Criterio de información")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figura_gmm, dpi=300)
plt.close()


# =========================================================
# GUARDADO DE RESULTADOS
# =========================================================

res_k_plot.to_excel(archivo_metricas_k, index=False)
res_gmm.to_excel(archivo_gmm, index=False)
df.to_excel(archivo_clustered, index=False)

with pd.ExcelWriter(salida("clustering_gmm_results.xlsx"), engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="series_clustered", index=False)
    res_k_plot.to_excel(writer, sheet_name="kmeans_metrics", index=False)
    res_gmm.to_excel(writer, sheet_name="gmm_aic_bic", index=False)
    tam_clusters.to_frame("n_series").to_excel(writer, sheet_name="cluster_sizes")
    prop_clusters.to_frame("proportion").to_excel(writer, sheet_name="cluster_props")


print("\nArchivos guardados:")
print("-", archivo_metricas_k)
print("-", archivo_gmm)
print("-", archivo_clustered)
print("-", salida("clustering_gmm_results.xlsx"))
print("-", figura_k)
print("-", figura_gmm)
print("-", figura_clusters)