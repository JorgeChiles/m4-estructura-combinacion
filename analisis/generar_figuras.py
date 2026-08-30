#!/usr/bin/env python3
"""
Genera las figuras del articulo en PDF vectorial (y PNG para vista rapida).

Salida: TESIS/figuras/*.pdf

Estilo pensado para revista: serif, sin adornos, paleta segura para
daltonismo, todo legible al tamano final de columna. Los PDF son vectoriales:
no pixelan al imprimir y pesan poco.

Uso:  python3 generar_figuras.py
"""
import os
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
CPX = AQUI / "m4-structural-complexity"
RES = CPX / "results"
MOD = AQUI / "TRANSPOSE_1000_RANDOM"
OUT = AQUI / "figuras"
OUT.mkdir(exist_ok=True)

# ── estilo ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# paleta segura para daltonismo (Okabe-Ito) + rampa secuencial azul
AZUL, NARANJA, VERDE = "#0072B2", "#D55E00", "#009E73"
GRIS, GRIS_C = "#4D4D4D", "#BFBFBF"
RAMPA = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]

# Los consolidados guardan los nombres tal como se escribieron en el codigo.
# Para el articulo se pasan a nomenclatura de publicacion: sin guion bajo,
# con acentos y con los nombres de modelo en su forma habitual.
NOMBRES = {
    "ARIMA_RF": "ARIMA-RF", "ARIMA_NN": "ARIMA-RN", "SARIMA_NN": "SARIMA-RN",
    "LSTM_Conv1D": "LSTM-ConvNet", "Dense NN": "Perceptrón",
    "ConvNet 1D": "ConvNet 1D", "Lineal": "Regresión lineal",
    "Bayesiano": "Regresión bayesiana", "LassoLars": "LassoLars",
    "Suavizado Exponencial": "Suavizado exponencial",
    "GaussianProcess": "Proceso gaussiano", "Random Forest": "Random Forest",
    "Naive Estacional": "Naive estacional", "naive2": "Naive2",
    "naive": "Naive", "snaive": "Naive estacional", "Naive": "Naive",
    "SVM": "SVM", "KNN": "KNN",
}


def pub(x):
    """Nombre de modelo listo para publicacion."""
    if hasattr(x, "map"):
        return x.map(lambda v: NOMBRES.get(v, v))
    return NOMBRES.get(x, x)


FRECS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
ES = {"Yearly": "Anual", "Quarterly": "Trimestral", "Monthly": "Mensual",
      "Weekly": "Semanal", "Daily": "Diaria", "Hourly": "Horaria"}


def guardar(fig, nombre):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{nombre}.{ext}")
    plt.close(fig)
    print(f"  figuras/{nombre}.pdf", flush=True)


# ── datos ────────────────────────────────────────────────────────────
print("cargando datos...", flush=True)
clus = pd.read_excel(RES / "df_features_clustered.xlsx")
Xp = np.load(RES / "X_pca.npy")
ent = pd.read_excel(RES / "df_entropy.xlsx")
err = pd.read_excel(RES / "forecasting_errors_completo.xlsx")

D = (clus.merge(ent, on=["serie", "category"])
         .merge(err[["serie", "category"] +
                    [c for c in err.columns if c.startswith("error_")]],
                on=["serie", "category"], how="left"))
DE = D.dropna(subset=["complexity_index", "error_naive_smape"])
print(f"  {len(D):,} series | {len(DE):,} con error de pronostico", flush=True)


# ══ FIG 1 · espacio PCA y particion ══════════════════════════════════
def fig_pca():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"wspace": 0.24})
    rng = np.random.default_rng(42)
    idx = rng.choice(len(Xp), 14000, replace=False)
    lab = clus["cluster"].values[idx]

    for c, col, nom in [(0, AZUL, "Cluster 0 · regular"),
                        (1, NARANJA, "Cluster 1 · complejo")]:
        m = lab == c
        ax[0].scatter(Xp[idx][m, 0], Xp[idx][m, 1], s=1.6, c=col,
                      alpha=.30, linewidths=0, rasterized=True, label=nom)
    ax[0].set_xlabel("PC$_1$"); ax[0].set_ylabel("PC$_2$")
    ax[0].set_title("(a) Partición en el espacio de componentes", loc="left")
    lg = ax[0].legend(markerscale=7, handletextpad=.2, frameon=False, loc="lower left")
    for h in lg.legend_handles:
        h.set_alpha(1)

    # densidad del indice por cluster
    for c, col, nom in [(0, AZUL, "Cluster 0"), (1, NARANJA, "Cluster 1")]:
        v = D.loc[D.cluster == c, "complexity_index"]
        ax[1].hist(v, bins=np.linspace(-8, 20, 90), color=col, alpha=.62,
                   density=True, label=nom, edgecolor="none")
    ax[1].set_xlabel("Índice de complejidad")
    ax[1].set_ylabel("Densidad")
    ax[1].set_xlim(-8, 20)
    ax[1].set_title("(b) Distribución del índice", loc="left")
    ax[1].legend(frameon=False)
    guardar(fig, "fig1_pca_clusters")


# ══ FIG 2 · seleccion del numero de grupos ═══════════════════════════
def fig_seleccion_k():
    km = pd.read_excel(RES / "kmeans_metrics.xlsx").sort_values("k")
    gm = pd.read_excel(RES / "gmm_bic_aic.xlsx").sort_values("k")
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw={"wspace": 0.34})

    ax[0].plot(km.k, km.silhouette, "o-", color=AZUL, ms=4, lw=1.2)
    ax[0].scatter([2], [km.silhouette.iloc[0]], s=64, facecolor="none",
                  edgecolor=NARANJA, lw=1.4, zorder=5)
    ax[0].set_ylabel("Silhouette"); ax[0].set_title("(a) Silhouette (mayor mejor)", loc="left")

    ax[1].plot(km.k, km.davies_bouldin, "o-", color=AZUL, ms=4, lw=1.2)
    ax[1].scatter([2], [km.davies_bouldin.iloc[0]], s=64, facecolor="none",
                  edgecolor=NARANJA, lw=1.4, zorder=5)
    ax[1].set_ylabel("Davies-Bouldin"); ax[1].set_title("(b) Davies-Bouldin (menor mejor)", loc="left")

    ax[2].plot(gm.k, gm.bic / 1e5, "o-", color=AZUL, ms=4, lw=1.2, label="BIC")
    ax[2].plot(gm.k, gm.aic / 1e5, "s--", color=GRIS, ms=3.4, lw=1.1, label="AIC")
    ax[2].set_ylabel(r"Criterio ($\times 10^5$)")
    ax[2].set_title("(c) GMM: sin codo", loc="left")
    ax[2].legend(frameon=False)

    for a in ax:
        a.set_xlabel("Número de grupos $k$")
        a.xaxis.set_major_locator(MultipleLocator(2))
    guardar(fig, "fig2_seleccion_k")


# ══ FIG 3 · cargas de la primera componente ══════════════════════════
def fig_loadings():
    L = pd.read_excel(RES / "pc1_loadings.xlsx").sort_values("loading_pc1")
    fig, ax = plt.subplots(figsize=(4.6, 6.2))
    col = [NARANJA if v < 0 else AZUL for v in L.loading_pc1]
    ax.barh(range(len(L)), L.loading_pc1, color=col, height=.72, linewidth=0)
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels([f.replace("_", " ") for f in L.feature], fontsize=7.2)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Carga sobre PC$_1$")
    ax.set_ylim(-.8, len(L) - .2)
    ax.grid(axis="y", visible=False)
    ax.text(.98, .015, "Carga positiva: más estructura temporal\n"
                       "Carga negativa: más irregularidad",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
            color=GRIS, linespacing=1.5)
    guardar(fig, "fig3_loadings_pc1")


# ══ FIG 4 · el indice frente al error ════════════════════════════════
def fig_indice_error():
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.85), gridspec_kw={"wspace": 0.26})

    # (a) error medio por decil del indice, global
    dec = pd.qcut(DE.complexity_index, 10, labels=False)
    g = DE.groupby(dec).agg(x=("complexity_index", "mean"),
                            y=("error_naive_smape", "mean"),
                            s=("error_naive_smape", "sem"))
    ax[0].errorbar(g.x, g.y, yerr=1.96 * g.s, fmt="o-", color=AZUL, ms=4.5,
                   lw=1.3, capsize=2.4, elinewidth=.9)
    ax[0].set_xlabel("Índice de complejidad (media del decil)")
    ax[0].set_ylabel("sMAPE Naive")
    ax[0].set_title("(a) Relación monótona, todas las series", loc="left")
    ax[0].text(.04, .93, f"$n$ = {len(DE):,}", transform=ax[0].transAxes,
               fontsize=7.6, color=GRIS, va="top")

    # (b) por cuartil y frecuencia
    q = pd.qcut(DE.complexity_index, 4, labels=False)
    piv = (DE.assign(q=q).groupby(["category", "q"]).error_naive_smape.mean()
             .unstack().reindex(FRECS))
    x = np.arange(len(FRECS)); w = .2
    for i in range(4):
        ax[1].bar(x + (i - 1.5) * w, piv[i], w, color=RAMPA[i], linewidth=0,
                  label=["Q1 bajo", "Q2", "Q3", "Q4 alto"][i])
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([ES[f] for f in FRECS], rotation=22, ha="right")
    ax[1].set_ylabel("sMAPE Naive")
    ax[1].set_title("(b) Por cuartil de complejidad y frecuencia", loc="left")
    ax[1].legend(frameon=False, ncol=2, columnspacing=1, handlelength=1.1)
    ax[1].grid(axis="x", visible=False)
    guardar(fig, "fig4_indice_error")


# ══ FIG 5 · correlacion por frecuencia ═══════════════════════════════
def fig_correlacion():
    filas = []
    for f in FRECS:
        g = DE[DE.category == f]
        filas.append({
            "f": ES[f], "n": len(g),
            "idx": g.complexity_index.corr(g.error_naive_smape),
            "lz": g.lz_complexity.corr(g.error_naive_smape),
            "se": g.sample_entropy.corr(g.error_naive_smape),
            "pe": g.perm_entropy.corr(g.error_naive_smape)})
    t = pd.DataFrame(filas)

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(t)); w = .2
    for i, (c, col, nom) in enumerate([
            ("idx", AZUL, "Índice de complejidad"),
            ("lz", NARANJA, "Lempel-Ziv"),
            ("se", VERDE, "Sample entropy"),
            ("pe", GRIS_C, "Entropía de permutación")]):
        ax.bar(x + (i - 1.5) * w, t[c], w, color=col, linewidth=0, label=nom)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.f}\n$n$={r.n:,}" for r in t.itertuples()], fontsize=7.6)
    ax.set_ylabel("Correlación con el sMAPE del Naive")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.16),
              handlelength=1.1, columnspacing=1.4)
    ax.grid(axis="x", visible=False)
    # la anotacion va al hueco de abajo a la izquierda: arriba pisaba
    # las barras de la frecuencia semanal
    ax.annotate("las series diarias no\ntienen estructura que medir",
                xy=(3.72, .02), xytext=(1.75, -.235), fontsize=7, color=GRIS,
                ha="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=.7,
                                connectionstyle="arc3,rad=.2"))
    guardar(fig, "fig5_correlacion_frecuencia")


# ══ FIG 6 · OWA de los modelos ═══════════════════════════════════════
def fig_owa():
    p = pd.read_excel(MOD / "owa_modelos_ponderado.xlsx")
    p = p[~p.modelo.isin(["naive", "snaive"])].copy()
    p = p[p.OWA_pond < 3].sort_values("OWA_pond", ascending=False)

    fig, ax = plt.subplots(figsize=(4.9, 5.6))
    d = p.OWA_pond - 1
    col = [VERDE if v < 0 else NARANJA for v in d]
    col = [GRIS_C if m == "naive2" else c for m, c in zip(p.modelo, col)]
    y = np.arange(len(p))
    ax.barh(y, d, color=col, height=.72, linewidth=0)
    ax.axvline(0, color="k", lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels(pub(p.modelo), fontsize=7.6)
    ax.set_xlabel("OWA $-$ 1  (desvío respecto de Naive2)")
    ax.set_ylim(-.8, len(p) - .2)
    ax.grid(axis="y", visible=False)
    # los valores van en una columna fija a la derecha: pegados al extremo
    # de la barra chocaban con los nombres de los modelos
    xr = max(d) * 1.30
    for yi, (m, nm) in enumerate(zip(p.OWA_pond, p.modelo)):
        ax.text(xr, yi, f"{m:.3f}", va="center", ha="right", fontsize=6.9,
                color=GRIS if nm == "naive2" else "black")
    ax.set_xlim(-.11, max(d) * 1.34)
    # la explicacion del color va en el pie de figura del articulo,
    # no encima de las barras
    guardar(fig, "fig6_owa_modelos")


# ══ FIG 7 · OWA por modelo y frecuencia (mapa de calor) ══════════════
def fig_owa_heatmap():
    r = pd.read_excel(MOD / "owa_modelos_resumen.xlsx")
    r = r[r.frecuencia != "TODAS"]
    piv = r.pivot(index="modelo", columns="frecuencia", values="OWA")
    piv = piv.reindex(columns=[f for f in FRECS if f in piv.columns])
    orden = piv.mean(axis=1).sort_values().index
    piv = piv.loc[orden]
    piv = piv[~piv.index.isin(["naive", "snaive"])]

    M = np.clip(piv.values, .55, 1.75)
    fig, ax = plt.subplots(figsize=(5.0, 6.4))
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=.55, vmax=1.75, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([ES[c] for c in piv.columns], rotation=35, ha="right", fontsize=7.6)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([pub(m) for m in piv.index], fontsize=7.2)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                        color="white" if (v < .72 or v > 1.5) else "#1a1a1a")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03,
                      ticks=[.6, .8, 1.0, 1.2, 1.4, 1.6])
    cb.set_label("OWA", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(1.0, color="k", lw=1.1)
    guardar(fig, "fig7_owa_heatmap")


# ══ FIG 8 · varianza por semilla ═════════════════════════════════════
def fig_semillas():
    # ahora con las seis frecuencias (4.500 mediciones); el archivo de tres
    # queda como respaldo por si el consolidado no existe
    arch = MOD / "varianza_redes_6frec.csv"
    if not arch.exists():
        arch = MOD / "varianza_redes.csv"
    v = pd.read_csv(arch).dropna(subset=["smape"])
    g = v.groupby(["frecuencia", "serie", "modelo"]).smape.agg(["mean", "min", "max"])
    g["rango"] = g["max"] - g["min"]
    orden = g.groupby("modelo")["mean"].median().sort_values().index

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"wspace": .3})

    datos = [g.xs(m, level="modelo")["rango"].values for m in orden]
    bp = ax[0].boxplot(datos, vert=False, widths=.6, showfliers=False,
                       patch_artist=True, medianprops=dict(color="k", lw=1.1))
    for b in bp["boxes"]:
        b.set(facecolor=AZUL, alpha=.55, linewidth=.7)
    ax[0].set_yticklabels([pub(m) for m in orden], fontsize=7.6)
    ax[0].set_xlabel("Rango de sMAPE entre 5 semillas")
    ax[0].set_title("(a) Dispersión por inicialización", loc="left")
    ax[0].grid(axis="y", visible=False)

    piv = v.pivot_table(index=["frecuencia", "serie", "semilla"],
                        columns="modelo", values="smape")
    gan = piv.idxmin(axis=1).reset_index()
    cambia = gan.groupby(["frecuencia", "serie"])[0].nunique()
    ORD = ["YEARLY", "QUARTELY", "MONTHLY", "WEEKLY", "DAILY", "HOURLY"]
    ETQ = {"YEARLY": "Anual", "QUARTELY": "Trimestral", "MONTHLY": "Mensual",
           "WEEKLY": "Semanal", "DAILY": "Diaria", "HOURLY": "Horaria"}
    fr = [f for f in ORD if f in cambia.index.get_level_values(0)]
    pct = [100 * (cambia.loc[f] > 1).mean() for f in fr]
    tot = 100 * (cambia > 1).mean()

    x = np.arange(len(fr))
    # la horaria se destaca: es la unica frecuencia donde las redes le ganan
    # al benchmark, y resulta ser la mas inestable de las seis
    col = [NARANJA if f != "HOURLY" else "#8C3A10" for f in fr]
    ax[1].bar(x, pct, .62, color=col, linewidth=0)
    ax[1].axhline(tot, color=GRIS, ls="--", lw=1, label=f"Total: {tot:.0f} %")
    for xi, pp in zip(x, pct):
        ax[1].text(xi, pp + 1.8, f"{pp:.0f}", ha="center", fontsize=7.4)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([ETQ[f] for f in fr], rotation=32, ha="right", fontsize=7.6)
    ax[1].set_ylabel("% de series")
    ax[1].set_ylim(0, 108)
    ax[1].set_title("(b) El mejor modelo cambia con la semilla", loc="left")
    ax[1].legend(frameon=False, loc="lower left", fontsize=7.6)
    ax[1].grid(axis="x", visible=False)
    guardar(fig, "fig8_varianza_semillas")


# ══ FIG 9 · validacion del benchmark Naive2 ══════════════════════════
def fig_validacion():
    v = pd.read_excel(RES / "m4_validacion_naive2.xlsx")
    v = v[v.category != "TODAS"]
    v["f"] = v.category.map(ES)
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    x = np.arange(len(v))
    ax.scatter(x - .09, v.sMAPE_M4, s=52, marker="_", color=GRIS,
               linewidths=1.8, label="Publicado (M4, 2020)")
    ax.scatter(x + .09, v.sMAPE_calculado, s=30, color=AZUL,
               label="Replicado en este trabajo", zorder=4)
    for xi, (a, b) in enumerate(zip(v.sMAPE_M4, v.sMAPE_calculado)):
        ax.plot([xi - .09, xi + .09], [a, b], color=GRIS_C, lw=.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(v.f, rotation=28, ha="right", fontsize=7.6)
    ax.set_ylabel("sMAPE del benchmark Naive2")
    ax.legend(frameon=False, fontsize=7.4)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(top=v[["sMAPE_M4", "sMAPE_calculado"]].max().max() * 1.14)
    ax.text(.98, .06, f"desvío máximo: {v.dif_sMAPE.abs().max():.3f}",
            transform=ax.transAxes, ha="right", fontsize=7.2, color=GRIS)
    guardar(fig, "fig9_validacion_naive2")


for f in (fig_pca, fig_seleccion_k, fig_loadings, fig_indice_error,
          fig_correlacion, fig_owa, fig_owa_heatmap, fig_semillas, fig_validacion):
    f()

print(f"\n{len(list(OUT.glob('*.pdf')))} figuras en {OUT}")
