#!/usr/bin/env python3
"""
El enrutamiento por índice frente a un meta-aprendiz completo (estilo FFORMS/FFORMPP).

La objeción que responde
------------------------
Los revisores señalan que el enrutamiento por índice nunca se compara contra los
métodos publicados de selección de modelo por atributos: FFORMS (clasifica cuál
método gana) y FFORMA (aprende pesos para combinar pronósticos).

Qué NO se puede hacer aquí, y por qué
-------------------------------------
FFORMA no es reproducible con estos datos: combina PRONÓSTICOS con pesos
aprendidos, y el pipeline guardó el error por serie y modelo, no las predicciones.
Su pool base tampoco es el mismo (usa los métodos del paquete forecast de R).
Citar su OWA publicado de 0.838 sobre M4 completo y compararlo con 0.895 sobre
4.773 series seria enganoso: distinto conjunto, distinto pool, distinta tarea.

Qué SÍ se puede hacer, y es la comparación pertinente
-----------------------------------------------------
Implementar el ENFOQUE sobre los mismos datos, el mismo pool de 26 modelos y el
mismo protocolo de particiones. Dos variantes:

    E · FFORMS      clasificador sobre los 29 atributos -> predice qué modelo gana
    F · FFORMPP     un regresor por modelo predice su error -> se elige el menor

Ambos usan el vector completo de 29 atributos. La regla del articulo usa UN
escalar y dos estratos. La pregunta es cuanto se pierde por esa simplificación.

Salidas:
    metaaprendizaje_resultados.xlsx
    figuras/fig12_metaaprendizaje.pdf

Uso:  python3 comparacion_metaaprendizaje.py [n_particiones]
"""
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
MOD = AQUI / "TRANSPOSE_1000_RANDOM"
RES = AQUI / "m4-structural-complexity" / "results"
OUT = AQUI / "figuras"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})
AZUL, NARANJA, VERDE, MORADO, GRIS = "#0072B2", "#D55E00", "#009E73", "#7B3294", "#4D4D4D"

N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FUERA = {"naive", "snaive", "Holt-Winters"}
SEMILLA_RF = 42


def cargar():
    d = pd.read_csv(MOD / "owa_modelos_por_serie.csv")
    d = d[~d.modelo.isin(FUERA)]
    L = pd.read_excel(RES / "pc1_loadings.xlsx")
    ATR = L.feature.tolist()
    f = pd.read_excel(RES / "df_features_complexity.xlsx",
                      usecols=["serie", "complexity_index"] + ATR)
    d = d.merge(f, on="serie", how="inner").dropna(subset=["smape", "mase"])
    n_mod = d.modelo.nunique()
    ok = d.groupby("serie").modelo.nunique() == n_mod
    d = d[d.serie.isin(ok[ok].index)]
    return d, ATR, n_mod


def owa(sub, base):
    if len(sub) == 0:
        return np.nan
    bs, bm = base.smape.mean(), base.mase.mean()
    return .5 * (sub.smape.mean() / bs + sub.mase.mean() / bm) if bs and bm else np.nan


def mejor(tabla):
    n2 = tabla[tabla.modelo == "naive2"]
    if n2.empty:
        return None
    bs, bm = n2.smape.mean(), n2.mase.mean()
    g = tabla.groupby("modelo").agg(s=("smape", "mean"), m=("mase", "mean"))
    return (.5 * (g.s / bs + g.m / bm)).idxmin()


def elegidos(TE, eleccion):
    """eleccion: dict serie -> modelo. Devuelve las filas correspondientes."""
    e = pd.DataFrame({"serie": list(eleccion), "modelo": list(eleccion.values())})
    return TE.merge(e, on=["serie", "modelo"])


def una_particion(d, ATR, semilla):
    series = d[["serie", "frecuencia"]].drop_duplicates()
    rng = np.random.default_rng(semilla)
    tr_ids = set()
    for _, g in series.groupby("frecuencia"):
        tr_ids |= set(rng.choice(g.serie.values, len(g) // 2, replace=False))
    TR, TE = d[d.serie.isin(tr_ids)], d[~d.serie.isin(tr_ids)]
    base_te = TE[TE.modelo == "naive2"]
    fila = {"semilla": semilla}

    # tabla serie x atributos (una fila por serie)
    Xtr = TR.drop_duplicates("serie").set_index("serie")[ATR]
    Xte = TE.drop_duplicates("serie").set_index("serie")[ATR]

    # ── A · modelo fijo
    mA = mejor(TR)
    fila["A_fijo"] = owa(TE[TE.modelo == mA], base_te)

    # ── B · por frecuencia
    reglaB = {f: mejor(g) for f, g in TR.groupby("frecuencia")}
    selB = pd.concat([TE[(TE.frecuencia == f) & (TE.modelo == m)]
                      for f, m in reglaB.items() if m])
    fila["B_frecuencia"] = owa(selB, base_te)

    # ── C · el del articulo: frecuencia x 2 estratos del indice
    partes = []
    for f, gtr in TR.groupby("frecuencia"):
        u = gtr.drop_duplicates("serie")
        corte = u.complexity_index.median()      # dos estratos, fijado a priori
        gte = TE[TE.frecuencia == f]
        for lo, hi in ((-np.inf, corte), (corte, np.inf)):
            sub = gtr[(gtr.complexity_index > lo) & (gtr.complexity_index <= hi)]
            m = mejor(sub)
            if m is None:
                continue
            s = gte[(gte.complexity_index > lo) & (gte.complexity_index <= hi)]
            partes.append(s[s.modelo == m])
    fila["C_indice"] = owa(pd.concat(partes), base_te) if partes else np.nan

    # ── E · FFORMS: clasificar cual modelo gana, con los 29 atributos
    ytr = TR.loc[TR.groupby("serie").smape.idxmin()].set_index("serie").modelo
    ytr = ytr.reindex(Xtr.index)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                 random_state=SEMILLA_RF, n_jobs=-1)
    clf.fit(Xtr.values, ytr.values)
    predE = dict(zip(Xte.index, clf.predict(Xte.values)))
    fila["E_fforms"] = owa(elegidos(TE, predE), base_te)

    # ── F · FFORMPP: predecir el error de CADA modelo y quedarse con el menor
    modelos = sorted(TR.modelo.unique())
    pred = np.full((len(Xte), len(modelos)), np.nan)
    for j, m in enumerate(modelos):
        sub = TR[TR.modelo == m].drop_duplicates("serie").set_index("serie")
        y = np.log1p(sub.smape.reindex(Xtr.index))       # log: el sMAPE tiene cola larga
        ok = y.notna().values
        rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                   random_state=SEMILLA_RF, n_jobs=-1)
        rf.fit(Xtr.values[ok], y.values[ok])
        pred[:, j] = rf.predict(Xte.values)
    predF = dict(zip(Xte.index, [modelos[i] for i in pred.argmin(axis=1)]))
    fila["F_fformpp"] = owa(elegidos(TE, predF), base_te)

    # ── D · oraculo
    orac = TE.loc[TE.groupby("serie").smape.idxmin()]
    fila["D_oraculo"] = owa(orac, base_te)
    return fila


def main():
    d, ATR, n_mod = cargar()
    print(f"{d.serie.nunique():,} series x {n_mod} modelos x {len(ATR)} atributos")
    print(f"{N_PART} particiones\n")

    R = pd.DataFrame([una_particion(d, ATR, s) for s in range(N_PART)])
    R.to_excel(AQUI / "metaaprendizaje_resultados.xlsx", index=False)

    NOM = {"A_fijo": "A · modelo fijo", "B_frecuencia": "B · por frecuencia",
           "C_indice": "C · frecuencia + índice (2 estratos)",
           "E_fforms": "E · clasificador sobre 29 atributos",
           "F_fformpp": "F · regresor por modelo, 29 atributos",
           "D_oraculo": "D · oráculo (cota)"}
    print("═══ OWA EN PRUEBA ═══")
    print(f"{'política':40s} {'media':>7s} {'de':>7s}")
    for c, nom in NOM.items():
        print(f"{nom:40s} {R[c].mean():7.4f} {R[c].std():7.4f}")

    from scipy.stats import wilcoxon
    print("\n═══ ¿EL META-APRENDIZ COMPLETO LE GANA AL ÍNDICE? ═══")
    for c, nom in (("E_fforms", "clasificador"), ("F_fformpp", "regresor por modelo")):
        dif = R.C_indice - R[c]        # positivo = el meta-aprendiz es mejor
        p = wilcoxon(R[c], R.C_indice)[1]
        gana = "el meta-aprendiz" if dif.mean() > 0 else "el índice"
        print(f"  vs {nom:22s} dif {dif.mean():+.4f}  gana {gana:17s} "
              f"({(dif>0).sum()}/{len(R)})  p={p:.3g}")

    h = R.A_fijo - R.D_oraculo
    print("\n  del hueco entre el modelo fijo y el oráculo, cada política recorre:")
    for c, nom in NOM.items():
        if c in ("A_fijo", "D_oraculo"):
            continue
        print(f"    {nom:40s} {100*((R.A_fijo - R[c])/h).mean():5.1f} %")

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    cols = ["A_fijo", "B_frecuencia", "C_indice", "E_fforms", "F_fformpp", "D_oraculo"]
    bp = ax.boxplot([R[c] for c in cols], widths=.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=1.1))
    for b, c in zip(bp["boxes"], [GRIS, AZUL, VERDE, MORADO, NARANJA, "#BFBFBF"]):
        b.set(facecolor=c, alpha=.65, linewidth=.7)
    ax.set_xticklabels(["A\nfijo", "B\nfrecuencia", "C\níndice", "E\nclasific.",
                        "F\nregresor", "D\noráculo"], fontsize=7.8)
    ax.axhline(1.0, color=NARANJA, ls="--", lw=1, zorder=0)
    ax.text(.06, 1.0, "Naive2", color=NARANJA, fontsize=7.5, va="bottom")
    ax.set_ylabel("OWA en prueba")
    ax.set_title(f"Índice frente a meta-aprendizaje sobre 29 atributos · "
                 f"{N_PART} particiones", loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig12_metaaprendizaje.{ext}")
    plt.close(fig)
    print("\nfiguras/fig12_metaaprendizaje.pdf\nmetaaprendizaje_resultados.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
