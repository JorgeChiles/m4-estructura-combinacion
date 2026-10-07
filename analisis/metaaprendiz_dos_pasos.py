#!/usr/bin/env python3
"""
El indice de dos estratos contra un meta-aprendiz de dos pasos (estilo Vaiciukynas).

La objecion que responde
------------------------
Los revisores piden comparar el enrutamiento por indice contra meta-aprendices
publicados sobre M4. Dos son pertinentes:

    Vaiciukynas et al. (2021)  dos pasos: un meta-aprendiz ORDENA los metodos y
                               otro predice el TAMANO del ensamble, por serie
    Di Gangi (2022)            combinacion convexa dispersa aprendida

Por que no se citan sus numeros y ya
------------------------------------
Vaiciukynas reporta sMAPE 9.21 sobre M4, pero evaluado en 12.561 series
MICROECONOMICAS, no en las 100.000: su Theta de referencia da 11.05 cuando el
Theta oficial de M4 da 12.309. Son escalas distintas y ponerlas en la misma
tabla seria enganoso, igual que con FFORMA.

Lo que si se puede: implementar el ENFOQUE sobre los mismos datos, el mismo pool
y el mismo protocolo, que es la comparacion que de verdad informa.

    H1  un regresor por modelo predice su error -> se promedian los k mejores,
        con k FIJO. Es FFORMPP extendido a combinacion.
    H2  ademas, un segundo regresor predice el k OPTIMO de cada serie.
        Es la propuesta de dos pasos.

Contra:

    G_k      el articulo: los k mejores de cada (frecuencia x estrato)
    Gk_auto  igual, con k elegido en entrenamiento por frecuencia

La pregunta: ¿aprender el orden y el tamano POR SERIE, con los 29 atributos,
supera a un escalar con dos estratos?

Salidas:
    metaaprendiz_dos_pasos.xlsx
    figuras/fig14_dos_pasos.pdf

Uso:  python3 metaaprendiz_dos_pasos.py [n_particiones]
"""
import importlib
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
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
MOD = AQUI / "TRANSPOSE_1000_RANDOM"
RES = AQUI / "m4-structural-complexity" / "results"
OUT = AQUI / "figuras"; OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(AQUI / "m4-structural-complexity" / "src"))
M4 = importlib.import_module("06_m4_metrics")
DATA = AQUI / "m4-structural-complexity" / "data"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": .02,
})
AZUL, NARANJA, VERDE, MORADO, GRIS = "#0072B2", "#D55E00", "#009E73", "#7B3294", "#4D4D4D"

N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 20
KS = (1, 2, 3, 4, 5)
FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}
FUERA = {"naive", "snaive", "Holt-Winters"}
SEMILLA_RF = 42


def cargar():
    """Pronosticos, reales, escala del MASE y los 29 atributos, por frecuencia."""
    P = pd.read_pickle(AQUI / "pronosticos_modelos.pkl.gz")
    P = P[~P.modelo.astype(str).isin(FUERA)]
    L = pd.read_excel(RES / "pc1_loadings.xlsx")
    ATR = L.feature.tolist()
    F = pd.read_excel(RES / "df_features_complexity.xlsx",
                      usecols=["serie", "complexity_index"] + ATR).set_index("serie")

    D = {}
    for f, (h, m) in FRECS.items():
        sub = P[P.frecuencia == f]
        if sub.empty:
            continue
        w = sub.pivot_table(index=["serie", "h"], columns="modelo", values="pred")
        modelos = [c for c in w.columns if w[c].notna().any()]
        w = w[modelos]
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        te = M4.load_m4(DATA / f"{f}-test.csv")
        cand = [s for s in w.index.get_level_values(0).unique()
                if s in tr and s in te and s in F.index]
        series = [s for s in cand
                  if len(w.loc[s]) >= h and w.loc[s].iloc[:h].notna().all().all()]
        n = len(series)
        A = np.empty((n, len(modelos), h)); R = np.empty((n, h)); E = np.empty(n)
        for i, s in enumerate(series):
            A[i] = w.loc[s].iloc[:h][modelos].values.T
            R[i] = np.asarray(te[s], float)[:h]
            y = np.asarray(tr[s], float)
            E[i] = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
        # Los faltantes se imputan POR COLUMNA. trend_strength y
        # seasonal_strength son NaN en las 23.000 anuales porque STL no aplica
        # con m=1; rellenarlas con una mediana global mezclaria escalas que van
        # de -0,7 a 12 y destruiria los atributos, no solo los faltantes.
        X = F.loc[series, ATR].values.astype(float)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X = np.where(np.isnan(X), med, X)
        D[f] = dict(series=np.array(series), modelos=np.array(modelos), P=A,
                    real=R, escala=E, X=X,
                    ci=F.loc[series, "complexity_index"].values,
                    i_n2=modelos.index("naive2"))
        print(f"── {f:10s} {n:5,} series x {len(modelos)} modelos x {len(ATR)} atributos",
              flush=True)
    return D, ATR


def metricas(pred, real, escala):
    den = np.abs(real) + np.abs(pred)
    sm = 200.0 * np.where(den > 0, np.abs(real - pred) / np.where(den > 0, den, 1),
                          0).mean(axis=1)
    return sm, np.abs(real - pred).mean(axis=1) / escala


def precalcular(D):
    for f, d in D.items():
        n, k, h = d["P"].shape
        SM = np.empty((n, k)); MA = np.empty((n, k))
        for j in range(k):
            SM[:, j], MA[:, j] = metricas(d["P"][:, j, :], d["real"], d["escala"])
        d["SM"], d["MA"] = SM, MA
        # sMAPE de promediar los j+1 mejores segun un orden dado: se recalcula
        # por particion, aca solo se reserva
    return D


def ranking(d, filas):
    j2 = d["i_n2"]
    bs, bm = d["SM"][filas, j2].mean(), d["MA"][filas, j2].mean()
    o = .5 * (d["SM"][filas].mean(axis=0) / bs + d["MA"][filas].mean(axis=0) / bm)
    return np.argsort(o)


def error_de_orden(d, orden_por_serie, k, filas):
    """Promedia los k primeros del orden PROPIO de cada serie."""
    idx = np.where(filas)[0]
    sel = orden_por_serie[idx, :k]                       # (n_sel, k)
    pred = np.take_along_axis(d["P"][idx], sel[:, :, None], axis=1).mean(axis=1)
    return metricas(pred, d["real"][idx], d["escala"][idx])


def una_particion(D, semilla):
    rng = np.random.default_rng(semilla)
    acc = {}
    fila = {"semilla": semilla}

    def guardar(nombre, sm, ma):
        acc.setdefault(nombre, []).append((sm, ma))

    # la particion de TODAS las frecuencias se sortea antes, porque el
    # meta-aprendiz del paso 1 se entrena sobre el conjunto agrupado
    mascaras = {g: rng.random(len(D[g]["series"])) < .5 for g in D}

    for f, d in D.items():
        n = len(d["series"])
        tr = mascaras[f]
        te = ~tr
        guardar("naive2", d["SM"][te, d["i_n2"]], d["MA"][te, d["i_n2"]])
        nm = d["P"].shape[1]

        # ── referencia del articulo: k mejores por (frecuencia x estrato)
        corte = np.median(d["ci"][tr])
        estr_tr = [tr & (d["ci"] <= corte), tr & (d["ci"] > corte)]
        estr_te = [te & (d["ci"] <= corte), te & (d["ci"] > corte)]
        for k in KS:
            sa, ma_ = [], []
            for mtr, mte in zip(estr_tr, estr_te):
                if mtr.sum() < 5 or mte.sum() == 0:
                    continue
                sel = ranking(d, mtr)[:k]
                p = d["P"][:, sel, :].mean(axis=1)
                a, b = metricas(p[mte], d["real"][mte], d["escala"][mte])
                sa.append(a); ma_.append(b)
            if sa:
                guardar(f"G{k}", np.concatenate(sa), np.concatenate(ma_))

        # ── paso 1: un regresor por modelo predice su error -> orden por serie
        # El meta-aprendiz se entrena AGRUPANDO las seis frecuencias, como en el
        # metodo publicado: con ~500 series por frecuencia el bosque no tiene
        # datos suficientes, y entrenarlo por separado cuesta 0,013 de OWA.
        pred_err = np.empty((n, nm))
        for j in range(nm):
            Xg = np.vstack([D[g]["X"][mascaras[g]] for g in D])
            yg = np.concatenate([np.log1p(np.clip(D[g]["SM"][mascaras[g], j], 0, None))
                                 for g in D])
            rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                       random_state=SEMILLA_RF, n_jobs=-1)
            rf.fit(Xg, yg)
            pred_err[:, j] = rf.predict(d["X"])
        orden = np.argsort(pred_err, axis=1)

        for k in KS:
            a, b = error_de_orden(d, orden, k, te)
            guardar(f"H1_{k}", a, b)

        # ── paso 2: un regresor predice el k optimo de cada serie
        # objetivo: el k que minimiza el error real al promediar el top-k
        # PREDICHO. Se calcula solo sobre entrenamiento.
        err_k = np.empty((n, len(KS)))
        for ik, k in enumerate(KS):
            sel = orden[:, :k]
            p = np.take_along_axis(d["P"], sel[:, :, None], axis=1).mean(axis=1)
            err_k[:, ik] = metricas(p, d["real"], d["escala"])[0]
        k_opt = np.array(KS)[err_k.argmin(axis=1)]
        rk = RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                   random_state=SEMILLA_RF, n_jobs=-1)
        rk.fit(d["X"][tr], k_opt[tr])
        k_hat = np.clip(np.rint(rk.predict(d["X"])).astype(int), min(KS), max(KS))

        idx = np.where(te)[0]
        p = np.empty((len(idx), d["P"].shape[2]))
        for r, i in enumerate(idx):
            p[r] = d["P"][i, orden[i, :k_hat[i]], :].mean(axis=0)
        a, b = metricas(p, d["real"][idx], d["escala"][idx])
        guardar("H2_dos_pasos", a, b)
        fila[f"k_medio_{f}"] = float(k_hat[idx].mean())

    n2s = np.concatenate([a for a, _ in acc["naive2"]])
    n2m = np.concatenate([b for _, b in acc["naive2"]])
    for nombre, lst in acc.items():
        sm = np.concatenate([a for a, _ in lst]); ma = np.concatenate([b for _, b in lst])
        fila[nombre] = .5 * (sm.mean() / n2s.mean() + ma.mean() / n2m.mean())
    return fila


def main():
    D, ATR = cargar()
    D = precalcular(D)
    print(f"\n{N_PART} particiones, mitad entrenamiento / mitad prueba\n", flush=True)
    R = pd.DataFrame([una_particion(D, s) for s in range(N_PART)])
    R.to_excel(AQUI / "metaaprendiz_dos_pasos.xlsx", index=False)

    from scipy.stats import wilcoxon
    print("═══ EL INDICE CONTRA EL META-APRENDIZ, A IGUAL k ═══\n")
    print(f"{'k':>3s} {'G (indice, 2 estratos)':>23s} {'H1 (29 atributos)':>19s} "
          f"{'a favor del indice':>19s} {'p':>10s}")
    for k in KS:
        g, h = R[f"G{k}"], R[f"H1_{k}"]
        print(f"{k:3d} {g.mean():23.4f} {h.mean():19.4f} {h.mean()-g.mean():+19.4f} "
              f"{wilcoxon(g, h).pvalue:10.3g}")

    print("\n═══ DOS PASOS: ORDEN Y TAMANO APRENDIDOS POR SERIE ═══\n")
    mejor_g = min(KS, key=lambda k: R[f"G{k}"].mean())
    print(f"  G{mejor_g} (mejor del articulo)      {R[f'G{mejor_g}'].mean():.4f} "
          f"(de {R[f'G{mejor_g}'].std():.4f})")
    print(f"  H2 dos pasos (Vaiciukynas)   {R.H2_dos_pasos.mean():.4f} "
          f"(de {R.H2_dos_pasos.std():.4f})")
    dif = R[f"G{mejor_g}"] - R.H2_dos_pasos
    print(f"  diferencia {dif.mean():+.4f}   "
          f"{'gana el meta-aprendiz' if dif.mean() > 0 else 'gana el indice'} en "
          f"{(dif > 0).sum() if dif.mean() > 0 else (dif < 0).sum()}/{len(R)}   "
          f"p = {wilcoxon(R.H2_dos_pasos, R[f'G{mejor_g}']).pvalue:.3g}")
    kk = [c for c in R.columns if c.startswith("k_medio_")]
    print("\n  tamano medio de ensamble que predice el paso 2:")
    for c in kk:
        print(f"    {c.replace('k_medio_',''):10s} {R[c].mean():.2f}")

    figura(R, mejor_g)
    print("\nmetaaprendiz_dos_pasos.xlsx")
    return 0


def figura(R, mejor_g):
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ks = list(KS)
    ax.errorbar(ks, [R[f"G{k}"].mean() for k in ks], yerr=[R[f"G{k}"].std() for k in ks],
                marker="o", ms=4.5, lw=1.4, capsize=3, color=NARANJA,
                label="índice, 2 estratos")
    ax.errorbar(ks, [R[f"H1_{k}"].mean() for k in ks],
                yerr=[R[f"H1_{k}"].std() for k in ks],
                marker="s", ms=4.5, lw=1.4, capsize=3, color=MORADO,
                label="meta-aprendiz, 29 atributos por serie")
    ax.axhline(R.H2_dos_pasos.mean(), color=VERDE, ls="--", lw=1.2)
    ax.text(ks[-1], R.H2_dos_pasos.mean(), "  dos pasos", color=VERDE, fontsize=7.5,
            va="center")
    ax.set_xticks(ks)
    ax.set_xlabel("modelos combinados $k$")
    ax.set_ylabel("OWA en prueba")
    ax.set_title("Ordenar por estrato frente a ordenar por serie", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.8)
    ax.grid(axis="x", visible=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig14_dos_pasos.{ext}")
    plt.close(fig)
    print("figuras/fig14_dos_pasos.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
