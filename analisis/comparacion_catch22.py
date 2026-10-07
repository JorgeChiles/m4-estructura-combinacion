#!/usr/bin/env python3
"""
El indice de 29 descriptores frente a catch22.

La objecion que responde
------------------------
Los tres revisores piden lo mismo: los 29 descriptores se eligieron por
interpretabilidad y cobertura de familias, no por un procedimiento de seleccion.
catch22 (Lubba et al., 2019) es el conjunto canonico: 22 caracteristicas
filtradas de las 4.791 de hctsa por desempeno de clasificacion y redundancia
mutua sobre 93 colecciones. Si un indice construido sobre catch22 rindiera igual
o mejor, el conjunto propio no estaria justificado.

Que se compara
--------------
Se construye un indice ANALOGO sobre catch22 --negativo de la primera componente
principal sobre las 22 caracteristicas estandarizadas-- y se lo somete a las
mismas tres pruebas que el indice del articulo:

    1. correlacion con el error del pronostico ingenuo
    2. utilidad para enrutar: mejor modelo por (frecuencia x estrato)
    3. utilidad para combinar: promedio de los k mejores por estrato

La tercera es la que importa, porque es donde el articulo situa el aporte de la
caracterizacion.

Sobre el signo
--------------
El signo de una componente principal es arbitrario. Para el indice de catch22 se
fija con el mismo criterio que el propio: se orienta de modo que correlacione
POSITIVAMENTE con el error del ingenuo, y se declara. Como aca la comparacion es
justamente contra el error, se usa solo la mitad de entrenamiento de cada
particion para fijar el signo y no se toca la de prueba.

Salidas:
    catch22_features.csv
    comparacion_catch22.xlsx

Uso:  python3 comparacion_catch22.py [n_particiones]
"""
import importlib
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
MOD = AQUI / "TRANSPOSE_1000_RANDOM"
RES = AQUI / "m4-structural-complexity" / "results"
sys.path.insert(0, str(AQUI / "m4-structural-complexity" / "src"))
M4 = importlib.import_module("06_m4_metrics")
DATA = AQUI / "m4-structural-complexity" / "data"

N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 30
KS = (1, 2, 3, 4, 5)
FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}
FUERA = {"naive", "snaive", "Holt-Winters"}


def calcular_catch22():
    """22 caracteristicas por serie, sobre las series evaluadas."""
    import pycatch22
    f_out = AQUI / "catch22_features.csv"
    if f_out.exists():
        print(f"── catch22 ya calculado: {f_out.name}", flush=True)
        return pd.read_csv(f_out).set_index("serie")
    P = pd.read_pickle(AQUI / "pronosticos_modelos.pkl.gz")
    filas, t0 = [], time.time()
    for f in FRECS:
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        series = sorted(set(P[P.frecuencia == f].serie.unique()) & set(tr))
        for i, sid in enumerate(series):
            y = np.asarray(tr[sid], float)
            y = y[np.isfinite(y)]
            if len(y) < 20:
                continue
            r = pycatch22.catch22_all(list(y), catch24=False)
            filas.append(dict(serie=sid, frecuencia=f,
                              **dict(zip(r["names"], r["values"]))))
        print(f"   {f:10s} {len(series):5,} series  [{time.time()-t0:.0f} s]", flush=True)
    F = pd.DataFrame(filas)
    F.to_csv(f_out, index=False)
    print(f"── {len(F):,} series x 22 caracteristicas -> {f_out.name}", flush=True)
    return F.set_index("serie")


def indice_desde(X, err_train, mask_train):
    """-PC1 estandarizado, con el signo fijado SOLO con entrenamiento."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))
    pc1 = PCA(n_components=1).fit_transform(Z).ravel()
    # signo: que correlacione positivamente con el error, decidido en train
    r = np.corrcoef(pc1[mask_train], err_train)[0, 1]
    return pc1 if (np.isfinite(r) and r > 0) else -pc1


def cargar():
    P = pd.read_pickle(AQUI / "pronosticos_modelos.pkl.gz")
    P = P[~P.modelo.astype(str).isin(FUERA)]
    C = calcular_catch22()
    cols22 = [c for c in C.columns if c != "frecuencia"]
    prop = pd.read_excel(RES / "df_features_complexity.xlsx",
                         usecols=["serie", "complexity_index"]).set_index("serie")

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
                if s in tr and s in te and s in C.index and s in prop.index]
        series = [s for s in cand
                  if len(w.loc[s]) >= h and w.loc[s].iloc[:h].notna().all().all()]
        n = len(series)
        A = np.empty((n, len(modelos), h)); R = np.empty((n, h)); E = np.empty(n)
        for i, s in enumerate(series):
            A[i] = w.loc[s].iloc[:h][modelos].values.T
            R[i] = np.asarray(te[s], float)[:h]
            y = np.asarray(tr[s], float)
            E[i] = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
        D[f] = dict(series=np.array(series), modelos=np.array(modelos), P=A,
                    real=R, escala=E, X22=C.loc[series, cols22].values.astype(float),
                    ci=prop.loc[series, "complexity_index"].values,
                    i_n2=modelos.index("naive2"))
        print(f"── {f:10s} {n:5,} series", flush=True)
    return D


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
    return D


def ranking(d, filas):
    j2 = d["i_n2"]
    bs, bm = d["SM"][filas, j2].mean(), d["MA"][filas, j2].mean()
    return np.argsort(.5 * (d["SM"][filas].mean(axis=0) / bs +
                            d["MA"][filas].mean(axis=0) / bm))


def una_particion(D, semilla):
    rng = np.random.default_rng(semilla)
    acc, fila = {}, {"semilla": semilla}

    def guardar(nombre, sm, ma):
        acc.setdefault(nombre, []).append((sm, ma))

    for f, d in D.items():
        n = len(d["series"])
        tr = rng.random(n) < .5
        te = ~tr
        guardar("naive2", d["SM"][te, d["i_n2"]], d["MA"][te, d["i_n2"]])

        idx22 = indice_desde(d["X22"], d["SM"][tr, d["i_n2"]], tr)
        for nom, ci in (("propio", d["ci"]), ("catch22", idx22)):
            corte = np.median(ci[tr])
            e_tr = [tr & (ci <= corte), tr & (ci > corte)]
            e_te = [te & (ci <= corte), te & (ci > corte)]
            for k in KS:
                sa, ma_ = [], []
                for mtr, mte in zip(e_tr, e_te):
                    if mtr.sum() < 5 or mte.sum() == 0:
                        continue
                    sel = ranking(d, mtr)[:k]
                    p = d["P"][:, sel, :].mean(axis=1)
                    a, b = metricas(p[mte], d["real"][mte], d["escala"][mte])
                    sa.append(a); ma_.append(b)
                if sa:
                    guardar(f"{nom}_{k}", np.concatenate(sa), np.concatenate(ma_))

    n2s = np.concatenate([a for a, _ in acc["naive2"]])
    n2m = np.concatenate([b for _, b in acc["naive2"]])
    for nombre, lst in acc.items():
        sm = np.concatenate([a for a, _ in lst]); ma = np.concatenate([b for _, b in lst])
        fila[nombre] = .5 * (sm.mean() / n2s.mean() + ma.mean() / n2m.mean())
    return fila


def main():
    D = precalcular(cargar())

    # ── 1 · correlacion con el error del ingenuo
    print("\n═══ 1 · CORRELACION CON EL ERROR DEL PRONOSTICO INGENUO ═══\n")
    from scipy.stats import pearsonr, spearmanr
    filas = []
    for f, d in D.items():
        err = d["SM"][:, d["i_n2"]]
        todo = np.ones(len(err), bool)
        i22 = indice_desde(d["X22"], err, todo)
        for nom, ci in (("propio", d["ci"]), ("catch22", i22)):
            ok = np.isfinite(ci) & np.isfinite(err)
            filas.append(dict(frecuencia=f, indice=nom,
                              pearson=pearsonr(ci[ok], err[ok])[0],
                              spearman=spearmanr(ci[ok], err[ok])[0]))
    CO = pd.DataFrame(filas).pivot(index="frecuencia", columns="indice",
                                   values=["pearson", "spearman"])
    print(CO.round(3).to_string())

    # ── 2 y 3 · enrutar y combinar
    print(f"\n═══ 2 y 3 · ENRUTAR Y COMBINAR, {N_PART} PARTICIONES ═══\n")
    R = pd.DataFrame([una_particion(D, s) for s in range(N_PART)])
    R.to_excel(AQUI / "comparacion_catch22.xlsx", index=False)
    from scipy.stats import wilcoxon
    print(f"{'k':>3s} {'indice propio':>15s} {'catch22':>10s} {'diferencia':>12s} "
          f"{'propio mejor':>13s} {'p':>10s}")
    for k in KS:
        a, b = R[f"propio_{k}"], R[f"catch22_{k}"]
        print(f"{k:3d} {a.mean():15.4f} {b.mean():10.4f} {b.mean()-a.mean():+12.4f} "
              f"{(a<b).sum():10d}/{len(R)} {wilcoxon(a,b).pvalue:10.3g}")
    print("\n(k=1 es enrutar eligiendo; k>=2 es combinar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
