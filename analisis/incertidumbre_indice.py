#!/usr/bin/env python3
"""
El aporte del índice, con la unidad de remuestreo correcta.

El problema
-----------
El artículo sostiene el aporte del índice con un Wilcoxon pareado sobre 30
particiones y con un bootstrap que remuestrea esas mismas 30 diferencias. Las
dos cosas tratan la PARTICIÓN como unidad, y 30 particiones de las mismas 5.674
series no son 30 muestras independientes: comparten las series. El p se vuelve
anticonservador y el intervalo sale demasiado angosto, porque ninguno de los dos
incorpora la fuente de variación que importa para generalizar, que es haber
sorteado estas series y no otras.

Lo que mide cada cosa
---------------------
No son intercambiables y conviene decir cuál responde qué:

  Wilcoxon sobre particiones   ¿el aporte es CONSISTENTE al re-dividir esta
                               muestra? Es una afirmación de reproducibilidad
                               interna, y como tal es válida.

  Bootstrap sobre series       ¿el aporte SOBREVIVE a que se hubieran sorteado
                               otras series? Es la afirmación de generalización,
                               y es la que el artículo necesita.

Diseño
------
El mismo que `incertidumbre_regimen.py` y `incertidumbre_decisiones.py`:
se remuestrean series CON REEMPLAZO dentro de cada frecuencia, se promedia el
protocolo sobre N_PART particiones dentro de cada réplica --porque el número
reportado es un promedio sobre particiones-- y se pondera por la poblacion de
M4. Semanal y horaria no se remuestrean: ahi la muestra ES la poblacion.

Se reporta, para cada tamano de conjunto k, el aporte del indice
B_k - G_k con su intervalo, y al lado el intervalo que daba el remuestreo de
particiones, para que se vea la diferencia.

Salidas:
    incertidumbre_indice.xlsx

Uso:  python3 incertidumbre_indice.py [n_replicas]
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

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
RES = AQUI / "m4-structural-complexity" / "results"
sys.path.insert(0, str(AQUI / "m4-structural-complexity" / "src"))
M4 = importlib.import_module("06_m4_metrics")
DATA = AQUI / "m4-structural-complexity" / "data"

N_REP = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_PART = 20
KS = (1, 2, 3, 4, 5)
FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}
POBLACION = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000,
             "Weekly": 359, "Daily": 4227, "Hourly": 414}
COMPLETAS = {"Weekly", "Hourly"}
FUERA = {"naive", "snaive", "Holt-Winters"}


def metricas(p, r, e):
    den = np.abs(r) + np.abs(p)
    return (200. * np.where(den > 0, np.abs(r - p) / np.where(den > 0, den, 1), 0).mean(1),
            np.abs(r - p).mean(1) / e)


def cargar():
    P = pd.read_pickle(AQUI / "pronosticos_modelos.pkl.gz")
    P = P[~P.modelo.astype(str).isin(FUERA)]
    idx = pd.read_excel(RES / "df_features_complexity.xlsx",
                        usecols=["serie", "complexity_index"]).set_index("serie")
    D = {}
    for f, (h, m) in FRECS.items():
        sub = P[P.frecuencia == f]
        w = sub.pivot_table(index=["serie", "h"], columns="modelo", values="pred")
        mods = [c for c in w.columns if w[c].notna().any()]
        w = w[mods]
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        te = M4.load_m4(DATA / f"{f}-test.csv")
        ss = [s for s in w.index.get_level_values(0).unique()
              if s in tr and s in te and s in idx.index
              and len(w.loc[s]) >= h and w.loc[s].iloc[:h].notna().all().all()]
        n = len(ss)
        A = np.empty((n, len(mods), h)); R = np.empty((n, h)); E = np.empty(n)
        for i, s in enumerate(ss):
            A[i] = w.loc[s].iloc[:h][mods].values.T
            R[i] = np.asarray(te[s], float)[:h]
            y = np.asarray(tr[s], float)
            E[i] = np.mean(np.abs(y[m:] - y[:-m]))
        SM = np.empty((n, len(mods))); MA = np.empty((n, len(mods)))
        for j in range(len(mods)):
            SM[:, j], MA[:, j] = metricas(A[:, j, :], R, E)
        D[f] = dict(P=A, real=R, escala=E, SM=SM, MA=MA, n=n,
                    i_n2=mods.index("naive2"),
                    ci=idx.loc[ss, "complexity_index"].values)
        print(f"── {f:10s} {n:5,} series", flush=True)
    return D


def una_particion(D, rng, w, fijo):
    """B_k y G_k sobre una partición de una remuestra ya sorteada."""
    acu = {f"{p}{k}": {"s": 0., "m": 0.} for p in "BG" for k in KS}
    bS = bM = 0.
    for f, d in D.items():
        idx = fijo[f]
        mitad = rng.random(len(idx)) < .5
        tr, te = idx[mitad], idx[~mitad]
        if len(tr) < 10 or len(te) < 10:
            return None
        j2 = d["i_n2"]
        bS += w[f] * d["SM"][te, j2].mean(); bM += w[f] * d["MA"][te, j2].mean()

        def rank(filas):
            bs, bm = d["SM"][filas, j2].mean(), d["MA"][filas, j2].mean()
            return np.argsort(.5 * (d["SM"][filas].mean(0) / bs
                                    + d["MA"][filas].mean(0) / bm))

        orden = rank(tr)
        for k in KS:                                   # B: solo frecuencia
            p = d["P"][:, orden[:k], :].mean(1)
            a, b = metricas(p[te], d["real"][te], d["escala"][te])
            acu[f"B{k}"]["s"] += w[f] * a.mean(); acu[f"B{k}"]["m"] += w[f] * b.mean()

        corte = np.median(d["ci"][tr])                 # G: frecuencia x estrato
        ci_tr, ci_te = d["ci"][tr], d["ci"][te]
        sa = {k: ([], []) for k in KS}
        for mtr, mte in ((tr[ci_tr <= corte], te[ci_te <= corte]),
                         (tr[ci_tr > corte], te[ci_te > corte])):
            if len(mtr) < 5 or len(mte) == 0:
                continue
            o = rank(mtr)
            for k in KS:
                p = d["P"][:, o[:k], :].mean(1)
                a, b = metricas(p[mte], d["real"][mte], d["escala"][mte])
                sa[k][0].append(a); sa[k][1].append(b)
        for k in KS:
            if sa[k][0]:
                acu[f"G{k}"]["s"] += w[f] * np.concatenate(sa[k][0]).mean()
                acu[f"G{k}"]["m"] += w[f] * np.concatenate(sa[k][1]).mean()
    return {c: .5 * (acu[c]["s"] / bS + acu[c]["m"] / bM) for c in acu}


def una_replica(D, rng, w):
    fijo = {f: (np.arange(d["n"]) if f in COMPLETAS
                else rng.integers(0, d["n"], d["n"])) for f, d in D.items()}
    acu = []
    for _ in range(N_PART):
        r = una_particion(D, rng, w, fijo)
        if r is None:
            return None
        acu.append(r)
    return {c: float(np.mean([a[c] for a in acu])) for c in acu[0]}


def main():
    w = pd.Series(POBLACION); w = w / w.sum()
    D = cargar()

    # ── (1) remuestreo de SERIES, que es la inferencia correcta
    rng = np.random.default_rng(20261029)
    reps = [r for r in (una_replica(D, rng, w) for _ in range(N_REP)) if r]
    print(f"\n── {len(reps)} réplicas × {N_PART} particiones", flush=True)

    # ── (2) remuestreo de PARTICIONES, que es lo que hacía el artículo
    rng2 = np.random.default_rng(7)
    fijo = {f: np.arange(d["n"]) for f, d in D.items()}      # sin remuestrear series
    parts = [p for p in (una_particion(D, rng2, w, fijo) for _ in range(30)) if p]

    filas = []
    print(f"\n{'k':>2s} {'aporte':>9s} {'IC sobre SERIES':>22s} {'IC sobre particiones':>24s}")
    for k in KS:
        ser = np.array([r[f"B{k}"] - r[f"G{k}"] for r in reps])
        lo, hi = np.percentile(ser, [2.5, 97.5])
        dpar = np.array([p[f"B{k}"] - p[f"G{k}"] for p in parts])
        bp = np.percentile([np.random.default_rng(i).choice(dpar, len(dpar)).mean()
                            for i in range(2000)], [2.5, 97.5])
        cruza = lo <= 0 <= hi
        print(f"{k:2d} {ser.mean():+9.4f}   [{lo:+.4f}, {hi:+.4f}]{'  CRUZA CERO' if cruza else '':>13s}"
              f"   [{bp[0]:+.4f}, {bp[1]:+.4f}]   ancho {(hi-lo)/(bp[1]-bp[0]):.1f}x")
        filas.append(dict(k=k, aporte=ser.mean(), ic_lo=lo, ic_hi=hi, cruza_cero=bool(cruza),
                          ancho_series=hi - lo, ancho_particiones=bp[1] - bp[0],
                          ic_part_lo=bp[0], ic_part_hi=bp[1]))
    pd.DataFrame(filas).to_excel(AQUI / "incertidumbre_indice.xlsx", index=False)
    print("\nincertidumbre_indice.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
