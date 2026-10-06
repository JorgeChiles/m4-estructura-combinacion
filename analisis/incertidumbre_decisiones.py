#!/usr/bin/env python3
"""
¿Cual de las dos decisiones pesa mas: combinar o estratificar?

La pregunta
-----------
El articulo afirma que la decision de combinar pesa mas que la de caracterizar
(0,026 contra 0,023). Esa comparacion nunca tuvo intervalo: se apoyaba en dos
medias sobre 30 particiones y en un Wilcoxon pareado.

El Wilcoxon responde otra cosa. Dice si DENTRO de una particion una ganancia
supera a la otra de forma consistente, y da p chiquisimos porque las particiones
comparten las mismas series. No dice si la diferencia sobrevive a que se
hubieran sorteado otras series, que es de lo que depende la afirmacion.

Diseño
------
El mismo que `incertidumbre_regimen.py`, que ya se valido:

  - se remuestrean series CON REEMPLAZO dentro de cada frecuencia;
  - dentro de cada replica se PROMEDIA sobre N_PART particiones, porque el
    numero que el articulo reporta es un promedio sobre particiones y un
    bootstrap de una sola mediria la incertidumbre de una corrida suelta;
  - se pondera por la poblacion de M4, que es invariante a cuantas series se
    hayan evaluado de cada frecuencia;
  - semanal y horaria no se remuestrean: ahi la muestra ES la poblacion.

Las dos cantidades que se comparan, ambas en OWA:

    ganancia de combinar      G1 - G5   (dentro de estratos, de 1 a 5 modelos)
    ganancia de estratificar  B1 - G1   (al elegir uno, de frecuencia a frec x indice)

Salidas:
    incertidumbre_decisiones.xlsx

Uso:  python3 incertidumbre_decisiones.py [n_replicas]
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


def cargar(arch):
    P = pd.read_pickle(AQUI / arch)
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
    """B1, G1 y G5 sobre una particion de una remuestra ya sorteada."""
    acu = {k: {"s": 0., "m": 0.} for k in ("B1", "G1", "G5")}
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

        p = d["P"][:, rank(tr)[:1], :].mean(1)          # B1: solo frecuencia
        a, b = metricas(p[te], d["real"][te], d["escala"][te])
        acu["B1"]["s"] += w[f] * a.mean(); acu["B1"]["m"] += w[f] * b.mean()

        corte = np.median(d["ci"][tr])
        ci_tr, ci_te = d["ci"][tr], d["ci"][te]
        sa = {1: ([], []), 5: ([], [])}
        for mtr, mte in ((tr[ci_tr <= corte], te[ci_te <= corte]),
                         (tr[ci_tr > corte], te[ci_te > corte])):
            if len(mtr) < 5 or len(mte) == 0:
                continue
            orden = rank(mtr)
            for k in (1, 5):
                p = d["P"][:, orden[:k], :].mean(1)
                a, b = metricas(p[mte], d["real"][mte], d["escala"][mte])
                sa[k][0].append(a); sa[k][1].append(b)
        for k in (1, 5):
            if sa[k][0]:
                acu[f"G{k}"]["s"] += w[f] * np.concatenate(sa[k][0]).mean()
                acu[f"G{k}"]["m"] += w[f] * np.concatenate(sa[k][1]).mean()
    return {k: .5 * (acu[k]["s"] / bS + acu[k]["m"] / bM) for k in acu}


def una_replica(D, rng, w):
    fijo = {f: (np.arange(d["n"]) if f in COMPLETAS
                else rng.integers(0, d["n"], d["n"])) for f, d in D.items()}
    acu = []
    for _ in range(N_PART):
        r = una_particion(D, rng, w, fijo)
        if r is None:
            return None
        acu.append(r)
    return {k: float(np.mean([a[k] for a in acu])) for k in ("B1", "G1", "G5")}


def main():
    w = pd.Series(POBLACION); w = w / w.sum()
    filas = []
    for etiqueta, arch in (("1.000 anuales",
                            "respaldo_pre_anuales/pronosticos_modelos.pkl.gz"),
                           ("2.000 anuales", "pronosticos_modelos.pkl.gz")):
        if not (AQUI / arch).exists():
            print(f"── falta {arch}"); continue
        print(f"\n═══ {etiqueta.upper()} ═══")
        D = cargar(arch)
        rng = np.random.default_rng(20261005)
        reps = [r for r in (una_replica(D, rng, w) for _ in range(N_REP)) if r]
        B1 = np.array([r["B1"] for r in reps])
        G1 = np.array([r["G1"] for r in reps])
        G5 = np.array([r["G5"] for r in reps])
        comb, estrat = G1 - G5, B1 - G1
        dif = comb - estrat
        lo, hi = np.percentile(dif, [2.5, 97.5])
        print(f"  ({len(reps)} réplicas × {N_PART} particiones, pesos poblacionales)")
        print(f"  ganancia de combinar      {comb.mean():+.4f}  "
              f"[{np.percentile(comb,2.5):+.4f}, {np.percentile(comb,97.5):+.4f}]")
        print(f"  ganancia de estratificar  {estrat.mean():+.4f}  "
              f"[{np.percentile(estrat,2.5):+.4f}, {np.percentile(estrat,97.5):+.4f}]")
        print(f"  diferencia                {dif.mean():+.4f}  IC 95 % [{lo:+.4f}, {hi:+.4f}]")
        v = ("EMPATAN: no se puede afirmar cuál pesa más" if lo <= 0 <= hi
             else ("combinar pesa más" if dif.mean() > 0 else "estratificar pesa más"))
        print(f"  -> {v}   (combinar mayor en {100*(dif>0).mean():.0f} % de las réplicas)")
        filas.append(dict(muestra=etiqueta, combinar=comb.mean(), estratificar=estrat.mean(),
                          dif=dif.mean(), ic_lo=lo, ic_hi=hi, cruza_cero=bool(lo <= 0 <= hi),
                          replicas=len(reps)))
    pd.DataFrame(filas).to_excel(AQUI / "incertidumbre_decisiones.xlsx", index=False)
    print("\nincertidumbre_decisiones.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
