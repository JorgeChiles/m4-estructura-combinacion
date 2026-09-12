#!/usr/bin/env python3
"""
¿La conclusión sobre el régimen sobrevive al error de muestreo?

La pregunta
-----------
La Seccion 7.5 concluye a partir de una diferencia de OWA entre dos politicas
--elegir el mejor modelo contra combinar los cinco mejores-- ponderada por la
poblacion de M4. Ahi mensual pesa el 48 %, y de mensual se evaluaron 1.000 de
48.000 series: el 2,1 %.

O sea que casi la mitad del numero que decide el regimen descansa sobre una
muestra chica. El bootstrap que ya existe (`comparacion_m4.py`) mide la
incertidumbre del NIVEL del OWA; este mide la de la DIFERENCIA entre politicas,
que es la cantidad de la que depende la conclusion.

Diseño
------
En cada replica se remuestrean series CON REEMPLAZO dentro de cada frecuencia y
se corre el protocolo completo sobre esa replica: mitad para aprender la
politica, mitad para evaluar.

La particion se PROMEDIA sobre N_PART repeticiones dentro de cada replica, no se
sortea una sola vez. Esto importa: el numero que el articulo reporta es un
promedio sobre 30 particiones, asi que un bootstrap de una particion por replica
mide la incertidumbre de otra cantidad --una corrida suelta-- y sobrestima la del
numero publicado. Con una sola particion el ruido de particion aportaba el 57 %
de la varianza en el banco con N-BEATS; promediando queda en lo que corresponde.

Semanal y horaria no se remuestrean: ahi la muestra ES la poblacion (359 de 359
y 414 de 414) y no hay error de muestreo que simular.

Diagnostico: mas series mensuales, ¿cerrarian el intervalo?
-----------------------------------------------------------
Con --congelar Monthly se anula el error de muestreo de esa frecuencia, que es
el limite de evaluar sus 48.000 series. Si el intervalo sigue conteniendo el
cero, entonces la indeterminacion no es falta de muestra y no hay computo que la
cierre. Es el caso: [-0.0122, +0.0050] con mensual congelado.

Salidas:
    incertidumbre_regimen.xlsx

Uso:  python3 incertidumbre_regimen.py [n_replicas] [--congelar Frec[,Frec...]]
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

_arg = [a for a in sys.argv[1:] if not a.startswith("--")]
N_REP = int(_arg[0]) if _arg else 600
N_PART = 30                               # particiones promediadas por replica
FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}
POBLACION = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000,
             "Weekly": 359, "Daily": 4227, "Hourly": 414}
COMPLETAS = {"Weekly", "Hourly"}          # la muestra es la poblacion
if "--congelar" in sys.argv:              # diagnostico: anular su error de muestreo
    COMPLETAS = COMPLETAS | set(sys.argv[sys.argv.index("--congelar") + 1].split(","))
FUERA = {"naive", "snaive", "Holt-Winters"}
K_COMB = 5


def cargar(archivo):
    P = pd.read_pickle(AQUI / archivo)
    P = P[~P.modelo.astype(str).isin(FUERA)]
    D = {}
    for f, (h, m) in FRECS.items():
        sub = P[P.frecuencia == f]
        w = sub.pivot_table(index=["serie", "h"], columns="modelo", values="pred")
        modelos = [c for c in w.columns if w[c].notna().any()]
        w = w[modelos]
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        te = M4.load_m4(DATA / f"{f}-test.csv")
        series = [s for s in w.index.get_level_values(0).unique()
                  if s in tr and s in te
                  and len(w.loc[s]) >= h and w.loc[s].iloc[:h].notna().all().all()]
        n = len(series)
        A = np.empty((n, len(modelos), h)); R = np.empty((n, h)); E = np.empty(n)
        for i, s in enumerate(series):
            A[i] = w.loc[s].iloc[:h][modelos].values.T
            R[i] = np.asarray(te[s], float)[:h]
            y = np.asarray(tr[s], float)
            E[i] = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
        d = dict(P=A, real=R, escala=E, i_n2=modelos.index("naive2"), n=n)
        SM = np.empty((n, len(modelos))); MA = np.empty((n, len(modelos)))
        for j in range(len(modelos)):
            SM[:, j], MA[:, j] = metricas(A[:, j, :], R, E)
        d["SM"], d["MA"] = SM, MA
        D[f] = d
    return D


def metricas(pred, real, escala):
    den = np.abs(real) + np.abs(pred)
    sm = 200.0 * np.where(den > 0, np.abs(real - pred) / np.where(den > 0, den, 1),
                          0).mean(axis=1)
    return sm, np.abs(real - pred).mean(axis=1) / escala


def una_particion(D, rng, w, fijo):
    """El protocolo sobre una particion de una remuestra ya sorteada."""
    S = {"elegir": 0.0, "combinar": 0.0}
    M_ = {"elegir": 0.0, "combinar": 0.0}
    bS = bM = 0.0
    for f, d in D.items():
        idx = fijo[f]
        mitad = rng.random(len(idx)) < .5
        tr, te = idx[mitad], idx[~mitad]
        if len(tr) < 10 or len(te) < 10:
            return None
        j2 = d["i_n2"]
        bs, bm = d["SM"][tr, j2].mean(), d["MA"][tr, j2].mean()
        o = .5 * (d["SM"][tr].mean(axis=0) / bs + d["MA"][tr].mean(axis=0) / bm)
        orden = np.argsort(o)
        bS += w[f] * d["SM"][te, j2].mean(); bM += w[f] * d["MA"][te, j2].mean()
        for nom, k in (("elegir", 1), ("combinar", K_COMB)):
            p = d["P"][:, orden[:k], :].mean(axis=1)
            sm, ma = metricas(p[te], d["real"][te], d["escala"][te])
            S[nom] += w[f] * sm.mean(); M_[nom] += w[f] * ma.mean()
    return {nom: .5 * (S[nom] / bS + M_[nom] / bM) for nom in S}


def una_replica(D, rng, w):
    """Una remuestra de series, promediada sobre N_PART particiones.

    Se sortean las series UNA vez y luego se promedia el protocolo sobre varias
    particiones, que es exactamente la cantidad que el articulo reporta.
    """
    fijo = {f: (np.arange(d["n"]) if f in COMPLETAS
                else rng.integers(0, d["n"], d["n"])) for f, d in D.items()}
    acu = []
    for _ in range(N_PART):
        r = una_particion(D, rng, w, fijo)
        if r is None:
            return None
        acu.append(r)
    return {nom: float(np.mean([a[nom] for a in acu])) for nom in ("elegir", "combinar")}


def main():
    w = pd.Series(POBLACION); w = w / w.sum()
    filas = []
    for etiqueta, archivo in (("banco clasico", "pronosticos_modelos.pkl.gz"),
                              ("con N-BEATS", "pronosticos_modelos_con_nbeats.pkl.gz")):
        if not (AQUI / archivo).exists():
            print(f"── falta {archivo}"); continue
        D = cargar(archivo)
        rng = np.random.default_rng(20260912)
        reps = [r for r in (una_replica(D, rng, w) for _ in range(N_REP)) if r]
        el = np.array([r["elegir"] for r in reps])
        co = np.array([r["combinar"] for r in reps])
        dif = co - el                       # positivo = combinar es PEOR
        lo, hi = np.percentile(dif, [2.5, 97.5])
        cruza = lo <= 0 <= hi
        congelado = COMPLETAS - {"Weekly", "Hourly"}
        print(f"\n═══ {etiqueta.upper()} ═══  ({len(reps)} réplicas × {N_PART} particiones"
              + (f", congelado: {', '.join(sorted(congelado))}" if congelado else "") + ")")
        print(f"  elegir uno      {el.mean():.4f}  [{np.percentile(el,2.5):.4f}, "
              f"{np.percentile(el,97.5):.4f}]")
        print(f"  combinar cinco  {co.mean():.4f}  [{np.percentile(co,2.5):.4f}, "
              f"{np.percentile(co,97.5):.4f}]")
        print(f"  diferencia      {dif.mean():+.4f}  IC 95 % [{lo:+.4f}, {hi:+.4f}]")
        if cruza:
            veredicto = "EL INTERVALO CRUZA CERO: no se puede afirmar cuál gana"
        else:
            veredicto = "gana elegir" if dif.mean() > 0 else "gana combinar"
        print(f"  -> {veredicto}   (combinar mejor en "
              f"{100*(dif<0).mean():.0f} % de las réplicas)")
        filas.append(dict(banco=etiqueta, elegir=el.mean(), combinar=co.mean(),
                          dif=dif.mean(), ic_lo=lo, ic_hi=hi, cruza_cero=cruza,
                          replicas=len(reps)))
    pd.DataFrame(filas).to_excel(AQUI / "incertidumbre_regimen.xlsx", index=False)
    print("\nincertidumbre_regimen.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
