#!/usr/bin/env python3
"""
¿Cuanto se mueve N-BEATS al cambiar la semilla?

Por que hace falta antes de escribir cualquier cifra
----------------------------------------------------
Al construir el modelo aparecio, sin buscarlo, el fenomeno que este trabajo
documenta: dos corridas de N-BEATS mensual con la misma semilla y los mismos
datos, cambiando solo el largo del planificador de la tasa de aprendizaje,
dieron sMAPE de validacion de 30,1 y 19,6. Reportar el resultado de UNA corrida
seria incoherente con el hallazgo central del articulo sobre inestabilidad.

Este guion evalua las corridas de varias semillas y reporta, por frecuencia:

    OWA medio, desvio, rango (max - min)
    si el rango supera la distancia al mejor modelo del banco, en cuyo caso
    NO se puede afirmar que N-BEATS lo supere

Uso:
    NB_ANCLA=reciente NB_SEMILLA=k NB_SUFIJO=_sk python3 entrenar_nbeats.py
    python3 dispersion_nbeats.py _s1 _s2 _s3 ...
"""
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "m4-structural-complexity" / "src"))
M4 = importlib.import_module("06_m4_metrics")
DATA = AQUI / "m4-structural-complexity" / "data"

FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}
POBLACION_M4 = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000,
                "Weekly": 359, "Daily": 4227, "Hourly": 414}
# mejor de los 26 modelos del banco, por frecuencia
MEJOR_BANCO = {"Yearly": 0.9139, "Quarterly": 0.9456, "Monthly": 0.9325,
               "Weekly": 0.8971, "Daily": 0.9791, "Hourly": 0.6260}


def medir(suf):
    f_pron = AQUI / f"nbeats_pronosticos{suf}.csv"
    if not f_pron.exists():
        return None
    N = pd.read_csv(f_pron)
    filas = []
    for f, (h, m) in FRECS.items():
        sub = N[N.frecuencia == f]
        if sub.empty:
            continue
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        te = M4.load_m4(DATA / f"{f}-test.csv")
        for sid, g in sub.groupby("serie"):
            if sid not in tr or sid not in te:
                continue
            g = g.sort_values("h")
            real = np.asarray(te[sid], float)[:h]
            p = g.pred.values[:len(real)]
            if len(p) < len(real) or not np.isfinite(p).all():
                continue
            filas.append(dict(frecuencia=f, serie=sid, semilla=suf,
                              smape=M4.smape(real, p),
                              mase=M4.mase(np.asarray(tr[sid], float), real, p, m)))
    return pd.DataFrame(filas)


def main():
    sufijos = sys.argv[1:] or ["", "_s1", "_s2", "_s3", "_s4"]
    partes = [d for d in (medir(s) for s in sufijos) if d is not None and len(d)]
    if len(partes) < 2:
        print("hacen falta al menos dos corridas con semillas distintas")
        return 1
    T = pd.concat(partes, ignore_index=True)
    B = pd.read_csv(AQUI / "TRANSPOSE_1000_RANDOM" / "owa_modelos_por_serie.csv")
    n2 = B[B.modelo == "naive2"].groupby("frecuencia").agg(s=("smape", "mean"),
                                                           m=("mase", "mean"))

    g = T.groupby(["semilla", "frecuencia"]).agg(s=("smape", "mean"),
                                                 m=("mase", "mean")).reset_index()
    g["OWA"] = [.5 * (r.s / n2.s[r.frecuencia] + r.m / n2.m[r.frecuencia])
                for _, r in g.iterrows()]

    print(f"{len(partes)} corridas con semillas distintas\n")
    print(f"{'Frecuencia':11s} {'media':>7s} {'desvio':>7s} {'min':>7s} {'max':>7s} "
          f"{'rango':>7s} {'mejor banco':>12s}  ¿lo supera?")
    resumen = []
    for f in FRECS:
        v = g[g.frecuencia == f].OWA.values
        if len(v) < 2:
            continue
        rango = v.max() - v.min()
        mb = MEJOR_BANCO[f]
        # solo se puede afirmar si TODAS las corridas quedan del mismo lado
        if v.max() < mb:
            veredicto = "si, en todas"
        elif v.min() > mb:
            veredicto = "no, en ninguna"
        else:
            veredicto = "DEPENDE DE LA SEMILLA"
        print(f"{f:11s} {v.mean():7.4f} {v.std():7.4f} {v.min():7.4f} {v.max():7.4f} "
              f"{rango:7.4f} {mb:12.4f}  {veredicto}")
        resumen.append(dict(frecuencia=f, media=v.mean(), desvio=v.std(),
                            minimo=v.min(), maximo=v.max(), rango=rango,
                            mejor_banco=mb, veredicto=veredicto))

    # global ponderado, por semilla
    w = pd.Series(POBLACION_M4); w = w / w.sum()
    bS = float((n2.s.reindex(w.index) * w).sum()); bM = float((n2.m.reindex(w.index) * w).sum())
    glob = []
    for sem, gg in g.groupby("semilla"):
        gg = gg.set_index("frecuencia")
        if len(gg) < len(FRECS):
            continue
        S = float((gg.s.reindex(w.index) * w).sum()); M = float((gg.m.reindex(w.index) * w).sum())
        glob.append(.5 * (S / bS + M / bM))
    if glob:
        glob = np.array(glob)
        print(f"\nGLOBAL ponderado: {glob.mean():.4f} +- {glob.std():.4f}   "
              f"[{glob.min():.4f}, {glob.max():.4f}]   rango {glob.max()-glob.min():.4f}")
        print(f"  referencia: ARIMA, el mejor del banco, 0.9360")

    R = pd.DataFrame(resumen)
    with pd.ExcelWriter(AQUI / "dispersion_nbeats.xlsx") as w_:
        R.to_excel(w_, sheet_name="por_frecuencia", index=False)
        g.to_excel(w_, sheet_name="por_semilla", index=False)
    T.to_csv(AQUI / "nbeats_por_serie_semillas.csv", index=False)
    print("\ndispersion_nbeats.xlsx · nbeats_por_serie_semillas.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
