#!/usr/bin/env python3
"""
Arma el banco ampliado: los 26 modelos mas N-BEATS, agregado por semillas.

Por que existe este archivo
---------------------------
La fusion se habia hecho a mano en una sesion y el pickle resultante quedaba sin
receta. Al ampliar la muestra anual de 1.000 a 2.000 series hubo que rehacerla y
no habia de donde copiarla. Queda escrita.

Como se agrega
--------------
MEDIANA entre las cinco corridas, no promedio. Esa eleccion no es cosmetica: en
semanal una de las cinco semillas diverge, y el promedio arrastra el OWA de la
frecuencia a 1,1696 mientras que la mediana lo deja en 0,7136. El promedio
hereda la corrida mala; la mediana la descarta.

Las semillas, y como se confirmaron
-----------------------------------
Las corridas originales no dejaron registro de sus semillas. Se reentreno anual
con 42 (sin sufijo) y 1 a 4, que es lo que sugieren los nombres de archivo, y
las predicciones de las 1.000 series anuales que ya existian salieron IDENTICAS
bit a bit a las guardadas. Eso confirma las semillas y, de paso, que el pipeline
es determinista a igual maquina y numero de hilos. (La irreproducibilidad que
documenta el articulo es entre numeros de hilos distintos, no entre corridas.)

Entradas:
    pronosticos_modelos.pkl.gz
    nbeats_pronosticos.csv, nbeats_pronosticos_s1..s4.csv

Salidas:
    pronosticos_modelos_con_nbeats.pkl.gz
    nbeats_ensamble_por_serie.csv     sMAPE y MASE por serie del ensamble

Uso:  python3 fusionar_nbeats.py
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

SUFIJOS = ["", "_s1", "_s2", "_s3", "_s4"]
FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}


def metricas(pred, real, escala):
    den = np.abs(real) + np.abs(pred)
    sm = 200.0 * np.where(den > 0, np.abs(real - pred) / np.where(den > 0, den, 1),
                          0).mean()
    return sm, np.abs(real - pred).mean() / escala


def main():
    falta = [s for s in SUFIJOS if not (AQUI / f"nbeats_pronosticos{s}.csv").exists()]
    if falta:
        print(f"faltan las corridas: {falta}")
        return 1

    corridas = []
    for s in SUFIJOS:
        d = pd.read_csv(AQUI / f"nbeats_pronosticos{s}.csv")
        corridas.append(d.set_index(["frecuencia", "serie", "h"]).pred)
        print(f"── nbeats_pronosticos{s or ' (42)'}: {len(d):,} filas", flush=True)

    # mediana entre semillas, alineando por (frecuencia, serie, h)
    M = pd.concat(corridas, axis=1, keys=range(len(SUFIJOS)))
    n_parciales = int(M.isna().any(axis=1).sum())
    if n_parciales:
        print(f"⚠ {n_parciales:,} filas sin las cinco corridas; se usa la mediana "
              f"de las disponibles")
    NB = M.median(axis=1, skipna=True).rename("pred").reset_index()
    NB["modelo"] = "N-BEATS"
    print(f"\n── ensamble: {len(NB):,} filas, "
          f"{NB.serie.nunique():,} series", flush=True)
    print(NB.groupby("frecuencia").serie.nunique().to_string())

    P = pd.read_pickle(AQUI / "pronosticos_modelos.pkl.gz")
    P = P[P.modelo != "N-BEATS"]              # por si se recorre dos veces
    T = pd.concat([P, NB[P.columns]], ignore_index=True)
    T.to_pickle(AQUI / "pronosticos_modelos_con_nbeats.pkl.gz")
    print(f"\n{len(T):,} filas -> pronosticos_modelos_con_nbeats.pkl.gz")

    # sMAPE y MASE por serie del ensamble, para las tablas por frecuencia
    filas = []
    for f, (h, m) in FRECS.items():
        sub = NB[NB.frecuencia == f]
        if sub.empty:
            continue
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        te = M4.load_m4(DATA / f"{f}-test.csv")
        for sid, g in sub.groupby("serie"):
            if sid not in tr or sid not in te:
                continue
            p = g.sort_values("h").pred.to_numpy(float)[:h]
            real = np.asarray(te[sid], float)[:h]
            y = np.asarray(tr[sid], float)
            if len(p) < h or len(y) <= m:
                continue
            esc = np.mean(np.abs(y[m:] - y[:-m]))
            if not np.isfinite(esc) or esc <= 0:
                continue
            sm, ma = metricas(p, real, esc)
            filas.append(dict(frecuencia=f, serie=sid, smape=sm, mase=ma))
    E = pd.DataFrame(filas)
    E.to_csv(AQUI / "nbeats_ensamble_por_serie.csv", index=False)
    print(f"{len(E):,} series -> nbeats_ensamble_por_serie.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
