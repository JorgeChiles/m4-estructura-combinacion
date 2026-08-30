#!/usr/bin/env python3
"""
Extrae y cachea los pronosticos de los 26 modelos, uno por serie.

Por que hace falta
------------------
Todo lo hecho hasta aca usa el ERROR por serie y modelo. Para probar politicas
que COMBINAN pronosticos eso no alcanza: el error de un promedio no es el
promedio de los errores. Hay que tener las predicciones.

Estan en los consolidados, en el bloque que arranca en "Datos Prueba" de cada
hoja. Leerlos con openpyxl es lento, asi que se hace una vez y se cachea.

Se reutiliza la maquinaria de `owa_modelos.py` en vez de reimplementarla: misma
deduplicacion de recorridas repetidas (" v2"), mismo control de que los reales
guardados coincidan con los oficiales de M4.

Salida:
    pronosticos_modelos.pkl.gz   frecuencia, serie, modelo, h, pred

Uso:  python3 extraer_pronosticos.py
"""
import importlib
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
MOD = AQUI / "TRANSPOSE_1000_RANDOM"
sys.path.insert(0, str(MOD))
OM = importlib.import_module("owa_modelos")
M4 = OM.M4
DATA = OM.DATA

# los benchmarks se recalculan, no se leen: asi el pool combinable incluye
# Naive2 con la misma implementacion que el denominador del OWA
BENCH = ("naive2", "naive", "snaive")


def main():
    t0 = time.time()
    partes = []
    for carpeta, base, nombre, h, m in OM.FRECS:
        if not os.path.isdir(MOD / carpeta):
            print(f"── {nombre}: falta la carpeta, se omite", flush=True)
            continue
        print(f"── {nombre} (h={h}, m={m})", flush=True)
        pron, rep = OM.leer_consolidados(str(MOD / carpeta), base)
        print(f"   {len(pron)} series leidas ({time.time()-t0:.0f} s acumulados)",
              flush=True)

        tr = M4.load_m4(DATA / f"{nombre}-train.csv")
        te = M4.load_m4(DATA / f"{nombre}-test.csv")

        filas, desaj = [], 0
        for sid, d in sorted(pron.items()):
            if sid not in tr or sid not in te:
                continue
            entren = np.asarray(tr[sid], float)
            real = np.asarray(te[sid], float)[:h]

            guardado = d.get("Datos Prueba")
            if guardado is not None and len(guardado) >= len(real):
                if not np.allclose(guardado[:len(real)], real, rtol=1e-3,
                                   atol=1e-3, equal_nan=True):
                    desaj += 1

            todos = dict(d)
            todos.pop("Datos Prueba", None)
            todos["naive2"] = M4.forecast_naive2(entren, h, m)
            todos["naive"] = M4.forecast_naive(entren, h)
            todos["snaive"] = M4.forecast_snaive(entren, h, m)

            for modelo, pred in todos.items():
                pred = np.asarray(pred, float)
                n = min(len(real), len(pred))
                if n == 0 or not np.isfinite(pred[:n]).all():
                    continue
                for j in range(n):
                    filas.append((nombre, sid, modelo, j + 1, float(pred[j])))
        P = pd.DataFrame(filas, columns=["frecuencia", "serie", "modelo", "h", "pred"])
        print(f"   {P.serie.nunique()} series x {P.modelo.nunique()} modelos "
              f"-> {len(P):,} filas" + (f"   ¡{desaj} desajustes!" if desaj else ""),
              flush=True)
        partes.append(P)

    T = pd.concat(partes, ignore_index=True)
    T["modelo"] = T.modelo.astype("category")
    T["frecuencia"] = T.frecuencia.astype("category")
    sal = AQUI / "pronosticos_modelos.pkl.gz"
    T.to_pickle(sal)
    print(f"\n{len(T):,} filas -> {sal.name} "
          f"({sal.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
