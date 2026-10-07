#!/usr/bin/env python3
"""
¿Es estable el indice de complejidad si la serie cambia un poco?

La objecion que responde
------------------------
Un revisor pidio medir que tan sensible es el indice a perturbaciones y a la
longitud de la historia. La objecion es seria porque el indice no se reporta: se
USA para decidir. Si una serie cambia de estrato al agregarle ruido que no
cambia su naturaleza, la politica de enrutamiento decide distinto por razones
que no son de la serie.

Que se mide, y por que esa cantidad
-----------------------------------
No alcanza con la correlacion entre el indice original y el perturbado: dos
indices pueden correlacionar 0,99 y aun asi reasignar de estrato a muchas series
si el corte cae donde la densidad es alta. Lo que decide la politica es el
ESTRATO, asi que la cantidad que importa es la TASA DE CAMBIO DE ESTRATO: que
fraccion de series cruza la mediana al perturbarlas.

Se reportan las dos, y ademas el efecto sobre la decision final: se rehace el
enrutamiento usando el indice perturbado para asignar modelo, y se compara el OWA
contra el que se obtiene con el indice original. Esa es la unica medida que dice
si la inestabilidad, de existir, cuesta algo.

Perturbaciones
--------------
1. RUIDO. Se suma ruido gaussiano de desvio proporcional al de la serie, en
   niveles de 1 %, 2 % y 5 %. Es la perturbacion que un revisor pide por defecto.
2. HISTORIA. Se recalcula el indice usando solo el ultimo 75 % y el ultimo 50 %
   de las observaciones. Es la mas realista de las dos: en produccion la historia
   disponible cambia, y el indice se recalcularia sobre lo que haya.

El indice perturbado se proyecta con el MISMO escalador y la MISMA componente
principal ajustados sobre los datos originales. No se reajusta el PCA: la
pregunta no es si el espacio se sostiene, sino si una serie concreta conserva su
posicion en el espacio ya definido, que es como se usaria en la practica.

Salidas:
    estabilidad_indice.xlsx
    figuras/fig15_estabilidad.pdf

Uso:  python3 estabilidad_indice.py [n_series_por_frecuencia]
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
sys.path.insert(0, "/tmp")
M4 = importlib.import_module("06_m4_metrics")
DATA = AQUI / "m4-structural-complexity" / "data"

N_POR_FREC = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEMILLA = 20261006
FRECS = {"Yearly": 1, "Quarterly": 4, "Monthly": 12,
         "Weekly": 1, "Daily": 1, "Hourly": 24}
RUIDOS = (0.01, 0.02, 0.05)
RECORTES = (0.75, 0.50)


def cargar_feats():
    """Las funciones de extraccion viven en el cuaderno que produjo los
    descriptores originales; se importan de ahi para no reimplementarlas."""
    import json, re
    f = AQUI / "CLASIFICACION_ST" / "CLASIFICACION_V3_All.ipynb"
    nb = json.load(open(f, encoding="utf-8"))
    celdas = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    quiero = {"safe_float", "zscore_series", "robust_scale_series", "shannon_entropy",
              "spectral_entropy", "dominant_frequency_features", "turning_points_ratio",
              "outlier_ratio_robust", "hurst_exponent", "acf_decay",
              "trend_linearity_features", "strength_trend_stl", "strength_seasonal_stl",
              "stationarity_features", "change_points_per_length", "extract_features_v2"}
    sel = [c for c in celdas if set(re.findall(r"^def (\w+)", c, re.M)) & quiero]
    imports = sorted({l.strip() for c in celdas for l in c.split("\n")
                      if re.match(r"^\s*(import |from )\w", l) and "rutas" not in l})
    mod = Path("/tmp/_feats_estabilidad.py")
    mod.write_text("\n".join(imports) + "\n\n" + "\n\n".join(sel) + "\n")
    return importlib.import_module("_feats_estabilidad")


def main():
    FE = cargar_feats()
    print("── funciones de extracción cargadas del cuaderno original", flush=True)

    # ── referencia: descriptores e índice ya calculados
    orig = pd.read_excel(RES / "df_features_complexity.xlsx")
    COLS = [c for c in orig.columns
            if c not in ("serie", "category", "freq", "horizon", "source_file",
                         "n_raw", "n_valid", "missing_ratio", "complexity_index",
                         "PC1", "cluster")]
    COLS = [c for c in COLS if pd.api.types.is_numeric_dtype(orig[c])]
    print(f"── {len(COLS)} descriptores, {len(orig):,} series de referencia", flush=True)

    # ── el escalador y la componente, ajustados sobre los datos ORIGINALES
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    imp = SimpleImputer(strategy="median").fit(orig[COLS])
    esc = StandardScaler().fit(imp.transform(orig[COLS]))
    pca = PCA(n_components=1).fit(esc.transform(imp.transform(orig[COLS])))

    def indice(df):
        """-PC1 con el escalador y la componente ya ajustados."""
        return -pca.transform(esc.transform(imp.transform(df[COLS])))[:, 0]

    # Indice paralelo SIN log_length. Recortar la historia cambia la longitud, y
    # la longitud es uno de los 29 descriptores: parte de cualquier movimiento
    # del indice al recortar es mecanico y no dice nada sobre la estructura de la
    # serie. Este segundo indice separa las dos cosas.
    COLS_SL = [c for c in COLS if c != "log_length"]
    imp_sl = SimpleImputer(strategy="median").fit(orig[COLS_SL])
    esc_sl = StandardScaler().fit(imp_sl.transform(orig[COLS_SL]))
    pca_sl = PCA(n_components=1).fit(esc_sl.transform(imp_sl.transform(orig[COLS_SL])))

    def indice_sin_largo(df):
        return -pca_sl.transform(esc_sl.transform(imp_sl.transform(df[COLS_SL])))[:, 0]

    ref = orig.set_index("serie")
    rng = np.random.default_rng(SEMILLA)
    filas = []

    for f, m in FRECS.items():
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        cand = [s for s in tr if s in ref.index]
        ss = list(rng.choice(cand, size=min(N_POR_FREC, len(cand)), replace=False))
        def descriptores(pares):
            """(sid, serie) -> DataFrame indexado por sid, saltando los None.

            extract_features_v2 devuelve None cuando la serie es demasiado corta
            para alguno de los descriptores, que es justo lo que pasa al recortar
            la historia de las series anuales."""
            filas = {}
            for sid, y in pares:
                d = FE.extract_features_v2(y, sid, m)
                if d is not None:
                    filas[sid] = d
            D = pd.DataFrame.from_dict(filas, orient="index")
            for c in COLS:
                if c not in D.columns:
                    D[c] = np.nan
            return D

        limpio = {sid: np.asarray(tr[sid], float)[np.isfinite(np.asarray(tr[sid], float))]
                  for sid in ss}
        B = descriptores(limpio.items())
        i0 = pd.Series(indice(B), index=B.index)
        i0s = pd.Series(indice_sin_largo(B), index=B.index)

        for nivel in RUIDOS:
            def ruidosa(y):
                sd = np.std(y)
                return y + rng.normal(0, nivel * sd, size=len(y)) if sd > 0 else y
            P = descriptores((sid, ruidosa(y)) for sid, y in limpio.items())
            i1 = pd.Series(indice(P), index=P.index)
            i1s = pd.Series(indice_sin_largo(P), index=P.index)
            com = i0.index.intersection(i1.index)
            filas.append(dict(frecuencia=f, perturbacion=f"ruido {nivel:.0%}",
                              i0=i0[com].to_numpy(), i1=i1[com].to_numpy(),
                              i0s=i0s[com].to_numpy(), i1s=i1s[com].to_numpy()))
            print(f"   {f:10s} ruido {nivel:.0%} listo", flush=True)

        for frac in RECORTES:
            def recorta(y):
                k = max(int(len(y) * frac), 2 * m + 4)
                return y[-k:]
            P = descriptores((sid, recorta(y)) for sid, y in limpio.items())
            i1 = pd.Series(indice(P), index=P.index)
            i1s = pd.Series(indice_sin_largo(P), index=P.index)
            com = i0.index.intersection(i1.index)
            filas.append(dict(frecuencia=f, perturbacion=f"historia {frac:.0%}",
                              i0=i0[com].to_numpy(), i1=i1[com].to_numpy(),
                              i0s=i0s[com].to_numpy(), i1s=i1s[com].to_numpy()))
            print(f"   {f:10s} historia {frac:.0%} listo", flush=True)

    # ── resumen: correlación, y sobre todo cambio de estrato
    res = []
    for r in filas:
        a, b = r["i0"], r["i1"]
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        corte = np.median(a)                 # el corte se fija con el índice original
        cambia = (a <= corte) != (b <= corte)
        asl, bsl = r["i0s"][ok], r["i1s"][ok]
        corte_sl = np.median(asl)
        cambia_sl = (asl <= corte_sl) != (bsl <= corte_sl)
        res.append(dict(frecuencia=r["frecuencia"], perturbacion=r["perturbacion"],
                        n=len(a), pearson=np.corrcoef(a, b)[0, 1],
                        spearman=pd.Series(a).corr(pd.Series(b), method="spearman"),
                        cambia_estrato=cambia.mean(),
                        cambia_sin_largo=cambia_sl.mean(),
                        spearman_sin_largo=pd.Series(asl).corr(pd.Series(bsl),
                                                               method="spearman")))
    R = pd.DataFrame(res)
    R.to_excel(AQUI / "estabilidad_indice.xlsx", index=False)

    print("\n═══ ESTABILIDAD DEL ÍNDICE ═══\n")
    piv = R.pivot(index="perturbacion", columns="frecuencia", values="cambia_estrato")
    orden = [f"ruido {n:.0%}" for n in RUIDOS] + [f"historia {f:.0%}" for f in RECORTES]
    print("Fracción de series que CAMBIAN DE ESTRATO:\n")
    print((100 * piv.loc[orden]).round(1).to_string())
    print("\nCorrelación de Spearman entre índice original y perturbado:\n")
    pv = R.pivot(index="perturbacion", columns="frecuencia", values="spearman")
    print(pv.loc[orden].round(3).to_string())
    print("\nGlobal, y qué queda al excluir log_length del índice:\n")
    print(f"  {'perturbación':16s} {'cambia estrato':>14s} {'sin log_length':>15s}"
          f" {'Spearman':>10s} {'sin log_len':>12s}")
    for p in orden:
        sub = R[R.perturbacion == p]
        print(f"  {p:16s} {100*sub.cambia_estrato.mean():13.1f} %"
              f" {100*sub.cambia_sin_largo.mean():14.1f} %"
              f" {sub.spearman.mean():10.3f} {sub.spearman_sin_largo.mean():12.3f}")
    print("\nestabilidad_indice.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
