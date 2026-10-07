#!/usr/bin/env python3
"""
¿La inversión de régimen depende de que N-BEATS falle en mensual?

La objecion que responde
------------------------
N-BEATS domina cuatro frecuencias y con el en el banco elegir supera a combinar.
Pero en mensual --el 48 % de la coleccion-- la implementacion no converge, y un
revisor puede objetar con razon que la inversion se demuestra sobre 5 de 6
frecuencias. El articulo argumenta que el fallo es CONSERVADOR: si mensual
funcionara, el modelo dominaria alli tambien y la inversion seria mas fuerte.
El argumento es plausible y hasta ahora no estaba verificado. Esto lo verifica.

Dos analisis
------------
1. COTA INFERIOR, sin nada sintetico. Se repiten las politicas excluyendo
   mensual. Si la inversion se sostiene sobre las cinco frecuencias restantes
   --el 52 % de la coleccion-- entonces no depende de mensual en ningun sentido.

2. BARRIDO DE CALIDAD. Se sustituye el pronostico mensual de N-BEATS por uno
   sintetico de calidad controlada y se observa donde cambia la conclusion:

       f(lambda) = lambda * (mejor del banco por serie) + (1-lambda) * ARIMA

   Con lambda=0 el modelo sintetico rinde como ARIMA, el mejor del banco en
   mensual; al subir lambda se vuelve progresivamente dominante.

   ADVERTENCIA, y hay que declararla en el articulo: el extremo del barrido usa
   el mejor modelo POR SERIE, que se conoce solo mirando la prueba. No es un
   pronostico alcanzable ni se lo presenta como tal. Es un dial para responder
   "si mensual tuviera un modelo de calidad X, ¿seguiria invirtiendose?", que es
   exactamente la pregunta del revisor. Ningun numero de este barrido debe
   citarse como desempeno de nada.

Salidas:
    sensibilidad_mensual.xlsx

Uso:  python3 sensibilidad_mensual.py [n_particiones]
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

N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 30
KS = (1, 5)                      # elegir uno contra combinar cinco
FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}
POBLACION = {"Yearly": 23000, "Quarterly": 24000, "Monthly": 48000,
             "Weekly": 359, "Daily": 4227, "Hourly": 414}
FUERA = {"naive", "snaive", "Holt-Winters"}
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def cargar():
    P = pd.read_pickle(AQUI / "pronosticos_modelos_con_nbeats.pkl.gz")
    P = P[~P.modelo.astype(str).isin(FUERA)]
    idx = pd.read_excel(RES / "df_features_complexity.xlsx",
                        usecols=["serie", "complexity_index"]).set_index("serie")
    D = {}
    for f, (h, m) in FRECS.items():
        sub = P[P.frecuencia == f]
        w = sub.pivot_table(index=["serie", "h"], columns="modelo", values="pred")
        modelos = [c for c in w.columns if w[c].notna().any()]
        w = w[modelos]
        tr = M4.load_m4(DATA / f"{f}-train.csv")
        te = M4.load_m4(DATA / f"{f}-test.csv")
        cand = [s for s in w.index.get_level_values(0).unique()
                if s in tr and s in te and s in idx.index]
        series = [s for s in cand
                  if len(w.loc[s]) >= h and w.loc[s].iloc[:h].notna().all().all()]
        n = len(series)
        A = np.empty((n, len(modelos), h)); R = np.empty((n, h)); E = np.empty(n)
        for i, s in enumerate(series):
            A[i] = w.loc[s].iloc[:h][modelos].values.T
            R[i] = np.asarray(te[s], float)[:h]
            y = np.asarray(tr[s], float)
            E[i] = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
        D[f] = dict(series=np.array(series), modelos=list(modelos), P=A, real=R,
                    escala=E, ci=idx.loc[series, "complexity_index"].values,
                    i_n2=modelos.index("naive2"))
        print(f"── {f:10s} {n:5,} series x {len(modelos)} modelos", flush=True)
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


def sustituir_mensual(D, lam):
    """Reemplaza el pronostico mensual de N-BEATS por uno sintetico."""
    d = D["Monthly"]
    j_nb = d["modelos"].index("N-BEATS")
    j_ar = d["modelos"].index("ARIMA")
    # mejor modelo por serie EXCLUYENDO N-BEATS (que esta roto)
    otros = [j for j in range(len(d["modelos"])) if j != j_nb]
    mejor = np.array(otros)[d["SM"][:, otros].argmin(axis=1)]
    P_mejor = d["P"][np.arange(len(mejor)), mejor, :]
    d["P"][:, j_nb, :] = lam * P_mejor + (1 - lam) * d["P"][:, j_ar, :]
    sm, ma = metricas(d["P"][:, j_nb, :], d["real"], d["escala"])
    d["SM"][:, j_nb], d["MA"][:, j_nb] = sm, ma
    b = d["SM"][:, d["i_n2"]].mean(), d["MA"][:, d["i_n2"]].mean()
    return .5 * (sm.mean() / b[0] + ma.mean() / b[1])


def ranking(d, filas):
    j2 = d["i_n2"]
    bs, bm = d["SM"][filas, j2].mean(), d["MA"][filas, j2].mean()
    return np.argsort(.5 * (d["SM"][filas].mean(axis=0) / bs +
                            d["MA"][filas].mean(axis=0) / bm))


def politicas(D, frecuencias, n_part):
    """OWA de elegir uno y de combinar cinco, por frecuencia, ponderado."""
    w = pd.Series({f: POBLACION[f] for f in frecuencias})
    w = w / w.sum()
    filas = []
    for semilla in range(n_part):
        rng = np.random.default_rng(semilla)
        acu = {k: {"s": 0.0, "m": 0.0} for k in KS}
        bs = bm = 0.0
        for f in frecuencias:
            d = D[f]
            n = len(d["series"])
            tr = rng.random(n) < .5
            te = ~tr
            bs += w[f] * d["SM"][te, d["i_n2"]].mean()
            bm += w[f] * d["MA"][te, d["i_n2"]].mean()
            orden = ranking(d, tr)
            for k in KS:
                p = d["P"][:, orden[:k], :].mean(axis=1)
                a, b = metricas(p[te], d["real"][te], d["escala"][te])
                acu[k]["s"] += w[f] * a.mean(); acu[k]["m"] += w[f] * b.mean()
        filas.append({f"k{k}": .5 * (acu[k]["s"] / bs + acu[k]["m"] / bm) for k in KS}
                     | {"semilla": semilla})
    return pd.DataFrame(filas)


def main():
    D = precalcular(cargar())
    from scipy.stats import wilcoxon
    res = []

    # ── 1 · cota inferior: sin mensual, sin nada sintetico
    cinco = [f for f in FRECS if f != "Monthly"]
    pob = sum(POBLACION[f] for f in cinco) / sum(POBLACION.values())
    R = politicas(D, cinco, N_PART)
    print(f"\n═══ 1 · COTA INFERIOR: LAS CINCO FRECUENCIAS QUE N-BEATS SI CUBRE ═══")
    print(f"    {100*pob:.0f} % de la coleccion, sin ningun dato sintetico\n")
    print(f"  elegir uno      OWA {R.k1.mean():.4f}  (de {R.k1.std():.4f})")
    print(f"  combinar cinco  OWA {R.k5.mean():.4f}  (de {R.k5.std():.4f})")
    d15 = R.k5 - R.k1
    print(f"  combinar {'EMPEORA' if d15.mean() > 0 else 'mejora'} en {abs(d15.mean()):.4f}"
          f"   en {(d15>0).sum()}/{len(R)} particiones   p = {wilcoxon(R.k5,R.k1).pvalue:.3g}")
    res.append(dict(escenario="sin mensual", owa_mensual=np.nan,
                    elegir=R.k1.mean(), combinar=R.k5.mean(), dif=d15.mean()))

    # ── 2 · barrido de calidad del mensual
    print(f"\n═══ 2 · BARRIDO: ¿QUE CALIDAD DEBERIA TENER EL MENSUAL? ═══")
    print("    OJO: el extremo del barrido usa el mejor modelo por serie, que se")
    print("    conoce mirando la prueba. Es un dial para el analisis, no un")
    print("    pronostico alcanzable. Ningun numero de aca es un desempeno.\n")
    P0 = D["Monthly"]["P"].copy()
    SM0, MA0 = D["Monthly"]["SM"].copy(), D["Monthly"]["MA"].copy()
    print(f"{'lambda':>7s} {'OWA mensual':>12s} {'elegir uno':>11s} "
          f"{'combinar 5':>11s} {'diferencia':>11s}  conclusion")
    for lam in LAMBDAS:
        D["Monthly"]["P"] = P0.copy()
        D["Monthly"]["SM"], D["Monthly"]["MA"] = SM0.copy(), MA0.copy()
        owa_m = sustituir_mensual(D, lam)
        R = politicas(D, list(FRECS), N_PART)
        dif = (R.k5 - R.k1).mean()
        concl = "elegir gana" if dif > 0 else "combinar gana"
        print(f"{lam:7.2f} {owa_m:12.4f} {R.k1.mean():11.4f} {R.k5.mean():11.4f} "
              f"{dif:+11.4f}  {concl}")
        res.append(dict(escenario=f"lambda={lam}", owa_mensual=owa_m,
                        elegir=R.k1.mean(), combinar=R.k5.mean(), dif=dif))

    # ── referencia: como esta hoy, con el mensual roto
    D["Monthly"]["P"] = P0; D["Monthly"]["SM"], D["Monthly"]["MA"] = SM0, MA0
    R = politicas(D, list(FRECS), N_PART)
    dif = (R.k5 - R.k1).mean()
    print(f"\n  medido (mensual roto, OWA 5.24): elegir {R.k1.mean():.4f}  "
          f"combinar {R.k5.mean():.4f}  dif {dif:+.4f}")
    res.append(dict(escenario="medido (mensual roto)", owa_mensual=5.2442,
                    elegir=R.k1.mean(), combinar=R.k5.mean(), dif=dif))

    pd.DataFrame(res).to_excel(AQUI / "sensibilidad_mensual.xlsx", index=False)
    print("\nsensibilidad_mensual.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
