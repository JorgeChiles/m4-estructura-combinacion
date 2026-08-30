#!/usr/bin/env python3
"""
¿Caracterizar sirve poco, o sirve poco para ELEGIR y sí para PONDERAR?

La objeción que responde
------------------------
El experimento de enrutamiento muestra que asignar modelo por frecuencia x
estrato de complejidad recorre apenas el 9,6 % de la distancia hacia el oraculo,
y que un meta-aprendiz con los 29 atributos recorre el 10,7 %. De ahi se seguiria
que caracterizar series para asignar modelo tiene techo bajo.

Pero FFORMA salio SEGUNDO en M4 (OWA 0.838) usando exactamente caracteristicas de
series. La diferencia no esta en las caracteristicas sino en que hace con ellas:

    este trabajo (C, F)   ELIGE un modelo
    FFORMA                COMBINA varios con pesos aprendidos

Y eso encaja con la causa ya diagnosticada: si la etiqueta "cual gana" es
inestable, elegir hereda la inestabilidad y combinar la promedia.

Diseño
------
Mismo protocolo que `experimento_enrutamiento.py`: la regla se aprende en la
mitad de las series y se evalua en la otra, sobre muchas particiones, con los
cortes de estrato calculados SOLO con entrenamiento.

    A   modelo fijo global
    B   el mejor de cada frecuencia                      (elige)
    C   el mejor de cada frecuencia x estrato            (elige)
    Bk  promedio de los k mejores de cada frecuencia     (combina)
    Gk  promedio de los k mejores de frecuencia x estrato (combina)
    D   oraculo por serie

La comparacion que decide es **Gk contra Bk**: si el indice sigue aportando
cuando se combina, entonces caracterizar sirve, y lo que fallaba era elegir.

Salidas:
    politica_combinar.xlsx
    figuras/fig13_combinar.pdf

Uso:  python3 politica_combinar.py [n_particiones]
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

N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 30
KS = (1, 2, 3, 4, 5)
FRECS = {"Yearly": (6, 1), "Quarterly": (8, 4), "Monthly": (18, 12),
         "Weekly": (13, 1), "Daily": (14, 1), "Hourly": (48, 24)}
# duplicados de benchmark y el que no cubre todas las frecuencias
FUERA = {"naive", "snaive", "Holt-Winters"}


def cargar():
    """Tensores por frecuencia: pronosticos, reales y escala del MASE."""
    P = pd.read_pickle(AQUI / "pronosticos_modelos.pkl.gz")
    P = P[~P.modelo.astype(str).isin(FUERA)]
    idx = pd.read_excel(RES / "df_features_complexity.xlsx",
                        usecols=["serie", "complexity_index"]).set_index("serie")

    D = {}
    for f, (h, m) in FRECS.items():
        sub = P[P.frecuencia == f]
        if sub.empty:
            continue
        w = sub.pivot_table(index=["serie", "h"], columns="modelo", values="pred")
        modelos = [c for c in w.columns if w[c].notna().any()]
        w = w[modelos]
        series = w.index.get_level_values(0).unique().tolist()

        tr = M4.load_m4(DATA / f"{f}-train.csv")
        te = M4.load_m4(DATA / f"{f}-test.csv")
        series = [s for s in series if s in tr and s in te and s in idx.index]

        # diseño completo: solo series donde TODOS los modelos tienen pronostico
        ok = []
        for s in series:
            b = w.loc[s]
            if len(b) >= h and b.iloc[:h].notna().all().all():
                ok.append(s)
        series = ok
        n = len(series)
        A = np.empty((n, len(modelos), h))
        R = np.empty((n, h))
        E = np.empty(n)
        for i, s in enumerate(series):
            A[i] = w.loc[s].iloc[:h][modelos].values.T
            R[i] = np.asarray(te[s], float)[:h]
            y = np.asarray(tr[s], float)
            E[i] = np.mean(np.abs(y[m:] - y[:-m])) if len(y) > m else np.nan
        D[f] = dict(series=np.array(series), modelos=np.array(modelos),
                    P=A, real=R, escala=E,
                    ci=idx.loc[series, "complexity_index"].values,
                    i_n2=modelos.index("naive2") if "naive2" in modelos else None)
        print(f"── {f:10s} {n:5,} series x {len(modelos)} modelos", flush=True)
    return D


def metricas(pred, real, escala):
    """sMAPE y MASE por serie de un bloque de pronosticos."""
    den = np.abs(real) + np.abs(pred)
    sm = 200.0 * np.where(den > 0, np.abs(real - pred) / np.where(den > 0, den, 1), 0).mean(axis=1)
    ma = np.abs(real - pred).mean(axis=1) / escala
    return sm, ma


def precalcular(D):
    """Metricas de cada modelo individual: se usan para rankear en train."""
    for f, d in D.items():
        n, k, h = d["P"].shape
        SM = np.empty((n, k)); MA = np.empty((n, k))
        for j in range(k):
            SM[:, j], MA[:, j] = metricas(d["P"][:, j, :], d["real"], d["escala"])
        d["SM"], d["MA"] = SM, MA
    return D


def owa_de(sm, ma, sm_n2, ma_n2):
    return .5 * (sm.mean() / sm_n2.mean() + ma.mean() / ma_n2.mean())


def ranking(d, filas):
    """Modelos ordenados por OWA sobre un subconjunto de series."""
    j2 = d["i_n2"]
    bs, bm = d["SM"][filas, j2].mean(), d["MA"][filas, j2].mean()
    o = .5 * (d["SM"][filas].mean(axis=0) / bs + d["MA"][filas].mean(axis=0) / bm)
    return np.argsort(o)


def una_particion(D, semilla):
    rng = np.random.default_rng(semilla)
    fila = {"semilla": semilla}
    # acumuladores globales (promedio simple sobre la muestra, como el resto
    # del experimento de enrutamiento)
    acc = {}

    def guardar(nombre, f, sm, ma):
        acc.setdefault(nombre, []).append((sm, ma))

    for f, d in D.items():
        n = len(d["series"])
        tr = rng.random(n) < .5
        te = ~tr
        j2 = d["i_n2"]
        guardar("naive2", f, d["SM"][te, j2], d["MA"][te, j2])

        orden_f = ranking(d, tr)
        # --- B y Bk: por frecuencia, eligiendo o combinando
        for k in KS:
            sel = orden_f[:k]
            pred = d["P"][:, sel, :].mean(axis=1)
            sm, ma = metricas(pred[te], d["real"][te], d["escala"][te])
            guardar(f"B{k}", f, sm, ma)

        # --- C y Gk: por frecuencia x estrato del indice
        corte = np.median(d["ci"][tr])          # solo con entrenamiento
        estr_tr = [tr & (d["ci"] <= corte), tr & (d["ci"] > corte)]
        estr_te = [te & (d["ci"] <= corte), te & (d["ci"] > corte)]
        for k in KS:
            sm_all, ma_all = [], []
            for mtr, mte in zip(estr_tr, estr_te):
                if mtr.sum() < 5 or mte.sum() == 0:
                    continue
                sel = ranking(d, mtr)[:k]
                pred = d["P"][:, sel, :].mean(axis=1)
                sm, ma = metricas(pred[mte], d["real"][mte], d["escala"][mte])
                sm_all.append(sm); ma_all.append(ma)
            if sm_all:
                guardar(f"G{k}", f, np.concatenate(sm_all), np.concatenate(ma_all))
            if k == 5:
                fila.setdefault("_sel", {})[f] = [
                    tuple(d["modelos"][ranking(d, m)[:5]]) for m in estr_tr
                    if m.sum() >= 5]

        # --- CONTROL: combinar 5 modelos AL AZAR
        # Si promediar 5 sorteados rindiera lo mismo que promediar los 5
        # mejores de entrenamiento, entonces el ranking no aportaria nada y
        # todo el efecto seria del promedio. Es el control que separa "combinar
        # ayuda" de "saber QUE combinar ayuda".
        sm_all, ma_all = [], []
        for mte in estr_te:
            if mte.sum() == 0:
                continue
            sel = rng.choice(d["P"].shape[1], 5, replace=False)
            pred = d["P"][:, sel, :].mean(axis=1)
            a, b = metricas(pred[mte], d["real"][mte], d["escala"][mte])
            sm_all.append(a); ma_all.append(b)
        if sm_all:
            guardar("azar5", f, np.concatenate(sm_all), np.concatenate(ma_all))

        # control mas exigente: 5 al azar DENTRO del top-10 de entrenamiento.
        # Descarta la explicacion facil de que el azar solo pierde por meter
        # modelos que divergen, y pregunta si el orden fino importa.
        sm_all, ma_all = [], []
        for mtr, mte in zip(estr_tr, estr_te):
            if mtr.sum() < 5 or mte.sum() == 0:
                continue
            top10 = ranking(d, mtr)[:10]
            sel = rng.choice(top10, 5, replace=False)
            pred = d["P"][:, sel, :].mean(axis=1)
            a, b = metricas(pred[mte], d["real"][mte], d["escala"][mte])
            sm_all.append(a); ma_all.append(b)
        if sm_all:
            guardar("azar5_top10", f, np.concatenate(sm_all), np.concatenate(ma_all))

        # --- k elegido en ENTRENAMIENTO, no fijado de antemano
        # El reproche es correcto: reportar k=5 porque es el mejor en prueba
        # seria elegir mirando el test. Aca k se elige con la mitad de
        # entrenamiento y se aplica a la de prueba.
        mejor_k, mejor_owa = None, np.inf
        for k in KS:
            sm_tr, ma_tr = [], []
            for mtr in estr_tr:
                if mtr.sum() < 5:
                    continue
                sel = ranking(d, mtr)[:k]
                pred = d["P"][:, sel, :].mean(axis=1)
                a, b = metricas(pred[mtr], d["real"][mtr], d["escala"][mtr])
                sm_tr.append(a); ma_tr.append(b)
            if not sm_tr:
                continue
            n2b = d["SM"][tr, j2].mean(), d["MA"][tr, j2].mean()
            o = .5 * (np.concatenate(sm_tr).mean() / n2b[0] +
                      np.concatenate(ma_tr).mean() / n2b[1])
            if o < mejor_owa:
                mejor_owa, mejor_k = o, k
        sm_all, ma_all = [], []
        for mtr, mte in zip(estr_tr, estr_te):
            if mtr.sum() < 5 or mte.sum() == 0:
                continue
            sel = ranking(d, mtr)[:mejor_k]
            pred = d["P"][:, sel, :].mean(axis=1)
            a, b = metricas(pred[mte], d["real"][mte], d["escala"][mte])
            sm_all.append(a); ma_all.append(b)
        if sm_all:
            guardar("Gk_auto", f, np.concatenate(sm_all), np.concatenate(ma_all))
        fila[f"k_{f}"] = mejor_k

        # --- oraculo por serie
        mejor = d["SM"][te].argmin(axis=1)
        sm = d["SM"][te][np.arange(te.sum()), mejor]
        ma = d["MA"][te][np.arange(te.sum()), mejor]
        guardar("oraculo", f, sm, ma)

    sel_guardado = fila.pop("_sel", None)
    n2sm = np.concatenate([a for a, _ in acc["naive2"]])
    n2ma = np.concatenate([b for _, b in acc["naive2"]])
    for nombre, lst in acc.items():
        sm = np.concatenate([a for a, _ in lst]); ma = np.concatenate([b for _, b in lst])
        fila[nombre] = owa_de(sm, ma, n2sm, n2ma)
    return fila, sel_guardado


def main():
    if not (AQUI / "pronosticos_modelos.pkl.gz").exists():
        print("falta pronosticos_modelos.pkl.gz; correr extraer_pronosticos.py")
        return 1
    D = precalcular(cargar())
    print(f"\n{N_PART} particiones, mitad entrenamiento / mitad prueba\n")
    salidas = [una_particion(D, s) for s in range(N_PART)]
    R = pd.DataFrame([a for a, _ in salidas])
    SEL = [b for _, b in salidas]
    R.to_excel(AQUI / "politica_combinar.xlsx", index=False)

    from scipy.stats import wilcoxon
    print("═══ ELEGIR CONTRA COMBINAR ═══\n")
    print(f"{'k':>3s} {'B (frecuencia)':>16s} {'G (frec x indice)':>18s} "
          f"{'aporta el indice':>17s} {'G mejor':>9s} {'p':>10s}")
    for k in KS:
        b, g = R[f"B{k}"], R[f"G{k}"]
        p = wilcoxon(g, b).pvalue
        print(f"{k:3d} {b.mean():16.4f} {g.mean():18.4f} {b.mean()-g.mean():+17.4f} "
              f"{(g<b).sum():6d}/{len(R)} {p:10.3g}")

    print(f"\n{'':22s} {'OWA':>8s} {'de':>7s}   {'cierra hacia el oraculo':>24s}")
    a = R.B1.mean()   # k=1 por frecuencia es el control del articulo
    hueco = R.B1 - R.oraculo
    for nom, col in ([("A/B  elegir por frecuencia", "B1"),
                      ("C    elegir por indice", "G1")]
                     + [(f"B{k}   combinar {k} por frecuencia", f"B{k}") for k in KS if k > 1]
                     + [(f"G{k}   combinar {k} por indice", f"G{k}") for k in KS if k > 1]
                     + [("D    oraculo por serie", "oraculo")]):
        cierra = 100 * ((R.B1 - R[col]) / hueco).mean()
        print(f"{nom:22s} {R[col].mean():8.4f} {R[col].std():7.4f} {cierra:23.1f} %")

    print("\n═══ CONTROL: ¿ALCANZA CON PROMEDIAR, O HAY QUE SABER QUE PROMEDIAR? ═══")
    print(f"  5 al azar          OWA {R.azar5.mean():.4f}  (de {R.azar5.std():.4f})")
    print(f"  5 mejores de train OWA {R.G5.mean():.4f}  (de {R.G5.std():.4f})")
    print(f"  aporta saber cuales: {R.azar5.mean()-R.G5.mean():+.4f}")
    from scipy.stats import wilcoxon as _w2
    print(f"  p = {_w2(R.G5, R.azar5).pvalue:.3g}   G5 mejor en "
          f"{(R.G5 < R.azar5).sum()}/{len(R)}")
    print(f"\n  El azar sobre TODO el banco pierde por meter modelos que divergen.")
    print(f"  Control mas exigente, 5 al azar dentro del top-10 de entrenamiento:")
    print(f"    azar dentro del top-10  OWA {R.azar5_top10.mean():.4f}  "
          f"(de {R.azar5_top10.std():.4f})")
    print(f"    top-5 de entrenamiento  OWA {R.G5.mean():.4f}")
    print(f"    aporta el orden fino: {R.azar5_top10.mean()-R.G5.mean():+.4f}   "
          f"p = {_w2(R.G5, R.azar5_top10).pvalue:.3g}   "
          f"G5 mejor en {(R.G5 < R.azar5_top10).sum()}/{len(R)}")

    # ── estabilidad del top-5: responde si combinar tiene ventaja injusta
    print("\n═══ ¿ES JUSTA LA COMPARACION? ESTABILIDAD DEL TOP-5 ═══")
    print("   Si el top-5 cambiara mucho entre particiones, combinar estaria")
    print("   aprovechando diversidad que elegir no puede aprovechar.\n")
    filas_est = []
    for f in D:
        conj = [set(t) for sel in SEL if sel and f in sel for t in sel[f]]
        if len(conj) < 2:
            continue
        jac = [len(a & b) / len(a | b) for i, a in enumerate(conj)
               for b in conj[i+1:]]
        # frecuencia de aparicion de cada modelo en el top-5
        from collections import Counter
        c = Counter(m for t in conj for m in t)
        nucleo = [m for m, v in c.items() if v >= .9 * len(conj)]
        filas_est.append(dict(frecuencia=f, jaccard=np.mean(jac),
                              nucleo=len(nucleo),
                              modelos=", ".join(sorted(nucleo)[:5])))
    EST = pd.DataFrame(filas_est)
    print(f"{'frecuencia':11s} {'Jaccard medio':>14s} {'nucleo estable':>15s}  modelos del nucleo")
    for _, r in EST.iterrows():
        print(f"{r.frecuencia:11s} {r.jaccard:14.3f} {r.nucleo:12d}/5    {r.modelos}")

    # ── k elegido en entrenamiento
    print("\n═══ k ELEGIDO EN ENTRENAMIENTO, NO FIJADO EN 5 ═══")
    print(f"  Gk_auto  OWA {R.Gk_auto.mean():.4f}  (de {R.Gk_auto.std():.4f})")
    print(f"  G5 fijo  OWA {R.G5.mean():.4f}  (de {R.G5.std():.4f})")
    print(f"  costo de tener que elegir k: {R.Gk_auto.mean()-R.G5.mean():+.4f}")
    kk = [c for c in R.columns if c.startswith("k_")]
    print("  k elegido por frecuencia:",
          {c[2:]: R[c].value_counts().idxmax() for c in kk})

    # ── intervalo del aporte del indice, pareado por particion
    print("\n═══ APORTE DEL INDICE AL COMBINAR: INTERVALO, NO SOLO p ═══")
    print("   El reproche es que 0.009 cae dentro de la incertidumbre de muestreo")
    print("   de +-0.015. Pero esa incertidumbre es del OWA GLOBAL; la diferencia")
    print("   entre dos politicas se evalua sobre LAS MISMAS series de cada")
    print("   particion, asi que el estadistico correcto es el pareado.\n")
    for k in KS:
        dif = R[f"B{k}"] - R[f"G{k}"]
        b = np.percentile([np.random.default_rng(i).choice(dif, len(dif)).mean()
                           for i in range(4000)], [2.5, 97.5])
        print(f"  k={k}: aporte {dif.mean():+.4f}  IC 95 % [{b[0]:+.4f}, {b[1]:+.4f}]"
              + ("   incluye cero" if b[0] <= 0 <= b[1] else "   no incluye cero"))

    with pd.ExcelWriter(AQUI / "politica_combinar_extra.xlsx") as wr:
        EST.to_excel(wr, sheet_name="estabilidad_top5", index=False)
    figura(R)
    print("\npolitica_combinar.xlsx · politica_combinar_extra.xlsx")
    return 0


def figura(R):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ks = list(KS)
    for col, nom, c in (("B", "por frecuencia", AZUL), ("G", "frecuencia × índice", NARANJA)):
        med = [R[f"{col}{k}"].mean() for k in ks]
        de = [R[f"{col}{k}"].std() for k in ks]
        ax.errorbar(ks, med, yerr=de, marker="o", ms=4.5, lw=1.4, capsize=3,
                    color=c, label=nom)
    ax.axhline(R.oraculo.mean(), color=GRIS, ls=":", lw=1)
    ax.text(ks[-1], R.oraculo.mean(), " oráculo", color=GRIS, fontsize=7.5, va="center")
    ax.set_xticks(ks)
    ax.set_xlabel("número de modelos combinados (k = 1 es elegir uno)")
    ax.set_ylabel("OWA en prueba")
    ax.set_title("Elegir un modelo frente a combinar varios", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", visible=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig13_combinar.{ext}")
    plt.close(fig)
    print("figuras/fig13_combinar.pdf")


if __name__ == "__main__":
    raise SystemExit(main())
