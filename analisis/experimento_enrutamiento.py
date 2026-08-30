#!/usr/bin/env python3
"""
¿Sirve el índice de complejidad para ELEGIR modelo, o solo correlaciona?

El reproche del revisor
-----------------------
El artículo muestra que el índice correlaciona con la dificultad del pronóstico,
pero nunca demuestra que USARLO mejore una decisión. Correlación no es utilidad.
Este experimento responde eso con los datos que ya están calculados: no se
reajusta ningún modelo, se reutilizan los errores por serie de los 26.

Diseño
------
Una regla de enrutamiento es una función  serie -> modelo.  Se APRENDE en un
conjunto de entrenamiento y se EVALÚA en uno de prueba; si se aprende y evalúa
sobre las mismas series, el resultado es sobreajuste disfrazado de hallazgo.

    A · Fijo global      un solo modelo para todo (el mejor en entrenamiento)
    B · Por frecuencia   el mejor modelo de cada frecuencia
    C · Por índice       el mejor de cada (frecuencia x estrato de complejidad)
    D · Oráculo          el mejor para cada serie; no alcanzable, es la cota

**B es el control que importa.** Si C no le gana a B, el índice no aporta nada
que la frecuencia no diera ya gratis, y hay que decirlo.

Los cortes de los estratos salen del ENTRENAMIENTO y se aplican a prueba: usar
los cuantiles de todo el conjunto seria fuga de información.

Se repite sobre muchas particiones aleatorias para no depender de un sorteo
afortunado.

Salidas:
    enrutamiento_resultados.xlsx
    figuras/fig11_enrutamiento.pdf

Uso:  python3 experimento_enrutamiento.py [n_particiones]
"""
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
OUT = AQUI / "figuras"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})
AZUL, NARANJA, VERDE, GRIS = "#0072B2", "#D55E00", "#009E73", "#4D4D4D"

N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 30
N_ESTRATOS = 2
# se excluyen los duplicados de benchmark y el que no cubre todas las frecuencias
FUERA = {"naive", "snaive", "Holt-Winters"}


def cargar():
    d = pd.read_csv(MOD / "owa_modelos_por_serie.csv")
    d = d[~d.modelo.isin(FUERA)]
    idx = pd.read_excel(RES / "df_features_complexity.xlsx",
                        usecols=["serie", "complexity_index"])
    d = d.merge(idx, on="serie", how="inner").dropna(subset=["smape", "mase"])

    # diseño completo: solo series donde TODOS los modelos tienen valor
    n_mod = d.modelo.nunique()
    ok = d.groupby("serie").modelo.nunique() == n_mod
    d = d[d.serie.isin(ok[ok].index)]
    return d, n_mod


def owa(sub, base):
    """OWA de un conjunto de elecciones contra el Naive2 de esas mismas series."""
    if len(sub) == 0:
        return np.nan
    s, m = sub.smape.mean(), sub.mase.mean()
    bs, bm = base.smape.mean(), base.mase.mean()
    return .5 * (s / bs + m / bm) if bs and bm else np.nan


def mejor(tabla):
    """Modelo con menor OWA dentro de una tabla ya filtrada."""
    n2 = tabla[tabla.modelo == "naive2"]
    if n2.empty:
        return None
    bs, bm = n2.smape.mean(), n2.mase.mean()
    g = tabla.groupby("modelo").agg(s=("smape", "mean"), m=("mase", "mean"))
    return (.5 * (g.s / bs + g.m / bm)).idxmin()


def una_particion(d, semilla, n_est):
    series = d[["serie", "frecuencia"]].drop_duplicates()
    rng = np.random.default_rng(semilla)
    # partición estratificada por frecuencia, mitad y mitad
    tr_ids = []
    for _, g in series.groupby("frecuencia"):
        k = len(g) // 2
        tr_ids += list(rng.choice(g.serie.values, k, replace=False))
    tr_ids = set(tr_ids)
    TR = d[d.serie.isin(tr_ids)]
    TE = d[~d.serie.isin(tr_ids)]
    base_te = TE[TE.modelo == "naive2"]

    fila = {"semilla": semilla, "estratos": n_est, "n_train": TR.serie.nunique(),
            "n_test": TE.serie.nunique()}

    # ── A · un modelo fijo para todo
    mA = mejor(TR)
    fila["A_fijo"] = owa(TE[TE.modelo == mA], base_te)
    fila["A_modelo"] = mA

    # ── B · el mejor de cada frecuencia
    reglaB = {f: mejor(g) for f, g in TR.groupby("frecuencia")}
    selB = pd.concat([TE[(TE.frecuencia == f) & (TE.modelo == m)]
                      for f, m in reglaB.items() if m])
    fila["B_frecuencia"] = owa(selB, base_te)
    fila["B_n_modelos"] = len(set(reglaB.values()))

    # ── C · el mejor de cada (frecuencia x estrato del índice)
    partes, reglaC = [], {}
    for f, gtr in TR.groupby("frecuencia"):
        u = gtr[["serie", "complexity_index"]].drop_duplicates()
        # cortes calculados SOLO con entrenamiento
        cortes = np.quantile(u.complexity_index, np.linspace(0, 1, n_est + 1))
        cortes[0], cortes[-1] = -np.inf, np.inf
        cortes = np.unique(cortes)
        gtr = gtr.assign(est=pd.cut(gtr.complexity_index, cortes, labels=False))
        gte = TE[TE.frecuencia == f].assign(
            est=pd.cut(TE[TE.frecuencia == f].complexity_index, cortes, labels=False))
        for e, sub in gtr.groupby("est"):
            m = mejor(sub)
            if m is None:
                continue
            reglaC[(f, e)] = m
            partes.append(gte[(gte.est == e) & (gte.modelo == m)])
    selC = pd.concat(partes) if partes else pd.DataFrame()
    fila["C_indice"] = owa(selC, base_te)
    fila["C_n_modelos"] = len(set(reglaC.values()))
    fila["C_n_reglas"] = len(reglaC)

    # ── D · oráculo por serie (cota superior, no alcanzable)
    piv = TE.pivot_table(index="serie", columns="modelo", values="smape")
    elec = piv.idxmin(axis=1).rename("modelo").reset_index()
    selD = TE.merge(elec, on=["serie", "modelo"])
    fila["D_oraculo"] = owa(selD, base_te)

    # cuanto del hueco entre A y el oraculo cierra cada regla
    hueco = fila["A_fijo"] - fila["D_oraculo"]
    if hueco > 0:
        fila["B_cierra_%"] = 100 * (fila["A_fijo"] - fila["B_frecuencia"]) / hueco
        fila["C_cierra_%"] = 100 * (fila["A_fijo"] - fila["C_indice"]) / hueco
    return fila


def main():
    d, n_mod = cargar()
    print(f"{d.serie.nunique():,} series con diseño completo x {n_mod} modelos")
    print(f"{N_PART} particiones aleatorias, mitad entrenamiento / mitad prueba\n")

    # Barrido del numero de estratos: si la ventaja del indice dependiera de
    # haber elegido 4, seria un artefacto de esa eleccion y no un hallazgo.
    todo = []
    for ne in (2, 3, 4, 5):
        todo += [una_particion(d, s, ne) for s in range(N_PART)]
    TODO = pd.DataFrame(todo)
    TODO.to_excel(AQUI / "enrutamiento_barrido.xlsx", index=False)

    from scipy.stats import wilcoxon as _w
    print("═══ SENSIBILIDAD AL NUMERO DE ESTRATOS ═══")
    print(f"{'estratos':>9s} {'B (frec)':>9s} {'C (indice)':>11s} {'ventaja':>9s} "
          f"{'C mejor':>9s} {'p':>10s} {'cierra %':>9s}")
    for ne, g in TODO.groupby("estratos"):
        dd = g.B_frecuencia - g.C_indice
        pp = _w(g.C_indice, g.B_frecuencia)[1]
        print(f"{ne:9d} {g.B_frecuencia.mean():9.4f} {g.C_indice.mean():11.4f} "
              f"{dd.mean():+9.4f} {(dd>0).sum():6d}/{len(g):<3d} {pp:10.2e} "
              f"{g['C_cierra_%'].mean():8.1f}")
    print()

    R = TODO[TODO.estratos == N_ESTRATOS].copy()
    R.to_excel(AQUI / "enrutamiento_resultados.xlsx", index=False)

    print("═══ OWA EN PRUEBA (menor es mejor) ═══")
    print(f"{'politica':34s} {'media':>7s} {'de':>7s} {'min':>7s} {'max':>7s}")
    for c, nom in [("A_fijo", "A · un modelo fijo para todo"),
                   ("B_frecuencia", "B · mejor por frecuencia"),
                   ("C_indice", "C · mejor por frecuencia + índice"),
                   ("D_oraculo", "D · oráculo por serie (cota)")]:
        v = R[c]
        print(f"{nom:34s} {v.mean():7.4f} {v.std():7.4f} {v.min():7.4f} {v.max():7.4f}")

    from scipy.stats import wilcoxon
    print("\n═══ ¿EL ÍNDICE APORTA ALGO QUE LA FRECUENCIA NO DÉ? ═══")
    dif = R.B_frecuencia - R.C_indice          # positivo = C mejor
    p = wilcoxon(R.C_indice, R.B_frecuencia)[1]
    print(f"  C mejor que B en {(dif > 0).sum()}/{len(R)} particiones")
    print(f"  ventaja media de C sobre B: {dif.mean():+.4f} de OWA")
    print(f"  Wilcoxon pareado: p = {p:.4g}")
    print(f"\n  del hueco entre el modelo fijo y el oráculo:")
    print(f"    la frecuencia sola cierra   {R['B_cierra_%'].mean():5.1f} %")
    print(f"    frecuencia + índice cierra  {R['C_cierra_%'].mean():5.1f} %")
    print(f"\n  modelos distintos usados: B {R.B_n_modelos.mean():.1f}   "
          f"C {R.C_n_modelos.mean():.1f}  (sobre {R.C_n_reglas.mean():.0f} reglas)")
    print(f"  modelo fijo elegido: {R.A_modelo.mode()[0]} "
          f"({(R.A_modelo == R.A_modelo.mode()[0]).sum()}/{len(R)} particiones)")

    veredicto = ("El índice SÍ aporta sobre la frecuencia sola."
                 if p < .05 and dif.mean() > 0 else
                 "El índice NO aporta de forma significativa sobre la frecuencia sola.")
    print(f"\n  >>> {veredicto}")

    # ── figura
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.1),
                                  gridspec_kw={"width_ratios": [1.25, 1], "wspace": .34})
    datos = [R.A_fijo, R.B_frecuencia, R.C_indice, R.D_oraculo]
    nombres = ["A\nfijo", "B\nfrecuencia", "C\nfrec. +\níndice", "D\noráculo"]
    bp = ax.boxplot(datos, widths=.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=1.1))
    for b, c in zip(bp["boxes"], [GRIS, AZUL, VERDE, "#BFBFBF"]):
        b.set(facecolor=c, alpha=.65, linewidth=.7)
    ax.set_xticklabels(nombres, fontsize=7.6)
    ax.axhline(1.0, color=NARANJA, ls="--", lw=1, zorder=0)
    ax.text(.06, 1.0, "Naive2", color=NARANJA, fontsize=7.5, va="bottom")
    ax.set_ylabel("OWA en el conjunto de prueba")
    ax.set_title(f"(a) Políticas · {N_PART} particiones", loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)

    # (b) la ventaja no depende de cuantos estratos se elijan
    res = TODO.groupby("estratos").agg(B=("B_frecuencia", "mean"),
                                       C=("C_indice", "mean"),
                                       sB=("B_frecuencia", "std"),
                                       sC=("C_indice", "std"))
    x = res.index.values
    ax2.errorbar(x, res.B, yerr=res.sB, fmt="s--", color=AZUL, ms=5, lw=1.2,
                 capsize=3, label="B · por frecuencia")
    ax2.errorbar(x, res.C, yerr=res.sC, fmt="o-", color=VERDE, ms=5, lw=1.4,
                 capsize=3, label="C · frecuencia + índice")
    ax2.set_xticks(x)
    ax2.set_xlabel("Estratos por frecuencia")
    ax2.set_ylabel("OWA en prueba")
    ax2.set_title("(b) Menos estratos, más ventaja", loc="left", fontsize=9)
    ax2.legend(frameon=False, fontsize=7.6)
    ax2.grid(axis="x", visible=False)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig11_enrutamiento.{ext}")
    plt.close(fig)
    print("\nfiguras/fig11_enrutamiento.pdf\nenrutamiento_resultados.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
