#!/usr/bin/env python3
"""
Test de Friedman con post-hoc de Nemenyi sobre los 26 modelos.

Por que por rangos
------------------
El OWA se define sobre promedios y es fragil: unas pocas series con pronostico
divergente arrastran la media (ARIMA-RN pasa de MASE mediano 1.57 a MASE medio
1782 por TRES series). Un test por rangos es inmune a eso: solo importa el
orden dentro de cada serie, no cuanto se aleja el peor.

Es ademas el procedimiento estandar para comparar varios metodos sobre muchos
conjuntos de datos (Demsar 2006): Friedman para rechazar que todos rindan
igual, y Nemenyi para decir que pares difieren.

    CD = q_alpha * sqrt( k(k+1) / (6N) )

con k modelos, N series y q_alpha el cuantil del rango estudentizado dividido
por raiz de dos.

Requiere diseno completo: cada modelo evaluado en cada serie. Los modelos que
no cubren todas las series se excluyen y se informa cuales.

Con --con-hibrido se agrega el orquestador propio como modelo 28 y se produce un
diagrama aparte. Va aparte a proposito: el diagrama principal responde "como se
ordenan los 26 modelos de la tesis", y meter ahi un modelo propio cambiaria los
rangos de todos por un motivo ajeno a esa pregunta.

Salidas:
    figuras/fig10_nemenyi.pdf
    figuras/fig13_nemenyi_hibrido.pdf   (solo con --con-hibrido)
    friedman_rangos.xlsx

Uso:  python3 test_friedman.py [--con-hibrido]
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
from scipy.stats import friedmanchisquare, studentized_range

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
MOD = AQUI / "TRANSPOSE_1000_RANDOM"
OUT = AQUI / "figuras"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.linewidth": .7,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})
AZUL, NARANJA, GRIS = "#0072B2", "#D55E00", "#4D4D4D"

NOMBRES = {
    "ARIMA_RF": "ARIMA-RF", "ARIMA_NN": "ARIMA-RN", "SARIMA_NN": "SARIMA-RN",
    "LSTM_Conv1D": "LSTM-ConvNet", "Dense NN": "Perceptrón",
    "Lineal": "Regresión lineal", "Bayesiano": "Regresión bayesiana",
    "Suavizado Exponencial": "Suavizado exp.", "GaussianProcess": "Proceso gaussiano",
    "Naive Estacional": "Naive estacional", "naive2": "Naive2",
}
pub = lambda m: NOMBRES.get(m, m)

# los benchmarks propios duplican a los modelos homonimos de la tesis
DUPLICADOS = ["naive", "snaive"]

CON_HIBRIDO = "--con-hibrido" in sys.argv
# de las tres variantes que produce el orquestador entra solo la principal:
# las otras dos son configuraciones del mismo sistema y meterlas las trata
# como si fueran metodos independientes
VARIANTE_HIBRIDA = "Orquestador"


def preparar(metrica, con_hibrido=False):
    """Matriz series x modelos con casos completos."""
    d = pd.read_csv(MOD / "owa_modelos_por_serie.csv")
    d = d[~d.modelo.isin(DUPLICADOS)]
    if con_hibrido:
        h = pd.read_csv(AQUI / "hibrido_por_serie.csv")
        h = h[h.modelo == VARIANTE_HIBRIDA]
        d = pd.concat([d, h[["frecuencia", "serie", "modelo", "smape", "mase"]]],
                      ignore_index=True)
    piv = d.pivot_table(index=["frecuencia", "serie"], columns="modelo",
                        values=metrica)
    # El diseno tiene que ser completo, pero hay dos motivos distintos por
    # los que falta un valor:
    #  - ESTRUCTURAL: Holt-Winters solo existe donde m>1 (cubre el 51 %).
    #    Se excluye el modelo; no hay forma de compararlo en igualdad.
    #  - PUNTUAL: ARIMA no converge en 15 series de 4.773 (cubre el 99.7 %).
    #    Excluir el modelo por eso seria absurdo -- ARIMA es el mejor de la
    #    evaluacion. Se descartan esas series.
    cobertura = piv.notna().mean()
    fuera = list(cobertura[cobertura < .90].index)
    piv = piv.drop(columns=fuera)
    antes = len(piv)
    piv = piv.dropna()
    return piv, fuera, antes - len(piv)


def nemenyi_cd(k, n, alpha=.05):
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6 * n)), q


def camarillas(orden, rangos, cd):
    """Grupos maximales de modelos sin diferencia significativa."""
    gr = []
    for i in range(len(orden)):
        j = i
        while j + 1 < len(orden) and rangos[orden[j + 1]] - rangos[orden[i]] < cd:
            j += 1
        if j > i:
            gr.append((i, j))
    # quedarse solo con los maximales
    return [g for g in gr if not any(o != g and o[0] <= g[0] and g[1] <= o[1] for o in gr)]


def diagrama(rangos, cd, n, k, ruta, titulo):
    """Diagrama de diferencias criticas (Demsar 2006).

    Geometria: el eje arriba, con el MEJOR rango a la izquierda. Debajo del
    eje, una banda con las camarillas (grupos sin diferencia significativa);
    mas abajo, las etiquetas, mitad a cada lado para que las lineas guia no
    se crucen."""
    orden = list(rangos.sort_values().index)        # de mejor a peor
    r = rangos.to_dict()
    lo, hi = np.floor(min(r.values())) - .3, np.ceil(max(r.values())) + .3
    mitad = (len(orden) + 1) // 2

    cam = camarillas(orden, r, cd)
    y_cam = -.09                                    # banda de camarillas
    y_top = y_cam - .075 * len(cam) - .17           # primera fila de etiquetas
    paso = .27
    alto = 1.5 + (abs(y_top) + paso * mitad) * 1.05

    fig, ax = plt.subplots(figsize=(7.0, alto))
    ax.set_xlim(lo, hi)                             # mejor (menor) a la izquierda
    ax.set_ylim(y_top - paso * mitad - .45, 1.30)
    ax.axis("off")

    # eje
    ax.plot([lo, hi], [0, 0], "k-", lw=.9)
    for t in np.arange(np.ceil(lo), np.floor(hi) + 1):
        ax.plot([t, t], [0, .09], "k-", lw=.9)
        ax.text(t, .17, f"{t:.0f}", ha="center", va="bottom", fontsize=8)
    ax.text(lo, .62, titulo, ha="left", fontsize=8.5, color=GRIS)

    # barra de diferencia critica, en el hueco de la derecha
    x1 = hi - .04 * (hi - lo)
    ax.plot([x1 - cd, x1], [.95, .95], "k-", lw=2.1, solid_capstyle="butt")
    for x in (x1 - cd, x1):
        ax.plot([x, x], [.88, 1.02], "k-", lw=.9)
    ax.text(x1 - cd / 2, 1.07, f"DC = {cd:.2f}", ha="center", fontsize=7.8)

    # etiquetas: las mejores a la izquierda, las peores a la derecha
    for i, m in enumerate(orden):
        izq = i < mitad
        fila = i if izq else len(orden) - 1 - i
        y = y_top - paso * fila
        xt = lo if izq else hi
        ax.plot([r[m], r[m]], [0, y], color=GRIS, lw=.6)
        ax.plot([r[m], xt], [y, y], color=GRIS, lw=.6)
        ax.text(xt + (-.008 if izq else .008) * (hi - lo), y,
                f"{pub(m)}  ({r[m]:.1f})", va="center",
                ha="right" if izq else "left", fontsize=7.4)

    # camarillas: grupos cuyas diferencias no alcanzan la DC
    for niv, (a, b) in enumerate(cam):
        y = y_cam - .075 * niv
        ax.plot([r[orden[a]] - .04, r[orden[b]] + .04], [y, y],
                color=NARANJA, lw=3.0, solid_capstyle="round", zorder=4)

    ax.text(lo, y_top - paso * mitad - .30,
            f"k = {k} modelos · N = {n:,} series · Nemenyi al 5 %",
            fontsize=7.2, color=GRIS, ha="left")
    for ext in ("pdf", "png"):
        fig.savefig(str(ruta) + "." + ext)
    plt.close(fig)


def main():
    filas_resumen = []
    for metrica, etiqueta in (("smape", "sMAPE"), ("mase", "MASE")):
        piv, fuera, series_caidas = preparar(metrica)
        n, k = piv.shape
        print(f"\n══ {etiqueta} ══")
        print(f"  diseño completo: {n:,} series x {k} modelos")
        if fuera:
            print(f"  modelos excluidos (no aplican en todas las frecuencias): "
                  f"{[pub(f) for f in fuera]}")
        if series_caidas:
            print(f"  series descartadas por algun ajuste fallido: {series_caidas}")

        stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
        print(f"  Friedman: chi2 = {stat:,.1f}   p = {p:.3g}")

        rangos = piv.rank(axis=1).mean()
        cd, q = nemenyi_cd(k, n)
        print(f"  q(0.05, {k}, inf)/raiz(2) = {q:.3f}   ->   DC = {cd:.3f}")
        print(f"  mejor: {pub(rangos.idxmin())} ({rangos.min():.2f})   "
              f"peor: {pub(rangos.idxmax())} ({rangos.max():.2f})")

        if metrica == "smape":
            diagrama(rangos, cd, n, k, OUT / "fig10_nemenyi",
                     "Rango promedio (menor es mejor)")
            print("  figuras/fig10_nemenyi.pdf")

        t = rangos.sort_values().rename("rango_promedio").reset_index()
        t["modelo"] = t.modelo.map(pub)
        t["puesto"] = range(1, len(t) + 1)
        t["metrica"] = etiqueta
        t["dif_vs_mejor"] = (t.rango_promedio - t.rango_promedio.min()).round(3)
        t["significativo"] = t.dif_vs_mejor > cd
        filas_resumen.append(t)

    res = pd.concat(filas_resumen, ignore_index=True)
    res.to_excel(AQUI / "friedman_rangos.xlsx", index=False)

    print("\n══ RANGO PROMEDIO POR sMAPE ══")
    s = res[res.metrica == "sMAPE"]
    print(s[["puesto", "modelo", "rango_promedio", "dif_vs_mejor",
             "significativo"]].round(2).to_string(index=False))

    print("\n══ ¿coinciden los dos ordenes? ══")
    a = res[res.metrica == "sMAPE"].set_index("modelo").puesto
    b = res[res.metrica == "MASE"].set_index("modelo").puesto
    j = pd.concat([a.rename("sMAPE"), b.rename("MASE")], axis=1).dropna()
    print(f"  correlación de Spearman entre ambos rankings: "
          f"{j.sMAPE.corr(j.MASE, method='spearman'):.3f}")
    print(f"  mayor discrepancia: "
          f"{(j.sMAPE - j.MASE).abs().idxmax()} "
          f"(puesto {int(j.loc[(j.sMAPE-j.MASE).abs().idxmax(),'sMAPE'])} vs "
          f"{int(j.loc[(j.sMAPE-j.MASE).abs().idxmax(),'MASE'])})")
    print("\nfriedman_rangos.xlsx")

    if CON_HIBRIDO:
        con_hibrido()
    return 0


def con_hibrido():
    """Misma prueba con el orquestador propio como modelo adicional."""
    from scipy.stats import wilcoxon

    print("\n\n════ CON EL ORQUESTADOR PROPIO ════")
    tablas = []
    for metrica, etiqueta in (("smape", "sMAPE"), ("mase", "MASE")):
        piv, fuera, caidas = preparar(metrica, con_hibrido=True)
        n, k = piv.shape
        stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
        rangos = piv.rank(axis=1).mean()
        cd, q = nemenyi_cd(k, n)
        orden = rangos.sort_values()
        puesto = list(orden.index).index(VARIANTE_HIBRIDA) + 1

        print(f"\n══ {etiqueta} ══")
        print(f"  {n:,} series x {k} modelos   Friedman chi2 = {stat:,.1f}  p = {p:.3g}")
        print(f"  DC = {cd:.3f}")
        print(f"  orquestador: rango {orden[VARIANTE_HIBRIDA]:.2f}, "
              f"puesto {puesto} de {k}")

        # contra quien empata y a quien le gana con significancia
        dif = orden - orden[VARIANTE_HIBRIDA]
        empata = [pub(m) for m in dif.index
                  if m != VARIANTE_HIBRIDA and abs(dif[m]) < cd]
        mejores = [pub(m) for m in dif.index if dif[m] < -cd]
        print(f"  significativamente mejores que el orquestador: "
              f"{mejores if mejores else 'ninguno'}")
        print(f"  empatados (dentro de la DC): {empata}")

        if metrica == "smape":
            diagrama(rangos, cd, n, k, OUT / "fig13_nemenyi_hibrido",
                     "Rango promedio con el orquestador propio (menor es mejor)")
            print("  figuras/fig13_nemenyi_hibrido.pdf")

            # prueba pareada directa contra los tres de la cima
            for rival in ("ARIMA", "SARIMA", "ETS"):
                if rival not in piv.columns:
                    continue
                w = wilcoxon(piv[VARIANTE_HIBRIDA], piv[rival])
                gana = (piv[VARIANTE_HIBRIDA] < piv[rival]).mean()
                print(f"  Wilcoxon vs {rival:7s}: p = {w.pvalue:.3g}   "
                      f"gana en {100*gana:.1f} % de las series")

        t = orden.rename("rango_promedio").reset_index()
        t["modelo"] = t.modelo.map(pub)
        t["puesto"] = range(1, len(t) + 1)
        t["metrica"] = etiqueta
        tablas.append(t)

    pd.concat(tablas, ignore_index=True).to_excel(
        AQUI / "friedman_rangos_hibrido.xlsx", index=False)
    print("\nfriedman_rangos_hibrido.xlsx")


if __name__ == "__main__":
    raise SystemExit(main())
