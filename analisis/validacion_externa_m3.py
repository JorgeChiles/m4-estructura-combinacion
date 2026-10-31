#!/usr/bin/env python3
"""
Validacion externa: el mismo aparato sobre las 3.003 series de M3.

La objecion que responde
------------------------
Todo el articulo vive dentro de M4. El indice se construye con M4, los estratos
se cortan con M4, las politicas se entrenan y se prueban en mitades de M4. Un
revisor de IJF va a preguntar lo obvio: cuanto de esto es M4. La pregunta no es
retorica ---los conjuntos de referencia tienen sesgos propios, y la literatura de
meta-aprendizaje esta llena de reglas que no sobreviven al cambio de corpus---.

Por que M3, y por que no se reentrena nada
------------------------------------------
M3 publico los PRONOSTICOS de los metodos que compitieron, serie por serie y
horizonte por horizonte (paquete Mcomp de Hyndman, objeto M3Forecast). Eso
permite reconstruir el error por serie y por modelo sin reajustar un solo modelo:
exactamente la tabla que el pipeline de M4 produce con owa_modelos_por_serie.csv.
La validacion externa sale, entonces, de datos publicados y no de una corrida
nueva, que es lo unico honesto cuando el punto es si la REGLA transfiere y no si
uno sabe entrenar modelos.

Dos de los 24 metodos (AAM1 y AAM2) solo pronosticaron las series trimestrales y
mensuales. El protocolo exige el pool completo en cada serie, asi que se quedan
fuera: 22 metodos sobre las 3.003 series, con NAIVE2 dentro del pool y, a la vez,
como denominador del OWA ---igual que en M4, donde naive2 tambien es candidato---.

Que se transfiere y que se vuelve a estimar
-------------------------------------------
Los descriptores se calculan con EL MISMO codigo que los calculo sobre M4
(rasgos_v2.py, extraido del notebook original). El indice se transfiere: medianas
de imputacion, medias y escalas del estandarizador, y los pesos de PC1, todo
estimado en M4 y aplicado a M3 sin tocar. Eso es lo que hace de esto una prueba
externa y no un reajuste con otro nombre. Como control se reporta tambien el
indice reestimado dentro de M3, que responde la otra pregunta: si lo que no
transfiere es el indice o la idea.

Las politicas se entrenan dentro de M3, porque eso es lo que hace cualquiera que
reciba un corpus nuevo: el protocolo de particiones es el del articulo (30
mitades aleatorias estratificadas por frecuencia) y se ejecuta llamando a la
MISMA funcion de comparacion_metaaprendizaje.py, no a una copia.

Lo que NO se puede comparar
---------------------------
Los OWA absolutos de M3 y M4 no son comparables entre si: distinto pool (22
metodos publicados en 2000 contra 26 modelos propios), distinto tamano, distintas
frecuencias ---M3 tiene 174 series OTHER y ninguna horaria, diaria o semanal--- y
distinto oraculo. Lo comparable es la ESTRUCTURA: el orden de las politicas, si
el indice le gana a la regla por frecuencia, si el clasificador se derrumba con
la perdida 0--1 y se recupera con la correcta, y que fraccion del hueco hasta el
oraculo queda sin recorrer.

Salidas:
    validacion_externa_m3.xlsx          hojas: resumen, particiones, metodos, indice
    figuras/fig17_validacion_m3.pdf
    m4-structural-complexity/results/m3_rasgos.csv          (cache)
    m4-structural-complexity/results/m3_errores_por_serie.csv (cache)

Uso:  python3 validacion_externa_m3.py [n_particiones] [--rehacer]
"""
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))

REHACER = "--rehacer" in sys.argv
N_PART = next((int(a) for a in sys.argv[1:] if a.isdigit()), 30)
# comparacion_metaaprendizaje lee sys.argv al importarse; se le deja limpio.
sys.argv = [sys.argv[0]]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import comparacion_metaaprendizaje as CM
import rasgos_v2 as RV

warnings.filterwarnings("ignore")

M3 = AQUI / "m4-structural-complexity" / "data_m3"
RES = AQUI / "m4-structural-complexity" / "results"
OUT = AQUI / "figuras"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .02,
})
AZUL, NARANJA, VERDE, MORADO, GRIS = "#0072B2", "#D55E00", "#009E73", "#7B3294", "#4D4D4D"
LILA = "#C77CFF"

# El periodo estacional de M3 con la convencion de M4. Las 174 series OTHER no
# declaran periodo, asi que van con m = 1, como las anuales.
FREQ = {"YEARLY": 1, "QUARTERLY": 4, "MONTHLY": 12, "OTHER": 1}
NOMBRE = {"YEARLY": "Yearly", "QUARTERLY": "Quarterly",
          "MONTHLY": "Monthly", "OTHER": "Other"}
FUERA_M3 = {"AAM1", "AAM2"}       # solo pronosticaron trimestral y mensual
BASE = "naive2"                   # NAIVE2 renombrado: asi corre el codigo de M4

# Lista de src/01_pca_complexity.py, en ese orden: es el que vio el PCA de M4.
FEATURES_CLUSTER_V2 = [
    "z_skew", "z_kurtosis", "z_entropy", "z_spectral_entropy",
    "z_outlier_ratio", "z_turning_points_ratio", "z_hurst",
    "z_acf1", "z_acf6", "z_acf_freq", "z_acf_decay",
    "z_trend_linearity_r2", "z_curvature_gain", "z_trend_slope",
    "trend_strength", "seasonal_strength",
    "dominant_frequency", "dominant_energy_ratio",
    "adf_pvalue", "kpss_pvalue", "stationarity_conflict",
    "diff_var_ratio", "change_points_per_length",
    "diff_skew", "diff_kurtosis", "diff_entropy",
    "diff_turning_points_ratio", "robust_entropy", "robust_outlier_ratio",
]


# =========================================================
# METRICAS  (copiadas literalmente de src/06_m4_metrics.py)
# =========================================================

def smape(real, pred):
    """sMAPE de M4, en porcentaje (0-200)."""
    real = np.asarray(real, float)
    pred = np.asarray(pred, float)
    den = np.abs(real) + np.abs(pred)
    mask = den != 0
    if mask.sum() == 0:
        return np.nan
    return float(200 * np.mean(np.abs(real[mask] - pred[mask]) / den[mask]))


def mase(train, real, pred, m):
    """MASE: error absoluto medio escalado por el naive estacional in-sample."""
    train = np.asarray(train, float)
    if len(train) <= m:
        return np.nan
    escala = np.mean(np.abs(train[m:] - train[:-m]))
    if not np.isfinite(escala) or escala == 0:
        return np.nan
    return float(np.mean(np.abs(np.asarray(real, float) - np.asarray(pred, float))) / escala)


# =========================================================
# 1 · DESCRIPTORES DE M3
# =========================================================

def cargar_m3():
    S = pd.read_csv(M3 / "m3_series.csv")
    V = pd.read_csv(M3 / "m3_valores.csv")
    tr = {k: g.sort_values("t").valor.to_numpy(float)
          for k, g in V[V.parte == "train"].groupby("id")}
    te = {k: g.sort_values("t").valor.to_numpy(float)
          for k, g in V[V.parte == "test"].groupby("id")}
    return S, tr, te


def rasgos_m3(S, tr):
    cache = RES / "m3_rasgos.csv"
    if cache.exists() and not REHACER:
        return pd.read_csv(cache)
    t0 = time.time()
    filas, cortas = [], []
    with threadpool_limits(limits=1):
        for i, r in enumerate(S.itertuples(), 1):
            f = RV.extract_features_v2(tr[r.id], r.id, FREQ[r.periodo],
                                       category=NOMBRE[r.periodo], horizon=r.h)
            if f is None:
                cortas.append(r.id)
            else:
                filas.append(f)
            if i % 500 == 0:
                print(f"    {i:5d}/{len(S)}  {time.time()-t0:5.1f} s")
    if cortas:
        print(f"    {len(cortas)} series demasiado cortas para los descriptores")
    R = pd.DataFrame(filas)
    R.to_csv(cache, index=False)
    print(f"    {len(R)} series x {R.shape[1]} columnas  ({time.time()-t0:.0f} s)")
    return R


# =========================================================
# 2 · ERROR POR SERIE Y MODELO, DE LOS PRONOSTICOS PUBLICADOS
# =========================================================

def errores_m3(S, tr, te):
    cache = RES / "m3_errores_por_serie.csv"
    if cache.exists() and not REHACER:
        return pd.read_csv(cache)
    P = pd.read_csv(M3 / "m3_pronosticos.csv")
    P = P[~P.metodo.isin(FUERA_M3)]
    P["metodo"] = P.metodo.replace({"NAIVE2": BASE})
    cols = [c for c in P.columns if c.startswith("f")]
    meta = S.set_index("id")
    filas = []
    for r in P.itertuples(index=False):
        d = meta.loc[r.id]
        h, m = int(d.h), int(d.m)
        pred = np.array([getattr(r, c) for c in cols[:h]], float)
        if not np.all(np.isfinite(pred)):
            continue
        real = te[r.id][:h]
        filas.append((NOMBRE[d.periodo], r.id, r.metodo,
                      smape(real, pred), mase(tr[r.id], real, pred, m), h))
    E = pd.DataFrame(filas, columns=["frecuencia", "serie", "modelo",
                                     "smape", "mase", "n_pron"])
    E.to_csv(cache, index=False)
    print(f"    {E.serie.nunique()} series x {E.modelo.nunique()} metodos")
    return E


# =========================================================
# 3 · EL INDICE DE M4, APLICADO A M3
# =========================================================

def indice_transferido(R3):
    """Devuelve (indice transferido, indice reestimado en M3, verificacion).

    Transferido = medianas de imputacion, medias/escalas del estandarizador y
    pesos de PC1 estimados en M4 y aplicados a M3 sin reajuste. Es la unica
    version que constituye prueba externa. Para que el ajuste sea exactamente el
    de src/01_pca_complexity.py se repite ese preprocesamiento sobre el mismo
    archivo de entrada y se verifica que reproduce el complexity_index guardado.
    """
    F4 = pd.read_excel(AQUI / "m4-structural-complexity" / "features_m4_all_v2.xlsx")
    X = F4[[c for c in FEATURES_CLUSTER_V2 if c in F4.columns]].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    sel = VarianceThreshold(threshold=0.0)
    sel.fit(X.fillna(X.median(numeric_only=True)))
    cols = list(X.columns[sel.get_support()])
    X = X[cols]

    imp = SimpleImputer(strategy="median").fit(X)
    esc = StandardScaler().fit(imp.transform(X))
    pca = PCA(n_components=0.90, random_state=42).fit(esc.transform(imp.transform(X)))

    # Verificacion: el indice recalculado tiene que ser el guardado.
    guardado = pd.read_excel(RES / "df_features_complexity.xlsx",
                             usecols=["serie", "complexity_index"])
    rehecho = -pca.transform(esc.transform(imp.transform(X)))[:, 0]
    dif_indice = float(np.abs(rehecho - guardado.set_index("serie")
                       .complexity_index.reindex(F4.serie).values).max())

    X3 = R3[cols].replace([np.inf, -np.inf], np.nan)
    trans = -pca.transform(esc.transform(imp.transform(X3)))[:, 0]

    # Control: el mismo preprocesamiento, pero estimado dentro de M3.
    imp3 = SimpleImputer(strategy="median").fit(X3)
    esc3 = StandardScaler().fit(imp3.transform(X3))
    pca3 = PCA(n_components=0.90, random_state=42).fit(esc3.transform(imp3.transform(X3)))
    propio = -pca3.transform(esc3.transform(imp3.transform(X3)))[:, 0]

    # El signo de PC1 es arbitrario. En M4 se fijo por convencion: PC1 positivo =
    # estructura explotable, asi que el indice (su inverso) crece con la
    # complejidad y correlaciona POSITIVO con el error del naive. Se alinea el
    # indice propio de M3 a esa misma convencion antes de compararlos.
    if np.corrcoef(propio, trans)[0, 1] < 0:
        propio = -propio

    pesos = pd.Series(pca.components_[0], index=cols)
    return trans, propio, dif_indice, pesos, pca.explained_variance_ratio_[0], \
        pca3.explained_variance_ratio_[0]


# =========================================================
# 4 · LAS POLITICAS, CON EL PROTOCOLO DEL ARTICULO
# =========================================================

POL = ["A_fijo", "B_frecuencia", "C_indice", "E_fforms", "Ep_costos",
       "F_fformpp", "D_oraculo"]


def correr_politicas(d, ATR, etiqueta):
    print(f"    {etiqueta}: {d.serie.nunique():,} series x "
          f"{d.modelo.nunique()} modelos x {N_PART} particiones")
    R = pd.DataFrame([CM.una_particion(d, ATR, s) for s in range(N_PART)])
    R["corpus"] = etiqueta
    return R


def perfil(T, idx, etiqueta):
    """Heterogeneidad del pool y del ganador, en un corpus.

    El enrutamiento solo puede comprar algo si el ganador CAMBIA de una serie a
    otra y si los modelos del pool se parecen poco. Estas son las cantidades que
    fijan ese margen, antes de entrenar nada: dispersion del OWA entre modelos,
    cuantos modelos quedan a tiro del mejor, cuan concentrada esta la etiqueta
    del oraculo, y en cuantas de las celdas de la politica C el ganador local es
    distinto del ganador global.
    """
    b = T[T.modelo == BASE]
    g = T.groupby("modelo").agg(s=("smape", "mean"), m=("mase", "mean"))
    owa = .5 * (g.s / b.smape.mean() + g.m / b.mase.mean())
    gan = T.loc[T.groupby("serie").smape.idxmin()].modelo
    p = gan.value_counts(normalize=True)
    H = -(p * np.log(p)).sum() / np.log(len(owa))

    glob = owa.idxmin()
    locales, celdas = set(), 0
    for f, gf in T.groupby("frecuencia"):
        corte = idx.reindex(gf.serie.unique()).median()
        for lo, hi in ((-np.inf, corte), (corte, np.inf)):
            ser = idx.index[(idx > lo) & (idx <= hi)]
            sub = gf[gf.serie.isin(ser)]
            if sub.empty:
                continue
            m = CM.mejor(sub)
            if m:
                locales.add(m)
                celdas += int(m != glob)
    n2 = T[T.modelo == BASE].set_index("serie").smape
    return {
        "corpus": etiqueta,
        "series": T.serie.nunique(),
        "modelos": len(owa),
        "mejor_modelo": glob,
        "OWA_mejor": owa.min(),
        "OWA_peor": owa.max(),
        "modelos_a_5pct_del_mejor": int((owa <= owa.min() * 1.05).sum()),
        # Si ningun candidato se dispara, la perdida 0--1 no puede fallar "por
        # unos pocos modelos que divergen", que es la objecion natural a la
        # Seccion S3. En M4 si hay candidatos que divergen; en M3 no.
        "modelos_con_OWA_mayor_2": int((owa > 2).sum()),
        "pct_pares_serie_modelo_smape_mayor_100": 100 * (T.smape > 100).mean(),
        "ganadores_distintos": gan.nunique(),
        "clase_mayoritaria_pct": 100 * p.iloc[0],
        "entropia_normalizada": H,
        "ganadores_locales_distintos": len(locales),
        "celdas_que_cambian_de_modelo": celdas,
        "spearman_indice_vs_naive2": spearmanr(
            idx.reindex(n2.index), n2).statistic,
    }


def curva_dificultad(T, idx):
    """sMAPE de naive2 por decil del indice, en multiplos de su propia media.

    Normalizar por la media del corpus es lo que permite poner las dos curvas en
    el mismo eje: no compara niveles de error ---M3 y M4 no son comparables en
    nivel--- sino la PENDIENTE, que es lo que el indice afirma capturar.
    """
    n2 = T[T.modelo == BASE].set_index("serie").smape
    i = idx.reindex(n2.index)
    dec = pd.qcut(i, 10, labels=False, duplicates="drop")
    return (n2.groupby(dec).mean() / n2.mean()).to_numpy()


def heterogeneidad(d3, idx3):
    d4, ATR4, _ = CM.cargar()
    i4 = d4.drop_duplicates("serie").set_index("serie").complexity_index
    H = pd.DataFrame([
        perfil(d4, i4, "M4 (27 modelos)"),
        perfil(d3, idx3, "M3 (22 métodos publicados)"),
    ])
    curvas = {"M4": curva_dificultad(d4, i4), "M3": curva_dificultad(d3, idx3)}
    for c in H.columns[1:]:
        v = H[c].tolist()
        f = (lambda x: f"{x:>12.3f}") if isinstance(v[0], float) else (lambda x: f"{x!s:>12}")
        print(f"    {c:38s}" + "".join(f(x) for x in v))
    return H, curvas


def cerrado(R):
    """Fraccion del hueco entre el modelo fijo y el oraculo que recorre cada
    politica, promediando la razon POR PARTICION (como el cuadro del articulo)."""
    h = R.A_fijo - R.D_oraculo
    return {c: 100 * ((R.A_fijo - R[c]) / h).mean() for c in POL}


# =========================================================
# MAIN
# =========================================================

def main():
    print("═══ 1 · M3: series y descriptores ═══")
    S, tr, te = cargar_m3()
    print(f"    {len(S)} series  " +
          "  ".join(f"{k}:{v}" for k, v in S.periodo.value_counts().items()))
    R3 = rasgos_m3(S, tr)

    print("\n═══ 2 · error por serie de los pronosticos publicados ═══")
    E = errores_m3(S, tr, te)
    n_mod = E.modelo.nunique()
    ok = E.groupby("serie").modelo.nunique() == n_mod
    E = E[E.serie.isin(ok[ok].index)]
    print(f"    {E.serie.nunique()} series con los {n_mod} metodos completos")

    print("\n═══ 3 · el indice de M4 aplicado a M3 ═══")
    trans, propio, dif_indice, pesos, var1_m4, var1_m3 = indice_transferido(R3)
    print(f"    verificacion del ajuste de M4: dif. max. con el indice "
          f"guardado = {dif_indice:.2e}")
    print(f"    varianza de PC1: M4 {100*var1_m4:.1f} %   M3 {100*var1_m3:.1f} %")
    r_idx = spearmanr(trans, propio).statistic
    print(f"    Spearman entre el indice transferido y el reestimado en M3: {r_idx:+.3f}")

    R3 = R3.assign(indice_trans=trans, indice_propio=propio)
    ATR = pd.read_excel(RES / "pc1_loadings.xlsx").feature.tolist()
    cols_r = ["serie", "indice_trans", "indice_propio"] + ATR
    d = E.merge(R3[cols_r], on="serie", how="inner").dropna(subset=["smape", "mase"])
    ok = d.groupby("serie").modelo.nunique() == n_mod
    d = d[d.serie.isin(ok[ok].index)]

    # El indice contra el error del naive, la relacion basica del articulo.
    u = d.drop_duplicates("serie").set_index("serie")
    n2 = d[d.modelo == BASE].set_index("serie").smape.reindex(u.index)
    rho_t = spearmanr(u.indice_trans, n2).statistic
    rho_p = spearmanr(u.indice_propio, n2).statistic
    print(f"    indice vs sMAPE de naive2 (Spearman): transferido {rho_t:+.3f}  "
          f"reestimado {rho_p:+.3f}")

    print("\n═══ 4 · OWA de los metodos publicados de M3 ═══")
    g = d.groupby("modelo").agg(s=("smape", "mean"), m=("mase", "mean"))
    b = d[d.modelo == BASE]
    MET = (0.5 * (g.s / b.smape.mean() + g.m / b.mase.mean())).sort_values()
    MET.name = "OWA"
    print(MET.to_frame().head(8).to_string(float_format=lambda x: f"{x:7.4f}"))

    print("\n═══ 5 · politicas sobre M3 ═══")
    Rt = correr_politicas(d.rename(columns={"indice_trans": "complexity_index"}),
                          ATR, "M3 · índice transferido de M4")
    Rp = correr_politicas(d.rename(columns={"indice_propio": "complexity_index"}),
                          ATR, "M3 · índice reestimado en M3")
    R4 = pd.read_excel(AQUI / "metaaprendizaje_resultados.xlsx")
    R4["corpus"] = "M4 (artículo)"

    NOM = {"A_fijo": "A · modelo fijo", "B_frecuencia": "B · por frecuencia",
           "C_indice": "C · frecuencia + índice", "E_fforms": "E · clasificador",
           "Ep_costos": "E' · clasificador con costos",
           "F_fformpp": "F · regresor por modelo", "D_oraculo": "D · oráculo"}
    print(f"\n{'política':30s} {'M4':>17s} {'M3 transf.':>17s} {'M3 propio':>17s}")
    for c, nom in NOM.items():
        print(f"{nom:30s} " + "".join(
            f"{X[c].mean():9.4f}±{X[c].std():<7.4f}" for X in (R4, Rt, Rp)))

    print(f"\n{'del hueco A→D, cada política recorre':30s} {'M4':>9s} "
          f"{'M3 tr.':>9s} {'M3 pr.':>9s}")
    ct = {k: cerrado(X) for k, X in (("M4", R4), ("M3t", Rt), ("M3p", Rp))}
    for c, nom in NOM.items():
        if c in ("A_fijo", "D_oraculo"):
            continue
        print(f"{nom:30s} " + "".join(f"{ct[k][c]:8.1f} %" for k in ct))

    print("\n═══ 6 · ¿se repiten las pruebas del artículo? ═══")
    pruebas = []
    for et, X in (("M4 (artículo)", R4), ("M3 transferido", Rt),
                  ("M3 reestimado", Rp)):
        for a, b_, nom in (("C_indice", "B_frecuencia", "C vs B · ¿el índice aporta?"),
                           ("Ep_costos", "C_indice", "E' vs C · ¿el aparato gana?"),
                           ("Ep_costos", "F_fformpp", "E' vs F · clasif. vs regresor"),
                           ("E_fforms", "B_frecuencia", "E vs B · ¿E se derrumba?")):
            dif = X[b_] - X[a]          # positivo = gana a (OWA menor)
            p = wilcoxon(X[a], X[b_]).pvalue
            pruebas.append({"corpus": et, "prueba": nom, "dif": dif.mean(),
                            "gana_a": int((dif > 0).sum()), "n": len(X), "p": p})
            print(f"  {et:15s} {nom:30s} dif {dif.mean():+.4f}  "
                  f"{(dif>0).sum():2d}/{len(X)}  p={p:.3g}")

    print("\n═══ 7 · ¿por qué el índice aporta menos en M3? ═══")
    het, curvas = heterogeneidad(d, u.indice_trans)

    # ── salidas
    resumen = pd.DataFrame({
        "politica": [NOM[c] for c in POL],
        "OWA_M4": [R4[c].mean() for c in POL],
        "OWA_M3_transferido": [Rt[c].mean() for c in POL],
        "OWA_M3_reestimado": [Rp[c].mean() for c in POL],
        "cierra_M4": [ct["M4"][c] for c in POL],
        "cierra_M3_transferido": [ct["M3t"][c] for c in POL],
        "cierra_M3_reestimado": [ct["M3p"][c] for c in POL],
    })
    indice = pd.DataFrame({
        "concepto": ["dif. max. al reproducir el índice de M4",
                     "varianza explicada por PC1 en M4",
                     "varianza explicada por PC1 en M3",
                     "Spearman índice transferido vs reestimado",
                     "Spearman índice transferido vs sMAPE naive2",
                     "Spearman índice reestimado vs sMAPE naive2"],
        "valor": [dif_indice, var1_m4, var1_m3, r_idx, rho_t, rho_p],
    })
    with pd.ExcelWriter(AQUI / "validacion_externa_m3.xlsx") as w:
        resumen.to_excel(w, sheet_name="resumen", index=False)
        pd.concat([R4, Rt, Rp]).to_excel(w, sheet_name="particiones", index=False)
        pd.DataFrame(pruebas).to_excel(w, sheet_name="pruebas", index=False)
        MET.to_frame().to_excel(w, sheet_name="metodos")
        indice.to_excel(w, sheet_name="indice", index=False)
        het.to_excel(w, sheet_name="heterogeneidad", index=False)
        pesos.to_frame("peso_pc1_m4").to_excel(w, sheet_name="pesos_pc1")

    figura(Rt, ct, curvas)
    print("\nvalidacion_externa_m3.xlsx\nfiguras/fig17_validacion_m3.pdf")
    return 0


def figura(Rt, ct, curvas):
    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(6.5, 2.4), layout="constrained",
        gridspec_kw=dict(width_ratios=[1.45, 1, 1]))

    cols = ["A_fijo", "B_frecuencia", "C_indice", "E_fforms", "Ep_costos",
            "F_fformpp", "D_oraculo"]
    colores = [GRIS, AZUL, VERDE, MORADO, LILA, NARANJA, "#BFBFBF"]
    bp = a1.boxplot([Rt[c] for c in cols], widths=.55, patch_artist=True,
                    showfliers=False, medianprops=dict(color="k", lw=1.1))
    for b, c in zip(bp["boxes"], colores):
        b.set(facecolor=c, alpha=.65, linewidth=.7)
    a1.set_xticklabels(["A", "B", "C", "E", "E$'$", "F", "D"], fontsize=7.6)
    a1.axhline(1.0, color=NARANJA, ls="--", lw=1, zorder=0)
    a1.text(.06, 1.0, "Naive2", color=NARANJA, fontsize=6.8, va="bottom")
    a1.set_ylabel("OWA en prueba", fontsize=8)
    a1.set_title("(a) M3, índice transferido", loc="left", fontsize=8.2)
    a1.grid(axis="x", visible=False)
    a1.tick_params(labelsize=7.4)

    # (b) la unica comparacion legitima entre corpus: no el OWA, sino que
    # fraccion del hueco hasta el oraculo recorre cada politica. El eje se corta
    # en -20: la barra de E en M4 vale -152 % y aplastaria todo lo demas.
    cl = [c for c in cols if c not in ("A_fijo", "D_oraculo")]
    x = np.arange(len(cl))
    PISO = -20
    for off, k, col, et in ((-.2, "M4", AZUL, "M4 · 27 modelos"),
                            (.2, "M3t", NARANJA, "M3 · 22 publicados")):
        v = np.array([ct[k][c] for c in cl])
        a2.bar(x + off, np.maximum(v, PISO), .4, color=col, alpha=.75, label=et)
        for xi, vi in zip(x + off, v):
            if vi < PISO:
                a2.text(xi, PISO + 1.5, f"{vi:.0f}", fontsize=6, rotation=90,
                        ha="center", va="bottom", color="w")
    a2.set_ylim(PISO, 22)
    a2.set_xticks(x)
    a2.set_xticklabels(["B", "C", "E", "E$'$", "F"], fontsize=7.6)
    a2.axhline(0, color="k", lw=.6)
    a2.set_ylabel("% del hueco A→D", fontsize=8)
    a2.set_title("(b) lo comparable", loc="left", fontsize=8.2)
    a2.legend(fontsize=6, frameon=False, loc="upper left")
    a2.grid(axis="x", visible=False)
    a2.tick_params(labelsize=7.4)

    # (c) lo que el indice SI transfiere: ordenar por dificultad. Las dos curvas
    # estan en multiplos de la media de su propio corpus, no en puntos de sMAPE.
    xd = np.arange(1, 11)
    for k, col, mk in (("M4", AZUL, "o"), ("M3", NARANJA, "s")):
        a3.plot(xd, curvas[k], color=col, marker=mk, ms=3, lw=1.2, label=k)
    a3.axhline(1.0, color="k", lw=.6, ls=":")
    a3.set_xticks([1, 5, 10])
    a3.set_xlabel("decil del índice", fontsize=7.6)
    a3.set_ylabel("sMAPE Naive2 (× media)", fontsize=8)
    a3.set_title("(c) la dificultad sí", loc="left", fontsize=8.2)
    a3.legend(fontsize=6.8, frameon=False, loc="upper left")
    a3.tick_params(labelsize=7.4)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig17_validacion_m3.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
