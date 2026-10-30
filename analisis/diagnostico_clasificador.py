#!/usr/bin/env python3
"""
Por que fracasa el clasificador: cuatro explicaciones, y cual aguanta.

La objecion que responde
------------------------
El articulo reporta que la politica E ---un bosque aleatorio que predice CUAL
modelo gana--- queda en OWA 1.4927, peor que no enrutar, y atribuye el fracaso a
una sola causa: la etiqueta es inestable. Es una explicacion plausible y no esta
probada. Hay al menos otras tres, y un revisor las va a nombrar:

  (1) DESBALANCE DE CLASES. La etiqueta es el argmin sobre 27 modelos. Si unos
      pocos modelos ganan casi todas las series, el bosque aprende un problema
      degenerado. Es la primera objecion de cualquiera que vea un multiclase.

  (2) COSTOS DE ERROR HETEROGENEOS. La clasificacion minimiza la perdida 0--1:
      equivocarse de modelo cuesta lo mismo siempre. Pero elegir el segundo mejor
      cuesta milesimas y elegir uno que diverge cuesta ordenes de magnitud. Si el
      dano esta concentrado en un punado de series, el problema no es que la
      etiqueta sea ruido: es que se esta optimizando la perdida equivocada.

  (3) CALIBRACION. El bosque entrega una probabilidad por clase y la politica E
      la tira: se queda con el argmax. Si esas probabilidades estan calibradas,
      la baja confianza es informacion aprovechable ---abstenerse y caer a la
      regla por frecuencia--- y descartarla es una decision del protocolo, no un
      limite del enfoque.

  (4) INESTABILIDAD DE LA ETIQUETA, que es la que el articulo ya afirma.

Las cuatro predicen el mismo fracaso y se distinguen por lo que pasa al
corregirlas. Eso es lo que mide este script.

Como se separan
---------------
  Desbalance      Se reentrena con class_weight="balanced". Si el desbalance es
                  la causa, corregirlo mejora. (Prediccion contraria: empeora,
                  porque sube la tasa de eleccion de clases raras, que son justo
                  las caras.)

  Costos          Dos cortes. Primero descriptivo: que fraccion del exceso de OWA
                  sobre la politica B viene del 1 % peor de las series. Si el dano
                  esta concentrado, la perdida 0--1 es el problema. Segundo
                  correctivo: decision de Bayes con la matriz de perdida empirica
                  ---elegir el modelo que minimiza el arrepentimiento ESPERADO
                  bajo las probabilidades del mismo bosque--- y poda del espacio
                  de etiquetas a los modelos que nunca divergen.

  Calibracion     Se mide el error esperado de calibracion de la probabilidad
                  maxima, y se prueba la regla con abstencion: si la confianza no
                  llega a tau, se usa el ganador de la frecuencia.

  Etiqueta        Se mide el margen: cuanto separa al primero del segundo. Si en
                  una fraccion grande de las series la diferencia es menor que la
                  dispersion por semilla que reporta el articulo, la etiqueta es
                  en efecto casi arbitraria, y eso acota la exactitud alcanzable
                  con independencia del clasificador.

Nada de esto cambia el numero que reporta el articulo: la politica E se
reproduce igual, con las mismas particiones y la misma semilla del bosque.

Salidas:
    diagnostico_clasificador.xlsx       una hoja por bloque
    figuras/fig16_clasificador.pdf

Uso:  python3 diagnostico_clasificador.py [n_particiones]
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
from sklearn.ensemble import RandomForestClassifier

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
AZUL, NARANJA, VERDE, MORADO, GRIS = "#0072B2", "#D55E00", "#009E73", "#7B3294", "#4D4D4D"

N_PART = int(sys.argv[1]) if len(sys.argv) > 1 else 30
FUERA = {"naive", "snaive", "Holt-Winters"}
SEMILLA_RF = 42
TAUS = (0.0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
MARGENES = (0.01, 0.02, 0.05)      # margen relativo primero-segundo
N_PODA = 10                        # modelos que sobreviven a la poda


def cargar():
    """Identica a comparacion_metaaprendizaje.py: mismas series, mismo pool."""
    d = pd.read_csv(MOD / "owa_modelos_por_serie.csv")
    d = d[~d.modelo.isin(FUERA)]
    L = pd.read_excel(RES / "pc1_loadings.xlsx")
    ATR = L.feature.tolist()
    f = pd.read_excel(RES / "df_features_complexity.xlsx",
                      usecols=["serie", "complexity_index"] + ATR)
    d = d.merge(f, on="serie", how="inner").dropna(subset=["smape", "mase"])
    n_mod = d.modelo.nunique()
    ok = d.groupby("serie").modelo.nunique() == n_mod
    d = d[d.serie.isin(ok[ok].index)]
    return d, ATR, n_mod


def owa(sub, base):
    if len(sub) == 0:
        return np.nan
    bs, bm = base.smape.mean(), base.mase.mean()
    return .5 * (sub.smape.mean() / bs + sub.mase.mean() / bm) if bs and bm else np.nan


def mejor(tabla):
    n2 = tabla[tabla.modelo == "naive2"]
    if n2.empty:
        return None
    bs, bm = n2.smape.mean(), n2.mase.mean()
    g = tabla.groupby("modelo").agg(s=("smape", "mean"), m=("mase", "mean"))
    return (.5 * (g.s / bs + g.m / bm)).idxmin()


def matriz(TE, modelos, col):
    """serie x modelo -> ndarray, en el orden de `modelos`."""
    return (TE.pivot_table(index="serie", columns="modelo", values=col)
              .reindex(columns=modelos))


def arg_mejor(T, modelos, idx):
    """Indice del modelo de menor sMAPE por serie.

    Desempata por el orden de las filas del CSV, que es como lo hace
    `comparacion_metaaprendizaje.py`. No es un detalle cosmetico: en el 1,3 % de
    las series el primero y el segundo tienen el MISMO sMAPE ---modelos que
    colapsan al mismo pronostico--- y ahi la etiqueta la fija el desempate. Para
    que la politica E reproduzca el numero publicado hay que desempatar igual."""
    nom = (T.loc[T.groupby("serie").smape.idxmin()]
             .set_index("serie").modelo.reindex(idx))
    pos = {m: i for i, m in enumerate(modelos)}
    return np.array([pos[m] for m in nom])


def una_particion(d, ATR, semilla):
    series = d[["serie", "frecuencia"]].drop_duplicates()
    rng = np.random.default_rng(semilla)
    tr_ids = set()
    for _, g in series.groupby("frecuencia"):
        tr_ids |= set(rng.choice(g.serie.values, len(g) // 2, replace=False))
    TR, TE = d[d.serie.isin(tr_ids)], d[~d.serie.isin(tr_ids)]
    base_te = TE[TE.modelo == "naive2"]
    bs, bm = base_te.smape.mean(), base_te.mase.mean()

    modelos = sorted(d.modelo.unique())
    Xtr = TR.drop_duplicates("serie").set_index("serie")[ATR]
    Xte = TE.drop_duplicates("serie").set_index("serie")[ATR]
    S = matriz(TE, modelos, "smape").loc[Xte.index]      # sMAPE de prueba
    M = matriz(TE, modelos, "mase").loc[Xte.index]
    Str = matriz(TR, modelos, "smape").loc[Xtr.index]
    frec_te = TE.drop_duplicates("serie").set_index("serie").frecuencia.loc[Xte.index]
    frec_tr = TR.drop_duplicates("serie").set_index("serie").frecuencia.loc[Xtr.index]
    n = len(Xte)

    def owa_de(ix):
        """OWA de elegir el modelo ix[i] para la serie i."""
        f = np.arange(n)
        return .5 * (S.values[f, ix].mean() / bs + M.values[f, ix].mean() / bm)

    fila = {"semilla": semilla, "n_te": n}

    # ── referencia: la regla por frecuencia (politica B), y el oraculo
    reglaB = {f: mejor(g) for f, g in TR.groupby("frecuencia")}
    iB = np.array([modelos.index(reglaB[f]) for f in frec_te])
    fila["B_frecuencia"] = owa_de(iB)
    iD = arg_mejor(TE, modelos, Xte.index)
    fila["D_oraculo"] = owa_de(iD)

    # ── etiqueta y bosque base: exactamente la politica E del articulo
    ytr = arg_mejor(TR, modelos, Xtr.index)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                 random_state=SEMILLA_RF, n_jobs=-1)
    clf.fit(Xtr.values, ytr)
    P = clf.predict_proba(Xte.values)                      # n x len(clf.classes_)
    clases = clf.classes_
    iE = clases[P.argmax(1)]
    fila["E_base"] = owa_de(iE)

    # ── (1) desbalance ---------------------------------------------------
    cuenta = np.bincount(ytr, minlength=len(modelos))
    p_cl = cuenta / cuenta.sum()
    nz = p_cl[p_cl > 0]
    fila["clases_usadas"] = int((cuenta > 0).sum())
    fila["share_mayoritaria"] = p_cl.max()
    fila["entropia_norm"] = float(-(nz * np.log(nz)).sum() / np.log(len(modelos)))
    fila["clases_bajo_1pct"] = int(((p_cl > 0) & (p_cl < .01)).sum())

    cb = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                class_weight="balanced",
                                random_state=SEMILLA_RF, n_jobs=-1)
    cb.fit(Xtr.values, ytr)
    fila["E_balanceado"] = owa_de(cb.classes_[cb.predict_proba(Xte.values).argmax(1)])

    # ── (2) costos heterogeneos -------------------------------------------
    # 2a · concentracion del dano: aporte de cada serie al exceso sobre B
    f_ = np.arange(n)
    exc = .5 * ((S.values[f_, iE] - S.values[f_, iB]) / bs
                + (M.values[f_, iE] - M.values[f_, iB]) / bm) / n
    tot = exc[exc > 0].sum()
    orden = np.argsort(-exc)
    for q in (0.001, 0.01, 0.05):
        k = max(int(np.ceil(q * n)), 1)
        fila[f"dano_peor_{q:.3%}".replace(".000%", "%")] = exc[orden[:k]].sum() / tot
    # 2b · OWA recortado: se descarta, para CADA politica, su propio 1 % peor
    for nom, ix in (("E", iE), ("B", iB), ("D", iD)):
        e = .5 * (S.values[f_, ix] / bs + M.values[f_, ix] / bm)
        keep = np.argsort(e)[:int(np.floor(.99 * n))]
        fila[f"{nom}_recortado99"] = .5 * (S.values[keep, ix[keep]].mean() / bs
                                           + M.values[keep, ix[keep]].mean() / bm)
    # 2c · cuantas series reciben un modelo catastrofico (sMAPE > 2x naive2)
    j2 = modelos.index("naive2")
    cat = S.values[f_, iE] > 2 * S.values[:, j2]
    fila["frac_catastrofica_E"] = cat.mean()
    fila["frac_catastrofica_B"] = (S.values[f_, iB] > 2 * S.values[:, j2]).mean()

    # 2d · decision de Bayes con la matriz de perdida empirica, por frecuencia.
    #      L[c, m] = arrepentimiento medio de elegir m entre las series de
    #      entrenamiento cuyo ganador es c. Se elige argmin_m sum_c p(c|x) L[c,m].
    reg_tr = Str.values - Str.values[np.arange(len(Xtr)), ytr][:, None]

    def perdida(mtr):
        """L[c, m] sobre el subconjunto de entrenamiento `mtr`."""
        L = np.zeros((len(clases), len(modelos)))
        glob = reg_tr[mtr].mean(0) if mtr.sum() else reg_tr.mean(0)
        for a, c in enumerate(clases):
            sel = mtr & (ytr == c)
            L[a] = reg_tr[sel].mean(0) if sel.sum() >= 3 else glob
        return L

    iCost, iPrior = np.empty(n, int), np.empty(n, int)
    todo = np.ones(len(Xtr), bool)
    for f in frec_te.unique():
        mtr, mte = (frec_tr == f).values, (frec_te == f).values
        L = perdida(mtr)
        iCost[mte] = (P[mte] @ L).argmin(1)
        # control 1 · la misma matriz, pero con la PROPORCION de entrenamiento en
        # vez de p(c|x): mide cuanto aportan los atributos y cuanto la sola
        # estructura de costos.
        pri = np.array([(mtr & (ytr == c)).sum() for c in clases], float)
        iPrior[mte] = (pri / max(pri.sum(), 1) @ L).argmin()
    fila["E_costos"] = owa_de(iCost)
    fila["E_costos_prior"] = owa_de(iPrior)
    # control 2 · una sola matriz global, sin condicionar por frecuencia: mide
    # cuanto de la ganancia viene de conocer la frecuencia.
    LG = perdida(todo)
    fila["E_costos_global"] = owa_de((P @ LG).argmin(1))

    # 2e · poda del espacio de etiquetas a los N_PODA modelos mas seguros en
    #      entrenamiento (menor sMAPE medio): la etiqueta se redefine como el
    #      mejor DENTRO de ese conjunto, de modo que las clases caras no existen.
    seguros = np.argsort(Str.values.mean(0))[:N_PODA]
    ypod = seguros[Str.values[:, seguros].argmin(1)]
    cp = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                random_state=SEMILLA_RF, n_jobs=-1)
    cp.fit(Xtr.values, ypod)
    fila["E_podado"] = owa_de(cp.classes_[cp.predict_proba(Xte.values).argmax(1)])

    # ── (3) calibracion ----------------------------------------------------
    conf = P.max(1)
    acierto = (iE == iD)
    fila["exactitud"] = acierto.mean()
    top3 = clases[np.argsort(-P, 1)[:, :3]]
    fila["exactitud_top3"] = (top3 == iD[:, None]).any(1).mean()
    bins = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(conf, bins) - 1, 0, 9)
    ece = 0.
    for k in range(10):
        m = b == k
        if m.any():
            ece += m.mean() * abs(acierto[m].mean() - conf[m].mean())
    fila["ece"] = ece
    fila["confianza_media"] = conf.mean()
    for t in TAUS:                      # abstencion: bajo tau, se usa la regla B
        ix = np.where(conf >= t, iE, iB)
        fila[f"E_tau{t:.2f}"] = owa_de(ix)
        fila[f"abstiene_tau{t:.2f}"] = (conf < t).mean()

    # ── (4) margen de la etiqueta -----------------------------------------
    ord_s = np.sort(S.values, 1)
    m1, m2 = ord_s[:, 0], ord_s[:, 1]
    rel = np.where(m1 > 0, (m2 - m1) / m1, 0.)
    for g in MARGENES:
        fila[f"margen_bajo_{g:.0%}"] = (rel < g).mean()
    fila["margen_mediano"] = float(np.median(rel))
    fila["empate_exacto"] = float((m2 == m1).mean())
    return fila, exc, conf, acierto


def main():
    d, ATR, n_mod = cargar()
    print(f"{d.serie.nunique():,} series x {n_mod} modelos x {len(ATR)} atributos")
    print(f"{N_PART} particiones\n", flush=True)

    filas, excs, confs, aciertos = [], [], [], []
    for s in range(N_PART):
        fi, ex, co, ac = una_particion(d, ATR, s)
        filas.append(fi); excs.append(ex); confs.append(co); aciertos.append(ac)
        print(f"   particion {s:2d}  E={fi['E_base']:.4f}  "
              f"bal={fi['E_balanceado']:.4f}  costos={fi['E_costos']:.4f}  "
              f"podado={fi['E_podado']:.4f}", flush=True)
    R = pd.DataFrame(filas)

    from scipy.stats import wilcoxon
    pr = lambda t: print(t, flush=True)

    pr("\n═══ 1 · DESBALANCE DE CLASES ═══")
    pr(f"  clases efectivamente usadas       {R.clases_usadas.mean():.1f} de {n_mod}")
    pr(f"  share de la clase mayoritaria     {R.share_mayoritaria.mean():.1%}")
    pr(f"  entropia normalizada              {R.entropia_norm.mean():.3f}")
    pr(f"  clases con menos del 1 %          {R.clases_bajo_1pct.mean():.1f}")
    dif = R.E_base - R.E_balanceado
    pr(f"\n  OWA  base {R.E_base.mean():.4f}   balanceado {R.E_balanceado.mean():.4f}"
       f"   dif {dif.mean():+.4f}  p={wilcoxon(R.E_base, R.E_balanceado)[1]:.3g}")
    pr(f"  {'corregir el desbalance MEJORA' if dif.mean() > 0 else 'corregir el desbalance EMPEORA'}")

    pr("\n═══ 2 · COSTOS DE ERROR HETEROGENEOS ═══")
    pr("  del exceso de OWA sobre la politica B, aporta:")
    for q in ("0.1%", "1%", "5%"):
        c = [c for c in R.columns if c.startswith("dano_peor") and q in c]
        if c:
            pr(f"     el {q:>4s} peor de las series   {R[c[0]].mean():6.1%}")
    pr(f"\n  series con modelo catastrofico (sMAPE > 2x naive2)")
    pr(f"     politica E   {R.frac_catastrofica_E.mean():.2%}")
    pr(f"     politica B   {R.frac_catastrofica_B.mean():.2%}")
    pr(f"\n  OWA completo y recortado al 99 %:")
    for nom, c in (("E clasificador", "E"), ("B frecuencia", "B"), ("D oraculo", "D")):
        comp = R[{"E": "E_base", "B": "B_frecuencia", "D": "D_oraculo"}[c]].mean()
        pr(f"     {nom:16s} {comp:7.4f}   recortado {R[f'{c}_recortado99'].mean():7.4f}")
    pr(f"\n  corregir la perdida:")
    pr(f"     decision de Bayes con costos   {R.E_costos.mean():.4f}"
       f"   (base {R.E_base.mean():.4f})  p={wilcoxon(R.E_base, R.E_costos)[1]:.3g}")
    pr(f"     etiquetas podadas a {N_PODA:2d}         {R.E_podado.mean():.4f}"
       f"   p={wilcoxon(R.E_base, R.E_podado)[1]:.3g}")
    pr(f"\n  controles sobre la decision de Bayes:")
    pr(f"     sin atributos, solo la prior    {R.E_costos_prior.mean():.4f}"
       f"   (aporte de los atributos {R.E_costos_prior.mean()-R.E_costos.mean():+.4f})")
    pr(f"     matriz global, sin frecuencia   {R.E_costos_global.mean():.4f}"
       f"   (aporte de la frecuencia  {R.E_costos_global.mean()-R.E_costos.mean():+.4f})")

    pr("\n═══ 3 · CALIBRACION ═══")
    pr(f"  exactitud del clasificador        {R.exactitud.mean():.1%}")
    pr(f"  exactitud en el top-3             {R.exactitud_top3.mean():.1%}")
    pr(f"  confianza media                   {R.confianza_media.mean():.3f}")
    pr(f"  error esperado de calibracion     {R.ece.mean():.3f}")
    pr(f"\n  regla con abstencion (bajo tau se usa la regla por frecuencia):")
    pr(f"     {'tau':>5s} {'abstiene':>9s} {'OWA':>8s}")
    for t in TAUS:
        pr(f"     {t:5.2f} {R[f'abstiene_tau{t:.2f}'].mean():8.1%} "
           f"{R[f'E_tau{t:.2f}'].mean():8.4f}")

    pr("\n═══ 4 · MARGEN DE LA ETIQUETA ═══")
    pr(f"  series con empate EXACTO entre 1.o y 2.o  {R.empate_exacto.mean():.1%}")
    pr(f"  margen relativo mediano entre 1.o y 2.o   {R.margen_mediano.mean():.1%}")
    for g in MARGENES:
        pr(f"  series con margen menor al {g:.0%}            "
           f"{R[f'margen_bajo_{g:.0%}'].mean():.1%}")

    with pd.ExcelWriter(AQUI / "diagnostico_clasificador.xlsx") as w:
        R.to_excel(w, sheet_name="particiones", index=False)
        pd.DataFrame({"tau": TAUS,
                      "abstiene": [R[f"abstiene_tau{t:.2f}"].mean() for t in TAUS],
                      "owa": [R[f"E_tau{t:.2f}"].mean() for t in TAUS]}
                     ).to_excel(w, sheet_name="abstencion", index=False)

    # ── figura: concentracion del dano y curva de abstencion
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7))
    # La curva se calcula DENTRO de cada particion y despues se resume. Agrupar
    # las 30 no sirve: una sola serie de la particion que se dispara a OWA 15,2
    # se lleva dos tercios del exceso acumulado y tapa el resto.
    rej = np.logspace(-2.2, 2, 160)
    curvas = []
    for e in excs:
        p = np.sort(e[e > 0])[::-1]
        if len(p) < 20:
            continue
        cum = 100 * np.cumsum(p) / p.sum()
        xi = 100 * np.arange(1, len(p) + 1) / len(e)
        curvas.append(np.interp(rej, xi, cum, left=cum[0], right=100))
    C = np.vstack(curvas)
    a1.fill_between(rej, np.percentile(C, 25, 0), np.percentile(C, 75, 0),
                    color=MORADO, alpha=.18, lw=0)
    a1.plot(rej, np.median(C, 0), color=MORADO, lw=1.4)
    a1.plot([.01, 100], [.01, 100], color=GRIS, ls=":", lw=.8)
    a1.set_xscale("log")
    a1.set_xlim(.01, 100)
    a1.set_ylim(0, 100)
    a1.set_xlabel("% de series que pierden, de peor a mejor")
    a1.set_ylabel("% del exceso de OWA")
    a1.set_title("El daño está concentrado", loc="left", fontsize=8.5)
    a2.plot([100 * R[f"abstiene_tau{t:.2f}"].mean() for t in TAUS],
            [R[f"E_tau{t:.2f}"].mean() for t in TAUS], "o-", color=AZUL, lw=1.3, ms=3.5)
    a2.axhline(R.B_frecuencia.mean(), color=NARANJA, ls="--", lw=1)
    a2.text(2, R.B_frecuencia.mean(), "regla por frecuencia", color=NARANJA,
            fontsize=7.2, va="bottom")
    a2.set_xlabel("% de series en que se abstiene")
    a2.set_ylabel("OWA en prueba")
    a2.set_title("Abstenerse por baja confianza", loc="left", fontsize=8.5)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig16_clasificador.{ext}")
    plt.close(fig)
    pr("\nfiguras/fig16_clasificador.pdf\ndiagnostico_clasificador.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
