#!/usr/bin/env python3
"""
N-BEATS sobre las mismas series, para que el banco no quede anclado en 2019.

Por que
-------
Los tres revisores senalan lo mismo: el banco de 26 modelos usa arquitecturas
neuronales genericas entrenadas serie por serie (perceptron, ConvNet 1D, LSTM,
Transformer) y no incluye ningun modelo GLOBAL moderno. N-BEATS
(Oreshkin, Carpov, Chapados y Bengio, ICLR 2020) es el caso testigo: entrena una
sola red sobre muchas series y reporta mejoras sobre el ganador de M4 sin
ingenieria de caracteristicas.

Que cambia y que no
-------------------
Agregar N-BEATS puede cambiar QUE modelos componen los conjuntos que las
politicas eligen. Lo que no puede cambiar es la comparacion ENTRE politicas
--elegir contra combinar, frecuencia contra frecuencia x estrato-- porque todas
operan sobre el mismo banco, sea cual sea.

Arquitectura
------------
Version generica del articulo original, sin los bloques interpretables:
pilas de bloques totalmente conectados, cada uno con salida doble
(backcast y forecast) y conexion residual por sustraccion. El backcast que
un bloque explica se resta de la entrada del siguiente, de modo que cada bloque
modela el residuo del anterior; los forecast se suman.

    entrada L -> [bloque_1 -> bloque_2 -> ... -> bloque_B] -> suma de forecasts

Se entrena una red por frecuencia sobre TODAS las series de esa frecuencia
(aprendizaje cruzado), con perdida sMAPE directa, que es la metrica de la
competencia. Las series se normalizan por el ultimo valor de la ventana, de modo
que la red aprende formas y no niveles.

Salida:
    nbeats_pronosticos.csv   frecuencia, serie, h, pred

Uso:  python3 entrenar_nbeats.py [frecuencia ...]
"""
import importlib
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
COMPLEJIDAD = AQUI / "m4-structural-complexity"
sys.path.insert(0, str(COMPLEJIDAD / "src"))
M4 = importlib.import_module("06_m4_metrics")
DATA = COMPLEJIDAD / "data"

# La semilla es variable a proposito. Este trabajo documenta que una sola
# corrida de una red no distingue una diferencia real del ruido de
# inicializacion (Seccion sobre varianza por semilla), de modo que reportar
# N-BEATS a partir de UNA corrida seria incoherente con su propio hallazgo.
SEMILLA = int(os.environ.get("NB_SEMILLA", "42"))
SUF = os.environ.get("NB_SUFIJO", "")
torch.manual_seed(SEMILLA)
np.random.seed(SEMILLA)

# frecuencia: horizonte, multiplicador de la ventana de entrada
# El articulo original usa L = 2H..7H; se toma 4H, que es su valor central y
# evita ajustar esta eleccion a los resultados.
FRECS = {"Yearly": 6, "Quarterly": 8, "Monthly": 18,
         "Weekly": 13, "Daily": 14, "Hourly": 48}
MULT_L = 4
# como se ancla la escala de cada ventana: "media" (toda la entrada),
# "reciente" (los ultimos H puntos) o "ultimo" (el ultimo valor)
ANCLA = os.environ.get("NB_ANCLA", "reciente")
EPOCAS = int(os.environ.get("NB_EPOCAS", "30"))
MAXWIN = int(os.environ.get("NB_MAXWIN", "150000"))
LOTE = 512
ANCHO = 256          # unidades por capa densa
CAPAS = 4            # capas densas por bloque
BLOQUES = 6          # bloques de la pila


class Bloque(nn.Module):
    """Bloque generico: cuatro capas densas y salida doble."""

    def __init__(self, L, H, ancho, capas):
        super().__init__()
        capa = []
        e = L
        for _ in range(capas):
            capa += [nn.Linear(e, ancho), nn.ReLU()]
            e = ancho
        self.tronco = nn.Sequential(*capa)
        self.back = nn.Linear(ancho, L)
        self.fore = nn.Linear(ancho, H)

    def forward(self, x):
        t = self.tronco(x)
        return self.back(t), self.fore(t)


class NBeats(nn.Module):
    """Pila con conexiones residuales dobles: el backcast se resta, el
    forecast se suma. Es el mecanismo que distingue a N-BEATS de un
    perceptron profundo cualquiera."""

    def __init__(self, L, H, ancho=ANCHO, capas=CAPAS, bloques=BLOQUES):
        super().__init__()
        self.bloques = nn.ModuleList(Bloque(L, H, ancho, capas) for _ in range(bloques))

    def forward(self, x):
        resid = x
        salida = 0.0
        for b in self.bloques:
            back, fore = b(resid)
            resid = resid - back
            salida = salida + fore
        return salida


def perdida_smape(pred, real):
    den = torch.abs(real) + torch.abs(pred)
    return (200.0 * torch.abs(real - pred) / torch.clamp(den, min=1e-8)).mean()


def ventanas(series, L, H, maxwin, rng):
    """Ventanas normalizadas por la MEDIA de la entrada.

    La normalizacion es por ventana y no global: la red debe aprender la forma
    de la continuacion, no la escala de la serie, que varia varios ordenes de
    magnitud entre series de M4.

    Se divide por la media y no por el ultimo valor. Con el ultimo valor, una
    ventana que termina en un valle estacional queda dividida por un numero
    chico y su objetivo escalado explota; en mensual, con m=12, eso pasa a
    menudo y la red no converge (sMAPE de validacion 30,1 contra 3 a 12 en las
    demas frecuencias). La media de la ventana es insensible a donde caiga el
    corte dentro del ciclo."""
    por_serie = max(1, maxwin // max(1, len(series)))
    X, Y = [], []
    for y in series:
        n = len(y) - L - H + 1
        if n <= 0:
            continue
        pos = range(n) if n <= por_serie else rng.choice(n, por_serie, replace=False)
        for i in pos:
            v = y[i:i + L + H].astype(np.float32)
            esc = (abs(v[:L].mean()) if ANCLA == "media" else
                   abs(v[L - H:L].mean()) if ANCLA == "reciente" else abs(v[L - 1]))
            if not np.isfinite(esc) or esc < 1e-8:
                continue
            X.append(v[:L] / esc)
            Y.append(v[L:] / esc)
    return np.array(X, np.float32), np.array(Y, np.float32)


def ejecutar(nombre, H):
    t0 = time.time()
    L = MULT_L * H
    tr = M4.load_m4(DATA / f"{nombre}-train.csv")
    te = M4.load_m4(DATA / f"{nombre}-test.csv")

    # se entrena con TODAS las series de la frecuencia; se evalua solo en las
    # que el banco de 26 modelos cubre, para que la comparacion sea sobre el
    # mismo conjunto
    P = pd.read_pickle(AQUI / "pronosticos_modelos.pkl.gz")
    evaluar = sorted(set(P[P.frecuencia == nombre].serie.unique()) & set(te))
    largas = [np.asarray(v, float) for v in tr.values() if len(v) >= L + H + 1]
    print(f"── {nombre}: L={L}, {len(tr):,} series ({len(largas):,} utiles para "
          f"entrenar), {len(evaluar):,} a evaluar", flush=True)
    if not largas:
        print("   ninguna serie alcanza el largo minimo; se omite", flush=True)
        return pd.DataFrame()

    rng = np.random.default_rng(SEMILLA)
    X, Y = ventanas(largas, L, H, MAXWIN, rng)
    print(f"   {len(X):,} ventanas de {L} -> {H}", flush=True)

    perm = rng.permutation(len(X))
    corte = int(len(X) * .9)
    itr, iva = perm[:corte], perm[corte:]
    Xt = torch.tensor(X[itr]); Yt = torch.tensor(Y[itr])
    Xv = torch.tensor(X[iva]); Yv = torch.tensor(Y[iva])

    modelo = NBeats(L, H)
    opt = torch.optim.Adam(modelo.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCAS)
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt, Yt),
                                     batch_size=LOTE, shuffle=True)
    mejor, estado = np.inf, None
    for ep in range(EPOCAS):
        modelo.train()
        for xb, yb in dl:
            opt.zero_grad()
            perdida_smape(modelo(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
        sched.step()
        modelo.eval()
        with torch.no_grad():
            lv = float(np.mean([perdida_smape(modelo(Xv[i:i + 2048]),
                                              Yv[i:i + 2048]).item()
                                for i in range(0, len(Xv), 2048)]))
        if lv < mejor - 1e-6:
            mejor, estado = lv, {k: v.clone() for k, v in modelo.state_dict().items()}
        if ep % 5 == 0 or ep == EPOCAS - 1:
            print(f"   epoca {ep+1:3d}/{EPOCAS}  sMAPE val {lv:7.3f}"
                  f"  [{time.time()-t0:.0f} s]", flush=True)
    if estado:
        modelo.load_state_dict(estado)
    modelo.eval()

    filas, saltadas = [], 0
    with torch.no_grad():
        for sid in evaluar:
            y = np.asarray(tr[sid], float)
            if len(y) < L:
                # series mas cortas que la ventana: se rellena por la izquierda
                # repitiendo el primer valor, que es la convencion habitual y
                # evita descartarlas
                y = np.concatenate([np.full(L - len(y), y[0]), y])
                saltadas += 1
            v = y[-L:].astype(np.float32)
            esc = (abs(v.mean()) if ANCLA == "media" else
                   abs(v[-H:].mean()) if ANCLA == "reciente" else abs(v[-1]))
            if not np.isfinite(esc) or esc < 1e-8:
                continue
            p = modelo(torch.tensor(v / esc)[None, :]).numpy().ravel() * esc
            for j, val in enumerate(p):
                filas.append((nombre, sid, j + 1, float(val)))
    print(f"   listo en {time.time()-t0:.0f} s"
          + (f", {saltadas} series rellenadas por ser mas cortas que L" if saltadas else ""),
          flush=True)
    return pd.DataFrame(filas, columns=["frecuencia", "serie", "h", "pred"])


def main():
    pedidas = sys.argv[1:] or list(FRECS)
    partes = [ejecutar(f, FRECS[f]) for f in pedidas if f in FRECS]
    T = pd.concat([p for p in partes if len(p)], ignore_index=True)
    sal = AQUI / f"nbeats_pronosticos{SUF}.csv"
    if sal.exists() and len(pedidas) < len(FRECS):
        viejo = pd.read_csv(sal)
        T = pd.concat([viejo[~viejo.frecuencia.isin(pedidas)], T], ignore_index=True)
    T.to_csv(sal, index=False)
    print(f"\n{len(T):,} filas -> {sal.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
