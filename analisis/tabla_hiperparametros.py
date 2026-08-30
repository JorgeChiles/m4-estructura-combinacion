#!/usr/bin/env python3
"""
Arma la tabla de hiperparametros del apendice, en LaTeX.

De donde sale cada cosa
-----------------------
- Modelos clasicos y de aprendizaje automatico: de la columna "Parametros"
  que el notebook guarda en cada hoja del consolidado. Es la configuracion
  REAL con la que se corrio, no la que uno cree que puso.
- Redes neuronales: la columna dice "No definidos" porque son modelos Keras
  sin get_params(). Se construyen y se lee su arquitectura efectiva.

Salidas:
    apendice_hiperparametros.tex
    hiperparametros.xlsx

Uso:  python3 tabla_hiperparametros.py
"""
import ast
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
MOD = AQUI / "TRANSPOSE_1000_RANDOM"
NB = MOD / "QUARTELY_DATA" / "Tesis_8.5_Quarterly.ipynb"

# Que claves reportar de cada modelo: las que definen el ajuste, no los
# defaults de scikit-learn que solo agregan ruido a la tabla.
CLAVES = {
    "Naive Estacional": ["periodo"],
    "Suavizado Exponencial": ["alpha_optimizado"],
    "Holt": ["alpha_optimizado", "beta_optimizado"],
    "Holt-Winters": ["alpha", "beta", "gamma", "seasonal", "seasonal_periods"],
    "ARIMA": ["p_d_q", "P_D_Q_m"],
    "SARIMA": ["p_d_q", "P_D_Q_m"],
    "ETS": ["alpha", "use_boxcox", "trend", "seasonal"],
    "ARIMA_RF": ["arima_order", "max_lag"],
    "ARIMA_NN": ["arima_order", "max_lag", "hidden_units"],
    "SARIMA_NN": ["sarima_order", "seasonal_order"],
    "Lineal": ["fit_intercept", "tol"],
    "LassoLars": ["alpha", "max_iter", "positive", "fit_intercept"],
    "Bayesiano": ["tol", "alpha_1", "lambda_1"],
    "Bagging": ["base_estimator", "n_estimators"],
    "Random Forest": ["n_estimators", "max_depth", "max_features", "min_samples_leaf"],
    "AdaBoost": ["base_estimator", "n_estimators", "learning_rate"],
    "XGBoost": ["n_estimators", "max_depth", "learning_rate", "subsample"],
    "LightGBM": ["n_estimators", "num_leaves", "learning_rate", "max_depth"],
    "SVM": ["kernel", "C", "gamma", "epsilon"],
    "KNN": ["n_neighbors", "weights", "metric"],
    "GaussianProcess": ["kernel", "alpha"],
}

# Varios modelos NO tienen hiperparametros fijos: el notebook los ajusta por
# serie con RandomizedSearchCV. Para el apendice interesa el ESPACIO DE
# BUSQUEDA, no el valor que salio en una serie cualquiera. Extraidos del
# notebook (celdas de definicion de cada modelo).
GRILLAS = {
 "LassoLars": (25, 3, r"alpha $\in$ logspace($-4$, 1, 60); max\_iter $\in$ "
               r"\{200, 500, 1000, 2000\}; fit\_intercept, positive, precompute"),
 "Bayesiano": (25, 3, r"alpha$_1$, alpha$_2$, lambda$_1$, lambda$_2$ $\in$ "
               r"logspace($-8$, $-2$, 20); tol $\in$ \{1e-6, 1e-5, 1e-4, 1e-3\}"),
 "Bagging": (30, 3, r"n\_estimators $\in$ \{50, 100, 200, 300, 500\}; max\_samples "
             r"y max\_features $\in$ \{0.5, 0.7, 0.9, 1.0\}; bootstrap"),
 "AdaBoost": (30, 3, r"n\_estimators $\in$ \{50, 100, 200, 300, 500\}; learning\_rate "
              r"$\in$ \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}; loss $\in$ \{linear, square, "
              r"exponential\}; profundidad del arbol base $\in$ \{2--6, None\}"),
 "Random Forest": (30, 3, r"n\_estimators $\in$ \{50, 100, 200, 300\}; max\_depth $\in$ "
                   r"\{None, 5, 10, 20, 30\}; min\_samples\_split, min\_samples\_leaf; "
                   r"max\_features $\in$ \{sqrt, log2, None\}"),
 "KNN": (30, 3, r"n\_neighbors $\in$ \{1, \dots, 30\}; weights $\in$ \{uniform, "
         r"distance\}; $p \in$ \{1, 2\}"),
 "SVM": (30, 3, r"kernel RBF; $C \in$ \{0.1, 1, 10\}; epsilon $\in$ \{0.01, 0.1\}; "
         r"gamma $\in$ \{scale, 0.01, 0.1\}"),
 "XGBoost": (30, 3, r"n\_estimators $\in$ \{100, 200, 300, 500\}; learning\_rate $\in$ "
             r"\{0.01, 0.05, 0.1, 0.2\}; max\_depth $\in$ \{3, 5, 7, 10\}; subsample y "
             r"colsample\_bytree $\in$ \{0.6, 0.8, 1.0\}"),
 "LightGBM": (30, 3, r"n\_estimators $\in$ \{300, 600, 900\}; learning\_rate $\in$ "
              r"\{0.02, 0.05, 0.1\}; max\_depth $\in$ \{4, 6, 8\}; num\_leaves $\in$ "
              r"\{15, 31, 63\}; min\_data\_in\_leaf, max\_bin"),
 "GaussianProcess": (30, 3, r"kernel entre combinaciones de RBF, Matérn y WhiteKernel; "
                     r"alpha $\in$ \{1e-8, 1e-6, 1e-4, 1e-3\}"),
}

FAMILIA = {
    "Naive": "Referencia", "Naive Estacional": "Referencia",
    "Suavizado Exponencial": "Estadístico", "Holt": "Estadístico",
    "Holt-Winters": "Estadístico", "ARIMA": "Estadístico",
    "SARIMA": "Estadístico", "ETS": "Estadístico",
    "ARIMA_RF": "Híbrido", "ARIMA_NN": "Híbrido", "SARIMA_NN": "Híbrido",
    "Lineal": "Aprendizaje automático", "LassoLars": "Aprendizaje automático",
    "Bayesiano": "Aprendizaje automático", "Bagging": "Aprendizaje automático",
    "Random Forest": "Aprendizaje automático", "AdaBoost": "Aprendizaje automático",
    "XGBoost": "Aprendizaje automático", "LightGBM": "Aprendizaje automático",
    "SVM": "Aprendizaje automático", "KNN": "Aprendizaje automático",
    "GaussianProcess": "Aprendizaje automático",
    "Dense NN": "Red neuronal", "ConvNet 1D": "Red neuronal", "LSTM": "Red neuronal",
    "Transformer": "Red neuronal", "LSTM_Conv1D": "Red neuronal",
}

NOMBRES = {
    "ARIMA_RF": "ARIMA-RF", "ARIMA_NN": "ARIMA-RN", "SARIMA_NN": "SARIMA-RN",
    "LSTM_Conv1D": "LSTM-ConvNet", "Dense NN": "Perceptrón",
    "Lineal": "Regresión lineal", "Bayesiano": "Regresión bayesiana",
    "GaussianProcess": "Proceso gaussiano", "Naive Estacional": "Naive estacional",
}
ORDEN = ["Referencia", "Estadístico", "Aprendizaje automático", "Híbrido", "Red neuronal"]


def leer_parametros():
    """Parametros guardados, agregados sobre varias series."""
    import openpyxl
    wb = openpyxl.load_workbook(MOD / "QUARTELY_DATA" /
                                "resultados_consolidados_quarterly.xlsx",
                                read_only=True, data_only=True)
    crudos = {}
    for hoja in wb.sheetnames[:60]:
        for fila in wb[hoja].iter_rows(values_only=True):
            if not fila or fila[0] in (None, "Modelo", "Serie", "Datos Prueba"):
                continue
            crudos.setdefault(str(fila[0]).strip(), []).append(str(fila[1]))
    wb.close()
    return crudos


def formatear(modelo, textos):
    """Extrae las claves de interes; marca las que varian entre series."""
    claves = CLAVES.get(modelo)
    if not claves:
        return None
    vistos = {k: set() for k in claves}
    for t in textos:
        t = t.replace("np.float64(", "(").replace("np.int64(", "(")
        try:
            d = ast.literal_eval(t)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for k in claves:
            if k in d:
                v = d[k]
                vistos[k].add(f"{v:.3g}" if isinstance(v, float) else str(v))
    partes = []
    for k, vals in vistos.items():
        if not vals:
            continue
        if len(vals) == 1:
            partes.append(f"{k} = {list(vals)[0]}")
        else:
            partes.append(f"{k}: por serie")
    return "; ".join(partes) if partes else None


def arquitecturas_keras():
    """Construye cada red y devuelve su arquitectura efectiva."""
    nb = json.load(open(NB, encoding="utf-8"))
    i_loop = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
                  and "for name in series_names" in "".join(c["source"]))
    g = {}
    cwd = os.getcwd()
    os.chdir(NB.parent)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for c in nb["cells"][:i_loop]:
                if c["cell_type"] != "code":
                    continue
                src = "\n".join(l for l in "".join(c["source"]).split("\n")
                                if not l.strip().startswith(("%", "!")))
                exec(compile(src, "<nb>", "exec"), g)
    finally:
        os.chdir(cwd)

    forma = (8, 6)          # ventana x rezagos, tamano tipico
    build = {
        "Dense NN":    lambda: g["modelo_MLP"](6),
        "ConvNet 1D":  lambda: g["modelo_Conv1d"](forma),
        "LSTM":        lambda: g["modelo_LSTM"](forma),
        "Transformer": lambda: g["modelo_Transformer"](forma),
        "LSTM_Conv1D": lambda: g["modelo_hibrido_LSTM_Conv1D"](forma),
    }
    out = {}
    for nombre, f in build.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mod = f()
            m = getattr(mod, "model", mod)
            capas = []
            for c in m.layers:
                t = type(c).__name__
                if t in ("InputLayer",):
                    continue
                u = getattr(c, "units", None) or getattr(c, "filters", None)
                r = getattr(c, "rate", None)
                capas.append(f"{t}({u})" if u else
                             (f"{t}({r})" if r is not None else t))
            n_par = int(sum(np.prod(w.shape) for w in m.trainable_weights))
            opt = type(m.optimizer).__name__ if getattr(m, "optimizer", None) else "?"
            out[nombre] = (" $\\to$ ".join(capas), n_par, opt)
        except Exception as e:
            out[nombre] = (f"(no se pudo construir: {type(e).__name__})", 0, "?")
    return out


def esc(s):
    """Escapa lo que rompe LaTeX, salvo en los tramos que ya son LaTeX."""
    if "$" in s or "\\_" in s:      # ya viene formateado
        return s.replace("%", r"\%")
    for a, b in [("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def main():
    crudos = leer_parametros()
    print(f"modelos con parametros guardados: {len(crudos)}")
    redes = arquitecturas_keras()
    print(f"redes reconstruidas: {sum(1 for v in redes.values() if v[1] > 0)}/5\n")

    filas = []
    for modelo, fam in FAMILIA.items():
        if modelo in redes:
            capas, npar, opt = redes[modelo]
            cfg = (f"{capas}. Optimizador {opt}, perdida MSE, "
                   f"50 épocas, lote 16, validación 20\\%. "
                   f"{npar:,} parametros entrenables")
        elif modelo == "Naive":
            cfg = "Repite la última observación"
        elif modelo in GRILLAS:
            n_iter, cv, grilla = GRILLAS[modelo]
            cfg = (f"Ajustado por serie con RandomizedSearchCV "
                   f"({n_iter} muestras, validación cruzada de {cv} pliegues) sobre: "
                   f"{grilla}")
        else:
            cfg = formatear(modelo, crudos.get(modelo, [])) or "Valores por defecto"
        filas.append({"familia": fam, "modelo": NOMBRES.get(modelo, modelo),
                      "config": cfg})

    df = pd.DataFrame(filas)
    df.to_excel(AQUI / "hiperparametros.xlsx", index=False)

    L = [r"% Tabla de hiperparametros -- generada por tabla_hiperparametros.py",
         r"% Requiere: \usepackage{booktabs}, \usepackage{longtable} y \usepackage{array}",
         r"% Las columnas van en ragged right: justificado en una columna angosta",
         r"% deja huecos y LaTeX lo reporta como underfull hbox.",
         r"\begin{longtable}{>{\raggedright\arraybackslash}p{0.21\linewidth}"
         r">{\raggedright\arraybackslash}p{0.71\linewidth}}",
         r"\caption{Configuración de los 26 modelos evaluados. Los valores "
         r"provienen de la configuración efectivamente registrada en cada "
         r"corrida, no de la documentación de las bibliotecas. "
         r"``por serie'' indica que el parámetro se selecciona automáticamente "
         r"para cada serie.}\label{tab:hiperparametros}\\",
         r"\toprule", r"\textbf{Modelo} & \textbf{Configuración} \\", r"\midrule",
         r"\endfirsthead", r"\toprule",
         r"\textbf{Modelo} & \textbf{Configuración} \\", r"\midrule", r"\endhead",
         r"\bottomrule", r"\endfoot"]
    for fam in ORDEN:
        sub = df[df.familia == fam]
        if sub.empty:
            continue
        L.append(r"\multicolumn{2}{l}{\textit{" + esc(fam) + r"}} \\[1mm]")
        for _, r in sub.iterrows():
            L.append(f"\\hspace{{.6em}}{esc(r.modelo)} & {esc(r.config)} \\\\")
        L.append(r"\addlinespace")
    L.append(r"\end{longtable}")
    (AQUI / "apendice_hiperparametros.tex").write_text("\n".join(L), encoding="utf-8")

    for fam in ORDEN:
        for _, r in df[df.familia == fam].iterrows():
            print(f"  {r.modelo:20s} {r.config[:88]}")
    print("\napendice_hiperparametros.tex\nhiperparametros.xlsx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
