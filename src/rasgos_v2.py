"""Los 29 descriptores estructurales, tal como se calcularon sobre M4.

ESTE ARCHIVO NO SE EDITA A MANO. Lo genera `extraer_rasgos.py` copiando
literalmente las celdas 0, 2 y 4 del notebook CLASIFICACION_V3_All.ipynb, que es
donde se calcularon los descriptores de las 100.000 series de M4. Se extrae a
modulo para que la validacion externa sobre M3 use EL MISMO codigo y no una
reimplementacion: si los descriptores se calcularan distinto, la transferencia
del indice no probaria nada.

Lo unico que se quita de las celdas es `import matplotlib.pyplot as plt`, que
aqui no hace falta y abre un backend grafico al importar.
"""

# ===== celda 0 del notebook =====
import os
import numpy as np
import pandas as pd
import traceback

from scipy.stats import skew, kurtosis
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.seasonal import STL

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

from threadpoolctl import threadpool_limits


# ===== celda 2 del notebook =====
# =========================================================
# INTENTO DE IMPORTAR "ruptures"
# =========================================================
try:
    # Intentamos importar la librería ruptures,
    # que sirve para detectar puntos de cambio estructural en series de tiempo.
    import ruptures as rpt

    # Si la importación funciona, dejamos una bandera en True.
    RUPTURES_OK = True

except Exception:
    # Si falla la importación, dejamos la bandera en False
    # para que el resto del código no se rompa.
    RUPTURES_OK = False


# =========================================================
# FUNCIÓN: safe_float
# =========================================================
def safe_float(x):
    # Esta función intenta convertir cualquier valor x a float.
    # Si no se puede, devuelve np.nan.
    try:
        # Convertimos x a float.
        x = float(x)

        # Si x es finito (no inf, no -inf, no nan), lo devolvemos.
        # Si no es finito, devolvemos np.nan.
        return x if np.isfinite(x) else np.nan

    except Exception:
        # Si cualquier cosa falla, devolvemos np.nan.
        return np.nan


# =========================================================
# FUNCIÓN: zscore_series
# =========================================================
def zscore_series(x):
    # Convertimos x a numpy array de tipo float.
    x = np.asarray(x, dtype=float)

    # Nos quedamos solo con valores finitos.
    x = x[np.isfinite(x)]

    # Si la serie queda vacía, devolvemos la serie vacía.
    if len(x) == 0:
        return x

    # Calculamos la media de la serie.
    mu = np.mean(x)

    # Calculamos la desviación estándar.
    sd = np.std(x)

    # Si la desviación estándar es cero,
    # significa que todos los valores son iguales.
    if sd == 0:
        # En ese caso devolvemos un vector de ceros del mismo tamaño.
        return np.zeros_like(x)

    # Si sd > 0, devolvemos la serie estandarizada:
    # (x - media) / desviación estándar
    return (x - mu) / sd


# =========================================================
# FUNCIÓN: robust_scale_series
# =========================================================
def robust_scale_series(x):
    # Convertimos a array float.
    x = np.asarray(x, dtype=float)

    # Filtramos valores finitos.
    x = x[np.isfinite(x)]

    # Si no hay datos, devolvemos el array vacío.
    if len(x) == 0:
        return x

    # Calculamos la mediana.
    med = np.median(x)

    # Calculamos percentiles 25 y 75.
    q1, q3 = np.percentile(x, [25, 75])

    # Calculamos el rango intercuartílico.
    iqr = q3 - q1

    # Si el IQR es cero, no podemos dividir.
    if iqr == 0:
        # Entonces centramos respecto a la mediana.
        return x - med

    # Si IQR > 0, escalamos de manera robusta:
    # (x - mediana) / IQR
    return (x - med) / iqr


# =========================================================
# FUNCIÓN: shannon_entropy
# =========================================================
def shannon_entropy(x, bins=20):
    # Convertimos a array float.
    x = np.asarray(x, dtype=float)

    # Nos quedamos con valores finitos.
    x = x[np.isfinite(x)]

    # Si hay muy pocos datos, no calculamos entropía.
    if len(x) < 5:
        return np.nan

    # Construimos un histograma con "bins" cajas.
    # density=True hace que se calcule una densidad.
    hist, _ = np.histogram(x, bins=bins, density=True)

    # Nos quedamos solo con valores positivos del histograma.
    hist = hist[hist > 0]

    # Si no hay bins positivos, devolvemos nan.
    if len(hist) == 0:
        return np.nan

    # Normalizamos para que sumen 1 y puedan interpretarse como probabilidades.
    p = hist / hist.sum()

    # Calculamos entropía de Shannon: -sum(p log p)
    return safe_float(-np.sum(p * np.log(p)))


# =========================================================
# FUNCIÓN: spectral_entropy
# =========================================================
def spectral_entropy(x):
    try:
        # Convertimos a array float.
        x = np.asarray(x, dtype=float)

        # Filtramos finitos.
        x = x[np.isfinite(x)]

        # Si la serie es demasiado corta, devolvemos nan.
        if len(x) < 8:
            return np.nan

        # Calculamos el periodograma:
        # f = frecuencias
        # pxx = potencia espectral
        f, pxx = periodogram(x)

        # Filtramos potencias finitas.
        pxx = pxx[np.isfinite(pxx)]

        # Filtramos potencias positivas.
        pxx = pxx[pxx > 0]

        # Si no queda nada, devolvemos nan.
        if len(pxx) == 0:
            return np.nan

        # Normalizamos la energía total para obtener "probabilidades espectrales".
        p = pxx / pxx.sum()

        # Calculamos entropía espectral.
        return safe_float(-np.sum(p * np.log(p)))

    except Exception:
        # Si algo falla, devolvemos nan.
        return np.nan


# =========================================================
# FUNCIÓN: dominant_frequency_features
# =========================================================
def dominant_frequency_features(x):
    try:
        # Convertimos a array float.
        x = np.asarray(x, dtype=float)

        # Filtramos finitos.
        x = x[np.isfinite(x)]

        # Si hay pocos datos, no calculamos nada.
        if len(x) < 8:
            return np.nan, np.nan

        # Calculamos periodograma.
        f, pxx = periodogram(x)

        # Si solo hay frecuencia cero o no hay suficientes puntos, devolvemos nan.
        if len(f) <= 1:
            return np.nan, np.nan

        # Quitamos la frecuencia cero, porque no nos interesa como frecuencia dominante.
        f = f[1:]
        pxx = pxx[1:]

        # Si no hay energía o la energía total es cero, devolvemos nan.
        if len(pxx) == 0 or np.sum(pxx) == 0:
            return np.nan, np.nan

        # Buscamos el índice de la mayor potencia espectral.
        idx = np.argmax(pxx)

        # Devolvemos:
        # 1) la frecuencia dominante
        # 2) la proporción de energía de esa frecuencia respecto al total
        return safe_float(f[idx]), safe_float(pxx[idx] / np.sum(pxx))

    except Exception:
        return np.nan, np.nan


# =========================================================
# FUNCIÓN: turning_points_ratio
# =========================================================
def turning_points_ratio(x):
    # Convertimos a array float.
    x = np.asarray(x, dtype=float)

    # Filtramos finitos.
    x = x[np.isfinite(x)]

    # Si la serie tiene menos de 3 puntos, no hay puntos de giro.
    if len(x) < 3:
        return np.nan

    # Contador de turning points.
    c = 0

    # Recorremos cada punto interno.
    for i in range(1, len(x) - 1):
        # Si el punto actual es máximo local o mínimo local, sumamos 1.
        if (x[i] > x[i - 1] and x[i] > x[i + 1]) or (x[i] < x[i - 1] and x[i] < x[i + 1]):
            c += 1

    # Devolvemos la proporción respecto al número máximo posible de turning points.
    return safe_float(c / max(1, len(x) - 2))


# =========================================================
# FUNCIÓN: outlier_ratio_robust
# =========================================================
def outlier_ratio_robust(x, thr=3.5):
    # Convertimos a array float.
    x = np.asarray(x, dtype=float)

    # Filtramos finitos.
    x = x[np.isfinite(x)]

    # Si hay muy pocos datos, devolvemos nan.
    if len(x) < 5:
        return np.nan

    # Calculamos la mediana.
    med = np.median(x)

    # Calculamos la MAD (median absolute deviation).
    mad = np.median(np.abs(x - med))

    # Si MAD = 0, no hay dispersión robusta.
    if mad == 0:
        return 0.0

    # Calculamos robust z-score.
    rz = 0.6745 * (x - med) / mad

    # Devolvemos la proporción de valores cuyo robust z-score
    # supera el umbral thr.
    return safe_float(np.mean(np.abs(rz) > thr))


# =========================================================
# FUNCIÓN: hurst_exponent
# =========================================================
def hurst_exponent(ts, max_lag=20):
    try:
        # Convertimos a array float.
        ts = np.asarray(ts, dtype=float)

        # Filtramos finitos.
        ts = ts[np.isfinite(ts)]

        # Si la serie es muy corta, no calculamos Hurst.
        if len(ts) < max_lag + 5:
            return np.nan

        # Definimos rezagos desde 2 hasta max_lag o la mitad de la serie.
        lags = range(2, min(max_lag, len(ts) // 2))

        # Aquí guardaremos la medida tau para cada lag.
        tau = []

        # Para cada lag, calculamos la desviación del incremento.
        for lag in lags:
            diff = ts[lag:] - ts[:-lag]
            sd = np.std(diff)

            # Guardamos sqrt(sd) si sd > 0; si no, nan.
            tau.append(np.sqrt(sd) if sd > 0 else np.nan)

        # Convertimos a arrays.
        lags = np.array(list(lags), dtype=float)
        tau = np.array(tau, dtype=float)

        # Nos quedamos solo con valores válidos.
        mask = np.isfinite(tau) & (tau > 0)

        # Si hay pocos puntos válidos, devolvemos nan.
        if mask.sum() < 5:
            return np.nan

        # Ajustamos recta en escala log-log.
        slope = np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]

        # Aproximamos exponente de Hurst.
        return safe_float(slope * 2.0)

    except Exception:
        return np.nan


# =========================================================
# FUNCIÓN: acf_decay
# =========================================================
def acf_decay(x, max_lag=12):
    try:
        # Convertimos a array float.
        x = np.asarray(x, dtype=float)

        # Filtramos finitos.
        x = x[np.isfinite(x)]

        # Si la serie es muy corta, devolvemos nan.
        if len(x) < max_lag + 2:
            return np.nan

        # Calculamos ACF hasta max_lag.
        a = acf(x, nlags=max_lag, fft=False)

        # Tomamos valor absoluto de los rezagos 1...max_lag.
        vals = np.abs(a[1:])

        # Buscamos el primer lag donde la ACF cae por debajo de 0.2.
        below = np.where(vals < 0.2)[0]

        # Si nunca cae, devolvemos max_lag.
        if len(below) == 0:
            return float(max_lag)

        # Si sí cae, devolvemos ese primer lag (+1 porque empezamos en lag 1).
        return safe_float(below[0] + 1)

    except Exception:
        return np.nan


# =========================================================
# FUNCIÓN: trend_linearity_features
# =========================================================
def trend_linearity_features(x):
    try:
        # Convertimos a array float.
        x = np.asarray(x, dtype=float)

        # Filtramos finitos.
        x = x[np.isfinite(x)]

        # Longitud de la serie.
        n = len(x)

        # Si hay muy pocos puntos, no calculamos nada.
        if n < 5:
            return np.nan, np.nan

        # Construimos variable tiempo para regresión lineal.
        t = np.arange(n).reshape(-1, 1)

        # Convertimos y a columna.
        y = x.reshape(-1, 1)

        # Ajuste lineal.
        lin = LinearRegression().fit(t, y)

        # R^2 del modelo lineal.
        r2_lin = lin.score(t, y)

        # Construimos regresores lineal + cuadrático.
        t2 = np.column_stack([np.arange(n), np.arange(n) ** 2])

        # Ajuste cuadrático.
        quad = LinearRegression().fit(t2, y)

        # R^2 del modelo cuadrático.
        r2_quad = quad.score(t2, y)

        # Devolvemos:
        # 1) R^2 lineal
        # 2) ganancia por curvatura = R^2 cuadrático - R^2 lineal
        return safe_float(r2_lin), safe_float(r2_quad - r2_lin)

    except Exception:
        return np.nan, np.nan


# =========================================================
# FUNCIÓN: strength_trend_stl
# =========================================================
def strength_trend_stl(s, period):
    try:
        # Ajustamos STL sobre la serie s con período dado.
        stl = STL(s, period=period, robust=True).fit()

        # Varianza del residual.
        vr = np.var(stl.resid)

        # Varianza de tendencia + residual.
        vt = np.var(stl.trend + stl.resid)

        # Si vt = 0, no podemos dividir.
        if vt == 0:
            return np.nan

        # Fuerza de tendencia según fórmula clásica.
        return safe_float(max(0, 1 - vr / vt))

    except Exception:
        return np.nan


# =========================================================
# FUNCIÓN: strength_seasonal_stl
# =========================================================
def strength_seasonal_stl(s, period):
    try:
        # Ajustamos STL.
        stl = STL(s, period=period, robust=True).fit()

        # Varianza residual.
        vr = np.var(stl.resid)

        # Varianza estacional + residual.
        vs = np.var(stl.seasonal + stl.resid)

        # Si vs = 0, no podemos dividir.
        if vs == 0:
            return np.nan

        # Fuerza de estacionalidad.
        return safe_float(max(0, 1 - vr / vs))

    except Exception:
        return np.nan


# =========================================================
# IMPORTS PARA PRUEBAS DE ESTACIONARIEDAD
# =========================================================
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


# =========================================================
# FUNCIÓN: stationarity_features
# =========================================================
def stationarity_features(x):
    # Diccionario de salida con valores por defecto.
    out = {
        "adf_pvalue": np.nan,              # p-value de ADF
        "kpss_pvalue": np.nan,             # p-value de KPSS
        "adf_stationary": np.nan,          # 1 si ADF sugiere estacionariedad
        "kpss_stationary": np.nan,         # 1 si KPSS sugiere estacionariedad
        "stationarity_conflict": np.nan,   # 1 si ADF y KPSS se contradicen
    }

    # Convertimos a array float.
    x = np.asarray(x, dtype=float)

    # Filtramos finitos.
    x = x[np.isfinite(x)]

    # Si la serie es muy corta, no calculamos pruebas.
    if len(x) < 10:
        return out

    # =========================
    # ADF
    # =========================
    try:
        # adfuller devuelve varios valores; [1] es el p-value.
        adf_p = adfuller(x, autolag="AIC")[1]

        # Guardamos p-value.
        out["adf_pvalue"] = safe_float(adf_p)

        # Si p < 0.05, rechazamos raíz unitaria => estacionaria.
        out["adf_stationary"] = float(adf_p < 0.05)

    except Exception:
        # Si falla, dejamos nan.
        pass

    # =========================
    # KPSS (con control de warnings)
    # =========================
    try:
        # A veces KPSS lanza warnings por interpolación.
        # Los silenciamos aquí para que no ensucien la salida.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)

            # kpss(...)[1] = p-value
            kpss_p = kpss(x, regression="c", nlags="auto")[1]

        # Cortamos p-value en un rango estable.
        kpss_p = np.clip(kpss_p, 1e-6, 1.0)

        # Guardamos p-value.
        out["kpss_pvalue"] = safe_float(kpss_p)

        # En KPSS, si p > 0.05, no rechazamos estacionariedad.
        out["kpss_stationary"] = float(kpss_p > 0.05)

    except Exception:
        pass

    # =========================
    # Conflicto
    # =========================
    if (
        np.isfinite(out["adf_stationary"]) and
        np.isfinite(out["kpss_stationary"])
    ):
        # Si una prueba dice estacionaria y la otra no, hay conflicto.
        out["stationarity_conflict"] = float(
            out["adf_stationary"] != out["kpss_stationary"]
        )

    return out


# =========================================================
# FUNCIÓN: change_points_per_length
# =========================================================
def change_points_per_length(x):
    try:
        # Si no existe ruptures, no podemos calcular esto.
        if not RUPTURES_OK:
            return np.nan

        # Convertimos a array float.
        x = np.asarray(x, dtype=float)

        # Filtramos finitos.
        x = x[np.isfinite(x)]

        # Si la serie es muy corta, devolvemos 0.
        if len(x) < 24:
            return 0.0

        # ruptures espera señal 2D: (n_muestras, n_variables)
        signal = x.reshape(-1, 1)

        # Ajustamos algoritmo PELT con modelo l2.
        algo = rpt.Pelt(model="l2").fit(signal)

        # Detectamos puntos de cambio usando penalización pen=3.
        bkps = algo.predict(pen=3)

        # ruptures incluye el final de la serie como breakpoint,
        # por eso restamos 1.
        n_cp = max(0, len(bkps) - 1)

        # Devolvemos densidad de puntos de cambio.
        return safe_float(n_cp / len(x))

    except Exception:
        return np.nan


# ===== celda 4 del notebook =====
def extract_features_v2(serie, nombre, freq, category=None, horizon=None):
    # Convertimos la serie de entrada a una Serie de pandas,
    # eliminamos valores faltantes y forzamos tipo float.
    s = pd.Series(serie).dropna().astype(float)

    # Extraemos los valores como array de numpy para trabajar más rápido.
    x = s.values

    # Definimos una longitud mínima necesaria para poder calcular features.
    # Si freq existe, exigimos al menos 2 ciclos completos.
    # Si no existe, exigimos al menos 8 observaciones.
    min_len = max(8, 2 * freq if freq is not None else 8)

    # Si la serie es demasiado corta, devolvemos None.
    if len(x) < min_len:
        return None

    # Generamos la versión normalizada por z-score.
    x_z = zscore_series(x)

    # Generamos la versión robustamente escalada.
    x_r = robust_scale_series(x)

    # Calculamos primeras diferencias de la serie z-score.
    # Si la serie tiene longitud 1 o menos, devolvemos array vacío.
    x_d = np.diff(x_z) if len(x_z) > 1 else np.array([])

    # Creamos el diccionario base de features.
    feat = {
        "serie": nombre,                              # nombre o ID de la serie
        "category": category,                        # categoría (Yearly, Monthly, etc.)
        "freq": freq,                                # frecuencia de la serie
        "log_length": safe_float(np.log1p(len(x))),  # log(1 + longitud) para estabilizar escala
        "horizon": horizon if horizon is not None else np.nan,  # horizonte de forecast
    }

    # =========================
    # FEATURES SOBRE SERIE Z-SCORE
    # =========================

    # Asimetría de la serie normalizada.
    feat["z_skew"] = safe_float(skew(x_z, bias=False)) if len(x_z) > 2 else np.nan

    # Curtosis de la serie normalizada.
    feat["z_kurtosis"] = safe_float(kurtosis(x_z, bias=False)) if len(x_z) > 3 else np.nan

    # Entropía de Shannon sobre la serie z-score.
    feat["z_entropy"] = shannon_entropy(x_z)

    # Entropía espectral sobre la serie z-score.
    feat["z_spectral_entropy"] = spectral_entropy(x_z)

    # Proporción robusta de outliers sobre la serie z-score.
    feat["z_outlier_ratio"] = outlier_ratio_robust(x_z)

    # Proporción de puntos de giro sobre la serie z-score.
    feat["z_turning_points_ratio"] = turning_points_ratio(x_z)

    # Exponente de Hurst sobre la serie z-score.
    feat["z_hurst"] = hurst_exponent(x_z)

    # =========================
    # AUTOCORRELACIONES
    # =========================
    try:
        # Definimos cuántos rezagos vamos a calcular.
        # Queremos al menos 12, o freq si freq es mayor,
        # pero sin exceder la mitad de la serie.
        nlags = min(max(12, freq if freq else 12), len(x_z) // 2 - 1)

        # Calculamos la ACF.
        a = acf(x_z, nlags=max(1, nlags), fft=False)

        # Guardamos la autocorrelación en lag 1.
        feat["z_acf1"] = safe_float(a[1]) if len(a) > 1 else np.nan

        # Guardamos la autocorrelación en lag 6.
        feat["z_acf6"] = safe_float(a[6]) if len(a) > 6 else np.nan

        # Guardamos la autocorrelación en el lag igual a la frecuencia.
        feat["z_acf_freq"] = safe_float(a[freq]) if freq is not None and len(a) > freq else np.nan

    except Exception:
        # Si falla el cálculo, ponemos nan.
        feat["z_acf1"] = np.nan
        feat["z_acf6"] = np.nan
        feat["z_acf_freq"] = np.nan

    # Calculamos una medida de decaimiento de autocorrelación.
    # Busca el primer lag donde |ACF| cae por debajo de 0.2.
    feat["z_acf_decay"] = acf_decay(x_z, max_lag=min(12, max(2, len(x_z) // 3)))

    # =========================
    # TENDENCIA Y CURVATURA
    # =========================

    # Calcula:
    # - R^2 del ajuste lineal
    # - ganancia al pasar de lineal a cuadrático
    lin_r2, curvature_gain = trend_linearity_features(x_z)

    feat["z_trend_linearity_r2"] = lin_r2
    feat["z_curvature_gain"] = curvature_gain

    # Además calculamos explícitamente la pendiente lineal.
    try:
        # Tiempo como variable explicativa.
        t = np.arange(len(x_z)).reshape(-1, 1)

        # Serie como variable respuesta.
        y = x_z.reshape(-1, 1)

        # Ajustamos regresión lineal.
        model = LinearRegression().fit(t, y)

        # Guardamos la pendiente.
        feat["z_trend_slope"] = safe_float(model.coef_[0][0])

    except Exception:
        feat["z_trend_slope"] = np.nan

    # =========================
    # TENDENCIA Y ESTACIONALIDAD POR STL
    # =========================

    # Solo tiene sentido calcular STL si:
    # - freq existe
    # - freq > 1
    # - hay al menos 2 ciclos completos
    if freq is not None and freq > 1 and len(x) >= 2 * freq:
        feat["trend_strength"] = strength_trend_stl(s, period=freq)
        feat["seasonal_strength"] = strength_seasonal_stl(s, period=freq)
    else:
        feat["trend_strength"] = np.nan
        feat["seasonal_strength"] = np.nan

    # =========================
    # FRECUENCIA DOMINANTE
    # =========================

    # Calculamos:
    # - frecuencia dominante
    # - proporción de energía de esa frecuencia
    dom_freq, dom_energy = dominant_frequency_features(x_z)

    feat["dominant_frequency"] = dom_freq
    feat["dominant_energy_ratio"] = dom_energy

    # =========================
    # ESTACIONARIEDAD
    # =========================

    # Agregamos al diccionario:
    # - adf_pvalue
    # - kpss_pvalue
    # - adf_stationary
    # - kpss_stationary
    # - stationarity_conflict
    feat.update(stationarity_features(x_z))

    # =========================
    # RAZÓN DE VARIANZAS EN DIFERENCIAS
    # =========================
    try:
        # Compara la varianza de la serie diferenciada con la varianza original.
        # Si la serie original tiene varianza cero, devolvemos nan.
        feat["diff_var_ratio"] = safe_float(np.var(np.diff(x_z)) / np.var(x_z)) if np.var(x_z) > 0 else np.nan
    except Exception:
        feat["diff_var_ratio"] = np.nan

    # =========================
    # PUNTOS DE CAMBIO
    # =========================

    # Densidad de puntos de cambio estructural.
    feat["change_points_per_length"] = change_points_per_length(x_z)

    # =========================
    # FEATURES SOBRE LA SERIE DIFERENCIADA
    # =========================
    if len(x_d) > 1:
        # Asimetría de la primera diferencia.
        feat["diff_skew"] = safe_float(skew(x_d, bias=False)) if len(x_d) > 2 else np.nan

        # Curtosis de la primera diferencia.
        feat["diff_kurtosis"] = safe_float(kurtosis(x_d, bias=False)) if len(x_d) > 3 else np.nan

        # Entropía de la primera diferencia.
        feat["diff_entropy"] = shannon_entropy(x_d)

        # Proporción de puntos de giro en la primera diferencia.
        feat["diff_turning_points_ratio"] = turning_points_ratio(x_d)

    else:
        # Si la serie diferenciada es demasiado corta, todo va en nan.
        feat["diff_skew"] = np.nan
        feat["diff_kurtosis"] = np.nan
        feat["diff_entropy"] = np.nan
        feat["diff_turning_points_ratio"] = np.nan

    # =========================
    # FEATURES ROBUSTOS
    # =========================

    # Entropía sobre la serie robustamente escalada.
    feat["robust_entropy"] = shannon_entropy(x_r)

    # Proporción robusta de outliers sobre la serie robustamente escalada.
    feat["robust_outlier_ratio"] = outlier_ratio_robust(x_r)

    # Finalmente devolvemos el diccionario completo de features.
    return feat
