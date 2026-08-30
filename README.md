# Caracterizar para ponderar, no para elegir

Código de *«Caracterizar para ponderar, no para elegir: estructura y combinación de
modelos de pronóstico en las 100.000 series de la competencia M4»*
(Peña Araya y Segura Pérez, UNAM).

## Qué contiene

```
src/        el pipeline: características, PCA, índice, clustering, entropías, métricas
analisis/   los experimentos del artículo: enrutamiento, meta-aprendizaje, combinación
figuras/    las figuras del artículo, en PDF vectorial
```

## Datos

No se incluyen: son públicos y pesan varios cientos de MB. Descargar los CSV
originales de la competencia desde el repositorio de los organizadores
(<https://github.com/Mcompetitions/M4-methods>, carpeta `Dataset`) y colocarlos en
`data/` con los nombres `Yearly-train.csv`, `Yearly-test.csv`, y así para las seis
frecuencias.

## Orden de ejecución

| | Guion | Qué produce |
|---|---|---|
| 1 | `src/01_pca_complexity.py` | los 29 descriptores, PCA e índice de complejidad |
| 2 | `src/02_clustering_gmm.py` | la partición binaria y sus métricas internas |
| 3 | `src/03_compute_entropy.py` | entropía de permutación, sample entropy y Lempel-Ziv |
| 4 | `src/04_forecasting_errors.py` | errores del pronóstico ingenuo por serie |
| 5 | `src/05_entropy_validation.py` | regresiones anidadas: índice contra entropías |
| 6 | `src/06_m4_metrics.py` | sMAPE, MASE y OWA oficiales; benchmark Naive2 |
| 7 | `src/owa_modelos.py` | OWA de los 26 modelos, por serie |
| 8 | `analisis/extraer_pronosticos.py` | cachea los pronósticos (hace falta para combinar) |
| 9 | `analisis/experimento_enrutamiento.py` | políticas A–D y barrido de estratos |
| 10 | `analisis/comparacion_metaaprendizaje.py` | políticas E y F, estilo FFORMS y FFORMPP |
| 11 | `analisis/politica_combinar.py` | elegir contra combinar, y los controles |
| 12 | `analisis/test_friedman.py` | Friedman y post-hoc de Nemenyi |
| 13 | `analisis/generar_figuras.py` | las figuras |

Los pasos 1 a 7 escriben en `results/`; los de `analisis/` leen de ahí.

## Reproducibilidad

Las etapas deterministas reproducen con **diferencia 0,00** sobre las 99.935 series al
reejecutarse: descriptores, PCA, índice, conglomerados, entropías y errores de los
modelos estadísticos y de aprendizaje automático.

Las que involucran redes neuronales **no reproducen**, y eso es un resultado del
artículo, no un descuido. Fijar la semilla resultó insuficiente: al cambiar únicamente
el número de hilos (`OMP_NUM_THREADS` de 4 a 2), ninguna de 2.250 mediciones coincidió.
Ver el apéndice de reproducibilidad del artículo para el entorno exacto.

## Entorno

Python 3.10.20, TensorFlow 2.21.0, NumPy 2.2.5, scikit-learn 1.7.2, statsmodels 0.14.6.
Todo en CPU. `pip install -r requirements.txt`.

## Cita

Pendiente de publicación. Mientras tanto, citar el repositorio.

## Licencia

MIT. Ver `LICENSE`.
