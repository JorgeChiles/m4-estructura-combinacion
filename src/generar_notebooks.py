#!/usr/bin/env python3
"""
Genera un notebook por cada script del pipeline, con celdas markdown que
explican el procedimiento.

Como funciona
-------------
El codigo de las celdas NO se escribe a mano: se extrae de los .py cortando
por los encabezados de seccion

    # =========================================================
    # TITULO DE LA SECCION
    # =========================================================

y se intercala con la explicacion correspondiente. Asi el notebook no puede
divergir del script: si cambia el .py, se regenera y listo.

El .py sigue siendo la version que se corre en serio. El notebook es para
leer, explicar y para la defensa de la tesis.

Uso:  python3 src/generar_notebooks.py
"""
import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
NB_DIR = SRC.parent / "notebooks"

BANNER = re.compile(r"^# ={10,}\s*$")


# =========================================================
# EXPLICACIONES
# =========================================================
# Por cada script: titulo, introduccion y un texto por seccion.
# Las secciones sin texto quedan como celda de codigo sola.

E = {}

E["01_pca_complexity.py"] = {
    "titulo": "01 · PCA e índice de complejidad estructural",
    "intro": """
Este es el primer paso del pipeline y el que define **el objeto central de la tesis**:
el *índice de complejidad estructural*.

**Entrada.** `features_m4_all_v2.xlsx`, que trae ~42 descriptores por serie calculados
en `CLASIFICACION_V3_All.ipynb`: fuerza de tendencia, fuerza estacional, autocorrelaciones,
no linealidad, curtosis, entropía, coeficientes de variación, etc., para las 100.000 series
de M4.

**Problema.** Esos descriptores están fuertemente correlacionados entre sí. Una serie con
tendencia marcada casi siempre tiene autocorrelación alta en el primer rezago. Usar los 42
por separado es redundante, y no da un número único con el que ordenar las series.

**Idea.** Aplicar PCA y quedarse con el primer componente. PC1 es la dirección de máxima
varianza conjunta: el eje que resume "cuánta estructura temporal explotable tiene la serie".

**El signo.** PCA no fija el signo de un componente. Acá PC1 sale con signo positivo para
series *estructuradas* (mucha tendencia y estacionalidad), así que el índice se define como

$$\\text{complexity\\_index} = -\\,\\text{PC1}$$

para que **más alto signifique más complejo**, es decir, menos estructura aprovechable.
Esa inversión es la que hace que el índice deba correlacionar **positivamente** con el error
de pronóstico, que es exactamente lo que valida el script 05.

**Salidas.** `df_features_complexity.xlsx` (todas las series con su índice y su cuartil),
`pc1_loadings.xlsx` (qué features pesan en PC1), `X_pca.npy` y `X_scaled.npy` (que reusa el 02).
""",
    "secciones": {
        "CONTROL DE HILOS": """
BLAS multihilo sobre matrices de 100.000 × 42 pelea consigo mismo: los hilos se sincronizan
más de lo que trabajan. Fijando todo en 1 el script corre más rápido y con memoria estable.
Va **antes** de importar numpy, porque las variables se leen al cargar la librería.
""",
        "SELECCIÓN Y LIMPIEZA DE FEATURES": """
Tres filtros en orden:

1. Se queda solo con las features de la lista `FEATURES_CLUSTER_V2` que existan en el archivo.
2. Convierte infinitos en `NaN` y descarta columnas enteramente vacías.
3. `VarianceThreshold(0.0)` elimina las columnas **constantes**: una feature que vale lo mismo
   en las 100.000 series no aporta varianza y ensuciaría el PCA.
""",
        "IMPUTACIÓN": """
Imputación por **mediana**, no por media: las features de M4 tienen colas largas
(series con saltos, con outliers) y la media se corre hacia los extremos.
""",
        "ESTANDARIZACIÓN": """
Paso obligatorio antes de PCA. PCA maximiza varianza, y sin estandarizar la feature con
la escala más grande domina el primer componente por una razón puramente de unidades.
Con `StandardScaler` todas entran con media 0 y desvío 1.
""",
        "PCA": """
`n_components=0.90` no fija un número de componentes: le pide a PCA **quedarse con los que
hagan falta para explicar el 90% de la varianza**. El índice usa solo PC1, pero el espacio
completo se guarda porque el script 02 hace clustering sobre él.
""",
        "ÍNDICE DE COMPLEJIDAD": """
Acá se define el índice y se invierte el signo (ver la introducción).

Además se corta en cuartiles con `qcut` → `low`, `mid_low`, `mid_high`, `high`.
Los cuartiles son **relativos a la muestra**: por construcción hay 25% en cada nivel.
Sirven para comparar grupos (el cociente de error alto/bajo que reporta el 05), no como
una escala absoluta de dificultad.
""",
        "LOADINGS PC1": """
Los *loadings* son la interpretación del índice: cuánto pesa cada feature original en PC1.
Sin esta tabla el índice sería una caja negra. Es la evidencia de que PC1 mide lo que
decimos que mide, y el material para la sección de resultados de la tesis.
""",
    },
}

E["02_clustering_gmm.py"] = {
    "titulo": "02 · Clustering y selección de número de grupos",
    "intro": """
Complementa al 01. Si el índice de complejidad es una escala **continua**, este paso
pregunta si además existen **grupos discretos** de series en el espacio PCA.

**Entrada.** `X_pca.npy` y `df_features_complexity.xlsx` que dejó el 01.

**Dos familias de método:**

- **MiniBatchKMeans** con K de 2 a 9, evaluado con *silhouette* (más alto mejor) y
  *Davies-Bouldin* (más bajo mejor). MiniBatch y no KMeans común porque son 100.000 puntos.
- **Gaussian Mixture** con 2 a 10 componentes, evaluado con **AIC y BIC**. GMM es un modelo
  probabilístico, así que admite criterios de información: penalizan agregar componentes.

**Por qué las dos.** Silhouette y Davies-Bouldin son métricas *geométricas* que suelen
premiar K chicos; AIC/BIC son *estadísticos*. Si ambas familias coinciden, la estructura de
grupos es real; si no coinciden, es señal de que el espacio es más bien continuo — que es
justamente lo que el índice del 01 modela mejor.

**Salidas.** `clustering_gmm_results.xlsx` (todo junto, en hojas separadas) más los archivos
sueltos y las tres figuras.
""",
    "secciones": {
        "MUESTRA PARA EVALUACIÓN": """
La selección de K se hace sobre una muestra de 30.000 series, no sobre las 100.000.
El motivo es `silhouette_score`: es **O(n²)** en memoria y tiempo, porque compara cada punto
con todos los demás. Con 100.000 puntos no termina.

El entrenamiento final sí usa las 100.000. La muestra es solo para *elegir* K.
`random_state=42` fija el sorteo.
""",
        "EVALUACIÓN DE KMEANS": """
Para cada K se ajusta el modelo y se calculan las dos métricas internas:

- **Silhouette** ∈ [-1, 1]: qué tan cerca está cada punto de su grupo comparado con el grupo
  vecino más cercano. Más alto es mejor.
- **Davies-Bouldin** ≥ 0: cociente entre dispersión interna y separación entre grupos.
  Más bajo es mejor.

El K elegido es el mejor por silhouette, desempatando por Davies-Bouldin.
""",
        "ENTRENAMIENTO FINAL KMEANS": """
Recién acá se ajusta sobre las 100.000 series, con el K elegido y `n_init=20`
(veinte inicializaciones distintas, se queda con la mejor) porque esta asignación
es la que queda guardada.
""",
        "GMM: AIC Y BIC": """
La lectura correcta de AIC/BIC no es el valor absoluto sino **dónde deja de bajar la curva**.
Si BIC sigue cayendo hasta el máximo K probado, el criterio no encontró un óptimo: es evidencia
en contra de que existan grupos bien definidos, y a favor de tratar la complejidad como
un continuo.
""",
    },
}

E["03_compute_entropy.py"] = {
    "titulo": "03 · Entropías clásicas (validación externa)",
    "intro": """
Este script calcula **tres medidas de complejidad ya establecidas en la literatura**.
No forman parte del índice: son el control externo con el que se lo valida en el 05.

El argumento de la tesis necesita esto. Un índice construido con PCA sobre features propias
podría estar midiendo cualquier cosa. Si correlaciona con medidas independientes y aceptadas,
deja de ser una construcción arbitraria.

Las tres miden complejidad desde ángulos distintos:

| Medida | Qué captura | Rango |
|---|---|---|
| **Entropía de permutación** | desorden en el *orden relativo* de valores vecinos | 0 a 1 (normalizada) |
| **Sample entropy** | qué tan impredecible es la serie: probabilidad de que patrones parecidos sigan pareciéndose | 0 a ∞ |
| **Lempel-Ziv** | cuánto se puede *comprimir* la serie binarizada | normalizada por `n/log₂ n` |

Ninguna usa las features del 01: se calculan directo sobre los valores crudos de la serie
de entrenamiento. Esa independencia es la que hace válida la validación.

**Entrada.** `df_features_complexity.xlsx` (para saber qué series recorrer) y los CSV de `data/`.
**Salida.** `df_entropy.xlsx`, una fila por serie con las tres medidas.
""",
    "secciones": {
        "ENTROPIA DE PERMUTACION": """
Bandt-Pompe (2002). Recorre la serie con una ventana de `m=3` y, en cada posición, se queda
**solo con el orden** de los valores (cuál es el menor, el del medio, el mayor). Con m=3 hay
3! = 6 patrones posibles. Después mide la entropía de Shannon de la distribución de patrones
y la divide por log(3!) para dejarla entre 0 y 1.

- Serie perfectamente monótona → siempre el mismo patrón → entropía **0**.
- Ruido blanco → los 6 patrones equiprobables → entropía **1**.

La gracia es que solo usa el orden: es inmune a cambios de escala y muy robusta a outliers.
""",
        "SAMPLE ENTROPY": """
Richman-Moorman (2000). Cuenta cuántos pares de ventanas de largo `m` se parecen dentro de una
tolerancia `r = 0.2·σ`, y cuántos **siguen pareciéndose** al extender a `m+1`. El resultado es
`-log(φ_{m+1} / φ_m)`: si extender la ventana rompe la mayoría de las coincidencias, la serie
es impredecible y la entropía es alta.

Es la más cara de las tres: el conteo de pares es **O(n²)** y no está vectorizado. Sobre las
series mensuales largas es lo que domina el tiempo del script.
""",
        "COMPLEJIDAD LEMPEL-ZIV": """
Primero **binariza** la serie contra su propia mediana (1 si está por encima, 0 si no) y después
cuenta cuántos patrones nuevos hacen falta para reconstruir esa cadena — el mismo principio que
la compresión LZ77.

Se normaliza por `n / log₂(n)`, que es el valor esperado para una secuencia aleatoria: así el
resultado no depende del largo de la serie y ≈1 significa "incompresible como el azar".

Vale la pena anticipar el resultado del 05: **de las tres, LZ es la que mejor correlaciona con
el error de pronóstico** (+0,429 global, por encima del propio índice de complejidad, +0,399).
""",
        "LOOP PRINCIPAL": """
Recorre las seis frecuencias y, dentro de cada una, las series listadas en el archivo de features.
Las series con menos de 10 observaciones se registran con `NaN` en las tres columnas en lugar de
descartarse, para que el archivo mantenga una fila por serie y el merge del 05 no pierda registros.

⚠️ **`MAX_SERIES_PER_CATEGORY_ENT`**: si vale 1000, el script procesa 1000 series por frecuencia
(6.000 en total). El `df_entropy.xlsx` que hay guardado tiene **99.935 filas**, así que se generó
con este parámetro en `None`. Está en `None` para que el script reproduzca el resultado real.
""",
    },
}

E["04_forecasting_errors.py"] = {
    "titulo": "04 · Errores de pronóstico de referencia",
    "preparacion": (
        "## Preparación\n\n"
        "Además de agregar `src/` al path, este notebook corre sobre una **muestra de 500 "
        "series por frecuencia**.\n\n"
        "La corrida completa son ~100.000 series con SARIMA y toma cerca de un día de máquina, "
        "sin producir ningún dato nuevo: los resultados completos ya están calculados en "
        "`forecasting_errors_partial.xlsx` y extendidos a las seis frecuencias por el 04b. "
        "La muestra alcanza para documentar el procedimiento con salidas reales.\n\n"
        "> ⚠️ Con `MAX_SERIES` puesta, **todas las salidas llevan el sufijo `_MUESTRA500`**. "
        "Es a propósito: una corrida de muestra no puede reemplazar a la completa, porque "
        "`forecasting_errors_partial.xlsx` es de donde salen el 04b y el 05.\n\n"
        "Para la corrida completa desde la terminal, sin la variable:\n\n"
        "```\npython3 src/04_forecasting_errors.py\n```",
        'import os\nimport sys\nfrom pathlib import Path\n\n'
        '_raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
        'sys.path.insert(0, str(_raiz / "src"))\n\n'
        '# 500 series por frecuencia. Sin esta linea corre las 100.000 (~1 dia).\n'
        'os.environ["MAX_SERIES"] = "500"'),
    "intro": """
Para validar que el índice de complejidad **sirve para algo**, hace falta la variable contra
la que compararlo: el error que efectivamente comete un modelo al pronosticar cada serie.

Este script ajusta cuatro métodos de referencia sobre las 100.000 series y guarda su sMAPE:

| Método | Qué hace |
|---|---|
| **Naive** | repite el último valor observado |
| **Seasonal naive** | repite el último ciclo estacional completo |
| **ARIMA(1,1,1)** | un ARIMA fijo, sin búsqueda de orden |
| **SARIMA(1,1,1)(1,1,1,s)** | idem con componente estacional |

Órdenes fijos a propósito: no interesa el mejor modelo posible por serie, sino una **medida
homogénea de dificultad**. Si una serie es difícil para estos cuatro, es difícil.

⚠️ **Dos advertencias importantes** para leer los resultados:

1. El sMAPE de este script queda en **fracción** (0,0398), no en porcentaje, y **no** es el
   sMAPE oficial de M4.
2. Las estacionalidades usadas acá (Weekly 52, Daily 7) **no son las de M4**, que define
   Weekly = 1 y Daily = 1. El `snaive` de este script, entonces, no es comparable con la
   competencia.

Las dos cosas se corrigen en el script **06**, que implementa las métricas oficiales.
Este script sigue siendo válido para lo suyo — ordenar series por dificultad — pero los
números publicables son los del 06.
""",
    "secciones": {
        "RUTAS": """
El bloque de arriba fija los hilos de BLAS en 1 **antes de importar numpy**, y no es un detalle
menor. `SARIMAX` con m=52 (las series semanales) abre un hilo por núcleo, y sobre matrices
chicas los hilos pasan más tiempo sincronizándose que calculando.

Está **medido**: una serie semanal tarda ~37 s con un hilo y **más de 39 minutos** con todos.
Sobre las 294 series semanales que admiten SARIMA, es la diferencia entre 3 horas y ocho días.
Puede ser la razón por la que la corrida original de mayo quedó trunca después de Monthly.
""",
        "FUNCIONES": """
Todas las funciones de pronóstico devuelven `NaN` en vez de lanzar excepción cuando no pueden
ajustar. Sobre 100.000 series hay de todo: series demasiado cortas para el orden estacional,
optimizaciones que no convergen, constantes. Con `raise` el script moriría en la serie 3.000;
con `NaN` se registra el fallo y se sigue, y después se puede contar cuántos hubo por método.

`safe_smape` protege la división cuando real y pronóstico son ambos cero: enmascara esos
puntos en lugar de devolver infinito.
""",
        "LOOP PRINCIPAL": """
Doble recorrido: frecuencia → serie. Cada serie produce una fila con cuatro errores y cuatro
columnas `status_*` (`ok`, `error`, `no_aplica`, `serie_corta`, `serie_no_encontrada`).

Guardar el *status* junto al error es lo que después permite distinguir un `NaN` porque el
modelo no aplicaba de un `NaN` porque falló el ajuste — que son cosas muy distintas al reportar.

Es la etapa más lenta del pipeline: SARIMA sobre 48.000 series mensuales son horas.
""",
    },
}

E["04b_completar_frecuencias.py"] = {
    "ejecutar_main": True,
    "titulo": "04b · Completar Weekly, Daily y Hourly",
    "intro": """
El archivo que dejó el 04 se llama `forecasting_errors_partial` por una razón: tiene
**95.000 series**, no 100.000. Cubre Yearly, Quarterly y Monthly — la población completa de
esas tres frecuencias — y le faltan enteras Weekly (359), Daily (4.227) y Hourly (414).

Sin esas 5.000 no se puede reportar la correlación por frecuencia, que resultó ser el
hallazgo más interesante de la validación: el índice funciona muy bien en Hourly (+0,661)
y prácticamente no funciona en Daily (+0,019).

Este script calcula solo las que faltan, con **las mismas funciones** que el 04 para que los
errores sean comparables, y concatena todo en `forecasting_errors_completo.xlsx`. No pisa el
parcial.

**Por qué está paralelizado.** SARIMA con m=52 sobre series semanales de 2.000+ puntos tarda
~150 s por serie: en secuencial serían ~15 horas. Con un `Pool` baja a ~2.
""",
    "secciones": {
        "CONFIGURACION": """
Ojo con estas estacionalidades: son las mismas que usa el 04 (Weekly 52, Daily 7, Hourly 24),
elegidas por criterio "natural", **no** las que define M4 (1, 1, 24). Se mantienen acá a
propósito, para que las 5.000 series nuevas sean consistentes con las 95.000 que ya estaban
calculadas. Mezclar convenciones en un mismo archivo sería peor.

Las métricas oficiales de M4 se calculan aparte, en el script 06.
""",
        "EVALUACION DE UNA SERIE": """
`evaluar()` está definida **a nivel de módulo, no adentro de otra función**. No es un detalle
de estilo: `multiprocessing` serializa la función con `pickle` para mandarla a cada worker, y
las funciones anidadas o las lambdas no son picklables. Definida adentro, el `Pool` falla.
""",
        "PROCESAMIENTO POR FRECUENCIA": """
`imap_unordered` en lugar de `map`: devuelve cada resultado apenas está listo, sin esperar a
que terminen todos ni respetar el orden de entrada. Como después se arma un DataFrame y se
ordena, el orden no importa, y así se puede ir imprimiendo progreso real con ETA.

El orden de procesamiento es Daily → Hourly → Weekly, de la más rápida a la más lenta, para
tener resultados parciales útiles temprano.
""",
        "PRINCIPAL": """
Nota sobre los hilos: al principio del archivo se fija `OMP_NUM_THREADS=1` **antes de importar
numpy**. Con un `Pool`, cada worker es un proceso independiente; si además cada uno abre 18
hilos de BLAS, se pelean por los mismos cores y el resultado es más lento que el secuencial.
Un core por worker, y el paralelismo lo maneja el `Pool`.
""",
    },
}

E["05_entropy_validation.py"] = {
    "titulo": "05 · Validación del índice de complejidad",
    "intro": """
El script que cierra el argumento. Responde dos preguntas:

1. **¿El índice mide complejidad?** → correlacionarlo con las tres entropías del 03.
2. **¿Sirve para predecir dificultad?** → correlacionarlo con el error de pronóstico del 04b,
   globalmente y por frecuencia.

**Resultados obtenidos** (99.935 series):

| Medida | Correlación con el error naive |
|---|---|
| `lz_complexity` | **+0,429** |
| `complexity_index` | **+0,399** |
| `sample_entropy` | +0,240 |
| `perm_entropy` | +0,057 |

Por frecuencia el índice se comporta muy distinto: Hourly +0,661, Weekly +0,634, Monthly +0,439,
Quarterly +0,395, Yearly +0,203 y **Daily +0,019**.

Ese Daily ≈ 0 no es un error del pipeline: las series diarias de M4 son prácticamente caminatas
aleatorias, sin estructura explotable que el índice pueda medir. Es un resultado, no una falla.

⚠️ El signo positivo es la predicción teórica: más complejidad → más error. Si hubiera dado
negativo, la inversión de signo del script 01 estaría al revés.
""",
    "secciones": {
        "CARGA Y UNION DE DATOS": """
Acá estaba el bug que impedía correr el script. Los errores de pronóstico **no** están en
`df_features_complexity.xlsx`: los produce el 04 en un archivo aparte. Sin este segundo merge,
la columna `error_naive_smape` no existe y el script muere con `KeyError`.

El bloque prefiere `forecasting_errors_completo.xlsx` (100.000 series, seis frecuencias) y solo
cae al parcial (95.000, tres frecuencias) si el completo todavía no se generó. Correrlo con el
parcial da resultados **sin Weekly, Daily ni Hourly**, que es justo la parte más interesante.
""",
        "CORRELACION GLOBAL": """
Matriz de correlaciones entre el índice, las tres entropías y el error naive.

Dos lecturas distintas:
- **índice vs entropías** → validación de constructo: ¿mide lo mismo que las medidas aceptadas?
- **índice vs error** → validez predictiva: ¿sirve para anticipar dificultad?

La segunda es la que importa para la tesis. Un índice que correlaciona perfecto con las
entropías pero nada con el error sería inútil.
""",
        "CORRELACION POR FRECUENCIA": """
Desagregar es imprescindible. La correlación global mezcla seis poblaciones con dinámicas y
horizontes distintos, y puede estar inflada simplemente porque las frecuencias difieren entre sí
en promedio (una forma del efecto de agregación / paradoja de Simpson).

Al abrir por frecuencia aparece el rango real: de +0,661 en Hourly a +0,019 en Daily.
""",
        "REGRESIONES": """
Tres modelos anidados sobre el error de pronóstico:

1. solo el índice de complejidad,
2. solo las entropías clásicas,
3. las dos cosas juntas.

La comparación de R² responde la pregunta que un jurado va a hacer: **¿el índice aporta algo
que las entropías ya conocidas no tengan?** Si el R² del modelo 3 es prácticamente igual al del
modelo 2, el índice es redundante. Los coeficientes del modelo 3 se guardan aparte para poder
discutir qué aporta cada componente.
""",
    },
}

E["06_m4_metrics.py"] = {
    "ejecutar_main": True,
    "titulo": "06 · Métricas oficiales de M4: sMAPE, MASE y OWA",
    "preparacion": (
        "## Preparación\n\n"
        "`CON_ARIMA` pide que se incluyan ARIMA(1,1,1) y SARIMA(1,1,1)(1,1,1,m), no solo los "
        "benchmarks. Son ~36 min sobre las 100.000 series.\n\n"
        "> Sin esta variable el notebook correría **sin** ARIMA y regeneraría "
        "`m4_metrics_por_serie.xlsx` sin esas columnas, pisando el resultado de la corrida larga.",
        'import os\nimport sys\nfrom pathlib import Path\n\n'
        '_raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
        'sys.path.insert(0, str(_raiz / "src"))\n\n'
        '# incluir ARIMA y SARIMA (~36 min). Sin esto, solo naive/snaive/naive2.\n'
        'os.environ["CON_ARIMA"] = "1"'),
    "intro": """
Implementa la métrica con la que **se decidió realmente la competencia M4**, que ninguno de los
scripts anteriores calcula.

### Las tres fórmulas

**sMAPE** — error porcentual simétrico, en **porcentaje** (el del script 04 queda en fracción):

$$\\text{sMAPE} = \\frac{200}{h}\\sum_{t=1}^{h} \\frac{|A_t - F_t|}{|A_t| + |F_t|}$$

**MASE** — error absoluto medio **escalado** por el error del naive estacional *dentro de la
muestra de entrenamiento*:

$$\\text{MASE} = \\frac{\\frac{1}{h}\\sum_{t=1}^{h}|A_t - F_t|}{\\frac{1}{n-m}\\sum_{t=m+1}^{n}|Y_t - Y_{t-m}|}$$

Ese denominador es lo que hace comparables frecuencias distintas: divide por la variabilidad
propia de la serie. MASE = 1 significa "igual de bueno que el naive estacional un paso adelante".

**OWA** — el promedio de las dos, cada una normalizada contra el benchmark **Naive2**:

$$\\text{OWA} = \\frac{1}{2}\\left(\\frac{\\text{sMAPE}}{\\text{sMAPE}_{Naive2}} + \\frac{\\text{MASE}}{\\text{MASE}_{Naive2}}\\right)$$

Por construcción Naive2 vale exactamente 1,000. **Por debajo de 1 se le gana al benchmark.**

### Por qué hace falta Naive2 y no alcanza con el naive común

Naive2 no repite el último valor. Es: testear estacionalidad → si da significativa,
desestacionalizar → aplicar naive sobre la serie ajustada → **volver a aplicar los índices
estacionales** al pronóstico. Como es el denominador de todos los OWA, un Naive2 mal
implementado corre todos los resultados.

### Estacionalidad según M4

`Yearly 1 · Quarterly 4 · Monthly 12 · Weekly 1 · Daily 1 · Hourly 24`

M4 trata Weekly y Daily como **no estacionales**. El script 04 usaba 52 y 7, que es la elección
"natural" pero no la de la competencia.

### Validación

El script compara su Naive2 contra los valores publicados en Makridakis, Spiliotis &
Assimakopoulos (2020). Resultado:

| Frecuencia | sMAPE calculado | sMAPE M4 | dif |
|---|---|---|---|
| Yearly | 16,342 | 16,342 | 0,000 |
| Quarterly | 11,024 | 11,012 | +0,012 |
| Monthly | 14,385 | 14,427 | −0,042 |
| Weekly | 9,161 | 9,161 | 0,000 |
| Daily | 3,045 | 3,045 | 0,000 |
| Hourly | 18,616 | 18,383 | +0,233 |
| **Todas** | **13,548** | **13,564** | **−0,016** |

Las tres frecuencias con m=1 dan **exactas** (ahí Naive2 = Naive, sin descomposición).
Las estacionales difieren en décimas porque `seasonal_decompose` de statsmodels no es idéntica
al código original de M4. El desvío global es de 0,016 puntos de sMAPE: la implementación
reproduce el benchmark.
""",
    "secciones": {
        "CONFIGURACION": """
`M4_EXACTO` merece explicación. El código oficial de M4 tiene una particularidad en el test de
estacionalidad: el primer término de la suma **no va al cuadrado**.

```python
s = acf(ts, 1)                          # <- sin cuadrado
for i in range(2, ppy): s += acf(ts, i) ** 2
```

La fórmula de libro (Box-Jenkins) eleva todos los términos. Es casi seguro un desliz del código
original, pero es el que produjo los resultados publicados. Se deja en `True` para reproducir M4;
en `False` usa la fórmula estándar.
""",
        "METRICAS": """
`smape` enmascara los puntos donde real y pronóstico son ambos cero (división 0/0) en lugar de
devolver infinito.

`mase` devuelve `NaN` si el denominador da cero, que pasa con series constantes: ahí el naive
estacional no comete error y el cociente no está definido.
""",
        "NAIVE2: TEST DE ESTACIONALIDAD Y DESESTACIONALIZACION": """
El corazón del script.

**`test_estacionalidad`** compara |acf(m)| contra la banda de confianza del 90%
(el 1,645 es el cuantil normal). Exige además al menos 3 ciclos completos: con menos, la
autocorrelación estacional no es estimable.

**`indices_estacionales`** hace descomposición clásica **multiplicativa** y normaliza los índices
para que sumen m. Devuelve unos (= sin estacionalidad) en tres casos: el test no da significativo,
la serie tiene valores ≤ 0 (el modelo multiplicativo no está definido), o la descomposición falla.

**`forecast_naive2`** divide la serie por su ciclo, toma el último valor ajustado, lo repite h
veces y lo **vuelve a multiplicar** por los índices que corresponden al horizonte futuro.
Sin ese último paso el pronóstico queda desestacionalizado y el error se dispara.
""",
        "EVALUACION DE UNA SERIE": """
Devuelve una fila con sMAPE y MASE de cada método, más metadatos útiles: `h`, `m`, largo del
entrenamiento y si la serie pasó el test de estacionalidad.

Esa columna `estacional` permite después cruzar los resultados con el índice de complejidad:
¿el índice del script 01 se comporta distinto en las series que M4 considera estacionales?
""",
        "AGREGACION: OWA": """
Detalle que se pasa por alto fácil: **OWA se calcula sobre los promedios, no serie por serie.**

Es decir, primero se promedian sMAPE y MASE sobre todas las series del grupo, y recién ahí se
hace el cociente contra el promedio de Naive2. Calcular el OWA de cada serie y después promediar
da un número distinto — y no es el de la competencia.
""",
        "VALIDACION CONTRA LOS VALORES PUBLICADOS": """
El control de calidad del script. Si nuestro Naive2 no reproduce el publicado, el denominador de
todos los OWA está mal y ningún resultado es comparable con la literatura.

Las frecuencias con m=1 (Yearly, Weekly, Daily) tienen que dar **exactas**, porque ahí Naive2
degenera en Naive y no interviene ninguna descomposición. Si esas tres no dan exactas, el error
está en el sMAPE o en el MASE, no en la parte estacional. Es un buen test diagnóstico.
""",
    },
}


# ─────────────────────────────────────────────────────────
# Este script NO vive en src/: esta en la carpeta de los resultados de la
# tesis, porque lee los consolidados de los modelos. Se documenta igual, con
# "origen" y "destino" apuntando alla.
# ─────────────────────────────────────────────────────────

TESIS = SRC.parent.parent / "TRANSPOSE_1000_RANDOM"

E["owa_modelos.py"] = {
    "ejecutar_main": True,
    "origen": TESIS / "owa_modelos.py",
    "destino": TESIS / "notebooks",
    "etiqueta": "TRANSPOSE_1000_RANDOM/owa_modelos.py",
    "titulo": "OWA de los 26 modelos de la tesis",
    "preparacion": (
        "## Preparación\n\n"
        "Dos cosas hacen falta acá:\n\n"
        "1. El script se ubica con `Path(__file__)`, y en un notebook `__file__` **no existe**. "
        "Se define apuntando al `.py` real.\n"
        "2. Las funciones de métrica se importan de `m4-structural-complexity/src/06_m4_metrics.py`, "
        "que el propio script agrega al path.",
        'from pathlib import Path\n\n'
        '# El script usa Path(__file__) para ubicarse; en el notebook no existe.\n'
        '_raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
        'if "__file__" not in globals():\n'
        '    __file__ = str(_raiz / "owa_modelos.py")'),
    "intro": """
Cierra el círculo de la tesis: pone los **26 modelos propios** en la misma escala con la que
se midieron los 61 métodos de la competencia M4.

**De dónde salen los pronósticos.** Cada hoja de los consolidados tiene, además de la tabla
de métricas, un bloque que arranca en la fila `Datos Prueba` con **el pronóstico de cada
modelo**. Eso es lo que se necesita: con los pronósticos se puede calcular cualquier métrica,
sin volver a correr nada.

**Por qué no se usa el SMAPE ya guardado.** El del notebook es
`100·mean(2|F−A|/(|A|+|F|+1e-6))`, que es la misma fórmula de M4 salvo el epsilon. Pero el
**MASE no está guardado**, y para calcularlo hace falta el pronóstico igual. Ya que hay que
leerlo, se recalcula todo con una sola implementación —la del script 06— para modelos y
benchmarks por igual.

**Fuente de verdad.** Los valores reales y las series de entrenamiento se leen de los CSV
**originales de M4**, no de los transpuestos. Así el denominador del MASE y el Naive2 son
idénticos a los del script 06. El script además verifica que los reales guardados en cada
consolidado coincidan con los oficiales: en las 4.773 series, **cero desajustes**.

### Resultado

Seis modelos le ganan a Naive2 (OWA reponderado a la población de M4):

| # | Modelo | OWA |
|---|---|---|
| 1 | **ARIMA** | **0,936** |
| 2 | **SARIMA** | 0,938 |
| 3 | LassoLars | 0,974 |
| 4 | Lineal | 0,978 |
| 5 | Holt-Winters | 0,995 |
| 6 | ETS | 0,998 |
| — | *Naive2* | *1,000* |

Como referencia, el ganador de M4 (Smyl, híbrido ES-RNN) obtuvo 0,821.

**Dos validaciones que salieron gratis:** el `Naive` de la tesis reproduce exactamente el
`naive` de este script (sMAPE 14,280 / MASE 3,305) y el `Naive Estacional` reproduce el
`snaive` (12,033 / 2,433). Dos implementaciones independientes dando el mismo número.
""",
    "secciones": {
        "CONFIGURACION": """
`h` y `m` son los oficiales de M4, no los "naturales": Weekly y Daily van con **m=1**.

`EXCLUIR` filtra las copias de respaldo (`_ORIGINAL`, `_PREVIO`, `CORRUPTO`…) que quedaron
en las carpetas de trabajo y no son resultados vigentes. Sin ese filtro se contarían corridas
viejas junto con las buenas.
""",
        "LECTURA DE LOS CONSOLIDADOS": """
**El detalle importante: los sufijos `v2`, `v3`.**

Cuando el notebook de la tesis encuentra que un modelo *ya* tenía pronóstico para esa serie,
no lo pisa: agrega una columna nueva.

```python
while col in df_pron.columns:
    k += 1
    col = f"{base} v{k}"
```

O sea que `"LSTM v2"` **no es otro modelo**: es la misma serie recorrida dos veces (pasa cuando
un lote se reinició y volvió a procesar una serie cuyo archivo individual seguía ahí). En YEARLY
hay 146 de estas. Se conserva la primera, para que haya exactamente un pronóstico por
(serie, modelo). Sin esta limpieza aparecerían "modelos" fantasma evaluados sobre 2 o 5 series.

**Sobre la lectura.** Se usa `openpyxl` en modo `read_only` en vez de `pd.read_excel`. Son 4.773
hojas repartidas en ~30 archivos; con pandas, que construye un DataFrame por hoja, esto tardaría
un orden de magnitud más. Cada hoja tiene apenas 40 filas, así que se levantan enteras a memoria
y se buscan las filas que interesan.
""",
        "CALCULO": """
Para cada serie se calculan además los tres benchmarks (`naive2`, `naive`, `snaive`) con las
**mismas funciones del script 06**. Es deliberado: si modelos y benchmarks se midieran con
implementaciones distintas, el cociente que define el OWA no sería válido.

El control de integridad compara los `Datos Prueba` guardados en el consolidado contra los
valores oficiales de M4. Si no coincidieran, la evaluación de esa serie no valdría nada — es
barato de verificar y caro de descubrir tarde.

### El detalle que más cambia los resultados

`resumir()` restringe el denominador Naive2 a **las mismas series que el modelo pudo pronosticar**.

Sin eso, Holt-Winters —que solo existe donde m>1, o sea 2.414 de las 4.773 series— quedaría
dividido por un Naive2 promediado sobre series que el modelo nunca vio. Con la corrección su OWA
pasa de **0,992 a 1,135**: la diferencia entre "le gana al benchmark" y "no le gana". El mismo
ajuste aplica, en menor medida, a las redes secuenciales, que se saltean las series cortas.
""",
        "OWA REPONDERADO A LA POBLACION DE M4": """
La muestra de la tesis **no tiene la mezcla de frecuencias de M4**:

| | muestra | M4 |
|---|---|---|
| Daily | 21% | 4,2% |
| Hourly | 8,7% | 0,4% |
| Monthly | 21% | 48% |

Daily y Hourly están fuertemente sobre-representadas, y son justo donde varios modelos se
descontrolan. Reponderar corrige la mezcla y da el número comparable con la literatura.

**Cómo se pondera.** Se ponderan sMAPE y MASE **por separado** y recién después se hace el
cociente. Es la forma correcta: M4 agrega como razón de promedios, no como promedio de razones,
y las dos cosas no son iguales.

Para modelos que no cubren las seis frecuencias, los pesos se renormalizan sobre las que sí tiene.
""",
        "PRINCIPAL": """
Tres tablas y cuatro archivos.

Vale la pena mirar `mase_extremos` antes de leer cualquier promedio. Cuatro modelos —ARIMA_NN,
SARIMA_NN, LSTM y LSTM_Conv1D— tienen OWA de 23 a 31, pero **sus medianas están entre 0,9 y 1,8**:

- `ARIMA_NN` en Hourly: OWA 4.341 por **3 series**
- `SARIMA_NN` en Monthly: OWA 114,8 por **17 series**
- `Holt`: 118 series extremas, **117 de ellas Hourly** — extrapola tendencia lineal a 48 pasos

No fallan en promedio: **fallan catastróficamente en casos puntuales**. Para una aplicación de
pronóstico eso es peor que ser mediocre de forma consistente, y es el argumento directo para
ponerle una cota de cordura al pronóstico antes de mostrarlo.
""",
    },
}


# =========================================================
# CONSTRUCCION DE NOTEBOOKS
# =========================================================

def partir_en_secciones(src):
    """Corta el fuente en (titulo, codigo) usando los banners. Lo previo al
    primer banner queda como preambulo."""
    lineas = src.split("\n")
    marcas = [i for i, l in enumerate(lineas) if BANNER.match(l)]

    pares = []
    i = 0
    while i + 1 < len(marcas):
        a, b = marcas[i], marcas[i + 1]
        # entre dos banners solo puede haber lineas de comentario: si no, no
        # es un encabezado sino dos banners sueltos
        cuerpo = lineas[a + 1:b]
        if cuerpo and all(l.strip().startswith("#") for l in cuerpo):
            titulo = " · ".join(l.strip().lstrip("#").strip() for l in cuerpo)
            pares.append((a, b, titulo))
            i += 2
        else:
            i += 1

    if not pares:
        return "\n".join(lineas), []

    preambulo = "\n".join(lineas[:pares[0][0]])
    secciones = []
    for k, (_, b, titulo) in enumerate(pares):
        fin = pares[k + 1][0] if k + 1 < len(pares) else len(lineas)
        codigo = "\n".join(lineas[b + 1:fin]).strip("\n")
        secciones.append((titulo, codigo))
    return preambulo, secciones


# El campo "source" se guarda como STRING, no como lista. Si se guarda como
# lista, cada elemento tiene que traer su propio "\n" al final: Jupyter las
# concatena sin agregar saltos y el codigo queda todo en una linea.
def md(texto):
    return {"cell_type": "markdown", "metadata": {},
            "source": texto.strip("\n")}


def code(texto):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": texto.rstrip("\n")}


GUARDA = """if __name__ == "__main__":
    sys.exit(main())"""

NOTA_MAIN = """# En el .py esto se ejecuta solo al correr el script. En el notebook queda
# a pedido, para poder leer las definiciones sin lanzar el calculo completo:
#
#     main()"""


LLAMADA_MAIN = """# En un notebook __name__ SI vale "__main__", asi que la guarda del .py se
# disparaba sola. Se reemplaza por una llamada explicita.
import multiprocessing as _mp

# macOS usa "spawn" por defecto desde Python 3.8: los workers arrancan un
# interprete nuevo e importan el modulo, y las funciones definidas en las
# celdas de un notebook viven en __main__, que no se puede reimportar. Con
# "fork" el worker hereda el proceso entero y las ve tal cual.
if _mp.get_start_method(allow_none=True) != "fork":
    _mp.set_start_method("fork", force=True)

main()"""


def sin_guarda_main(codigo, ejecutar=False):
    """Reemplaza el bloque `if __name__ == "__main__"` por una nota.

    En un notebook __name__ SI vale "__main__", asi que la guarda se dispara
    sola: al ejecutarlo arrancaria el calculo entero y ademas sys.exit()
    cortaria el kernel."""
    if GUARDA not in codigo:
        return codigo
    return codigo.replace(GUARDA, LLAMADA_MAIN if ejecutar else NOTA_MAIN)


# Preparacion por defecto: los scripts de src/ solo necesitan que src/ este
# en el path para poder importar rutas.py.
PREP_SRC = (
    "## Preparación\n\nAgrega `src/` al path para poder importar `rutas.py`, "
    "que resuelve dónde se leen y escriben los archivos.",
    'import sys\nfrom pathlib import Path\n\n'
    '_raiz = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()\n'
    'sys.path.insert(0, str(_raiz / "src"))',
)


def construir(archivo, spec):
    origen = spec.get("origen", SRC / archivo)
    src = origen.read_text(encoding="utf-8")
    preambulo, secciones = partir_en_secciones(src)

    etiqueta = spec.get("etiqueta", f"src/{archivo}")
    prep_md, prep_code = spec.get("preparacion", PREP_SRC)

    celdas = [
        md(f"# {spec['titulo']}\n\n"
           f"*Generado desde `{etiqueta}` por `src/generar_notebooks.py`.*\n\n"
           f"> El archivo que se corre en producción es el `.py`. Este notebook documenta el\n"
           f"> procedimiento; si cambia el script, se regenera con\n"
           f"> `python3 src/generar_notebooks.py`."),
        md(spec["intro"]),
        md(prep_md),
        code(prep_code),
    ]

    # el primer "banner" de varios scripts es el encabezado del archivo:
    # si el titulo contiene el nombre del .py, se muestra como imports
    if secciones and archivo.split(".")[0] in secciones[0][0]:
        titulo, codigo = secciones.pop(0)
        celdas += [md("## Imports"), code((preambulo + "\n" + codigo).strip())]
    elif preambulo.strip():
        celdas += [md("## Imports y documentación del script"), code(preambulo.strip())]

    for titulo, codigo in secciones:
        if not codigo.strip():
            continue
        celdas.append(md(f"## {titulo.title()}"))
        expl = spec["secciones"].get(titulo)
        if expl:
            celdas.append(md(expl))
        celdas.append(code(sin_guarda_main(codigo, spec.get("ejecutar_main", False))))

    # nbformat 4.5+ exige un id por celda
    for i, c in enumerate(celdas):
        c["id"] = f"celda-{i:02d}"

    return {"cells": celdas,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4, "nbformat_minor": 5}


def tiene_salidas(ruta):
    """True si el notebook ya fue ejecutado y conserva resultados."""
    try:
        nb = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return any(c.get("outputs") for c in nb.get("cells", [])
               if c.get("cell_type") == "code")


def main():
    NB_DIR.mkdir(exist_ok=True)
    forzar = "--forzar" in sys.argv
    saltados = []

    for archivo, spec in E.items():
        nb = construir(archivo, spec)
        carpeta = spec.get("destino", NB_DIR)
        carpeta.mkdir(exist_ok=True)
        destino = carpeta / (archivo.replace(".py", ".ipynb"))

        # NO pisar un notebook que ya tiene resultados adentro. Paso esto:
        # se regeneraron los notebooks despues de ejecutarlos y se perdieron
        # 20 h de salidas (los datos seguian en results/, pero el notebook
        # quedaba vacio). Con --forzar se sobrescribe igual.
        if destino.exists() and not forzar and tiene_salidas(destino):
            saltados.append(destino)
            continue

        destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        try:
            mostrar = destino.relative_to(SRC.parent)
        except ValueError:              # notebook fuera de este proyecto
            mostrar = destino.relative_to(SRC.parent.parent)
        print(f"  {mostrar}   {n_md} markdown + {n_code} código")
    if saltados:
        print(f"\n⚠ {len(saltados)} notebook(s) NO se regeneraron porque ya tienen "
              f"resultados adentro:")
        for d in saltados:
            print(f"    {d.name}")
        print("  Para sobrescribirlos igual:  python3 src/generar_notebooks.py --forzar")
    print(f"\n{len(E) - len(saltados)} de {len(E)} notebooks generados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
