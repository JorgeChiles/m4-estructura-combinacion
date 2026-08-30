#!/usr/bin/env python3
"""
Rutas compartidas por los scripts del pipeline.

Todo lo que el pipeline PRODUCE va a results/. En la raiz quedan solo las
entradas originales (features_m4_all_v2.xlsx, el notebook) y data/ con los
CSV de M4.

Por que existe entrada()
------------------------
Los scripts encadenan: 02 lee lo que escribio 01, 05 lee lo que escribieron
01, 03 y 04b. Si un archivo todavia esta en la raiz (corridas viejas, antes
de esta reorganizacion), entrada() lo encuentra igual en vez de fallar.
Busca en results/ primero y usa la raiz como respaldo.

Uso:
    from rutas import BASE_DIR, DATA_DIR, RES_DIR, entrada, salida

    df = pd.read_excel(entrada("df_features_complexity.xlsx"))
    df.to_excel(salida("df_entropy.xlsx"), index=False)
"""
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:                      # pegado en un notebook
    BASE_DIR = Path.cwd()
    if BASE_DIR.name == "src":
        BASE_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
RES_DIR = BASE_DIR / "results"


def salida(nombre):
    """Ruta donde escribir un resultado. Crea results/ si no existe."""
    RES_DIR.mkdir(exist_ok=True)
    return RES_DIR / nombre


def entrada(nombre):
    """Ruta desde donde leer: results/ primero, raiz del proyecto como respaldo."""
    p = RES_DIR / nombre
    return p if p.exists() else BASE_DIR / nombre
