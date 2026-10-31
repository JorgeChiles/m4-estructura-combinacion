#!/usr/bin/env python3
"""Genera rasgos_v2.py copiando las celdas del notebook que calculan los 29
descriptores.

La validacion externa sobre M3 tiene que calcular los descriptores con el mismo
codigo que los calculo sobre M4, o no prueba nada: una reimplementacion que
difiera en un detalle ---el orden de los rezagos de la ACF, el periodo que se le
pasa a STL, el umbral del detector de puntos de cambio--- haria que el indice
transferido no fuera el mismo indice. El codigo vive en el notebook
CLASIFICACION_V3_All.ipynb (celdas 0, 2 y 4), que es la entrada original del
pipeline, asi que en vez de copiarlo a mano se extrae.

Uso:  python3 extraer_rasgos.py
"""
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
NB = AQUI / "m4-structural-complexity" / "CLASIFICACION_V3_All.ipynb"
DEST = AQUI / "rasgos_v2.py"
CELDAS = (0, 2, 4)          # imports, funciones auxiliares, extract_features_v2

CABECERA = '''"""Los 29 descriptores estructurales, tal como se calcularon sobre M4.

ESTE ARCHIVO NO SE EDITA A MANO. Lo genera `extraer_rasgos.py` copiando
literalmente las celdas 0, 2 y 4 del notebook CLASIFICACION_V3_All.ipynb, que es
donde se calcularon los descriptores de las 100.000 series de M4. Se extrae a
modulo para que la validacion externa sobre M3 use EL MISMO codigo y no una
reimplementacion: si los descriptores se calcularan distinto, la transferencia
del indice no probaria nada.

Lo unico que se quita de las celdas es `import matplotlib.pyplot as plt`, que
aqui no hace falta y abre un backend grafico al importar.
"""
'''


def main():
    nb = json.loads(NB.read_text())
    partes = []
    for i in CELDAS:
        txt = "".join(nb["cells"][i]["source"]).rstrip()
        txt = "\n".join(l for l in txt.split("\n") if "matplotlib" not in l)
        partes.append(f"# ===== celda {i} del notebook =====\n{txt}")
    DEST.write_text(CABECERA + "\n" + "\n\n\n".join(partes) + "\n")
    print(f"{DEST.name}: celdas {CELDAS} de {NB.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
