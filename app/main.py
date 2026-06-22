import cv2

from tensorflow import keras
from ultralytics import YOLO

from procesar_sudoku import procesar_sudoku
from clasificar_celdas import clasificar_celdas
from resolver_sudoku import (
    puzzle_parcial_valido,
    resolver_sudoku as resolver_con_backtracking,
    sudoku_valido
)


ruta_imagen = (
    "../modelo_yolo/img_proc/images/test/"
    "sudoku_168_jpg.rf.pfhTWZY1G9sbBr6PPDKz.jpg"
)


# Cargar los tres modelos
modelo_yolo = YOLO(
    "models/best.pt"
)

modelo_class = keras.models.load_model(
    "models/best_class_0_9.keras"
)

modelo_juego = keras.models.load_model(
    "models/best_juego.keras"
)


# Cargar la imagen
imagen = cv2.imread(
    ruta_imagen
)

if imagen is None:

    raise ValueError(
        "No se ha podido cargar la imagen."
    )


# Detectar, corregir y dividir el tablero
tablero, celdas = procesar_sudoku(
    imagen,
    modelo_yolo
)


# Clasificar las 81 celdas
(
    puzzle,
    confianzas,
    contenido_detectado,
    celdas_preparadas
) = clasificar_celdas(
    celdas,
    modelo_class
)


# Validar el puzzle reconocido
if not puzzle_parcial_valido(
    puzzle
):

    raise ValueError(
        "El puzzle reconocido contiene contradicciones."
    )


# BACKTRACKING
# MRV selecciona la celda más restrictiva.
# La CNN ordena los candidatos.
# Si un candidato falla, se borra y se prueba el siguiente.
solucion, estadisticas = (
    resolver_con_backtracking(
        puzzle,
        modelo_juego,
        max_nodos=100000
    )
)


if solucion is None:

    raise ValueError(
        "No se ha encontrado una solución."
    )


# Validar y mostrar el resultado
valido = sudoku_valido(
    solucion
)

print("Puzzle reconocido:")
print(puzzle)

print("\nSudoku resuelto:")
print(solucion)

print(
    "\nSudoku válido:",
    valido
)

print(
    "Nodos:",
    estadisticas["nodos"]
)

print(
    "Retrocesos:",
    estadisticas["retrocesos"]
)
