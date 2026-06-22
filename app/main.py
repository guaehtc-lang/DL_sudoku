import cv2

from tensorflow import keras
from ultralytics import YOLO

from procesar_sudoku import procesar_sudoku
from clasificar_celdas import clasificar_celdas
from resolver_sudoku import (
    puzzle_parcial_valido,
    resolver_sudoku,
    sudoku_valido
)


# Ruta de la imagen de prueba
ruta_imagen = (
    "../modelo_yolo/img_proc/images/test/"
    "sudoku_168_jpg.rf.pfhTWZY1G9sbBr6PPDKz.jpg"
)


# Cargar los modelos
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


# YOLO detecta el tablero
# OpenCV corrige la perspectiva y genera 81 celdas
tablero, celdas = procesar_sudoku(
    imagen,
    modelo_yolo
)


# La CNN clasifica las 81 celdas
(
    puzzle,
    confianzas,
    contenido_detectado,
    celdas_preparadas
) = clasificar_celdas(
    celdas,
    modelo_class
)


# Comprobar que el puzzle no tiene contradicciones
if not puzzle_parcial_valido(
    puzzle
):

    raise ValueError(
        "El puzzle reconocido contiene contradicciones."
    )


# Resolver el Sudoku
# En resolver_sudoku.py se ejecutan:
# CNN + MRV + backtracking
solucion, estadisticas = resolver_sudoku(
    puzzle,
    modelo_juego
)

if solucion is None:

    raise ValueError(
        "No se ha encontrado una solución."
    )


# Comprobar la solución
valido = sudoku_valido(
    solucion
)

if not valido:

    raise ValueError(
        "La solución obtenida no es válida."
    )


# Mostrar resultados
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
