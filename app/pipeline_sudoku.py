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


# Los modelos se cargan una sola vez al importar este módulo.
modelo_yolo = YOLO(
    "models/best.pt"
)

modelo_class = keras.models.load_model(
    "models/best_class_0_9.keras"
)

modelo_juego = keras.models.load_model(
    "models/best_juego.keras"
)


def cargar_imagen(ruta_imagen):
    """Carga una imagen local y comprueba que sea válida."""

    imagen = cv2.imread(
        ruta_imagen
    )

    if imagen is None:
        raise ValueError(
            "No se ha podido cargar la imagen."
        )

    return imagen


def analizar_imagen(imagen):
    """Detecta el tablero y convierte sus 81 celdas en números."""

    tablero, celdas = procesar_sudoku(
        imagen,
        modelo_yolo
    )

    (
        puzzle,
        confianzas,
        contenido_detectado,
        celdas_preparadas
    ) = clasificar_celdas(
        celdas,
        modelo_class
    )

    return (
        tablero,
        puzzle,
        confianzas,
        contenido_detectado,
        celdas_preparadas
    )


def resolver_puzzle(
    puzzle,
    max_nodos=100000
):
    """Valida el puzzle y ejecuta CNN + MRV + backtracking."""

    if not puzzle_parcial_valido(
        puzzle
    ):

        raise ValueError(
            "El puzzle reconocido contiene contradicciones."
        )

    solucion, estadisticas = resolver_con_backtracking(
        puzzle,
        modelo_juego,
        max_nodos=max_nodos
    )

    valido = sudoku_valido(
        solucion
    )

    return (
        solucion,
        valido,
        estadisticas
    )


def mostrar_resultados(
    puzzle,
    solucion,
    valido,
    estadisticas
):
    """Muestra el resultado de la prueba ejecutada desde main.py."""

    print("Puzzle reconocido:")
    print(puzzle)

    print("\nSudoku resuelto:")
    print(solucion)

    print("\nSudoku válido:", valido)
    print("Nodos:", estadisticas["nodos"])
    print(
        "Retrocesos:",
        estadisticas["retrocesos"]
    )
