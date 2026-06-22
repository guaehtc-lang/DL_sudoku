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


modelo_yolo = YOLO(
    "models/best.pt"
)

modelo_class = keras.models.load_model(
    "models/best_class_0_9.keras"
)

modelo_juego = keras.models.load_model(
    "models/best_juego.keras"
)


def analizar_imagen(imagen):

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

    if not puzzle_parcial_valido(
        puzzle
    ):

        raise ValueError(
            "El puzzle reconocido contiene contradicciones."
        )

    solucion, estadisticas = resolver_sudoku(
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


def main(
    imagen,
    max_nodos=100000
):

    (
        tablero,
        puzzle,
        confianzas,
        contenido_detectado,
        celdas_preparadas
    ) = analizar_imagen(
        imagen
    )

    (
        solucion,
        valido,
        estadisticas
    ) = resolver_puzzle(
        puzzle,
        max_nodos=max_nodos
    )

    return (
        tablero,
        puzzle,
        solucion,
        valido,
        confianzas,
        contenido_detectado,
        estadisticas
    )


if __name__ == "__main__":

    ruta_imagen = (
        "../modelo_yolo/img_proc/images/test/"
        "sudoku_168_jpg.rf.pfhTWZY1G9sbBr6PPDKz.jpg"
    )

    imagen = cv2.imread(
        ruta_imagen
    )

    if imagen is None:
        raise ValueError(
            "No se ha podido cargar la imagen de prueba."
        )

    (
        tablero,
        puzzle,
        solucion,
        valido,
        confianzas,
        contenido_detectado,
        estadisticas
    ) = main(
        imagen
    )

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
