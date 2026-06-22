import cv2
import numpy as np


def agrupar(posiciones, distancia=8):
    """Agrupa líneas próximas y devuelve una posición media."""

    posiciones = sorted(posiciones)
    grupos = []

    for posicion in posiciones:

        if not grupos:
            grupos.append([posicion])

        elif posicion - np.mean(grupos[-1]) <= distancia:
            grupos[-1].append(posicion)

        else:
            grupos.append([posicion])

    return [
        int(np.mean(grupo))
        for grupo in grupos
    ]


def procesar_sudoku(imagen, modelo_yolo):
    """Detecta el tablero, corrige la perspectiva y genera 81 celdas."""

    if imagen is None:
        raise ValueError(
            "La imagen no es válida."
        )

    # YOLO localiza el Sudoku dentro de la imagen.
    resultado = modelo_yolo(
        imagen,
        verbose=False
    )[0]

    if len(resultado.boxes) == 0:
        raise ValueError(
            "No se ha detectado ningún Sudoku."
        )

    indice = int(
        resultado.boxes.conf.argmax().item()
    )

    x1, y1, x2, y2 = (
        resultado.boxes.xyxy[indice]
        .cpu()
        .numpy()
        .astype(int)
    )

    tablero = imagen[
        y1:y2,
        x1:x2
    ]

    if tablero.size == 0:
        raise ValueError(
            "El recorte del tablero está vacío."
        )

    # OpenCV detecta las líneas exteriores de la cuadrícula.
    gris = cv2.cvtColor(
        tablero,
        cv2.COLOR_BGR2GRAY
    )

    gris = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    binaria = cv2.adaptiveThreshold(
        gris,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    lineas = cv2.HoughLinesP(
        binaria,
        1,
        np.pi / 180,
        80,
        minLineLength=80,
        maxLineGap=15
    )

    if lineas is None:
        raise ValueError(
            "No se ha detectado la cuadrícula."
        )

    horizontales = []
    verticales = []

    for linea in lineas:

        x1, y1, x2, y2 = linea[0]

        if abs(x2 - x1) > abs(y2 - y1):

            horizontales.append(
                int((y1 + y2) / 2)
            )

        else:

            verticales.append(
                int((x1 + x2) / 2)
            )

    horizontales = agrupar(
        horizontales
    )

    verticales = agrupar(
        verticales
    )

    if len(horizontales) < 2 or len(verticales) < 2:
        raise ValueError(
            "No se han detectado los bordes."
        )

    origen = np.float32([
        [verticales[0], horizontales[0]],
        [verticales[-1], horizontales[0]],
        [verticales[-1], horizontales[-1]],
        [verticales[0], horizontales[-1]]
    ])

    destino = np.float32([
        [0, 0],
        [449, 0],
        [449, 449],
        [0, 449]
    ])

    # La transformación deja el tablero recto y cuadrado.
    matriz = cv2.getPerspectiveTransform(
        origen,
        destino
    )

    tablero = cv2.warpPerspective(
        tablero,
        matriz,
        (450, 450)
    )

    celdas = []

    for fila in range(9):
        for columna in range(9):

            celda = tablero[
                fila * 50:(fila + 1) * 50,
                columna * 50:(columna + 1) * 50
            ]

            celdas.append(celda)

    return tablero, celdas
