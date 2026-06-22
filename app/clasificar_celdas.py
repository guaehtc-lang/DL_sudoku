import cv2
import numpy as np


def preparar_celda(
    celda,
    margen=5,
    area_minima=80
):

    if len(celda.shape) == 3:

        celda = cv2.cvtColor(
            celda,
            cv2.COLOR_BGR2GRAY
        )

    celda = celda[
        margen:-margen,
        margen:-margen
    ]

    celda = cv2.GaussianBlur(
        celda,
        (3, 3),
        0
    )

    _, binaria = cv2.threshold(
        celda,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    cantidad, etiquetas, estadisticas, _ = (
        cv2.connectedComponentsWithStats(
            binaria,
            connectivity=8
        )
    )

    alto_imagen, ancho_imagen = (
        binaria.shape
    )

    componentes_validos = []

    for indice in range(1, cantidad):

        x, y, ancho, alto, area = (
            estadisticas[indice]
        )

        toca_borde = (
            x <= 1
            or y <= 1
            or x + ancho >= ancho_imagen - 1
            or y + alto >= alto_imagen - 1
        )

        proporcion = (
            ancho / alto
            if alto > 0
            else 0
        )

        if toca_borde:
            continue

        if area < area_minima:
            continue

        if ancho < 3 or alto < 10:
            continue

        if proporcion < 0.12 or proporcion > 1.5:
            continue

        componentes_validos.append(
            (
                area,
                indice,
                x,
                y,
                ancho,
                alto
            )
        )

    if len(componentes_validos) == 0:

        return (
            np.zeros(
                (28, 28),
                dtype=np.uint8
            ),
            False
        )

    _, indice, x, y, ancho, alto = max(
        componentes_validos,
        key=lambda componente: componente[0]
    )

    componente = np.zeros_like(
        binaria
    )

    componente[
        etiquetas == indice
    ] = 255

    contenido = componente[
        y:y + alto,
        x:x + ancho
    ]

    escala = 20 / max(
        ancho,
        alto
    )

    nuevo_ancho = max(
        1,
        int(round(ancho * escala))
    )

    nuevo_alto = max(
        1,
        int(round(alto * escala))
    )

    contenido = cv2.resize(
        contenido,
        (
            nuevo_ancho,
            nuevo_alto
        ),
        interpolation=cv2.INTER_AREA
    )

    imagen = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    inicio_x = (
        28 - nuevo_ancho
    ) // 2

    inicio_y = (
        28 - nuevo_alto
    ) // 2

    imagen[
        inicio_y:inicio_y + nuevo_alto,
        inicio_x:inicio_x + nuevo_ancho
    ] = contenido

    momentos = cv2.moments(
        imagen
    )

    if momentos["m00"] != 0:

        centro_x = (
            momentos["m10"]
            / momentos["m00"]
        )

        centro_y = (
            momentos["m01"]
            / momentos["m00"]
        )

        mover_x = int(
            round(14 - centro_x)
        )

        mover_y = int(
            round(14 - centro_y)
        )

        matriz = np.float32([
            [1, 0, mover_x],
            [0, 1, mover_y]
        ])

        imagen = cv2.warpAffine(
            imagen,
            matriz,
            (28, 28)
        )

    return imagen, True


def clasificar_celdas(celdas, modelo_class):

    celdas_preparadas = []
    contenido_detectado = []

    for celda in celdas:

        celda_preparada, tiene_contenido = (
            preparar_celda(
                celda
            )
        )

        celdas_preparadas.append(
            celda_preparada
        )

        contenido_detectado.append(
            tiene_contenido
        )

    celdas_preparadas = np.array(
        celdas_preparadas
    )

    contenido_detectado = np.array(
        contenido_detectado
    )

    entrada = (
        celdas_preparadas.astype(
            "float32"
        )
        / 255.0
    ).reshape(
        -1,
        28,
        28,
        1
    )

    probabilidades = modelo_class.predict(
        entrada,
        verbose=0
    )

    numeros = np.argmax(
        probabilidades,
        axis=1
    )

    confianzas = np.max(
        probabilidades,
        axis=1
    )

    numeros[
        ~contenido_detectado
    ] = 0

    confianzas[
        ~contenido_detectado
    ] = 1.0

    puzzle = numeros.reshape(
        9,
        9
    )

    confianzas = confianzas.reshape(
        9,
        9
    )

    contenido_detectado = (
        contenido_detectado.reshape(
            9,
            9
        )
    )

    return (
        puzzle,
        confianzas,
        contenido_detectado,
        celdas_preparadas
    )
