import numpy as np

from tensorflow import keras


def puzzle_parcial_valido(tablero):
    """Comprueba que las pistas no se repitan en filas, columnas o bloques."""

    tablero = np.array(
        tablero,
        dtype=int
    ).reshape(9, 9)

    for i in range(9):

        fila = tablero[i, :]
        fila = fila[fila != 0]

        if len(fila) != len(set(fila)):
            return False

        columna = tablero[:, i]
        columna = columna[columna != 0]

        if len(columna) != len(set(columna)):
            return False

    for fila in range(0, 9, 3):
        for columna in range(0, 9, 3):

            bloque = tablero[
                fila:fila + 3,
                columna:columna + 3
            ].flatten()

            bloque = bloque[
                bloque != 0
            ]

            if len(bloque) != len(set(bloque)):
                return False

    return True


def sudoku_valido(tablero):
    """Comprueba que el Sudoku completo contiene los números del 1 al 9."""

    if tablero is None:
        return False

    tablero = np.array(
        tablero,
        dtype=int
    ).reshape(9, 9)

    numeros = set(
        range(1, 10)
    )

    for i in range(9):

        if set(tablero[i, :]) != numeros:
            return False

        if set(tablero[:, i]) != numeros:
            return False

    for fila in range(0, 9, 3):
        for columna in range(0, 9, 3):

            bloque = tablero[
                fila:fila + 3,
                columna:columna + 3
            ]

            if set(
                bloque.flatten()
            ) != numeros:
                return False

    return True


def crear_candidatos(tableros):
    """Marca los números permitidos en cada celda vacía."""

    # One-hot transforma cada número en un canal independiente.
    one_hot = keras.utils.to_categorical(
        tableros,
        num_classes=10
    ).astype("float32")

    numeros = one_hot[
        ...,
        1:
    ]

    usados_fila = numeros.max(
        axis=2
    )[:, :, None, :]

    usados_columna = numeros.max(
        axis=1
    )[:, None, :, :]

    bloques = numeros.reshape(
        -1,
        3,
        3,
        3,
        3,
        9
    )

    usados_bloque = bloques.max(
        axis=(2, 4)
    )

    usados_bloque = np.repeat(
        np.repeat(
            usados_bloque,
            3,
            axis=1
        ),
        3,
        axis=2
    )

    candidatos = 1 - np.maximum.reduce([
        np.broadcast_to(
            usados_fila,
            numeros.shape
        ),
        np.broadcast_to(
            usados_columna,
            numeros.shape
        ),
        usados_bloque
    ])

    candidatos[
        tableros != 0
    ] = 0

    return candidatos.astype(
        "float32"
    )


def crear_entrada_cnn(tableros):
    """Crea los 19 canales de entrada: 10 del tablero y 9 candidatos."""

    one_hot = keras.utils.to_categorical(
        tableros,
        num_classes=10
    ).astype("float32")

    candidatos = crear_candidatos(
        tableros
    )

    return np.concatenate(
        [
            one_hot,
            candidatos
        ],
        axis=-1
    )


def candidatos_validos(
    tablero,
    fila,
    columna
):
    """Devuelve los números permitidos por las reglas del Sudoku."""

    usados = set(
        tablero[fila, :]
    )

    usados.update(
        tablero[:, columna]
    )

    fila_bloque = (
        fila // 3
    ) * 3

    columna_bloque = (
        columna // 3
    ) * 3

    usados.update(
        tablero[
            fila_bloque:fila_bloque + 3,
            columna_bloque:columna_bloque + 3
        ].flatten()
    )

    return [
        numero
        for numero in range(1, 10)
        if numero not in usados
    ]


def seleccionar_celda_mrv(tablero):
    """MRV elige la celda vacía con menos candidatos posibles."""

    mejor_celda = None
    mejores_candidatos = None

    for fila in range(9):
        for columna in range(9):

            if tablero[
                fila,
                columna
            ] != 0:
                continue

            candidatos = candidatos_validos(
                tablero,
                fila,
                columna
            )

            if len(candidatos) == 0:
                return fila, columna, []

            if (
                mejores_candidatos is None
                or len(candidatos)
                < len(mejores_candidatos)
            ):

                mejor_celda = (
                    fila,
                    columna
                )

                mejores_candidatos = candidatos

            if len(mejores_candidatos) == 1:
                break

        if (
            mejores_candidatos is not None
            and len(mejores_candidatos) == 1
        ):
            break

    if mejor_celda is None:
        return None

    fila, columna = mejor_celda

    return (
        fila,
        columna,
        mejores_candidatos
    )


def predecir_probabilidades(
    tablero,
    modelo_juego
):
    """La CNN ordena los candidatos de mayor a menor probabilidad."""

    entrada = crear_entrada_cnn(
        tablero.reshape(
            1,
            9,
            9
        )
    )

    probabilidades = modelo_juego(
        entrada,
        training=False
    ).numpy()[0]

    return probabilidades


def resolver_sudoku(
    puzzle,
    modelo_juego,
    max_nodos=100000
):
    """Resuelve el Sudoku combinando CNN, MRV y backtracking."""

    tablero = np.array(
        puzzle,
        dtype=int
    ).reshape(9, 9).copy()

    if not puzzle_parcial_valido(
        tablero
    ):

        raise ValueError(
            "El puzzle inicial contiene contradicciones."
        )

    estadisticas = {
        "nodos": 0,
        "retrocesos": 0
    }

    # BACKTRACKING
    # Prueba candidatos ordenados por la CNN. Si una elección no conduce
    # a una solución, borra ese número y prueba el siguiente candidato.
    def buscar():

        if estadisticas["nodos"] >= max_nodos:
            return False

        estadisticas["nodos"] += 1

        if 0 not in tablero:

            return sudoku_valido(
                tablero
            )

        seleccion = seleccionar_celda_mrv(
            tablero
        )

        if seleccion is None:
            return False

        fila, columna, candidatos = (
            seleccion
        )

        if len(candidatos) == 0:

            estadisticas[
                "retrocesos"
            ] += 1

            return False

        probabilidades = predecir_probabilidades(
            tablero,
            modelo_juego
        )

        candidatos = sorted(
            candidatos,
            key=lambda numero: probabilidades[
                fila,
                columna,
                numero - 1
            ],
            reverse=True
        )

        for numero in candidatos:

            tablero[
                fila,
                columna
            ] = numero

            if buscar():
                return True

            # Retroceso: se deshace la elección incorrecta.
            tablero[
                fila,
                columna
            ] = 0

            estadisticas[
                "retrocesos"
            ] += 1

        return False

    resuelto = buscar()

    if resuelto:

        return (
            tablero.copy(),
            estadisticas
        )

    return (
        None,
        estadisticas
    )
