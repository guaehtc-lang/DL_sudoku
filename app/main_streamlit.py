import cv2
import numpy as np
import streamlit as st

from pipeline_sudoku import analizar_imagen, resolver_puzzle


st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🧩",
    layout="wide"
)


if "uploader_version" not in st.session_state:
    st.session_state.uploader_version = 0


def limpiar_resultados():
    """Borra los datos de la imagen anterior."""

    for clave in [
        "archivo_id",
        "imagen",
        "tablero",
        "puzzle",
        "solucion",
        "error"
    ]:

        if clave in st.session_state:
            del st.session_state[clave]


def reiniciar_aplicacion():

    limpiar_resultados()
    st.session_state.uploader_version += 1


def convertir_imagen(archivo):
    """Convierte el archivo subido en una imagen de OpenCV."""

    datos = np.frombuffer(
        archivo.getvalue(),
        dtype=np.uint8
    )

    imagen = cv2.imdecode(
        datos,
        cv2.IMREAD_COLOR
    )

    if imagen is None:
        raise ValueError(
            "No se ha podido leer la imagen."
        )

    return imagen


def crear_imagen_solucion(
    puzzle_inicial,
    solucion
):
    """Dibuja el Sudoku final y diferencia las pistas originales."""

    tamaño = 720
    margen = 28
    celda = (tamaño - 2 * margen) // 9

    color_fondo = (247, 249, 252)
    color_pista = (205, 222, 240)
    color_resuelta = (255, 255, 255)
    color_linea = (55, 75, 102)
    color_pista_texto = (43, 65, 96)
    color_resuelta_texto = (45, 45, 45)

    imagen = np.full(
        (tamaño, tamaño, 3),
        color_fondo,
        dtype=np.uint8
    )

    puzzle_inicial = np.array(
        puzzle_inicial,
        dtype=int
    ).reshape(9, 9)

    solucion = np.array(
        solucion,
        dtype=int
    ).reshape(9, 9)

    # Fondos de las celdas
    for fila in range(9):
        for columna in range(9):

            x1 = margen + columna * celda
            y1 = margen + fila * celda
            x2 = x1 + celda
            y2 = y1 + celda

            color = (
                color_pista
                if puzzle_inicial[fila, columna] != 0
                else color_resuelta
            )

            cv2.rectangle(
                imagen,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

    # Cuadrícula
    for i in range(10):

        grosor = 4 if i % 3 == 0 else 1
        posicion = margen + i * celda

        cv2.line(
            imagen,
            (margen, posicion),
            (margen + 9 * celda, posicion),
            color_linea,
            grosor,
            cv2.LINE_AA
        )

        cv2.line(
            imagen,
            (posicion, margen),
            (posicion, margen + 9 * celda),
            color_linea,
            grosor,
            cv2.LINE_AA
        )

    # Números
    for fila in range(9):
        for columna in range(9):

            numero = int(
                solucion[fila, columna]
            )

            es_pista = (
                puzzle_inicial[fila, columna] != 0
            )

            color_texto = (
                color_pista_texto
                if es_pista
                else color_resuelta_texto
            )

            escala = 1.65 if es_pista else 1.45
            grosor = 3 if es_pista else 2

            texto = str(numero)
            fuente = cv2.FONT_HERSHEY_SIMPLEX

            (ancho, alto), _ = cv2.getTextSize(
                texto,
                fuente,
                escala,
                grosor
            )

            x = (
                margen
                + columna * celda
                + (celda - ancho) // 2
            )

            y = (
                margen
                + fila * celda
                + (celda + alto) // 2
            )

            cv2.putText(
                imagen,
                texto,
                (x, y),
                fuente,
                escala,
                color_texto,
                grosor,
                cv2.LINE_AA
            )

    return imagen


st.title("🧩 Resolución automática de Sudokus")

st.write(
    "Sube una fotografía de un Sudoku "
    "y pulsa Resolver."
)


archivo = st.file_uploader(
    "Selecciona una imagen",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key=(
        f"archivo_sudoku_"
        f"{st.session_state.uploader_version}"
    )
)


# Pantalla inicial
if archivo is None:

    st.info(
        "Sube una imagen para comenzar."
    )

    st.stop()


archivo_id = (
    archivo.name,
    archivo.size
)


# Procesar una imagen nueva
if st.session_state.get("archivo_id") != archivo_id:

    limpiar_resultados()
    st.session_state.archivo_id = archivo_id

    try:

        imagen = convertir_imagen(
            archivo
        )

        with st.spinner(
            "Detectando el Sudoku..."
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

        st.session_state.imagen = imagen
        st.session_state.tablero = tablero
        st.session_state.puzzle = puzzle

    except Exception:

        st.session_state.error = (
            "La imagen no se ha podido procesar correctamente. "
            "Prueba con otra fotografía más clara y centrada."
        )


# Pantalla de error
if "error" in st.session_state:

    st.error(
        st.session_state.error
    )

    if st.button(
        "Reiniciar y subir otra imagen",
        type="primary",
        use_container_width=True
    ):

        reiniciar_aplicacion()
        st.rerun()

    st.stop()


# Pantalla de resultado
if "solucion" in st.session_state:

    imagen_solucion = crear_imagen_solucion(
        st.session_state.puzzle,
        st.session_state.solucion
    )

    st.markdown("### Sudoku resuelto")

    izquierda, centro, derecha = st.columns(
        [1, 2, 1]
    )

    with centro:

        st.image(
            cv2.cvtColor(
                imagen_solucion,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        if st.button(
            "Probar con otro Sudoku",
            type="primary",
            use_container_width=True
        ):

            reiniciar_aplicacion()
            st.rerun()

    st.stop()


# Pantalla previa a resolver
columna_1, columna_2 = st.columns(2)

with columna_1:

    st.markdown("### Imagen original")

    st.image(
        cv2.cvtColor(
            st.session_state.imagen,
            cv2.COLOR_BGR2RGB
        ),
        use_container_width=True
    )


with columna_2:

    st.markdown("### Tablero detectado")

    st.image(
        cv2.cvtColor(
            st.session_state.tablero,
            cv2.COLOR_BGR2RGB
        ),
        use_container_width=True
    )


if st.button(
    "Resolver",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Resolviendo Sudoku..."
        ):

            solucion, valido, estadisticas = (
                resolver_puzzle(
                    st.session_state.puzzle
                )
            )

        if solucion is None or not valido:

            raise ValueError

        st.session_state.solucion = solucion
        st.rerun()

    except Exception:

        st.session_state.error = (
            "La imagen no se ha reconocido correctamente "
            "o el Sudoku contiene contradicciones. "
            "Prueba con otra imagen."
        )

        st.rerun()
