import cv2
import numpy as np
import pandas as pd
import streamlit as st

from main import analizar_imagen, resolver_puzzle


st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🧩",
    layout="wide"
)


def reiniciar_resultados():

    claves = [
        "imagen",
        "tablero",
        "puzzle",
        "confianzas",
        "contenido_detectado",
        "solucion",
        "estadisticas"
    ]

    for clave in claves:

        if clave in st.session_state:
            del st.session_state[clave]


def convertir_imagen(archivo):

    bytes_imagen = np.asarray(
        bytearray(archivo.read()),
        dtype=np.uint8
    )

    imagen = cv2.imdecode(
        bytes_imagen,
        cv2.IMREAD_COLOR
    )

    if imagen is None:
        raise ValueError(
            "No se ha podido leer la imagen."
        )

    return imagen


def preparar_puzzle_editado(tabla):

    puzzle = tabla.to_numpy()

    if np.isnan(puzzle).any():
        raise ValueError(
            "La matriz contiene celdas vacías no válidas."
        )

    puzzle = puzzle.astype(int)

    if np.any(puzzle < 0) or np.any(puzzle > 9):
        raise ValueError(
            "Solo se permiten valores entre 0 y 9."
        )

    return puzzle


def mostrar_tablero(tablero, titulo):

    tabla = pd.DataFrame(
        tablero,
        index=range(1, 10),
        columns=range(1, 10)
    )

    st.markdown(f"### {titulo}")

    st.dataframe(
        tabla,
        use_container_width=True
    )


st.title("🧩 Resolución automática de Sudokus")

st.write(
    "Sube una fotografía, revisa la matriz reconocida "
    "y corrige cualquier número antes de resolver."
)


archivo = st.file_uploader(
    "Selecciona una imagen",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    on_change=reiniciar_resultados
)


if archivo is not None:

    if "puzzle" not in st.session_state:

        try:

            imagen = convertir_imagen(
                archivo
            )

            with st.spinner(
                "Detectando y leyendo el Sudoku..."
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
            st.session_state.confianzas = confianzas
            st.session_state.contenido_detectado = (
                contenido_detectado
            )

        except Exception as error:

            st.error(
                f"Error al procesar la imagen: {error}"
            )

            st.stop()


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


    st.markdown("---")
    st.markdown("### Matriz reconocida")

    st.caption(
        "Usa 0 para representar una celda vacía. "
        "Corrige cualquier error antes de resolver."
    )

    puzzle_df = pd.DataFrame(
        st.session_state.puzzle,
        index=range(1, 10),
        columns=range(1, 10)
    )

    puzzle_editado = st.data_editor(
        puzzle_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            columna: st.column_config.NumberColumn(
                str(columna),
                min_value=0,
                max_value=9,
                step=1,
                format="%d"
            )
            for columna in puzzle_df.columns
        },
        key="editor_puzzle"
    )


    columna_resolver, columna_reiniciar = st.columns(
        [1, 1]
    )

    with columna_resolver:

        resolver = st.button(
            "Resolver Sudoku",
            type="primary",
            use_container_width=True
        )

    with columna_reiniciar:

        reiniciar = st.button(
            "Reiniciar",
            use_container_width=True
        )


    if reiniciar:

        reiniciar_resultados()
        st.rerun()


    if resolver:

        try:

            puzzle_corregido = preparar_puzzle_editado(
                puzzle_editado
            )

            with st.spinner(
                "Resolviendo Sudoku..."
            ):

                (
                    solucion,
                    valido,
                    estadisticas
                ) = resolver_puzzle(
                    puzzle_corregido
                )

            if solucion is None:

                st.error(
                    "No se ha encontrado una solución."
                )

            elif not valido:

                st.error(
                    "La solución obtenida no es válida."
                )

            else:

                st.session_state.solucion = solucion
                st.session_state.estadisticas = (
                    estadisticas
                )

                st.success(
                    "Sudoku resuelto correctamente."
                )

        except Exception as error:

            st.error(
                f"No se puede resolver el Sudoku: {error}"
            )


    if "solucion" in st.session_state:

        st.markdown("---")

        columna_puzzle, columna_solucion = st.columns(2)

        with columna_puzzle:

            mostrar_tablero(
                preparar_puzzle_editado(
                    puzzle_editado
                ),
                "Puzzle corregido"
            )

        with columna_solucion:

            mostrar_tablero(
                st.session_state.solucion,
                "Solución"
            )

        st.markdown("### Estadísticas")

        metrica_1, metrica_2 = st.columns(2)

        metrica_1.metric(
            "Nodos explorados",
            st.session_state.estadisticas[
                "nodos"
            ]
        )

        metrica_2.metric(
            "Retrocesos",
            st.session_state.estadisticas[
                "retrocesos"
            ]
        )


else:

    st.info(
        "Sube una imagen para comenzar."
    )
