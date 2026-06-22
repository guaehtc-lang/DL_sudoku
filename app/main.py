from pipeline_sudoku import (
    cargar_imagen,
    analizar_imagen,
    resolver_puzzle,
    mostrar_resultados
)


if __name__ == "__main__":

    ruta_imagen = (
        "../modelo_yolo/img_proc/images/test/"
        "sudoku_168_jpg.rf.pfhTWZY1G9sbBr6PPDKz.jpg"
    )

    imagen = cargar_imagen(
        ruta_imagen
    )

    (
        tablero,
        puzzle,
        confianzas,
        contenido_detectado,
        celdas_preparadas
    ) = analizar_imagen(
        imagen
    )

    solucion, valido, estadisticas = resolver_puzzle(
        puzzle
    )

    mostrar_resultados(
        puzzle,
        solucion,
        valido,
        estadisticas
    )
