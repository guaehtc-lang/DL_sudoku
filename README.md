# Sudoku Solver con Deep Learning

Proyecto de Deep Learning que detecta un Sudoku en una imagen, reconoce sus números y resuelve el tablero automáticamente.

## Funcionamiento

```text
Imagen
→ YOLO detecta el tablero
→ OpenCV corrige la perspectiva y divide 81 celdas
→ CNN clasifica cada celda como vacía o número 1–9
→ CNN + MRV + backtracking resuelven el Sudoku
→ Streamlit muestra el resultado
```

## Modelos

- **YOLOv8**: detecta el tablero de Sudoku.
- **Modelo de clasificación**: reconoce celdas vacías y números impresos del 1 al 9.
- **Modelo de juego**: ordena los candidatos de cada celda.
- **Backtracking**: corrige decisiones incorrectas hasta encontrar una solución válida.

## Estructura

```text
DL_sudoku/
├── modelo_yolo/
├── modelo_class/
├── modelo_juego/
├── app/
├── README.md
└── requirements.txt
```

La carpeta `app/models/` contiene los tres modelos utilizados por la aplicación:

```text
best.pt
best_class_0_9.keras
best_juego.keras
```

## Ejecutar la aplicación

Desde la carpeta raíz del proyecto:

```bash
cd app
streamlit run main_streamlit.py
```

Después:

1. Sube una imagen de un Sudoku.
2. Pulsa **Resolver**.
3. La aplicación mostrará el tablero resuelto.
4. Si la imagen no se reconoce correctamente, podrás reiniciar y probar otra.

## Dataset

El archivo completo `sudoku.csv` no se incluye en GitHub por su tamaño.  
Se conserva `sudoku_5.csv` como muestra de la estructura utilizada.

## Resultados principales

- Detección YOLO con métricas altas en validación y test.
- Clasificador 0–9 con una accuracy aproximada del **99,92 %** sobre datos sintéticos.
- Resolución completa mediante CNN y backtracking.
- Aplicación Streamlit integrada y funcional.

## Limitaciones

El reconocimiento depende de la calidad de la imagen. Fotografías inclinadas, borrosas, con sombras intensas o números poco definidos pueden producir errores.
