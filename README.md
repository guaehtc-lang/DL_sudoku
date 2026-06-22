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

### 1. Detección del tablero

Archivo:

```text
app/procesar_sudoku.py
```

Carga el modelo YOLO `best.pt`, localiza el Sudoku en la imagen y recorta el tablero detectado.

### 2. Corrección y división en 81 celdas

Archivo:

```text
app/procesar_sudoku.py
```

OpenCV corrige la perspectiva para obtener una cuadrícula frontal y regular. Después divide el tablero en 81 celdas.

### 3. Reconocimiento de números

Archivo:

```text
app/clasificar_celdas.py
```

Cada celda se limpia y se redimensiona a `28 × 28` píxeles. El modelo `best_class_0_9.keras` clasifica el contenido:

```text
0 = celda vacía
1–9 = número reconocido
```

El resultado es una matriz de `9 × 9` con el Sudoku inicial.

### 4. Resolución del Sudoku

Archivo:

```text
app/resolver_sudoku.py
```

El modelo `best_juego.keras` calcula qué números son más probables en cada celda vacía.

El algoritmo MRV selecciona primero la celda con menos candidatos. Después, el backtracking prueba los candidatos en orden de probabilidad y retrocede cuando una elección provoca una contradicción.

### 5. Coordinación del proceso

Archivo:

```text
app/pipeline_sudoku.py
```

Conecta las distintas fases:

```text
detección
→ procesamiento
→ clasificación
→ validación
→ resolución
```

También carga los tres modelos utilizados por la aplicación.

### 6. Ejecución por consola

Archivo:

```text
app/main.py
```

Contiene únicamente las llamadas necesarias para ejecutar el proceso completo desde Python.

### 7. Interfaz gráfica

Archivo:

```text
app/main_streamlit.py
```

Permite subir una imagen, ejecutar el proceso y mostrar el Sudoku resuelto en el navegador.

## Modelos

- **YOLOv8**: detecta el tablero de Sudoku.
- **Modelo de clasificación**: reconoce celdas vacías y números impresos del 1 al 9.
- **Modelo de juego**: ordena los candidatos de cada celda.
- **MRV + backtracking**: seleccionan las celdas más restrictivas y corrigen decisiones incorrectas hasta encontrar una solución válida.

## Estructura del proyecto

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
4. Si la imagen no se reconoce correctamente, reinicia el proceso y prueba otra imagen.

## Dataset

El archivo completo `sudoku.csv` no se incluye en GitHub por su tamaño.

Se conserva `sudoku_5.csv` como muestra de la estructura utilizada para entrenar el modelo de juego.

## Resultados principales

- Detección YOLO evaluada en validación y test.
- Clasificador 0–9 con una accuracy aproximada del **99,92 %** sobre datos sintéticos.
- Resolución completa mediante CNN, MRV y backtracking.
- Aplicación Streamlit integrada y funcional.

## Limitaciones

El reconocimiento depende de la calidad de la imagen. Fotografías inclinadas, borrosas, con sombras intensas o números poco definidos pueden producir errores.
