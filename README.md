# Sudoku Solver con Deep Learning

Proyecto de Deep Learning que detecta un Sudoku en una imagen, reconoce sus números y resuelve el tablero automáticamente.

---

## 1. Objetivo del proyecto

El objetivo es transformar una fotografía de un Sudoku en un tablero resuelto.

```text
Imagen
→ detectar el tablero
→ corregir la perspectiva
→ dividir en 81 celdas
→ reconocer los números
→ resolver el Sudoku
→ mostrar el resultado
```

---

## 2. Estructura del proyecto

```text
DL_sudoku/
├── modelo_yolo/
├── modelo_class/
├── modelo_juego/
├── app/
├── README.md
└── requirements.txt
```

Cada carpeta resuelve una parte distinta del problema.

---

# 3. Modelo YOLO

Carpeta:

```text
modelo_yolo/
```

## Objetivo

Detectar dónde está el tablero de Sudoku dentro de la imagen.

## Archivos principales

```text
modelo_yolo_colab_entrenamiento.ipynb
modelo_yolo_local.ipynb
modelo/best.pt
data.yaml
```

### `modelo_yolo_colab_entrenamiento.ipynb`

Se utiliza para entrenar YOLO en Google Colab.

```text
dataset de imágenes
→ entrenamiento YOLOv8
→ guardado de best.pt
```

### `modelo_yolo_local.ipynb`

Se utiliza para evaluar el modelo ya entrenado.

Muestra:

- Precision
- Recall
- mAP50
- mAP50-95
- detección gráfica sobre una imagen
- recorte del tablero

### `best.pt`

Es el modelo YOLO final utilizado por la aplicación.

---

# 4. OpenCV

OpenCV no reconoce números ni resuelve el Sudoku.

Su función es preparar correctamente la imagen para los modelos.

## Archivo

```text
app/procesar_sudoku.py
```

Después de que YOLO detecta el tablero, OpenCV:

1. recorta el Sudoku;
2. detecta sus límites;
3. corrige la perspectiva;
4. genera una vista frontal;
5. divide el tablero en 81 celdas.

```text
imagen original
→ tablero detectado
→ tablero corregido
→ 81 celdas
```

## Preparación de las celdas

Archivo:

```text
app/clasificar_celdas.py
```

OpenCV también:

1. convierte cada celda a escala de grises;
2. elimina márgenes;
3. aplica binarización;
4. elimina restos de cuadrícula;
5. detecta si la celda está vacía;
6. centra el número;
7. redimensiona a 28 × 28 píxeles.

---

# 5. Modelo de clasificación

Carpeta:

```text
modelo_class/
```

## Objetivo

Clasificar cada celda como:

```text
0 = celda vacía
1–9 = número reconocido
```

## Archivos principales

```text
modelo_class_colab_0_9.ipynb
modelo/best_class_0_9.keras
```

### `modelo_class_colab_0_9.ipynb`

Genera imágenes sintéticas de celdas y entrena una CNN.

Se utilizan números impresos porque los Sudokus no contienen dígitos manuscritos.

No se usa MNIST porque:

- MNIST contiene números escritos a mano;
- no incluye una clase específica para celdas vacías;
- las celdas reales contienen ruido y restos de cuadrícula;
- el dominio visual es diferente.

### `best_class_0_9.keras`

Es el modelo final que reconoce las 81 celdas.

Resultado principal:

```text
Accuracy aproximada: 99,92 %
```

La salida del modelo es una matriz de 9 × 9 con el Sudoku reconocido.

---

# 6. Dataset del modelo de juego

El archivo `sudoku.csv` se utiliza únicamente para entrenar `modelo_juego`; no se incluye en GitHub por su tamaño y se conserva `sudoku_5.csv` como muestra.


# 7. Modelo de juego

Carpeta:

```text
modelo_juego/
```

## Objetivo

Aprender qué números son más probables en las celdas vacías.

## Archivos principales

```text
modelo_juego_colab.ipynb
modelo/best_cnn_1M.keras
modelo/best_lstm_1M.keras
modelo/best_juego.keras
modelo/tipo_modelo.txt
```

En el notebook se comparan dos modelos:

```text
CNN
LSTM
```

La CNN obtiene mejores resultados y se guarda como:

```text
best_juego.keras
```

---

# 8. CNN + MRV + backtracking

La resolución final combina tres elementos.

## CNN

La CNN calcula qué números son más probables en cada celda vacía.

Ejemplo:

```text
5 → 70 %
7 → 20 %
2 → 10 %
```

La CNN ordena los candidatos, pero no garantiza por sí sola una solución correcta.

## MRV

MRV significa:

```text
Minimum Remaining Values
```

Selecciona primero la celda con menos candidatos legales.

Esto reduce el número de combinaciones que hay que probar.

## Backtracking

El backtracking prueba un candidato.

Si aparece una contradicción:

```text
se borra el número
→ se vuelve atrás
→ se prueba otro candidato
```

## Flujo conjunto

```text
MRV elige la celda más restrictiva
→ la CNN ordena los candidatos
→ backtracking prueba cada opción
→ si falla, retrocede
→ si funciona, continúa
```

El backtracking se ejecuta en:

```text
app/resolver_sudoku.py
```

---

# 9. Aplicación final

Carpeta:

```text
app/
```

## Archivos

```text
procesar_sudoku.py
clasificar_celdas.py
resolver_sudoku.py
main.py
main_streamlit.py
prueba_integracion.ipynb
models/
```

## `procesar_sudoku.py`

```text
YOLO detecta el tablero
→ OpenCV corrige la perspectiva
→ se generan 81 celdas
```

## `clasificar_celdas.py`

```text
81 celdas
→ limpieza OpenCV
→ CNN 0–9
→ matriz del Sudoku
```

## `resolver_sudoku.py`

```text
puzzle reconocido
→ CNN
→ MRV
→ backtracking
→ solución válida
```

## `prueba_integracion.ipynb`

Valida el proceso completo paso a paso.

Permite mostrar:

- imagen original;
- tablero corregido;
- 81 celdas;
- puzzle reconocido;
- solución final;
- nodos y retrocesos.

## `main.py`

Es una versión directa de la prueba de integración.

```text
cargar modelos
→ cargar imagen
→ procesar_sudoku()
→ clasificar_celdas()
→ validar puzzle
→ resolver_sudoku()
→ validar solución
```

## `main_streamlit.py`

Es la interfaz gráfica final.

Permite:

1. subir una imagen;
2. detectar el Sudoku;
3. resolverlo;
4. mostrar el resultado;
5. reiniciar y probar otra imagen.

---

# 10. Modelos utilizados por la aplicación

Dentro de:

```text
app/models/
```

se encuentran:

```text
best.pt
best_class_0_9.keras
best_juego.keras
```

Relación:

```text
best.pt
→ detecta el tablero

best_class_0_9.keras
→ reconoce las celdas

best_juego.keras
→ ordena los candidatos para resolver
```

---

# 11. Ejecutar la aplicación

Desde la carpeta raíz:

```bash
cd app
streamlit run main_streamlit.py
```

Después:

1. subir una imagen;
2. pulsar **Resolver**;
3. comprobar el Sudoku resuelto;
4. usar **Probar con otro Sudoku** para reiniciar.

---

# 12. Resultados principales

- YOLO detecta correctamente el tablero.
- OpenCV corrige la perspectiva y genera 81 celdas.
- El clasificador 0–9 alcanza aproximadamente un 99,92 % de accuracy sintética.
- La CNN de juego mejora frente a la LSTM.
- CNN + MRV + backtracking generan soluciones válidas.
- La integración completa funciona en Streamlit.

---

# 13. Limitaciones

El sistema depende de la calidad de la imagen.

Puede fallar con:

- imágenes borrosas;
- sombras intensas;
- perspectiva extrema;
- números poco definidos;
- cuadrículas incompletas;
- Sudokus con símbolos no impresos.

La aplicación muestra un mensaje de error y permite reiniciar el proceso.
