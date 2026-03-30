<div style="width: 100%; clear: both;">
<div style="float: left; width: 50%;">
<img src="http://www.uoc.edu/portal/_resources/common/imatges/marca_UOC/UOC_Masterbrand.jpg", align="left">
</div>
<div style="float: right; width: 50%;">
<p style="margin: 0; padding-top: 22px; text-align:right;">M2.875 · Deep Learning · PEC2</p>
<p style="margin: 0; text-align:right;">2025-2 · Máster universitario en Ciencia de datos (Data science)</p>
<p style="margin: 0; text-align:right; padding-button: 100px;">Estudios de Informática, Multimedia y Telecomunicación</p>
</div>
</div>
<div style="width:100%;">&nbsp;</div>


# PEC 2: Redes neuronales recurrentes con Keras


<u>Consideraciones generales</u>:

- Esta PEC debe realizarse de forma **estrictamente individual**. Cualquier indicio de copia será penalizado con un suspenso (D) para todas las partes implicadas y la posible evaluación negativa de la asignatura de forma íntegra.
- Es necesario que el estudiante indique **todas las fuentes** que ha utilizado para la realización de la PEC. De no ser así, se considerará que el estudiante ha cometido plagio, siendo penalizado con un suspenso (D) y la posible evaluación negativa de la asignatura de forma íntegra.

<u>Formato de la entrega</u>:

- Algunos ejercicios pueden suponer varios minutos de ejecución, por lo que la entrega debe hacerse en **formato notebook** y en **formato html**, donde se vea el código, los resultados y comentarios de cada ejercicio. Se puede exportar el notebook a HTML desde el menú File $\to$ Download as $\to$ HTML.
- Existe un tipo de celda especial para albergar texto. Este tipo de celda os será muy útil para responder a las diferentes preguntas teóricas planteadas a lo largo de la actividad. Para cambiar el tipo de celda a este tipo, en el menú: Cell $\to$ Cell Type $\to$ Markdown.

# 0. Contexto y carga de librerías

En esta práctica exploraremos el uso de **Redes Neuronales Recurrentes (RNN)** y sus variantes modernas (LSTM, GRU) aplicadas a diferentes problemas de series temporales y procesamiento de legunaje natural. Las RNN son especialmente útiles cuando los datos presentan una **estructura secuencial**, ya que permiten capturar dependencias a lo largo del tiempo o del texto.

En los ejercicios se trabajarán desde ejemplos sintéticos hasta aplicaciones reales en finanzas, análisis de sentiemientos y predicción meteorológica, con el objetivo de comprender tanto el uso práctico de estas arquitecturas como sus limitaciones.

A continuación se incluyen las librerías necesarias para realizar esta PEC:

```python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input, GRU, LSTM, Dropout, Embedding, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
```
En este ejercicio vamos a explorar cómo una **Red Neuronal Recurrente (RNN)** puede aprender patrones en datos de series temporales. Las RNN son un tipo de red diseñado para manejar secuencias, ya que no procesan los datos como entradas independientes, sino que mantienen un estado interno que les permite recordar información sobre pasos previos. Esto las hace especialmente útiles en tareas como predicción de series temporales, procesamiento de lenguaje natural (NLP) o análisis de señales.

Trabajaremos con una serie temporal para la predicción de la temperatura del aceite de transformadores eléctricos. Más información sobre el dataset: https://github.com/zhouhaoyi/ETDataset

En el modelado de series temporales, el parámetro `window_size` es fundamental. Representa el **número de pasos anteriores** que el modelo utilizará como entrada para predecir el siguiente valor. Podemos imaginarlo como una **ventana deslizante** (o sliding window) que se mueve a lo largo de la serie. En cada posición, recoge un bloque de datos pasados y lo utiliza para estimar el valor futuro inmediato. Por ejemplo, si `window_size = 5`, y tenemos la serie `1, 2, 3, 4, 5, 6, 7`, la red recibiría como entrada `[1, 2, 3, 4, 5]` y debería predecir como salida `6`.


Por otra parte, la **reproducibilidad** es la capacidad de obtener los **mismos resultados** cuando un experimento se repite bajo las mismas condiciones: mismo código, mismos datos de entrada, mismos hiperparámetros, mismo entorno de ejecución. En deep learning, lograr la reproducibilidad es más difícil por varias razones:
- Inicialización aleatoria de pesos de la red.
- Barajado aleatorio (shuffling) de los datos durante el entrenamiento.
- Aritmética de coma flotante. Las sumas de coma flotante no tienen porque seguir la propiedad asociativa debido al redondeo. Además, en las GPUs el orden de las operaciones puede cambiar entre ejecuciones.

Para solucionar esto, se fijan semillas aleatorias (**random seeds**) en todos los componentes que generan números aleatorios:
```python
np.random.seed()
random.seed()
tf.random.set_seed() (para TensorFlow)
```


<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">

**Ejercicio [2.5 pts.].** Carga los datos con la serie temporal, entrena una RNN simple con distintos tamaños de ventana, compara los resultados y visualizalos. Para ello:
- Descarga el dataset con una muestra reducida de los datos https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv
- Realiza un breve análisis exploratorio de los datos.
- Preprocesa los datos de entrada.  
    - Crea una función para convertir los datos en secuencias supervisadas. Dado un tamaño de ventana (`window_size`) ha de dividir la serie en secuencias de entrada (una subventana de longitud `window_size`) y valores de salida (el valor siguiente a cada ventana).
    - Divide los datos en 70% para entrenamiento, 15% para validación y 15% para test.
    - Escala los valores entre 0 y 1.     
- Crea un modelo secuencial con:
    - Una capa `SimpleRNN` y una capa de salida `Dense`.
    - Compila el modelo con loss `mean_squared_error` y optimizador `Adam`.
    - Usa semillas aleatorias para mantener la reproducibilidad.
- Calcula el RMSE (Root Mean Squared Error) y MAE (Mean Squared Error) para los tres subconjuntos de datos. Guarda los resultados en una tabla para hacer comparaciones posteriores.
- Visualiza las curvas de loss por epoch y los valores reales vs. valores predichos. Recuerda desescalar las variables para ver los valores reales y no los normalizados.

Responde a las siguientes preguntas:
- ¿Por qué deben convertirse los datos de series temporales en ventanas deslizantes?
- ¿Qué limitaciones tiene una SimpleRNN para datos con secuencias temporales largas?
</div>


