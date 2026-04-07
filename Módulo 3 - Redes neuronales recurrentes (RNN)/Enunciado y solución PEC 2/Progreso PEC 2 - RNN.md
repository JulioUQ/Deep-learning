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
# Proporciona funciones para interactuar con el sistema operativo (como rutas de archivos)
import os  

# Permite modificar aspectos del entorno de ejecución de Python, como la lista de rutas de búsqueda de módulos (sys.path)
import sys

# Sube un nivel desde /PEC/
root_dir = os.path.abspath('..')  
sys.path.append(root_dir)

# Análisis y manipulación de datos
import pandas as pd

# Estilo tablas
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
  
# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo visualizacion
sns.set(style="whitegrid")
  
# Operaciones matematicas y vectorizacion
import numpy as np
import random
  
# Librerias para DL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
  
# Para guardar modelos y resultados
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input, GRU, LSTM, Dropout, Embedding, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
  
# Comprobar valores de RAM (CPU/GPU) y Disco
import psutil
import shutil
  
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

## 1.1. Carga del *Electricity Transformer Dataset (ETDataset)*

El dataset ETT-small (*Electricity Transformer Temperature*) se enmarca en el problema de la distribución eléctrica, donde uno de los mayores retos es predecir la demanda futura de energía. Esta demanda es altamente variable y depende de múltiples factores como el día de la semana, estaciones del año, condiciones climáticas o eventos puntuales. La falta de modelos precisos de predicción a largo plazo obliga a los gestores a sobredimensionar la capacidad, lo que provoca ineficiencias operativas, desperdicio energético y desgaste innecesario del equipamiento.

En este contexto, la variable **temperatura del aceite del transformador (OT, *Oil Temperature*)** juega un papel clave, ya que actúa como un indicador directo del estado y la carga del transformador. Un incremento excesivo en la temperatura puede señalar condiciones de sobrecarga o riesgo operativo, mientras que valores estables indican un funcionamiento dentro de rangos seguros. Por tanto, modelar y predecir esta variable permite:

* Optimizar la gestión de la carga eléctrica.
* Prevenir fallos en transformadores.
* Reducir costes operativos y mantenimiento.
* Evitar sobreestimaciones en la planificación energética.

```python
# Ruta del dataset
ruta_csv = r'..\Data\ETTh1.csv'
  
# Cargar el dataset
data = pd.read_csv(ruta_csv, sep=',', decimal='.', parse_dates=['date'], index_col='date')
  
# Primera vista de la distribucion y calidad de los datos
resumen_dataset(data, nombre="ETTh1")
```

============================================================ RESUMEN DEL DATASET: ETTh1 ============================================================ INFORMACIÓN GENERAL 
- Dimensiones : 17,420 filas × 7 columnas 
- Rango temporal : 2016-07-01 00:00:00 → 2018-06-26 19:00:00 
- Tipos de datos: float64 7 Name: count, dtype: int64 

PRIMERAS FILAS 
HUFL HULL MUFL MULL LUFL LULL OT date 2016-07-01 00:00:00 5.83 2.01 1.60 0.46 4.20 1.34 30.53 2016-07-01 01:00:00 5.69 2.08 1.49 0.43 4.14 1.37 27.79 2016-07-01 02:00:00 5.16 1.74 1.28 0.35 3.78 1.22 27.79 2016-07-01 03:00:00 5.09 1.94 1.28 0.39 3.81 1.28 25.04 2016-07-01 04:00:00 5.36 1.94 1.49 0.46 3.87 1.28 21.95 

ESTADÍSTICAS DESCRIPTIVAS HUFL HULL MUFL MULL LUFL LULL OT count 17420.00 17420.00 17420.00 17420.00 17420.00 17420.00 17420.00 mean 7.38 2.24 4.30 0.88 3.07 0.86 13.32 std 7.07 2.04 6.83 1.81 1.16 0.60 8.57 min -22.71 -4.76 -25.09 -5.93 -1.19 -1.37 -4.08 25% 5.83 0.74 3.30 -0.28 2.32 0.67 6.96 50% 8.77 2.21 5.97 0.96 2.83 0.98 11.40 75% 11.79 3.68 8.64 2.20 3.62 1.22 18.08 max 23.64 10.11 17.34 7.75 8.50 3.05 46.01 

VALORES NULOS  No hay valores nulos ============================================================

## 1.3. Interpretación inicial del dataset *ETTh1* en contexto energético

### A. Estructura temporal y naturaleza del problema

El dataset presenta **17.420 observaciones** con **7 variables numéricas**, todas en formato continuo (float64). El índice temporal abarca desde julio de 2016 hasta finales de junio de 2018, lo que sugiere una serie temporal de frecuencia horaria (aprox. 2 años completos).

### B. Variables explicativas y variable objetivo

Todas las variables son numéricas continuas, siendo **OT (Oil Temperature)** la variable de interés. El resto de variables (`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`) parecen corresponder a **diferentes cargas o mediciones operativas del transformador** en distintos niveles (alto, medio, bajo).

Una breve descripcion de cada una sería:

| Variable |Descripción                             |
| -------- |--------------------------------------- |
| `date`   |Marca temporal                          |
| `HUFL`   |High Useful Load                        |
| `HULL`   |High Useless Load                       |
| `MUFL`   |Middle Useful Load                      |
| `MULL`   |Middle Useless Load                     |
| `LUFL`   |Low Useful Load                         |
| `LULL`   |Low Useless Load                        |
| `OT`     |**Oil Temperature (variable objetivo)** |

### C. Calidad del dato

La ausencia de valores nulos indica un dataset **completo y listo para modelado**, asi como la presencia de datos normalizados o transformados. Este preprocesado viene indicado en la descripcion de la [documentacion](https://github.com/zhouhaoyi/ETDataset/tree/main) del dataset. 

### D. Variabilidad y comportamiento de las variables

Las variables presentan una elevada variabilidad, con desviaciones estándar en algunos casos comparables a sus medias, lo que refleja las fluctuaciones propias de la demanda energética y la carga del sistema. En particular, la variable *OT* muestra un rango amplio y una dispersión significativa, lo que indica que la temperatura del aceite responde de forma dinámica a los cambios en la carga, incluyendo posibles picos en situaciones de mayor exigencia.

### E. Posibles outliers y eventos extremos

La presencia de valores extremos en varias variables sugiere la existencia de picos de demanda o condiciones operativas fuera de lo habitual. En este contexto, estos valores no deben interpretarse directamente como errores, sino como posibles eventos relevantes que pueden aportar información clave sobre situaciones de estrés en el sistema.

### F. Implicaciones para la predicción de OT

La variabilidad observada sugiere que la predicción de *OT* es un problema complejo, con posibles relaciones no lineales entre variables. Aunque la ausencia de valores nulos simplifica el preprocesado, la dispersión de los datos hace recomendable aplicar técnicas de normalización para facilitar el entrenamiento de modelos de Deep Learning.

### 1.3.1. Distribución de la variable objetivo (OT)

El análisis conjunto de la serie temporal y su distribución permite entender mejor el comportamiento de la temperatura del aceite del transformador (OT). A lo largo del tiempo, la variable muestra un patrón claramente no estacionario, con cambios progresivos en su nivel medio. En los primeros meses del periodo analizado se observan temperaturas elevadas, que posteriormente descienden y dan lugar a una dinámica más estable con oscilaciones moderadas. Este comportamiento sugiere la presencia de una componente estacional relevante, probablemente vinculada a factores externos como las condiciones ambientales o la variabilidad en la demanda energética.

Al observar el detalle a corto plazo, se aprecia que la temperatura no evoluciona de forma suave, sino que presenta fluctuaciones frecuentes junto con algunos picos y caídas abruptas. Estas variaciones reflejan la naturaleza dinámica del sistema eléctrico, donde la carga puede cambiar rápidamente en función del consumo o de condiciones operativas específicas.

Por su parte, la distribución de la variable confirma este comportamiento. La mayor parte de los valores se concentra en un rango intermedio de temperaturas, mientras que los valores más altos aparecen con menor frecuencia, dando lugar a una distribución asimétrica con cola hacia la derecha. Estos valores extremos, aunque poco frecuentes, son especialmente relevantes, ya que pueden estar asociados a situaciones de mayor estrés o carga en el transformador.

![[Pasted image 20260331091009.png]]


### 1.3.2. Distribución de las variables descriptivas

#### A. Comportamiento temporal de las variables

El análisis de las series temporales de las variables descriptivas muestra que todas ellas presentan un comportamiento claramente no estacionario, con cambios en el nivel medio y en la variabilidad a lo largo del tiempo. Se observan patrones cíclicos recurrentes, probablemente asociados a la naturaleza periódica de la demanda eléctrica (ciclos diarios y estacionales).

En particular, variables como HUFL y MUFL presentan una elevada variabilidad con frecuentes oscilaciones y caídas bruscas, lo que sugiere una fuerte sensibilidad a cambios en la carga del sistema. Por otro lado, variables como LUFL y LULL muestran un comportamiento más estable, aunque también presentan cambios de régimen en determinados periodos.

Además, se identifican eventos puntuales abruptos (picos y caídas), que podrían corresponder a situaciones operativas específicas o anomalías en la demanda. Estos eventos son relevantes, ya que pueden influir directamente en la temperatura del transformador.
```python
# Variables descriptivas (excluyendo la variable objetivo)
features = data.drop(columns=["OT"])
  
# SERIES TEMPORALES
fig, axes = plt.subplots(len(features.columns), 1, figsize=(14, 12), sharex=True)
  
for i, col in enumerate(features.columns):
    sns.lineplot(x=features.index, y=features[col], ax=axes[i], linewidth=0.7)
    axes[i].set_title(f"{col}")
    axes[i].set_ylabel(col)
  
plt.suptitle("Series temporales de las variables descriptivas", fontsize=14)
plt.tight_layout()
  
# Guardar figura
plt.savefig(r"..\images\series_temporales_variables_descriptivas.png", dpi=300, bbox_inches="tight")
plt.show()
```

![[Pasted image 20260331091038.png]]

#### B. Distribución de las variables

El análisis de los histogramas revela que las variables descriptivas presentan, en general, distribuciones no perfectamente normales, con distintos grados de asimetría y dispersión.

Variables como HUFL y MUFL muestran distribuciones más dispersas y con colas pronunciadas, lo que indica la presencia de valores extremos y una alta variabilidad en la carga.
Variables como HULL y MULL presentan distribuciones más centradas y relativamente simétricas, lo que sugiere un comportamiento más estable.
En el caso de LUFL y LULL, se observa una mayor concentración de valores en rangos específicos, junto con cierta asimetría, lo que podría reflejar distintos niveles operativos del sistema.

En conjunto, la presencia de colas largas y posibles valores atípicos indica que el sistema experimenta ocasionalmente condiciones extremas, que no deben considerarse errores, sino parte del comportamiento real del proceso.

```python
# DISTRIBUCIONES
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
  
axes = axes.flatten()
  
for i, col in enumerate(features.columns):
    sns.histplot(features[col], bins=40, kde=True, ax=axes[i])
    axes[i].set_title(f"Distribución de {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frecuencia")
  
plt.suptitle("Distribución de las variables descriptivas", fontsize=14)
plt.tight_layout()

# Guardar figura
plt.savefig(r"..\images\histogramas_variables_descriptivas.png", dpi=300, bbox_inches="tight")
  
plt.show()
```

![[Pasted image 20260331091115.png]]

### 1.3.3. Correlación entre variables

Las variables operativas del transformador muestran relaciones coherentes con la naturaleza del sistema eléctrico, destacando **fuertes interdependencias entre variables de carga de un mismo nivel** y **comportamientos más independientes en relación con la temperatura del aceite (OT)**.

* **Correlaciones muy fuertes (|r| > 0.9):**  Las variables de carga en niveles similares están altamente correlacionadas entre sí (**HUFL–MUFL: 0.99**, **HULL–MULL: 0.93**). Esto indica una gran coherencia en la medición de la carga del transformador, donde distintas variables capturan prácticamente el mismo comportamiento operativo. Este resultado sugiere una clara **redundancia de información** dentro de estos grupos.

* **Correlaciones moderadas (0.3 < |r| < 0.7):**  Se observan relaciones moderadas entre variables de distintos niveles de carga, como **HULL–LULL: 0.38** y **LUFL–LULL: 0.34**. Estas correlaciones reflejan que, aunque las cargas en diferentes niveles están relacionadas, no evolucionan de forma completamente conjunta, aportando cierta diversidad en la información del sistema.

* **Correlaciones débiles (0.1 < |r| < 0.3):**  Algunas variables presentan relaciones débiles, como **HUFL–LUFL: 0.29**, **HULL–LUFL: 0.26** o **MUFL–LUFL: 0.18**, lo que sugiere que los distintos niveles de carga mantienen cierta independencia. En relación con la variable objetivo, las correlaciones más altas son **HULL–OT: 0.22** y **MULL–OT: 0.22**, indicando una relación positiva pero limitada entre la carga y la temperatura del aceite.

* **Correlaciones casi nulas (|r| < 0.1):**  La mayoría de las variables presentan correlaciones muy bajas con la variable objetivo (**HUFL–OT: 0.06**, **MUFL–OT: 0.05**, **LULL–OT: 0.07**), lo que indica que no existe una relación lineal directa significativa. Asimismo, algunas relaciones entre variables explicativas también son prácticamente inexistentes, reflejando comportamientos independientes dentro del sistema.

En resumen, las variables de carga muestran una **alta redundancia dentro de cada nivel (high, middle)**, mientras que la variable objetivo (*OT*) presenta **baja correlación lineal con todas ellas**, lo que sugiere que su comportamiento depende de relaciones más complejas. Desde el punto de vista del modelado, esto indica que modelos lineales podrían ser insuficientes, siendo más adecuado el uso de modelos capaces de capturar **dependencias temporales y no lineales**, como las redes neuronales recurrentes.

```python
# Matriz de correlación
correlation_matrix = data.corr()
  
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix,
            annot=True,  
            cmap='coolwarm',  
            center=0,  
            fmt='.2f',  
            square=True,  
            linewidths=0.5,  
            cbar_kws={'label': 'Coeficiente de Correlación'})
  
plt.title('Matriz de Correlación - ETTh1',
          fontsize=14, fontweight='bold', pad=15)
  
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
  
# Guardar si quieres
plt.savefig(r"..\images\correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
```

![[Pasted image 20260331093527.png]]

```python
# Top correlaciones
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
upper_triangle = correlation_matrix.where(mask)
  
correlations_flat = upper_triangle.stack().reset_index()
correlations_flat.columns = ['Variable 1', 'Variable 2', 'Correlación']
correlations_flat['Abs_Correlación'] = correlations_flat['Correlación'].abs()
  
top_correlations = correlations_flat.sort_values(
    'Abs_Correlación', ascending=False
).head(10)
  
print("Top 10 Correlaciones Más Fuertes:\n")
  
for _, row in top_correlations.iterrows():
    tipo = "Positiva" if row['Correlación'] > 0 else "Negativa"
    print(f"{row['Variable 1']:5s} <-> {row['Variable 2']:5s}: "
          f"{row['Correlación']:6.2f} ({tipo})")
```

Top 10 Correlaciones Más Fuertes: 
HUFL <-> MUFL : 0.99 (Positiva) 
HULL <-> MULL : 0.93 (Positiva) 
HULL <-> LULL : 0.38 (Positiva) 
LUFL <-> LULL : 0.33 (Positiva) 
HUFL <-> LUFL : 0.29 (Positiva) 
HULL <-> LUFL : 0.26 (Positiva) 
HULL <-> OT : 0.22 (Positiva) 
MULL <-> OT : 0.22 (Positiva) 
MUFL <-> LUFL : 0.18 (Positiva) 
MULL <-> LUFL : 0.13 (Positiva)

## 1.3. Preprocesado de datos

En esta fase, transformaremos el dataframe original en estructuras que la `SimpleRNN` pueda procesar (tensores de 3 dimensiones: `[muestras, pasos_de_tiempo, características]`).

### 1.3.1. División del dataset (Train, Val, Test)

Dividimos de forma secuencial: 70% entrenamiento, 15% validación y 15% test.

> Hay que dividir ANTES de escalar. Si escalamos con todos los datos juntos, el modelo "ve" información del futuro (data leakage). El scaler se ajusta sólo con el train.


```python
# Tamaños de las particiones
n = len(data)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
  
# División secuencial
train_df = data.iloc[:train_end]
val_df = data.iloc[train_end:val_end]
test_df = data.iloc[val_end:]
  
print(f"Tamaño total: {n}")
print(f"Entrenaiento: {len(train_df)} | Validación: {len(val_df)} | Test: {len(test_df)}")
```

Tamaño total: 17420 
Entrenamiento: 12194 | Validación: 2613 | Test: 2613

### 1.3.2. Escalado de valores
Las redes neuronales recurrentes son muy sensibles a la escala de los datos debido a las funciones de activación (como `tanh`). Utilizaremos `MinMaxScaler` para llevar los valores al rango $[0, 1]$. 

> **Nota:** Ajustamos (`fit`) el escalador solo con los datos de **entrenamiento** y transformamos el resto con esos mismos parámetros.

```python
scaler = MinMaxScaler(feature_range=(0, 1))
  
# Ajustar solo con train y transformar todos
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)
  
# Guardamos el escalador de la variable objetivo (OT) para desescalar después
# OT es la última columna (índice -1)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(train_df[['OT']])
```

### 1.3.3. Función para crear secuencias (Sliding Window)

Esta función recorrerá los datos usando una ventana deslizante. Si `window_size = 24` (un día completo en datos horarios), usaremos las filas de la 0 a la 23 para predecir el valor de la fila 24.

En mi caso voy a definir distintos tamaños de ventana a probar de 1 dia, 2 dias y 1 semana.

> Para garantizar que los resultados sean consistentes cada vez que se ejecute la celda, invocamos las semillas antes de la creación del modelo.

```python
# Tamaños de ventana a probar
# 24h (1 día), 48h (2 días), 168h (1 semana)
WINDOW_SIZES = [24, 48, 168]
  
# Diccionario para guardar los datos de cada ventana
all_sequences = {}
  
# Bucle de preprocesamiento
for ws in WINDOW_SIZES:
    print(f"Generando secuencias para window_size = {ws}...")
    # Creamos las secuencias usando las funciones definidas antes
    X_train_ws, y_train_ws = create_sequences(train_scaled, ws)
    X_val_ws,   y_val_ws   = create_sequences(val_scaled,   ws)
    X_test_ws,  y_test_ws  = create_sequences(test_scaled,  ws)
    # Guardamos todo en el diccionario
    all_sequences[ws] = {
        'X_train': X_train_ws, 'y_train': y_train_ws,
        'X_val': X_val_ws,     'y_val': y_val_ws,
        'X_test': X_test_ws,   'y_test': y_test_ws
    }
  
    print(f" - X_train: {X_train_ws.shape} | y_train: {y_train_ws.shape}")
    print(f" - X_val:   {X_val_ws.shape}   | y_val: {y_val_ws.shape}\n")
```

Generando secuencias para window_size = 24... 
- X_train: (12170, 24, 7) | y_train: (12170,) 
- X_val: (2589, 24, 7) | y_val: (2589,) 

Generando secuencias para window_size = 48... 
- X_train: (12146, 48, 7) | y_train: (12146,) 
- X_val: (2565, 48, 7) | y_val: (2565,) 

Generando secuencias para window_size = 168... 
- X_train: (12026, 168, 7) | y_train: (12026,) 
- X_val: (2445, 168, 7) | y_val: (2445,)

## 1.4. Definición y Construcción del Modelo RNN

Una vez organizados los datos en el diccionario, se pone en marcha el bucle encargado de construir y entrenar los modelos de forma secuencial. Cada configuración se entrena de manera independiente y, para poder analizarlas posteriormente, los modelos resultantes se almacenan en el diccionario `trained_models`, mientras que sus historiales de entrenamiento se guardan en `trained_histories`.

Al revisar el `model.summary()` aparece un detalle especialmente interesante, y es que el número total de parámetros (4.673) es exactamente el mismo para los tres tamaños de ventana. Esto es debido a que en una RNN, los pesos no dependen de la longitud de la secuencia de entrada, sino que se **comparten a lo largo de todos los pasos temporales**. Es decir, la red utiliza la misma 'memoria' para procesar cada instante, independientemente de si la ventana abarca 24, 48 o 168 horas. Por eso, ampliar la ventana no incrementa el número de parámetros.

Si se desglosa el cálculo, se entiende mejor.
* La capa SimpleRNN con 64 unidades concentra la mayor parte de los parámetros, combinando los pesos de entrada (7 × 64), los pesos recurrentes (64 × 64) y los sesgos, lo que suma 4.608 parámetros. 
* A esto se añade la capa densa final, que conecta las 64 salidas de la RNN con una única neurona, aportando 65 parámetros adicionales. 
* El resultado total es, por tanto, 4.673 parámetros, idéntico en todos los modelos al no depender del tamaño de la ventana temporal.


```python
SEED = 42

# Guardar modelos e historiales de entrenamiento
trained_models = {}
trained_histories = {}

# En este caso, 7 variables
n_features = train_scaled.shape[1]

# Bucle de Entrenamiento y Evaluación
for ws in WINDOW_SIZES:
    print(f"\n{'='*60}")
    print(f" CONFIGURANDO MODELO PARA WINDOW_SIZE = {ws}")
    print(f"{'='*60}")

    # Resetear semillas para reproducibilidad
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# --- Definición del Modelo SimpleRNN ---
    model = Sequential([
        Input(shape=(ws, n_features)),
        SimpleRNN(64, activation="tanh"), # 64 unidades para capturar patrones complejos
        Dense(1) # Salida única: la temperatura OT
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
  
    # Mostrar arquitectura
    print(f"\nResumen del modelo (ws={ws}):")
    model.summary()
  
# --- Entrenamiento con EarlyStopping ---
    # Si la val_loss no mejora en 5 épocas, para y recupera el mejor peso
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
  
    history = model.fit(
            all_sequences[ws]['X_train'], all_sequences[ws]['y_train'],
            validation_data=(all_sequences[ws]['X_val'], all_sequences[ws]['y_val']),
            epochs=30,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1
        )
    # Guardamos los objetos para los siguientes lotes
    trained_models[ws] = model
    trained_histories[ws] = history
```

============================================================ CONFIGURANDO MODELO PARA WINDOW_SIZE = 24 ============================================================ Resumen del modelo (ws=24):
Model: "sequential_4"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn_4 (SimpleRNN)        │ (None, 64)             │         4,608 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_4 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 4,673 (18.25 KB)
 Trainable params: 4,673 (18.25 KB)
 Non-trainable params: 0 (0.00 B)

============================================================ CONFIGURANDO MODELO PARA WINDOW_SIZE = 48 ============================================================ Resumen del modelo (ws=48):
Model: "sequential_5"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn_5 (SimpleRNN)        │ (None, 64)             │         4,608 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 4,673 (18.25 KB)
 Trainable params: 4,673 (18.25 KB)
 Non-trainable params: 0 (0.00 B)
 
============================================================ CONFIGURANDO MODELO PARA WINDOW_SIZE = 168 ============================================================ Resumen del modelo (ws=168):
Model: "sequential_6"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn_6 (SimpleRNN)        │ (None, 64)             │         4,608 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 4,673 (18.25 KB)
 Trainable params: 4,673 (18.25 KB)
 Non-trainable params: 0 (0.00 B)

## 1.5. Cálculo de Métricas (RMSE y MAE)

Para evaluar el rendimiento del modelo, se calcularon las métricas de error (RMSE y MAE) en los tres subconjuntos (entrenamiento, validación y test) aplicando previamente la transformación inversa del `target_scaler` para expresar los resultados en grados Celsius, lo que facilita su interpretación directa.

El análisis de los resultados muestra con bastante claridad que la **ventana de 24 horas es la más adecuada**. Este modelo, que utiliza un horizonte temporal equivalente a un día, es el que mejor equilibra ajuste y capacidad de generalización, obteniendo los errores más bajos especialmente en validación y test.

A medida que se amplía la ventana temporal a 48 y, sobre todo, a 168 horas, el rendimiento empeora de forma progresiva. Este comportamiento sugiere que la arquitectura SimpleRNN tiene dificultades para manejar dependencias largas, perdiendo eficacia cuando la secuencia de entrada crece demasiado.

Un aspecto llamativo aparece al comparar los errores entre conjuntos: el RMSE en entrenamiento (1.04) es superior al de test (0.69) en la mejor configuración. Aunque no es lo habitual, este fenómeno puede darse en series temporales cuando el tramo final de los datos (utilizado como test) presenta un comportamiento más estable o menos ruidoso que el inicial, lo que facilita las predicciones.

```python
results_list = []
  
for ws in WINDOW_SIZES:
    model = trained_models[ws]
    data = all_sequences[ws]
    # Diccionario para iterar sobre los conjuntos
    subconjuntos = {
        'Entrenamiento': (data['X_train'], data['y_train']),
        'Validación': (data['X_val'], data['y_val']),
        'Test': (data['X_test'], data['y_test'])
    }

    for nombre, (X, y_real_scaled) in subconjuntos.items():
        # Predicción
        y_pred_scaled = model.predict(X, verbose=0)
        # Desescalado a valores reales (°C)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_real = target_scaler.inverse_transform(y_real_scaled.reshape(-1, 1))
        # Cálculo de métricas
        rmse = root_mean_squared_error(y_real, y_pred)
        mae = mean_absolute_error(y_real, y_pred)
        results_list.append({
            "Window Size": ws,
            "Conjunto": nombre,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })
  
# Generación de la tabla comparativa final
df_resultados = pd.DataFrame(results_list)
print("\nTABLA COMPARATIVA DE RESULTADOS:")
print(df_resultados.to_string(index=False))
```

TABLA COMPARATIVA DE RESULTADOS: 
Window Size Conjunto RMSE MAE 
24 Entrenamiento 1.04 0.73 
24 Validación 0.67 0.50 
24 Test 0.69 0.48 
48 Entrenamiento 1.04 0.74 
48 Validación 0.73 0.56 
48 Test 0.76 0.54 
168 Entrenamiento 1.15 0.85 
168 Validación 0.77 0.58 
168 Test 0.89 0.65

## 1.6. Visualización de Loss y Predicciones

Continuamos centrandonos en dos aspectos fundamentales: 1) las curvas de aprendizaje (*loss*); y 2) la comparación visual entre los valores reales y las predicciones en el conjunto de test.

Dado que el conjunto de test abarca unas 2613 horas, representar todos los puntos en un único gráfico provoca que las líneas (la azul para los valores reales y la roja para las predicciones) se superpongan hasta formar una especie de 'mancha' de color. Esta densidad dificulta distinguir con precisión el grado de acierto del modelo, ya que los errores quedan visualmente diluidos.

En cuanto a las **curvas de *loss***, se observa un comportamiento claro y consistente. Tanto el error de entrenamiento como el de validación descienden con rapidez en las primeras épocas y después se estabilizan. Este patrón indica que el mecanismo de *EarlyStopping* ha actuado correctamente, deteniendo el entrenamiento en el momento adecuado. Además, la convergencia es suave y sin oscilaciones bruscas, lo que sugiere que el ratio de aprendizaje del optimizador Adam está bien ajustado al problema.

Por otro lado, la vista completa de **real vs. predicho** permite evaluar el comportamiento global del modelo. A pesar de la saturación visual, se aprecia que la red logra capturar la tendencia general de la serie. De hecho, la red identifica correctamente los periodos de subida y bajada sostenida de la temperatura. Sin embargo, el modelo tiende a suavizar la señal, comportándose de forma similar a una media móvil, lo que le impide alcanzar con precisión los valores más extremos.

```python
# Bucle de visualización de resultados
for ws in WINDOW_SIZES:
    history = trained_histories[ws]
    model = trained_models[ws]
    data = all_sequences[ws]

    # Preparar datos de Test para la gráfica
    y_test_pred = target_scaler.inverse_transform(model.predict(data['X_test'], verbose=0))
    y_test_real = target_scaler.inverse_transform(data['y_test'].reshape(-1, 1))
  
    plt.figure(figsize=(15, 5))
  
    # --- Gráfica de Loss (MSE) ---
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Curvas de Loss - Ventana {ws}')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
  
    # --- Gráfica Real vs Predicho ---
    plt.subplot(1, 2, 2)
    plt.plot(y_test_real, label='Real (OT)', color='blue', alpha=0.7)
    plt.plot(y_test_pred, label='Predicho (OT)', color='red', linestyle='--')
    plt.title(f'Real vs Predicho (Test) - Ventana {ws}')
    plt.xlabel('Tiempo (Horas)')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
  
    plt.tight_layout()
  
    plt.savefig(f"../images/SimpleRNN_WindowSize_{ws}.png", dpi=300, bbox_inches="tight")
    plt.show()
```

![[Pasted image 20260331121752.png]]
![[Pasted image 20260331121757.png]]
![[Pasted image 20260331121803.png]]

Al acercarnos a las primeras 200 horas (aproximadamente ocho días y medio) la visualización deja de ser una panorámica general y empieza a revelar matices clave del comportamiento del modelo. En este tramo más corto es posible observar con mayor claridad cómo responde la predicción frente a la realidad.

Uno de los aspectos más evidentes es el **retraso (*lag*)**. La línea predicha parece reaccionar con cierto desfase respecto a la real, como si fuera siempre un paso por detrás. Este patrón sugiere que el modelo tiende a apoyarse excesivamente en el valor inmediatamente anterior, asumiendo que el siguiente será muy similar.

También se aprecia mejor la **precisión en los picos**. En lugar de alcanzar los máximos y mínimos reales, el modelo tiende a suavizarlos. Esto indica una dificultad para capturar cambios bruscos.

Por último, este zoom permite identificar claramente el **ciclo diario**. En 200 horas se observan varias oscilaciones completas de subida y bajada, lo que facilita comprobar si el modelo ha aprendido correctamente el patrón día-noche. Aunque logra reproducir la forma general de estas variaciones, los desfases y la pérdida de intensidad en los picos revelan que aún no capta del todo la dinámica completa del sistema.

```python
# Bucle de visualización de resultados
for ws in WINDOW_SIZES:
    history = trained_histories[ws]
    model = trained_models[ws]
    data = all_sequences[ws]

    # Preparar datos de Test para la gráfica
    y_test_pred = target_scaler.inverse_transform(model.predict(data['X_test'], verbose=0))
    y_test_real = target_scaler.inverse_transform(data['y_test'].reshape(-1, 1))
  
    plt.figure(figsize=(15, 5))
  
    # --- Gráfica de Loss (MSE) ---
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Curvas de Loss - Ventana {ws}')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
  
    # --- Gráfica Real vs Predicho (primeras 200 horas del Test)---
    plt.subplot(1, 2, 2)
    plt.plot(y_test_real[:200], label='Real (OT)', color='blue', alpha=0.7)
    plt.plot(y_test_pred[:200], label='Predicho (OT)', color='red', linestyle='--')
    plt.title(f'Real vs Predicho (Test) - Ventana {ws}')
    plt.xlabel('Tiempo (Horas)')
    plt.ylabel('Temperatura (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
  
    plt.tight_layout()
  
    plt.savefig(f"../images/SimpleRNN_WindowSize_{ws}.png", dpi=300, bbox_inches="tight")
    plt.show()
```


![[Pasted image 20260331121901.png]]![[Pasted image 20260331121906.png|697]]
![[Pasted image 20260331121914.png]]

<div style="background-color: #fcf2f2; border-color: #dfb5b4; border-left: 5px solid #dfb5b4; padding: 0.5em;">
<p><strong>Solución:</strong> 

<strong>¿Por qué deben convertirse los datos en ventanas deslizantes?</strong>

Las series temporales son flujos de datos continuos sin etiquetas explícitas. Sin embargo, las redes neuronales requieren una estructura de aprendizaje supervisado para poder entrenarse. En este sentido, la técnica de ventana deslizante (*sliding window*) es esencial porque:
* <strong>Crea pares de entrada-salida (X -> y):</strong> Transforma una secuencia plana en ejemplos discretos donde el pasado reciente actúa como características (*features*) y el valor siguiente como la etiqueta (*target*).
* <strong>Define un tamaño de entrada fijo:</strong> Las arquitecturas como la SimpleRNN necesitan que el vector de entrada tenga una dimensión predefinida para procesar la información.
* <strong>Estructura la dependencia temporal:</strong> Permite que el modelo aprenda de forma sistemática cómo los patrones observados en un intervalo de tiempo específico (por ejemplo, 24 horas) se correlacionan con el resultado inmediato posterior.

<strong>¿Qué limitaciones tiene una SimpleRNN para secuencias largas?</strong>

La principal limitación de la SimpleRNN es el fenómeno conocido como desvanecimiento del gradiente (*vanishing gradient*).

Este problema ocurre por lo siguiente:

* <strong>Propagación del error:</strong> Durante el entrenamiento, el error se propaga hacia atrás en el tiempo a través de cada paso de la secuencia. En ventanas largas (como la de 168h), los gradientes se multiplican repetidamente por los pesos de la red.
* <strong>Pérdida de información:</strong> Si esos pesos son pequeños, el gradiente disminuye exponencialmente hasta volverse insignificante. Como resultado, los pesos de las primeras etapas de la ventana no se actualizan eficazmente.
* <strong>'Olvido' a largo plazo:</strong> En la práctica, esto significa que la red pierde la capacidad de aprender dependencias lejanas. De modo que el modelo solo "recuerda" y reacciona a los eventos más recientes de la secuencia, ignorando patrones importantes que ocurrieron al inicio de la ventana.

Para mitigar esto, se desarrollaron arquitecturas más complejas como **LSTM (Long Short-Term Memory)** y **GRU (Gated Recurrent Unit)**, que utilizan 'puertas' lógicas para decidir qué información preservar y cuál descartar a lo largo del tiempo.
</p>
</div>


---
# 2. Problema de predicción multivariante

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
**Ejercicio [2 pts.].** Entrena modelos recurrentes avanzados para predicción multivariante de series temporales y compara su rendimiento. Para ello:
- Usa el mismo dataset y preprocesamiento que en el ejercicio anterior.

- Implementa y compara los dos modelos siguientes:
    - Un modelo con una capa `LSTM`, una capa `Dropout` y una capa de salida `Dense`.
    - Un modelo con una capa `GRU`, una capa `Dropout` y una capa de salida `Dense`.
    - Compila ambos modelos con loss `mean_squared_error` y optimizador `Adam`.
    - Usa semillas aleatorias para mantener la reproducibilidad.
    - Usa el mismo número de neuronas, tamaño de batch y epochs que en el ejercicio anterior.

- Añade callbacks durante el entrenamiento:
    - EarlyStopping para detener el entrenamiento si no mejora la validación.
    - ReduceLROnPlateau para reducir el learning rate cuando la validación se estanque.

- Entrena ambos modelos y calcula RMSE y MAE en entrenamiento, validación y test.
- Añade los resultados a la tabla comparativa creada en el ejercicio anterior.
- Visualiza las curvas de loss por epoch y los valores reales vs. valores predichos.
- Realiza un breve análisis de la comparación de resultados entre SimpleRNN, LSTM y GRU. Indica si se aprecia overfitting en algún modelo.
</div>

## 2.1. Modelos Recurrentes Avanzados (LSTM y GRU)

Como se ha comentado en el apartado de Solución del ejercicio 1, las redes LSTM y GRU fueron diseñadas para solucionar el problema del desvanecimiento del gradiente que observamos en la SimpleRNN. Utilizan mecanismos de 'puertas' para decidir qué información fluye a través del tiempo, permitiendo capturar dependencias mucho más largas y complejas.

### 2.1.1. Configuración de Callbacks y Parámetros

En este apartado además de ustilizar `EarlyStopping`, añadimos `ReduceLROnPlateau`, que actuará como un 'acelerador inteligente', reduciendo la tasa de aprendizaje si el modelo deja de mejorar para intentar encontrar un mínimo global más preciso.

```python
# Parámetros heredados del Ejercicio 1
WS_BEST = 24  # Usamos la mejor ventana del ejercicio anterior
EPOCHS = 30
BATCH_SIZE = 64
NEURONS = 64
N_FEATURES = train_scaled.shape[1]

# Definición de Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
  
# Recuperamos los datos del diccionario all_sequences
X_train_2 = all_sequences[WS_BEST]['X_train']
y_train_2 = all_sequences[WS_BEST]['y_train']
X_val_2   = all_sequences[WS_BEST]['X_val']
y_val_2   = all_sequences[WS_BEST]['y_val']
X_test_2  = all_sequences[WS_BEST]['X_test']
y_test_2  = all_sequences[WS_BEST]['y_test']
```

## 2.2. Implementación y Entrenamiento de Modelos

Entrenaremos ambos modelos en un bucle para asegurar que las condiciones sean idénticas.

```python
model_types = ['LSTM', 'GRU']
trained_models_adv = {}
trained_histories_adv = {}

for m_type in model_types:
    print(f"\n{'='*60}\n ENTRENANDO MODELO: {m_type}\n{'='*60}")

    # Semillas para reproducibilidad
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

    # Definición de arquitectura según el tipo
    model = Sequential()
    model.add(Input(shape=(WS_BEST, N_FEATURES)))
    if m_type == 'LSTM':
        model.add(LSTM(NEURONS, activation="tanh"))

    else:
        model.add(GRU(NEURONS, activation="tanh"))
    model.add(Dropout(0.2)) # Capa para prevenir overfitting
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    # Entrenamiento
    history = model.fit(
        X_train_2, y_train_2,
        validation_data=(X_val_2, y_val_2),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    trained_models_adv[m_type] = model
    trained_histories_adv[m_type] = history
```

model_types = ['LSTM', 'GRU']
trained_models_adv = {}
trained_histories_adv = {}

for m_type in model_types:
    print(f"\n{'='*60}\n ENTRENANDO MODELO: {m_type}\n{'='*60}")
    
    # Semillas para reproducibilidad
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)
    
    # Definición de arquitectura según el tipo
    model = Sequential()
    model.add(Input(shape=(WS_BEST, N_FEATURES)))
    
    if m_type == 'LSTM':
        model.add(LSTM(NEURONS, activation="tanh"))
    else:
        model.add(GRU(NEURONS, activation="tanh"))
        
    model.add(Dropout(0.2)) # Capa para prevenir overfitting
    model.add(Dense(1))
    
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    
    # Entrenamiento
    history = model.fit(
        X_train_2, y_train_2,
        validation_data=(X_val_2, y_val_2),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    trained_models_adv[m_type] = model
    trained_histories_adv[m_type] = history

## 2.3. Evaluación y Actualización de la Tabla Comparativa
Calculamos las métricas y las añadimos a la tabla que empezaste en el Ejercicio 1 para ver la evolución.

```python
# Lista para las nuevas métricas
adv_results = []

for m_type in model_types:
    model = trained_models_adv[m_type]
    subconjuntos = {
        'Entrenamiento': (X_train_2, y_train_2),
        'Validación': (X_val_2, y_val_2),
        'Test': (X_test_2, y_test_2)
    }

    for nombre, (X, y_real_scaled) in subconjuntos.items():
        y_pred_scaled = model.predict(X, verbose=0)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_real = target_scaler.inverse_transform(y_real_scaled.reshape(-1, 1))
        rmse = root_mean_squared_error(y_real, y_pred)
        mae = mean_absolute_error(y_real, y_pred)
        adv_results.append({
            "Window Size": WS_BEST,
            "Modelo": m_type,
            "Conjunto": nombre,
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })
  
# Unimos con los resultados anteriores (filtramos solo los de WS=24 del Ej1 para comparar)
df_previo = df_resultados[df_resultados['Window Size'] == 24].copy()
df_previo['Modelo'] = 'SimpleRNN'
df_final = pd.concat([df_previo, pd.DataFrame(adv_results)], ignore_index=True)
  
print("\nCOMPARATIVA FINAL DE ARQUITECTURAS (WS=24):")
print(df_final.sort_values(by=['Conjunto', 'RMSE']).to_string(index=False))
```

COMPARATIVA FINAL DE ARQUITECTURAS (WS=24): 
Window Size Conjunto RMSE MAE Modelo 
24 Entrenamiento 1.04 0.73 SimpleRNN 
24 Entrenamiento 1.09 0.77 GRU 
24 Entrenamiento 1.14 0.82 LSTM
24 Test 0.69 0.48 SimpleRNN 
24 Test 0.70 0.48 GRU
24 Test 0.78 0.58 LSTM
24 Validación 0.66 0.49 GRU
24 Validación 0.67 0.50 SimpleRNN 
24 Validación 0.70 0.53 LSTM

## 2.5. Visualización de Resultados
Generamos las gráficas comparativas para LSTM y GRU.

```pyhton
for m_type in model_types:
    history = trained_histories_adv[m_type]
    model = trained_models_adv[m_type]
    y_test_pred = target_scaler.inverse_transform(model.predict(X_test_2, verbose=0))

    y_test_real = target_scaler.inverse_transform(y_test_2.reshape(-1, 1))
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss: {m_type}')
    plt.legend(); plt.grid(alpha=0.3)

    # Zoom 200h
    plt.subplot(1, 2, 2)
    plt.plot(y_test_real, label='Real', color='blue', alpha=0.6)
    plt.plot(y_test_pred, label=f'Predicho {m_type}', color='green', linestyle='--')

    plt.title(f'{m_type}: Real vs Predicho (Zoom 200h)')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../images/LSTM-GRU__{m_type}.png", dpi=300, bbox_inches="tight")
    plt.show()
```

![[Pasted image 20260406135906.png]]
![[Pasted image 20260406135911.png]]


```python
for m_type in model_types:
    history = trained_histories_adv[m_type]
    model = trained_models_adv[m_type]
    y_test_pred = target_scaler.inverse_transform(model.predict(X_test_2, verbose=0))
    y_test_real = target_scaler.inverse_transform(y_test_2.reshape(-1, 1))
    plt.figure(figsize=(15, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'Loss: {m_type}')
    plt.legend(); plt.grid(alpha=0.3)

    # Zoom 200h
    plt.subplot(1, 2, 2)
    plt.plot(y_test_real[:200], label='Real', color='blue', alpha=0.6)
    plt.plot(y_test_pred[:200], label=f'Predicho {m_type}', color='green', linestyle='--')
    plt.title(f'{m_type}: Real vs Predicho (Zoom 200h)')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"../images/LSTM-GRU__{m_type}_subsample200.png", dpi=300, bbox_inches="tight")
    plt.show()
```

![[Pasted image 20260406135952.png]]

![[Pasted image 20260406135957.png]]