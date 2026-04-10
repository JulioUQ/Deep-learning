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
  
# Permite modificar aspectos del entoro de ejecución de Python, como la lista de rutas de búsqueda de módulos (sys.path)
import sys
  
# Sube un nivel desde /PEC/
root_dir = os.path.abspath('..')
sys.path.append(root_dir)
  
# Análisis y manipulación de datos
import pandas as pd
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
  
# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
# Estilo
sns.set(style="whitegrid")
  
# Operaciones matematicas y vectorizacion
import numpy as np
import random

# Librerias para DL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
  
# Para guardar modelos y resultados
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRN, Dense, Input, GRU, LSTM, Dropout, Embedding, Attention, GlobalAveragePooling1D
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
  
# Guardar los modelos entrenados y sus historiales
import pickle
```

# 1. RNN básica

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

He creado dos funciones que se utilizan a lo largo de la PEC:

- La primera, **`resumen_dataset()`**, genera un resumen completo del DataFrame, mostrando su forma, tipos de datos, valores nulos, y estadísticas básicas, además del rango de fechas.
- La segunda, **`create_sequences()`**, Convierte un array de datos en secuencias X (ventanas) e y (objetivo).

```python
def resumen_dataset(df, nombre="Dataset"):
    '''
    Imprime un resumen completo del dataset, incluyendo:
        - Dimensiones
        - Rango temporal
        - Tipos de datos
        - Primeras filas
        - Estadísticas descriptivas
        - Valores nulos (con conteo y porcentaje)
    '''
    n_filas, n_cols = df.shape
    fecha_min, fecha_max = df.index.min(), df.index.max()
  
    print("="*60)
    print(f"RESUMEN DEL DATASET: {nombre}")
    print("="*60)
  
    # Información general
    print("\nINFORMACIÓN GENERAL")
    print(f"- Dimensiones        : {n_filas:,} filas × {n_cols} columnas")
    print(f"- Rango temporal     : {fecha_min} -> {fecha_max}")
  
    # Tipos de datos
    print("\n- Tipos de datos:")
    print(df.dtypes.value_counts())
  
    # Primeras filas
    print("\nPRIMERAS FILAS")
    print(df.head())
  
    # Estadísticas
    print("\nESTADÍSTICAS DESCRIPTIVAS")
    print(df.describe())
  
    # Nulos
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
  
    print("\nVALORES NULOS")
    if len(nulos) == 0:
        print("No hay valores nulos")
    else:
        nulos_pct = (nulos / n_filas * 100).round(2)
        resumen_nulos = pd.DataFrame({
            "Nulos": nulos,
            "%": nulos_pct
        })
        print(resumen_nulos)
  
    print("="*60)

  
def create_sequences(data, window_size):
    """
    Convierte un array de datos en secuencias X (ventanas) e y (objetivo).
  
    Params:
    - data: Array de datos con forma [muestras, features]
    - window_size: Número de pasos de tiempo a incluir en cada secuencia X
  
    Returns:
    X: Secuencia de 'window_size' pasos con todas las variables.
        - Tendrá forma [muestras, window_size, features]
    y: Valor de la variable 'OT' en el paso siguiente.
        - Tendrá forma [muestras,] (predicción del siguiente valor de 'OT')
    """


    X, y = [], []
  
    for i in range(len(data) - window_size):
        # Tomamos todas las variables en la ventana i : i + window_size
        X.append(data[i : (i + window_size), :])
  
        # El objetivo (target) es la columna 'OT' (última posición) del siguiente paso
        y.append(data[i + window_size, -1])
  
    return np.array(X), np.array(y)


def create_sequences_multistep(data, window_size, horizon):
    """

    """
  
    X, y = [], []
  
    for i in range(len(data) - window_size - horizon + 1):
        # Secuencia de entrada (todas las variables)
        X.append(data[i : (i + window_size), :])
  
        # Secuencia de salida: las siguientes 'horizon' horas de la variable 'OT' (índice -1)
        y.append(data[i + window_size : i + window_size + horizon, -1])
  
    return np.array(X), np.array(y)

  
def evaluate_multistep(model, X, y_scaled):
    """
    Función para evaluar multistep desescalando correctamente
    """
    y_pred_scaled = model.predict(X, verbose=0)
  
    # Desescalar matrices [N, 10] con un scaler de [N, 1]
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)
    y_real = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(y_scaled.shape)
  
    # Error promedio por muestra
    mse = mean_squared_error(y_real, y_pred)
    rmse = root_mean_squared_error(y_real, y_pred)
    mae = mean_absolute_error(y_real, y_pred)
  
    return mse, rmse, mae, y_real, y_pred
```

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
data_raw = pd.read_csv(ruta_csv, sep=',', decimal='.', parse_dates=['date'], index_col='date')
  
# Primera vista de la distribucion y calidad de los datos
resumen_dataset(data_raw, nombre="ETTh1")
```

## 1.2. Análisis exploratorio de los datos
### 1.2.1. Interpretación inicial del dataset *ETTh1* en contexto energético

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

### 1.2.2. Distribución de la variable objetivo (OT)

El análisis conjunto de la serie temporal y su distribución permite entender mejor el comportamiento de la temperatura del aceite del transformador (OT). A lo largo del tiempo, la variable muestra un patrón claramente no estacionario, con cambios progresivos en su nivel medio. En los primeros meses del periodo analizado se observan temperaturas elevadas, que posteriormente descienden y dan lugar a una dinámica más estable con oscilaciones moderadas. Este comportamiento sugiere la presencia de una componente estacional relevante, probablemente vinculada a factores externos como las condiciones ambientales o la variabilidad en la demanda energética.

Al observar el detalle a corto plazo, se aprecia que la temperatura no evoluciona de forma suave, sino que presenta fluctuaciones frecuentes junto con algunos picos y caídas abruptas. Estas variaciones reflejan la naturaleza dinámica del sistema eléctrico, donde la carga puede cambiar rápidamente en función del consumo o de condiciones operativas específicas.

Por su parte, la distribución de la variable confirma este comportamiento. La mayor parte de los valores se concentra en un rango intermedio de temperaturas, mientras que los valores más altos aparecen con menor frecuencia, dando lugar a una distribución asimétrica con cola hacia la derecha. Estos valores extremos, aunque poco frecuentes, son especialmente relevantes, ya que pueden estar asociados a situaciones de mayor estrés o carga en el transformador.

```python
# Variable objetivo: temperatura del aceite del transformador (OT = Oil Temperature)
target_col = "OT"
  
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
  
# Serie completa
axes[0].plot(data_raw[target_col], color="steelblue", linewidth=0.7)
axes[0].set_title("Serie completa — Oil Temperature (OT)")
axes[0].set_ylabel("Temperatura (°C)")
axes[2].set_xlabel("Temperatura (°C)")
  
# Zoom en los primeros 500 puntos (≈ 20 días horarios)
axes[1].plot(data_raw[target_col].iloc[:500], color="darkorange", linewidth=0.9)
axes[1].set_title("Zoom — primeros 500 registros")
axes[1].set_ylabel("Temperatura (°C)")
axes[2].set_xlabel("Temperatura (°C)")
  
# Distribución
axes[2].hist(data_raw[target_col], bins=50, color="slategray", edgecolor="white")
axes[2].set_title("Distribución de OT")
axes[2].set_xlabel("Temperatura (°C)")
axes[1].set_ylabel("Temperatura (°C)")
axes[2].set_ylabel("Frecuencia")
  
plt.tight_layout()
  
# Guardar figura
plt.savefig(r"..\images\TS-hist_OT.png", dpi=300, bbox_inches="tight")
plt.show()
```

![[Pasted image 20260331091009.png]]


### 1.2.3. Distribución de las variables descriptivas

#### A. Comportamiento temporal de las variables

El análisis de las series temporales de las variables descriptivas muestra que todas ellas presentan un comportamiento claramente no estacionario, con cambios en el nivel medio y en la variabilidad a lo largo del tiempo. Se observan patrones cíclicos recurrentes, probablemente asociados a la naturaleza periódica de la demanda eléctrica (ciclos diarios y estacionales).

En particular, variables como HUFL y MUFL presentan una elevada variabilidad con frecuentes oscilaciones y caídas bruscas, lo que sugiere una fuerte sensibilidad a cambios en la carga del sistema. Por otro lado, variables como LUFL y LULL muestran un comportamiento más estable, aunque también presentan cambios de régimen en determinados periodos.

Además, se identifican eventos puntuales abruptos (picos y caídas), que podrían corresponder a situaciones operativas específicas o anomalías en la demanda. Estos eventos son relevantes, ya que pueden influir directamente en la temperatura del transformador.
```python
# Variables descriptivas (excluyendo la variable objetivo)
features = data_raw.drop(columns=["OT"])
  
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

### 1.2.4. Correlación entre variables

Las variables operativas del transformador muestran relaciones coherentes con la naturaleza del sistema eléctrico, destacando **fuertes interdependencias entre variables de carga de un mismo nivel** y **comportamientos más independientes en relación con la temperatura del aceite (OT)**.

* **Correlaciones muy fuertes (|r| > 0.9):**  Las variables de carga en niveles similares están altamente correlacionadas entre sí (**HUFL–MUFL: 0.99**, **HULL–MULL: 0.93**). Esto indica una gran coherencia en la medición de la carga del transformador, donde distintas variables capturan prácticamente el mismo comportamiento operativo. Este resultado sugiere una clara **redundancia de información** dentro de estos grupos.

* **Correlaciones moderadas (0.3 < |r| < 0.7):**  Se observan relaciones moderadas entre variables de distintos niveles de carga, como **HULL–LULL: 0.38** y **LUFL–LULL: 0.34**. Estas correlaciones reflejan que, aunque las cargas en diferentes niveles están relacionadas, no evolucionan de forma completamente conjunta, aportando cierta diversidad en la información del sistema.

* **Correlaciones débiles (0.1 < |r| < 0.3):**  Algunas variables presentan relaciones débiles, como **HUFL–LUFL: 0.29**, **HULL–LUFL: 0.26** o **MUFL–LUFL: 0.18**, lo que sugiere que los distintos niveles de carga mantienen cierta independencia. En relación con la variable objetivo, las correlaciones más altas son **HULL–OT: 0.22** y **MULL–OT: 0.22**, indicando una relación positiva pero limitada entre la carga y la temperatura del aceite.

* **Correlaciones casi nulas (|r| < 0.1):**  La mayoría de las variables presentan correlaciones muy bajas con la variable objetivo (**HUFL–OT: 0.06**, **MUFL–OT: 0.05**, **LULL–OT: 0.07**), lo que indica que no existe una relación lineal directa significativa. Asimismo, algunas relaciones entre variables explicativas también son prácticamente inexistentes, reflejando comportamientos independientes dentro del sistema.

En resumen, las variables de carga muestran una **alta redundancia dentro de cada nivel (high, middle)**, mientras que la variable objetivo (*OT*) presenta **baja correlación lineal con todas ellas**, lo que sugiere que su comportamiento depende de relaciones más complejas. Desde el punto de vista del modelado, esto indica que modelos lineales podrían ser insuficientes, siendo más adecuado el uso de modelos capaces de capturar **dependencias temporales y no lineales**, como las redes neuronales recurrentes.

```python
# Matriz de correlación
correlation_matrix = data_raw.corr()
  
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
n = len(data_raw)
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

Esta función (`create_sequences()`) recorrerá los datos usando una ventana deslizante. Es decir, si `window_size = 24` (un día completo en datos horarios), usaremos las filas de la 0 a la 23 para predecir el valor de la fila 24.

En mi caso voy a definir distintos tamaños de ventana a probar de 24h (1 día), 48h (2 días) y 168h (1 semana).

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
# Hiperparámetros
SEED = 42
UNITS = 64 # número de neuronas. Si el modelo sobreentrena lo bajo a 32 o añado una capa de Dropout(0.2)
BATCH = 64  # catidad de datos para actualizar el modelo. Si el entrenamiento va muy lento lo subo
EPOCHS = 200 # pasadas completas por todos los datos de entrenamiento
PATIENCE = 10 # Si se queda en un mínimo local lo subo
N_FEATURES = train_scaled.shape[1] # En este caso, 7 variables
  
# Callback
# Si la val_loss no mejora en 10 épocas, para y recupera el mejor peso
early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
```

```python
# --- CONFIGURACIÓN DE PERSISTENCIA ---
save_dir = r"../SimpleRNN"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
# Guardar modelos e historiales de entrenamiento
trained_models = {}
trained_histories = {}
  
# Bucle de Entrenamiento y Evaluación
for ws in WINDOW_SIZES:
    print(f"\n{'='*60}")
    print(f" CONFIGURANDO MODELO PARA WINDOW_SIZE = {ws}")
    print(f"{'='*60}")
    
    # Resetear semillas para reproducibilidad
    np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)
  
# --- Definición del Modelo SimpleRNN ---
    model = Sequential([
        Input(shape=(ws, N_FEATURES)),
        SimpleRNN(UNITS, activation="tanh"),
        Dense(1) # la temperatura OT
    ])
  
    model.compile(optimizer="adam", loss="mean_squared_error")


    # Mostrar arquitectura
    print(f"\nResumen del modelo (ws={ws}):")
    model.summary()


# --- Entrenamiento con EarlyStopping ---
    # Si la val_loss no mejora en 5 épocas, para y recupera el mejor peso
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
            all_sequences[ws]['X_train'], all_sequences[ws]['y_train'],
            validation_data=(all_sequences[ws]['X_val'], all_sequences[ws]['y_val']),
            epochs=EPOCHS,
            batch_size=BATCH,
            callbacks=[early_stop],
            verbose=1
        )

    # --- GUARDADO DISRUPTIVO ---
    # 1. Guardar el modelo en formato nativo de Keras
    model_filename = f"{save_dir}/modelo_rnn_ws_{ws}.keras"
    model.save(model_filename)
    
    # 2. Guardar el historial (history.history es un dict) para no perder las gráficas
    history_filename = f"{save_dir}/historia_ws_{ws}.pkl"
    with open(history_filename, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"\nModelo y historial guardados para ws={ws} en {save_dir}/")  
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

Epoch 1/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 5ms/step - loss: 0.0059 - val_loss: 5.9672e-04 Epoch 2/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 9.5527e-04 - val_loss: 3.4489e-04 Epoch 3/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 8.0963e-04 - val_loss: 3.0651e-04 Epoch 4/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 7.6111e-04 - val_loss: 2.5464e-04 Epoch 5/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 6.8163e-04 - val_loss: 2.2397e-04 Epoch 6/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 6.2448e-04 - val_loss: 2.0747e-04 Epoch 7/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 5.8225e-04 - val_loss: 2.0099e-04 Epoch 8/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 5.5121e-04 - val_loss: 1.9208e-04 Epoch 9/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 5.2933e-04 - val_loss: 1.7947e-04 Epoch 10/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 5.1414e-04 - val_loss: 1.7673e-04 Epoch 11/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 5.0313e-04 - val_loss: 1.8093e-04 Epoch 12/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.9476e-04 - val_loss: 1.8567e-04 Epoch 13/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.8804e-04 - val_loss: 1.8757e-04 Epoch 14/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.8237e-04 - val_loss: 1.8647e-04 Epoch 15/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.7749e-04 - val_loss: 1.8336e-04 Epoch 16/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.7329e-04 - val_loss: 1.7907e-04 Epoch 17/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.6967e-04 - val_loss: 1.7415e-04 Epoch 18/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.6657e-04 - val_loss: 1.6891e-04 Epoch 19/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.6388e-04 - val_loss: 1.6367e-04 Epoch 20/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.6147e-04 - val_loss: 1.5883e-04 Epoch 21/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 4.5917e-04 - val_loss: 1.5490e-04 Epoch 22/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.5685e-04 - val_loss: 1.5228e-04 Epoch 23/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.5450e-04 - val_loss: 1.5094e-04 Epoch 24/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.5220e-04 - val_loss: 1.5036e-04 Epoch 25/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.5008e-04 - val_loss: 1.4967e-04 Epoch 26/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.4840e-04 - val_loss: 1.4800e-04 Epoch 27/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.4757e-04 - val_loss: 1.4524e-04 Epoch 28/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.4828e-04 - val_loss: 1.4313e-04 Epoch 29/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.5100e-04 - val_loss: 1.4632e-04 Epoch 30/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.5294e-04 - val_loss: 1.7889e-04 Epoch 31/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.4806e-04 - val_loss: 2.2232e-04 Epoch 32/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.4146e-04 - val_loss: 2.2844e-04 Epoch 33/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - loss: 4.3767e-04 - val_loss: 2.2432e-04 Epoch 34/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.3511e-04 - val_loss: 2.1858e-04 Epoch 35/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.3299e-04 - val_loss: 2.1268e-04 Epoch 36/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.3107e-04 - val_loss: 2.0711e-04 Epoch 37/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.2927e-04 - val_loss: 2.0216e-04 Epoch 38/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 3ms/step - loss: 4.2756e-04 - val_loss: 1.9806e-04 Modelo y historial guardados para ws=24 en ..\SimpleRNN/ 

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

Epoch 1/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 2s 7ms/step - loss: 0.0059 - val_loss: 6.3663e-04 Epoch 2/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 9.0940e-04 - val_loss: 3.2464e-04 Epoch 3/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 8.0614e-04 - val_loss: 2.7068e-04 Epoch 4/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 7.5022e-04 - val_loss: 2.3955e-04 Epoch 5/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 6.8637e-04 - val_loss: 2.2428e-04 Epoch 6/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 6.3806e-04 - val_loss: 2.1657e-04 Epoch 7/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 6.0153e-04 - val_loss: 2.1287e-04 Epoch 8/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 5.7342e-04 - val_loss: 2.1147e-04 Epoch 9/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 5.5150e-04 - val_loss: 2.1145e-04 Epoch 10/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 5.3428e-04 - val_loss: 2.1226e-04 Epoch 11/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 5.2066e-04 - val_loss: 2.1357e-04 Epoch 12/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 5.0977e-04 - val_loss: 2.1520e-04 Epoch 13/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 5.0093e-04 - val_loss: 2.1700e-04 Epoch 14/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 4.9366e-04 - val_loss: 2.1887e-04 Epoch 15/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 4.8758e-04 - val_loss: 2.2065e-04 Epoch 16/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 4.8244e-04 - val_loss: 2.2227e-04 Epoch 17/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 4.7806e-04 - val_loss: 2.2369e-04 Epoch 18/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 6ms/step - loss: 4.7429e-04 - val_loss: 2.2493e-04 Epoch 19/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - loss: 4.7101e-04 - val_loss: 2.2609e-04 Modelo y historial guardados para ws=48 en ..\SimpleRNN/ 

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

Epoch 1/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 4s 17ms/step - loss: 0.0059 - val_loss: 4.8463e-04 Epoch 2/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 9.0751e-04 - val_loss: 3.4651e-04 Epoch 3/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 7.6312e-04 - val_loss: 2.8592e-04 Epoch 4/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 6.9113e-04 - val_loss: 2.4577e-04 Epoch 5/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 6.1580e-04 - val_loss: 2.3357e-04 Epoch 6/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 5.8101e-04 - val_loss: 2.6440e-04 Epoch 7/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 5.5851e-04 - val_loss: 2.7217e-04 Epoch 8/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 5.3688e-04 - val_loss: 2.7003e-04 Epoch 9/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 5.1857e-04 - val_loss: 2.6437e-04 Epoch 10/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 5.0353e-04 - val_loss: 2.5767e-04 Epoch 11/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 4.9132e-04 - val_loss: 2.5073e-04 Epoch 12/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.8150e-04 - val_loss: 2.4361e-04 Epoch 13/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 4.7363e-04 - val_loss: 2.3609e-04 Epoch 14/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.6730e-04 - val_loss: 2.2793e-04 Epoch 15/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.6214e-04 - val_loss: 2.1888e-04 Epoch 16/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.5791e-04 - val_loss: 2.0889e-04 Epoch 17/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.5440e-04 - val_loss: 1.9807e-04 Epoch 18/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.5146e-04 - val_loss: 1.8689e-04 Epoch 19/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 4.4892e-04 - val_loss: 1.7613e-04 Epoch 20/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.4660e-04 - val_loss: 1.6685e-04 Epoch 21/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.4433e-04 - val_loss: 1.5990e-04 Epoch 22/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.4198e-04 - val_loss: 1.5539e-04 Epoch 23/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.3955e-04 - val_loss: 1.5285e-04 Epoch 24/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.3709e-04 - val_loss: 1.5156e-04 Epoch 25/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.3463e-04 - val_loss: 1.5100e-04 Epoch 26/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 4.3221e-04 - val_loss: 1.5089e-04 Epoch 27/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.2982e-04 - val_loss: 1.5106e-04 Epoch 28/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.2746e-04 - val_loss: 1.5144e-04 Epoch 29/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.2514e-04 - val_loss: 1.5192e-04 Epoch 30/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.2287e-04 - val_loss: 1.5240e-04 Epoch 31/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 4.2069e-04 - val_loss: 1.5279e-04 Epoch 32/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 4.1864e-04 - val_loss: 1.5304e-04 Epoch 33/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.1681e-04 - val_loss: 1.5323e-04 Epoch 34/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.1618e-04 - val_loss: 1.5336e-04 Epoch 35/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.1417e-04 - val_loss: 1.5473e-04 Epoch 36/200 188/188 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 4.1125e-04 - val_loss: 1.5902e-04 Modelo y historial guardados para ws=168 en ..\SimpleRNN/
## 1.5. Análisis de Resultados y Comportamiento del Modelo

El análisis de las métricas consolidadas confirma que **la ventana de 24 horas constituye una de las configuraciones óptimas para este problema**, proporcionando un excelente equilibrio entre precisión y capacidad de generalización. Con este horizonte temporal, el modelo alcanza en el conjunto de test un **MSE de 0.43**, un **RMSE de 0.66** y un **MAE de 0.46**, lo que indica que la red es capaz de capturar los patrones térmicos diarios con un error inferior a medio grado Celsius. Además, en validación se obtienen los mejores resultados globales (**RMSE de 0.60 y MAE de 0.44**), lo que refuerza la robustez de esta configuración.

No obstante, resulta especialmente relevante observar que la ventana de **168 horas presenta un rendimiento prácticamente equivalente**, e incluso ligeramente superior en el conjunto de test (**MSE de 0.42, RMSE de 0.65 y MAE de 0.45**). Sin embargo, esta mejora es marginal frente al incremento significativo en la longitud de la secuencia de entrada, lo que sugiere que, aunque el modelo es capaz de manejar información de más largo plazo, **el grueso de la capacidad predictiva sigue concentrándose en el corto plazo (24 horas)**.

Por el contrario, la ventana de **48 horas muestra un deterioro claro del rendimiento**, con un **RMSE de 0.76 y un MAE de 0.54 en test**, así como peores resultados en validación (**RMSE de 0.73 y MAE de 0.56**). Este comportamiento sugiere la existencia de una franja intermedia en la que la información adicional no solo no aporta valor, sino que introduce ruido o patrones menos consistentes que dificultan el aprendizaje del modelo.

En el conjunto de entrenamiento, las diferencias entre configuraciones son menos acusadas, con valores muy similares para 24 y 168 horas (**RMSE de 0.97 y MAE de 0.67** en ambos casos), y un ligero empeoramiento para 48 horas (**RMSE de 1.04 y MAE de 0.74**). Esto indica que el modelo es capaz de ajustarse de forma similar durante el aprendizaje, pero las diferencias emergen principalmente en la capacidad de generalización.

Desde un punto de vista técnico, aunque la ventana de 168 horas no degrada el rendimiento, tampoco aporta mejoras significativas, lo que sugiere que la red no está aprovechando de forma efectiva la información más lejana en el tiempo. En consecuencia, **el modelo basa fundamentalmente sus predicciones en dependencias temporales de corto alcance**.

Finalmente, se mantiene el patrón observado de un mejor rendimiento en test que en entrenamiento para las configuraciones óptimas (**RMSE de 0.66 frente a 0.97 en la ventana de 24 horas**). Esto podría indicar que el conjunto de test presenta una dinámica más estable y menos ruidosa que el conjunto de entrenamiento, permitiendo que el modelo generalice con mayor precisión sobre este tramo de datos.

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
        mse = mean_squared_error(y_real, y_pred)
  
        results_list.append({
            "Window Size": ws,
            "Conjunto": nombre,
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
  
        })
  
# Generación de la tabla comparativa final
df_resultados_Ej1 = pd.DataFrame(results_list)
print("="*60)
print("TABLA COMPARATIVA DE RESULTADOS:")
print("="*60)
print(df_resultados_Ej1.sort_values(by=['Conjunto', 'RMSE', 'MAE']).to_string(index=False))
print("="*60)
```

============================================================ TABLA COMPARATIVA DE RESULTADOS: ============================================================ Window Size Conjunto MSE RMSE MAE 
168 Entrenamiento 0.94 0.97 0.67 
24 Entrenamiento 0.95 0.97 0.67 
48 Entrenamiento 1.09 1.04 0.74 
168 Test 0.42 0.65 0.45 
24 Test 0.43 0.66 0.46 
48 Test 0.57 0.76 0.54
24 Validación 0.36 0.60 0.44 
168 Validación 0.38 0.62 0.45 
48 Validación 0.53 0.73 0.56 ============================================================

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

Las redes LSTM y GRU fueron diseñadas para solucionar el problema del desvanecimiento del gradiente que observamos en la SimpleRNN. Utilizan mecanismos de 'puertas' para decidir qué información fluye a través del tiempo, permitiendo capturar dependencias mucho más largas y complejas.

### 2.1.1. Configuración de Callbacks y Parámetros

En este apartado además de utilizar `EarlyStopping`, añadimos `ReduceLROnPlateau`, que actuará como un 'acelerador inteligente', reduciendo la tasa de aprendizaje si el modelo deja de mejorar para intentar encontrar un mínimo global más preciso.
```python
# Parámetros heredados del Ejercicio 1
WS_BEST = 24  # Usamos la mejor ventana del ejercicio anterior
  
# Recuperamos los datos del diccionario all_sequences
X_train_2 = all_sequences[WS_BEST]['X_train']
y_train_2 = all_sequences[WS_BEST]['y_train']
X_val_2   = all_sequences[WS_BEST]['X_val']
y_val_2   = all_sequences[WS_BEST]['y_val']
X_test_2  = all_sequences[WS_BEST]['X_test']
y_test_2  = all_sequences[WS_BEST]['y_test']
```

## 2.2. Definición y Construcción de Modelos LSTM y GRU
 
El bucle de entrenamiento para los modelos avanzados sigue una estructura similar a la utilizada en el primer ejercicio, garantizando que ambos se evalúen bajo las mismas condiciones (mismo tamaño de ventana de 24 horas y mismas semillas aleatorias). Sin embargo, al observar los resúmenes (`model.summary()`) de las nuevas arquitecturas, la diferencia de complejidad estructural frente a la SimpleRNN se hace evidente de inmediato.

El número de parámetros entrenables experimenta un aumento drástico, reflejando el complejo mecanismo interno que utilizan estas redes para combatir el desvanecimiento del gradiente:

* **El Modelo LSTM:** Presenta un total de **18.497 parámetros**. Esta cifra cuadruplica aproximadamente los de la SimpleRNN ($4.673$). La razón técnica es que una celda LSTM no es una simple función de activación, sino que contiene cuatro redes neuronales interactuando internamente (tres "puertas" (olvido, entrada y salida) y un estado de celda candidato). Al desglosarlo, la capa LSTM requiere calcular cuatro conjuntos de pesos para las $7$ variables de entrada y las $64$ unidades ocultas, generando $18.432$ parámetros, a los que se suma la capa densa de salida ($65$).

* **El Modelo GRU:** Resulta ser una versión ligeramente más eficiente, con **14.081 parámetros**. La arquitectura GRU fue diseñada para simplificar la LSTM combinando las puertas de olvido y entrada en una sola "puerta de actualización" y fusionando los estados ocultos. Al tener internamente tres operaciones matemáticas en lugar de cuatro, requiere menos pesos ($14.016$ en la capa recurrente), lo que teóricamente debería traducirse en un entrenamiento más rápido manteniendo un rendimiento competitivo.

> **Nota sobre la Regularización:** En ambos modelos, la inclusión de la capa `Dropout(0.2)` añade $0$ parámetros entrenables, ya que su única función es "apagar" aleatoriamente el $20\%$ de las conexiones durante el entrenamiento. Esta técnica es fundamental en modelos con tanta capacidad (especialmente en la LSTM) para forzar a la red a no memorizar patrones específicos y evitar el sobreajuste.

Finalmente, el comportamiento dinámico del entrenamiento también ha cambiado. Los registros muestran cómo el *callback* `ReduceLROnPlateau` intervino repetidas veces en ambos modelos, reduciendo el *learning rate* a la mitad (hasta llegar a órdenes de magnitud de $10^{-5}$) al detectar estancamientos en el error de validación, permitiendo un ajuste fino ("*fine-tuning*") que el optimizador Adam por defecto no habría logrado por sí solo.

```python
# configuración de persistencia para los nuevos modelos
save_dir_adv = "../LSTM_y_GRU"
if not os.path.exists(save_dir_adv):
    os.makedirs(save_dir_adv)

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
        model.add(LSTM(UNITS, activation="tanh"))
    else:
        model.add(GRU(UNITS, activation="tanh"))
    model.add(Dropout(0.2)) # Capa para prevenir overfitting
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
  
# --- Entrenamiento con Callbacks ---
    # Si la val_loss no mejora en 5 épocas, para y recupera el mejor peso
    # Definición de Callbacks
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
  
    # Entrenamiento
    history = model.fit(
        X_train_2, y_train_2,
        validation_data=(X_val_2, y_val_2),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    trained_models_adv[m_type] = model
    trained_histories_adv[m_type] = history
  
     # --- GUARDADO DISRUPTIVO ---
    model_filename = f"{save_dir_adv}/modelo_{m_type}_ws_{WS_BEST}.keras"
    model.save(model_filename)
  
    history_filename = f"{save_dir_adv}/historia_{m_type}_ws_{WS_BEST}.pkl"
    with open(history_filename, 'wb') as f:
        pickle.dump(history.history, f)
  
    trained_models_adv[m_type] = model
    trained_histories_adv[m_type] = history
```

============================================================ ENTRENANDO MODELO: LSTM ============================================================

Model: "sequential_16"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                     │ (None, 64)             │        18,432 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_15 (Dense)                │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 18,497 (72.25 KB)
 Trainable params: 18,497 (72.25 KB)
 Non-trainable params: 0 (0.00 B)

Epoch 1/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - loss: 0.0049 - val_loss: 6.1717e-04 - learning_rate: 0.0010 Epoch 2/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.0021 - val_loss: 4.8428e-04 - learning_rate: 0.0010 Epoch 3/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.0018 - val_loss: 3.5835e-04 - learning_rate: 0.0010 Epoch 4/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.0016 - val_loss: 3.3052e-04 - learning_rate: 0.0010 Epoch 5/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 0.0015 - val_loss: 3.2143e-04 - learning_rate: 0.0010 Epoch 6/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 10ms/step - loss: 0.0013 - val_loss: 2.3125e-04 - learning_rate: 0.0010 Epoch 7/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0012 - val_loss: 3.4525e-04 - learning_rate: 0.0010 Epoch 8/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.0011 - val_loss: 3.1629e-04 - learning_rate: 0.0010 Epoch 9/200 184/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0011 Epoch 9: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257. 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.0011 - val_loss: 2.8033e-04 - learning_rate: 0.0010 Epoch 10/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 9.9663e-04 - val_loss: 2.1662e-04 - learning_rate: 5.0000e-04 Epoch 11/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 9.6580e-04 - val_loss: 1.9780e-04 - learning_rate: 5.0000e-04 Epoch 12/200 186/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0010 Epoch 12: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 9.5168e-04 - val_loss: 2.0860e-04 - learning_rate: 5.0000e-04 Epoch 13/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 8.7827e-04 - val_loss: 2.0341e-04 - learning_rate: 2.5000e-04 Epoch 14/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 8.7595e-04 - val_loss: 2.4720e-04 - learning_rate: 2.5000e-04 Epoch 15/200 184/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 8.9442e-04 Epoch 15: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 8.6746e-04 - val_loss: 2.0563e-04 - learning_rate: 2.5000e-04 Epoch 16/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 8.5889e-04 - val_loss: 2.1548e-04 - learning_rate: 1.2500e-04 Epoch 17/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 8.3473e-04 - val_loss: 2.1434e-04 - learning_rate: 1.2500e-04 Epoch 18/200 189/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 8.6663e-04 Epoch 18: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05. 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 8.4741e-04 - val_loss: 2.1766e-04 - learning_rate: 1.2500e-04 Epoch 19/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 8.3432e-04 - val_loss: 2.1871e-04 - learning_rate: 6.2500e-05 Epoch 20/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 8.1205e-04 - val_loss: 2.1643e-04 - learning_rate: 6.2500e-05 Epoch 21/200 186/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 8.2072e-04 Epoch 21: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05. 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 7ms/step - loss: 8.1887e-04 - val_loss: 2.2688e-04 - learning_rate: 6.2500e-05

============================================================ ENTRENANDO MODELO: GRU ============================================================

Model: "sequential_17"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ gru (GRU)                       │ (None, 64)             │        14,016 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_16 (Dense)                │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 14,081 (55.00 KB)
 Trainable params: 14,081 (55.00 KB)
 Non-trainable params: 0 (0.00 B)

Epoch 1/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step - loss: 0.0082 - val_loss: 2.9008e-04 - learning_rate: 0.0010 Epoch 2/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0021 - val_loss: 5.2424e-04 - learning_rate: 0.0010 Epoch 3/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - loss: 0.0017 - val_loss: 3.1573e-04 - learning_rate: 0.0010 Epoch 4/200 189/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0016 Epoch 4: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0015 - val_loss: 2.7639e-04 - learning_rate: 0.0010 Epoch 5/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0014 - val_loss: 1.7083e-04 - learning_rate: 5.0000e-04 Epoch 6/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0013 - val_loss: 1.6985e-04 - learning_rate: 5.0000e-04 Epoch 7/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 1s 8ms/step - loss: 0.0013 - val_loss: 2.1131e-04 - learning_rate: 5.0000e-04 Epoch 8/200 184/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0012 Epoch 8: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0012 - val_loss: 1.6893e-04 - learning_rate: 5.0000e-04 Epoch 9/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - loss: 0.0012 - val_loss: 2.3436e-04 - learning_rate: 2.5000e-04 Epoch 10/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0011 - val_loss: 1.6619e-04 - learning_rate: 2.5000e-04 Epoch 11/200 188/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0011 Epoch 11: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0011 - val_loss: 1.6230e-04 - learning_rate: 2.5000e-04 Epoch 12/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - loss: 0.0011 - val_loss: 1.9912e-04 - learning_rate: 1.2500e-04 Epoch 13/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0010 - val_loss: 2.0166e-04 - learning_rate: 1.2500e-04 Epoch 14/200 190/191 ━━━━━━━━━━━━━━━━━━━━ 0s 8ms/step - loss: 0.0010 Epoch 14: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - loss: 0.0010 - val_loss: 1.6844e-04 - learning_rate: 1.2500e-04 Epoch 15/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0010 - val_loss: 1.6965e-04 - learning_rate: 6.2500e-05 Epoch 16/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0010 - val_loss: 2.0280e-04 - learning_rate: 6.2500e-05 Epoch 17/200 185/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0010 Epoch 17: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0010 - val_loss: 2.0060e-04 - learning_rate: 6.2500e-05 Epoch 18/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - loss: 0.0010 - val_loss: 1.6353e-04 - learning_rate: 3.1250e-05 Epoch 19/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0010 - val_loss: 1.9550e-04 - learning_rate: 3.1250e-05 Epoch 20/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 0s 7ms/step - loss: 0.0010 Epoch 20: ReduceLROnPlateau reducing learning rate to 1.5625000742147677e-05. 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 8ms/step - loss: 0.0010 - val_loss: 1.7384e-04 - learning_rate: 3.1250e-05 Epoch 21/200 191/191 ━━━━━━━━━━━━━━━━━━━━ 2s 9ms/step - loss: 9.9686e-04 - val_loss: 1.7793e-04 - learning_rate: 1.5625e-05

## 2.3. Análisis Comparativo: SimpleRNN vs. Modelos Avanzados

ESTE APARTADO QUIERO QUE LO RESUELVAS TU TENIENDO EN CUENTA TODA LA PRACTICA QUE LLEVAMOS REALIZADA.

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
        mse = mean_squared_error(y_real, y_pred)
  
        adv_results.append({
            "Window Size": WS_BEST,
            "Modelo": m_type,
            "Conjunto": nombre,
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })
  
# Unimos con los resultados anteriores (filtramos solo los de WS=24 del Ej1 para comparar)
df_resultados_Ej1_ws24 = df_resultados_Ej1[df_resultados_Ej1['Window Size'] == 24].copy()

df_resultados_Ej1_ws24['Modelo'] = 'SimpleRNN'
df_resultados_Ej2 = pd.concat([df_resultados_Ej1_ws24, pd.DataFrame(adv_results)], ignore_index=True)
  
print("="*60)
print("COMPARATIVA FINAL DE ARQUITECTURAS (WS=24):")
print("="*60)
print(df_resultados_Ej2.sort_values(by=['Conjunto', 'RMSE', 'MAE']).to_string(index=False))
print("="*60)
  
# Guardar en Excel
ruta_excel = r'../LSTM_y_GRU/df_resultados_Ej2.xlsx'
  
with pd.ExcelWriter(ruta_excel, engine='openpyxl', mode='w') as writer:
    df_resultados_Ej2.to_excel(writer, sheet_name='Comparativa_SimpleRNN-LSTM-GRU', index=False)
```

  
============================================================ COMPARATIVA FINAL DE ARQUITECTURAS (WS=24): ============================================================ Window Size Conjunto MSE RMSE MAE Modelo 
24 Entrenamiento 0.95 0.97 0.67 SimpleRNN 
24 Entrenamiento 1.11 1.05 0.75 GRU 
24 Entrenamiento 1.29 1.14 0.82 LSTM 
24 Test 0.43 0.66 0.46 SimpleRNN 
24 Test 0.47 0.68 0.49 GRU 
24 Test 0.61 0.78 0.58 LSTM 
24 Validación 0.36 0.60 0.44 SimpleRNN 
24 Validación 0.41 0.64 0.48 GRU 
24 Validación 0.50 0.70 0.53 LSTM ============================================================
## 2.5. Visualización de Loss y Predicciones (LSTM y GRU)

Al igual que en el análisis de la arquitectura simple, evaluamos el comportamiento de los modelos avanzados a través de sus curvas de aprendizaje y la representación visual de sus predicciones frente a los datos reales.

En las **curvas de _loss_** de ambos modelos, se observa una convergencia estable y sostenida. Un detalle técnico fundamental que diferencia estas gráficas de las del primer ejercicio es que la curva de validación (_Val Loss_) se mantiene consistentemente por debajo de la curva de entrenamiento (_Train Loss_). Este fenómeno es el efecto directo de la capa `Dropout(0.2)`: durante el entrenamiento, a la red se le "apagan" neuronas, lo que eleva artificialmente su error; sin embargo, durante la validación, la red opera con toda su capacidad, obteniendo un mejor rendimiento. La ausencia de repuntes al alza en la validación confirma que no hay _overfitting_ y que la combinación de _EarlyStopping_ y _ReduceLROnPlateau_ ha gestionado el entrenamiento de forma impecable.

En la vista panorámica de **real vs. predicho**, tanto la LSTM como la GRU logran trazar con éxito la tendencia macroscópica de la serie temporal a lo largo del conjunto de test. A pesar de la saturación visual propia de representar más de 2600 horas, se aprecia que los modelos mantienen la estabilidad y no divergen frente a la volatilidad de la serie, capturando fielmente los periodos de calentamiento y enfriamiento prolongados del transformador.

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
    plt.plot(y_test_real, label='Real', color='blue', alpha=0.6)
    plt.plot(y_test_pred, label=f'Predicho {m_type}', color='green', linestyle='--')

    plt.title(f'{m_type}: Real vs Predicho')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()
```


![[Pasted image 20260406135906.png]]
![[Pasted image 20260406135911.png]]

Al aplicar el **zoom a las primeras 200 horas**, los matices del comportamiento predictivo a corto plazo salen a la luz. A pesar de la enorme complejidad matemática de las puertas lógicas que incorporan la LSTM y la GRU, el **retraso (_lag_)** sigue estando visible. Los modelos continúan exhibiendo cierta dependencia de la "predicción ingenua" (asumir que el valor siguiente será igual al actual), reaccionando a los cambios bruscos un paso por detrás en lugar de anticiparlos. Esto se hace especialmente evidente en la profunda y repentina caída de temperatura que se registra en torno a la hora 115, donde la predicción persigue al desplome real con un ligero desfase.

Finalmente, en cuanto a la **precisión en los picos** y el ciclo diario, ambos modelos avanzados siguen mostrando la tendencia a suavizar los valores extremos, funcionando como un filtro de la señal original. No obstante, al comparar minuciosamente el encaje de ambas gráficas, la línea de predicción de la **GRU** parece ceñirse de forma sutilmente más estrecha a los recovecos de la curva real que la de la LSTM. Esta confirmación visual cierra el círculo del análisis, respaldando la conclusión numérica obtenida en la tabla de métricas: para esta ventana de 24 horas, la arquitectura GRU resulta más eficiente y ajustada que la LSTM.

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

    plt.show()
```

![[Pasted image 20260406135952.png]]

![[Pasted image 20260406135957.png]]

En conclusión, la comparativa demuestra el principio de parsimonia en Deep Learning. Cuando las dependencias temporales son a corto plazo (un ciclo diario), introducir arquitecturas complejas (LSTM) junto con fuerte regularización puede degradar el rendimiento. La elección de la arquitectura debe estar guiada por la ventana de contexto necesaria, siendo la SimpleRNN o la GRU las opciones más eficientes para este pronóstico a corto plazo.

**Pregunta: Realiza un breve análisis de la comparación de resultados entre SimpleRNN, LSTM y GRU. Indica si se aprecia overfitting en algún modelo.**

**Solución: El analisis se ha realizado junto a los resultados obtenidos en cada apartado**

---
---
# 3. LSTM con PyTorch

En este ejercicio implementarás un modelo LSTM para predicción de series temporales usando **PyTorch**.

Hasta ahora los modelos recurrentes se han implementado con TensorFlow/Keras, donde el entrenamiento se realiza mediante `model.fit()`. En PyTorch, en cambio, el entrenamiento se controla de forma explícita mediante un **bucle de entrenamiento manual**, lo que ofrece mayor flexibilidad.

Dado que PyTorch no se ha trabajado todavía en profundidad en la asignatura, el ejercicio se plantea de forma guiada, proporcionando la estructura básica del código que deberás completar, ejecutar y analizar.

**IMPORTANTE: Es recomendable realizar este ejercicio para preparar las próximas PEC, que serán en PyTorch. Si decides no hacerlo, puedes continuar con el resto de ejercicios, pero tendrás que realizar el Ejecicio 6 de búsqueda de hiperparámetros con Optuna para poder conseguir un 10 en la PEC.**

---
# 3. LSTM con PyTorch

**Ejercicio [1 pts.].** Continúa el ejercicio anterior, pero ahora implementa los modelos recurrentes usando PyTorch en lugar de TensorFlow/Keras. Para ello:

**Paso 1. Preparación de datos**

Usa el mismo dataset (ETTh1) y preprocesamiento que en los ejercicios anteriores. Reutiliza las secuencias de entrada, particiones de entrenamiento/validación/test y normalización ya definidas.

**Paso 2. Conversión de datos a tensores de PyTorch**

PyTorch trabaja con objetos llamados tensores (`torch.Tensor`).Convierte los datos de entrenamiento, validación y test a tensores usando `torch.tensor(..., dtype=torch.float32)`.

**Paso 3. Creación de TensorDataset y DataLoader**

En PyTorch los datos suelen gestionarse mediante dos estructuras:

- `TensorDataset`. Agrupa los tensores de entrada y salida. `TensorDataset(X, y)`.
- `DataLoader`. Permite: iterar por batches, barajar los datos y alimentar el modelo durante el entrenamiento. Debes crear tres `DataLoader`:
    - entrenamiento (`shuffle=True`)
    - validación (`shuffle=False`)
    - test (`shuffle=False`)

Usa el mismo **tamaño de batch** que en los ejercicios anteriores.

**Paso 4. Implementación del modelo LSTM**

En PyTorch los modelos se implementan creando una clase que hereda de `torch.nn.Module`. Debes definir un modelo con la siguiente arquitectura: - Una capa `LSTM`. - Una capa `Dropout`. - Una capa de salida `Linear` para predicción de la serie temporal.

**Constructor `__init__`**

En el constructor debes definir las capas del modelo:

- `nn.LSTM`
- `nn.Dropout`
- `nn.Linear`

El número de neuronas ocultas debe ser el mismo usado en los ejercicios anteriores

**Método `forward`**

El método `forward()` define cómo fluye la información a través del modelo.

Pasos que debes implementar:

1. Pasar la secuencia de entrada `x` por la capa LSTM
2. Seleccionar la salida correspondiente al último instante temporal
3. Aplicar dropout
4. Pasar el resultado por la capa lineal para obtener la predicción final

**Paso 5. Configuración del entrenamiento**

Define los elementos necesarios para entrenar el modelo:

- **Función de pérdida**. Usa el MSE: `criterion = nn.MSELoss()`
- **Optimizador**. Usa el optimizador **Adam**: `optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)`

**Paso 6. Implementación del bucle de entrenamiento manual**

A diferencia de Keras, en PyTorch el entrenamiento debe implementarse manualmente.

Para cada epoch:

**Paso 6.1. Fase de entrenamiento**

Primero pon el modelo en modo entrenamiento: `model.train()`. Después itera sobre los batches del `DataLoader` de entrenamiento (creado en el paso 3).

Para cada batch:

1. Mueve los datos al dispositivo correspondiente (`cpu` o `gpu`).
2. Reinicia los gradientes `optimizer.zero_grad()`.
3. Realiza la propagación hacia adelante `pred = model(xb)`.
4. Calcula la pérdida `loss = criterion(pred, yb)`.
5. Realiza la backpropagation `loss.backward()`.
6. Actualiza los pesos `optimizer.step()`.

Acumula la pérdida media de entrenamiento de cada epoch.

**Paso 6.2. Fase de validación**

Después de cada epoch evalúa el modelo en el conjunto de validación.

1. Cambia el modelo a modo evaluación `model.eval()`.
2. Desactiva el cálculo de gradientes `with torch.no_grad():`.
3. Itera sobre el `DataLoader` de validación y calcula la pérdida.

En esta fase **NO** se debe:

- hacer `backward()`
- actualizar los pesos

**Paso 7. Evaluación del modelo**

Una vez entrenado el modelo, evalúalo en entrenamiento, validación y test

**Paso 8. Comparación de resultados**

Calcula RMSE y MAE en entrenamiento, validación y test. Incluye los nuevos resultados en la tabla comparativa y analiza los resultados obtenidos.

**Solución:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Variables heredadas (Ver variables en los Ejercicios 1 y 2)
# Renombre de UNITS para ser consistente con el codigo facilitado por el profesor
N_UNITS =  UNITS
  
# Recuperamos los datos del diccionario all_sequences
X_train_3 = all_sequences[WS_BEST]['X_train']
y_train_3 = all_sequences[WS_BEST]['y_train']
X_val_3   = all_sequences[WS_BEST]['X_val']
y_val_3   = all_sequences[WS_BEST]['y_val']
X_test_3  = all_sequences[WS_BEST]['X_test']
y_test_3  = all_sequences[WS_BEST]['y_test']

# Conversión de datos a tensores
# Uso .reshape(-1, 1) en las 'y' para que tengan forma [batch, 1] y no [batch]
X_train_t = torch.tensor(X_train_3, dtype=torch.float32)
y_train_t = torch.tensor(y_train_3, dtype=torch.float32).reshape(-1, 1)
  
X_val_t = torch.tensor(X_val_3, dtype=torch.float32)
y_val_t = torch.tensor(y_val_3, dtype=torch.float32).reshape(-1, 1)
  
X_test_t = torch.tensor(X_test_3, dtype=torch.float32)
y_test_t = torch.tensor(y_test_3, dtype=torch.float32).reshape(-1, 1)
  
# Creación del TensorDataset
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
  
train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

# Implementación del Modelo LSTM en PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=N_UNITS, dropout=DROPOUT):
        super().__init__()
        # batch_first=True es crucial porque los tensores son [batch, seq, features
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)    # Capa LSTM
        self.dropout = nn.Dropout(dropout)                                                       # Capa Dropout
        self.fc = nn.Linear(hidden_size, 1)                                                      # Capa Linear


    def forward(self, x):
        out, _ = self.lstm(x)     # Pasar la secuencia de entrada `x` por la capa LSTM
        out = out[:, -1, :]       # Seleccionar la salida correspondiente al último instante temporal
        out = self.dropout(out)   # Aplicar dropout
        out = self.fc(out)        # Pasar el resultado por la capa lineal para obtener la predicción final
        return out
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Entrenamiento
def train_model(model):
    model.to(device)                                          # Mueve los datos al dispositivo correspondiente (`cpu` o `gpu`).
    criterion = nn.MSELoss()                                  #  Función de pérdida
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #  Optimizador

    for epoch in range(EPOCHS):
        # ENTRENAMIENTO
        model.train()                               # Pon el modelo en modo entrenamiento
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb =  xb.to(device), yb.to(device)  # Mueve los datos al dispositivo correspondiente (`cpu` o `gpu`)
            optimizer.zero_grad()                   # Reinicia los gradientes
            pred = model(xb)                        # Realiza la propagación hacia adelante
            loss = criterion(pred, yb)              # Calcula la pérdida
            loss.backward()                         # Realiza la backpropagation
            optimizer.step()                        # Actualiza los pesos
            train_loss += loss.item()               # Acumula la pérdida del batch actual (loss.item() convierte el tensor de PyTorch a un valor numérico de Python)
        train_loss /= len(train_loader)             # Calcula la pérdida media de entrenamiento dividiendo entre el número total de batches
        
        # VALIDACIÓN
        model.eval()                                   # Cambia el modelo a modo evaluación

        val_loss = 0

        with torch.no_grad():                          # Desactiva el cálculo de gradientes
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)  # Mueve los datos al dispositivo correspondiente (`cpu` o `gpu`)
                pred = model(xb)                       # Realiza la propagación hacia adelante
                loss = criterion(pred, yb)             # Calcula la pérdida
                val_loss += loss.item()                # Acumula la pérdida del batch actual durante la fase de validación
        val_loss /= len(val_loader)                    # Calcula la pérdida media de validación dividiendo entre el número total de batches

        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.5f} - val_loss: {val_loss:.5f}")

# Evaluación del modelo
def evaluate_model(model, loader):
    model.eval()                                # Pon el modelo en modo evaluación
    preds = []
    trues = []

    with torch.no_grad():                       # Desactiva el cálculo de gradientes para ahorrar memoria y acelerar la evaluación
        for xb, yb in loader:                   # Itera sobre los batches del DataLoader
            xb = xb.to(device)                  # Mueve los datos al dispositivo correspondiente
            pred = model(xb).cpu().numpy()      # Realiza la predicción del modelo y la convierte de tensor de PyTorch a array de NumPy
            preds.append(pred)                  # Guarda las predicciones del batch en la lista
            trues.append(yb.numpy())            # Guarda los valores reales del batch en la lista
  
    preds = np.vstack(preds)                    # Une verticalmente todas las predicciones de los batches en un único array
    trues = np.vstack(trues)                   # Une verticalmente todos los valores reales en un único array
  
    # Desescalamos antes de medir el error para poder comparar con Keras
    preds_real = target_scaler.inverse_transform(preds)
    trues_real = target_scaler.inverse_transform(trues)
  
    # Cálculo de errores (incluido el MSE para tener las 3 métricas)
    mse = mean_squared_error(trues_real, preds_real)
    rmse = root_mean_squared_error(trues_real, preds_real)
    mae = mean_absolute_error(trues_real, preds_real)
  
    return mse, rmse, mae        
    
    
# Definición de la LSTM
lstm_model = LSTMModel(input_size=N_FEATURES)

# Entrenamiento del modelo
train_model(lstm_model)

# Evaluación del modelo
lstm_train_mse, lstm_train_rmse, lstm_train_mae = evaluate_model(lstm_model, train_loader)
lstm_val_mse, lstm_val_rmse, lstm_val_mae = evaluate_model(lstm_model, val_loader)
lstm_test_mse, lstm_test_rmse, lstm_test_mae = evaluate_model(lstm_model, test_loader)

# Creamos una lista con los resultados específicos del modelo de PyTorch
resultados_pytorch = [
    {"Window Size": 24, "Conjunto": "Entrenamiento", "MSE": round(lstm_train_mse, 4), "RMSE": round(lstm_train_rmse, 4), "MAE": round(lstm_train_mae, 4), "Modelo": "LSTM (PyTorch)"},
    {"Window Size": 24, "Conjunto": "Validación", "MSE": round(lstm_val_mse, 4), "RMSE": round(lstm_val_rmse, 4), "MAE": round(lstm_val_mae, 4), "Modelo": "LSTM (PyTorch)"},
    {"Window Size": 24, "Conjunto": "Test", "MSE": round(lstm_test_mse, 4), "RMSE": round(lstm_test_rmse, 4), "MAE": round(lstm_test_mae, 4), "Modelo": "LSTM (PyTorch)"}
]
 
# Lo convertimos a DataFrame temporal
df_pytorch = pd.DataFrame(resultados_pytorch)
  
# Cocatenamos tu df_final (Keras) con el nuevo df_pytorch para crear el dataframe final
df_resultados_Ej3 = pd.concat([df_resultados_Ej2, df_pytorch], ignore_index=True)
  
print("="*60)
print("COMPARATIVA FINAL DE ARQUITECTURAS (WS=24) - KERAS VS PYTORCH:")
print("="*60)
print(df_resultados_Ej3.sort_values(by=['Conjunto', 'RMSE', 'MAE']).to_string(index=False))
print("="*60)
```

Epoch 1/200 - train_loss: 0.02415 - val_loss: 0.00380 Epoch 2/200 - train_loss: 0.00313 - val_loss: 0.00154 Epoch 3/200 - train_loss: 0.00257 - val_loss: 0.00079 Epoch 4/200 - train_loss: 0.00222 - val_loss: 0.00142 Epoch 5/200 - train_loss: 0.00197 - val_loss: 0.00109 Epoch 6/200 - train_loss: 0.00178 - val_loss: 0.00057 Epoch 7/200 - train_loss: 0.00169 - val_loss: 0.00047 Epoch 8/200 - train_loss: 0.00163 - val_loss: 0.00060 Epoch 9/200 - train_loss: 0.00153 - val_loss: 0.00028 Epoch 10/200 - train_loss: 0.00146 - val_loss: 0.00031 Epoch 11/200 - train_loss: 0.00135 - val_loss: 0.00026 Epoch 12/200 - train_loss: 0.00129 - val_loss: 0.00024 Epoch 13/200 - train_loss: 0.00123 - val_loss: 0.00021 Epoch 14/200 - train_loss: 0.00115 - val_loss: 0.00031 Epoch 15/200 - train_loss: 0.00111 - val_loss: 0.00035 Epoch 16/200 - train_loss: 0.00101 - val_loss: 0.00018 Epoch 17/200 - train_loss: 0.00099 - val_loss: 0.00025 Epoch 18/200 - train_loss: 0.00097 - val_loss: 0.00070 Epoch 19/200 - train_loss: 0.00090 - val_loss: 0.00018 Epoch 20/200 - train_loss: 0.00085 - val_loss: 0.00018 Epoch 21/200 - train_loss: 0.00082 - val_loss: 0.00018 Epoch 22/200 - train_loss: 0.00079 - val_loss: 0.00026 Epoch 23/200 - train_loss: 0.00076 - val_loss: 0.00021 Epoch 24/200 - train_loss: 0.00073 - val_loss: 0.00018 Epoch 25/200 - train_loss: 0.00073 - val_loss: 0.00021 Epoch 26/200 - train_loss: 0.00068 - val_loss: 0.00015 Epoch 27/200 - train_loss: 0.00066 - val_loss: 0.00024 Epoch 28/200 - train_loss: 0.00065 - val_loss: 0.00016 Epoch 29/200 - train_loss: 0.00062 - val_loss: 0.00015 Epoch 30/200 - train_loss: 0.00063 - val_loss: 0.00015 Epoch 31/200 - train_loss: 0.00062 - val_loss: 0.00039 Epoch 32/200 - train_loss: 0.00061 - val_loss: 0.00022 Epoch 33/200 - train_loss: 0.00057 - val_loss: 0.00016 Epoch 34/200 - train_loss: 0.00060 - val_loss: 0.00015 Epoch 35/200 - train_loss: 0.00060 - val_loss: 0.00019 Epoch 36/200 - train_loss: 0.00059 - val_loss: 0.00038 Epoch 37/200 - train_loss: 0.00056 - val_loss: 0.00031 Epoch 38/200 - train_loss: 0.00056 - val_loss: 0.00019 Epoch 39/200 - train_loss: 0.00057 - val_loss: 0.00014 Epoch 40/200 - train_loss: 0.00056 - val_loss: 0.00014 Epoch 41/200 - train_loss: 0.00057 - val_loss: 0.00015 Epoch 42/200 - train_loss: 0.00057 - val_loss: 0.00020 Epoch 43/200 - train_loss: 0.00055 - val_loss: 0.00016 Epoch 44/200 - train_loss: 0.00055 - val_loss: 0.00018 Epoch 45/200 - train_loss: 0.00053 - val_loss: 0.00014 Epoch 46/200 - train_loss: 0.00054 - val_loss: 0.00017 Epoch 47/200 - train_loss: 0.00055 - val_loss: 0.00034 Epoch 48/200 - train_loss: 0.00055 - val_loss: 0.00015 Epoch 49/200 - train_loss: 0.00056 - val_loss: 0.00016 Epoch 50/200 - train_loss: 0.00055 - val_loss: 0.00031 Epoch 51/200 - train_loss: 0.00054 - val_loss: 0.00038 Epoch 52/200 - train_loss: 0.00054 - val_loss: 0.00033 Epoch 53/200 - train_loss: 0.00055 - val_loss: 0.00015 Epoch 54/200 - train_loss: 0.00055 - val_loss: 0.00014 Epoch 55/200 - train_loss: 0.00055 - val_loss: 0.00014 Epoch 56/200 - train_loss: 0.00052 - val_loss: 0.00021 Epoch 57/200 - train_loss: 0.00055 - val_loss: 0.00055 Epoch 58/200 - train_loss: 0.00053 - val_loss: 0.00024 Epoch 59/200 - train_loss: 0.00053 - val_loss: 0.00017 Epoch 60/200 - train_loss: 0.00052 - val_loss: 0.00014 Epoch 61/200 - train_loss: 0.00053 - val_loss: 0.00022 Epoch 62/200 - train_loss: 0.00054 - val_loss: 0.00027 Epoch 63/200 - train_loss: 0.00054 - val_loss: 0.00024 Epoch 64/200 - train_loss: 0.00053 - val_loss: 0.00068 Epoch 65/200 - train_loss: 0.00055 - val_loss: 0.00016 Epoch 66/200 - train_loss: 0.00054 - val_loss: 0.00019 Epoch 67/200 - train_loss: 0.00053 - val_loss: 0.00026 Epoch 68/200 - train_loss: 0.00054 - val_loss: 0.00024 Epoch 69/200 - train_loss: 0.00053 - val_loss: 0.00021 Epoch 70/200 - train_loss: 0.00053 - val_loss: 0.00022 Epoch 71/200 - train_loss: 0.00053 - val_loss: 0.00026 Epoch 72/200 - train_loss: 0.00052 - val_loss: 0.00016 Epoch 73/200 - train_loss: 0.00054 - val_loss: 0.00023 Epoch 74/200 - train_loss: 0.00053 - val_loss: 0.00015 Epoch 75/200 - train_loss: 0.00052 - val_loss: 0.00030 Epoch 76/200 - train_loss: 0.00052 - val_loss: 0.00036 Epoch 77/200 - train_loss: 0.00053 - val_loss: 0.00015 Epoch 78/200 - train_loss: 0.00053 - val_loss: 0.00035 Epoch 79/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 80/200 - train_loss: 0.00053 - val_loss: 0.00014 Epoch 81/200 - train_loss: 0.00052 - val_loss: 0.00014 Epoch 82/200 - train_loss: 0.00052 - val_loss: 0.00017 Epoch 83/200 - train_loss: 0.00053 - val_loss: 0.00016 Epoch 84/200 - train_loss: 0.00053 - val_loss: 0.00031 Epoch 85/200 - train_loss: 0.00053 - val_loss: 0.00018 Epoch 86/200 - train_loss: 0.00052 - val_loss: 0.00016 Epoch 87/200 - train_loss: 0.00052 - val_loss: 0.00019 Epoch 88/200 - train_loss: 0.00053 - val_loss: 0.00024 Epoch 89/200 - train_loss: 0.00053 - val_loss: 0.00073 Epoch 90/200 - train_loss: 0.00052 - val_loss: 0.00022 Epoch 91/200 - train_loss: 0.00052 - val_loss: 0.00014 Epoch 92/200 - train_loss: 0.00052 - val_loss: 0.00026 Epoch 93/200 - train_loss: 0.00053 - val_loss: 0.00016 Epoch 94/200 - train_loss: 0.00052 - val_loss: 0.00020 Epoch 95/200 - train_loss: 0.00052 - val_loss: 0.00023 Epoch 96/200 - train_loss: 0.00053 - val_loss: 0.00015 Epoch 97/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 98/200 - train_loss: 0.00053 - val_loss: 0.00017 Epoch 99/200 - train_loss: 0.00052 - val_loss: 0.00016 Epoch 100/200 - train_loss: 0.00052 - val_loss: 0.00020 Epoch 101/200 - train_loss: 0.00051 - val_loss: 0.00017 Epoch 102/200 - train_loss: 0.00051 - val_loss: 0.00019 Epoch 103/200 - train_loss: 0.00052 - val_loss: 0.00014 Epoch 104/200 - train_loss: 0.00052 - val_loss: 0.00044 Epoch 105/200 - train_loss: 0.00053 - val_loss: 0.00056 Epoch 106/200 - train_loss: 0.00051 - val_loss: 0.00013 Epoch 107/200 - train_loss: 0.00052 - val_loss: 0.00034 Epoch 108/200 - train_loss: 0.00052 - val_loss: 0.00018 Epoch 109/200 - train_loss: 0.00051 - val_loss: 0.00037 Epoch 110/200 - train_loss: 0.00052 - val_loss: 0.00029 Epoch 111/200 - train_loss: 0.00051 - val_loss: 0.00018 Epoch 112/200 - train_loss: 0.00050 - val_loss: 0.00019 Epoch 113/200 - train_loss: 0.00051 - val_loss: 0.00016 Epoch 114/200 - train_loss: 0.00051 - val_loss: 0.00047 Epoch 115/200 - train_loss: 0.00052 - val_loss: 0.00016 Epoch 116/200 - train_loss: 0.00051 - val_loss: 0.00018 Epoch 117/200 - train_loss: 0.00051 - val_loss: 0.00031 Epoch 118/200 - train_loss: 0.00052 - val_loss: 0.00017 Epoch 119/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 120/200 - train_loss: 0.00052 - val_loss: 0.00019 Epoch 121/200 - train_loss: 0.00053 - val_loss: 0.00015 Epoch 122/200 - train_loss: 0.00051 - val_loss: 0.00035 Epoch 123/200 - train_loss: 0.00051 - val_loss: 0.00024 Epoch 124/200 - train_loss: 0.00051 - val_loss: 0.00023 Epoch 125/200 - train_loss: 0.00051 - val_loss: 0.00030 Epoch 126/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 127/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 128/200 - train_loss: 0.00052 - val_loss: 0.00032 Epoch 129/200 - train_loss: 0.00052 - val_loss: 0.00015 Epoch 130/200 - train_loss: 0.00051 - val_loss: 0.00022 Epoch 131/200 - train_loss: 0.00051 - val_loss: 0.00021 Epoch 132/200 - train_loss: 0.00053 - val_loss: 0.00015 Epoch 133/200 - train_loss: 0.00052 - val_loss: 0.00019 Epoch 134/200 - train_loss: 0.00051 - val_loss: 0.00021 Epoch 135/200 - train_loss: 0.00051 - val_loss: 0.00017 Epoch 136/200 - train_loss: 0.00052 - val_loss: 0.00017 Epoch 137/200 - train_loss: 0.00051 - val_loss: 0.00024 Epoch 138/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 139/200 - train_loss: 0.00051 - val_loss: 0.00019 Epoch 140/200 - train_loss: 0.00052 - val_loss: 0.00014 Epoch 141/200 - train_loss: 0.00052 - val_loss: 0.00027 Epoch 142/200 - train_loss: 0.00050 - val_loss: 0.00024 Epoch 143/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 144/200 - train_loss: 0.00051 - val_loss: 0.00028 Epoch 145/200 - train_loss: 0.00051 - val_loss: 0.00016 Epoch 146/200 - train_loss: 0.00052 - val_loss: 0.00014 Epoch 147/200 - train_loss: 0.00050 - val_loss: 0.00017 Epoch 148/200 - train_loss: 0.00050 - val_loss: 0.00021 Epoch 149/200 - train_loss: 0.00051 - val_loss: 0.00016 Epoch 150/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 151/200 - train_loss: 0.00051 - val_loss: 0.00034 Epoch 152/200 - train_loss: 0.00051 - val_loss: 0.00016 Epoch 153/200 - train_loss: 0.00050 - val_loss: 0.00015 Epoch 154/200 - train_loss: 0.00051 - val_loss: 0.00017 Epoch 155/200 - train_loss: 0.00051 - val_loss: 0.00015 Epoch 156/200 - train_loss: 0.00051 - val_loss: 0.00015 Epoch 157/200 - train_loss: 0.00051 - val_loss: 0.00045 Epoch 158/200 - train_loss: 0.00051 - val_loss: 0.00034 Epoch 159/200 - train_loss: 0.00050 - val_loss: 0.00017 Epoch 160/200 - train_loss: 0.00051 - val_loss: 0.00019 Epoch 161/200 - train_loss: 0.00051 - val_loss: 0.00026 Epoch 162/200 - train_loss: 0.00050 - val_loss: 0.00030 Epoch 163/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 164/200 - train_loss: 0.00050 - val_loss: 0.00022 Epoch 165/200 - train_loss: 0.00050 - val_loss: 0.00021 Epoch 166/200 - train_loss: 0.00050 - val_loss: 0.00023 Epoch 167/200 - train_loss: 0.00051 - val_loss: 0.00023 Epoch 168/200 - train_loss: 0.00052 - val_loss: 0.00017 Epoch 169/200 - train_loss: 0.00050 - val_loss: 0.00025 Epoch 170/200 - train_loss: 0.00051 - val_loss: 0.00017 Epoch 171/200 - train_loss: 0.00052 - val_loss: 0.00020 Epoch 172/200 - train_loss: 0.00052 - val_loss: 0.00023 Epoch 173/200 - train_loss: 0.00049 - val_loss: 0.00018 Epoch 174/200 - train_loss: 0.00050 - val_loss: 0.00014 Epoch 175/200 - train_loss: 0.00050 - val_loss: 0.00019 Epoch 176/200 - train_loss: 0.00052 - val_loss: 0.00018 Epoch 177/200 - train_loss: 0.00050 - val_loss: 0.00020 Epoch 178/200 - train_loss: 0.00052 - val_loss: 0.00020 Epoch 179/200 - train_loss: 0.00050 - val_loss: 0.00014 Epoch 180/200 - train_loss: 0.00051 - val_loss: 0.00020 Epoch 181/200 - train_loss: 0.00051 - val_loss: 0.00013 Epoch 182/200 - train_loss: 0.00050 - val_loss: 0.00032 Epoch 183/200 - train_loss: 0.00051 - val_loss: 0.00017 Epoch 184/200 - train_loss: 0.00050 - val_loss: 0.00032 Epoch 185/200 - train_loss: 0.00050 - val_loss: 0.00015 Epoch 186/200 - train_loss: 0.00050 - val_loss: 0.00014 Epoch 187/200 - train_loss: 0.00051 - val_loss: 0.00014 Epoch 188/200 - train_loss: 0.00052 - val_loss: 0.00022 Epoch 189/200 - train_loss: 0.00051 - val_loss: 0.00015 Epoch 190/200 - train_loss: 0.00051 - val_loss: 0.00035 Epoch 191/200 - train_loss: 0.00050 - val_loss: 0.00022 Epoch 192/200 - train_loss: 0.00050 - val_loss: 0.00023 Epoch 193/200 - train_loss: 0.00050 - val_loss: 0.00027 Epoch 194/200 - train_loss: 0.00050 - val_loss: 0.00021 Epoch 195/200 - train_loss: 0.00049 - val_loss: 0.00016 Epoch 196/200 - train_loss: 0.00050 - val_loss: 0.00037 Epoch 197/200 - train_loss: 0.00051 - val_loss: 0.00051 Epoch 198/200 - train_loss: 0.00051 - val_loss: 0.00018 Epoch 199/200 - train_loss: 0.00051 - val_loss: 0.00025 Epoch 200/200 - train_loss: 0.00050 - val_loss: 0.00018


============================================================ COMPARATIVA FINAL DE ARQUITECTURAS (WS=24) - KERAS VS PYTORCH: ============================================================ Window Size Conjunto MSE RMSE MAE Modelo 24 Entrenamiento 0.93 0.97 0.66 LSTM (PyTorch) 24 Entrenamiento 0.95 0.97 0.67 SimpleRNN 24 Entrenamiento 1.11 1.05 0.75 GRU 24 Entrenamiento 1.29 1.14 0.82 LSTM 24 Test 0.42 0.65 0.44 LSTM (PyTorch) 24 Test 0.43 0.66 0.46 SimpleRNN 24 Test 0.47 0.68 0.49 GRU 24 Test 0.61 0.78 0.58 LSTM 24 Validación 0.36 0.60 0.44 SimpleRNN 24 Validación 0.41 0.64 0.48 GRU 24 Validación 0.45 0.67 0.51 LSTM (PyTorch) 24 Validación 0.50 0.70 0.53 LSTM ============================================================

<div style="background-color: #fcf2f2; border-color: #dfb5b4; border-left: 5px solid #dfb5b4; padding: 0.5em;">
<p><strong>Solución:</strong> </p>

Al evaluar el conjunto de Test, la **LSTM de PyTorch lidera el ranking global** con un MAE de 0.44°C y un RMSE de 0.65, seguida de cerca por la SimpleRNN (MAE = 0.46) y la GRU (MAE = 0.49), ambas de Keras. La LSTM de Keras sigue siendo la arquitectura con peor rendimiento (MAE = 0.58), confirmando el patrón observado en el ejercicio anterior.

La diferencia entre las dos implementaciones de LSTM no reside en las ecuaciones de la red, sino en la dinámica del bucle de entrenamiento. En Keras, el modelo fue detenido por `EarlyStopping` en la época 21; en PyTorch, al implementar el bucle manualmente y prescindir de este freno, el modelo iteró durante las 200 épocas completas. Esto se traduce también en el conjunto de entrenamiento, donde la LSTM de PyTorch alcanza un RMSE de 0.97, igualando a la SimpleRNN y muy por encima del 1.14 de la LSTM de Keras. Como se indicó en el ejercicio 2, la LSTM de Keras sufría un ligero *underfitting* causado por la combinación del `Dropout(0.2)` y la parada temprana. El bucle manual de PyTorch demuestra que, si se otorga al optimizador Adam el tiempo suficiente, la red supera esa penalización y converge hacia un mínimo considerablemente más preciso.

En conclusión, la implementación en PyTorch reivindica la capacidad predictiva de la LSTM, demostrando que puede superar a los modelos más sencillos cuando el régimen de entrenamiento es suficientemente exhaustivo. No obstante, el principio de parsimonia plantea una reflexión final: si una SimpleRNN o una GRU logran resultados competitivos (MAE de 0.46 y 0.49 respectivamente) en muchas menos épocas, con menos parámetros y sin forzar el entrenamiento, se confirman como las arquitecturas más eficientes para modelar este transformador a corto plazo, mientras que la LSTM solo alcanza su potencial real bajo condiciones de entrenamiento más exhaustivas.
</div>

---
---

# 4. Predicción multistep con atención

En este ejercicio se abordará un problema de **predicción multistep** en series temporales. A diferencia de los ejercicios anteriores, donde el modelo predecía solo el siguiente instante, ahora el modelo deberá predecir **varios pasos consecutivos en el futuro** a partir de una ventana de observaciones pasadas.

Supongamos que tenemos una serie temporal y que la última observación disponible ocurre en el instante `t`. El objetivo del modelo es predecir una secuencia completa de valores futuros consecutivos:

`t+1, t+2, t+3, ..., t+n`

Es decir, el modelo produce `n` predicciones seguidas en el tiempo. Es importante notar que todas estas predicciones se generan **simultáneamente en una única inferencia** del modelo. Esto es diferente de otro tipo de problema en series temporales donde se predicen **instantes específicos separados**, por ejemplo: `t+1, t+12, t+24`. Este segundo caso es común en predicción meteorológica o energética, donde interesa estimar valores a ciertas horas concretas del futuro.

En este ejercicio trabajaremos con predicción multistep directa, donde el modelo aprende a generar todo el horizonte futuro a la vez.

**Ejercicio [2.5 pts.].** En este ejercicio vas a construir un modelo LSTM con mecanismo de atención para predecir múltiples pasos futuros de la serie temporal ETTh1. Además, explorarás qué partes de la secuencia de entrada influyen más en las predicciones mediante visualización de los pesos de atención.

- Usa el dataset ETTh1.
- Modifica la función para crear secuencias para poder realizar predicciones a múltipes pasos adelante (multistep prediction).
- Divide los datos en un 70% para entrenamiento, 15% validación y 15% test. Escala los valores entre 0 y 1.
- Implementa un modelo secuencial en Tensorflow con:
    - Una capa `LSTM`.
    - Una capa de atención (`Attention`) sobre los timesteps de la LSTM.
    - Una capa `Dropout`.
    - Una capa `Dense` para predecir X pasos futuros.
    
- Usa un tamaño de ventana de 48 y un horizonte de 10. Este horizonte de 10 implica que el modelo produce 10 predicciones simultáneamente.
- Entrena el modelo usando `EarlyStopping` y `ReduceLROnPlateau`. Usa el mismo número de neuronas, tamaño de batch y epochs que en los ejercicios anteriores.
- Calcula RMSE y MAE en entrenamiento, validación y test y añádelos a la tabla comparativa. Las métricas RMSE y MAE se calcularán **promediando el error por muestra** dentro de la ventana de predicción.
- Visualiza el error por horizonte cada horizonte de manera gráfica para el RMSE.
- Extrae los pesos de atención para algunos ejemplos de conjunto de test. Visualiza los pesos de atención sobre la secuencia de entrada para identificar qué timesteps influyen más en la predicción. ¿Que conclusiones puedes extraer en base a la gráfica?

## 4.1. Preparación de Datos

Utilizamos los datos ya escalados y divididos en el ejercicio 1, pues concuerda con los criterios indicados en este ejercicio.

A diferencia del ejercicio 1, en este caso generamos las secuencias con la función `create_sequences_multistep()`, que permite realizar **predicciones multistep**.

En lugar de predecir un único valor, cada ventana de tamaño `window_size` se utiliza para estimar los siguientes `horizon` valores de la variable `OT`. Por ejemplo, con `window_size = 48` y `horizon = 10`, se emplean las filas 0–47 como entrada y se predicen las filas 48–57.

```python
# --- Parámetros del Ejercicio 4 ---
WS_48 = 48
HORIZON = 10
```

```python
# Creamos las secuencias multistep con los datos escalados y divididos del Ejercicio 1
X_train_ws_48, y_train_ws_48 = create_sequences_multistep(train_scaled, WS_48, HORIZON)
X_val_ws_48,   y_val_ws_48   = create_sequences_multistep(val_scaled,   WS_48, HORIZON)
X_test_ws_48,  y_test_ws_48  = create_sequences_multistep(test_scaled,  WS_48, HORIZON)
  
# Guardamos todo en el diccionario
all_sequences_Ej4 = {
    'X_train': X_train_ws_48, 'y_train': y_train_ws_48,
    'X_val': X_val_ws_48,     'y_val': y_val_ws_48,
    'X_test': X_test_ws_48,   'y_test': y_test_ws_48
    }
  
print(f"Secuencias mutistep para window_size = {WS_48}:")
print(f" - X_train: {X_train_ws_48.shape} | y_train: {y_train_ws_48.shape}")
print(f" - X_val:   {X_val_ws_48.shape}   | y_val: {y_val_ws_48.shape}")
print(f" - X_test:  {X_test_ws_48.shape}  | y_test: {y_test_ws_48.shape}")
```


## 4.2. Definición y Entrenamiento del Modelo LSTM Multistep con Atención

Para abordar este problema de predicción múltiple directa, la arquitectura del modelo requiere un diseño más sofisticado que en los ejercicios anteriores. De acuerdo con las especificaciones técnicas de Keras, la inclusión de una capa de Atención (`Attention`) imposibilita el uso de la clase `Sequential` convencional, ya que este mecanismo exige procesar flujos de datos no lineales (requiere evaluar simultáneamente tensores de tipo *query* y *value*). Por ello, el modelo ha sido implementado utilizando la **API Funcional de TensorFlow** (como indica el profesor en el foro).

La arquitectura diseñada sigue este flujo de procesamiento (indicado en el enunciado):

1. **Memoria Secuencial (LSTM):** La capa de entrada proyecta la ventana de 48 horas hacia una capa LSTM. Es indispensable configurar esta capa con `return_sequences=True`. De este modo, la LSTM no colapsa la información en un único vector final, sino que expone el estado oculto de **cada una de las 48 horas** a la siguiente capa.
2. **Mecanismo de Self-Attention:** Las 48 representaciones temporales generadas por la LSTM ingresan en la capa de Atención. El mecanismo cruza esta información consigo misma para calcular un "mapa de importancia" dinámico, permitiendo a la red discriminar qué horas del pasado son cruciales y cuáles son irrelevantes para proyectar el futuro. Esto se denomina self-attention, donde la red aprende qué *timesteps* de su propia secuencia son más relevantes para la predicción. Además, se ha habilitado el parámetro `return_attention_scores=True` para generar una matriz secundaria de pesos que extraeremos en la fase de evaluación visual.
3. **Colapso Dimensional y Regularización:** Dado que la capa de Atención devuelve un tensor tridimensional temporal, utilizamos una capa `GlobalAveragePooling1D` para promediar y "aplanar" la señal antes de aplicar la regularización mediante la capa `Dropout` al 20% y la capa `Dense`.
4. **Predicción Simultánea (Capa Densa):** La red finaliza en una capa `Dense` configurada con 10 neuronas (el horizonte definido), permitiendo emitir el bloque completo de las 10 predicciones futuras en una sola inferencia temporal.

Para el entrenamiento se han replicado los parámetros de ejercicios anteriores, utilizando el optimizador Adam y la pérdida basada en el Error Cuadrático Medio (*MSE*). Al igual que en el ejercicio 2, la sinergia entre los *callbacks* `EarlyStopping` y `ReduceLROnPlateau` asegura que el modelo afine los pesos de manera óptima sin incurrir en sobreajuste. 

Asimismo, y aprovechando la flexibilidad de la API Funcional, se ha definido paralelamente un submodelo (`attention_extractor`) que comparte los mismos pesos entrenados pero cuya única salida es la matriz de atención.

```python
# Semillas
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)
  
# --- Definición de la Arquitectura (API Funcional) ---
# Capa de Entrada
inputs = Input(shape=(WS_48, N_FEATURES))
  
# Capa LSTM: debe devolver toda la secuencia (return_sequences=True)
# para que la capa de Atención pueda evaluar todos los pasos temporales
lstm_out = LSTM(UNITS, return_sequences=True, activation="tanh")(inputs)
  
# Cap de Self-Attention: compara la salida de la LSTM consigo misma (query y value)
# Activmos return_attention_scores=True para poder graficarlos en el siguiente apartado
attn_out, attn_weights = Attntion()([lstm_out, lstm_out], return_attention_scores=True)
  
# Redcimos la dimensionalidad temporal (de 3D a 2D) para pasarla a las capas densas
pooled = GlobalAveragePooling1D()(attn_out)
drop = Dropout(DROPOUT)(pooled)
  
# Capa de Salida (Predice 10 pasos a la vez)
outputs = Dense(HORIZON)(drop)
  
# --- Modelo principal para entrenamiento e inferencia ---
model_attn = Model(inputs=inputs, outputs=outputs, name="LSTM_Attention_Multistep")
model_attn.compile(optimizer="adam", loss="mean_squared_error")
  
# --- Modelo secundario para extraer los pesos ---
# Este odelo recibe la misma entrada pero devuelve la matriz con los pesos de atención
attention_extractor = Model(inputs=inputs, outputs=attn_weights)
  
model_attn.summary()
  
# --- Entrenamiento con Callbacks ---
early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
  
print(f"\nEntrenando modelo multistep (Horizonte={HORIZON})...")
history_attn = model_attn.fit(
    all_sequences_Ej4['X_train'], all_sequences_Ej4['y_train'],
    validation_data=(all_sequences_Ej4['X_val'], all_sequences_Ej4['y_val']),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
```

Model: "LSTM_Attention_Multistep"

┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ input_layer_17      │ (None, 48, 7)     │          0 │ -                 │
│ (InputLayer)        │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ lstm_1 (LSTM)       │ (None, 48, 64)    │     18,432 │ input_layer_17[0… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ attention           │ [(None, 48, 64),  │          0 │ lstm_1[0][0],     │
│ (Attention)         │ (None, 48, 48)]   │            │ lstm_1[0][0]      │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ global_average_poo… │ (None, 64)        │          0 │ attention[0][0]   │
│ (GlobalAveragePool… │                   │            │                   │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dropout_2 (Dropout) │ (None, 64)        │          0 │ global_average_p… │
├─────────────────────┼───────────────────┼────────────┼───────────────────┤
│ dense_17 (Dense)    │ (None, 10)        │        650 │ dropout_2[0][0]   │
└─────────────────────┴───────────────────┴────────────┴───────────────────┘

 Total params: 19,082 (74.54 KB)

 Trainable params: 19,082 (74.54 KB)

 Non-trainable params: 0 (0.00 B)
 
Entrenando modelo multistep (Horizonte=10)... Epoch 1/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 6s 19ms/step - loss: 0.0184 - val_loss: 0.0034 - learning_rate: 0.0010 Epoch 2/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0070 - val_loss: 0.0020 - learning_rate: 0.0010 Epoch 3/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 4s 18ms/step - loss: 0.0059 - val_loss: 0.0017 - learning_rate: 0.0010 Epoch 4/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 4s 19ms/step - loss: 0.0053 - val_loss: 0.0017 - learning_rate: 0.0010 Epoch 5/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 18ms/step - loss: 0.0050 - val_loss: 0.0014 - learning_rate: 0.0010 Epoch 6/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 17ms/step - loss: 0.0047 - val_loss: 0.0012 - learning_rate: 0.0010 Epoch 7/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 18ms/step - loss: 0.0042 - val_loss: 0.0020 - learning_rate: 0.0010 Epoch 8/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 0.0039 - val_loss: 0.0016 - learning_rate: 0.0010 Epoch 9/200 187/190 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0037 Epoch 9: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257. 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 0.0037 - val_loss: 0.0025 - learning_rate: 0.0010 Epoch 10/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 0.0034 - val_loss: 0.0019 - learning_rate: 5.0000e-04 Epoch 11/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 0.0033 - val_loss: 0.0022 - learning_rate: 5.0000e-04 Epoch 12/200 187/190 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0032 Epoch 12: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628. 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 15ms/step - loss: 0.0032 - val_loss: 0.0021 - learning_rate: 5.0000e-04 Epoch 13/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 4s 20ms/step - loss: 0.0031 - val_loss: 0.0021 - learning_rate: 2.5000e-04 Epoch 14/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 4s 21ms/step - loss: 0.0030 - val_loss: 0.0017 - learning_rate: 2.5000e-04 Epoch 15/200 189/190 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - loss: 0.0030 Epoch 15: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814. 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 0.0030 - val_loss: 0.0020 - learning_rate: 2.5000e-04 Epoch 16/200 190/190 ━━━━━━━━━━━━━━━━━━━━ 3s 16ms/step - loss: 0.0029 - val_loss: 0.0021 - learning_rate: 1.2500e-04