<div style="width: 100%; clear: both;">
<div style="float: left; width: 50%;">
<img src="http://www.uoc.edu/portal/_resources/common/imatges/marca_UOC/UOC_Masterbrand.jpg", align="left">
</div>
<div style="float: right; width: 50%;">
<p style="margin: 0; padding-top: 22px; text-align:right;">M2.875 · Deep Learning · PEC1</p>
<p style="margin: 0; text-align:right;">2025-2 · Máster universitario en Ciencia de datos (Data science)</p>
<p style="margin: 0; text-align:right; padding-button: 100px;">Estudios de Informática, Multimedia y Telecomunicación</p>
</div>
</div>
<div style="width:100%;">&nbsp;</div>


# PEC 1: Redes neuronales convolucionales con Keras y Pytorch - Clasificación de imágenes satelitales

A lo largo de esta práctica vamos a implementar varios modelos de redes neuronales para clasificar las imágenes de una base de datos de imágenes satelitales. Concretamente se desarrollaran las siguientes tareas:
1. Descarga, análisis y pre-procesado de los datos (1.5 pts)
2. Red neuronal completamente conectada (Dense NN) (1.5 pts)
3. Red convolucional (CNN) pequeña (2 pts)
4. Autoencoders (2 pts)
5. Transfer Learning con modelos eficientes: EfficientNetB0 (2 pts)
6. Introducción a Pytorch: Replicando nuestra CNN (1 pt)

<u>Consideraciones generales</u>:

- La solución propuesta no debe utilizar métodos, funciones o parámetros declarados **_deprecated_** en versiones futuras.
- Esta PEC se debe hacer de manera **estrictamente individual**. Cualquier indicio de copia será penalizado con un suspenso (D) para todas las partes implicadas y la posible evaluación negativa de la totalidad de la asignatura.
- Es necesario que el estudiante indique **todas las fuentes** que ha utilizado para la realización de la PEC. En caso contrario, se considerará que el alumno ha cometido plagio, siendo penalizado con un suspenso (D) y la posible evaluación negativa de la totalidad de la asignatura.
- Si se utiliza cualquier **IA generativa** en la resolución del PAC **se debe referenciar** en aquellas secciones donde se ha utilizado, como cualquier otra fuente.

<u>Formato de la entrega</u>:

- Algunos ejercicios pueden suponer varios minutos de ejecución, de manera que la entrega se debe hacer en formato **Notebook** y en formato **html**, donde se vea el código, los resultados y los comentarios de cada ejercicio. Podéis exportar el cuaderno a HTML en Jupyter Notebook desde el menú File $\to$ Download as $\to$ HTML.
- Hay un tipo especial de celda para alojar el texto. Este tipo de celdas será muy útil para responder a las diferentes preguntas teóricas planteadas a lo largo de la actividad. Para cambiar el tipo de celda a este tipo, en el menú: Cell $\to$ Cell Type $\to$ Markdown.

<u>Dataset utilizado</u>:

En este PAC utilizaremos **UC Merced Land Use Dataset**, un conjunto de datos de libre acceso que consta de imágenes aéreas de alta resolución (2 m) de una región agrícola en el Valle Central de California.

## 0. Contexto y carga de librerías
Las imágenes tomadas por satélite son clave en la supervisión del uso y la cobertura del suelo, cuestiones relevantes para la gestión ambiental, la planificación urbana, la sostenibilidad y para combatir el cambio climático.

En esta práctica, trabajaremos con la base de datos [UC Merced Land Use Data](https://www.kaggle.com/datasets/zeadomar/uc-mercedland), que consiste en imágenes satelitales de 256x256 píxeles de 21 escenas diferentes: las clases son diversas, y contienen escenas e imágenes de aviones o ríos, entre otras categorías.

Concretamente trabajaremos con una versión aumentada de dicha base de datos que está disponible en un [repositorio de Kaggle](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification). En esta versión se han llevado a cabo varios procesos de aumentación de datos de tal forma que el número de imágenes por clase pasa de 100 a 500.

**Nota: Se recomienda realizar la práctica en el entorno que ofrece la plataforma Kaggle, ya que ofrece un entorno gratuito con 30 horas semanales de uso de GPU.**

A lo largo de toda la práctica, para la creación de las distintas redes, iremos alternando el uso del modelo [Sequential](https://keras.io/guides/sequential_model/) y el modelo [Functional](https://keras.io/guides/functional_api/) de Keras a través de las clases [Sequential](https://keras.io/api/models/sequential/) y [Model](https://keras.io/api/models/model/) respectivamente.

Empezamos cargando las librerías mas relevantes:

```python
# Importamos tensorflow
import tensorflow as tf
print("TF version   : ", tf.__version__)

# Necesitaremos GPU
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# keras version is 2.11.0
import keras
print("Keras version   : ", keras.__version__)
```

```python
# Importamos los elementos de keras que utilizaremos con mayor frecuencia
from keras.utils import image_dataset_from_directory
from keras.layers import (
    GlobalAveragePooling2D, Flatten,
    Dense, Dropout, Conv2D, Conv2DTranspose, BatchNormalization,
    MaxPooling2D, UpSampling2D, Rescaling, Resizing)

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import Sequential, Model
```

```python
# Importamos el resto de librerías que necesitaremos para la PEC
import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
```

## 1. Descarga, análisis y pre-procesado de los datos (1,5 puntos)

En este apartado exploraremos la base de datos y prepararemos la carga de las imágenes para los modelos de los siguientes apartados.

Para la descarga de la base de datos tenemos 2 opciones dependiendo de si decidimos trabajar en local o desde el entorno de Kaggle:
- Si trabajamos en local debemos descargar la base de datos desde el siguiente [enlace](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification/download?datasetVersionNumber=3) (es un archivo .zip que ocupa 2 GB) y descomprimirlo.
- Si trabajamos desde Kaggle. Debemos subir el Notebook del enunciado a la plataforma (para ello podéis seguir los 6 primeros pasos del siguiente [artículo](https://rajputankit22.medium.com/how-to-upload-my-own-notebook-to-kaggle-2b0dedbb5a6b)) y después, una vez subido el notebook, en la pestaña 'Input', clickar el botón '+ Add Input' y en la caja de búsqueda introducir la dirección 'https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification'. Una vez encontrado el dataset darle al botón '+' (Add Dataset), y desde ese momento ya tendréis accesible la base de datos en la ruta <code>../input/</code>.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("apollo2506/landuse-scene-classification")

print("Path to dataset files:", path)
```

Una vez tenemos la base de datos accesible vamos a inspeccionarla.

Las imágenes se encuentran agrupadas de 2 formas diferentes:
- En la carpeta <code>/landuse-scene-classification/images/</code> se encuentra el total de las imágenes separadas por clases (cada clase en una carpeta distinta). Pero no se ha realizado una separación en conjunto de entrenamiento y test (o entrenamiento, validación y test).
- En la carpeta <code>/landuse-scene-classification/images_train_test_val/</code> se encuentran 3 carpetas (<code>test</code>, <code>train</code> y <code>validation</code>) en las que el total de imágenes se ha separado de forma aleatoria. En cada una de las 3 carpetas, tenemos imágenes de las 21 clases agrupadas en sus correspondientes carpetas. En la carpeta raíz <code>/landuse-scene-classification/</code> tenemos 3 archivos .csv con la distribución de cada carpeta.

En esta práctica utilizaremos el dataset ya particionado, es decir, trabajaremos con las imágenes que se encuentran en la ruta <code>/landuse-scene-classification/images_train_test_val/</code>.

### 1.1. Análisis de los archivos .csv

A partir de los archivos .csv podemos ver cómo se han distribuído los datos. Por ejemplo:

```python
import pandas as pd 

# Carga de los datos
train = pd.read_csv(r'C:\Users\usuario\Documents\GitHub\Deep-learning\Módulo 2 - Redes neuronales convolucionales (CNN)\Enunciado de la PEC\data\apollo2506\landuse-scene-classification\versions\3\train.csv')

test = pd.read_csv(r'C:\Users\usuario\Documents\GitHub\Deep-learning\Módulo 2 - Redes neuronales convolucionales (CNN)\Enunciado de la PEC\data\apollo2506\landuse-scene-classification\versions\3\test.csv')

validation = pd.read_csv(r'C:\Users\usuario\Documents\GitHub\Deep-learning\Módulo 2 - Redes neuronales convolucionales (CNN)\Enunciado de la PEC\data\apollo2506\landuse-scene-classification\versions\3\validation.csv')
```

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio [0,5 pts.]:</strong> A partir de los 3 archivos .csv se pide:
    <ul>
        <li>Extraer los nombres de las 21 clases (esto sólo hace falta hacerlo en uno de los 3 archivos)</li>
        <li>¿Cuántas instancias tenemos en total para cada conjunto de datos?</li>
        <li>Comprobar que las clases están balanceadas en los 3 conjuntos de datos (contando para cada conjunto, cuantas instancias/ejemplos tenemos para cada clase)</li>
    </ul>        
</div>

```python
# Extraer nombres de las clases
classes = sorted(train['Label'].unique())
print(f"Número de clases: {len(classes)}")
print("Clases:", classes)
```

Número de clases: 21 Clases: [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6), np.int64(7), np.int64(8), np.int64(9), np.int64(10), np.int64(11), np.int64(12), np.int64(13), np.int64(14), np.int64(15), np.int64(16), np.int64(17), np.int64(18), np.int64(19), np.int64(20)]

```python
# Número de instancias por conjunto
print("\nNúmero de instancias por conjunto:")
print(f"Train: {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test: {len(test)}")
```

Número de instancias por conjunto: Train: 7350 Validation: 2100 Test: 1050

```python
# Número de instancias por clase
print("\nDistribución de clases en TRAIN:")
print(train['Label'].value_counts().sort_index())
  
print("\nDistribución de clases en VALIDATION:")
print(validation['Label'].value_counts().sort_index())

print("\nDistribución de clases en TEST:")
print(test['Label'].value_counts().sort_index())
```

Distribución de clases en TRAIN: Label 0 350 1 350 2 350 3 350 4 350 5 350 6 350 7 350 8 350 9 350 10 350 11 350 12 350 13 350 14 350 15 350 16 350 17 350 18 350 19 350 20 350 Name: count, dtype: int64 Distribución de clases en VALIDATION: Label 0 100 1 100 2 100 3 100 4 100 5 100 6 100 7 100 8 100 9 100 10 100 11 100 12 100 13 100 14 100 15 100 16 100 17 100 18 100 19 100 20 100 Name: count, dtype: int64 Distribución de clases en TEST: Label 0 50 1 50 2 50 3 50 4 50 5 50 6 50 7 50 8 50 9 50 10 50 11 50 12 50 13 50 14 50 15 50 16 50 17 50 18 50 19 50 20 50 Name: count, dtype: int64

<div style="background-color: #fcf2f2; border-color: #dfb5b4; border-left: 5px solid #dfb5b4; padding: 0.5em;">
<strong>Comentarios:</strong>
<br><br>
A partir del análisis de los archivos '.csv' se observa que el dataset contiene clases diferentes, identificadas mediante etiquetas numéricas que van de 0 a 20.

En cuanto a la partición de los datos, el conjunto se distribuye de la siguiente forma:
- **Train:** 7350 imágenes  
- **Validation:** 2100 imágenes  
- **Test:** 1050 imágenes  

El análisis de la distribución de clases muestra que **todas las clases están perfectamente balanceadas en los tres subconjuntos**. En el conjunto de entrenamiento hay **350 imágenes por clase**, en el conjunto de validación **100 imágenes por clase**, y en el conjunto de test **50 imágenes por clase**.

Este balanceo es ventajoso para el entrenamiento de modelos de clasificación, ya que evita sesgos hacia determinadas clases y permite que el modelo aprenda representaciones de todas ellas de forma equilibrada.
<br><br>
</div>

### 1.2. Análisis de las carpetas de imágenes.

Aunque se supone que cada archivo .csv refleja a la perfección el contenido de cada conjunto de datos, no está demás cerciorarse que el contenido del mismo se corresponde con lo anotado en cada archivo. Para ello se pide:

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio[0,5 pts]:</strong> Proporciona, a partir de las carpetas de imágenes, el número de imágenes que tenemos en cada categoría para cada conjunto de datos, comprobando que coincide con lo estipulado en el archivo .csv, y visualiza a modo de ejemplo una imagen por cada categoría. ¿Qué rango dinámico (valores mínimo y máximo) tienen las imágenes?
</div>

```python
base_path = r"C:\Users\usuario\Documents\GitHub\Deep-learning\Módulo 2 - Redes neuronales convolucionales (CNN)\Enunciado de la PEC\data\apollo2506\landuse-scene-classification\versions\3"

images_path = os.path.join(base_path, "images_train_test_val")

# Usar los CSVs que ya cargaste
csv_dict = {"train": train, "validation": validation, "test": test}
  
for dataset_name in ["train", "validation", "test"]:
    path = os.path.join(images_path, dataset_name)
    class_counts = {}
    # Contar imágenes en carpetas
    for class_folder in sorted(os.listdir(path)):
        class_path = os.path.join(path, class_folder)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith('.png')])
            class_counts[class_folder] = num_images
    print(f"\n{dataset_name.upper()}:")
    print(f"Total imágenes en carpetas: {sum(class_counts.values())}")
    print(f"Total imágenes en CSV: {len(csv_dict[dataset_name])}")
```

TRAIN: Total imágenes en carpetas: 7350 Total imágenes en CSV: 7350 VALIDATION: Total imágenes en carpetas: 2100 Total imágenes en CSV: 2100 TEST: Total imágenes en carpetas: 1050 Total imágenes en CSV: 1050

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Obtener los nombres de las clases
train_path = os.path.join(images_path, "train")
class_names = sorted([d for d in os.listdir(train_path)
                      if os.path.isdir(os.path.join(train_path, d))])

print(f"Clases encontradas: {len(class_names)}")

# Almacenar estadísticas
stats_list = []

# Crear figura con 21 imágenes (7 filas x 3 columnas)
fig, axes = plt.subplots(7, 3, figsize=(15, 25))
axes = axes.flatten()

for idx, class_name in enumerate(class_names):
    class_path = os.path.join(train_path, class_name)
    
    # Obtener la primera imagen de la clase
    images = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        img_path = os.path.join(class_path, images[0])
        img = Image.open(img_path)
        
        # Convertir a array para obtener estadísticas
        img_array = np.array(img)

        # Calcular estadísticas
        min_val = img_array.min()
        max_val = img_array.max()
        mean_val = img_array.mean()

        # Guardar estadísticas
        stats_list.append({
            'clase': class_name,
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'shape': img_array.shape,
            'dtype': img_array.dtype
        })

        # Mostrar en subplot
        axes[idx].imshow(img)
        axes[idx].set_title(
            f"{class_name}\n"
            f"Size: {img.size} | Min: {min_val} | Max: {max_val}\n"
            f"Mean: {mean_val:.2f} | Type: {img_array.dtype}",
            fontsize=9, fontweight='bold'
        )
        axes[idx].axis('off')
plt.tight_layout()
plt.show()

# RESUMEN GLOBAL
print("\n" + "="*80)
print("RESUMEN GLOBAL DEL RANGO DINÁMICO")
print("="*80)

all_mins = [s['min'] for s in stats_list]
all_maxs = [s['max'] for s in stats_list]
all_means = [s['mean'] for s in stats_list]  

print(f"\n{'Clase':<25} {'Min':<10} {'Max':<10} {'Media':<10}")
print("-"*80)
for stat in stats_list:
    print(f"{stat['clase']:<25} {stat['min']:<10.0f} {stat['max']:<10.0f} {stat['mean']:<10.2f}")

print("\n" + "="*80)
print(f"Valor MÍNIMO global:        {min(all_mins):.0f}")
print(f"Valor MÁXIMO global:        {max(all_maxs):.0f}")
print(f"Media global de píxeles:    {np.mean(all_means):.2f}")
print(f"\nFormato de imagen:")
print(f"  - Data type: {stats_list[0]['dtype']}")
print(f"  - Tamaño: {stats_list[0]['shape']}")
print(f"\n✓ CONCLUSIÓN: Las imágenes están en formato UINT8")
print(f"✓ Rango dinámico: [{min(all_mins):.0f}, {max(all_maxs):.0f}]")
print("="*80)
```

Clases encontradas: 21

![[Pasted image 20260306183826.png]]

Las imágenes conservan su formato nativo UINT8 con un rango dinámico completo de [0, 255].

Es fundamental destacar que las imágenes no han sufrido una normalización previa, lo que preserva la variabilidad visual completa de las tomas satelitales y las deja listas para ser normalizadas eficientemente dentro de las propias redes (mediante la capa Rescaling).

### 1.3. Creación de los conjuntos de datos en formato Keras/Tensorflow
​
Con el objetivo de crear una base de datos en el formato Keras/Tensorflow a partir de las imágenes proporcionadas utilizaremos la función <code>**tf.keras.utils.image_dataset_from_directory()**</code> ya que nos permite crear bases de datos a partir de imágenes guardadas en carpetas.

La documentación de esta función se encuentra tanto en la web de [Keras](https://keras.io/api/data_loading/image/) como en la de [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory) .

Además, aprovecharemos para redimensionar las imágenes y pasarlas a tamaño 224x224, que es el tamaño con el que se ha entrenado la red EfficientNetB0 que utilizaremos en un apartado posterior.

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio[0,5 pts]:</strong> Utiliza la función  <code>image_dataset_from_directory()</code> para generar 3 conjuntos de datos (<code>train_data</code>, <code>val_data</code> y <code>test_data</code>) a partir de las carpetas analizadas. Las imágenes deben ser redimensionadas a tamaño 224x224 píxels RGB (224,224,3) y agrupadas en lotes de tamaño 32 (batch=32) manteniendo su rango dinámico.
</div>

```python
from keras.utils import image_dataset_from_directory

# Rutas de los conjuntos
train_dir = os.path.join(images_path, "train")
val_dir = os.path.join(images_path, "validation")
test_dir = os.path.join(images_path, "test")

# Crear train_data
train_data = image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode='rgb'
)

# Crear val_data
val_data = image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode='rgb'
)
 
# Crear test_data
test_data = image_dataset_from_directory(
    test_dir,
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    color_mode='rgb'
)

# 18 segundos de ejecución
```

Found 7350 files belonging to 21 classes. Found 2100 files belonging to 21 classes. Found 1050 files belonging to 21 classes. ✓ Conjuntos de datos creados exitosamente

Los números coinciden perfectamente con lo que vimos en los CSVs:

* **Train:** 7350 imágenes (350 por clase)
* **Validation:** 2100 imágenes (100 por clase)
* **Test:** 1050 imágenes (50 por clase)

Esto confirma que `image_dataset_from_directory()` ha detectado todas las imágenes correctamente.

```python
# Verificar las dimensiones
print("\nDimensiones de los conjuntos:")
for images, labels in train_data.take(1):
    print(f"Train - Imágenes shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"Rango de valores: min={images.numpy().min()}, max={images.numpy().max()}")

for images, labels in val_data.take(1):
    print(f"Validation - Imágenes shape: {images.shape}, Labels shape: {labels.shape}")

for images, labels in test_data.take(1):
    print(f"Test - Imágenes shape: {images.shape}, Labels shape: {labels.shape}")
```

Dimensiones de los conjuntos: Train - Imágenes shape: (32, 224, 224, 3), Labels shape: (32,) Rango de valores: min=0.0, max=255.0 Validation - Imágenes shape: (32, 224, 224, 3), Labels shape: (32,) Rango de valores: min=0.0, max=255.0 Test - Imágenes shape: (32, 224, 224, 3), Labels shape: (32,) Rango de valores: min=0.0, max=255.0

Esto significa:

* 32: Tamaño del batch (lote) especificado.
* 224, 224: Tamaño de imagen redimensionada.
* 3: Canales RGB (rojo, verde, azul).
* 0-255: Rango dinámico original de las imágenes.

De modo que, las imágenes han sido redimensionadas de 256x256 a 224x224, y mantienen su rango dinámico original (0-255).

## 2. Red neuronal completamente conectada (Dense NN) (1,5 puntos)

En este apartado, vamos a entrenar y evaluar un modelo muy sencillo completamente conectado para establecer un resultado de referencia.

Dado que en una red neuronal artificial las entradas son unidimensionales, lo primero que tenemos que hacer es redimensionar los datos de entrada (las imágenes) para convertirlos en arrays de una dimensión.

Como trabajar con imágenes de tamaño 224x224 en una red completamente conectada implicaría entrenar un número de parámetros excesivamente elevado definiremos un modelo en el que se realizará previamente un redimensionado de las imágenes de entrada a un tamaño de 32x32 y un achatamiento (*flattening*) de los píxeles para así generar un vector unidimensional de tamaño 3072 (32x32x3).

Posteriormente entrenaremos un clasificador (una red completamente conectada) para llevar a cabo la clasificación de nuestros datos.

En este apartado utilizaremos las capas [Resizing](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/resizing/), [Rescaling](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/), [Flatten](https://keras.io/api/layers/reshaping_layers/flatten/), [Dense](https://keras.io/api/layers/core_layers/dense/) y [Dropout](https://keras.io/api/layers/regularization_layers/dropout/) de keras.


<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
    <strong>Ejercicio:</strong> Implementa un modelo <strong>secuencial</strong> de Keras (a partir de la clase <code>Sequential()</code>) con las siguientes especificaciones:
    <ul>
        <li>Una capa que reduzca las dimensiones de entrada de (224,224) a (32,32)</li>
        <li>Una capa de reescalado para conseguir que los valores de la imagen estén entre 0 y 1</li>
        <li>Una capa Flatten para convertir la imagen en un vector de 3072 posiciones</li>
        <li>Una capa completamente conectada de 1024 neuronas y activación ReLU</li>
        <li>Una capa de Dropout (con probabilidad 0.5)</li>
        <li>Una capa de salida completamente conectada correspondiente a la clasificación final cuyo número de neuronas debe ser igual al múmero de clases de la base de datos y con la función de activación adecuada para llevar a cabo esta tarea de clasificación.</li>
    </ul>
        
Compilar y entrenar el modelo siguiendo las siguientes indicaciones:
     <ul>
         <li>Utilizar el optimizador Adam con  <i>learning rate</i> de 0.0001.</li>
         <li>Entrenar durante 100 épocas utilizando  <i>EarlyStopping</i> con una persistencia de 10 épocas, monitorizando la función de pérdida en el conjunto de validación, y guardando los pesos que mejor resultado hayan obtenido.</li>
         <li>Monitorizar la métrica  <i>accuracy</i> durante entrenamiento y validación.</li>
         <li>Mostrar las gráficas de accuracy y loss. En cada gráfica debe visualizarse la curva de entrenamiento y la de validación. NOTA: Se recomienda hacer una función que imprima ambas gráficas para poder reutilizarla en próximos apartados.</li>
         <li>Realizar la evaluación del modelo una vez ha finalizado el entrenamiento para mostrar la pérdida y la exactitud final sobre los datos de test.</li>
    </ul>
    Preguntas a responder: ¿Cúal es el número de parámetros a entrenar? ¿y el tiempo de entrenamiento? ¿Qué precisión se obtiene con este modelo? Comenta los resultados y las gráficas de entrenamiento.<br/>    
    <strong> NOTA: se recomienda, al final de la creación de cada modelo, utilizar la función <code>summary()</code> para comprobar la estructura de la red creada, así como el numero de parámetros que se deben entrenar. Se recomienda hacerlo en todos los ejercicios.</strong>
</div>

```python
# Definición de la red
# Nº de clases/ Nº de neuronas de la capa de salida
num_classes = len(class_names)
  
# Implementar red neuronal completamente conectada (Dense Neural Network)
# Crear el modelo Sequential con capas Dense
model_dense = Sequential([

    # Reducir tamaño de imagen
    Resizing(32, 32, input_shape=(224,224,3)),

    # Normalización
    Rescaling(1./255),

    # Convertir imagen en vector
    Flatten(),

    # Capa densa
    Dense(1024, activation='relu'),

    # Regularización
    Dropout(0.5),

    # Capa de salida
    Dense(num_classes, activation='softmax')
])

# Visualización de la estructura de la red
model_dense.summary()
```
Model: "sequential_1"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ resizing_1 (Resizing)           │ (None, 32, 32, 3)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ rescaling_1 (Rescaling)         │ (None, 32, 32, 3)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten_1 (Flatten)             │ (None, 3072)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1024)           │     3,146,752 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 1024)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 21)             │        21,525 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 3,168,277 (12.09 MB)

 Trainable params: 3,168,277 (12.09 MB)

 Non-trainable params: 0 (0.00 B)

```python
# Compilación de la red

optimizer = Adam(learning_rate=1e-3)

model_dense.compile(
    optimizer=optimizer,                    # Actualizar los pesos de la red
    loss='sparse_categorical_crossentropy', # Funcion de pérdida a minimizar
    metrics=['accuracy']                    # Métrica de evaluacion
)

  

# Detener entreno cuando no haya mejoras en 15 epochs seguidas
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

# Entrenamiento de la red
start_time = time.time()

history_dense = model_dense.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stop]
)

training_time = time.time() - start_time
print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")
```
Epoch 1/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 25s 103ms/step - accuracy: 0.0680 - loss: 3.1031 - val_accuracy: 0.0857 - val_loss: 3.0024 Epoch 2/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.0920 - loss: 2.9981 - val_accuracy: 0.0900 - val_loss: 2.9658 Epoch 3/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 75ms/step - accuracy: 0.1033 - loss: 2.9544 - val_accuracy: 0.1262 - val_loss: 2.9394 Epoch 4/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 75ms/step - accuracy: 0.1269 - loss: 2.9093 - val_accuracy: 0.1400 - val_loss: 2.9050 Epoch 5/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 76ms/step - accuracy: 0.1410 - loss: 2.8565 - val_accuracy: 0.1167 - val_loss: 2.8818 Epoch 6/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.1507 - loss: 2.8236 - val_accuracy: 0.1624 - val_loss: 2.8459 Epoch 7/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 76ms/step - accuracy: 0.1653 - loss: 2.7817 - val_accuracy: 0.1676 - val_loss: 2.8283 Epoch 8/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 76ms/step - accuracy: 0.1801 - loss: 2.7362 - val_accuracy: 0.1643 - val_loss: 2.8072 Epoch 9/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 74ms/step - accuracy: 0.1894 - loss: 2.7071 - val_accuracy: 0.1743 - val_loss: 2.7839 Epoch 10/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 74ms/step - accuracy: 0.2084 - loss: 2.6711 - val_accuracy: 0.1833 - val_loss: 2.7665 Epoch 11/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 73ms/step - accuracy: 0.2128 - loss: 2.6422 - val_accuracy: 0.1748 - val_loss: 2.7477 Epoch 12/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 74ms/step - accuracy: 0.2268 - loss: 2.6083 - val_accuracy: 0.1924 - val_loss: 2.7326 Epoch 13/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 75ms/step - accuracy: 0.2367 - loss: 2.5791 - val_accuracy: 0.1895 - val_loss: 2.7308 Epoch 14/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 17s 75ms/step - accuracy: 0.2411 - loss: 2.5487 - val_accuracy: 0.1990 - val_loss: 2.7113 Epoch 15/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 78ms/step - accuracy: 0.2520 - loss: 2.5241 - val_accuracy: 0.1919 - val_loss: 2.7083 Epoch 16/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 76ms/step - accuracy: 0.2599 - loss: 2.4943 - val_accuracy: 0.1933 - val_loss: 2.6937 Epoch 17/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 78ms/step - accuracy: 0.2733 - loss: 2.4563 - val_accuracy: 0.1857 - val_loss: 2.7103 Epoch 18/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 76ms/step - accuracy: 0.2750 - loss: 2.4292 - val_accuracy: 0.2110 - val_loss: 2.6807 Epoch 19/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 20s 85ms/step - accuracy: 0.2820 - loss: 2.4105 - val_accuracy: 0.2038 - val_loss: 2.6776 Epoch 20/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 20s 87ms/step - accuracy: 0.2988 - loss: 2.3739 - val_accuracy: 0.2152 - val_loss: 2.6572 Epoch 21/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.3156 - loss: 2.3408 - val_accuracy: 0.2033 - val_loss: 2.6778 Epoch 22/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.3192 - loss: 2.3262 - val_accuracy: 0.2205 - val_loss: 2.6445 Epoch 23/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.3282 - loss: 2.2918 - val_accuracy: 0.2019 - val_loss: 2.6700 Epoch 24/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.3287 - loss: 2.2704 - val_accuracy: 0.2095 - val_loss: 2.6432 Epoch 25/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 78ms/step - accuracy: 0.3454 - loss: 2.2471 - val_accuracy: 0.2100 - val_loss: 2.6514 Epoch 26/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.3596 - loss: 2.2026 - val_accuracy: 0.2143 - val_loss: 2.6317 Epoch 27/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 78ms/step - accuracy: 0.3672 - loss: 2.1720 - val_accuracy: 0.2100 - val_loss: 2.6596 Epoch 28/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.3718 - loss: 2.1464 - val_accuracy: 0.2205 - val_loss: 2.6315 Epoch 29/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 76ms/step - accuracy: 0.3785 - loss: 2.1254 - val_accuracy: 0.2314 - val_loss: 2.6260 Epoch 30/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 76ms/step - accuracy: 0.3914 - loss: 2.0957 - val_accuracy: 0.2343 - val_loss: 2.6080 Epoch 31/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 76ms/step - accuracy: 0.3920 - loss: 2.0828 - val_accuracy: 0.2186 - val_loss: 2.6120 Epoch 32/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.4128 - loss: 2.0312 - val_accuracy: 0.2314 - val_loss: 2.6250 Epoch 33/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 76ms/step - accuracy: 0.4131 - loss: 2.0164 - val_accuracy: 0.2281 - val_loss: 2.6065 Epoch 34/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 78ms/step - accuracy: 0.4235 - loss: 1.9949 - val_accuracy: 0.2152 - val_loss: 2.6211 Epoch 35/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 77ms/step - accuracy: 0.4354 - loss: 1.9696 - val_accuracy: 0.2333 - val_loss: 2.6136 Epoch 36/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 78ms/step - accuracy: 0.4438 - loss: 1.9290 - val_accuracy: 0.2300 - val_loss: 2.6241 Epoch 37/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 18s 79ms/step - accuracy: 0.4452 - loss: 1.9083 - val_accuracy: 0.2319 - val_loss: 2.6092 Epoch 38/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 83ms/step - accuracy: 0.4582 - loss: 1.8938 - val_accuracy: 0.2329 - val_loss: 2.6076 Epoch 39/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 20s 86ms/step - accuracy: 0.4585 - loss: 1.8845 - val_accuracy: 0.2414 - val_loss: 2.6030 Epoch 40/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 84ms/step - accuracy: 0.4611 - loss: 1.8582 - val_accuracy: 0.2457 - val_loss: 2.5913 Epoch 41/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 84ms/step - accuracy: 0.4784 - loss: 1.8250 - val_accuracy: 0.2452 - val_loss: 2.5994 Epoch 42/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 84ms/step - accuracy: 0.4895 - loss: 1.7902 - val_accuracy: 0.2395 - val_loss: 2.5960 Epoch 43/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 20s 86ms/step - accuracy: 0.4878 - loss: 1.7805 - val_accuracy: 0.2367 - val_loss: 2.6027 Epoch 44/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 20s 86ms/step - accuracy: 0.4999 - loss: 1.7335 - val_accuracy: 0.2381 - val_loss: 2.6264 Epoch 45/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 84ms/step - accuracy: 0.5039 - loss: 1.7312 - val_accuracy: 0.2419 - val_loss: 2.6097 Epoch 46/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 83ms/step - accuracy: 0.5135 - loss: 1.6995 - val_accuracy: 0.2333 - val_loss: 2.6118 Epoch 47/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 82ms/step - accuracy: 0.5076 - loss: 1.6911 - val_accuracy: 0.2333 - val_loss: 2.5946 Epoch 48/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 83ms/step - accuracy: 0.5336 - loss: 1.6510 - val_accuracy: 0.2452 - val_loss: 2.5973 Epoch 49/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 19s 82ms/step - accuracy: 0.5350 - loss: 1.6358 - val_accuracy: 0.2371 - val_loss: 2.6111 Epoch 50/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 20s 85ms/step - accuracy: 0.5450 - loss: 1.6021 - val_accuracy: 0.2452 - val_loss: 2.6317 Tiempo de entrenamiento: 916.01 segundos



```python
# Plot del training loss y la accuracy
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.figure(figsize=(14,5))
  
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, label='Train accuracy')
    plt.plot(epochs, val_acc, label='Validation accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
  
    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()
```

```python
plot_training_history(history_dense)
```
![[Evolucion de accuracy y loss_val por epoch_dense.png]]
```python
test_loss, test_acc = model_dense.evaluate(test_data, verbose=2)
  
print(f"\nResultados en TEST:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
```

33/33 - 168s - 5s/step - accuracy: 0.1676 - loss: 2.8289 Resultados en TEST: Loss: 2.8289 Accuracy: 0.1676

### Número de parámetros a entrenar

El modelo implementado presenta un total de **3.168.277 parámetros entrenables**, tal y como se observa en el resumen de la arquitectura (`model.summary()`). La gran mayoría de estos parámetros se concentran en la primera capa densa (`Dense(1024)`), que conecta el vector de entrada de **3072 características** (resultado de aplicar `Flatten` sobre la imagen redimensionada a 32×32×3) con las **1024 neuronas** de la capa oculta. Esta capa contiene **3.146.752 parámetros**, lo que representa prácticamente la totalidad del modelo.

La capa de salida (`Dense(21)`) añade **21.525 parámetros adicionales**, correspondientes a las conexiones entre las 1024 neuronas de la capa anterior y las 21 neuronas de salida, una por cada clase del dataset. El resto de capas (`Resizing`, `Rescaling`, `Flatten` y `Dropout`) no contienen parámetros entrenables.
### Tiempo de entrenamiento

El tiempo total de entrenamiento se ha realizado en unos pocos minutos (aproximadamente **3365.41 segundos). Este tiempo es relativamente reducido debido ya que la arquitectura utilizada es sencilla y el número de capas es limitado, aunque el número total de parámetros es elevado debido al uso de capas densas.

Además, se ha utilizado un mecanismo de **EarlyStopping**, que detiene el entrenamiento si la pérdida en el conjunto de validación no mejora durante **10 épocas consecutivas**. Esto permite evitar entrenamientos innecesariamente largos y reduce el riesgo de sobreentrenamiento.

### Precisión obtenida

La evaluación final del modelo sobre el conjunto de **test** arroja una **accuracy de 0.1676**, lo que significa que el modelo clasifica correctamente aproximadamente el **16.8% de las imágenes**.

Si se considera que el problema consta de **21 clases**, una clasificación completamente aleatoria tendría una precisión aproximada de: (1/21)/00.476.

Por tanto, el modelo obtiene un rendimiento superior al azar, aunque la precisión sigue siendo relativamente baja para un problema de clasificación de imágenes.

### Comentario de las gráficas de entrenamiento

Las gráficas de evolución de **accuracy** y **loss** durante el entrenamiento muestran que el modelo consigue aprender ciertos patrones presentes en el conjunto de entrenamiento, ya que la precisión de entrenamiento aumenta progresivamente en las primeras épocas y la pérdida disminuye.

Sin embargo, la mejora en el conjunto de **validación** es mucho más limitada. La precisión de validación tiende a estabilizarse rápidamente y la pérdida deja de disminuir tras varias épocas, lo que indica que el modelo alcanza pronto su capacidad máxima de generalización. Este comportamiento sugiere que la arquitectura utilizada tiene dificultades para capturar patrones complejos presentes en las imágenes.

### Interpretación de los resultados

El bajo rendimiento obtenido se explica principalmente por las **limitaciones de las redes completamente conectadas cuando se aplican a imágenes**.

Al utilizar una capa `Flatten`, la red pierde completamente la **estructura espacial de la imagen**, tratándose cada píxel como una característica independiente. Esto impide que el modelo pueda detectar patrones visuales relevantes como bordes, texturas o formas, que son fundamentales en tareas de visión por computador.

Además, la gran cantidad de parámetros concentrados en la capa densa aumenta la complejidad del modelo sin aprovechar adecuadamente la información espacial presente en las imágenes.

### Conclusión

En conclusión, el modelo implementado en este apartado sirve como **referencia inicial para el problema de clasificación**, permitiendo establecer un punto de comparación para arquitecturas más avanzadas.

Los resultados obtenidos muestran que las redes completamente conectadas no son la opción más adecuada para el análisis de imágenes, lo que justifica el uso de **redes neuronales convolucionales (CNN)** en los siguientes apartados de la práctica, ya que estas arquitecturas están específicamente diseñadas para explotar la estructura espacial de los datos visuales y suelen ofrecer un rendimiento significativamente superior.
## 3. Red convolucional pequeña (2 puntos)

Dadas las bajas prestaciones del modelo anterior vamos a probar otro tipo de redes con el objetivo de obtener unos mejores resultados en la tarea de clasificación que debemos llevar a cabo.

Las redes convolucionales (CNN) son especialmente adecuadas para modelar datos donde hay patrones en 2 dimensiones, como es el caso de las imágenes.

En la tarea de clasificación, la estructura de una CNN se divide en dos grandes bloques:

* **Bloque extractor de características**: En este bloque se generan diferentes niveles de abstracción de la imagen de entrada mediante capas convolucionales. Cuanto más profundas son estas capas, más preparadas están para la tarea de clasificación.
* **Clasificador**: Este bloque está formado por capas totalmente conectadas, la salida de este bloque será la probabilidad asociada a cada clase.

En el apartado anterior, el bloque "extractor de características" era extremadamente simple, por no decir inexistente. En este apartado, vamos a hacer uso de capas convolucionales para poder aprender mejores abstracciones de las imágenes de entrada con el fin de mejorar su clasificación.

En este apartado utilizaremos las capas [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/),  [MaxPooling2D](https://keras.io/api/layers/pooling_layers/max_pooling2d/), [GlobalAveragePooling2D](https://keras.io/api/layers/pooling_layers/global_average_pooling2d/), [Dense](https://keras.io/api/layers/core_layers/dense/) y [Dropout](https://keras.io/api/layers/regularization_layers/dropout/) de keras.

**Nota: Se recomienda, a partir de este punto realizar el entrenamiento en una máquina con GPU (puede activarse en plataformas como Google Colab o Kaggle) con el fin de reducir los tiempos de entrenamiento.**

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
    <strong>Ejercicio [2 puntos]:</strong> A partir del modelo <strong>funcional</strong> de keras (y la clase <code>Model()</code>), implementa una red con las siguientes características:
    <ul>
        <li>Un bloque extractor de características que conste de:
            <ul>
                <li>Una capa de entrada de dimensiones adecuadas a los datos.</li>
                <li>Una capa de rescalado para conseguir que los valores de la imagen estén entre 0 y 1.</li>
                <li>3 capas convolucionales con tamaño de kernel (5x5) para la primera y (3x3) para las 2 siguientes. Se utilizará padding '<i>same</i>' y activación ReLU. El número de filtros para cada capa convolucional será 16, 32 y 64 respectivamente.</li>
                <li>A cada capa convolucional le sigue una capa de <i>Max Pooling</i></li>
                <li>Una capa de <i>average pooling</i> (GlobalAveragePooling2D) para reducir las dimensiones a un vector de 64 dimensiones.</li>
            </ul></li>
        <li>El clasificador final sigue la estructura del modelo del apartado anterior:
            <ul>
                <li>Una capa completamente conectada de 1024 neuronas y activación ReLU</li>
                <li>Una capa de Dropout (con probabilidad 0.5)</li>
                <li>Una capa de salida completamente conectada correspondiente a la clasificación final cuyo número de neuronas debe ser igual al múmero de clases de la base de datos y con la función de activación adecuada para llevar a cabo esta tarea de clasificación.</li>
            </ul></li>
    </ul>
    
Compilar y entrenar el modelo siguiendo las siguientes indicaciones:
     <ul>
         <li>Utilizar el optimizador Adam con  <i>learning rate</i> de 0.001.</li>
          <li>Entrenar durante 100 épocas utilizando  <i>EarlyStopping</i> con una persistencia de 10 épocas, monitorizando la función de pérdida en el conjunto de validación, y guardando los pesos que mejor resultado hayan obtenido.</li>
         <li>Monitorizar la métrica  <i>accuracy</i> durante entrenamiento y validación.</li>
         <li>Mostrar las gráficas de <i>accuracy</i> y <i>loss</i>. En cada gráfica debe visualizarse la curva de entrenamiento y la de validación.</li>
         <li>Realizar la evaluación del modelo una vez ha finalizado el entrenamiento para mostrar la pérdida y la exactitud final sobre los datos de test.</li>
    </ul>
    Preguntas a responder: ¿Cúal es el número de parámetros a entrenar? ¿y el tiempo de entrenamiento? ¿Qué precisión se obtiene con este modelo? Comenta los resultados y las gráficas de entrenamiento.
</div>

```python
# Definición de la red
inputs = tf.keras.Input(shape=(224, 224, 3))

# Nº de clases/ Nº de neuronas de la capa de salida
num_classes = len(class_names)

# Normalización
x = Rescaling(1./255)(inputs)

# --- Bloque extractor de características ---
# Conv 1
x = Conv2D(16, (5,5), padding='same', activation='relu')(x)
x = MaxPooling2D()(x)

# Conv 2
x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
x = MaxPooling2D()(x)
  
# Conv 3
x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
x = MaxPooling2D()(x)

# Reducción a vector de características
x = GlobalAveragePooling2D()(x)

# --- Clasificador ---
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

outputs = Dense(num_classes, activation='softmax')(x)

# Crear modelo
model_cnn = Model(inputs=inputs, outputs=outputs)

# Ver estructura
model_cnn.summary()
```

Model: "functional_1"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_1 (InputLayer)      │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ rescaling_1 (Rescaling)         │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 224, 224, 16)   │         1,216 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 112, 112, 16)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 112, 112, 32)   │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 56, 56, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 56, 56, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 28, 28, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 64)             │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 1024)           │        66,560 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 1024)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 21)             │        21,525 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 112,437 (439.21 KB)

 Trainable params: 112,437 (439.21 KB)

 Non-trainable params: 0 (0.00 B)

```python
# Compilación de la red
optimizer = Adam(learning_rate=0.001)

model_cnn.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
  
# Detener entreno cuando no haya mejoras en 10 epochs seguidas
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Entrenamiento
start_time = time.time()
  
history_cnn = model_cnn.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stop]
)
  
training_time_cnn = time.time() - start_time
print(f"Tiempo de entrenamiento: {training_time_cnn:.2f} segundos")
```
Epoch 1/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 99s 418ms/step - accuracy: 0.0822 - loss: 2.9110 - val_accuracy: 0.1486 - val_loss: 2.6421 Epoch 2/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 77s 333ms/step - accuracy: 0.2019 - loss: 2.5027 - val_accuracy: 0.2524 - val_loss: 2.3257 Epoch 3/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 75s 327ms/step - accuracy: 0.2999 - loss: 2.2046 - val_accuracy: 0.3376 - val_loss: 2.0659 Epoch 4/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 75s 325ms/step - accuracy: 0.3548 - loss: 2.0047 - val_accuracy: 0.3810 - val_loss: 1.9396 Epoch 5/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 77s 335ms/step - accuracy: 0.3996 - loss: 1.8435 - val_accuracy: 0.4333 - val_loss: 1.7500 Epoch 6/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 79s 321ms/step - accuracy: 0.4276 - loss: 1.7368 - val_accuracy: 0.4471 - val_loss: 1.6957 Epoch 7/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 319ms/step - accuracy: 0.4521 - loss: 1.6564 - val_accuracy: 0.4686 - val_loss: 1.5822 Epoch 8/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 76s 332ms/step - accuracy: 0.4803 - loss: 1.5520 - val_accuracy: 0.5381 - val_loss: 1.4143 Epoch 9/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.5147 - loss: 1.4630 - val_accuracy: 0.5371 - val_loss: 1.3752 Epoch 10/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 318ms/step - accuracy: 0.5332 - loss: 1.3825 - val_accuracy: 0.5576 - val_loss: 1.3154 Epoch 11/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 72s 314ms/step - accuracy: 0.5535 - loss: 1.3212 - val_accuracy: 0.5919 - val_loss: 1.2494 Epoch 12/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 323ms/step - accuracy: 0.5705 - loss: 1.2612 - val_accuracy: 0.5829 - val_loss: 1.2604 Epoch 13/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 75s 324ms/step - accuracy: 0.6031 - loss: 1.1813 - val_accuracy: 0.6124 - val_loss: 1.1739 Epoch 14/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 317ms/step - accuracy: 0.6178 - loss: 1.1286 - val_accuracy: 0.6386 - val_loss: 1.1011 Epoch 15/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 318ms/step - accuracy: 0.6312 - loss: 1.0892 - val_accuracy: 0.6619 - val_loss: 1.0323 Epoch 16/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 317ms/step - accuracy: 0.6448 - loss: 1.0392 - val_accuracy: 0.6567 - val_loss: 1.0518 Epoch 17/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 316ms/step - accuracy: 0.6551 - loss: 1.0110 - val_accuracy: 0.6390 - val_loss: 1.0647 Epoch 18/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 317ms/step - accuracy: 0.6718 - loss: 0.9810 - val_accuracy: 0.6886 - val_loss: 0.9216 Epoch 19/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 318ms/step - accuracy: 0.6844 - loss: 0.9298 - val_accuracy: 0.6667 - val_loss: 1.0260 Epoch 20/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 319ms/step - accuracy: 0.6860 - loss: 0.9072 - val_accuracy: 0.7348 - val_loss: 0.8412 Epoch 21/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 320ms/step - accuracy: 0.7112 - loss: 0.8522 - val_accuracy: 0.7186 - val_loss: 0.8614 Epoch 22/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 317ms/step - accuracy: 0.7053 - loss: 0.8567 - val_accuracy: 0.7229 - val_loss: 0.8433 Epoch 23/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 320ms/step - accuracy: 0.7177 - loss: 0.8255 - val_accuracy: 0.7376 - val_loss: 0.8173 Epoch 24/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 319ms/step - accuracy: 0.7204 - loss: 0.8052 - val_accuracy: 0.7462 - val_loss: 0.7534 Epoch 25/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 323ms/step - accuracy: 0.7404 - loss: 0.7473 - val_accuracy: 0.7319 - val_loss: 0.8137 Epoch 26/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 316ms/step - accuracy: 0.7475 - loss: 0.7338 - val_accuracy: 0.7738 - val_loss: 0.7239 Epoch 27/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 319ms/step - accuracy: 0.7487 - loss: 0.7201 - val_accuracy: 0.7248 - val_loss: 0.8724 Epoch 28/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 318ms/step - accuracy: 0.7483 - loss: 0.7216 - val_accuracy: 0.7671 - val_loss: 0.7065 Epoch 29/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 75s 326ms/step - accuracy: 0.7739 - loss: 0.6767 - val_accuracy: 0.7152 - val_loss: 0.8359 Epoch 30/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.7686 - loss: 0.6865 - val_accuracy: 0.7824 - val_loss: 0.6694 Epoch 31/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 318ms/step - accuracy: 0.7721 - loss: 0.6599 - val_accuracy: 0.7867 - val_loss: 0.6452 Epoch 32/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 318ms/step - accuracy: 0.7833 - loss: 0.6307 - val_accuracy: 0.7790 - val_loss: 0.6732 Epoch 33/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.7793 - loss: 0.6348 - val_accuracy: 0.7952 - val_loss: 0.6428 Epoch 34/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 316ms/step - accuracy: 0.7947 - loss: 0.5945 - val_accuracy: 0.7819 - val_loss: 0.6848 Epoch 35/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.8049 - loss: 0.5845 - val_accuracy: 0.7748 - val_loss: 0.6579 Epoch 36/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 318ms/step - accuracy: 0.8029 - loss: 0.5650 - val_accuracy: 0.7929 - val_loss: 0.6242 Epoch 37/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8059 - loss: 0.5582 - val_accuracy: 0.7967 - val_loss: 0.6345 Epoch 38/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 316ms/step - accuracy: 0.8174 - loss: 0.5443 - val_accuracy: 0.8071 - val_loss: 0.6181 Epoch 39/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8120 - loss: 0.5430 - val_accuracy: 0.8119 - val_loss: 0.5719 Epoch 40/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8188 - loss: 0.5203 - val_accuracy: 0.8243 - val_loss: 0.5828 Epoch 41/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.8254 - loss: 0.5020 - val_accuracy: 0.8081 - val_loss: 0.5746 Epoch 42/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 320ms/step - accuracy: 0.8257 - loss: 0.4972 - val_accuracy: 0.8267 - val_loss: 0.5431 Epoch 43/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8242 - loss: 0.5012 - val_accuracy: 0.8252 - val_loss: 0.5242 Epoch 44/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 319ms/step - accuracy: 0.8322 - loss: 0.4959 - val_accuracy: 0.8014 - val_loss: 0.5885 Epoch 45/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 75s 324ms/step - accuracy: 0.8410 - loss: 0.4648 - val_accuracy: 0.8481 - val_loss: 0.4793 Epoch 46/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 319ms/step - accuracy: 0.8415 - loss: 0.4557 - val_accuracy: 0.8386 - val_loss: 0.5075 Epoch 47/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 75s 325ms/step - accuracy: 0.8593 - loss: 0.4097 - val_accuracy: 0.8176 - val_loss: 0.5735 Epoch 48/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8521 - loss: 0.4149 - val_accuracy: 0.8514 - val_loss: 0.4620 Epoch 49/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 323ms/step - accuracy: 0.8507 - loss: 0.4349 - val_accuracy: 0.8152 - val_loss: 0.5760 Epoch 50/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 320ms/step - accuracy: 0.8618 - loss: 0.4116 - val_accuracy: 0.8124 - val_loss: 0.5708 Epoch 51/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.8612 - loss: 0.3956 - val_accuracy: 0.8505 - val_loss: 0.4977 Epoch 52/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 320ms/step - accuracy: 0.8581 - loss: 0.4090 - val_accuracy: 0.8400 - val_loss: 0.5266 Epoch 53/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 82s 320ms/step - accuracy: 0.8627 - loss: 0.3839 - val_accuracy: 0.8405 - val_loss: 0.5025 Epoch 54/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 323ms/step - accuracy: 0.8653 - loss: 0.3820 - val_accuracy: 0.8510 - val_loss: 0.4741 Epoch 55/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8650 - loss: 0.3906 - val_accuracy: 0.8486 - val_loss: 0.4846 Epoch 56/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.8815 - loss: 0.3527 - val_accuracy: 0.8543 - val_loss: 0.4690 Epoch 57/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 323ms/step - accuracy: 0.8801 - loss: 0.3395 - val_accuracy: 0.8752 - val_loss: 0.4209 Epoch 58/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 320ms/step - accuracy: 0.8797 - loss: 0.3424 - val_accuracy: 0.8543 - val_loss: 0.4977 Epoch 59/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8762 - loss: 0.3511 - val_accuracy: 0.8581 - val_loss: 0.4846 Epoch 60/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 73s 317ms/step - accuracy: 0.8857 - loss: 0.3318 - val_accuracy: 0.8576 - val_loss: 0.4660 Epoch 61/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 323ms/step - accuracy: 0.8834 - loss: 0.3338 - val_accuracy: 0.8671 - val_loss: 0.4572 Epoch 62/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.8876 - loss: 0.3151 - val_accuracy: 0.8495 - val_loss: 0.4895 Epoch 63/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 320ms/step - accuracy: 0.8905 - loss: 0.3028 - val_accuracy: 0.8529 - val_loss: 0.4999 Epoch 64/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8993 - loss: 0.2971 - val_accuracy: 0.8214 - val_loss: 0.6102 Epoch 65/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8860 - loss: 0.3309 - val_accuracy: 0.8657 - val_loss: 0.4406 Epoch 66/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 321ms/step - accuracy: 0.9068 - loss: 0.2718 - val_accuracy: 0.8676 - val_loss: 0.4322 Epoch 67/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 74s 322ms/step - accuracy: 0.8956 - loss: 0.2943 - val_accuracy: 0.8681 - val_loss: 0.4319 Tiempo de entrenamiento: 4990.43 segundos
```python
plot_training_history(history_cnn)
```

![[Evolucion de accuracy y loss_val por epoch_CNN.png]]

```python
test_loss, test_acc = model_cnn.evaluate(test_data, verbose=2)

print(f"\nResultados en TEST:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
```

33/33 - 4s - 136ms/step - accuracy: 0.8638 - loss: 0.4153 Resultados en TEST: Loss: 0.4153 Accuracy: 0.8638

### Número de parámetros a entrenar

 El modelo convolucional implementado presenta un total de **112.437 parámetros entrenables**, tal y como se observa en el resumen de la arquitectura (`model.summary()`). A diferencia del modelo completamente conectado del apartado anterior, en este caso los parámetros se distribuyen entre varias capas convolucionales y las capas densas finales.  

Las capas convolucionales utilizan un número relativamente reducido de parámetros (1.216, 4.640 y 18.496 respectivamente), ya que comparten pesos a lo largo de toda la imagen. Tras el bloque extractor de características, la capa **GlobalAveragePooling2D** reduce la representación a un vector de **64 características**, lo que permite que la capa densa posterior tenga únicamente **66.560 parámetros**, muy por debajo de los más de tres millones de parámetros que tenía la red del apartado anterior.  

En conjunto, este modelo tiene **aproximadamente 28 veces menos parámetros** que la red completamente conectada (3.168.277 frente a 112.437), lo que lo hace mucho más eficiente y mejor adaptado al problema de clasificación de imágenes.

### Tiempo de entrenamiento  

El tiempo total de entrenamiento del modelo ha sido aproximadamente **4990.43 segundos**. Este tiempo es algo superior al del modelo del apartado anterior, ya que las operaciones convolucionales sobre imágenes de tamaño **224×224** implican un mayor coste computacional.

  Además, el entrenamiento se ha detenido antes de alcanzar las 100 épocas gracias al uso de **EarlyStopping**, que monitoriza la pérdida en el conjunto de validación y detiene el entrenamiento cuando esta deja de mejorar durante **10 épocas consecutivas**. Esto permite evitar sobreentrenamiento y reducir el tiempo total de entrenamiento.
### Precisión obtenida  

Tras evaluar el modelo sobre el conjunto de **test**, se obtiene una **accuracy de 0.8638**, lo que significa que el modelo clasifica correctamente aproximadamente el **86.4% de las imágenes**.

Este resultado supone una **mejora muy significativa** respecto al modelo completamente conectado del apartado 2, que alcanzaba únicamente una precisión de **16.8%**. Por tanto, la red convolucional consigue aprender representaciones mucho más útiles de las imágenes para realizar la tarea de clasificación.

### Comentario de las gráficas de entrenamiento

Las gráficas de evolución de **accuracy** y **loss** muestran un proceso de aprendizaje claro y progresivo. Durante las primeras épocas, tanto la precisión de entrenamiento como la de validación aumentan rápidamente, mientras que la función de pérdida disminuye de forma consistente.

A medida que avanza el entrenamiento, las curvas de entrenamiento y validación se mantienen relativamente cercanas, lo que indica una **buena capacidad de generalización** del modelo. Aunque en las últimas épocas se observa una ligera separación entre ambas curvas (con mayor precisión en entrenamiento), el comportamiento general sugiere que el modelo no está sufriendo un sobreajuste severo.

En general, las gráficas reflejan que el modelo logra aprender representaciones cada vez más discriminativas de las imágenes gracias al bloque convolucional.

### Interpretación de los resultados

La mejora obtenida respecto al modelo del apartado anterior se debe principalmente al uso de **capas convolucionales**, que permiten explotar la estructura espacial de las imágenes. Estas capas son capaces de detectar patrones locales como bordes, texturas o formas, y combinarlos progresivamente para generar representaciones más abstractas útiles para la clasificación.

Además, el uso de **MaxPooling** reduce progresivamente las dimensiones espaciales, mientras que **GlobalAveragePooling2D** resume las características aprendidas en un vector compacto que alimenta al clasificador final.

En comparación con la red completamente conectada del apartado 2, este modelo es **mucho más eficiente en número de parámetros y significativamente más preciso**, lo que confirma que las arquitecturas convolucionales son mucho más adecuadas para tareas de visión por computador.

## 4. Autoencoders (2 puntos)

En el apartado anterior hemos podido observar que, utilizando el tipo de redes adecuado, podemos obtener mejores resultados entrenando un número de parámetros muy inferior. Esto es debido a que las CNN consiguen extraer las características principales de los datos proporcionados (imágenes en nuestro caso).

En este apartado vamos a observar esta capacidad desde otro punto de vista: el de **codificar y decodificar una imagen**.

Para ello diseñaremos un autoencoder que sea capaz de reducir el tamaño de los datos de entrada pero captando las características principales de las imágenes para poder llevar a cabo una buena reconstrucción de las mismas.

Empezaremos rescalando externamente los datos que vamos a utilizar, para que estén en el rango (0,1), en lugar de realizarlo dentro de la red como hemos hecho en el apartado anterior:

```python
# data rescalling
normalization_layer = Rescaling(1./255)
  
normalized_train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
normalized_val_data = val_data.map(lambda x, y: (normalization_layer(x), y))
```

Además, en un autoencoder, en lugar de utilizar las etiquetas como objetivo (que es lo que se utiliza en un problema de clasificación), deben ser las propias imágenes las que se utilicen como objetivo de la red. Por tanto, crearemos una nueva base de datos de entrenamiento y validación donde son las propias imágenes las que hagan de etiquetas:

```python
train_data_auto = normalized_train_data.map(lambda x, y: (x, x))
val_data_auto = normalized_val_data.map(lambda x, y: (x, x))
```

Comprobamos la estructura de la nueva base de datos:

```pyton
image_batch, label_batch = iter(train_data_auto).get_next()
print("Las dimensiones de un batch de imágenes es: {}".format(image_batch.shape))
print("Las dimensiones de un batch de etiquetas es: {}".format(label_batch.shape))
```

Las dimensiones de un batch de imágenes es: (32, 224, 224, 3) Las dimensiones de un batch de etiquetas es: (32, 224, 224, 3)

Y que los datos tienen el rango dinámico adecuado:

```python
first_image = image_batch[0]
print("En la primera imagen los valores mínimo y máximo son {} y {}, respectivamente"
      .format(np.min(first_image),np.max(first_image)))
```

En la primera imagen los valores mínimo y máximo son 0.0 y 0.8117647767066956, respectivamente

### 4.1. Diseño y entrenamiento del autoencoder

Una vez ya tenemos los datos en el formato adecuado vamos a diseñar el autoencoder. Para ello utilizaremos el bloque extractor del apartado anterior como codificador (exceptuando la capa de GlobalAveragePooling, ya que necesitamos preservar la estructura espacial) y reflejaremos su estructura en el decodificador utilizando las capas [Conv2DTranspose](https://keras.io/api/layers/convolution_layers/convolution2d_transpose/) y [UpSampling2D](https://keras.io/api/layers/reshaping_layers/up_sampling2d/) de keras.

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
    <strong>Ejercicio [1 punto]:</strong> A partir del modelo <strong>funcional</strong> de keras (y la clase <code>Model()</code>), implementa un autoencoder con las siguientes características:
    <ul>
        <li>El bloque codificador debe tener:
            <ul>
                <li>Una capa de entrada de dimensiones adecuadas a los datos.</li>
                <li>3 capas convolucionales con tamaño de kernel (5x5) para la primera y (3x3) para las 2 siguientes. Se utilizará padding '<i>same</i>' y activación ReLU. El número de filtros para cada capa convolucional será 16, 32 y 64 respectivamente.</li>
                <li>A cada capa convolucional le sigue una capa de <i>Max Pooling</i></li>
            </ul></li>
        <li>El bloque decodificador debe tener:
            <ul>
                <li>3 capas convolucionales con tamaño de kernel (3x3) para las 2 primeras y (5x5) para la última. Se utilizará padding '<i>same</i>' y activación ReLU. El número de filtros para cada capa convolucional será 64, 32 y 16, respectivamente</li>
                <li>A cada capa convolucional le sigue una capa de <i>UpSampling2D</i></li>
                <li>Una última capa convolucional con tamaño de kernel (3x3), con 3 filtros y activación sigmoide.</li>
            </ul></li>
    </ul>
    
Compilar y entrenar el modelo siguiendo las siguientes indicaciones:
     <ul>
         <li>Utilizar el optimizador Adam con  <i>learning rate</i> de 0.001.</li>
         <li>Utilizar como función de pérdida el error cuadrático medio.</li>
         <li>Entrenar durante 100 épocas utilizando  <i>EarlyStopping</i> con una persistencia de 10 épocas, monitorizando la función de pérdida en el conjunto de validación, y guardando los pesos que mejor resultado hayan obtenido.</li>
         <li>Monitorizar la pérdida durante entrenamiento y validación.</li>
         <li>Mostrar las gráficas del <i>loss</i> (la curva de entrenamiento y la de validación).</li>
    </ul>
</div>

```python
# Definición del modelo Autoencoder
inputs = tf.keras.Input(shape=(224,224,3))

# ----------- CODIFICADOR (ENCODER) -----------
# Conv1
x = Conv2D(16, (5,5), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2,2), padding='same')(x)

# Conv2
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)

# Conv3
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

# ----------- ECODIFICADOR (DECODER) -----------
# Conv1 decoder
x = Conv2DTranspose(64, (3,3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2,2))(x)

# Conv2 decoder
x = Conv2DTranspose(32, (3,3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)

# Conv3 decoder
x = Conv2DTranspose(16, (5,5), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
  
# Capa de salida
outputs = Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)

# Crear modelo
autoencoder = Model(inputs, outputs)

# Ver estructura
autoencoder.summary()
```

Model: "functional_2"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)      │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 224, 224, 16)   │         1,216 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 112, 112, 16)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 112, 112, 32)   │         4,640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_4 (MaxPooling2D)  │ (None, 56, 56, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (Conv2D)               │ (None, 56, 56, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_5 (MaxPooling2D)  │ (None, 28, 28, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose                │ (None, 28, 28, 64)     │        36,928 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ up_sampling2d (UpSampling2D)    │ (None, 56, 56, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_1              │ (None, 56, 56, 32)     │        18,464 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ up_sampling2d_1 (UpSampling2D)  │ (None, 112, 112, 32)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_transpose_2              │ (None, 112, 112, 16)   │        12,816 │
│ (Conv2DTranspose)               │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ up_sampling2d_2 (UpSampling2D)  │ (None, 224, 224, 16)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_6 (Conv2D)               │ (None, 224, 224, 3)    │           435 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 92,995 (363.26 KB)

 Trainable params: 92,995 (363.26 KB)

 Non-trainable params: 0 (0.00 B)
 
```python
optimizer = Adam(learning_rate=0.001)
  
autoencoder.compile(
    optimizer=optimizer,
    loss='mse'
)
  
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

start_time = time.time()
  
history_auto = autoencoder.fit(
    train_data_auto,
    validation_data=val_data_auto,
    epochs=100,
    callbacks=[early_stop]
)

training_time_auto = time.time() - start_time
print(f"Tiempo de entrenamiento: {training_time_auto:.2f} segundos")
```

Epoch 1/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 181s 765ms/step - loss: 0.0161 - val_loss: 0.0070 Epoch 2/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 171s 742ms/step - loss: 0.0062 - val_loss: 0.0050 Epoch 3/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 729ms/step - loss: 0.0050 - val_loss: 0.0056 Epoch 4/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 730ms/step - loss: 0.0045 - val_loss: 0.0053 Epoch 5/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 727ms/step - loss: 0.0042 - val_loss: 0.0039 Epoch 6/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 726ms/step - loss: 0.0039 - val_loss: 0.0036 Epoch 7/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 723ms/step - loss: 0.0037 - val_loss: 0.0034 Epoch 8/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 721ms/step - loss: 0.0037 - val_loss: 0.0035 Epoch 9/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 718ms/step - loss: 0.0034 - val_loss: 0.0032 Epoch 10/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 720ms/step - loss: 0.0033 - val_loss: 0.0030 Epoch 11/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 719ms/step - loss: 0.0032 - val_loss: 0.0031 Epoch 12/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 165s 715ms/step - loss: 0.0032 - val_loss: 0.0029 Epoch 13/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 719ms/step - loss: 0.0030 - val_loss: 0.0028 Epoch 14/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 719ms/step - loss: 0.0030 - val_loss: 0.0028 Epoch 15/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 722ms/step - loss: 0.0029 - val_loss: 0.0027 Epoch 16/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 722ms/step - loss: 0.0028 - val_loss: 0.0026 Epoch 17/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 724ms/step - loss: 0.0028 - val_loss: 0.0025 Epoch 18/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 727ms/step - loss: 0.0026 - val_loss: 0.0025 Epoch 19/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 720ms/step - loss: 0.0027 - val_loss: 0.0025 Epoch 20/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 724ms/step - loss: 0.0025 - val_loss: 0.0024 Epoch 21/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 726ms/step - loss: 0.0026 - val_loss: 0.0023 Epoch 22/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 723ms/step - loss: 0.0025 - val_loss: 0.0025 Epoch 23/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 724ms/step - loss: 0.0025 - val_loss: 0.0024 Epoch 24/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 721ms/step - loss: 0.0024 - val_loss: 0.0029 Epoch 25/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 719ms/step - loss: 0.0024 - val_loss: 0.0023 Epoch 26/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 720ms/step - loss: 0.0023 - val_loss: 0.0025 Epoch 27/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 724ms/step - loss: 0.0023 - val_loss: 0.0024 Epoch 28/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 724ms/step - loss: 0.0023 - val_loss: 0.0022 Epoch 29/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 723ms/step - loss: 0.0022 - val_loss: 0.0021 Epoch 30/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 720ms/step - loss: 0.0022 - val_loss: 0.0020 Epoch 31/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 723ms/step - loss: 0.0022 - val_loss: 0.0022 Epoch 32/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 725ms/step - loss: 0.0022 - val_loss: 0.0025 Epoch 33/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 727ms/step - loss: 0.0021 - val_loss: 0.0020 Epoch 34/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 170s 736ms/step - loss: 0.0021 - val_loss: 0.0021 Epoch 35/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 723ms/step - loss: 0.0021 - val_loss: 0.0021 Epoch 36/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 724ms/step - loss: 0.0020 - val_loss: 0.0019 Epoch 37/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 723ms/step - loss: 0.0021 - val_loss: 0.0019 Epoch 38/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 725ms/step - loss: 0.0020 - val_loss: 0.0019 Epoch 39/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 729ms/step - loss: 0.0020 - val_loss: 0.0018 Epoch 40/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 169s 732ms/step - loss: 0.0020 - val_loss: 0.0019 Epoch 41/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 181s 786ms/step - loss: 0.0020 - val_loss: 0.0019 Epoch 42/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 730ms/step - loss: 0.0020 - val_loss: 0.0018 Epoch 43/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 730ms/step - loss: 0.0020 - val_loss: 0.0018 Epoch 44/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 728ms/step - loss: 0.0019 - val_loss: 0.0018 Epoch 45/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 729ms/step - loss: 0.0019 - val_loss: 0.0019 Epoch 46/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 727ms/step - loss: 0.0019 - val_loss: 0.0018 Epoch 47/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 729ms/step - loss: 0.0019 - val_loss: 0.0019 Epoch 48/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 723ms/step - loss: 0.0019 - val_loss: 0.0024 Epoch 49/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 723ms/step - loss: 0.0019 - val_loss: 0.0021 Epoch 50/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 728ms/step - loss: 0.0018 - val_loss: 0.0017 Epoch 51/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 726ms/step - loss: 0.0018 - val_loss: 0.0017 Epoch 52/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 725ms/step - loss: 0.0018 - val_loss: 0.0017 Epoch 53/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 728ms/step - loss: 0.0019 - val_loss: 0.0017 Epoch 54/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 725ms/step - loss: 0.0018 - val_loss: 0.0016 Epoch 55/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 166s 721ms/step - loss: 0.0017 - val_loss: 0.0018 Epoch 56/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 727ms/step - loss: 0.0018 - val_loss: 0.0031 Epoch 57/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 175s 758ms/step - loss: 0.0018 - val_loss: 0.0016 Epoch 58/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 176s 764ms/step - loss: 0.0017 - val_loss: 0.0016 Epoch 59/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 725ms/step - loss: 0.0017 - val_loss: 0.0017 Epoch 60/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 724ms/step - loss: 0.0027 - val_loss: 0.0020 Epoch 61/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 726ms/step - loss: 0.0019 - val_loss: 0.0019 Epoch 62/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 728ms/step - loss: 0.0018 - val_loss: 0.0020 Epoch 63/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 726ms/step - loss: 0.0018 - val_loss: 0.0018 Epoch 64/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 167s 726ms/step - loss: 0.0018 - val_loss: 0.0017 Epoch 65/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 204s 734ms/step - loss: 0.0018 - val_loss: 0.0017 Epoch 66/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 728ms/step - loss: 0.0017 - val_loss: 0.0016 Epoch 67/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 168s 728ms/step - loss: 0.0017 - val_loss: 0.0018 Tiempo de entrenamiento: 11274.21 segundos

```python
plt.figure(figsize=(8,5))
plt.plot(history_auto.history['loss'], label='Train Loss')
plt.plot(history_auto.history['val_loss'], label='Validation Loss')
plt.title("Evolución del error de reconstrucción")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.show()
```

![[Evolucion de accuracy y loss_val por epoch_autoencoders.png]]

### 4.2. Evaluación del autoencoder

La evaluación del modelo obtenido puede hacerse en este caso tanto de forma cuantitativa (calculando el MSE entre las imágenes originales y reconstruídas del conjunto de test) como cualitativa (mostrando imágenes originales y reconstruídas).

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
    <strong>Ejercicio [1 punto]:</strong> Realizar las siguientes operaciones para evaluar las prestaciones del modelo obtenido:
    <ul>
        <li>Partiendo del conjunto de test obtenido en el primer apartado de la practica:
            <ul>
                <li>Llevar a cabo el reescalado de los datos utilizando la capa <code>normalization_layer</code> tal y como se ha hecho con los conjuntos de entrenamiento y test al inicio de este bloque.</li>
                <li>Generar el conjunto de datos <code>test_data_auto</code> en el que las imágenes sean también el objetivo y substituyan a las etiquetas. </li>
            </ul></li>
        <li>Realizar la evaluación del modelo una vez ha finalizado el entrenamiento para mostrar la pérdida final a partir de los datos de test.</li>
        <li>Imprimir por pantalla 4 parejas de imágenes (original y reconstruída). Nota: a la hora de representar las imágenes correctamente, recordad que su rango dinámico deben ser números enteros entre 0 y 255.</li>
    </ul>
    Preguntas: ¿Consideras que la reconstrucción es adecuada? ¿Qué <i>ratio</i> de compresión se consigue con este autoencoder? Consideramos como <i>ratio</i> de compresión la relación entre el tamaño original de la imagen (224,224,3) y el de la representación más perqueña que llega a hacer el codificador (tamaño de la salida de su última capa).
</div>

```python
# Normalización de los datos
normalized_test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

# Crear dataset para autoencoder (imagen -> imagen)
test_data_auto = normalized_test_data.map(lambda x, y: (x, x))

# Evaluación del modelo (cuantitativa)
test_loss = autoencoder.evaluate(test_data_auto, verbose=2)
  
print("\nResultado en TEST:")
print(f"MSE (loss): {test_loss:.6f}")
```

33/33 - 12s - 375ms/step - loss: 0.0016 Resultado en TEST: MSE (loss): 0.001627

```python
# Visualización de los datos
# Evaluación del modelo (cualitativa)
# # Obtener un batch de imágenes
image_batch, _ = iter(test_data_auto).get_next()
  
# Generar reconstrucciones
reconstructed_images = autoencoder.predict(image_batch)

# Mostrar 4 ejemplos
plt.figure(figsize=(10,6))

for i in range(4):
    # Imagen original
    plt.subplot(2,4,i+1)
    plt.imshow((image_batch[i].numpy()*255).astype("uint8"))
    plt.title("Original")
    plt.axis("off")
    
    # Imagen reconstruida
    plt.subplot(2,4,i+5)
    plt.imshow((reconstructed_images[i]*255).astype("uint8"))
    plt.title("Reconstruida")
    plt.axis("off")
plt.tight_layout()
plt.show()
```

![[OriginalvsReconstruida_autoencoders.png]]
### ### Número de parámetros a entrenar

El autoencoder implementado presenta un total de **92.995 parámetros entrenables**, tal y como se puede observar en el resumen de la arquitectura del modelo (`autoencoder.summary()`).
  
Estos parámetros se distribuyen entre las distintas capas convolucionales del **codificador** y del **decodificador**. En el codificador, las capas **Conv2D** con 16, 32 y 64 filtros se encargan de extraer progresivamente características de la imagen mientras se reduce su resolución mediante **MaxPooling2D**. Gracias al uso de convoluciones, el número de parámetros se mantiene relativamente bajo, ya que los filtros **comparten pesos a lo largo de toda la imagen**.

En el decodificador, las capas **Conv2DTranspose** y **UpSampling2D** realizan el proceso inverso, reconstruyendo progresivamente la resolución de la imagen hasta recuperar el tamaño original **224×224×3**. Finalmente, una capa convolucional con **3 filtros y activación sigmoide** genera la imagen reconstruida.

En conjunto, el modelo tiene un número moderado de parámetros, lo que permite que pueda **aprender representaciones comprimidas de las imágenes sin requerir una arquitectura excesivamente compleja**.

### Tiempo de entrenamiento
  
El tiempo total de entrenamiento del modelo ha sido aproximadamente **11.274 segundos**. Este tiempo es relativamente elevado debido principalmente al **tamaño de las imágenes de entrada (224×224)** y a que el autoencoder debe procesarlas completamente tanto en la fase de **codificación como en la de decodificación**.

Además, a diferencia de los modelos de clasificación, el autoencoder intenta **reconstruir todos los píxeles de la imagen**, lo que implica un mayor número de operaciones durante cada iteración de entrenamiento.

El uso de **EarlyStopping** permite detener el entrenamiento automáticamente cuando la pérdida en el conjunto de validación deja de mejorar durante varias épocas consecutivas, evitando así continuar entrenando innecesariamente y reduciendo el tiempo total de entrenamiento.

### Evaluación cuantitativa (MSE)

Para evaluar el rendimiento del autoencoder se ha utilizado la **función de pérdida Mean Squared Error (MSE)** entre las imágenes originales y las imágenes reconstruidas del conjunto de **test**.

Tras realizar la evaluación del modelo con los datos de prueba, se obtiene un **MSE de 0.001627**, lo que indica que el error medio entre los píxeles originales y reconstruidos es bastante pequeño.

Este valor sugiere que el modelo ha aprendido correctamente a **capturar las características más relevantes de las imágenes** y es capaz de reconstruirlas con bastante precisión.

### Evaluación cualitativa de las reconstrucciones  

Además de la evaluación numérica, se han representado **cuatro parejas de imágenes** formadas por la imagen original y su correspondiente reconstrucción generada por el autoencoder.

Al comparar visualmente ambas imágenes se observa que la **estructura general, las formas principales y los colores** se mantienen correctamente en las reconstrucciones. Sin embargo, también se aprecia una **ligera pérdida de detalle**, ya que las imágenes reconstruidas aparecen algo más suavizadas o difuminadas.

Este comportamiento es habitual en los autoencoders, ya que durante el proceso de compresión el modelo intenta conservar únicamente la **información más relevante de la imagen**, perdiéndose algunos detalles finos o texturas.  

En general, puede considerarse que la **reconstrucción es adecuada**, ya que las imágenes reconstruidas siguen siendo claramente reconocibles y mantienen la mayor parte de su contenido visual.

### Ratio de compresión del autoencoder

El **ratio de compresión** se calcula comparando el tamaño de la imagen original con el tamaño de la representación más pequeña generada por el codificador.

La imagen original tiene dimensiones: [224x224x3] = 150528 valores de píxel.  

Tras pasar por el codificador, la representación más comprimida tiene dimensiones: [28x28x64] = valores de pixel.

Por tanto, el **ratio de compresión** aproximado es: 150528/50176 = 3.

Esto significa que el autoencoder consigue representar la imagen utilizando aproximadamente **tres veces menos información que la imagen original**, manteniendo aun así suficiente información para reconstruirla de forma adecuada.

### Interpretación de los resultados

Los resultados obtenidos muestran que el autoencoder es capaz de **aprender representaciones comprimidas de las imágenes preservando la mayor parte de su información visual**.  

El bajo valor de MSE indica que la reconstrucción es bastante precisa, mientras que la comparación visual confirma que las imágenes reconstruidas conservan correctamente su estructura principal.

Aunque existe una ligera pérdida de detalle fino, esto es normal en modelos de compresión, ya que el objetivo del autoencoder es **capturar las características más importantes de la imagen en un espacio latente más reducido**.  

En conjunto, el modelo consigue un **equilibrio razonable entre compresión y calidad de reconstrucción**, lo que demuestra la utilidad de los autoencoders para tareas como **reducción de dimensionalidad, compresión de imágenes o extracción automática de características**.

## 5. Transfer learning con modelos eficientes: EfficientNetB0 (2 puntos)

Las redes neuronales convolucionales profundas nos brindan la posibilidad de mejorar la capacidad de aprendizaje de un modelo. Arquitecturas clásicas como VGG16 fueron pioneras, pero hoy en día existen familias de modelos mucho más optimizadas, como **EfficientNet**, que logran mejores precisiones con muchos menos parámetros.

### 5.1. Transfer Learning
En este apartado, aplicaremos [transfer learning](https://keras.io/guides/transfer_learning/) utilizando el modelo [EfficientNetB0](https://keras.io/api/applications/efficientnet/#efficientnetb0-function) preentrenado en [Imagenet](http://www.image-net.org/). Lo adaptaremos para clasificar las 21 categorías de nuestra base de datos. Una gran ventaja de EfficientNet en Keras es que **ya incluye la capa de preprocesamiento (reescalado) internamente**, por lo que podemos pasarle directamente nuestras imágenes en el rango [0, 255].

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio[1 punto]:</strong> Implementa una red siguiendo los siguientes pasos:
    <ul>
        <li>Partir del modelo EfficientNetB0 con los pesos entrenados en Imagenet (sin la parte superior de clasificación, <code>include_top=False</code>) y congelarlos.</li>
        <li>Añadir una capa <code>GlobalAveragePooling2D</code> a la salida del modelo base.</li>
        <li>Añadir una capa densa de 50 neuronas con activación ReLU, seguida de la capa de salida con el número de neuronas adecuado para llevar a cabo la clasificación y la función de activación correspondiente.</li>
    </ul>
Compilar y entrenar el modelo siguiendo las siguientes indicaciones:
     <ul>
         <li>Utilizar el optimizador Adam con <i>learning rate</i> de 0.0001.</li>
         <li>Entrenar durante 100 épocas utilizando <i>EarlyStopping</i> con una persistencia de 10 épocas, monitorizando la <i>accuracy</i> en validación y guardando los mejores pesos.</li>
         <li>Mostrar las gráficas de accuracy y loss (entrenamiento y validación).</li>
         <li>Evaluar el modelo final sobre los datos de test.</li>
    </ul>
    Preguntas a responder: ¿Cuál es el número de parámetros a entrenar? ¿Qué precisión se obtiene comparado con la red convolucional simple del apartado 3? Comenta los resultados y las gráficas de entrenamiento.
</div>

```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
  
# Cargar el modelo base preentrenado
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
  
base_model.trainable = False

# Definición de la red
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(50, activation='relu')(x)
outputs = layers.Dense(21, activation='softmax')(x)
  
model_effnet = models.Model(inputs=base_model.input, outputs=outputs)

# Compilación de la red
model_effnet.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
  
# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# Entrenamiento de la red
start = time.time()
  
history_effnet = model_effnet.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[early_stop]
)
  
training_time = time.time() - start
print("Tiempo de entrenamiento:", training_time)
```

Epoch 1/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 391s 2s/step - accuracy: 0.8057 - loss: 1.1053 - val_accuracy: 0.8514 - val_loss: 0.8096 Epoch 2/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 304s 1s/step - accuracy: 0.8733 - loss: 0.6695 - val_accuracy: 0.8824 - val_loss: 0.5574 Epoch 3/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 301s 1s/step - accuracy: 0.9011 - loss: 0.4840 - val_accuracy: 0.9019 - val_loss: 0.4346 Epoch 4/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 301s 1s/step - accuracy: 0.9156 - loss: 0.3868 - val_accuracy: 0.9162 - val_loss: 0.3605 Epoch 5/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 294s 1s/step - accuracy: 0.9282 - loss: 0.3237 - val_accuracy: 0.9276 - val_loss: 0.3106 Epoch 6/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 296s 1s/step - accuracy: 0.9384 - loss: 0.2784 - val_accuracy: 0.9343 - val_loss: 0.2752 Epoch 7/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 292s 1s/step - accuracy: 0.9461 - loss: 0.2437 - val_accuracy: 0.9414 - val_loss: 0.2472 Epoch 8/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 293s 1s/step - accuracy: 0.9498 - loss: 0.2188 - val_accuracy: 0.9462 - val_loss: 0.2259 Epoch 9/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 296s 1s/step - accuracy: 0.9593 - loss: 0.1939 - val_accuracy: 0.9510 - val_loss: 0.2064 Epoch 10/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 296s 1s/step - accuracy: 0.9638 - loss: 0.1741 - val_accuracy: 0.9552 - val_loss: 0.1927 Epoch 11/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 292s 1s/step - accuracy: 0.9644 - loss: 0.1629 - val_accuracy: 0.9590 - val_loss: 0.1784 Epoch 12/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 295s 1s/step - accuracy: 0.9676 - loss: 0.1500 - val_accuracy: 0.9600 - val_loss: 0.1683 Epoch 13/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 295s 1s/step - accuracy: 0.9722 - loss: 0.1361 - val_accuracy: 0.9610 - val_loss: 0.1581 Epoch 14/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 308s 1s/step - accuracy: 0.9739 - loss: 0.1264 - val_accuracy: 0.9614 - val_loss: 0.1506 Epoch 15/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 313s 1s/step - accuracy: 0.9773 - loss: 0.1178 - val_accuracy: 0.9633 - val_loss: 0.1440 Epoch 16/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 311s 1s/step - accuracy: 0.9789 - loss: 0.1113 - val_accuracy: 0.9624 - val_loss: 0.1360 Epoch 17/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 309s 1s/step - accuracy: 0.9812 - loss: 0.1016 - val_accuracy: 0.9633 - val_loss: 0.1308 Epoch 18/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 307s 1s/step - accuracy: 0.9826 - loss: 0.0937 - val_accuracy: 0.9667 - val_loss: 0.1245 Epoch 19/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 305s 1s/step - accuracy: 0.9857 - loss: 0.0865 - val_accuracy: 0.9671 - val_loss: 0.1190 Epoch 20/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 307s 1s/step - accuracy: 0.9839 - loss: 0.0851 - val_accuracy: 0.9676 - val_loss: 0.1159 Epoch 21/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 304s 1s/step - accuracy: 0.9880 - loss: 0.0771 - val_accuracy: 0.9690 - val_loss: 0.1108 Epoch 22/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 332s 1s/step - accuracy: 0.9880 - loss: 0.0730 - val_accuracy: 0.9690 - val_loss: 0.1073 Epoch 23/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 308s 1s/step - accuracy: 0.9876 - loss: 0.0691 - val_accuracy: 0.9676 - val_loss: 0.1035 Epoch 24/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 305s 1s/step - accuracy: 0.9899 - loss: 0.0635 - val_accuracy: 0.9710 - val_loss: 0.1006 Epoch 25/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 308s 1s/step - accuracy: 0.9906 - loss: 0.0603 - val_accuracy: 0.9695 - val_loss: 0.0972 Epoch 26/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 301s 1s/step - accuracy: 0.9899 - loss: 0.0582 - val_accuracy: 0.9710 - val_loss: 0.0951 Epoch 27/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 300s 1s/step - accuracy: 0.9918 - loss: 0.0530 - val_accuracy: 0.9719 - val_loss: 0.0926 Epoch 28/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 299s 1s/step - accuracy: 0.9939 - loss: 0.0504 - val_accuracy: 0.9714 - val_loss: 0.0905 Epoch 29/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 302s 1s/step - accuracy: 0.9939 - loss: 0.0480 - val_accuracy: 0.9733 - val_loss: 0.0884 Epoch 30/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 301s 1s/step - accuracy: 0.9943 - loss: 0.0448 - val_accuracy: 0.9748 - val_loss: 0.0853 Epoch 31/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 305s 1s/step - accuracy: 0.9941 - loss: 0.0449 - val_accuracy: 0.9733 - val_loss: 0.0859 Epoch 32/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 297s 1s/step - accuracy: 0.9927 - loss: 0.0431 - val_accuracy: 0.9743 - val_loss: 0.0813 Epoch 33/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 307s 1s/step - accuracy: 0.9939 - loss: 0.0412 - val_accuracy: 0.9757 - val_loss: 0.0787 Epoch 34/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 302s 1s/step - accuracy: 0.9952 - loss: 0.0370 - val_accuracy: 0.9757 - val_loss: 0.0764 Epoch 35/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 302s 1s/step - accuracy: 0.9954 - loss: 0.0374 - val_accuracy: 0.9771 - val_loss: 0.0757 Epoch 36/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 304s 1s/step - accuracy: 0.9965 - loss: 0.0325 - val_accuracy: 0.9757 - val_loss: 0.0762 Epoch 37/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 298s 1s/step - accuracy: 0.9973 - loss: 0.0310 - val_accuracy: 0.9757 - val_loss: 0.0737 Epoch 38/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 305s 1s/step - accuracy: 0.9966 - loss: 0.0319 - val_accuracy: 0.9752 - val_loss: 0.0716 Epoch 39/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 301s 1s/step - accuracy: 0.9981 - loss: 0.0275 - val_accuracy: 0.9786 - val_loss: 0.0700 Epoch 40/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 301s 1s/step - accuracy: 0.9976 - loss: 0.0269 - val_accuracy: 0.9767 - val_loss: 0.0716 Epoch 41/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 311s 1s/step - accuracy: 0.9967 - loss: 0.0271 - val_accuracy: 0.9771 - val_loss: 0.0690 Epoch 42/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 303s 1s/step - accuracy: 0.9982 - loss: 0.0235 - val_accuracy: 0.9776 - val_loss: 0.0683 Epoch 43/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 309s 1s/step - accuracy: 0.9974 - loss: 0.0246 - val_accuracy: 0.9776 - val_loss: 0.0678 Epoch 44/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 307s 1s/step - accuracy: 0.9976 - loss: 0.0238 - val_accuracy: 0.9776 - val_loss: 0.0674 Epoch 45/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 305s 1s/step - accuracy: 0.9981 - loss: 0.0231 - val_accuracy: 0.9776 - val_loss: 0.0652 Epoch 46/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 306s 1s/step - accuracy: 0.9978 - loss: 0.0206 - val_accuracy: 0.9781 - val_loss: 0.0640 Epoch 47/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 304s 1s/step - accuracy: 0.9981 - loss: 0.0198 - val_accuracy: 0.9790 - val_loss: 0.0621 Epoch 48/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 305s 1s/step - accuracy: 0.9978 - loss: 0.0201 - val_accuracy: 0.9781 - val_loss: 0.0625 Epoch 49/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 306s 1s/step - accuracy: 0.9980 - loss: 0.0199 - val_accuracy: 0.9800 - val_loss: 0.0610 Epoch 50/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 309s 1s/step - accuracy: 0.9976 - loss: 0.0201 - val_accuracy: 0.9800 - val_loss: 0.0623 Epoch 51/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 306s 1s/step - accuracy: 0.9985 - loss: 0.0174 - val_accuracy: 0.9790 - val_loss: 0.0619 Epoch 52/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 307s 1s/step - accuracy: 0.9981 - loss: 0.0174 - val_accuracy: 0.9805 - val_loss: 0.0598 Epoch 53/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 300s 1s/step - accuracy: 0.9985 - loss: 0.0161 - val_accuracy: 0.9819 - val_loss: 0.0572 Epoch 54/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 294s 1s/step - accuracy: 0.9995 - loss: 0.0142 - val_accuracy: 0.9790 - val_loss: 0.0576 Epoch 55/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 297s 1s/step - accuracy: 0.9992 - loss: 0.0151 - val_accuracy: 0.9776 - val_loss: 0.0592 Epoch 56/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 303s 1s/step - accuracy: 0.9977 - loss: 0.0166 - val_accuracy: 0.9810 - val_loss: 0.0568 Epoch 57/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 318s 1s/step - accuracy: 0.9985 - loss: 0.0147 - val_accuracy: 0.9810 - val_loss: 0.0574 Epoch 58/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 312s 1s/step - accuracy: 0.9986 - loss: 0.0140 - val_accuracy: 0.9795 - val_loss: 0.0590 Epoch 59/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 317s 1s/step - accuracy: 0.9985 - loss: 0.0134 - val_accuracy: 0.9819 - val_loss: 0.0561 Epoch 60/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 309s 1s/step - accuracy: 0.9986 - loss: 0.0132 - val_accuracy: 0.9819 - val_loss: 0.0553 Epoch 61/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 311s 1s/step - accuracy: 0.9992 - loss: 0.0111 - val_accuracy: 0.9819 - val_loss: 0.0547 Epoch 62/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 315s 1s/step - accuracy: 0.9986 - loss: 0.0128 - val_accuracy: 0.9829 - val_loss: 0.0541 Epoch 63/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 315s 1s/step - accuracy: 0.9982 - loss: 0.0121 - val_accuracy: 0.9800 - val_loss: 0.0567 Epoch 64/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 307s 1s/step - accuracy: 0.9992 - loss: 0.0109 - val_accuracy: 0.9810 - val_loss: 0.0564 Epoch 65/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 306s 1s/step - accuracy: 0.9989 - loss: 0.0108 - val_accuracy: 0.9824 - val_loss: 0.0540 Epoch 66/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 313s 1s/step - accuracy: 0.9992 - loss: 0.0104 - val_accuracy: 0.9819 - val_loss: 0.0541 Epoch 67/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 313s 1s/step - accuracy: 0.9996 - loss: 0.0104 - val_accuracy: 0.9829 - val_loss: 0.0524 Epoch 68/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 318s 1s/step - accuracy: 0.9990 - loss: 0.0104 - val_accuracy: 0.9814 - val_loss: 0.0537 Epoch 69/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 317s 1s/step - accuracy: 0.9999 - loss: 0.0084 - val_accuracy: 0.9838 - val_loss: 0.0516 Epoch 70/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 309s 1s/step - accuracy: 0.9992 - loss: 0.0083 - val_accuracy: 0.9833 - val_loss: 0.0508 Epoch 71/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 313s 1s/step - accuracy: 0.9995 - loss: 0.0082 - val_accuracy: 0.9819 - val_loss: 0.0525 Epoch 72/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 316s 1s/step - accuracy: 0.9995 - loss: 0.0083 - val_accuracy: 0.9810 - val_loss: 0.0524 Epoch 73/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 319s 1s/step - accuracy: 0.9992 - loss: 0.0088 - val_accuracy: 0.9810 - val_loss: 0.0514 Epoch 74/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 314s 1s/step - accuracy: 0.9990 - loss: 0.0088 - val_accuracy: 0.9833 - val_loss: 0.0515 Epoch 75/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 312s 1s/step - accuracy: 0.9988 - loss: 0.0088 - val_accuracy: 0.9824 - val_loss: 0.0508 Epoch 76/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 319s 1s/step - accuracy: 0.9996 - loss: 0.0077 - val_accuracy: 0.9814 - val_loss: 0.0507 Epoch 77/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 311s 1s/step - accuracy: 0.9986 - loss: 0.0085 - val_accuracy: 0.9819 - val_loss: 0.0483 Epoch 78/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 316s 1s/step - accuracy: 0.9997 - loss: 0.0072 - val_accuracy: 0.9829 - val_loss: 0.0503 Epoch 79/100 230/230 ━━━━━━━━━━━━━━━━━━━━ 313s 1s/step - accuracy: 0.9989 - loss: 0.0074 - val_accuracy: 0.9833 - val_loss: 0.0486 Tiempo de entrenamiento: 24280.33340549469

```pyton
plot_training_history(history_effnet)
```

![[Evolucion de accuracy y loss_val por epoch_effnet.png]]

```pyton
test_loss, test_acc = model_effnet.evaluate(test_data)
  
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)
```

test_loss, test_acc = model_effnet.evaluate(test_data)

print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

### Número de parámetros a entrenar

El modelo utilizado en este apartado se basa en la arquitectura **EfficientNetB0**, una red convolucional profunda optimizada para lograr un alto rendimiento con un número relativamente reducido de parámetros. En este caso se ha utilizado el modelo **preentrenado en ImageNet**, eliminando su parte superior de clasificación (`include_top=False`) para poder adaptarlo a nuestro problema de clasificación de **21 categorías**.

Durante la fase de _transfer learning_, los pesos del modelo base se han **congelado** (`base_model.trainable = False`), lo que implica que las capas convolucionales de EfficientNet no se entrenan y únicamente se actualizan los parámetros de las nuevas capas añadidas al modelo.

Las capas añadidas han sido:

- Una capa **GlobalAveragePooling2D**, que transforma los mapas de características en un vector de características.
- Una **capa densa de 50 neuronas con activación ReLU**, que aprende combinaciones de las características extraídas por EfficientNet.
- Una **capa de salida de 21 neuronas con activación softmax**, encargada de realizar la clasificación multiclase.

Gracias a esta estrategia, el número de parámetros que realmente se entrenan es **muy reducido en comparación con entrenar toda la red desde cero**, ya que se aprovechan las representaciones visuales aprendidas previamente por EfficientNet en el conjunto de datos ImageNet.

### Tiempo de entrenamiento

El entrenamiento del modelo ha tenido una duración aproximada de **24280 segundos** (alrededor de **6.7 horas**).

Este tiempo es considerablemente mayor que en los modelos anteriores de la práctica, principalmente debido a que **EfficientNetB0 es una arquitectura profunda y compleja**, que requiere realizar un mayor número de operaciones convolucionales sobre imágenes de tamaño **224×224**.

El modelo se entrenó durante un máximo de **100 épocas**, utilizando el callback **EarlyStopping** con una persistencia de **10 épocas**, monitorizando la **accuracy de validación**. Gracias a este mecanismo, el entrenamiento se detiene automáticamente cuando el rendimiento en validación deja de mejorar, evitando así sobreentrenamiento y reduciendo el tiempo total de entrenamiento.

En este caso, el entrenamiento finalizó en la **época 79**, cuando el modelo dejó de mejorar en el conjunto de validación.

### Precisión obtenida

Tras evaluar el modelo sobre el conjunto de **test**, se obtiene una **precisión muy elevada**, cercana al **98%**, con una pérdida muy baja.

Este resultado supone una **mejora muy significativa respecto a la red convolucional simple del apartado 3**, que alcanzaba una precisión aproximada del **86%**. Por tanto, el uso de **transfer learning con EfficientNetB0** permite obtener un rendimiento claramente superior en la tarea de clasificación de imágenes.

La mejora se debe a que EfficientNet ha sido previamente entrenado en **ImageNet**, un conjunto de datos con más de un millón de imágenes, lo que le permite aprender **características visuales muy generales y robustas**, como bordes, texturas, formas y patrones complejos. Estas representaciones pueden reutilizarse eficazmente para nuevos problemas mediante transfer learning.

### Comentario de las gráficas de entrenamiento

Las gráficas de evolución de **accuracy** y **loss** muestran un proceso de aprendizaje estable y progresivo.

Durante las primeras épocas se observa un **aumento rápido de la precisión**, pasando aproximadamente de **0.80 a más de 0.95 en pocas épocas**, lo que indica que el modelo se adapta rápidamente al nuevo problema de clasificación aprovechando las características ya aprendidas por EfficientNet.

A medida que avanza el entrenamiento, la **accuracy de entrenamiento continúa aumentando hasta valores cercanos a 1**, mientras que la **accuracy de validación se estabiliza alrededor de 0.98**. Las curvas de pérdida también muestran una disminución progresiva tanto en entrenamiento como en validación.

Aunque en las últimas épocas aparece una ligera separación entre las curvas de entrenamiento y validación, el comportamiento general indica que el modelo **generaliza adecuadamente** y no presenta un sobreajuste severo.

### Interpretación de los resultados

Los resultados obtenidos muestran claramente la **eficacia del transfer learning en problemas de visión por computador**. Al utilizar una red profunda preentrenada como EfficientNetB0, el modelo puede aprovechar características visuales aprendidas previamente en grandes conjuntos de datos, lo que permite alcanzar **altos niveles de precisión incluso con datasets más pequeños**.

En comparación con la red convolucional simple del apartado 3, el modelo basado en EfficientNet logra **una mejora notable en la precisión**, pasando de aproximadamente **86% a alrededor de 98%**, lo que demuestra la superioridad de las arquitecturas modernas de deep learning cuando se combinan con técnicas de transfer learning.

En conjunto, este apartado muestra cómo el uso de **modelos preentrenados y arquitecturas eficientes** permite mejorar significativamente el rendimiento en tareas de clasificación de imágenes, aunque a costa de **mayores tiempos de entrenamiento debido a la complejidad del modelo**.

### 5.2. Fine-tunning
Para mejorar los resultados del <i>transfer learning</i>, aplicaremos <i>fine-tunning</i>, que consiste en descongelar el modelo base y reentrenar la red completa durante unas pocas épocas con un <i>learning rate</i> muy pequeño para ajustar los pesos finamente a nuestro dominio.

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio[1 punto]:</strong> Volver a compilar el modelo con los siguientes cambios:
    <ul>
        <li>Descongelar los pesos del modelo EfficientNetB0 (<code>base_model_eff.trainable = True</code>).</li>
    </ul>
Compilar y entrenar el modelo siguiendo las siguientes indicaciones:
     <ul>
         <li>Utilizar el optimizador Adam con <i>learning rate</i> de 0.00001.</li>
         <li>Entrenar durante 10 épocas.</li>
         <li>Mostrar las gráficas y realizar la evaluación final sobre los datos de test.</li>
    </ul>
    Comenta los resultados globales del proceso.
</div>


```python
# Definición del modelo
base_model_eff = model_effnet.layers[0]

# Descongelar EfficientNetB0 para entrenamiento completo
base_model_eff.trainable = True

# Re-Compilación de la red
from tensorflow.keras.optimizers import Adam

model_effnet.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Entrenamiento
history_finetune = model_effnet.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)
```
  
Epoch 1/10

230/230 ━━━━━━━━━━━━━━━━━━━━ 444s 2s/step - accuracy: 0.9989 - loss: 0.0099 - val_accuracy: 0.9829 - val_loss: 0.0517 Epoch 2/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 449s 2s/step - accuracy: 0.9990 - loss: 0.0091 - val_accuracy: 0.9833 - val_loss: 0.0520 Epoch 3/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 447s 2s/step - accuracy: 0.9986 - loss: 0.0097 - val_accuracy: 0.9833 - val_loss: 0.0515 Epoch 4/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 440s 2s/step - accuracy: 0.9990 - loss: 0.0085 - val_accuracy: 0.9838 - val_loss: 0.0516 Epoch 5/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 461s 2s/step - accuracy: 0.9992 - loss: 0.0089 - val_accuracy: 0.9833 - val_loss: 0.0514 Epoch 6/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 463s 2s/step - accuracy: 0.9989 - loss: 0.0100 - val_accuracy: 0.9829 - val_loss: 0.0515 Epoch 7/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 460s 2s/step - accuracy: 0.9989 - loss: 0.0089 - val_accuracy: 0.9829 - val_loss: 0.0511 Epoch 8/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 454s 2s/step - accuracy: 0.9990 - loss: 0.0089 - val_accuracy: 0.9833 - val_loss: 0.0508 Epoch 9/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 432s 2s/step - accuracy: 0.9996 - loss: 0.0082 - val_accuracy: 0.9833 - val_loss: 0.0510 Epoch 10/10 230/230 ━━━━━━━━━━━━━━━━━━━━ 433s 2s/step - accuracy: 0.9993 - loss: 0.0084 - val_accuracy: 0.9833 - val_loss: 0.0506

```python
# Resultados
plot_training_history(history_finetune)
```

![[Evolucion de accuracy y loss_val por epoch_effnet_finetunning.png]]

### 5.2. Fine-tunning — Comentario de resultados

El proceso de **fine-tunning** consiste en descongelar el modelo base (**EfficientNetB0**) para permitir que **todos sus pesos se ajusten ligeramente al nuevo conjunto de datos**. A diferencia del _transfer learning_ inicial, donde el modelo base permanece congelado y solo se entrenan las capas finales, en esta fase se permite modificar toda la red, pero utilizando un **learning rate muy pequeño (0.00001)** para evitar cambios bruscos en los pesos previamente aprendidos.

En este caso se ha re-compilado el modelo utilizando el optimizador **Adam**, que es ampliamente utilizado en redes profundas debido a su buena capacidad de convergencia y adaptación del aprendizaje.

Durante el entrenamiento se observan los siguientes comportamientos:

- La **precisión de entrenamiento** se mantiene extremadamente alta durante todas las épocas, alrededor de **0.999**.    
- La **precisión de validación** se estabiliza aproximadamente en **0.983**.
- La **función de pérdida de validación** se mantiene prácticamente constante alrededor de **0.05**, con pequeñas oscilaciones entre épocas.

Estos resultados indican que el modelo ya estaba **muy bien ajustado tras el transfer learning inicial**, por lo que el fine-tunning únicamente produce **pequeñas mejoras marginales**. Esto es algo esperado cuando el modelo base ya representa bien las características visuales del conjunto de datos.

A partir de las gráficas de entrenamiento se pueden extraer varias conclusiones:

1. **Estabilidad del entrenamiento**  
    Tanto la _accuracy_ como la _loss_ presentan curvas muy estables, lo que indica que el modelo converge correctamente y no aparecen comportamientos erráticos.
2. **Ausencia de sobreajuste significativo**  
    La diferencia entre precisión de entrenamiento y validación es muy pequeña. Aunque el entrenamiento alcanza valores ligeramente superiores, la validación permanece cercana, lo que sugiere una buena capacidad de generalización.    
3. **Mejora limitada con fine-tunning**  
    Debido a que el modelo ya estaba altamente optimizado tras el _transfer learning_, el fine-tunning solo permite **ajustar finamente los pesos**, sin generar mejoras muy grandes en la precisión final.


En conclusión, el **fine-tunning permite adaptar ligeramente las representaciones internas del modelo al dominio específico del problema**, manteniendo una precisión muy elevada y confirmando que el modelo basado en EfficientNetB0 es altamente eficaz para esta tarea de clasificación.

## 6. Introducción a PyTorch: Red Convolucional Básica (1 punto)

Hasta ahora hemos utilizado TensorFlow/Keras, que nos ofrecen una API de alto nivel muy cómoda mediante los métodos `.fit()` y `.evaluate()`. Sin embargo, en el mundo de la investigación y en gran parte de la industria, **PyTorch** es el <i>framework</i> dominante debido a su flexibilidad y a que permite un control más profundo y "pythonico" del flujo de ejecución.

En PyTorch, nosotros mismos debemos definir cómo los datos pasan por la red (`forward pass`) y cómo iteramos sobre los lotes para actualizar los pesos en lo que llamamos el **bucle de entrenamiento** (*training loop*).

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Ejercicio [1 punto]:</strong> Implementa una red convolucional sencilla y entrénala usando PyTorch para clasificar nuestra base de datos.
    <ul>
        <li><strong>Carga de datos:</strong> Utiliza <code>torchvision.datasets.ImageFolder</code> y <code>DataLoader</code> para cargar las imágenes de entrenamiento desde la ruta <code>train_dir</code>. Aplica transformaciones para redimensionar a 224x224 y convertir a Tensor (lo cual rescala automáticamente a valores entre 0 y 1). Lotes de tamaño 32.</li>
        <li><strong>Definición del modelo:</strong> Crea una clase que herede de <code>nn.Module</code>. Debe contener:
            <ul>
                <li>Dos capas convolucionales (`nn.Conv2d`), cada una seguida de una activación ReLU y un <i>Max Pooling</i> (`nn.MaxPool2d`).</li>
                <li>Una capa <code>nn.AdaptiveAvgPool2d((1, 1))</code> para colapsar las dimensiones espaciales.</li>
                <li>Una capa lineal de clasificación (`nn.Linear`) con salida a 21 clases.</li>
            </ul>
        </li>
        <li><strong>Bucle de entrenamiento:</strong> Define como función de pérdida <code>nn.CrossEntropyLoss</code> y utiliza el optimizador Adam (lr=0.001). Escribe un bucle de entrenamiento de <strong>5 épocas</strong> que recorra el DataLoader de entrenamiento e imprima por pantalla la pérdida (<i>loss</i>) de cada época.</li>
    </ul>
    Comenta brevemente las diferencias que has notado al programar este modelo en PyTorch respecto a Keras.
</div>

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transformaciones de datoS
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()   # Convierte a tensor y escala automáticamente a [0,1]
])

# Carga del dataset
train_dataset = datasets.ImageFolder(
    root=train_dir,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# Definición del modelo
class SimpleCNN(nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
  
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(32, 21)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Inicializar modelo
model = SimpleCNN()

# Función de pérdida
criterion = nn.CrossEntropyLoss()

# Optimizador
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Bucle de entrenamiento
epochs = 5

  
for epoch in range(epochs):                                     # Iterar sobre el número de epochs
    running_loss = 0.0                                          # Acumulador de pérdida para este epoch
    for images, labels in train_loader:                         # Iterar sobre los lotes de datos
        optimizer.zero_grad()                                   # Limpiar los gradientes acumulados
        outputs = model(images)                                 # Forward pass: calcular las predicciones
        loss = criterion(outputs, labels)                       # Calcular la pérdida
        loss.backward()                                         # Backward pass: calcular los gradientes
        optimizer.step()                                        # Actualizar los pesos de la red
        running_loss += loss.item()                             # Acumular la pérdida para este epoch
    epoch_loss = running_loss / len(train_loader)               # Promedio de pérdida por epoch
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}") # Imprimir la pérdida promedio al final de cada epoch
```

Epoch 1/5 - Loss: 3.0101 Epoch 2/5 - Loss: 2.8809 Epoch 3/5 - Loss: 2.7779 Epoch 4/5 - Loss: 2.6497 Epoch 5/5 - Loss: 2.5449


#  Diferencias observadas entre PyTorch y Keras

Las principales diferencias observadas al implementar este modelo respecto a Keras son:

### Mayor control del entrenamiento

En PyTorch es necesario definir manualmente:

- el **forward pass**
- el **bucle de entrenamiento**
- el **cálculo de la pérdida**
- la **actualización de los pesos**

Mientras que en Keras todo esto se gestiona automáticamente mediante:

```
model.fit()
```

### Código más explícito

PyTorch obliga a escribir cada paso del entrenamiento:

```
forward
loss
backward
optimizer.step()
```

Esto hace el código **más flexible y transparente**, lo cual es una de las razones por las que PyTorch es muy utilizado en investigación.

### Mayor flexibilidad

Al controlar completamente el bucle de entrenamiento es más fácil implementar:

- arquitecturas personalizadas
- métodos de entrenamiento avanzados
- algoritmos experimentales.

<div style="background-color: #c3fac4; border-color: #dfb5b4; border-left: 5px solid #dfb5b4; padding: 0.5em;">

<strong>Fuentes:</strong>

<br><br>

Pon aquí las fuentes utilizadas en la elaboración de la PEC (incluyendo herramientas de IA generativa).

<br><br>

    <ul>

        <li><strong>Fuentes utilizadas en el ejercicio 1:</strong>

  

**Apollo2506. (2023).** *Land Use Scene Classification Dataset (Augmented UC Merced Land Use Dataset).* Kaggle.

[https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification)

  

* Utilizado para **descargar el dataset de imágenes satelitales** empleado en la práctica.

* Proporciona las **imágenes organizadas por clases** y los archivos **CSV con la partición en train, validation y test**.

  
  

**Rajput, A. (2022).** *How to upload my own notebook to Kaggle.* Medium.

[https://rajputankit22.medium.com/how-to-upload-my-own-notebook-to-kaggle-2b0dedbb5a6b](https://rajputankit22.medium.com/how-to-upload-my-own-notebook-to-kaggle-2b0dedbb5a6b)

  

* Utilizado como **guía para cargar notebooks propios en la plataforma Kaggle** y añadir datasets externos al entorno de trabajo.

  
  

**UC Merced Vision and Learning Center. (2010).** *UC Merced Land Use Dataset.* University of California, Merced.

[http://weegee.vision.ucmerced.edu/datasets/landuse.html](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

  

* Fuente original del **dataset UC Merced Land Use**, que describe las **21 categorías de uso del suelo** y las características de las imágenes satelitales utilizadas en tareas de clasificación.

  
  

**Keras Team. (2024).** *Image data loading: image_dataset_from_directory.* Keras Documentation.

[https://keras.io/api/data_loading/image/](https://keras.io/api/data_loading/image/)

  

* Utilizado para **comprender el funcionamiento de la función `image_dataset_from_directory()`**, empleada para crear los conjuntos de datos de entrenamiento, validación y test a partir de las carpetas de imágenes.

  

**TensorFlow Team. (2024).** *tf.keras.utils.image_dataset_from_directory.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)

  

* Utilizado como referencia técnica para **los parámetros de la función**, como `image_size`, `batch_size`, `color_mode` y la creación de datasets compatibles con TensorFlow/Keras.

  
  

**OpenAI. (2026).** *ChatGPT (GPT-5.3).* / **Anthropic. (2026).** *Claude (Haiku-4.5).*

  

* Utilizado como **herramienta de apoyo para revisar explicaciones conceptuales, redacción de comentarios del notebook y aclaración de dudas sobre el código y el análisis de datos**.

</li>

        <li><strong>Fuentes utilizadas en el ejercicio 2:</strong>

  

**Keras Team. (2024).** *Sequential model guide*. Keras Documentation.

[https://keras.io/guides/sequential_model/](https://keras.io/guides/sequential_model/)

  

* Utilizado para **comprender la construcción de modelos secuenciales en Keras mediante la clase `Sequential()`**, que se emplea para definir la arquitectura de la red neuronal completamente conectada.

  

**Keras Team. (2024).** *Dense layer.* Keras Documentation.

[https://keras.io/api/layers/core_layers/dense/](https://keras.io/api/layers/core_layers/dense/)

  

* Utilizado como referencia para la **implementación de capas completamente conectadas (`Dense`)**, incluyendo el uso de la función de activación **ReLU** en la capa oculta y **Softmax** en la capa de salida para la clasificación multiclase.

  

**Keras Team. (2024).** *Flatten layer.* Keras Documentation.

[https://keras.io/api/layers/reshaping_layers/flatten/](https://keras.io/api/layers/reshaping_layers/flatten/)

  

* Utilizado para comprender cómo **convertir una imagen tridimensional (altura × anchura × canales) en un vector unidimensional**, paso necesario para poder alimentar una red neuronal densa.

  

**Keras Team. (2024).** *Dropout layer.* Keras Documentation.

[https://keras.io/api/layers/regularization_layers/dropout/](https://keras.io/api/layers/regularization_layers/dropout/)

  

* Utilizado para aplicar **regularización mediante la técnica de Dropout**, reduciendo el riesgo de sobreajuste al desactivar aleatoriamente neuronas durante el entrenamiento.

  

**Keras Team. (2024).** *Image preprocessing layers: Resizing and Rescaling.* Keras Documentation.

[https://keras.io/api/layers/preprocessing_layers/image_preprocessing/](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/)

  

* Utilizado para **redimensionar las imágenes de entrada de 224×224 a 32×32 (`Resizing`)** y **normalizar los valores de píxel al rango [0,1] (`Rescaling`)**, pasos necesarios antes de introducir las imágenes en la red neuronal.

  

**TensorFlow Team. (2024).** *Adam optimizer.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)

  

* Utilizado como referencia para la **configuración del optimizador Adam**, incluyendo el parámetro **learning rate**, empleado para actualizar los pesos durante el entrenamiento.

  

**TensorFlow Team. (2024).** *EarlyStopping callback.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

  

* Utilizado para implementar **la técnica de Early Stopping**, que detiene el entrenamiento cuando la función de pérdida en validación deja de mejorar durante un número determinado de épocas.

  

**OpenAI. (2026).** *ChatGPT (GPT-5.3).* / **Anthropic. (2026).** *Claude (Haiku-4.5).*

  

* Utilizado como **herramienta de apoyo para revisar la explicación del modelo, interpretar los resultados del entrenamiento y mejorar la redacción de los comentarios del ejercicio**.

</li>

        <li><strong>Fuentes utilizadas en el ejercicio 3:</strong>

  

**LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.

[https://doi.org/10.1109/5.726791](https://doi.org/10.1109/5.726791)

  

* Referencia fundacional sobre **redes neuronales convolucionales (CNN)**. Se utiliza como base conceptual para entender cómo las **capas convolucionales permiten extraer patrones espaciales en imágenes**, lo que justifica su uso en tareas de visión por computador.

  

**Keras Team. (2024).** *Functional API guide.* Keras Documentation.

[https://keras.io/guides/functional_api/](https://keras.io/guides/functional_api/)

  

* Utilizado para comprender la **definición de modelos mediante la API funcional de Keras**, empleada en este ejercicio para construir la arquitectura de la red convolucional utilizando la clase `Model()`.

  

**Keras Team. (2024).** *Conv2D layer.* Keras Documentation.

[https://keras.io/api/layers/convolution_layers/convolution2d/](https://keras.io/api/layers/convolution_layers/convolution2d/)

  

* Utilizado como referencia para la implementación de **capas convolucionales (`Conv2D`)**, incluyendo parámetros como el **número de filtros, tamaño del kernel, padding y función de activación ReLU**.

  

**Keras Team. (2024).** *MaxPooling2D layer.* Keras Documentation.

[https://keras.io/api/layers/pooling_layers/max_pooling2d/](https://keras.io/api/layers/pooling_layers/max_pooling2d/)

  

* Utilizado para comprender el funcionamiento de las **capas de Max Pooling**, que permiten **reducir las dimensiones espaciales de las representaciones internas** y conservar las características más relevantes.

  

**Keras Team. (2024).** *GlobalAveragePooling2D layer.* Keras Documentation.

[https://keras.io/api/layers/pooling_layers/global_average_pooling2d/](https://keras.io/api/layers/pooling_layers/global_average_pooling2d/)

  

* Utilizado para aplicar **Global Average Pooling**, técnica que reduce los mapas de características generados por las capas convolucionales a un **vector de características compacto**, facilitando su uso en el clasificador final.

  

**Keras Team. (2024).** *Dense layer.* Keras Documentation.

[https://keras.io/api/layers/core_layers/dense/](https://keras.io/api/layers/core_layers/dense/)

  

* Utilizado para implementar las **capas completamente conectadas del clasificador final**, encargadas de transformar las características extraídas por la CNN en probabilidades de pertenencia a cada clase.

  

**Keras Team. (2024).** *Dropout layer.* Keras Documentation.

[https://keras.io/api/layers/regularization_layers/dropout/](https://keras.io/api/layers/regularization_layers/dropout/)

  

* Utilizado para aplicar **regularización mediante Dropout**, reduciendo el riesgo de sobreajuste durante el entrenamiento del modelo.

  

**TensorFlow Team. (2024).** *EarlyStopping callback.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

  

* Utilizado para implementar la técnica de **Early Stopping**, que detiene el entrenamiento cuando la pérdida en validación deja de mejorar durante varias épocas consecutivas.

  

**OpenAI. (2026).** *ChatGPT (GPT-5.3).* / **Anthropic. (2026).** *Claude (Haiku-4.5).*

  

* Utilizado como **herramienta de apoyo para revisar la explicación conceptual de las redes convolucionales, interpretar los resultados del entrenamiento y mejorar la redacción de los comentarios del ejercicio**.

</li>

        <li><strong>Fuentes utilizadas en el ejercicio 4:</strong>

  

**Hinton, G. E., & Salakhutdinov, R. R. (2006).** *Reducing the dimensionality of data with neural networks.* Science, 313(5786), 504–507.

[https://doi.org/10.1126/science.1127647](https://doi.org/10.1126/science.1127647)

  

* Referencia fundamental sobre **autoencoders y reducción de dimensionalidad mediante redes neuronales**. Se utiliza como base conceptual para entender cómo un autoencoder puede **comprimir los datos de entrada en una representación latente de menor dimensión y posteriormente reconstruirlos**.

  

**Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning.* MIT Press.

[https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)

  

* Utilizado para comprender el **funcionamiento general de los autoencoders**, incluyendo la estructura formada por **codificador (encoder) y decodificador (decoder)**, así como su uso en **aprendizaje no supervisado y reducción de dimensionalidad**.

  

**Keras Team. (2024).** *Conv2D layer.* Keras Documentation.

[https://keras.io/api/layers/convolution_layers/convolution2d/](https://keras.io/api/layers/convolution_layers/convolution2d/)

  

* Utilizado como referencia para la implementación de las **capas convolucionales del codificador**, encargadas de extraer características de las imágenes mientras se reduce progresivamente su resolución espacial.

  

**Keras Team. (2024).** *Conv2DTranspose layer.* Keras Documentation.

[https://keras.io/api/layers/convolution_layers/convolution2d_transpose/](https://keras.io/api/layers/convolution_layers/convolution2d_transpose/)

  

* Utilizado para comprender el funcionamiento de las **capas de convolución transpuesta**, empleadas en el **decodificador** para reconstruir la imagen aumentando progresivamente su resolución.

  

**Keras Team. (2024).** *UpSampling2D layer.* Keras Documentation.

[https://keras.io/api/layers/reshaping_layers/up_sampling2d/](https://keras.io/api/layers/reshaping_layers/up_sampling2d/)

  

* Utilizado para aplicar **operaciones de aumento de resolución espacial** durante el proceso de decodificación, permitiendo recuperar gradualmente el tamaño original de la imagen.

  

**Keras Team. (2024).** *MaxPooling2D layer.* Keras Documentation.

[https://keras.io/api/layers/pooling_layers/max_pooling2d/](https://keras.io/api/layers/pooling_layers/max_pooling2d/)

  

* Utilizado para comprender el funcionamiento de las **capas de Max Pooling**, que reducen las dimensiones espaciales de los mapas de características en el **codificador**, contribuyendo al proceso de compresión de la información.

  

**TensorFlow Team. (2024).** *EarlyStopping callback.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

  

* Utilizado para implementar la técnica de **Early Stopping**, que permite detener el entrenamiento automáticamente cuando la pérdida en validación deja de mejorar durante varias épocas consecutivas.

  

**TensorFlow Team. (2024).** *Adam optimizer.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)

  

* Utilizado como referencia para la **configuración del optimizador Adam**, incluyendo el uso de un **learning rate de 0.001** para el entrenamiento del autoencoder.

  

**Keras Team. (2024).** *Rescaling layer.* Keras Documentation.

[https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/)

  

* Utilizado para comprender el proceso de **normalización de los valores de los píxeles de las imágenes al rango [0,1]**, necesario antes de entrenar el modelo.

  

**OpenAI. (2026).** *ChatGPT (GPT-5.3).* / **Anthropic. (2026).** *Claude (Haiku-4.5).*

  

* Utilizado como **herramienta de apoyo para revisar la explicación conceptual del autoencoder, interpretar los resultados del entrenamiento y mejorar la redacción de los comentarios incluidos en el ejercicio**.

</li>

        <li><strong>Fuentes utilizadas en el ejercicio 5:</strong>

  

**Tan, M., & Le, Q. (2019).** *EfficientNet: Rethinking model scaling for convolutional neural networks.* Proceedings of the 36th International Conference on Machine Learning (ICML).

[https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)

  

* Artículo original que introduce la familia de modelos **EfficientNet**. Se utiliza como referencia conceptual para comprender cómo estas arquitecturas optimizan el rendimiento de las redes convolucionales mediante un **escalado equilibrado de profundidad, anchura y resolución**, permitiendo obtener alta precisión con menos parámetros.

  

**Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009).** *ImageNet: A large-scale hierarchical image database.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[https://doi.org/10.1109/CVPR.2009.5206848](https://doi.org/10.1109/CVPR.2009.5206848)

  

* Referencia del conjunto de datos **ImageNet**, utilizado para **preentrenar el modelo EfficientNetB0**. ImageNet contiene millones de imágenes etiquetadas y permite que los modelos aprendan **características visuales generales** reutilizables mediante transfer learning.

  

**Keras Team. (2024).** *EfficientNet models.* Keras Documentation.

[https://keras.io/api/applications/efficientnet/](https://keras.io/api/applications/efficientnet/)

  

* Utilizado como referencia para **cargar el modelo EfficientNetB0 preentrenado en ImageNet** mediante `EfficientNetB0(weights='imagenet', include_top=False)` y comprender los parámetros disponibles en su implementación.

  

**Keras Team. (2024).** *Transfer learning & fine-tuning guide.* Keras Documentation.

[https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/)

  

* Utilizado para comprender el proceso de **transfer learning y fine-tuning**, incluyendo estrategias como **congelar el modelo base**, añadir nuevas capas de clasificación y posteriormente **descongelar la red para ajustar los pesos finamente**.

  

**Keras Team. (2024).** *GlobalAveragePooling2D layer.* Keras Documentation.

[https://keras.io/api/layers/pooling_layers/global_average_pooling2d/](https://keras.io/api/layers/pooling_layers/global_average_pooling2d/)

  

* Utilizado para comprender el funcionamiento de la capa **GlobalAveragePooling2D**, empleada para transformar los mapas de características generados por EfficientNet en un vector de características antes de la clasificación final.

  

**Keras Team. (2024).** *Dense layer.* Keras Documentation.

[https://keras.io/api/layers/core_layers/dense/](https://keras.io/api/layers/core_layers/dense/)

  

* Utilizado para implementar las **capas completamente conectadas añadidas al modelo preentrenado**, incluyendo la capa oculta de **50 neuronas con activación ReLU** y la capa de salida **Softmax** para clasificación multiclase.

  

**TensorFlow Team. (2024).** *Adam optimizer.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam)

  

* Utilizado como referencia para la **configuración del optimizador Adam**, incluyendo el uso de **learning rates diferentes** para las fases de transfer learning (0.0001) y fine-tuning (0.00001).

  

**TensorFlow Team. (2024).** *EarlyStopping callback.* TensorFlow Documentation.

[https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)

  

* Utilizado para implementar **Early Stopping**, deteniendo el entrenamiento cuando la **accuracy de validación deja de mejorar** durante varias épocas consecutivas.

  

**OpenAI. (2026).** *ChatGPT (GPT-5.3).* / **Anthropic. (2026).** *Claude (Haiku-4.5).*

  

* Utilizado como **herramienta de apoyo para revisar la explicación conceptual del transfer learning y fine-tuning, interpretar los resultados del entrenamiento y mejorar la redacción de los comentarios del ejercicio**.

</li>

        <li><strong>Fuentes utilizadas en el ejercicio 6:</strong>

  

**Paszke, A., Gross, S., Massa, F., et al. (2019).** *PyTorch: An imperative style, high-performance deep learning library.*

Advances in Neural Information Processing Systems (NeurIPS).

  

[https://arxiv.org/abs/1912.01703](https://arxiv.org/abs/1912.01703)

  

* Artículo original que describe el framework **PyTorch**. Se utiliza como referencia conceptual para comprender el paradigma de **programación imperativa** utilizado por PyTorch, donde el flujo de ejecución se define dinámicamente y el usuario controla explícitamente el **forward pass**, el **cálculo de gradientes** y el **bucle de entrenamiento**.

  

**TorchVision Team. (2024).** *ImageFolder dataset.* TorchVision Documentation.

  

[https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)

  

* Utilizado para cargar el conjunto de imágenes mediante **`torchvision.datasets.ImageFolder`**, que permite estructurar el dataset de clasificación a partir de directorios organizados por clase.

  

**PyTorch Team. (2024).** *DataLoader.* PyTorch Documentation.

  

[https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)

  

* Utilizado para crear el **DataLoader**, que permite iterar sobre los datos en **lotes (batch_size=32)**, gestionar el barajado de los datos (`shuffle=True`) y facilitar el proceso de entrenamiento.

  

**PyTorch Team. (2024).** *torch.nn module.* PyTorch Documentation.

  

[https://pytorch.org/docs/stable/nn.html](https://pytorch.org/docs/stable/nn.html)

  

* Utilizado como referencia para la implementación de la arquitectura de la red neuronal mediante **`nn.Module`**, incluyendo las capas **`Conv2d`**, **`ReLU`**, **`MaxPool2d`**, **`AdaptiveAvgPool2d`** y **`Linear`** utilizadas en la red convolucional básica.

  

**PyTorch Team. (2024).** *CrossEntropyLoss.* PyTorch Documentation.

  

[https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

  

* Utilizado como referencia para la función de pérdida **`nn.CrossEntropyLoss`**, adecuada para problemas de **clasificación multiclase** como el planteado en la práctica.

  

**PyTorch Team. (2024).** *Adam optimizer.* PyTorch Documentation.

  

[https://pytorch.org/docs/stable/generated/torch.optim.Adam.html](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)

  

* Utilizado para configurar el optimizador **Adam** con **learning rate = 0.001**, encargado de actualizar los pesos del modelo durante el entrenamiento.

  

**PyTorch Team. (2024).** *TorchVision transforms.* TorchVision Documentation.

  

[https://pytorch.org/vision/stable/transforms.html](https://pytorch.org/vision/stable/transforms.html)

  

* Utilizado para aplicar transformaciones a las imágenes antes del entrenamiento, incluyendo **`Resize((224,224))`** y **`ToTensor()`**, que convierte las imágenes en tensores y normaliza automáticamente los valores de píxel al rango **[0,1]**.

  

**OpenAI. (2026).** *ChatGPT (GPT-5.3).* / **Anthropic. (2026).** *Claude (Haiku-4.5).*

  
  

* Utilizado como herramienta de apoyo para **revisar la implementación del modelo en PyTorch, explicar el funcionamiento del bucle de entrenamiento y mejorar la redacción de la comparación entre PyTorch y Keras**.

</li>        

    </ul>

</div>