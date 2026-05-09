# 0. Contexto y carga de librerías

En este ejercicio práctico nos inspiramos en el paper “Text Classification: Neural Networks vs Machine Learning Models vs Pre-trained Models” (https://arxiv.org/pdf/2412.21022). En realidad, en este ejercicio trabajaremos con text generation, pero el paper nos va a ser útil para entender cómo comparar modelos, tipos de gráficas o ablation studies que se pueden aplicar.  El objetivo del trabajo es explorar y entender de manera práctica cómo distintos tipos de redes neuronales secuenciales pueden abordar la tarea de clasificación de texto.


### Consideraciones sobre complejidad computacional y uso de GPU en la PEC:
**Atención -> Ejecutar correctamente esta PEC lleva tiempo y organización**.

Vamos a trabajar con transformers, arquitecturas grandes, etc. Será necesaria la utilización de GPU para acelerar los cálculos y, además, el tiempo de computación puede ser elevado. La ejecución final de todo el ejercicio puede llevar fácilmente varias horas en GPU. Por ello, aquí tienes algunos consejos fundamentales para facilitar tu trabajo:
* Comienza programando y debugueando tu ejercicio con subconjuntos pequeños que puedas ejecutar en minutos. Incluso en local, en cpu. También puedes considerar ir haciendo la PEC por partes sin necesidad de ejecutar todo. Cuando tu código esté preparado, escala tu ejercicio.
* Si utilizas plataformas como Kaggle y Colab, ten en cuenta sus limitaciones de tiempo y organízate de acuerdo.
* Kaggle es más estable que Colab.
* Si utilizas Colab Pro, ten en cuenta que si lo linqueas con Kaggle, también tendrás más horas de cómputo en Kaggle. https://www.kaggle.com/blog/level-up-your-compute-more-gpu-hours-on-kaggle-wit
* Comienza la PEC pronto, para que te de tiempo a ejecutarla y comentar tus conclusiones antes de entregarla.
* A la hora de elegir las GPUs, ten en cuenta cual es más rápida. e.g. https://www.topcpu.net/es/gpu-c/tesla-p100-pcie-16-gb-vs-tesla-t4
* Por mucho que tengas acceso a 2xGPU, si tu código no está adaptado a multiGPU, simplemente tendrás una GPU sin utilizar.
* Te parece largo? Echa un vistazo a los tiempos de entrenamiento de LLMs (interesante Megatron-Turing): https://en.wikipedia.org/wiki/List_of_large_language_models

```python
import math
import copy
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification

import evaluate
metric = evaluate.load("accuracy")

from tqdm import tqdm
from tabulate import tabulate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
```

# 1. Baseline. RNN vs Transformers
Vamos a comenzar comparando text generation, en particular, vamos a implementar desde cero dos modelos sencillos de text generation. Un modelo LSTM y un modelo basado en transformers y compararlos entre ellos.

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">

Antes de entrenar modelos de lenguaje, necesitamos convertir el texto en unidades básicas que podamos utilizar y hay varios pasos para preparar un texto para entrenar un modelo de generación de lenguaje.
1. Tokenización. Estas unidades básicas pueden ser de muy diferentes tipos: palabras, bpe, caracteres, etc. En esta primera parte utilizaremos palabras. El objetivo es separar el texto en unidades básicas (https://www.traceloop.com/blog/a-comprehensive-guide-to-tokenizing-text-for-llms)
2. Codificación de tokens a números.
3. Padding y secuencias fijas. Las secuencias de entrenamiento deben tener la misma longitud. Aquí utilizaremos una longitud de 50. El modelo intentará predecir la última palabra, en base a las anteriores. Si la secuencia es más corta, simplemente rellenala con un padding.

**Ejercicio 1. Preparación de los datos [1 pts.].**

- Cargar el dataset wikitext-2-raw-v1 desde el repositorio de hugginface https://huggingface.co/datasets/Salesforce/wikitext
- Preparación de tus datos de train, validation y text de la siguiente forma:
  * Tokeniza el texto
  * Crea secuencias de hasta 50 palabras más la palabra objetivo (la 51) (con cada texto crea el máximo número de secuencias posibles)
  * Quizás entrenar con todas las secuencias sea demasiado. Elige de forma adecuada un subconjunto de ellas.
- Muestra en una tabla los datos que había al principio, el número de secuencias totales generadas y con las que te has quedado al final. La tabla debe estar bien formateada con lineas
- Imprime en pantalla 10 secuencias de cada uno de tus subconjuntos finales
</div>

## 1.1. Carga de datos

Se ha utilizado el dataset `wikitext-2-raw-v1` disponible en HuggingFace, compuesto por fragmentos de texto extraídos de Wikipedia. Este dataset se encuentra dividido en tres subconjuntos independientes: entrenamiento (*train*), validación (*validation*) y prueba (*test*).

La carga del dataset se realiza mediante la librería `datasets`, que descarga automáticamente los ficheros y los almacena en caché local para reutilizaciones posteriores. Cada elemento del dataset contiene una única columna denominada `text`, que almacena fragmentos de texto sin procesar.

```python
# Carga el dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

print(dataset)
```

```python
# Carga el dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

print(dataset)
```

README.md: 
 10.5k/? [00:00<00:00, 863kB/s]

c:\Users\usuario\AppData\Local\Programs\Python\Python313\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\usuario\.cache\huggingface\hub\datasets--Salesforce--wikitext. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`

test-00000-of-00001.parquet: 100%
 733k/733k [00:00<00:00, 33.5MB/s]

Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`

train-00000-of-00001.parquet: 100%
 6.36M/6.36M [00:00<00:00, 45.1MB/s]

Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`

validation-00000-of-00001.parquet: 100%
 657k/657k [00:00<00:00, 46.2MB/s]
Generating test split: 100%
 4358/4358 [00:00<00:00, 6797.90 examples/s]
Generating train split: 100%
 36718/36718 [00:00<00:00, 373783.02 examples/s]
Generating validation split: 100%
 3760/3760 [00:00<00:00, 86241.67 examples/s]

DatasetDict({
    test: Dataset({
        features: ['text'],
        num_rows: 4358
    })
    train: Dataset({
        features: ['text'],
        num_rows: 36718
    })
    validation: Dataset({
        features: ['text'],
        num_rows: 3760
    })
})

## 1.2. Preparación de los conjuntos de datos: *train*, *test*, *validation*

### 1.2.1. Tokenización simple a nivel de palabra

Para transformar el texto en unidades básicas manejables por los modelos, en este paso se implementa una tokenización sencilla a nivel de palabra. Inicialmente, se filtran y eliminan los fragmentos de texto vacíos en los tres subconjuntos (*train*, *validation* y *test*) para limpiar los datos. 

```python
train_texts = [t for t in dataset["train"]["text"] if len(t.strip()) > 0]
valid_texts = [t for t in dataset["validation"]["text"] if len(t.strip()) > 0]
test_texts  = [t for t in dataset["test"]["text"] if len(t.strip()) > 0]
```

Posteriormente, se define una función de tokenización que convierte todo el texto a minúsculas y lo divide basándose en los espacios en blanco.

```python
# Tokenización
def tokenize_text(text):
    text = text.lower().strip()
    return text.split()
```

A partir de los textos procesados del conjunto de entrenamiento, se contabilizan las apariciones de cada palabra para construir un vocabulario base. A este diccionario se le incorporan dos *tokens* especiales fundamentales: `<PAD>` (índice 0), que servirá más adelante para rellenar y estandarizar la longitud de las secuencias de entrada, y `<UNK>` (índice 1), utilizado para representar cualquier palabra desconocida que no esté presente en el conjunto de entrenamiento. El proceso concluye asignando un identificador numérico único a cada término, dando como resultado un vocabulario final de 66.651 *tokens*.

```python
# Construir vocabulario
counter = Counter()

for text in train_texts:
    counter.update(tokenize_text(text))

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<UNK>": 1
}

vocab = dict(SPECIAL_TOKENS)

for word, _ in counter.items():
    vocab[word] = len(vocab)

idx2word = {v: k for k, v in vocab.items()}

print(f"Vocab size: {len(vocab)}")
```
Vocab size: 66651

### 1.2.2. Generación de secuencias

El siguiente paso consiste en transformar las listas de *tokens* en secuencias numéricas estructuradas y de longitud fija. Este proceso se lleva a cabo mediante dos operaciones principales:

En primer lugar, la función de **codificación (`encode()`)** mapea cada palabra a su identificador numérico correspondiente utilizando el vocabulario previamente construido. Si se encuentra alguna palabra que no formaba parte del conjunto de entrenamiento, se gestiona automáticamente asignándole el identificador del *token* desconocido (`<UNK>`).

```python
def encode(tokens):
    '''"
    Convierte una lista de tokens a índices usando el vocabulario.
    '''
    return [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
```

En segundo lugar, la **construcción de secuencias (`create_sequences()`)** aplica una técnica de ventana deslizante sobre cada texto. De este modo, en cada paso, se toman hasta 50 palabras previas como contexto y se define la palabra inmediatamente posterior como el objetivo a predecir (`target`).

```python
def create_sequences(texts, seq_len=50):
    '''
    Construye secuencias de entrenamiento para un modelo de lenguaje autoregresivo.
    '''
    sequences = []

    # iteramos por cada texto y construimos secuencias de longitud seq_len
    for text in texts:
        tokens = tokenize_text(text)

        # solo consideramos textos con al menos 2 tokens
        if len(tokens) < 2:
            continue

        # convertimos tokens a índices usando el vocabulario
        encoded = encode(tokens)

        # ventanas deslizantes
        for i in range(1, len(encoded)):
            start = max(0, i - seq_len)
            seq = encoded[start:i]
            target = encoded[i]

            # padding izquierda
            seq = [vocab["<PAD>"]] * (seq_len - len(seq)) + seq

            # guardamos la secuencia y su token objetivo
            sequences.append((seq, target))
    
    return sequences
```

Dado que las redes neuronales requieren que todas las entradas tengan la misma dimensión, se aplica una técnica de **padding a la izquierda**. Si el contexto disponible tiene menos de 50 palabras (por ejemplo, al principio de una frase), los huecos vacíos al inicio de la secuencia se rellenan con el identificador del *token* `<PAD>`.

Al aplicar esta transformación, el dataset original se expande de forma masiva, ya que cada texto genera tantas secuencias como palabras contenga (menos la primera). Como resultado, obtenemos 2.028.143 de secuencias estructuradas para la fase de entrenamiento, 211.425 para validación y  238.320 para el testeo final del modelo.

```python
# Construir secuencias de entrenamiento
SEQ_LEN = 50 # longitud máxima de secuencia

# Construimos secuencias para cada split
train_sequences = create_sequences(train_texts, SEQ_LEN)
valid_sequences = create_sequences(valid_texts, SEQ_LEN)
test_sequences  = create_sequences(test_texts, SEQ_LEN)

# Imprimir estadísticas
print(f"Train sequences: {len(train_sequences):,}")
print(f"Validation sequences: {len(valid_sequences):,}")
print(f"Test sequences: {len(test_sequences):,}")
```

Train sequences: 2,028,143
Validation sequences: 211,425
Test sequences: 238,320

### 1.2.3. Sub-muestreo de las secuencias

Para consolidar y visualizar el impacto de las transformaciones realizadas en las etapas previas, en este apartado se genera una tabla resumen utilizando la librería `tabulate`. Esta tabla ofrece una perspectiva general y estructurada de cómo ha evolucionado el volumen de datos a lo largo de todo el proceso de preparación.

Como se puede observar en los resultados, se parte de una cantidad moderada de fragmentos de texto originales (23.767 en el caso del conjunto de entrenamiento). Tras aplicar el proceso de ventana deslizante para generar el contexto y la palabra objetivo, esta cifra se multiplica exponencialmente hasta superar los dos millones de secuencias totales generadas.

Finalmente, la columna de *Secuencias finales* refleja la aplicación del sub-muestreo, confirmando que los conjuntos han sido acotados con éxito a los límites manejables definidos en el paso anterior (60.000 secuencias para el entrenamiento y 8.000 tanto para la validación como para el testeo). Este resumen tabular no solo ilustra la magnitud de la expansión de los datos en tareas de modelado de lenguaje, sino que certifica que los subconjuntos están correctamente dimensionados y listos para la fase de entrenamiento computacional.

```python
# Reducimos tamaño para hacer entrenamientos razonables
MAX_TRAIN = 60000
MAX_VALID = 8000
MAX_TEST  = 8000

# fijamos semilla para reproducibilidad
random.seed(42)

# muestreamos aleatoriamente los subconjuntos de secuencias para cada split
train_sequences = random.sample(
    train_sequences,
    min(MAX_TRAIN, len(train_sequences))
)

valid_sequences = random.sample(
    valid_sequences,
    min(MAX_VALID, len(valid_sequences))
)

test_sequences = random.sample(
    test_sequences,
    min(MAX_TEST, len(test_sequences))
)
```

## 1.3. Tabla resumen de los datos

En este apartado se genera una tabla resumen utilizando la librería `tabulate`. Como se puede observar en los resultados, se parte de una cantidad moderada de fragmentos de texto originales (23.767 en el caso del conjunto de entrenamiento). Tras aplicar el proceso de ventana deslizante para generar el contexto y la palabra objetivo, esta cifra se multiplica exponencialmente hasta superar los dos millones de secuencias totales generadas.

Finalmente, la columna de *Secuencias finales* refleja la aplicación del sub-muestreo, confirmando que los conjuntos han sido acotados con éxito a los límites manejables definidos en el paso anterior. 

```python
# Imprimir tabla resumen por subconjunto
table = [
    [
        "Train",
        len(train_texts),
        len(create_sequences(train_texts, SEQ_LEN)),
        len(train_sequences)
    ],
    [
        "Validation",
        len(valid_texts),
        len(create_sequences(valid_texts, SEQ_LEN)),
        len(valid_sequences)
    ],
    [
        "Test",
        len(test_texts),
        len(create_sequences(test_texts, SEQ_LEN)),
        len(test_sequences)
    ]
]

print(
    tabulate(
        table,
        headers=[
            "Subconjunto",
            "Textos originales",
            "Secuencias totales generadas",
            "Secuencias finales"
        ],
        tablefmt="grid"
    )
)
```

+---------------+---------------------+--------------------------------+----------------------+
| Subconjunto   |   Textos originales |   Secuencias totales generadas |   Secuencias finales |
+===============+=====================+================================+======================+
| Train         |               23767 |                        2028143 |                60000 |
+---------------+---------------------+--------------------------------+----------------------+
| Validation    |                2461 |                         211425 |                 8000 |
+---------------+---------------------+--------------------------------+----------------------+
| Test          |                2891 |                         238320 |                 8000 |
+---------------+---------------------+--------------------------------+----------------------+

## 1.4. Visualización de ejemplos por subonjunto

En este último paso se procede a la inspección visual de las secuencias generadas. 

En primer lugar, se implementa una función de **decodificación (`decode_sequence()`)** que realiza el proceso inverso. Es decir, convierte los identificadores numéricos de las secuencias de vuelta a su formato textual original, omitiendo de forma intencionada los *tokens* de relleno (`<PAD>`) para facilitar la lectura.

En segundo lugar, la función de **impresión (`print_examples()`)** muestra los diez primeros ejemplos mediante una función de cada uno de los subconjuntos (entrenamiento, validación y prueba). En cada bloque se aprecia:

* **INPUT:** Representa el texto de entrada (compuesto por hasta 50 palabras).
* **TARGET:** Es la palabra inmediatamente posterior.

Mediante la visualización de estos ejemplos prácticos, se observa cómo los signos de puntuación se tratan como *tokens* independientes y, de manera muy destacada, se comprueba el funcionamiento del *token* `<UNK>`. Como es evidente en varios ejemplos de los conjuntos de validación y test (e.g., `<UNK> de san juan de <UNK>`), cualquier término, nombre propio o símbolo que no formaba parte del vocabulario inicial de 66.651 palabras ha sido correctamente detectado y sustituido por el identificador de palabra desconocida.

```python
def decode_sequence(seq):
    return [idx2word[idx] for idx in seq if idx != vocab["<PAD>"]]

def print_examples(data, name, n=10):

    print("\n" + "="*80)
    print(name.upper())
    print("="*80)

    for i in range(n):

        seq, target = data[i]

        words = decode_sequence(seq)
        target_word = idx2word[target]

        print(f"\nEjemplo {i+1}")
        print("INPUT :", " ".join(words))
        print("TARGET:", target_word)

print_examples(train_sequences, "train")
print_examples(valid_sequences, "validation")
print_examples(test_sequences, "test")
```

================================================================================
TRAIN
================================================================================

Ejemplo 1
INPUT : 38 enters auburn , it passes by auburn high school before heading north through densely populated blocks filled with homes . the divided highway ends abruptly at swift street , where ny 38 turns west to follow the two @-@ lane undivided swift street west for seven blocks to ny
TARGET: 34

Ejemplo 2
INPUT : by the end of the 19th century , the city of galveston had a population of 37 @,@ 000 . its position on the natural harbor of galveston bay along the gulf of mexico made it the center of trade in texas . it was one of
TARGET: the

Ejemplo 3
INPUT : 1 , 1849 — atherton was chairman of the senate finance committee . mckay introduced a version into the house on february 20 ; debate began the same day . the dollar was attacked by congressmen from the whig party , then in the minority , on the grounds that
TARGET: it

Ejemplo 4
INPUT : , derby . on 5 september , he scored his first competitive goal for england in the 2 – 0 win against albania at st james ' park , newcastle . this was during qualifying for the 2002 fifa world cup . england qualified for the world cup , and
TARGET: after

Ejemplo 5
INPUT : in the upcoming weeks of holiday sales where it peaked in the year 's last weeks with 486 @,@ 000 and 760 @,@ 000 units sold at the pinnacle . the album moved 760 @,@ 000 copies during the christmas week of 1995 , the album 's highest sales week
TARGET: .

Ejemplo 6
INPUT : to promote the film , eon productions produced a one @-@ hour colour television programme entitled welcome to japan , mr. bond first aired on 2 june 1967 in the
TARGET: united

Ejemplo 7
INPUT : in august , 2011 , beyoncé performed " crazy in love " during her revue show 4 intimate nights with beyoncé . she performed a slowed @-@ down , jazzier version of the song and danced with
TARGET: a

Ejemplo 8
INPUT : protagonists of the water margin . he is also given the nickname " iron arm " ( 铁臂膀 ) , which carried over into the title of his fictional biography iron arm , golden sabre . while the tale fails to explain the reason for the moniker , it does
TARGET: mention

Ejemplo 9
INPUT : = = production =
TARGET: =

Ejemplo 10
INPUT : toniná in 668 . his rule is marked by warfare and the frequent depiction of bound captives on his monuments . ruler 2 established the use of in @-@ the @-@ round sculptural style that came to typify the stelae of toniná . a monument dated to 682 depicts three
TARGET: naked

================================================================================
VALIDATION
================================================================================

Ejemplo 1
INPUT : feature . they made it a joyous occasion for themselves as well as their guests . they were hardly an overly wealthy family , and their table was never notable for an <UNK> of the good things of life , but whenever they gave a dinner they cast all thoughts
TARGET: of

Ejemplo 2
INPUT : small <UNK> and four larger ' <UNK> ' . <UNK> <UNK> ( old norse : ' harbour bay ' ) in the south is the most sheltered anchorage and the surrounding cliffs contain a natural rock arch . <UNK> <UNK> to the east ( old norse : ' house bay
TARGET: '

Ejemplo 3
INPUT : amtrak 's crescent line connects meridian with the cities of new york , new york ; philadelphia , pennsylvania ; baltimore , maryland ; washington , d.c.
TARGET: ;

Ejemplo 4
INPUT : the canadian online explorer rated the show a 7 out of 10 , which was lower than the 8 out of 10 given to the 2007 edition by jason <UNK> . after the event , an accident occurred which resulted in the death of one man and the injury of
TARGET: another

Ejemplo 5
INPUT : scientologists at her request and that of scientology 's office of special affairs , and that she was in personal financial trouble and about to go on trial for tax <UNK> at the time she applied for asylum . on a 2000 visit to clearwater , florida , <UNK> <UNK>
TARGET: of

Ejemplo 6
INPUT : = =
TARGET: lyrics

Ejemplo 7
INPUT : " training day " received mixed reviews from television critics , with many commenting
TARGET: on

Ejemplo 8
INPUT : with britain accepting 200 men from victoria , 260 from new south wales and the south australian ship hmcs protector , under the command of captain william <UNK> . most of these forces were made up of naval brigade <UNK> , who had been trained in both ship handling and
TARGET: soldiering

Ejemplo 9
INPUT : according to guitarist kerry king : " yeah , ' slayer are nazis , fascists , communists ' — all that fun shit . and of course we got the most flak for it in germany .
TARGET: i

Ejemplo 10
INPUT : the <UNK> was informed in october 1985 by federal law enforcement officials that leslie l. <UNK> , an investigative journalist who had written a 20 @-@ part series on the <UNK> movement
TARGET: in

================================================================================
TEST
================================================================================

Ejemplo 1
INPUT : , in effect , attacked straight east across the river and was trying to seize the two avenues of advance into <UNK> above and below lake u @-@ p 'o . on august 31 , 1950 , lake u @-@ p 'o was a large body of water although in
TARGET: most

Ejemplo 2
INPUT : central australian artists frequently paint particular " <UNK> " , or stories , for which they have responsibility or rights . these stories are used to pass " important knowledge , cultural
TARGET: values

Ejemplo 3
INPUT : how to curse was performed at bush theatre in the london borough of hammersmith and fulham . <UNK> starred in two films in 2008 , daylight robbery by filmmaker paris <UNK> , and donkey punch directed by olly blackburn . in may 2008 , <UNK> made a guest appearance on
TARGET: a

Ejemplo 4
INPUT : a number of design faults of the tetrarch were revealed through its operational use . its size limited the possible crew to three , a driver in the hull and
TARGET: a

Ejemplo 5
INPUT : squadron suffered heavy casualties during the invasion ; only one valentine and three <UNK> out of twelve tanks were functional by 7 may , and the squadron had suffered seven killed and six wounded . it remained in madagascar until early 1943 , when it was shipped to india and
TARGET: took

Ejemplo 6
INPUT : the second battle of naktong bulge was an engagement between united nations ( un ) and north korean ( <UNK> ) forces early in the korean war from september 1 to september 15 , 1950 , along the naktong river
TARGET: in

Ejemplo 7
INPUT : order to clear land for his planned palatial complex , the <UNK> <UNK> . in 68 , the rebellion of <UNK> in gaul and later the acclamation of <UNK> in <UNK> drove nero from the throne . facing a false report of being denounced as a public enemy who was
TARGET: to

Ejemplo 8
INPUT : two modern statues which commemorate <UNK> can be found in towns associated with him . there is an equestrian statue in gloucester , england , a town which was founded in his honour . it
TARGET: is

Ejemplo 9
INPUT : some of the country 's oldest schools are founded in <UNK> , these are the university of santo tomas ( 1611 ) , <UNK> de san juan de <UNK> ( 1620 )
TARGET: ,

Ejemplo 10
INPUT : , unhappy with the politics of washington and suffering from poor health . although he was no longer active in politics , he continued to express opinions on the subjects of the day , opposing the 1820 missouri compromise and <UNK> the " great moderation & <UNK> " of federalist
TARGET: governor

<div style="background-color: #fcf2f2; border-color: #dfb5b4; border-left: 5px solid #dfb5b4; padding: 0.5em;">
<p><strong>Solución:</strong> La carga y el preprocesamiento se han ido explicando en cada apartado.</p>
</div>

---
---

<div style="background-color: #EDF7FF; border-color: #7C9DBF; border-left: 5px solid #7C9DBF; padding: 0.5em;">

**Ejercicio 2. Entrenamiento LSTM vs Transformer [2pts.].**

- En esta parte monta dos modelos sencillos. Uno basado en LSTM y otro en Transformer con dos cabezas de atención. Algunos requisitos son:
  * El total de parámetros entrenables de cada uno de ellos debe ser entre 5 y 6 millones de parámetros.
  * La idea es que ambos modelos sean comparables, así que se lo más consistente posible entre ambas arquitecturas. Por ejemplo: uno no puede tener 15 capas y el otro 1 capa.
  * Resto de parámetros es libre. Pero debes intentar evitar overfitting introduciendo regularizaciones, considerando lr, número de épocas, etc.

Se pide:
- Construir ambas arquitecturas y entrenar los modelos. Durante el entrenamiento, en cada época, genera una frase de 10 palabras que comienza con "Artificial intelligence is". El objetivo es que observes como va mejorando la capacidad de generación del modelo (0.5p)
- Mostrar curvas de train loss, validation loss, accuracy validation comparando ambos modelos. Las gráficas deben ser legibles, informativas y claras. En caso contrario se considerarán incorrectas. (0.5p)
- Elegir de ambas arquitecturas el modelo con menor loss en validation y mostrar en una tabla formateada con valores de accuracy sobre test y sobre validation del mejor modelo. (0.5p)
- Explicar en dos líneas tus principales conclusiones al comparar ambas arquitecturas en este problema. (0.5p)

Consideraciones:
- Probablemente tengas que ejecutar el entrenamiento varias veces para afinar los parámetros, escalar los datos o el número de épocas.
- Antes de realizar este ejercicio, es recomendable leer el enunciado de los siguientes ejercicios. El motivo es que deberás conservar algunos checkpoints de este ejercicio para ser utilizados después.
</div>
## 1.5. Preparación de DataLoaders y Arquitecturas

### 1.5.1. Selección de hiperparámetros

Se ajustaron los hiperparámetros de ambas arquitecturas para mantener un número de parámetros entrenables comparable y dentro del rango solicitado en el enunciado (5–6 millones de parámetros).

Dado que el vocabulario obtenido tras la tokenización contiene 66.651 tokens, las capas de embeddings y proyección de salida dominan el tamaño total del modelo. Por este motivo se redujeron las dimensiones internas de embedding (`EMBED_DIM`) y representación oculta (`HIDDEN_DIM`) hasta obtener arquitecturas equilibradas y computacionalmente manejables.

```python
# SELECCIÓN DE HIPERPARÁMETROS

# Longitud máxima de cada secuencia de entrada
SEQ_LEN = 50

# Número de muestras procesadas por batch
BATCH_SIZE = 64

# Dimensión de los embeddings de palabras
EMBED_DIM = 40

# Tamaño del estado oculto de la LSTM
HIDDEN_DIM = 48

# Número de capas LSTM / Transformer
NUM_LAYERS = 2

# Probabilidad de desactivación en dropout
DROPOUT = 0.3

# Número de cabezas de atención del Transformer
NHEAD = 2

# Tasa de aprendizaje del optimizador
LEARNING_RATE = 1e-3

# Número máximo de épocas de entrenamiento
EPOCHS = 15

# Número de épocas sin mejora antes de parar
PATIENCE = 3
```

## 1.5.2. Adaptación de los datos a PyTorch

Se construyeron `DataLoaders` independientes para train, validation y test con el objetivo de alimentar los modelos en batches. En entrenamiento se activó `shuffle=True` para mejorar la generalización del modelo, mientras que en validación y test se mantuvo el orden original.

```python
# Adaptación de los datos a PyTorch

class LanguageModelDataset(Dataset):
    '''
    Dataset personalizado para modelado de lenguaje.

    Convierte cada secuencia y su palabra objetivo
    en tensores de PyTorch preparados para entrenamiento.
    '''
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        seq, target = self.sequences[idx]

        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )

# Creamos datasets para cada conjunto de datos
train_ds = LanguageModelDataset(train_sequences)
valid_ds = LanguageModelDataset(valid_sequences)
test_ds  = LanguageModelDataset(test_sequences)

# Creamos dataloaders para cada conjunto de datos, 
# con shuffle solo en el conjunto de entrenamiento
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
```

## 1.5.3. Diseño e implementacion de modelos sencillos: LSTM y Transformer

En este apartado se implementan dos arquitecturas secuenciales para generación de texto: un modelo basado en LSTM y otro basado en Transformer. Ambos modelos se han diseñado manteniendo una complejidad y número de parámetros similares, permitiendo así realizar una comparación más consistente entre ambas aproximaciones.

### A. LSTM - `LSTMLanguageModel`

La arquitectura LSTM procesa las secuencias de forma recurrente, manteniendo información contextual mediante estados ocultos. Se utilizaron dos capas LSTM y dropout para reducir overfitting y mantener una complejidad comparable con el Transformer.

```python
# LSTM

class LSTMLanguageModel(nn.Module):
    '''
    Modelo de lenguaje basado en LSTM.

    Procesa secuencias de texto mediante capas recurrentes
    y predice la siguiente palabra de la secuencia.
    '''
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout
    ):
        super().__init__()

        # la capa de embedding convierte índices de palabras a vectores densos
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        # la capa LSTM procesa las secuencias de embeddings palabra a palabra, 
        # manteniendo un estado oculto que captura información del contexto previo
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # el dropout se aplica a la salida del LSTM para evitar sobreajuste
        self.dropout = nn.Dropout(dropout)

        # la capa fully connected toma la salida del LSTM y produce logits para cada palabra en el vocabulario
        # Transforma el espacio de características del LSTM al espacio de vocabulario para hacer la predicción
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        # convertimos los índices de palabras a embeddings densos
        x = self.embedding(x)

        # el LSTM devuelve la salida para cada token de la secuencia, 
        # pero solo nos interesa la salida del último token
        out, _ = self.lstm(x)

        # solo nos interesa la salida del último token de la secuencia, que es donde el modelo hace su predicción
        out = out[:, -1, :]

        # aplicamos dropout a la salida del LSTM antes de pasarla a la capa fully connected
        out = self.dropout(out)

        # la capa fully connected produce los logits para cada palabra en el vocabulario, 
        # que luego se usarán para calcular la pérdida y hacer predicciones
        logits = self.fc(out)

        return logits
```

### B. Transformer - `TransformerLanguageModel`

El Transformer procesa toda la secuencia simultáneamente mediante mecanismos de atención. En este caso se utilizaron dos cabezas de atención (nhead=2) tal y como solicita el enunciado, incorporando además codificación posicional para conservar información sobre el orden de las palabras.

> Referencia Atention is all you need
```python
# TRANSFORMER

class PositionalEncoding(nn.Module):
    '''
    Implementa codificación posicional sinusoidal para Transformers.

    Permite al modelo incorporar información sobre
    la posición de cada token dentro de la secuencia.
    '''
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # creamos una matriz de codificación posicional de tamaño (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # cada posición se codifica usando funciones sinusoidales de diferentes frecuencias
        position = torch.arange(0, max_len).unsqueeze(1)

        # las dimensiones pares se codifican con funciones seno y las impares con funciones coseno
        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        # aplicamos las funciones seno y coseno a las posiciones para crear la codificación posicional
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # añadimos una dimensión extra para que se pueda sumar a los embeddings de las palabras
        pe = pe.unsqueeze(0)

        # registramos la matriz de codificación posicional como un buffer del módulo,
        self.register_buffer("pe", pe)

    def forward(self, x):

        # sumamos la codificación posicional a los embeddings de las palabras para que el modelo pueda distinguir
        # la posición de cada token
        return x + self.pe[:, :x.size(1)] # type: ignore

class TransformerLanguageModel(nn.Module):
    '''
    Modelo de lenguaje basado en Transformer Encoder.

    Utiliza mecanismos de atención para procesar
    toda la secuencia simultáneamente.
    '''
    def __init__(
        self,
        vocab_size,
        embed_dim,
        nhead,
        num_layers,
        dropout
    ):
        super().__init__()

        # la capa de embedding convierte índices de palabras a vectores densos
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        # la codificación posicional se suma a los embeddings de las palabras para que el modelo pueda distinguir 
        # la posición de cada token
        self.pos_encoding = PositionalEncoding(embed_dim)

        # el TransformerEncoderLayer es la unidad básica del Transformer, que incluye mecanismos de atención y feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        # el TransformerEncoder se compone de varias capas de TransformerEncoderLayer apiladas,
        # lo que permite al modelo capturar relaciones complejas en la secuencia
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # el dropout se aplica a la salida del Transformer para evitar sobreajuste
        self.dropout = nn.Dropout(dropout)

        # la capa fully connected toma la salida del Transformer y produce logits para cada palabra en el vocabulario,
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        # convertimos los índices de palabras a embeddings densos
        x = self.embedding(x)

        # sumamos la codificación posicional a los embeddings de las palabras para que el modelo pueda distinguir
        x = self.pos_encoding(x)

        # el Transformer procesa toda la secuencia a la vez usando mecanismos de atención,
        x = self.transformer(x)
        x = x[:, -1, :]

        # aplicamos dropout a la salida del Transformer antes de pasarla a la capa fully connected
        x = self.dropout(x)

        # la capa fully connected produce los logits para cada palabra en el vocabulario, 
        # que luego se usarán para calcular la pérdida y hacer predicciones
        logits = self.fc(x)

        return logits
```

## 1.5.4. Inicialización de los modelos

Observando el output de la función de conteo (`count_parameters()`), se confirma que ambos modelos cumplen la restricción indicada en el enunciado, manteniendo entre 5 y 6 millones de parámetros entrenables. Además, las arquitecturas son comparables en profundidad y complejidad, permitiendo realizar una comparación más justa entre LSTM y Transformer.

```python
# Inicializamos ambos modelos con los mismos hiperparámetros para una comparación justa

VOCAB_SIZE = len(vocab)

lstm_model = LSTMLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

transformer_model = TransformerLanguageModel(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("LSTM params:", count_parameters(lstm_model))
print("Transformer params:", count_parameters(transformer_model))
```
LSTM params: 5968035
Transformer params: 5744027

## 1.5.5. Generación de texto

Mediante la función de generacion de texto (`generate_text()`), se observa cualitativamente cómo evoluciona el modelo durante el entrenamiento. A medida que las épocas avanzan, se espera que las frases generadas pasen de repeticiones simples a secuencias más coherentes y estructuradas.

```python
# Generación de texto

def generate_text(model, seed_text, max_words=10):
    '''
    Genera texto a partir de un texto semilla utilizando el modelo entrenado.
    '''
    # el modelo se pone en modo evaluación para desactivar dropout y otros comportamientos específicos de entrenamiento
    model.eval()

    # tokenizamos el texto semilla y lo convertimos a índices usando el vocabulario
    words = seed_text.lower().split()

    # iteramos para generar max_words palabras adicionales
    for _ in range(max_words):
        encoded = [
            vocab.get(w, vocab["<UNK>"])
            for w in words
        ]

        # solo consideramos las últimas SEQ_LEN palabras para generar la siguiente palabra,
        encoded = encoded[-SEQ_LEN:]

        # padding izquierda para asegurar que la secuencia tenga longitud SEQ_LEN, rellenando con el token <PAD> si es necesario
        encoded = [vocab["<PAD>"]] * (SEQ_LEN - len(encoded)) + encoded

        # convertimos la secuencia de índices a un tensor y la movemos al dispositivo (CPU o GPU)
        x = torch.tensor(encoded).unsqueeze(0).to(DEVICE)


        # hacemos una pasada hacia adelante por el modelo para obtener los logits de la siguiente palabra
        with torch.no_grad():
            logits = model(x)

        # obtenemos el índice de la palabra con mayor logit, que es la predicción del modelo para la siguiente palabra
        pred = torch.argmax(logits, dim=-1).item()

        # convertimos el índice de la palabra predicha de vuelta a una palabra usando el vocabulario inverso
        next_word = idx2word.get(pred, "<UNK>") # type: ignore

        # añadimos la palabra predicha a la lista de palabras para generar la siguiente palabra en la próxima iteración
        words.append(next_word)

    return " ".join(words)
```

## 1.5.6. Evaluación de los modelos

Durante la evaluación mediante la función (`evaluate_model()`) se calcularon tanto la pérdida (`loss`) como la precisión (`accuracy`) sobre validación y test. Estas métricas permiten comparar el rendimiento de ambas arquitecturas y detectar posibles problemas de sobreajuste.

```python
# Evaluacion del modelo en el conjunto de validación o test

def evaluate_model(model, loader, criterion):
    '''
    Evalúa el modelo en un conjunto de datos dado por el loader, calculando la pérdida promedio y la precisión.
    '''

    # el modelo se pone en modo evaluación para desactivar dropout y otros comportamientos específicos de entrenamiento
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # iteramos por el loader sin calcular gradientes para evaluar el modelo en el conjunto de validación o test
    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = criterion(logits, y)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total
```

## 1.5.7. Entrenamiento de los modelos

Durante el entrenamiento mediante la función (`train_model()`) se almacenaron métricas de `loss` y `accuracy` para posteriormente comparar ambas arquitecturas mediante gráficas. Además, se implementó `early stopping` para detener automáticamente el entrenamiento cuando la pérdida de validación (`loss validation`) deja de mejorar, reduciendo así el riesgo de overfitting y el tiempo de entrenamiento innecesario.

```python
# Entrenamiento del modelo con early stopping

def train_model(model, train_loader, valid_loader, epochs):
    '''
    Entrena el modelo utilizando el conjunto de entrenamiento y evalúa su rendimiento en el 
    conjunto de validación después de cada época.
    '''
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_acc": []
    }

    best_model = None
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x, y in tqdm(train_loader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            # Evita exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )
            optimizer.step()
            running_loss += loss.item()

        # ====================================================
        # MÉTRICAS TRAIN
        # ====================================================
        train_loss = running_loss / len(train_loader)

        # ====================================================
        # VALIDACIÓN
        # ====================================================
        valid_loss, valid_acc = evaluate_model(
            model,
            valid_loader,
            criterion
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        # ====================================================
        # LOGS
        # ====================================================
        print(f"\nEpoch {epoch+1}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Valid loss: {valid_loss:.4f}")
        print(f"Valid acc : {valid_acc:.4f}")

        generated = generate_text(
            model,
            "Artificial intelligence is",
            max_words=10
        )

        print("\nGenerated text:")
        print(generated)

        # ====================================================
        # EARLY STOPPING
        # ====================================================
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0
            print("\nValidation loss improved")

        else:
            epochs_without_improvement += 1
            print(
                f"\nNo improvement for "
                f"{epochs_without_improvement} epoch(s)"
            )

            if epochs_without_improvement >= PATIENCE:
                print("\nEarly stopping activated")
                break

    # ========================================================
    # RECUPERAR MEJOR MODELO
    # ========================================================
    model.load_state_dict(best_model)
    return model, history
```

## 1.5.7. Resultados del entrenamiento

Los resultados muestran que ambos modelos aprenden progresivamente durante las primeras épocas, reduciendo el `train loss` y mejorando ligeramente la `validation accuracy`. Sin embargo, la `validation loss` deja de mejorar rápidamente, por lo que el mecanismo de `early stopping` detiene el entrenamiento para evitar sobreajuste.

En las frases generadas se aprecia que, al inicio, ambos modelos repiten palabras muy frecuentes como *“the”*, algo habitual en modelos pequeños entrenados sobre un vocabulario muy grande. A medida que avanzan las épocas, comienzan a aparecer estructuras algo más coherentes, aunque todavía simples.

También se observa que el Transformer requiere bastante más tiempo por época que el LSTM, algo esperable debido al coste computacional del mecanismo de atención. Finalmente, las métricas de validación y test son muy similares entre ambos modelos, indicando que las dos arquitecturas presentan un rendimiento comparable en este problema.

```python
# ============================================================
# INICIAR ENTRENAMIENTO
# ============================================================

lstm_model, lstm_history = train_model(
    lstm_model,
    train_loader,
    valid_loader,
    epochs=EPOCHS
)

transformer_model, transformer_history = train_model(
    transformer_model,
    train_loader,
    valid_loader,
    epochs=EPOCHS
)
```

100%|██████████| 938/938 [01:46<00:00,  8.84it/s]

Epoch 1
Train loss: 7.8532
Valid loss: 7.4244
Valid acc : 0.0635

Generated text:
artificial intelligence is the the the the the the the the the the

Validation loss improved
100%|██████████| 938/938 [01:43<00:00,  9.09it/s]

Epoch 2
Train loss: 7.1418
Valid loss: 7.3765
Valid acc : 0.0960

Generated text:
artificial intelligence is the the the , , the the the the ,

Validation loss improved
100%|██████████| 938/938 [01:45<00:00,  8.89it/s]

Epoch 3
Train loss: 6.9428
Valid loss: 7.3549
Valid acc : 0.1118

Generated text:
artificial intelligence is the the the the the the the the the the

Validation loss improved
100%|██████████| 938/938 [01:41<00:00,  9.26it/s]

Epoch 4
Train loss: 6.7999
Valid loss: 7.3762
Valid acc : 0.1136

Generated text:
artificial intelligence is the the the the the the the the the the

No improvement for 1 epoch(s)
100%|██████████| 938/938 [01:49<00:00,  8.59it/s]

Epoch 5
Train loss: 6.6809
Valid loss: 7.3798
Valid acc : 0.1168

Generated text:
artificial intelligence is the the first of the first of the the first

No improvement for 2 epoch(s)
100%|██████████| 938/938 [01:44<00:00,  9.01it/s]

Epoch 6
Train loss: 6.5719
Valid loss: 7.4448
Valid acc : 0.1179

Generated text:
artificial intelligence is the first of the first of the first of the

No improvement for 3 epoch(s)

Early stopping activated
100%|██████████| 938/938 [08:02<00:00,  1.94it/s]

Epoch 1
Train loss: 7.8977
Valid loss: 7.2945
Valid acc : 0.1022

Generated text:
artificial intelligence is the the the the the the the the the the

Validation loss improved
100%|██████████| 938/938 [08:02<00:00,  1.94it/s]

Epoch 2
Train loss: 7.0966
Valid loss: 7.2491
Valid acc : 0.1092

Generated text:
artificial intelligence is the first , the first , the first , the

Validation loss improved
100%|██████████| 938/938 [08:03<00:00,  1.94it/s]

Epoch 3
Train loss: 6.8787
Valid loss: 7.2640
Valid acc : 0.1121

Generated text:
artificial intelligence is the first , and the first , and the first

No improvement for 1 epoch(s)
100%|██████████| 938/938 [07:58<00:00,  1.96it/s]

Epoch 4
Train loss: 6.7407
Valid loss: 7.2996
Valid acc : 0.1116

Generated text:
artificial intelligence is the first , the first , the first , the

No improvement for 2 epoch(s)
100%|██████████| 938/938 [07:58<00:00,  1.96it/s]

Epoch 5
Train loss: 6.6356
Valid loss: 7.3446
Valid acc : 0.1144

Generated text:
artificial intelligence is the first " , and the first , and the

No improvement for 3 epoch(s)

Early stopping activated

## 1.5.8. Visualización de la evolucion por modelo

PON EL TEXTO AQUI

```python
# ============================================================
# PLOTS
# ============================================================

lstm_epochs = range(
    1,
    len(lstm_history["train_loss"]) + 1
)

transformer_epochs = range(
    1,
    len(transformer_history["train_loss"]) + 1
)

# ============================================================
# LOSS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    lstm_epochs,
    lstm_history["train_loss"],
    label="LSTM Train Loss"
)

plt.plot(
    lstm_epochs,
    lstm_history["valid_loss"],
    label="LSTM Validation Loss"
)

plt.plot(
    transformer_epochs,
    transformer_history["train_loss"],
    label="Transformer Train Loss"
)

plt.plot(
    transformer_epochs,
    transformer_history["valid_loss"],
    label="Transformer Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# VALIDATION ACCURACY
# ============================================================
plt.figure(figsize=(10, 6))

plt.plot(
    lstm_epochs,
    lstm_history["valid_acc"],
    label="LSTM Validation Accuracy"
)

plt.plot(
    transformer_epochs,
    transformer_history["valid_acc"],
    label="Transformer Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid()
plt.show()
```

## 1.5.9. Evaluación y comparación del rendimiento de los modelos

La tabla muestra los resultados de ambos modelos sobre los conjuntos de validación y test. 

El Transformer obtiene una menor pérdida de validación, siendo seleccionado como el mejor modelo, aunque ambas arquitecturas presentan valores de `accuracy` muy similares.

Los resultados son coherentes para con la dificultad de predecir la siguiente palabra exacta sobre un vocabulario de  más de 66.000 tokens con modelos de entre 5 y 6 millones de parámetros. Es de destacar que el LSTM supera ligeramente al Transformer en `Validation accuracy` con un coste computacional por época significativamente menor, lo que puede ser relevante en escenarios con recursos limitados.

```python
criterion = nn.CrossEntropyLoss()

lstm_val_loss, lstm_val_acc = evaluate_model(
    lstm_model,
    valid_loader,
    criterion
)

lstm_test_loss, lstm_test_acc = evaluate_model(
    lstm_model,
    test_loader,
    criterion
)

tr_val_loss, tr_val_acc = evaluate_model(
    transformer_model,
    valid_loader,
    criterion
)

tr_test_loss, tr_test_acc = evaluate_model(
    transformer_model,
    test_loader,
    criterion
)

results = [
    ["LSTM", lstm_val_acc, lstm_test_acc],
    ["Transformer", tr_val_acc, tr_test_acc]
]

print(
    tabulate(
        results,
        headers=["Modelo", "Validation Accuracy", "Test Accuracy"],
        tablefmt="grid",
        floatfmt=".4f"
    )
)
```

+-------------+-------------------+-----------------------+-----------------+
| Modelo      |   Validation Loss |   Validation Accuracy |   Test Accuracy |
+=============+===================+=======================+=================+
| LSTM        |            7.3549 |                0.1118 |          0.1105 |
+-------------+-------------------+-----------------------+-----------------+
| Transformer |            7.2491 |                0.1092 |          0.1080 |
+-------------+-------------------+-----------------------+-----------------+
El mejor modelo de acuerdo a la perdida de validación es: Transformer