# Apuntes Completos — Transformadores (Teoría + Práctica)

## Deep Learning · M2.875 · Máster en Ciencia de Datos

# PARTE I: FUNDAMENTOS TEÓRICOS

## 1. ¿Qué es un Transformador?

Un **transformador** es una arquitectura de red neuronal diseñada originalmente para la transformación de datos secuencia a secuencia (seq2seq), propuesta como alternativa a las redes recurrentes. Desde su introducción en el paper seminal **"Attention is All You Need" (Vaswani et al., 2017)**, se ha convertido en una arquitectura generalista capaz de competir y superar a redes especializadas (recurrentes y convolucionales) en texto, imagen, audio, aprendizaje por refuerzo y datos multimodales.

El elemento central y distintivo de la arquitectura es el **mecanismo de autoatención** (_self-attention_): la capacidad de que cada elemento de una secuencia atienda a todos los demás elementos a la vez, sin necesidad de procesar la secuencia paso a paso.

---

## 2. Por qué los transformadores: limitaciones de las RNN

Antes de los transformadores, las **redes recurrentes (RNN/LSTM)** eran el estándar para datos secuenciales. Tienen tres tipos de problemas fundamentales:

1. **Dependencias de largo alcance.** Una red recurrente mantiene un estado interno que se va actualizando token a token. La información de tokens lejanos puede "diluirse" al propagarse a través de muchos pasos. Las LSTM mitigaron este problema parcialmente, pero no lo eliminaron.

2. **Vanishing/exploding gradients.** Al desplegar una RNN de longitud L durante el entrenamiento, los gradientes deben propagarse a través de L capas. Esto causa que se desvanezcan o exploten, dificultando la optimización.

3. **No paralelizable.** La naturaleza secuencial de las RNN impide calcular el estado del token t sin haber calculado antes el estado t-1. Esto limita mucho el aprovechamiento de GPUs modernas.

Los transformadores **eliminan la recurrencia**: todos los tokens se procesan en paralelo, las dependencias de largo alcance se modelan directamente mediante atención, y los problemas de gradiente dependen únicamente de la profundidad de la red (resuelta con redes residuales desde He et al., 2016).

---

## 3. El mecanismo de autoatención

### 3.1 Definición formal

Sea **X** la secuencia de entrada ($ℓ_X$ elementos, dimensión $d_e$) y **Z** la secuencia de contexto ($ℓ_Z$ elementos). Se definen tres transformaciones lineales:

- $Q = W_q · X + b_q$    _(Query — "lo que busco")_
- $K = W_k · Z + b_k$    _(Key — "lo que ofrezco")_
- $V = W_v · Z + b_v$    _(Value — "el contenido que entrego")_

La **matriz de puntuación** (similitud escalada) es:

$$S = (1/√d_attn) · K^T · Q$$


La **salida** es:

$$Ṽ = V · softmax(S)$$


Cuando X = Z (secuencia de contexto igual a la de entrada), se llama **autoatención** (_self-attention_).

### 3.2 Analogía con bases de datos

La terminología Q/K/V proviene de una analogía con bases de datos relacionales: ante una consulta (Query), se compara con todas las claves (Keys) para obtener una puntación de similitud, y el resultado es una combinación ponderada de los valores (Values) según esa similitud. En los transformadores, esta búsqueda es **probabilística y diferenciable**.

### 3.3 El papel de la máscara (_mask_)

La operación de máscara se aplica sobre la matriz de puntuación S antes del softmax, poniendo −∞ en las posiciones que no deben atenderse:

- **Sin máscara** (Mask ≡ 1): contexto **bidireccional** — cada token atiende a todos los demás. Típico de modelos tipo BERT (encoders).
- **Con máscara causal** (Mask[t,t'] = [[t ≤ t']]): contexto **unidireccional** — cada token solo atiende a los anteriores. Típico de GPT (decoders).
- **Máscara cruzada**: útil en arquitecturas codificador-decodificador para que el decoder atienda al encoder.

### 3.4 Autoatención múltiple (_Multi-Head Attention_)

En vez de una sola transformación Q/K/V, se aplican **H transformaciones en paralelo** (cabezales), cada una con sus propios parámetros:

```
M = Concat[D_1(Q_1,K_1,V_1), ..., D_H(Q_H,K_H,V_H)] · W_O
```

Si H = d / d_v, la dimensión de salida iguala la de entrada. Beneficios: más diversidad en las comparaciones, mayor capacidad representacional.

**Intuición:** la primera capa compara pares de atributos, la segunda compara pares de pares, y así sucesivamente. En las capas profundas, el número de atributos combinados crece exponencialmente.

---

## 4. Arquitectura completa del Transformador

La arquitectura original (Vaswani et al., 2017) consta de:

```
Entrada → Embedding + Positional Encoding → N × Encoder → N × Decoder → Linear + Softmax → Salida
```

### 4.1 Tratamiento de entrada

1. **Tokenización**: el texto se divide en _tokens_ (subpalabras, palabras o caracteres). Los métodos más comunes son BPE (_Byte Pair Encoding_), Unigram y WordPiece.
2. **Embedding de token**: cada token se mapea a un vector denso (representación numérica).
3. **Positional Encoding**: se suma una señal que codifica la posición del token en la secuencia (normalmente funciones seno/coseno de diferentes frecuencias). Esto es necesario porque la autoatención, a diferencia de las RNN, no tiene noción implícita del orden.

### 4.2 Bloque Encoder

Cada bloque encoder contiene:

1. Multi-Head Self-Attention (sin máscara, bidireccional)
2. Suma residual + LayerNorm
3. Red completamente conectada (FFN): dos capas lineales con activación ReLU/GELU
4. Suma residual + LayerNorm

### 4.3 Bloque Decoder

Cada bloque decoder contiene:

1. **Masked** Multi-Head Self-Attention (con máscara causal sobre la secuencia de salida)
2. Suma residual + LayerNorm
3. Multi-Head **Cross-Attention** (Q viene del decoder, K y V del último encoder)
4. Suma residual + LayerNorm
5. FFN
6. Suma residual + LayerNorm

### 4.4 Normalización de capas (_Layer Normalization_)

Presente en todos los bloques. Controla explícitamente la media y varianza de las activaciones de cada neurona. A diferencia de _batch normalization_, opera sobre la dimensión de características de cada elemento, no del batch.

### 4.5 Tratamiento de salida

La capa de salida realiza la transformación inversa del embedding: toma la representación interna y produce una distribución de probabilidad sobre el vocabulario mediante una capa lineal + softmax.

---

## 5. Tipos de transformadores

### 5.1 Secuencia a secuencia (Encoder-Decoder)

- Usa codificadores y decodificadores.
- Aplicaciones: traducción automática, resumen, respuesta a preguntas.
- Ejemplo: el modelo original Vaswani et al. (2017), **T5** (Raffel et al., 2020).

**Proceso:** la secuencia de entrada se codifica completamente; luego el decoder genera la salida token a token, condicionado a la representación del encoder y a los tokens ya generados.

### 5.2 Bidireccional / Autocodificación (Solo Encoder)

- Solo usa la parte codificadora. Sin máscara → contexto bidireccional (cada token atiende a todos).
- Preentrenamiento: se corrompe la entrada aleatoriamente (enmascarando tokens) y el modelo aprende a reconstruirla (**Masked Language Modeling, MLM**).
- Aplicaciones principales: clasificación de textos, extracción de información.
- Ejemplo: **BERT** (Devlin et al., 2018), RoBERTa, ALBERT, DistilBERT.

**BERT-base**: 12 capas encoder, 12 cabezales, d_e=768, ~110M parámetros.
**BERT-mini** (gaunernst/bert-mini-uncased): 4 capas, 4 cabezales, d_e=256, ~11M parámetros.
**DistilBERT**: versión destilada de BERT-base con 6 capas, d_e=768, ~66M parámetros. Retiene ~97% del rendimiento con la mitad de los parámetros.

### 5.3 Autorregresivo (Solo Decoder)

- Solo usa la parte decodificadora. Con máscara causal → contexto unidireccional.
- Preentrenamiento: predicción del siguiente token.
- Aplicaciones: generación de texto, chatbots.
- Ejemplo: familia **GPT** (GPT-1: 117M, GPT-2: 1.5B, GPT-3: 175B parámetros).

**Diferencia clave con BERT:** GPT ve el contexto solo hacia la izquierda (tokens anteriores); BERT ve el contexto completo en ambas direcciones.

---

## 6. Codificación posicional (_Positional Encoding_)

Como la autoatención es invariante a la permutación de los tokens (si cambias el orden, el resultado no cambia sin información posicional), es necesario inyectar información sobre la posición. El método original usa funciones sinusoidales:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

El embedding final es la suma del embedding del token y su codificación posicional.

---

## 7. Embeddings pre-transformer

Antes de los transformadores, se usaban embeddings estáticos: **Word2Vec**, **GloVe**, **FastText**. Representan palabras como vectores densos con propiedades interesantes (palabras similares tienen vectores cercanos; relaciones lineales: rey − hombre + mujer ≈ reina). Su limitación es que no capturan contexto: la misma palabra tiene siempre el mismo vector independientemente del contexto.

Los transformadores aprenden los embeddings de forma contextual durante el entrenamiento completo (end-to-end).

---

## 8. Transformadores para Visión (ViT)

La única diferencia esencial frente a los transformadores de texto es el **tokenizador**: la imagen se divide en parches (_patches_) pequeños no superpuestos (e.g., 16×16 píxeles), que se tratan como tokens. A cada parche se le suma un embedding posicional.

Los modelos ViT han demostrado rendimiento comparable o superior a CNNs en clasificación, detección y segmentación. Los errores de ViT son además más consistentes con la percepción humana que los de las CNNs.

**Variantes de clasificación:**

1. Adaptación directa del ViT original (patches directos)
2. Atributos de una CNN preentrenada alimentados al transformer
3. Destilación de conocimiento desde una CNN maestra
4. Atención localizada/jerárquica (e.g., Swin Transformer)

**Para detección:** DETR (Detection Transformer) — usa secuencia de "object queries" aleatorias como entrada del decoder, y la función de pérdida resuelve el emparejamiento óptimo con Hungarian matching.

**Para segmentación:** MaskFormer/Mask2Former — combina un backbone (CNN o ViT) con un decoder tipo DETR para generar máscaras de segmentación semántica, de instancias y panóptica con una única arquitectura.

---

## 9. Transformadores para Audio

El tokenizador habitual es el **espectrograma de MEL**: representación 2D de la señal de audio en dominio de frecuencia (escala MEL ≈ percepción humana de las frecuencias). Sobre él se aplican convoluciones para reducir las dimensiones al espacio interno del transformer.

**Ejemplos relevantes:**

- **Whisper** (Radford et al., 2022): transformer seq2seq multilingüe para transcripción de voz.
- **Wav2Vec 2.0**: similar a BERT pero para audio; aprende representaciones no supervisadas de secuencias de audio.

---

## 10. Transformadores Multimodales

Combinan entradas de distinta naturaleza (texto + imagen, texto + audio, etc.). Los transformadores son especialmente adecuados porque su espacio de modelado es equivalente a un grafo completamente conectado, sin los sesgos inductivos de las CNNs o RNNs que podrían no ser adecuados para ciertos tipos de datos.

**Estrategias de fusión en la autoatención:**

|Estrategia|Descripción|
|---|---|
|Suma temprana|Se suman los vectores de los distintos canales antes de la atención|
|Concatenación temprana|Se concatenan los vectores y se procesan juntos|
|Jerárquica multi→mono|Cada canal tiene su propio transformer, luego se fusionan|
|Jerárquica mono→multi|Un transformer conjunto primero, luego se separan|
|Cruzada|El Query de un canal actúa sobre las claves del otro|

**Aplicaciones:** MMBT (imagen+texto), VideoBERT (vídeo+texto), Med-BERT (registros médicos), Gato (DeepMind, 2022: modelo generalista con 604 tareas, imagen+texto+control de robots).

---

# PARTE II: PRÁCTICA — PEC3 TRANSFORMERS CON PYTORCH

## Ejercicio 1: Preparación de datos para generación de texto

**Dataset:** WikiText-2-raw-v1 (HuggingFace). Texto extraído de Wikipedia en inglés.

**Pipeline de preprocesamiento:**

1. Tokenización a nivel de palabra (lowercase + split por espacios).
2. Construcción del vocabulario desde el conjunto de entrenamiento (66.651 tokens). Tokens especiales: `<PAD>` (índice 0), `<UNK>` (índice 1).
3. Generación de secuencias con **ventana deslizante**: por cada texto, se crea una secuencia de hasta 50 tokens de contexto y la palabra siguiente como target. Padding a la izquierda.
4. Sub-muestreo: de los ~2M de secuencias generadas, se toman 60.000 (train), 8.000 (validation), 8.000 (test).

```python
# Estructura de cada muestra:
# INPUT:  secuencia de 50 tokens (con padding izquierdo si es necesario)
# TARGET: siguiente token (índice en el vocabulario)
```

**Tabla resumen del dataset:**

|Subconjunto|Textos originales|Secuencias generadas|Secuencias finales|
|---|---|---|---|
|Train|23.767|2.028.143|60.000|
|Validation|2.461|211.425|8.000|
|Test|2.891|238.320|8.000|

---

## Ejercicio 2: LSTM vs Transformer para generación de texto

**Objetivo:** comparar dos arquitecturas de ~5-6M de parámetros entrenables sobre la misma tarea de predicción de la siguiente palabra.

### Arquitectura LSTM

```
Embedding(66651, 40) → LSTM(input=40, hidden=48, layers=2, dropout=0.3) → Dropout → Linear(48, 66651)
```

- Procesa los tokens secuencialmente, manteniendo estado oculto.
- Solo usa la salida del último token para predecir la siguiente palabra.

### Arquitectura Transformer

```
Embedding(66651, 40) → PositionalEncoding → 2×TransformerEncoderLayer(d_model=40, nhead=2) → Dropout → Linear(40, 66651)
```

- Codificación posicional sinusoidal.
- Capas personalizadas `TransformerEncoderLayerWithAttn` con `need_weights=True` para extraer pesos de atención.
- Procesa toda la secuencia en paralelo; usa el último token para predecir.

### Hiperparámetros

```python
SEQ_LEN = 50, BATCH_SIZE = 64, EMBED_DIM = 40, HIDDEN_DIM = 48
NUM_LAYERS = 2, DROPOUT = 0.3, NHEAD = 2, LEARNING_RATE = 1e-3, EPOCHS = 15, PATIENCE = 3
```

### Resultados

|Modelo|Val Loss|Val Accuracy|Test Accuracy|
|---|---|---|---|
|LSTM|7.3568|10.72%|10.47%|
|**Transformer**|**7.3122**|**10.92%**|**10.52%**|

**Conclusiones:**

- El Transformer obtiene menor pérdida de validación → mejor seleccionado como modelo.
- El LSTM es más rápido por época (mayor tiempo de cómputo del mecanismo de atención).
- Las métricas de accuracy son bajas (~11%) pero esperables dado el vocabulario de 66.651 palabras.
- El _early stopping_ actúa en torno a la época 3, frenando el sobreajuste temprano.
- Evolución del texto generado: de "the the the the…" (época 1) a "the first of the first…" (época final), mostrando aprendizaje progresivo.

---

## Ejercicio 3: Visualización de la autoatención

Se extrajeron y visualizaron los **pesos de autoatención** del Transformer de la época 1 vs. época final para la frase: _"a new tricycle undercarriage was fitted , with the main"_.

### Matriz de self-attention

- **Época 1:** atención dispersa y uniforme entre todos los pares de tokens.
- **Época final:** atención más selectiva, concentrada en tokens semánticamente relevantes como _tricycle_ (sustantivo principal) y _main_. Las palabras menos informativas (_a_, artículos) reciben menor peso.

### Token alignment (estilo BertViz)

Visualización de las conexiones query→key con opacidad proporcional al peso de atención.

- **Época 1:** conexiones difusas, similares entre tokens.
- **Época final:** patrones más definidos, con _tricycle_ y _undercarriage_ como puntos focales.

**Interpretación:** A medida que el modelo aprende, la atención se vuelve más especializada y significativa, enfocándose en las palabras con mayor carga semántica.

```python
# Para extraer pesos de atención:
logits, all_attn = model(x, return_attentions=True)
# all_attn: lista de tensores (uno por capa)
# all_attn[0]: pesos de la primera capa, shape (batch, seq, seq)
```

---

## Ejercicio 4: Fine-tuning downstream con BERT-mini

### Modelo: `gaunernst/bert-mini-uncased`

Arquitectura:

- 4 capas encoder, 4 cabezales de atención, d_e=256, FFN=1024
- **11.171.074 parámetros entrenables**
- Tokenizador: WordPiece, vocabulario de 30.522 tokens

### Tarea: Clasificación de sentimiento binaria — Dataset IMDB

25.000 muestras de entrenamiento, 25.000 de test (reseñas de cine: positivo/negativo).

### Estrategia de fine-tuning

```python
# Adaptación downstream: se reemplaza la cabeza original por una capa de clasificación
model = AutoModelForSequenceClassification.from_pretrained("gaunernst/bert-mini-uncased", num_labels=2)
# Hiperparámetros
MAX_LENGTH = 256, BATCH_SIZE = 32, LR = 2e-5, WEIGHT_DECAY = 0.01, EPOCHS = 5, PATIENCE = 2
# Optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)
# Scheduler lineal decreciente
lr_scheduler = get_scheduler("linear", ...)
```

Se aplica **padding dinámico** por batch (`DataCollatorWithPadding`) para reducir consumo de memoria.

### Resultados por época

|Época|Train Loss|Val Loss|Val Acc|Test Acc|
|---|---|---|---|---|
|1|0.4197|0.3246|85.72%|85.91%|
|2|0.3063|0.3061|86.92%|87.13%|
|**3**|**0.2643**|**0.2910**|**87.84%**|**87.75%**|
|4|0.2405|0.2991|87.64%|87.92%|
|5|0.2206|0.3035|87.80%|88.07%|

**Mejor modelo: época 3** (menor val_loss = 0.2910). **Accuracy final en test: 88%**, tiempo de inferencia: **21.73s**(~0.87ms/muestra).

---

## Ejercicio 5: Fine-tuning DistilBERT (modelo completo)

### Modelo: `distilbert-base-uncased`

Arquitectura:

- 6 capas transformer, 12 cabezales, d_e=768, FFN=3072
- **66.955.010 parámetros entrenables** (~6× más que BERT-mini)
- No se congelaron capas: el dataset IMDB (25.000 muestras) es suficientemente grande, y DistilBERT ya es una arquitectura comprimida.

### Resultados

| Época | Train Loss | Val Loss   | Val Acc    | Test Acc   |
| ----- | ---------- | ---------- | ---------- | ---------- |
| **1** | **0.2921** | **0.2566** | **90.52%** | **90.22%** |
| 2     | 0.1713     | 0.2663     | 89.84%     | 90.86%     |
| 3     | 0.1011     | 0.3096     | 89.96%     | 91.01%     |

**Mejor modelo: época 1** (menor val_loss). **Accuracy final en test: ~91%**, tiempo de inferencia: **215.39s**(~8.6ms/muestra).

**Observación importante:** DistilBERT es ~10× más lento en inferencia que BERT-mini, con solo ~3 puntos más de accuracy. El trade-off depende del caso de uso.

---

## Ejercicio 6: Modelo personalizado desde cero (MiniBERT-custom)

**Objetivo:** construir y entrenar desde cero un transformer tipo BERT que sea 5-10× más rápido en inferencia que BERT-mini.

### Diseño: `MiniBertForSequenceClassification`

```
Configuración:
  hidden_size = 160
  num_hidden_layers = 2
  num_attention_heads = 4
  intermediate_size = 256 (FFN)
  max_position_embeddings = 128
  MAX_LENGTH = 64 tokens (vs 256 en los anteriores)
  
Módulos:
  MiniEmbeddings: word + position + token_type embeddings
  MiniEncoder: 2 × MiniTransformerLayer
    MiniTransformerLayer = MiniSelfAttention + Norm + FFN + Norm + conexiones residuales
  Pooler: linear(160, 160) + Tanh sobre token [CLS]
  Classifier: linear(160, 2)
  
Total: 5.302.754 parámetros
```

**Diferencias clave respecto a fine-tuning:**

- Pesos inicializados aleatoriamente (normal(0, 0.02))
- LR más alto: 3e-4 (vs 2e-5 en fine-tuning)
- Warmup del 10% de los pasos
- Gradient clipping con max_norm=1.0

### Resultados

|Época|Train Loss|Val Loss|Val Acc|Test Acc|
|---|---|---|---|---|
|1|0.6342|0.4774|76.76%|74.80%|
|**2**|**0.4178**|**0.4704**|**77.60%**|**74.61%**|
|3|0.2975|0.5545|77.56%|73.97%|
|4|0.2073|0.6828|76.28%|72.82%|

**Mejor modelo: época 2**. **Accuracy final en test: 72.82%**, tiempo de inferencia: **4.29s** (~0.17ms/muestra).

**Resultado:** 5× más rápido que BERT-mini, con 15 puntos menos de accuracy (esperable al entrenar sin conocimiento previo).

---

## Ejercicio 7: Knowledge Distillation

### Concepto

_Knowledge Distillation_ (Hinton et al., 2015): un modelo **teacher** grande y preciso transfiere su conocimiento a un modelo **student** pequeño mediante las distribuciones de probabilidad suavizadas de su capa de salida (_soft targets_), en lugar de las etiquetas duras del dataset.

**Intuición:** los soft targets contienen más información que las etiquetas duras porque reflejan la "confianza relativa" del teacher sobre todas las clases, no solo la correcta.

### Función de pérdida combinada

```python
def distillation_loss(student_logits, teacher_logits, labels, alpha, temperature):
    # Soft targets: suavizar con temperatura T
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T**2)
    # Hard labels
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss
```

**Parámetros:**

- **T (temperatura):** valores > 1 suavizan las distribuciones del teacher, amplificando la información de las clases incorrectas. Se usó T=4.0.
- **α (alpha):** pondera la contribución de KD loss vs CE loss. Se usó α=0.7 (70% del teacher).

### Configuración

- **Teacher:** mejor checkpoint de DistilBERT (época 1, val_loss=0.2461). **Pesos congelados** durante todo el proceso.
- **Student:** mejor checkpoint de MiniBERT-custom (época 2). Parte ya entrenado, no desde cero.
- LR=1e-4 (menor que en el ejercicio 6, porque el student ya tiene una base).
- DataLoaders del ejercicio 6 (MAX_LENGTH=64, batch_size=64).

### Resultados

|Época|Train Loss|Val Loss|Val Acc|Test Acc|
|---|---|---|---|---|
|1|0.4070|0.4706|77.88%|75.38%|
|2|0.3408|0.4349|78.68%|76.53%|
|3|0.3062|0.4275|78.00%|76.17%|
|**4**|**0.2853**|**0.4267**|**77.80%**|**76.26%**|

**Mejor modelo: época 4**. **Accuracy final en test: 76.16%**, tiempo de inferencia: **4.76s** (~0.19ms/muestra).

---

## Tabla comparativa final de los 4 modelos

|Modelo|Parámetros|Accuracy Test|Tiempo Inferencia|
|---|---|---|---|
|bert-mini-uncased (fine-tuning)|11.171.074|**88.07%**|21.73 s|
|distilbert-base-uncased (fine-tuning)|66.955.010|**91.01%**|215.39 s|
|minibert-custom (desde cero)|5.302.754|72.82%|**4.29 s**|
|minibert-kd (Knowledge Distillation)|5.302.754|**76.16%**|**4.76 s**|

**Conclusiones generales:**

- DistilBERT maximiza precisión pero es 10× más lento que BERT-mini.
- BERT-mini ofrece buen equilibrio: 88% de accuracy en ~21s de inferencia.
- Entrenar desde cero un modelo pequeño da peores resultados (73%) pero es el más rápido.
- **Knowledge Distillation mejora +3.3 puntos** al modelo pequeño sin incrementar el coste de inferencia, siendo la estrategia más eficiente para modelos compactos en producción.

---

## Reflexión sobre los ejercicios 4-7

La destilación de conocimiento es la técnica más eficiente estudiada para modelos compactos: permite extraer rendimiento del teacher sin modificar la arquitectura ni el coste de inferencia del student. El parámetro α es crítico: valores altos (> 0.7) hacen que el student ignore las etiquetas reales; valores bajos reducen el efecto de la destilación. La temperatura T=4.0 con α=0.7 ofrece el mejor balance. La combinación de fine-tuning previo del student (ejercicio 6) con destilación posterior es clave para que el student ya parta de una base útil.

---

# PARTE III: PREGUNTAS Y SOLUCIONES

## BLOQUE A — Preguntas conceptuales (teoría)

---

**P1. ¿Cuál es el problema principal de las RNNs que motivó la creación de los transformadores?**

Las RNNs tienen tres limitaciones fundamentales: (1) dificultad para resolver dependencias de largo alcance porque la información se diluye al propagarse a través de muchos pasos; (2) problemas de vanishing/exploding gradients al desplegar la red durante el entrenamiento; (3) no son paralelizables porque cada estado depende del anterior. Los transformadores eliminan la recurrencia procesando toda la secuencia en paralelo mediante atención, lo que resuelve estos tres problemas.

---

**P2. ¿Qué representan Q, K y V en el mecanismo de atención?**

Son tres transformaciones lineales aplicadas a los datos de entrada. Q (Query) representa "lo que busco"; K (Key) representa "lo que ofrezco para comparar"; V (Value) representa "el contenido que entrego si hay coincidencia". La analogía es una búsqueda en base de datos: la similitud entre Q y K determina los pesos con los que se pondera la combinación de los valores V. En autoatención, Q, K y V provienen de la misma secuencia de entrada.

---

**P3. ¿Por qué se escala la puntuación por 1/√d_attn en la ecuación de atención?**

Cuando la dimensión d_attn es grande, el producto escalar Q^T·K tiende a tener valores muy altos en magnitud. Esto lleva al softmax a regiones de gradientes muy pequeños (saturación), dificultando el entrenamiento. Dividir por √d_attn mantiene los valores en un rango donde el gradiente del softmax es más favorable.

---

**P4. ¿Qué diferencia hay entre un transformer encoder-only (BERT) y uno decoder-only (GPT)?**

BERT usa solo encoders sin máscara, procesando el contexto bidireccional (cada token atiende a todos los demás). Se preentrana con MLM (Masked Language Modeling). Es ideal para tareas de clasificación y comprensión. GPT usa solo decoders con máscara causal, procesando el contexto unidireccional (cada token solo atiende a los anteriores). Se preentrana prediciendo el siguiente token. Es ideal para generación de texto.

---

**P5. ¿Por qué se necesita la codificación posicional?**

La autoatención es invariante a la permutación: si se permutan los tokens de entrada, el resultado de la atención no cambia (salvo por el orden de los resultados). Esto significa que el modelo no tiene forma de saber si "Juan golpeó a María" es distinto de "María golpeó a Juan". La codificación posicional inyecta información sobre la posición de cada token, permitiendo al modelo distinguir el orden.

---

**P6. ¿Qué es la autoatención múltiple (Multi-Head Attention) y para qué sirve?**

Es la aplicación de H cabezales de atención en paralelo, cada uno con sus propios parámetros Q, K, V. Las salidas de todos los cabezales se concatenan y se proyectan linealmente. Permite al modelo capturar diferentes tipos de relaciones entre tokens de forma simultánea: un cabezal puede enfocarse en relaciones sintácticas, otro en semánticas, otro en posicionales, etc.

---

**P7. ¿Qué hace la normalización de capas (LayerNorm) y por qué se usa en transformadores?**

Normaliza las activaciones de cada muestra individualmente, controlando la media y varianza sobre la dimensión de características. En transformadores se usa después de cada sublayer (atención y FFN) junto con conexiones residuales (Add & Norm). Ayuda a estabilizar el entrenamiento de redes profundas y mejora la convergencia, sin los problemas de las batch normalization con secuencias de longitud variable.

---

**P8. ¿Cómo se adaptan los transformadores para visión (ViT)?**

La única diferencia es el tokenizador: en vez de dividir texto en palabras o subpalabras, la imagen se divide en parches (_patches_) pequeños no superpuestos (e.g., 16×16 píxeles). Cada parche se aplana y se proyecta linealmente al espacio interno del transformer. Se añade un embedding posicional para preservar la información espacial. Después de esta adaptación, el transformer procesa los parches como si fueran tokens de texto.

---

**P9. Explica las tres variantes de segmentación y cómo las aborda Mask2Former.**

La **segmentación semántica** asigna una clase a cada píxel (sin distinguir instancias individuales). La **segmentación de instancias** distingue objetos individuales de la misma clase. La **segmentación panóptica** combina ambas. Mask2Former usa un backbone (CNN o ViT) + decoder de píxeles + transformer decoder tipo DETR, generando una secuencia de "mask queries" que, combinadas con los embeddings de píxeles, producen las máscaras de segmentación directamente. Una única arquitectura resuelve los tres tipos.

---

**P10. ¿Qué es la destilación de conocimiento (Knowledge Distillation)?**

Técnica (Hinton et al., 2015) en la que un modelo grande (_teacher_) transfiere conocimiento a uno pequeño (_student_) mediante las distribuciones de probabilidad suavizadas de su salida (_soft targets_), en lugar de las etiquetas duras del dataset. Los soft targets contienen más información porque reflejan la confianza relativa del teacher sobre todas las clases. La función de pérdida combina la divergencia KL entre distribuciones suavizadas (KD loss) y la cross-entropy con las etiquetas reales (CE loss), ponderadas por α.

---

## BLOQUE B — Preguntas sobre la práctica

---

**P11. ¿Qué es la ventana deslizante y por qué se usa en los modelos de lenguaje?**

Es la técnica de generar múltiples muestras de entrenamiento a partir de un mismo texto, desplazando una ventana de tamaño fijo: la secuencia de los t tokens anteriores es el input y el token t+1 es el target. Se usa porque permite crear muchas muestras de entrenamiento (~2M secuencias de 23.767 textos en la práctica) y porque es la manera natural de entrenar modelos autorregresivos de lenguaje que predicen la siguiente palabra dado el contexto.

---

**P12. ¿Por qué se usa padding a la izquierda en los modelos de generación de texto y cómo afecta a la predicción?**

En la generación de texto, el modelo predice la siguiente palabra a partir del contexto más reciente. Si la secuencia de contexto es más corta que SEQ_LEN, se rellena por la izquierda con tokens `<PAD>`. Esto coloca el contenido real al final de la secuencia, que es donde el modelo hace su predicción (usa el embedding del último token). El padding por la derecha situaría el token de predicción en una posición rodeada de `<PAD>`, lo que no tendría sentido.

---

**P13. En la práctica, el Transformer tiene menor val_loss que el LSTM pero no siempre mejor accuracy. ¿Por qué?**

La val_loss mide la incertidumbre del modelo (log-probabilidad asignada a la respuesta correcta), mientras que accuracy solo mide si la predicción exacta coincide. Es posible que el Transformer asigne mayor probabilidad a la respuesta correcta (menor loss) sin que sea siempre el argmax predicho. En vocabularios grandes como el de WikiText-2 (66.651 tokens), esta discrepancia es habitual porque muchas palabras pueden ser plausibles en un contexto dado.

---

**P14. ¿Por qué en el fine-tuning de BERT-mini se usa LR=2e-5 y en el modelo custom desde cero se usa LR=3e-4?**

En el fine-tuning, el modelo ya tiene pesos preentrenados con representaciones lingüísticas muy informativas. Un LR pequeño (2e-5) realiza ajustes delicados sin destruir ese conocimiento previo. En el modelo desde cero, los pesos son aleatorios y el modelo necesita dar pasos más grandes para escapar de la inicialización aleatoria. Un LR mayor (3e-4) acelera el aprendizaje inicial.

---

**P15. ¿Qué rol juega el token [CLS] en la clasificación con BERT?**

El token `[CLS]` (_classification token_) es un token especial que se añade siempre al principio de la secuencia. Después de pasar por todos los encoders, su representación en la última capa captura información contextual de toda la secuencia (porque la autoatención bidireccional hace que atienda a todos los tokens). Este vector se pasa por el pooler (linear + tanh) y luego por la capa de clasificación. El diseño original de BERT usa este token como representación agregada de la secuencia para tareas de clasificación.

---

**P16. ¿Por qué la Knowledge Distillation mejora la accuracy del modelo small?**

El student aprende de dos fuentes: (1) las etiquetas duras del dataset y (2) las distribuciones suavizadas del teacher. Estas últimas contienen información "de confianza relativa" sobre todas las clases: por ejemplo, si una reseña es positiva pero tiene algunos elementos negativos, el teacher asignará, digamos, 85% a positivo y 15% a negativo, en lugar de 100%/0%. Esta señal más suave y rica permite al student aprender representaciones mejores que si solo ve las etiquetas binarias. Además, los soft targets actúan como regularizador, reduciendo el sobreajuste.

---

**P17. ¿Qué son los hiperparámetros T (temperatura) y α en Knowledge Distillation?**

**T** suaviza las distribuciones del teacher: T=1 es la distribución normal; T>1 "aplana" las probabilidades, haciendo que clases con baja probabilidad sean más visibles para el student (más información transferida). T=4 es un valor habitual que proporciona un buen balance.

**α** pondera entre aprender del teacher (KD loss) y aprender de las etiquetas (CE loss): α=0 → solo etiquetas duras; α=1 → solo soft targets del teacher. α=0.7 prioriza el teacher (70%) mientras mantiene la supervisión directa (30%).

---

**P18. ¿Qué es `DataCollatorWithPadding` y por qué es mejor que padding fijo?**

`DataCollatorWithPadding` aplica padding dinámico a nivel de batch: cada batch se rellena solo hasta la longitud del token más largo dentro de ese batch, no hasta la longitud máxima global (e.g., 256). Esto reduce el número de tokens de padding procesados innecesariamente, ahorrando memoria y tiempo de cómputo. Si todos los textos de un batch son cortos (e.g., máximo 50 tokens), no se crean tensores de 256 tokens llenos de padding.

---

**P19. ¿Qué implica que la val_loss mejore pero la val_accuracy no lo haga al mismo ritmo?**

Puede pasar cuando el modelo ya clasifica correctamente la mayoría de los ejemplos (accuracy alta) pero su confianza en las predicciones correctas sigue aumentando (la probabilidad asignada a la clase correcta aumenta, lo que reduce la cross-entropy). También puede ocurrir a la inversa: que el modelo aprenda distribuciones más suaves que aumentan la loss pero cambian algunas predicciones. En la práctica KD, la loss KL combina distribuciones suavizadas, por lo que los valores de loss no son directamente comparables con los de la cross-entropy estándar.

---

**P20. ¿Cuáles son las decisiones de diseño clave para acelerar la inferencia de un transformer?**

Las principales palancas para reducir el tiempo de inferencia son: (1) **reducir el número de capas** (de 4 a 2 en el modelo custom reduce cuadráticamente el coste de atención); (2) **reducir la dimensión oculta** (d_e=160 vs 256 en bert-mini); (3) **reducir la longitud máxima de secuencia** (MAX_LENGTH=64 vs 256: la autoatención es O(n²) en la longitud de la secuencia, por lo que reducir de 256 a 64 tokens supone una reducción de 16× en el coste de atención); (4) **aumentar el batch size** para maximizar la utilización de la GPU.

---

## BLOQUE C — Preguntas de nivel avanzado

---

**P21. ¿Cómo resuelve DETR el problema del matching entre predicciones y ground truth en detección de objetos?**

En detección, las predicciones del modelo y el ground truth no están alineados: el modelo produce N bounding boxes en orden arbitrario, mientras que el ground truth tiene M objetos reales. DETR resuelve esto con **emparejamiento bipartito óptimo** (Hungarian matching): se calcula la función de pérdida para todas las posibles asignaciones predicción-ground truth y se elige la que minimiza la pérdida total. Esto garantiza un entrenamiento estable sin necesidad de Non-Maximum Suppression (NMS).

---

**P22. ¿Por qué los transformadores son más adecuados que las CNNs para datos multimodales?**

Las CNNs incorporan priores estructurales (localidad espacial, invarianza a la traslación) que son beneficiosos para imágenes pero pueden ser desventajosos al combinar modalidades heterogéneas. El espacio de modelado de los transformadores es equivalente a un grafo completamente conectado, sin sesgos inductivos específicos de modalidad. Esto les da flexibilidad para aprender las relaciones entre modalidades directamente desde los datos, integrando texto, imagen, audio o cualquier otro tipo de entrada con el mismo mecanismo de atención.

---

**P23. Explica la diferencia entre fine-tuning y training from scratch en el contexto de transformadores.**

En **fine-tuning**, el modelo parte de pesos preentrenados en un corpus masivo (el modelo ya "sabe" mucho sobre el lenguaje). Solo se ajustan los pesos para la tarea específica con un LR pequeño. En **training from scratch**, los pesos son aleatorios y el modelo debe aprender todo desde cero con los datos disponibles. El fine-tuning es mucho más eficiente en datos y tiempo, y alcanza mejor rendimiento en tareas downstream. Training from scratch requiere enormes cantidades de datos y tiempo de cómputo (como GPT-3 con 499×10⁹ tokens de entrenamiento), pero ofrece total control sobre la arquitectura y el dominio.

---

**P24. ¿Qué es el espectrograma de MEL y por qué se usa en lugar de la señal de audio cruda?**

El espectrograma de MEL es una representación 2D de una señal de audio donde el eje vertical es la frecuencia en escala MEL (que linealiza la percepción humana de la frecuencia: somos más sensibles a las bajas frecuencias) y el eje horizontal es el tiempo. Se prefiere sobre la señal cruda porque: (1) reduce la dimensionalidad enormemente (44.100 muestras/segundo → matriz compacta), (2) está alineado con la percepción auditiva humana, enfatizando las frecuencias que percibimos mejor, y (3) es una representación estacionaria por ventanas que captura la estructura temporal del sonido.

---

**P25. ¿Qué significa que la arquitectura de un transformer sea equivalente a un grafo completamente conectado?**

Desde la perspectiva de la geometría topológica, en cada capa del transformer cada token está conectado a todos los demás mediante la autoatención. Esto es exactamente un grafo completamente conectado (o grafo completo): cada nodo tiene una arista hacia todos los demás. En cambio, una CNN corresponde a un grafo con conexiones locales (solo nodos cercanos en el espacio), y una RNN a un grafo con conexiones temporales secuenciales. El grafo completamente conectado es la estructura más general, lo que le da a los transformadores máxima flexibilidad para aprender cualquier tipo de relación entre tokens.

---

_Documento elaborado a partir del módulo "Transformadores" (FUOC, De la Torre Gallart, 2024) y de la PEC3 de Deep Learning (M2.875, UOC, 2025-2)._