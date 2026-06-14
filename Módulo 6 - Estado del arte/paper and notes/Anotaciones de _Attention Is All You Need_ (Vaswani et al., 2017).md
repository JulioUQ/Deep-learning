# Resumen completo de _Attention Is All You Need_ (Vaswani et al., 2017)

## Introducción

El artículo presenta **Transformer**, una nueva arquitectura de redes neuronales diseñada para tareas de procesamiento de secuencias, especialmente traducción automática. Hasta ese momento, los modelos más avanzados estaban basados en **redes neuronales recurrentes (RNNs)** o **redes convolucionales (CNNs)**, normalmente combinadas con mecanismos de atención.

Los autores proponen eliminar completamente las recurrencias y convoluciones y construir el modelo únicamente sobre mecanismos de **atención (attention)**. El objetivo es superar las limitaciones de paralelización y eficiencia de las RNNs, permitiendo entrenamientos más rápidos y una mejor captura de dependencias a larga distancia.

# Problema de los modelos anteriores

Las RNN procesan las secuencias elemento a elemento, lo que implica:

- Dependencia secuencial entre pasos.
- Imposibilidad de paralelizar completamente el entrenamiento.
- Dificultades para aprender relaciones entre elementos muy alejados en la secuencia.
- Mayor coste computacional para secuencias largas.

Aunque los mecanismos de atención ya se utilizaban para aliviar estos problemas, seguían dependiendo de una arquitectura recurrente subyacente. El Transformer elimina completamente esta dependencia.

# Idea principal: la atención como único mecanismo

La tesis central del artículo es que:

> Para tareas de modelado de secuencias, la atención por sí sola es suficiente para capturar dependencias complejas entre elementos.

En lugar de procesar palabras una tras otra, el modelo permite que cada palabra atienda simultáneamente a todas las demás palabras relevantes de la secuencia.

# Arquitectura Transformer

La arquitectura sigue el esquema clásico **encoder-decoder**, pero reemplaza las capas recurrentes por mecanismos de atención.

## Encoder

Está compuesto por:

- 6 capas idénticas.
- Cada capa contiene:
    - Multi-Head Self-Attention.
    - Feed-Forward Network.

Cada subcapa incorpora:

- Conexiones residuales.
- Normalización de capa (_Layer Normalization_).    

La dimensión de representación utilizada es:

- d_model = 512.

## Decoder

También está compuesto por:

- 6 capas idénticas.

Cada capa contiene:

1. Self-Attention enmascarada.
2. Atención sobre la salida del encoder.
3. Feed-Forward Network.

El enmascaramiento evita que una posición vea palabras futuras durante la generación, preservando la naturaleza autoregresiva del modelo.

# Scaled Dot-Product Attention

Es el núcleo matemático del Transformer.

La atención recibe:

- Queries (Q)
- Keys (K)
- Values (V)

El mecanismo calcula:

Attention(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V

Funcionamiento:

1. Se calcula la similitud entre Query y Keys.
2. Se divide por √dk para evitar valores excesivamente grandes.
3. Se aplica Softmax para obtener pesos.
4. Los pesos se utilizan para combinar los Values.

La división por √dk estabiliza el entrenamiento y evita gradientes demasiado pequeños cuando la dimensionalidad aumenta.

# Multi-Head Attention

En lugar de calcular una única atención, el Transformer calcula varias atenciones en paralelo.

Configuración utilizada:

- 8 cabezas de atención.
- Cada cabeza trabaja sobre una proyección distinta del espacio de representación.

Ventajas:

- Captura diferentes tipos de relaciones simultáneamente.
- Permite atender a distintos aspectos sintácticos y semánticos.
- Reduce el efecto de promediado de una única atención.

Los autores muestran que demasiadas o muy pocas cabezas degradan el rendimiento.

# Tipos de atención utilizados

## 1. Encoder Self-Attention

Cada palabra puede atender a cualquier otra palabra de la secuencia de entrada.

Ejemplo:

En una frase larga, una palabra puede relacionarse directamente con otra situada muchas posiciones atrás.

## 2. Decoder Self-Attention

Cada posición sólo puede ver:

- Su propio estado.
- Estados anteriores.

No puede ver palabras futuras.

## 3. Encoder-Decoder Attention

Permite que cada palabra generada en la salida consulte toda la representación de entrada producida por el encoder.

# Redes Feed-Forward

Tras la atención, cada posición pasa por una red neuronal independiente:

FFN(x)=max(0,xW_1+b_1)W_2+b_2

Características:

- Dos transformaciones lineales.
- Activación ReLU.
- Aplicación independiente para cada posición.

Dimensiones:

- Entrada: 512
- Capa oculta: 2048
- Salida: 512.

# Positional Encoding

Como el Transformer no posee recurrencia ni convoluciones, necesita información explícita sobre el orden de las palabras.

Para ello añade codificaciones posicionales mediante funciones seno y coseno de distintas frecuencias:

- Permiten representar posiciones absolutas y relativas.
- Facilitan extrapolar a secuencias más largas que las vistas durante el entrenamiento.

Los autores comprobaron que las codificaciones aprendidas y las sinusoidales producen resultados muy similares.

# ¿Por qué funciona mejor la Self-Attention?

Los autores comparan atención, recurrencia y convolución según tres criterios:

### 1. Complejidad computacional

La atención es más eficiente cuando la longitud de la secuencia es menor que la dimensión de representación, situación habitual en traducción automática.

### 2. Paralelización

La atención permite procesar todos los elementos simultáneamente.

Las RNN requieren procesamiento secuencial.

### 3. Dependencias a larga distancia

En self-attention cualquier palabra puede conectarse con cualquier otra mediante una única operación.

En RNNs la información debe atravesar múltiples pasos intermedios.

Esto facilita aprender relaciones de largo alcance.

# Entrenamiento

## Datos

### Inglés-Alemán

- WMT 2014.
- 4,5 millones de pares de frases.
- Vocabulario de aproximadamente 37.000 tokens.

### Inglés-Francés

- WMT 2014.
    
- 36 millones de pares de frases.
    
- Vocabulario de 32.000 tokens.
    

---

## Hardware

Se utilizó:

- 8 GPUs NVIDIA P100.
    

Tiempo de entrenamiento:

### Transformer Base

- 100.000 pasos.
    
- Aproximadamente 12 horas.
    

### Transformer Big

- 300.000 pasos.
    
- Aproximadamente 3,5 días.
    

---

## Optimización

Se utilizó:

- Adam Optimizer.
    
- β₁ = 0.9
    
- β₂ = 0.98
    
- ε = 10⁻⁹
    

La tasa de aprendizaje aumenta inicialmente (warmup) y posteriormente disminuye siguiendo una ley proporcional a la raíz cuadrada inversa del número de iteraciones.

---

## Regularización

Se aplicaron:

### Dropout

- Valor típico: 0.1.
    

### Label Smoothing

- ε = 0.1.
    

Mejora la precisión final y el BLEU aunque aumente ligeramente la incertidumbre del modelo.

---

# Resultados

## Traducción Inglés-Alemán

### Transformer Base

- BLEU = 27.3
    

### Transformer Big

- BLEU = 28.4
    

Supera a todos los modelos anteriores, incluyendo conjuntos de modelos (_ensembles_).

---

## Traducción Inglés-Francés

### Transformer Big

- BLEU = 41.8
    

Obtiene un nuevo estado del arte utilizando menos de una cuarta parte del coste computacional de los mejores sistemas anteriores.

---

# Experimentos de ablación

Los autores analizaron diversos componentes:

### Número de cabezas

- Una sola cabeza empeora significativamente el rendimiento.
    
- Demasiadas cabezas también perjudican los resultados.
    

### Tamaño del modelo

- Modelos más grandes producen mejores resultados.
    

### Dropout

- Fundamental para evitar sobreajuste.
    

### Positional Encoding

- Las versiones sinusoidales y aprendidas funcionan prácticamente igual.
    

---

# Generalización a otras tareas

Para demostrar que el Transformer no sirve únicamente para traducción, se evaluó en:

## Parsing sintáctico del inglés

Resultados:

- 91.3 F1 usando únicamente Penn Treebank.
    
- 92.7 F1 en configuración semi-supervisada.
    

Superó a la mayoría de sistemas previos, quedando únicamente por detrás de algunas arquitecturas altamente especializadas.

---

# Interpretabilidad

Los autores analizaron visualizaciones de atención y observaron que diferentes cabezas aprenden funciones distintas:

- Resolución de correferencias (pronombres y referencias).
    
- Relaciones sintácticas.
    
- Dependencias a larga distancia.
    
- Conexiones semánticas entre palabras.
    

Por ejemplo, algunas cabezas aprendieron a relacionar automáticamente un pronombre posesivo ("its") con el sustantivo al que hacía referencia.

---

# Conclusiones

El artículo introduce una de las arquitecturas más influyentes de la historia de la inteligencia artificial: el **Transformer**.

Sus principales aportaciones son:

1. Eliminar completamente recurrencias y convoluciones.
    
2. Basar todo el procesamiento en mecanismos de atención.
    
3. Mejorar la paralelización y reducir tiempos de entrenamiento.
    
4. Capturar dependencias a larga distancia de forma más efectiva.
    
5. Alcanzar nuevos estados del arte en traducción automática.
    
6. Generalizar correctamente a otras tareas de procesamiento del lenguaje.
    

La importancia histórica del trabajo radica en que sentó las bases de prácticamente todos los modelos modernos de IA generativa, incluyendo familias como BERT, GPT, T5, PaLM y muchos otros sistemas actuales.