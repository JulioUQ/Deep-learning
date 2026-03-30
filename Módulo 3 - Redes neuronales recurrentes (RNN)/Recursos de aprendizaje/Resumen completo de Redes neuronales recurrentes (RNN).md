---
Cita: Lozano Bagén, T. (2023) Redes neuronales recurrentes. [Recurso de aprendizaje textual]. 1ª edición. Barcelona:Fundació Universitat Oberta de Catalunya (FUOC).
---
- ---
### 1. Fundamentos de las Redes Recurrentes

Las redes neuronales recurrentes están diseñadas específicamente para procesar datos con una estructura secuencial intrínseca, como series temporales o textos.

- **Recurrencia y Memoria:** A diferencia de las redes tradicionales (donde la información fluye en una sola dirección), las RNN poseen conexiones que pueden ir de capas posteriores a capas anteriores. Esto permite que la red utilice información generada en el registro anterior, creando un "estado interno" o "memoria" que influye en cómo la red procesa los datos actuales.
    
- **Entrenamiento y Problemas de Gradiente:** Se entrenan utilizando una adaptación llamada "retropropagación en el tiempo" (desenrollando la red para aplicar el algoritmo clásico). Sin embargo, al multiplicar el gradiente consigo mismo a través de muchos pasos temporales, surgen problemas de "desaparición del gradiente" (la red olvida dependencias lejanas) o "explosión del gradiente" (el entrenamiento se vuelve inestable).
    

### 2. Tipología de Celdas Recurrentes

Para solucionar los problemas del gradiente, se utilizan celdas especiales que controlan el flujo de información mediante estructuras llamadas "puertas" (compuestas por neuronas con funciones de activación).

- **LSTM (Long Short Term Memory):** Es capaz de aprender dependencias tanto lejanas como cercanas gracias a un canal de memoria y a tres puertas de control:
    
    - _Puerta de olvido:_ Decide qué parte de la memoria anterior se debe descartar.
    - _Puerta de entrada:_ Controla qué nueva información se añade a la memoria.
    - _Puerta de salida:_ Determina qué parte del estado actual se emitirá como respuesta en ese paso.

- **GRU (Gated Recurrent Unit):** Es una simplificación de la celda LSTM que fusiona el estado de la red con su salida, utilizando menos parámetros. Controla la información mediante una _puerta de reset_ (selecciona qué memoria usar) y una _puerta de actualización_ (decide qué memoria retener y qué olvidar).

### 3. Arquitecturas Avanzadas

Las celdas se pueden organizar de diferentes maneras según la complejidad del problema:

- **Redes Bidireccionales:** Procesan la secuencia tanto en su orden original como en orden inverso. Son muy útiles para textos, ya que el contexto de una palabra puede depender de palabras que aparecen tanto antes como después de ella.

- **Redes Profundas:** Apilan varias capas de celdas recurrentes. Cada capa aprende conceptos más abstractos que la anterior, aunque requieren de grandes cantidades de datos para poder entrenarse correctamente.

- **Arquitectura Codificador-Decodificador (Seq2Seq):** Un "codificador" consume la secuencia de entrada y la comprime en un vector de estado, que luego es utilizado por el "decodificador" para generar una nueva secuencia de salida. Es el estándar para problemas como la traducción automática.

- **Mecanismo de Atención:** Mejora las arquitecturas Seq2Seq al permitir que el decodificador utilice una suma ponderada de _todos_ los estados del codificador (en lugar de solo el último vector). Esto ayuda a la red a "prestar atención" a las partes más relevantes de la secuencia de entrada en cada paso específico de la generación de la salida.

### 4. Consejos Prácticos y Aplicaciones

- **Series Temporales:** En lugar de aplicar la red a los datos crudos, es recomendable aplicar técnicas clásicas para descomponer la serie en tendencia, estacionalidad y componente irregular. Modelar cada componente por separado suele generar mejores predicciones.

- **Procesamiento de Textos:** Para evitar vectores inmensos y escasos al representar un diccionario, se recomienda usar una "capa de embedding" (como Word2Vec). Esta técnica comprime las palabras en vectores densos donde las palabras con significados similares tienen representaciones numéricas similares, lo que facilita enormemente el aprendizaje de la red.

---
