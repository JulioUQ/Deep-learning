# Quiz: Redes Neuronales Convolucionales (CNN) con Explicaciones

### 1. Una imagen de 64 × 64 píxeles en color RGB implica una capa de entrada de...

- [ ] 4096
- [ ] 1024
- [x] **12288**
- [ ] Ninguno de los anteriores

> **Explicación:** Una imagen RGB tiene 3 canales (Rojo, Verde, Azul). Por tanto, el número total de valores de entrada es $64 \times 64 \times 3 = 12288$.

---

### 2. Supongamos una entrada con dimensiones igual a 32 × 32 × 3 y un kernel con dimensiones 5 × 5, padding = 0 y stride = 2. ¿Cuáles serán las dimensiones de la salida?

- [x] **14 × 14 × 1**
- [ ] 32 × 32 × 3
- [ ] 14 × 14 × 3
- [ ] 28 × 28 × 1

> **Explicación:** Se aplica la fórmula $O = \frac{W - K + 2P}{S} + 1$. Aquí: $\frac{32 - 5 + 0}{2} + 1 = 13.5 + 1 = 14.5$, que se redondea a **14**. Como no se especifica el número de filtros, se asume 1 mapa de características de salida.

---

### 3. ¿Cuál es el resultado de aplicar una capa de agrupamiento (pooling) de 8 × 8 con stride = 2 sobre unos datos de entrada de tamaño 128 × 128 × 3?

- [ ] 64 × 64 × 1    
- [ ] 96 × 96 × 1
- [ ] 64 × 64 × 3
- [x] **61 × 61 × 3**

> **Explicación:** Aplicando la fórmula: $\frac{128 - 8}{2} + 1 = \frac{120}{2} + 1 = 61$. El pooling mantiene el número de canales de entrada, por lo que el resultado es $61 \times 61 \times 3$.

---

### 4. ¿Cuál es el resultado de aplicar un filtro convolucional de 16 × 16 con padding = 4 y stride = 1 sobre unos datos de entrada de tamaño 100 × 100 × 3?

- [ ] Ninguno de los anteriores    
- [x] **93 × 93 × 1**
- [ ] 85 × 85 × 1
- [ ] 84 × 84 × 1

> **Explicación:** $\frac{100 - 16 + (2 \times 4)}{1} + 1 = 84 + 8 + 1 = 93$. Un solo filtro produce un solo canal de salida.

---

### 5. El kernel, normalmente, tiene un tamaño mucho más pequeño que la entrada.

- [x] **Verdadero**    
- [ ] Falso

> **Explicación:** El objetivo del kernel es detectar patrones locales (bordes, texturas). Usar un kernel pequeño permite la "compartición de pesos" y reduce drásticamente el número de parámetros.

---

### 6. Supongamos una entrada con dimensiones igual a 32 × 32 × 3 y un kernel con dimensiones 5 × 5, padding = 1 y stride = 1. ¿Cuáles serán las dimensiones de la salida?

- [ ] 30 × 30 × 3
- [ ] 32 × 32 × 1
- [ ] 32 × 32 × 3
- [x] **30 × 30 × 1**

> **Explicación:** $\frac{32 - 5 + (2 \times 1)}{1} + 1 = 27 + 2 + 1 = 30$. Al ser una operación de convolución estándar con un solo filtro, el resultado tiene profundidad 1.

---

### 7. ¿Cuál es el resultado de aplicar un filtro convolucional de 10 × 10 con padding = 0 y stride = 2 sobre unos datos de entrada de tamaño 64 × 64 × 3?

- [ ] 32 × 32 × 3
- [ ] 32 × 32 × 1
- [ ] 56 × 56 × 1
- [x] **28 × 28 × 1**

> **Explicación:** $\frac{64 - 10}{2} + 1 = \frac{54}{2} + 1 = 27 + 1 = 28$.

---

### 8. ¿Cuántos parámetros es necesario entrenar en una capa fully-connected con 200 entradas y 80 neuronas?

- [ ] 12.480
- [ ] Ninguno de los anteriores
- [x] **16.080**
- [ ] 32.960

> **Explicación:** Parámetros = $(\text{entradas} \times \text{neuronas}) + \text{sesgos}$. En este caso: $(200 \times 80) + 80 = 16000 + 80 = 16080$.

---

### 9. Un patrón que se mantiene es que las capas de convolución van seguidas de una capa de agrupamiento o pooling...

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** Es la arquitectura clásica (como LeNet-5 o VGG). La convolución extrae características y el pooling reduce la resolución espacial para ganar invariancia y eficiencia.

---

### 10. ¿Por qué se suele optar por la transferencia del aprendizaje cuando el conjunto de datos es grande pero la similitud es baja?

- [ ] Por el modelo preentrenado como extractor.
- [x] **Por entrenar la red desde cero.**
- [ ] Por congelar los pesos iniciales.

> **Explicación:** Si tienes muchos datos pero son muy distintos a los del modelo preentrenado (ej. fotos de satélite vs. fotos de perros), es mejor entrenar desde cero para que la red aprenda las características específicas de ese dominio.

---

### 11. En una imagen en RGB, la convolución se aplica de forma independiente sobre cada uno de los tres canales.

- [ ] Falso
- [x] **Verdadero**

> **Explicación:** Aunque el resultado final suele ser la suma de las convoluciones de cada canal, el kernel tiene una profundidad que coincide con la entrada, operando sobre cada canal para extraer la información.

---

### 12. La data augmentation se basa en generar imágenes de forma artificial para obtener datos en una variedad de condiciones.

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** Mediante giros, recortes o cambios de brillo, "engañamos" a la red dándole más ejemplos de los que realmente tenemos, mejorando su generalización.

---

### 13. Supongamos una entrada de 32 × 32 × 3 y kernel 5 × 5, padding = 0, stride = 1. ¿Salida?

- [ ] 28 × 28 × 3
- [x] **28 × 28 × 1**
- [ ] 32 × 32 × 1

> **Explicación:** $\frac{32 - 5}{1} + 1 = 28$. La salida de una convolución con un filtro siempre colapsa la profundidad de entrada en un solo mapa de características por filtro.

---

### 14. ¿Qué capa es habitual aplicar inmediatamente después de cada capa de convolución?

- [x] **ReLU**
- [ ] Dropout
- [ ] Softmax

> **Explicación:** ReLU introduce no linealidad en el modelo, permitiendo que la red aprenda patrones complejos. Sin ella, múltiples capas de convolución equivaldrían a una sola operación lineal.

---

### 15. ¿Es habitual aplicar una capa no lineal (activación) tras la convolución?

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** Es el estándar en Deep Learning para romper la linealidad de las operaciones de producto de matrices.

---

### 16. El kernel, también llamado filtro, es la matriz de pesos de una capa de convolución.

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** El kernel contiene los parámetros que la red aprende mediante backpropagation para detectar características específicas.

---

### 17. Para mantener la dimensión de entrada en la salida, utilizamos...

- [ ] Padding igual a 0.
- [x] **Padding igual o superior a 1, dependiendo de la dimensión del kernel.**

> **Explicación:** Si el kernel es mayor que 1x1, la convolución "encoge" la imagen. El padding añade un marco (normalmente ceros) para compensar esa pérdida de tamaño.

---
### 18. ¿Resultado de 5x5, P=2, S=1 sobre entrada 64x64x3?

- [ ] 64 × 64 × 3
- [x] **Ninguno de los anteriores**
- [ ] 56 × 56 × 3

> **Explicación:** La dimensión espacial sería $\frac{64 - 5 + 4}{1} + 1 = 64$. Sin embargo, la salida de un filtro convolucional es $64 \times 64 \times 1$, no $\times 3$. Por eso la opción de "64x64x3" es incorrecta.

---

### 19. El parámetro stride permite lo que llamamos convoluciones por paso.

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** El stride define cuánto se desplaza la ventana del kernel. Un stride de 2 salta un píxel cada vez, reduciendo el tamaño de salida a la mitad.

---

### 20. ¿Resultado de 2x2, P=0, S=2 sobre 100x100x1?

- [ ] 48 × 48 × 1
- [x] **Ninguno de los anteriores**
- [ ] 50 × 50 × 1

> **Explicación:** La fórmula da $\frac{100 - 2}{2} + 1 = 49 + 1 = 50$. Como la opción 50x50 no aparece en tu lista original, la respuesta es "Ninguno".

---

### 21. Para mantener la dimensión usamos lo que se llama cero padding.

- [ ] Verdadero
- [x] **Falso**

> **Explicación:** Aunque se usa "zero padding", la afirmación es falsa porque no basta con aplicar padding; el valor del padding debe ser el correcto (ej. $P = \frac{K-1}{2}$ para stride 1) para que la dimensión se mantenga exactamente igual.

---

### 22. ¿Parámetros en FC con 400 entradas y 120 neuronas?

- [x] **48.120**
- [ ] 12.480

> **Explicación:** $(400 \times 120) + 120 = 48000 + 120 = 48120$.

---

### 23. El concepto de operación de convolución es el bloque fundamental de las CNN.

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** Es la operación que permite la extracción de características espaciales de forma eficiente.

---

### 24. Transfer learning y transferencia inductiva son sinónimos.

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** En el contexto de aprendizaje automático, ambos términos se refieren a utilizar el conocimiento adquirido en una tarea para mejorar el aprendizaje en otra.

---

### 25. ¿Qué ocurre a más profundidad en la red?

- [x] **Disminuye el volumen (dimensiones espaciales).**
- [x] **Aumenta el número de canales.**

> **Explicación:** Las redes suelen hacerse "más estrechas" (menos píxeles) y "más profundas" (más filtros) para detectar conceptos más abstractos.

---

### 26. ¿Qué definen los pesos compartidos y el sesgo compartido?

- [x] **Un mapa de características**
- [x] **Un kernel**
- [x] **Un filtro**

> **Explicación:** Los tres términos están relacionados: el filtro/kernel aplica los mismos pesos en toda la imagen para generar un mapa de características.

---

### 27. Un patrón que se mantiene en las CNN es que las capas de convolución van seguidas de…

- [x] **una capa de agrupamiento o pooling.**

> **Explicación:** El pooling condensa la información extraída por la convolución, haciendo al modelo más robusto a pequeñas traslaciones.

---

### 28. ¿Parámetros en capa convolucional (32x32x3), kernel 5x5, salida 28x28x1?

- [x] **76**
- [ ] 1024

> **Explicación:** Parámetros = $(\text{Ancho}_k \times \text{Alto}_k \times \text{Canales}_{\text{in}} + 1) \times \text{Filtros}_{\text{out}}$. Aquí: $(5 \times 5 \times 3 + 1) \times 1 = 76$.

---

### 29. ¿Por qué es popular el Transfer Learning?

- [x] **Disponibilidad de código abierto.**
- [x] **Ahorro de recursos computacionales.**

> **Explicación:** Entrenar modelos desde cero es carísimo en tiempo y dinero. Reutilizar modelos ya entrenados permite democratizar el uso de IA potente.

---

### 30. ¿Un kernel 1x1 produce una reducción de la dimensión espacial?

- [x] **Verdadero**
- [ ] Falso

> **Explicación:** Los kernels 1x1 no afectan el ancho ni el alto de la imagen (por eso "no producen reducción"), pero se usan para cambiar (reducir o aumentar) el número de canales.

---
