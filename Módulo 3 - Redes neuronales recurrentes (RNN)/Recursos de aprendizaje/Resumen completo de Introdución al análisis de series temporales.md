
### 1. Introducción a las Series Temporales

Las series temporales son conjuntos de datos donde el tiempo es la variable principal, organizados de forma cronológica y equiespaciada (por ejemplo, datos diarios del PIB o registros climáticos).

- **Tipos de Datos:** Antes de analizar un problema, es vital distinguir entre _datos transversales_ (múltiples individuos en un instante de tiempo), _series temporales_ (un individuo a lo largo del tiempo) y _datos de panel_ (múltiples individuos a lo largo del tiempo).

- **Objetivos Analíticos:** El tratamiento de estos datos se divide en dos grandes tareas:

    - _Análisis de series temporales (TSA):_ Busca entender las causas y fuerzas subyacentes que dan forma a los datos (el "por qué").
    - _Predicción (TSF - Time Series Forecasting):_ Utiliza el histórico para pronosticar u obtener estimaciones de valores en el futuro.

### 2. Componentes de una Serie Temporal

El análisis a menudo implica descomponer matemáticamente la serie en cuatro elementos o factores que se suman o multiplican:

- **Tendencia General:** Es el movimiento a largo plazo de la serie, que indica si los datos tienen un comportamiento global al alza, a la baja o se mantienen estables.

- **Variaciones Estacionales:** Patrones a corto plazo que se repiten de manera periódica (usualmente cada año) debido a condiciones climáticas (como las estaciones) o convenciones sociales (como las vacaciones).

- **Variaciones Cíclicas:** Oscilaciones (como subidas y bajadas) que duran más de un año. No tienen una periodicidad fija y son muy comunes en los ciclos económicos.

- **Variaciones Aleatorias/Irregulares:** Fluctuaciones erráticas e impredecibles causadas por eventos excepcionales (desastres, crisis) que aportan ruido a la serie y no pueden ser modeladas.

### 3. Estacionariedad

Para que muchos modelos matemáticos de predicción funcionen correctamente, asumen que los datos son "estacionarios".

- **Concepto:** Una serie es estacionaria cuando sus propiedades estadísticas (como la media y la varianza) se mantienen constantes a lo largo del tiempo, lo que la hace mucho más fácil de predecir.

- **Evaluación y Transformación:** Para saber si una serie es estacionaria se puede usar el _Test de Dickey-Fuller aumentado_. Si la serie no lo es (por ejemplo, porque tiene una tendencia creciente), se suele aplicar un proceso matemático llamado **diferenciación** (restar el valor actual menos el anterior) para eliminar tendencias o estacionalidades y volverla estacionaria.

### 4. Aprendizaje Automático y Pronósticos

Para poder utilizar algoritmos tradicionales de _Machine Learning_ (aprendizaje supervisado) en series temporales, los datos deben ser reestructurados previamente.

- **Método de la Ventana Deslizante:** Convierte la serie en un formato de "entradas y salidas". Consiste en utilizar los valores observados en pasos de tiempo anteriores (ventana o _lag_) como variables de entrada (X) para enseñar al modelo a predecir el instante de tiempo siguiente (y).

- **Estrategias para Múltiples Pasos:** Cuando se necesita pronosticar más de un valor a futuro, se pueden usar diferentes estrategias:

    - _Directa:_ Entrenar un modelo distinto e independiente para cada paso futuro.
    - _Recursiva:_ Usar un solo modelo iterativamente, utilizando las propias predicciones recién generadas como entrada para el siguiente paso temporal.
    - _Híbrida:_ Combinación de las dos anteriores.
    - _De salida múltiple:_ Un único modelo complejo capaz de arrojar todas las predicciones simultáneamente.
    
- **Métricas de Evaluación:** Para saber si el modelo es preciso, se utilizan métricas que comparan la realidad con la predicción, como el MAE (Error Absoluto Medio), el MSE (Error Cuadrático Medio) o el MAPE (Error Porcentual Absoluto Medio).

---

