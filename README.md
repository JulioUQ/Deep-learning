# **Deep Learning**

## 1. Presentación general

El repositorio introduce los fundamentos del **aprendizaje profundo** dentro del ámbito de la inteligencia artificial y la ciencia de datos. Se centra en modelos capaces de aprender representaciones complejas a partir de datos, especialmente **redes neuronales convolucionales (CNN)** para análisis de imágenes y **redes neuronales recurrentes (RNN)** para series temporales, además de transformadores y modelos generativos.

El curso combina teoría, práctica en Python y uso de herramientas como **Google Colab**, con énfasis en el aprendizaje autónomo y la participación en foros.

---

## 2. Objetivos formativos

El estudiante desarrollará:

### Conocimientos

- Comprensión del ciclo completo de un proyecto de ciencia de datos.
- Identificación de técnicas, herramientas y limitaciones del análisis de datos.

### Habilidades

- Diseño de soluciones basadas en inteligencia artificial.
- Implementación avanzada de modelos y herramientas de programación.
- Evaluación crítica de soluciones tecnológicas.

### Competencias

- Pensamiento crítico en proyectos de datos.
- Capacidad para diseñar y gestionar proyectos adaptándose a nuevas tecnologías.

También se espera dominar:

- Arquitecturas y entrenamiento de redes neuronales.
- CNN, RNN, transformadores y modelos generativos.
- Principios éticos y de integridad académica.

---

## 3. Contenidos principales

El programa se organiza en cinco bloques:

1. **Redes neuronales artificiales** y fundamentos del deep learning.
2. **Redes convolucionales (CNN)**: fundamentos, arquitecturas y aplicaciones.
3. **Redes recurrentes (RNN)** para datos secuenciales.
4. **Transformadores** y mecanismos de atención.
5. **Modelos generativos** como autoencoders y GANs.

---

## 4. Metodología de aprendizaje

El curso sigue un enfoque activo:

- Estudio de materiales teóricos.
- Resolución de actividades prácticas abiertas.
- Participación en debates y foros.
- Búsqueda autónoma de información y argumentación de soluciones.

Se requieren conocimientos previos de **programación en Python** y **machine learning**, además de comprensión básica de textos técnicos en inglés.

Se recomienda consultar los ejemplos prácticos del libro "Deep learning: Principios y fundamentos", disponibles en lenguaje Python, en el siguiente repositorio de código abierto:

- [https://github.com/jcasasr/Libro-Deep-Learning](https://github.com/jcasasr/Libro-Deep-Learning)

Además de realizar el siguiente curso de 3 horas (video + slides) ofrecido para desarrolladores como una rapida introduccion de los fundamentos del deep learning, con TensorFlow.

- https://cloud.google.com/blog/products/ai-machine-learning/learn-tensorflow-and-deep-learning-without-a-phd

Otros son recursos son:

- https://d2l.ai/
- https://www.deeplearningbook.org/
---

## 5. Sistema de evaluación

La asignatura se aprueba únicamente mediante **evaluación continua**:

### Pruebas principales

- **5 PECs** (pruebas de evaluación continua):    
    - PEC1–PEC4: 20% cada una
    - PEC5: 10%

- **Tests teóricos autocorregidos**: 10% final.

### Actividades orales asíncronas

- Asociadas a las PEC1–PEC4.
- Representan el **20% de cada PEC**.
- No realizarlas implica **No Presentado** en la PEC correspondiente.

### Entregas fuera de plazo

- Hasta 7 días de retraso con penalización (máximo 7/10).
- Más tarde: actividad no presentada.

---

## 6. Integridad académica y uso de IA

El trabajo debe ser **individual, original y correctamente citado**.  
El uso indebido de inteligencia artificial, plagio o suplantación puede implicar **suspenso o sanciones disciplinarias**.  
La universidad puede verificar identidad y autoría mediante pruebas orales o grabaciones.

---

## 7. Calendario general

El semestre incluye:

- Tests teóricos en febrero, marzo y abril de 2026.
- Entregas de PEC entre marzo y junio de 2026.
- Actividades orales disponibles unos días antes y después de cada PEC.

---

# Conclusión

 El repositorio proporciona una formación integral en **deep learning aplicado a imágenes y series temporales**, combinando fundamentos teóricos, práctica técnica y evaluación continua rigurosa. Su enfoque prioriza la autonomía del estudiante, el pensamiento crítico y la integridad académica para preparar profesionales capaces de desarrollar proyectos reales de ciencia de datos.

 ----

# Entorno de trabajo para RNN

## Crear entorno con Python 3.12
py -3.12 -m venv tf-env

## Activarlo
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
tf-env\Scripts\activate

## Ahora python apunta a 3.12 y puedes instalar TensorFlow
python -m pip install --upgrade pip
python -m pip install tensorflow numpy pandas matplotlib scikit-learn torch tensorflow-datasets