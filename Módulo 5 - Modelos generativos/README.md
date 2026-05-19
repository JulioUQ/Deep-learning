# Módulo 5: Modelos generativos

Este módulo introduce los fundamentos de los **modelos generativos** dentro del ámbito del _Deep Learning_. A diferencia de los modelos predictivos tradicionales, cuyo objetivo principal es clasificar o predecir resultados, los modelos generativos son capaces de **crear nuevos datos** similares a los utilizados durante el entrenamiento.

Actualmente, los modelos generativos representan una de las áreas más relevantes y activas de investigación en inteligencia artificial, siendo la base de tecnologías modernas como:

- Generación de imágenes
- Creación de texto
- Síntesis de audio y vídeo
- Modelos multimodales
- Inteligencia artificial generativa

El contenido del módulo proporciona una base teórica sólida para comprender los enfoques más importantes utilizados actualmente.

---

# Competencias y Objetivos

Los objetivos principales del módulo son:

- Definir y comprender los modelos generativos como una nueva categoría de modelos predictivos.

- Analizar distintas aproximaciones para la generación de datos mediante aprendizaje profundo.

- Introducir los fundamentos teóricos de:
    - _Generative Adversarial Networks_ (**GANs**)
    - Auto-codificadores variacionales (**VAEs**)
    - Modelos basados en difusión

- Comprender las ventajas, limitaciones y aplicaciones de cada enfoque.

- Adquirir la base necesaria para continuar profundizando mediante bibliografía especializada e investigación actual.

---

# Contenidos del módulo

A lo largo del módulo se estudiarán diferentes arquitecturas y técnicas generativas, entre ellas:

## GANs (_Generative Adversarial Networks_)

Las GANs utilizan dos redes neuronales enfrentadas:

- Un **generador**, encargado de crear datos falsos.
- Un **discriminador**, encargado de distinguir entre datos reales y generados.

Ambas redes compiten entre sí durante el entrenamiento, permitiendo generar datos cada vez más realistas.

### Aplicaciones comunes

- Generación de imágenes
- Super-resolución
- Deepfakes
- Transferencia de estilo

---

## Auto-codificadores variacionales (VAE)

Los VAEs permiten aprender una representación latente probabilística de los datos para posteriormente generar nuevas muestras similares.

### Características principales

- Aprendizaje no supervisado
- Representación compacta de los datos
- Generación controlada de muestras

### Aplicaciones comunes

- Generación de imágenes
- Reducción de dimensionalidad
- Compresión de datos

---

## Modelos basados en difusión

Los modelos de difusión generan datos eliminando progresivamente ruido añadido a una muestra aleatoria.

Actualmente son la base de muchos sistemas modernos de IA generativa.

### Aplicaciones comunes

- Generación de imágenes fotorrealistas
- Edición de imágenes
- Modelos tipo Stable Diffusion y DALL·E

---

# Conclusión

El estudio de los modelos generativos constituye una de las áreas más innovadoras e importantes del Deep Learning moderno. Este módulo proporciona una introducción sólida a las arquitecturas más relevantes utilizadas actualmente y sienta las bases para continuar explorando el campo de la inteligencia artificial generativa avanzada.

---

# Recursos y bibliografía

## Bibliografía básica

- **M7 - Modelos Generativos**
## Bibliografía accesoria

- [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow](https://discovery.biblioteca.uoc.edu/discovery/fulldisplay?context=L&vid=34CSUC_UOC%3AVU1&search_scope=MyInst_and_CI&isFrbr=true&tab=Everything&docid=alma991000734279006712&lang=es&utm_source=chatgpt.com)
- [DeepAI (Dive into Deep Learning)](https://d2l.ai/?utm_source=chatgpt.com)