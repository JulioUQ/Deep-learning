<div align="center">

![Curso](https://img.shields.io/badge/CURSO-6A1B9A?style=for-the-badge) ![Ruta de ciencia de datos 3/3](https://img.shields.io/badge/RUTA_DE_CIENCIA_DE_DATOS-3%2F3-8E24AA?style=for-the-badge)

# Deep learning

Proyectos de aprendizaje profundo con su teoría: redes densas, convolucionales, recurrentes, transformers y modelos generativos con TensorFlow/Keras y PyTorch.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![TensorFlow · Keras](https://img.shields.io/badge/TensorFlow_%C2%B7_Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

<sub>Ruta de ciencia de datos: [1. Estadística básica](https://github.com/JulioUQ/Curso-de-Estadistica-Basica) → [2. Aprendizaje automático](https://github.com/JulioUQ/curso-aprendizaje-automatico) → **3. Deep learning**</sub>

</div>

---

## Descripción general

Tercer paso de la ruta de ciencia de datos, después de [aprendizaje automático](https://github.com/JulioUQ/curso-aprendizaje-automatico). Cada módulo reúne la teoría y un proyecto completo, desde la preparación de los datos hasta la evaluación del modelo.

## Contenido

| Módulo | Proyecto | Técnicas |
| --- | --- | --- |
| [1. Redes neuronales](M%C3%B3dulo%201%20-%20Redes%20neuronales%20artificiales) | Fundamentos | Perceptrón, retropropagación, funciones de activación |
| [2. Redes convolucionales](M%C3%B3dulo%202%20-%20Redes%20neuronales%20convolucionales%20%28CNN%29) | Clasificación de imágenes satelitales (UC Merced) | Red densa, CNN, autoencoders, *transfer learning* con EfficientNetB0, CNN en PyTorch |
| [3. Redes recurrentes](M%C3%B3dulo%203%20-%20Redes%20neuronales%20recurrentes%20%28RNN%29) | Predicción de series temporales (Electricity Transformer Dataset) | SimpleRNN, LSTM y GRU, predicción multivariante, ajuste con Optuna |
| [4. Transformers](M%C3%B3dulo%204%20-%20Transformadores) | Generación de texto (WikiText-2) | LSTM frente a Transformer en PyTorch, Hugging Face `transformers` y `datasets` |
| [5. Modelos generativos](M%C3%B3dulo%205%20-%20Modelos%20generativos) | Generación de imágenes de dígitos (MNIST) | GAN con perceptrones y modelo de difusión (DDPM) |
| [6. Estado del arte](M%C3%B3dulo%206%20-%20Estado%20del%20arte) | Revisión de avances recientes | — |

## Tecnologías y herramientas

| Ámbito | Herramientas |
| --- | --- |
| Frameworks | TensorFlow/Keras, PyTorch, torchvision |
| NLP | Hugging Face transformers, datasets, evaluate |
| Optimización | Optuna |
| Datos y gráficos | pandas, NumPy, matplotlib, seaborn |

## Instalación y uso

Con Python 3.12:

```bash
py -3.12 -m pip install -r requirements.txt
jupyter lab
```

Los módulos 2 a 5 se benefician de una GPU; también se pueden ejecutar en Google Colab.

## Aspectos técnicos destacados

- El mismo problema resuelto con arquitecturas de complejidad creciente para comparar su rendimiento.
- Uso de Keras y PyTorch en paralelo.
- Comparación cualitativa entre dos familias de modelos generativos: GAN y difusión.

## Autor

**Julio Úbeda Quesada** · Data Scientist

[![Portfolio](https://img.shields.io/badge/Portfolio-0B2545?style=flat-square&logo=astro&logoColor=white)](https://juliouq.github.io/Portfolio-Web---JulioUQ/es/) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square)](https://www.linkedin.com/in/juliouq/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/JulioUQ)

<sub>Material elaborado durante el Máster Universitario en Ciencia de Datos.</sub>
