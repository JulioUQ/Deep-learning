# Dataset de Clasificación de Escenas de Uso del Suelo

Dataset de imágenes satelitales para la **clasificación automática de tipos de uso del suelo**.

> Debido al peso de los datos no se han incluido en el repositorio, si se quieren utilizar el enlace se encuentra o en el cuaderno jupyter de la PEC o en la descripcion del repositorio.

## 1. Objetivo del dataset

El objetivo principal de este conjunto de datos es **identificar y clasificar diferentes tipos de uso del suelo en imágenes satelitales**.

Las imágenes proceden de satélites **Landsat** y representan diferentes entornos terrestres. Este dataset se utiliza principalmente para **entrenar modelos de visión por computador y aprendizaje profundo (Deep Learning)** capaces de reconocer automáticamente distintos tipos de paisajes o estructuras en imágenes satelitales.

Este tipo de modelos se aplica en ámbitos como:

- planificación urbana
- gestión del territorio
- monitorización ambiental
- análisis geoespacial

---

## 2. Contenido del dataset

El dataset contiene **imágenes satelitales organizadas en 21 categorías de uso del suelo**, por ejemplo:

- edificios
- campos de béisbol
- autopistas
- zonas residenciales
- áreas agrícolas
- bosques
- aparcamientos
- ríos
- playas

### Características de las imágenes

- **Tamaño de imagen:** 256 × 256 píxeles
- **Número original de imágenes por clase:** 100
- **Número total de clases:** 21

Para aumentar la cantidad de datos disponibles para el entrenamiento de modelos de inteligencia artificial, se aplicaron **técnicas de aumento de datos (*data augmentation*)**.

Cada imagen original fue transformada **4 veces mediante variaciones**, generando nuevas versiones de la misma imagen.

De esta forma:

- **Imágenes finales por clase:** 500
- **Número total aproximado de imágenes:** 10.500

Este aumento del dataset permite **entrenar modelos más robustos y con mejor capacidad de generalización**.

---

## 3. Origen del dataset

Este dataset deriva del conocido **UC Merced Land Use Dataset**, desarrollado por la **Universidad de California en Merced**.

Las imágenes fueron:

- **extraídas manualmente** de imágenes satelitales de gran tamaño
- procedentes de la colección **USGS National Map Urban Area Imagery**
- correspondientes a **diferentes zonas urbanas de Estados Unidos**

### Resolución espacial

Las imágenes originales tienen una resolución aproximada de:

- **1 pie por píxel** (aproximadamente **30 cm por píxel**)

---

## 4. Publicación científica asociada

Si se publican resultados utilizando este dataset, se debe citar el siguiente trabajo:

**Yi Yang y Shawn Newsam (2010)**  
_"Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification"_  
ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS).

Este trabajo propone técnicas para clasificar imágenes de uso del suelo utilizando **representaciones visuales basadas en características espaciales**.

---
