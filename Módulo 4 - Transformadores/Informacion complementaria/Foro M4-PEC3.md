# Foro M4 - PEC3

## Consulta 1: Recuperar pesos de atención en el ejercicio 3

**Autor:** David Cobo Prieto  
**Fecha:** 3 de mayo, 23:09  
**Última respuesta:** 7 de mayo, 11:58

### Pregunta

En el ejercicio 3 se indica:

> “Utiliza `return_attentions=True` en tu modelo para recuperar los pesos de la matriz de atención.”

Sin embargo, en la documentación lo más parecido que aparece es:

> `need_weights` (`bool`) – If specified, returns `attn_output_weights` in addition to `attn_outputs`. Set `need_weights=False` to use the optimized `scaled_dot_product_attention` and achieve the best performance for MHA. Default: `True`.

¿Es correcto usar este parámetro para obtener los pesos?

### Respuesta del profesor

**Autor:** Abelardo Carlos Martínez Lorenzo  
**Fecha:** 7 de mayo, 11:58  
**Última edición:** 7 de mayo, 12:01

Hola David,

Sí, es correcto. En PyTorch, `need_weights=True` es el parámetro que permite recuperar los pesos de atención en `MultiheadAttention`.

Saludos.

---

## Consulta 2: Número razonable de épocas de entrenamiento

**Autor:** Mario Méndez Martín  
**Fecha:** 1 de mayo, 12:08  
**Última respuesta:** 7 de mayo, 12:04

### Pregunta

¿Cuántas épocas es razonable que entrenemos los modelos?

He usado 10 épocas y el rendimiento es bastante malo, pero si aumento las épocas el tiempo de entrenamiento va a aumentar demasiado, por lo que no sé qué hacer.

Un saludo.

### Respuesta de un compañero

**Autor:** Alejandro Martínez Molina  
**Fecha:** 5 de mayo, 18:50

Buenas Mario,

Yo estoy de momento en el ejercicio 2 y he probado 15 épocas con `EarlyStopping`, que de momento me está saltando, y el tiempo de entrenamiento no ha sido para nada elevado. Eso sí, estoy usando Kaggle Notebooks; no he probado a hacerlo con mi GPU en local.

### Respuesta del profesor

**Autor:** Abelardo Carlos Martínez Lorenzo  
**Fecha:** 7 de mayo, 12:04

Hola Mario,

No hay un número fijo de épocas obligatorio. Puedes usar pocas épocas, por ejemplo **10-15**, y apoyarte en **early stopping** si ves que la validación deja de mejorar.

En este ejercicio no se espera obtener una generación de texto perfecta. Lo importante es que:

- El entrenamiento esté bien planteado.
- Compares LSTM y Transformer de forma consistente.
- Muestres las curvas de:
  - `train loss`
  - `validation loss`
  - `validation accuracy`
- Comentes los resultados.

Saludos.

---

## Consulta 3: Referencias para implementar la PEC3 con PyTorch

**Autora:** Beatriz Padín Romero  
**Fecha:** 20 de abril, 20:56  
**Última respuesta:** 23 de abril, 18:28

### Pregunta

Hola,

Estoy empezando a hacer la PEC3 y me encuentro bastante perdida. Se trata de implementar los modelos con PyTorch, pero no he visto en la bibliografía referencias que me ayuden a llevar a cabo los ejercicios.

Para la red LSTM seguiré lo que hice en el ejercicio 3 de la PEC2, pero no veo nada fácil afrontar los demás ejercicios. Lo que me despista es que en ese ejercicio de la PEC2 se dice:

> “Dado que PyTorch no se ha trabajado todavía en profundidad en la asignatura, el ejercicio se plantea de forma guiada.”

Pero no veo que se nos hayan dado nociones de PyTorch en ninguno de los documentos de la bibliografía, aparte del libro *Dive into Deep Learning*, que no es para nada adecuado para empezar.

¿Es que me he saltado algo en los recursos de aprendizaje, o quizás se supone que deberíamos controlar PyTorch antes de empezar?

He estado buscando bibliografía por mi cuenta, pero no acabo de encontrar cosas que me gusten. Agradecería mucho alguna referencia básica para poder llevar a cabo la prueba.

Gracias y saludos.

### Respuesta del profesor

**Autor:** Abelardo Carlos Martínez Lorenzo  
**Fecha:** 23 de abril, 14:41  
**Última edición:** 23 de abril, 14:49

Hola Beatriz,

Tienes razón en que la bibliografía base del M4 no introduce PyTorch.

Por esa razón, en la página de la PEC3, además del enunciado actual, en el apartado de contenidos y recursos también hemos subido la **PEC3 del curso pasado resuelta**, precisamente para que tengáis una referencia de trabajo en PyTorch dentro de la propia asignatura.

Además, se ha incluido como recurso el libro *Dive into Deep Learning*:

- https://d2l.ai

En concreto, el **capítulo 11**, donde se trabajan transformers con implementación en PyTorch.

Dicho esto, entendemos que la PEC requiere cierto trabajo autónomo y que puede ser necesario apoyarse también en documentación adicional. Por eso, si lo necesitáis, os podemos orientar hacia algunos enlaces oficiales de PyTorch especialmente útiles para esta práctica.

Como referencias básicas, recomendamos especialmente la documentación oficial de PyTorch sobre:

- Tensores y modelos básicos.
- `Dataset` y `DataLoader`.
- Construcción de modelos con `nn.Module`.
- Para la parte de modelos preentrenados, la documentación de Hugging Face con PyTorch.

Una buena estrategia para orientarse es:

1. Reutilizar la lógica de la LSTM de la PEC2 adaptándola a PyTorch.
2. Montar después una versión sencilla del Transformer.
3. Dejar para el final la parte de fine-tuning con Hugging Face.

### Enlaces de interés

#### Para ejercicios 1-3: PyTorch puro

- PyTorch Learn the Basics
- Quickstart
- Datasets & DataLoaders
- Building the Neural Network
- Optimizing Model Parameters
- TransformerEncoderLayer
- TransformerEncoder

#### Opcional para el ejercicio 3

- MultiheadAttention

#### Para ejercicios 4-7: Hugging Face y modelos preentrenados

- Transformers Quickstart
- Text classification
- Trainer

Saludos,

Abelardo.

### Respuesta de Beatriz

**Autora:** Beatriz Padín Romero  
**Fecha:** 23 de abril, 18:28

Muchas gracias por la información.

Un saludo.