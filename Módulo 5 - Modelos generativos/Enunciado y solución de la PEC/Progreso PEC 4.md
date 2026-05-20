## 0. Importaciones necesarias

<p style="color: #00007d;">
  <strong>Algunas consideraciones antes de iniciar con la PEC:</strong>
</p>

<ol style="color: #00007d;">
  <li>Esta PEC está desarrollada con <strong>PyTorch</strong>. Todas las implementaciones, modelos y funciones utilizadas en el notebook están basadas en esta librería.</li>
  <li>Al ejecutar el notebook pueden aparecer mensajes informativos relacionados con PyTorch o con la inicialización de la GPU. <strong>No son errores</strong>, sino avisos propios del entorno. <strong>Puedes ignorarlos</strong>, ya que no afectan al funcionamiento del código.</li>
  <li>Si utilizas GPU en Kaggle, Colab u otro entorno, PyTorch la detectará automáticamente mediante <code>torch.cuda.is_available()</code>. En caso de no disponer de GPU, el notebook funcionará igualmente, aunque el entrenamiento puede requerir más tiempo.</li>
  <li>El notebook utiliza librerías como <code>torch</code>, <code>torchvision</code> y <code>matplotlib</code>. Kaggle ya incluye todas estas librerías, por lo que <strong>no necesitas instalar nada adicional</strong>.</li>
  <li>En la <strong>Etapa 2 (DDPM)</strong> el sampling implica un bucle iterativo de cientos de pasos, por lo que la generación de muestras puede tardar varios minutos. Esto es esperado y forma parte del comportamiento intrínseco de los modelos de difusión.</li>
</ol>

```python
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

# Configurar dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No hay GPU disponible")

# Semilla para reproducibilidad
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed(42)
```

Usando dispositivo: cuda
Tesla T4

## 1. Definición de constantes e hiperparámetros

<p>
Trabajaremos con el conjunto de datos <strong>MNIST</strong>, que contiene imágenes de dígitos manuscritos del 0 al 9.
</p>

<ul>
  <li>Incluye <strong>70.000 imágenes</strong>: <strong>60.000 de entrenamiento</strong> y <strong>10.000 de test</strong>.</li>
  <li>Cada imagen es de <strong>28x28 píxeles</strong> en escala de grises (<strong>1 canal</strong>).</li>
  <li>Las clases son los dígitos del 0 al 9 (<strong>10 categorías</strong>).</li>
</ul>

<p>
En esta práctica trabajaremos con el conjunto de entrenamiento, ya que el objetivo es <strong>generar nuevas imágenes</strong> de dígitos a partir de un vector de ruido (en la GAN) o a partir de ruido gaussiano puro (en el DDPM). Antes de continuar, define las siguientes variables:
</p>

<ul>
  <li><code>batch_size</code>: número de imágenes procesadas por lote durante el entrenamiento. Utilizaremos 128 muestras.</li>
  <li><code>num_channels</code>: cantidad de canales por imagen.</li>
  <li><code>num_classes</code>: total de categorías del dataset.</li>
  <li><code>image_size</code>: tamaño (alto y ancho) de las imágenes.</li>
  <li><code>image_dim</code>: número total de píxeles aplanados por imagen (útil para la GAN basada en MLP).</li>
  <li><code>latent_dim</code>: tamaño del vector de ruido usado como entrada del generador. En este caso utilizaremos 100.</li>
</ul>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 1.1 [0,25 pts.]:</strong>
<p>Completa el siguiente código con los valores correctos para MNIST:</p>
</div>

```python
batch_size = 128                            # Tamaño del lote para entrenamiento
num_channels = 1                            # Número de canales de las imágenes (1 para escala de grises)
num_classes = 10                            # Número de clases en CIFAR-10
image_size = 28                             # Tamaño de las imágenes (28x28 píxeles)
image_dim =  image_size**2 * num_channels   # Dimensión de la imagen aplanada
latent_dim = 100                            # Dimensión del vector latente (ruido)    
```
<!-- Bloque para respuesta del estudiante -->
<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.8em;">
  <strong>Respuesta a la pregunta 1.2:</strong>

  <ul>
    <li>
      <em>
        `num_channels`: MNIST utiliza un solo canal porque las imágenes están en escala de grises, es decir, no contienen información de color. Esto afecta a la arquitectura del generador de dos formas:
      </em>
      <ol>
        <li><em>La capa final del generador produce 1 canal en lugar de 3 (como ocurre en imágenes RGB).</em></li>
        <li><em>El espacio de representación es más simple: 28×28×1 = 784 dimensiones, frente a las 2.352 dimensiones de una imagen RGB.</em></li>
      </ol>
    </li>
    <li>
      <em>
        `latent_dim`: El vector latente es la entrada estocástica que captura la variabilidad de las muestras generadas. El tamaño de este vector tiene diferentes consecuencias:
      </em>
      <ul>
        <li>
          <em>
            <strong>Dimensión muy pequeña (ej. 2):</strong> el generador dispone de poco espacio para codificar la diversidad, por lo que las muestras generadas tienden a ser poco variadas.
          </em>
        </li>
        <li>
          <em>
            <strong>Dimensión muy grande (ej. 1000):</strong> requiere más parámetros y capacidad computacional, lo que dificulta el aprendizaje y aumenta el riesgo de <i>overfitting</i>.
          </em>
        </li>
      </ul>
    </li>
  </ul>
</div>

num_channels: CIFAR-10 utiliza 3 canales ya que son imágenes a color, cada canal corresponde a cada uno de los colores de RGB (rojo, verde y azul), por lo que la forma de los tensores será 32x32x3. Esto implica que el generador producirá imágenes con 3 mapas de características (uno por cada canal de color) y el discriminador debe aceptar imágenes con 3 canales de entrada, lo que aumenta su complejidad, como se describe en el apartado "Modelos generativos basados en redes antagónicas o adversarías de los recursos de aprendizaje.

latent_dim: El vector latente es una representación compacta del espacio de variaciones posibles de las imágenes generadas. Para elegir su dimensión se debe considerar que a mayor dimensión, habrá mayor diversidad en las imágenes generadas. Además, a mayor tamaño del vector, mayor coste computacional del entrenamiento del modelo (cp. 5 Modelos generativos basados en autocodificadores variacionales - VAEs, Modelos Generativos.).

## 2. Carga del dataset y preprocesado

<p>
Cargaremos MNIST con <code>torchvision.datasets.MNIST</code> y aplicaremos las transformaciones necesarias para convertir las imágenes en tensores normalizados al rango <code>[-1, 1]</code>. Esta normalización es importante porque el generador de la GAN tendrá una activación final <code>tanh</code> (que produce salidas en <code>[-1, 1]</code>), y queremos que el discriminador reciba imágenes reales en el mismo rango que las falsas.
</p>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
  <strong>Pregunta 2 [0,25 pts.]:</strong> Completar el siguiente código.
<ul>
  <li><strong>2.1</strong> Define la transformación que convierte la imagen a tensor y la normaliza al rango <code>[-1, 1]</code>.</li>
  <li> <strong>2.2</strong> Carga el conjunto de entrenamiento de MNIST aplicando la transformación definida.</li>
  <li><strong>2.3</strong> Crea el <code>DataLoader</code> con el <code>batch_size</code> definido y mezclando las muestras (<code>shuffle=True</code>).</li>
</ul>
</div>


## Normalización

Explicación:

* `ToTensor():` Convierte la imagen PIL a tensor de rango [0, 1]
* `Normalize(mean=0.5, std=0.5):` Aplica la fórmula (x - 0.5) / 0.5, resultando en rango [-1, 1]
* Esta normalización es crucial porque el generador usará activación tanh que produce salidas en [-1, 1]

```python
# 2.1 Transformaciones para normalizar al rango [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Convierte a tensor en rango [0, 1]
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normaliza a [-1, 1]
])

# 2.2 Carga del dataset de entrenamiento
train_dataset = datasets.MNIST(
    root='./data',          # Directorio de descarga
    train=True,             # Conjunto de entrenamiento
    download=True,          # Descargar si no existe
    transform=transform     # Aplicar transformaciones
)

# 2.3 DataLoader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,           # Mezclar muestras
    pin_memory=True,
    num_workers=2
)

print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)}")
print(f"Número de batches por época: {len(train_loader)}")

# Verificación rápida del rango
sample, _ = train_dataset[0]
print(f"Forma de una imagen: {sample.shape}")
print(f"Rango de valores: [{sample.min():.2f}, {sample.max():.2f}]")
```

100%|██████████| 9.91M/9.91M [00:02<00:00, 3.43MB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 495kB/s]
100%|██████████| 1.65M/1.65M [00:00<00:00, 3.37MB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 4.36MB/s]

Tamaño del dataset de entrenamiento: 60000
Número de batches por época: 469
Forma de una imagen: torch.Size([1, 28, 28])
Rango de valores: [-1.00, 1.00]

```python
# Visualización rápida de algunas imágenes reales del dataset
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    # Des-normalizar para visualizar: (x * 0.5) + 0.5
    img_show = img * 0.5 + 0.5
    ax.imshow(img_show.squeeze(), cmap='gray')
    ax.set_title(f"Etiqueta: {label}")
    ax.axis('off')
plt.suptitle("Muestras reales de MNIST", fontsize=14)
plt.tight_layout()
plt.show()
```

![Muestra reales de MNIST](C:\Users\usuario\Documents\GitHub\Deep-learning\Módulo 5 - Modelos generativos\figures\Muestra reales de MNIST.png)

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
  <strong>Pregunta 2.4 [0,25 pts.]:</strong>
  Explica de forma razonada qué ocurriría si entrenásemos la GAN sin normalizar las imágenes al rango <code>[-1, 1]</code> (por ejemplo, dejándolas en el rango <code>[0, 1]</code> que produce <code>ToTensor()</code>) mientras el generador conserva su activación final <code>tanh</code>.
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 2.4:</strong>
  <p><em>Si entrenamos la GAN sin normalizar las imágenes al rango [-1, 1], dejándolas en [0, 1] mientras el generador mantiene tanh, el discriminador aprendería un "atajo" trivial basándose únicamente en el rango de valores (imágenes reales positivas, generadas con valores negativos de tanh), sin necesidad de aprender características visuales reales. Esto crearía un discriminador que rechaza todo simplemente detectando si los valores son negativos, proporcionando un gradiente engañoso al generador que nunca aprendería a producir dígitos realistas. Además, el generador estaría en conflicto físico con tanh (que no puede producir positivos obligatoriamente), lo que causa inestabilidad.</em></p>
</div>


a) Por un lado, ya que las redes asumen las entradas en rangos pequeños normalizados (0,1 o -1, 1), si las entradas tienen rango (0, 255) las activaciones internas crecerían demasiado y se produciría una explosión del gradiente. Por otro lado, al utilizarse una función tangente hiperbólica como salida del generador, con un rango (-1, 1) se daría una incompatibilidad entre el discriminador y el generador.

b) Al esperar Pytorch un formato "channels-first", interpretaría que la imágen tiene 32 canáles de color, por lo que la red no podría entrenarse ya que se aplicarían los filtros convolucionales sobre ejes incorrectos.

<div style="background-color: #fff8e6; border: 2px solid #ddb86b; padding: 1em; border-radius: 6px; margin-top: 1em;">
<h2 style="color: #8a6700; margin-top: 0;">ETAPA 1 - GAN con MLP (4,5 puntos)</h2>
<p>
En esta primera etapa implementaremos una <strong>GAN simple</strong> donde tanto el generador como el discriminador son <strong>perceptrones multicapa (MLP)</strong>. Aunque las GAN convolucionales (DCGAN) suelen dar mejores resultados, una GAN basada en MLP es suficiente para MNIST y permite ver con claridad la mecánica del juego adversario sin la complejidad añadida de las convoluciones.
</p>
<p>
La idea central es la siguiente:
</p>
<ul>
  <li>El <strong>generador G</strong> recibe un vector de ruido <code>z ~ N(0, I)</code> de dimensión <code>latent_dim=100</code> y produce una imagen de 784 valores en <code>[-1, 1]</code>.</li>
  <li>El <strong>discriminador D</strong> recibe una imagen aplanada (real o falsa) y produce un escalar en <code>[0, 1]</code> que estima la probabilidad de que la imagen sea real.</li>
  <li>Ambas redes se entrenan en alternancia: D quiere maximizar su acierto, G quiere engañar a D.</li>
</ul>
</div>

## 3. Implementación del Generador (MLP)

<p>
El generador es una red densamente conectada que mapea un vector latente <code>z</code> de dimensión <code>latent_dim</code> a una imagen aplanada de <code>image_dim = 784</code> valores. Usaremos una arquitectura sencilla con varias capas ocultas, activaciones <code>LeakyReLU</code> en las capas intermedias y <code>tanh</code> en la salida (para que las imágenes generadas estén en el rango <code>[-1, 1]</code>, igual que las reales).
</p>

<p>Arquitectura propuesta:</p>
<ul>
  <li><code>Linear(latent_dim, 256)</code> → <code>LeakyReLU(0.2)</code></li>
  <li><code>Linear(256, 512)</code> → <code>LeakyReLU(0.2)</code></li>
  <li><code>Linear(512, 1024)</code> → <code>LeakyReLU(0.2)</code></li>
  <li><code>Linear(1024, image_dim)</code> → <code>Tanh()</code></li>
</ul>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 3.1 [0,5 pts.]:</strong> Completa la clase <code>Generator</code> siguiendo la arquitectura descrita arriba. La salida del método <code>forward</code> debe ser un tensor con forma <code>(batch_size, image_dim)</code>.
</div>

**Explicación:** El generador es un MLP que mapea un vector latente de 100 dimensiones a una imagen aplanada de 784 píxeles. Usa capas densas con `LeakyReLU` (para no perder información en negativas) y termina con `tanh` para asegurar que la salida esté en [-1, 1].

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, image_dim):
        super().__init__()
        # Define el bloque secuencial con las capas indicadas
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            
            nn.Linear(1024, image_dim),
            nn.Tanh()
        )

    def forward(self, z):
        # z tiene forma (batch_size, latent_dim)
        # debe retornar una imagen aplanada (batch_size, image_dim)
        return self.net(z)

# Test rápido de la arquitectura
G_test = Generator(latent_dim, image_dim).to(device)
z_test = torch.randn(4, latent_dim, device=device)
out_test = G_test(z_test)
print(f"Entrada (z):  {z_test.shape}")
print(f"Salida (img): {out_test.shape}")
print(f"Rango salida: [{out_test.min():.2f}, {out_test.max():.2f}]")
```

Entrada (z):  torch.Size([4, 100])
Salida (img): torch.Size([4, 784])
Rango salida: [-0.20, 0.17]

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 3.2 [0,5 pts.]:</strong>
Explica brevemente:
<ul>
  <li>¿Por qué usamos <code>tanh</code> como activación final del generador y no <code>sigmoid</code> o ninguna activación?</li>
  <li>¿Por qué usamos <code>LeakyReLU</code> en lugar de <code>ReLU</code> en las capas intermedias del generador?</li>
</ul>
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 3.2:</strong>
  <p><em>a) Usamos tanh porque produce salidas en el rango [-1, 1], que coincide exactamente con el rango de las imágenes reales normalizadas. Si usásemos sigmoid (rango [0, 1]), habría una incongruencia entre distribuciones. Sin activación, los valores serían ilimitados y el discriminador no podría comparar directamente.</em></p>
  <p><em>b) LeakyReLU(0.2) permite que los gradientes negativos fluyan hacia atrás (multiplicados por 0.2), evitando el problema del "dying ReLU". ReLU puro hace zero todos los valores negativos, lo que puede causar que neuronas mueran durante el entrenamiento y dejen de aprender.</em></p>
</div>

## 4. Implementación del Discriminador (MLP)

<p>
El discriminador es una red densa que recibe una imagen aplanada (<code>image_dim</code> valores en <code>[-1, 1]</code>) y produce un escalar en <code>[0, 1]</code> que estima la probabilidad de que la imagen sea real. Usaremos una arquitectura inversa a la del generador:
</p>

<ul>
  <li><code>Linear(image_dim, 1024)</code> → <code>LeakyReLU(0.2)</code> → <code>Dropout(0.3)</code></li>
  <li><code>Linear(1024, 512)</code> → <code>LeakyReLU(0.2)</code> → <code>Dropout(0.3)</code></li>
  <li><code>Linear(512, 256)</code> → <code>LeakyReLU(0.2)</code></li>
  <li><code>Linear(256, 1)</code> → <code>Sigmoid()</code></li>
</ul>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 4.1 [0,5 pts.]:</strong> Completa la clase <code>Discriminator</code> siguiendo la arquitectura descrita.
</div>

**Explicación:** El discriminador es un MLP que recibe una imagen aplanada de 784 píxeles y produce un escalar en [0, 1] (probabilidad de que sea real). La arquitectura es inversa a la del generador: va comprimiendo las dimensiones (1024 → 512 → 256 → 1).

```python
class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        # Define el bloque secuencial
        self.net = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x tiene forma (batch_size, image_dim)
        # debe retornar (batch_size, 1) con valores en [0, 1]
        return self.net(x)

# Test rápido
D_test = Discriminator(image_dim).to(device)
img_test = torch.randn(4, image_dim, device=device)
out_D = D_test(img_test)
print(f"Entrada D: {img_test.shape}")
print(f"Salida D:  {out_D.shape}")
print(f"Rango:     [{out_D.min():.3f}, {out_D.max():.3f}]")
```

Entrada D: torch.Size([4, 784])
Salida D:  torch.Size([4, 1])
Rango:     [0.483, 0.506]

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 4.2 [0,5 pts.]:</strong>
¿Por qué incluimos <code>Dropout</code> en el discriminador y no en el generador? ¿Qué papel cumple aquí?
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 4.2:</strong>
  <p><em>Incluimos Dropout(0.3) en las primeras capas del discriminador como regularización para evitar overfitting y mejorar la estabilidad del entrenamiento adversarial. El discriminador tiende a ser muy poderoso y puede aprender a explotar atajos (como rangos de valores), por lo que dropout lo fuerza a aprender características más robustas. No incluimos dropout en el generador porque queremos que sea lo más determinista posible en su generación, y dropout añadiría ruido innecesario.
  </em></p>
</div>

## 5. Funciones de pérdida y optimizadores

<p>
Para entrenar la GAN usamos la <strong>función de pérdida BCE (Binary Cross-Entropy)</strong>, que mide qué tan bien el discriminador clasifica entre reales (etiqueta 1) y falsas (etiqueta 0).
</p>

<p>
A diferencia de un problema supervisado clásico, aquí necesitamos <strong>dos optimizadores separados</strong>: uno para los parámetros del generador y otro para los del discriminador. La razón es que cada red tiene un objetivo distinto y se actualiza con su propia pérdida.
</p>

<p>
Usaremos el optimizador <strong>Adam</strong> con <code>lr=2e-4</code> y <code>betas=(0.5, 0.999)</code>, que son los hiperparámetros recomendados en el paper original de DCGAN.
</p>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 5.1 [0,5 pts.]:</strong> Completa el código siguiente:
<ul>
  <li><strong>5.1.a</strong> Instancia el generador y el discriminador y envíalos al <code>device</code>.</li>
  <li><strong>5.1.b</strong> Define la función de pérdida <code>BCELoss</code>.</li>
  <li><strong>5.1.c</strong> Define los dos optimizadores Adam, uno para G y otro para D, con los hiperparámetros indicados.</li>
</ul>
</div>

**Explicación:**

* BCELoss: Binary Cross-Entropy es apropiado porque el discriminador realiza una clasificación binaria (real/falso)
* Adam con lr=2e-4 y betas=(0.5, 0.999): Estos son los hiperparámetros recomendados en el paper DCGAN. El learning rate bajo (2e-4) asegura estabilidad, y los betas promueven convergencia lenta y controlada

```python
# 5.1.a Instanciar las redes
G = Generator(latent_dim, image_dim).to(device)
D = Discriminator(image_dim).to(device)

# 5.1.b Pérdida
criterion = nn.BCELoss()

# 5.1.c Optimizadores
lr = 2e-4
betas = (0.5, 0.999)

opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

print("Generador:")
print(G)
print("\nDiscriminador:")
print(D)

# Contar parámetros
n_params_G = sum(p.numel() for p in G.parameters())
n_params_D = sum(p.numel() for p in D.parameters())
print(f"\nParámetros del generador:    {n_params_G:,}")
print(f"Parámetros del discriminador: {n_params_D:,}")
```

Generador:
Generator(
  (net): Sequential(
    (0): Linear(in_features=100, out_features=256, bias=True)
    (1): LeakyReLU(negative_slope=0.2)
    (2): Linear(in_features=256, out_features=512, bias=True)
    (3): LeakyReLU(negative_slope=0.2)
    (4): Linear(in_features=512, out_features=1024, bias=True)
    (5): LeakyReLU(negative_slope=0.2)
    (6): Linear(in_features=1024, out_features=784, bias=True)
    (7): Tanh()
  )
)

Discriminador:
Discriminator(
  (net): Sequential(
    (0): Linear(in_features=784, out_features=1024, bias=True)
    (1): LeakyReLU(negative_slope=0.2)
    (2): Dropout(p=0.3, inplace=False)
    (3): Linear(in_features=1024, out_features=512, bias=True)
    (4): LeakyReLU(negative_slope=0.2)
    (5): Dropout(p=0.3, inplace=False)
    (6): Linear(in_features=512, out_features=256, bias=True)
    (7): LeakyReLU(negative_slope=0.2)
    (8): Linear(in_features=256, out_features=1, bias=True)
    (9): Sigmoid()
  )
)

Parámetros del generador:    1,486,352
Parámetros del discriminador: 1,460,225

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 5.2 [0,25 pts.]:</strong>
¿Por qué definimos dos optimizadores separados (uno para G y otro para D) y no uno solo que actualice todos los parámetros a la vez? ¿Qué problema tendríamos con un único optimizador?
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 5.2:</strong>
  <p><em>Necesitamos dos optimizadores separados porque el generador y el discriminador tienen objetivos opuestos: el discriminador quiere maximizar su acierto (minimizar su pérdida), mientras que el generador quiere engañar al discriminador (minimizar una pérdida diferente). Si usásemos un único optimizador, los gradientes de ambas redes se mezclarían, causando actualizaciones conflictivas que pueden llevar a divergencia del entrenamiento o a un equilibrio incorrecto. Cada optimizador actualiza solo los parámetros de su red, permitiendo que ambas evolucionen según su propia dinámica adversarial.</em></p>
</div>

## 6. Bucle de entrenamiento de la GAN

<p>
El bucle de entrenamiento de una GAN tiene la siguiente estructura para cada batch de imágenes reales:
</p>

<ol>
  <li><strong>Paso del discriminador</strong>:
    <ul>
      <li>Calcular la salida de D sobre las imágenes reales y la pérdida BCE con etiqueta 1.</li>
      <li>Generar imágenes falsas con G a partir de ruido <code>z</code> y calcular la salida de D sobre ellas (con <code>.detach()</code>) y la pérdida BCE con etiqueta 0.</li>
      <li>Sumar ambas pérdidas, hacer backward y <code>opt_D.step()</code>.</li>
    </ul>
  </li>
  <li><strong>Paso del generador</strong>:
    <ul>
      <li>Generar nuevas imágenes falsas con G y pasarlas por D (sin <code>detach()</code>).</li>
      <li>Calcular la pérdida BCE con etiqueta <strong>1</strong> (queremos que D crea que son reales).</li>
      <li>Backward y <code>opt_G.step()</code>.</li>
    </ul>
  </li>
</ol>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 6.1 [1,0 pts.]:</strong> Completa el bucle de entrenamiento de la GAN. Observa que ya están preparadas las variables auxiliares (etiquetas reales/falsas) y el reshape de las imágenes. Solo debes completar las líneas marcadas con <code># TODO</code>.
</div>

**Explicación del flujo:**

* Paso del discriminador: Entrena D a distinguir reales (etiqueta 1) de falsas (etiqueta 0). Usa .detach() en fake_imgs para evitar actualizar G
* Paso del generador: Entrena G a engañar a D. Usa imágenes falsas SIN .detach() para que los gradientes fluyan hasta G


```python
def train_gan(G, D, train_loader, opt_G, opt_D, criterion, n_epochs, latent_dim, device):
    G.train()
    D.train()
    history = {'loss_D': [], 'loss_G': []}

    for epoch in range(n_epochs):
        loss_D_epoch = 0.0
        loss_G_epoch = 0.0

        for batch_idx, (real_imgs, _) in enumerate(train_loader):
            bs = real_imgs.size(0)
            real_imgs = real_imgs.view(bs, -1).to(device)  # aplanar (bs, 784)

            # Etiquetas
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # ---------- 6.1.a Paso del discriminador ----------
            opt_D.zero_grad()

            # Salida de D sobre imágenes reales y su pérdida BCE
            d_real = D(real_imgs)
            loss_real = criterion(d_real, real_labels)

            # Generar imágenes falsas con z ~ N(0, I)
            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = G(z)

            # Salida de D sobre falsas (usar .detach()!) y su pérdida BCE
            d_fake = D(fake_imgs.detach())
            loss_fake = criterion(d_fake, fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # ---------- 6.1.b Paso del generador ----------
            opt_G.zero_grad()

            # Pasar las imágenes falsas por D (sin detach!) y calcular la pérdida
            # con etiqueta REAL (queremos engañar a D)
            d_fake_for_G = D(fake_imgs)
            loss_G = criterion(d_fake_for_G, real_labels)

            loss_G.backward()
            opt_G.step()

            loss_D_epoch += loss_D.item()
            loss_G_epoch += loss_G.item()

        loss_D_epoch /= len(train_loader)
        loss_G_epoch /= len(train_loader)
        history['loss_D'].append(loss_D_epoch)
        history['loss_G'].append(loss_G_epoch)

        print(f"Época {epoch+1}/{n_epochs} | loss_D = {loss_D_epoch:.4f} | loss_G = {loss_G_epoch:.4f}")

    return history

# Entrenamiento de la GAN
N_EPOCHS_GAN = 25  # Aprox. 3-5 minutos en GPU, 20-30 minutos en CPU

set_seed(42)
G = Generator(latent_dim, image_dim).to(device)
D = Discriminator(image_dim).to(device)
opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

history_gan = train_gan(G, D, train_loader, opt_G, opt_D, criterion, N_EPOCHS_GAN, latent_dim, device)
```
Época 1/25 | loss_D = 0.8039 | loss_G = 1.9615
Época 2/25 | loss_D = 0.5137 | loss_G = 3.5043
Época 3/25 | loss_D = 0.5851 | loss_G = 2.9235
Época 4/25 | loss_D = 0.4095 | loss_G = 3.1753
Época 5/25 | loss_D = 0.3740 | loss_G = 3.1249
Época 6/25 | loss_D = 0.3851 | loss_G = 2.8207
Época 7/25 | loss_D = 0.4830 | loss_G = 2.5436
Época 8/25 | loss_D = 0.5757 | loss_G = 2.3240
Época 9/25 | loss_D = 0.5894 | loss_G = 2.2555
Época 10/25 | loss_D = 0.7089 | loss_G = 1.9713
Época 11/25 | loss_D = 0.7844 | loss_G = 1.8308
Época 12/25 | loss_D = 0.8338 | loss_G = 1.6889
Época 13/25 | loss_D = 0.8924 | loss_G = 1.5765
Época 14/25 | loss_D = 0.9175 | loss_G = 1.5223
Época 15/25 | loss_D = 0.9230 | loss_G = 1.4934
Época 16/25 | loss_D = 0.9418 | loss_G = 1.4947
Época 17/25 | loss_D = 0.9636 | loss_G = 1.4269
Época 18/25 | loss_D = 0.9976 | loss_G = 1.3812
Época 19/25 | loss_D = 1.0010 | loss_G = 1.3816
Época 20/25 | loss_D = 1.0179 | loss_G = 1.3473
Época 21/25 | loss_D = 1.0201 | loss_G = 1.3406
Época 22/25 | loss_D = 1.0418 | loss_G = 1.3109
Época 23/25 | loss_D = 1.0475 | loss_G = 1.3022
Época 24/25 | loss_D = 1.0413 | loss_G = 1.3075
Época 25/25 | loss_D = 1.0494 | loss_G = 1.2827

```python
# Curvas de pérdida
plt.figure(figsize=(8, 4))
plt.plot(history_gan['loss_D'], label='Loss D', color='tab:blue')
plt.plot(history_gan['loss_G'], label='Loss G', color='tab:orange')
plt.xlabel('Época')
plt.ylabel('Pérdida (BCE)')
plt.title('Curvas de pérdida de la GAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

![Curva de perdida de la GAN](C:\Users\usuario\Documents\GitHub\Deep-learning\Módulo 5 - Modelos generativos\figures\Curva de perdida de la GAN.png)

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 6.2 [0,25 pts.]:</strong>
¿Por qué es necesario llamar a <code>.detach()</code> sobre las imágenes falsas cuando se calcula la pérdida del discriminador? ¿Qué pasaría si lo olvidásemos?
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 6.2:</strong>
  <p><em>El .detach() asegura que cada red solo se actualiza en su propio paso, manteniendo la dinámica adversarial correcta. .detach() desconecta el grafo de computación, evitando que los gradientes de la pérdida del discriminador fluyan hacia atrás al generador. Si lo olvidásemos, durante el paso del discriminador estaríamos inadvertidamente actualizando también los parámetros de G (a través de los gradientes de D), causando conflictos. G se actualizaría tanto en su propio paso como en el paso de D, resultando en inestabilidad total. </em></p>
</div>

## 7. Generación de muestras con la GAN entrenada

<p>
Una vez entrenada la GAN, podemos generar nuevas imágenes simplemente sampleando vectores latentes <code>z</code> de una normal estándar y pasándolos por el generador. Recordamos que las imágenes salen en <code>[-1, 1]</code>, por lo que para visualizarlas debemos des-normalizarlas a <code>[0, 1]</code>.
</p>

```python
def generar_muestras_gan(G, n_samples, latent_dim, device):
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        fake_imgs = G(z)
        # Reshape a (n, 1, 28, 28) y des-normalizar
        fake_imgs = fake_imgs.view(n_samples, 1, image_size, image_size)
        fake_imgs = fake_imgs * 0.5 + 0.5  # de [-1,1] a [0,1]
    return fake_imgs


def mostrar_grid(imgs, nrow=8, title=""):
    grid = make_grid(imgs.cpu(), nrow=nrow, padding=2)
    plt.figure(figsize=(nrow, imgs.size(0) // nrow))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# Generar 64 muestras de la GAN
muestras_gan = generar_muestras_gan(G, n_samples=64, latent_dim=latent_dim, device=device)
mostrar_grid(muestras_gan, nrow=8, title="Muestras generadas por la GAN (MLP)")
```

![Muestras generadas por la GAN (MLP)](C:\Users\usuario\Documents\GitHub\Deep-learning\Módulo 5 - Modelos generativos\figures\Muestras generadas por la GAN (MLP).png)

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 7 [0,5 pts.]:</strong>
Observa el grid de imágenes generadas por la GAN. Comenta:
<ul>
  <li><code>[0,25 pts.]</code> ¿Qué calidad visual presentan? ¿Se distinguen los dígitos? ¿Hay artefactos como ruido, blur o estructuras irregulares?</li>
  <li><code>[0,25 pts.]</code> ¿Observas algún caso de <em>mode collapse</em> (varios <code>z</code> distintos producen el mismo dígito)? ¿Cómo lo verificarías de forma sistemática?</li>
</ul>
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 7:</strong>
  <p><em>a) Las imágenes generadas presentan características de dígitos manuscritos (como 0, 1, 2, 3, etc.), aunque con diferente grado de claridad. Los contornos están generalmente definidos y se distinguen formas reconocibles de dígitos en la mayoría de casos. Pueden observarse artefactos como: suavidad excesiva (blur), ruido granular, o estructuras incompletas en ciertos dígitos. La calidad generalmente mejora con más épocas de entrenamiento, aunque una GAN basada en MLP tiene limitaciones arquitectónicas comparada con DCGAN (que usa convoluciones).</em></p>
  <p><em>b) El mode collapse ocurre cuando la GAN genera principalmente un mismo tipo de dígito (ej. solo 0s y 1s). Para verificarlo sistemáticamente: (1) Entrenar un clasificador MNIST preentrenado sobre las muestras generadas y analizar la distribución de predicciones (¿están balanceadas entre 0-9 o concentradas en pocos dígitos?); (2) Calcular la diversidad de muestras usando métricas como Inception Score (IS) o Fréchet Inception Distance (FID); (3) Generar múltiples vectores latentes aleatorios y observar si producen dígitos variados o repetidos. En nuestro caso, con 25 épocas de entrenamiento típicamente observamos cierta diversidad, aunque no perfecta.</em></p>
</div>


<div style="background-color: #fff8e6; border: 2px solid #ddb86b; padding: 1em; border-radius: 6px; margin-top: 1em;">
<h2 style="color: #8a6700; margin-top: 0;">ETAPA 2 - Modelo de difusión DDPM (4,5 puntos)</h2>

<p>
En esta segunda etapa implementaremos un <strong>Denoising Diffusion Probabilistic Model (DDPM)</strong> desde cero, sobre el mismo dataset MNIST. A diferencia de las GAN, los modelos de difusión <strong>no tienen un juego adversario</strong>: se entrenan con una sola pérdida MSE estable y previsible.
</p>

<h4 style="color: #8a6700;">Idea central</h4>

<p>Un modelo de difusión define dos procesos:</p>

<ol>
  <li><strong>Forward process (proceso directo)</strong>: tomamos una imagen real <code>x_0</code> y le añadimos ruido gaussiano de forma progresiva en <code>T</code> pasos hasta convertirla en ruido puro <code>x_T ~ N(0, I)</code>. Este proceso está fijado de antemano (no se aprende).</li>
  <li><strong>Reverse process (proceso inverso)</strong>: aprendemos una red neuronal que, dado <code>x_t</code> (imagen ruidosa en el paso <code>t</code>), predice el ruido que se añadió. Aplicando esta red iterativamente desde <code>t=T</code> hasta <code>t=0</code>, podemos partir de ruido puro y obtener una imagen limpia.</li>
</ol>

<p>
El milagro del DDPM es que aunque el forward process es fijo, podemos calcular <code>x_t</code> a partir de <code>x_0</code> en un solo paso (sin pasar por todos los intermedios) gracias a una propiedad de la cadena de Markov gaussiana. Esto hace el entrenamiento muy rápido.
</p>
</div>

```python
# Configurar dispositivo (GPU si está disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No hay GPU disponible")
```

Usando dispositivo: cuda
Tesla T4

## 8. Forward process: añadir ruido a las imágenes

<p>
El forward process se define con un <strong>schedule de varianzas</strong> $\beta_1, \beta_2, \ldots, \beta_T$ que crece linealmente desde un valor pequeño $\beta_1$ hasta un valor mayor $\beta_T$. Aquí usaremos:
</p>

<ul>
  <li>$T = 1000$ (número total de pasos de difusión)</li>
  <li>$\beta_1 = 10^{-4}$, $\beta_T = 0.02$ (schedule lineal)</li>
</ul>

<p>A partir de los $\beta_t$ definimos:</p>

<ul>
  <li>$\alpha_t = 1 - \beta_t$</li>
  <li>$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ (producto acumulado)</li>
</ul>

<p>La propiedad mágica del DDPM es que <strong>podemos saltarnos a cualquier paso $t$ directamente</strong> con la fórmula cerrada:</p>

<p style="text-align:center;">
$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim N(0, I)$$
</p>

<p>donde $\epsilon$ es ruido gaussiano estándar. Esto se llama la función <code>q_sample</code> y es la única función de forward que necesitamos.</p>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 8.1 [0,5 pts.]:</strong> Completa la función que crea el schedule de difusión. Debes calcular <code>betas</code>, <code>alphas</code> y <code>alphas_cumprod</code>.
</div>

```python
T = 1000           # número de pasos de difusión
beta_start = 1e-4
beta_end   = 0.02

def make_diffusion_schedule(T, beta_start, beta_end, device):
    # 8.1.a: betas linealmente espaciados
    betas = torch.linspace(beta_start, beta_end, T, device=device)

    # 8.1.b: alphas = 1 - betas
    alphas = 1.0 - betas

    # 8.1.c: producto acumulado
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        'betas':                          betas,
        'alphas':                         alphas,
        'alphas_cumprod':                 alphas_cumprod,
        'sqrt_alphas_cumprod':            torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod':  torch.sqrt(1.0 - alphas_cumprod),
    }

  schedule = make_diffusion_schedule(T, beta_start, beta_end, device)

# Verificación
print(f"betas:           min={schedule['betas'].min():.5f},  max={schedule['betas'].max():.4f}")
print(f"alphas_cumprod:  t=0 → {schedule['alphas_cumprod'][0]:.4f},  t=T-1 → {schedule['alphas_cumprod'][-1]:.5f}")
```

betas:           min=0.00010,  max=0.0200
alphas_cumprod:  t=0 → 0.9999,  t=T-1 → 0.00004

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 8.2 [0,3 pts.]:</strong>
Implementa la función <code>q_sample(x_0, t, noise)</code> que aplica la fórmula del forward process en un paso $t$ arbitrario:
$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon$$
</div>

```python
def q_sample(x_0, t, noise, schedule):
    # x_0:   imagen limpia,        forma (B, 1, 28, 28)
    # t:     paso de difusión,      forma (B,) tensor de enteros
    # noise: ruido eps,             forma (B, 1, 28, 28)

    # Recogemos los coeficientes correspondientes a cada t del batch
    sqrt_acp = schedule['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
    sqrt_omacp = schedule['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)

    # Aplica la fórmula
    x_t = sqrt_acp * x_0 + sqrt_omacp * noise

    return x_t
```

### Visualización del forward process

<p>
Para entender intuitivamente lo que hace el forward process, tomemos una imagen real de MNIST y veamos cómo se va degradando al aumentar $t$:
</p>

```python
# Tomamos una imagen real
x_0_demo, _ = train_dataset[0]
x_0_demo = x_0_demo.unsqueeze(0).to(device)  # (1, 1, 28, 28)

# Visualización en distintos pasos t
ts_a_visualizar = [0, 30, 100, 200, 300, 500, 700, 900, T - 1]

fig, axes = plt.subplots(1, len(ts_a_visualizar), figsize=(16, 2.5))
for i, t_val in enumerate(ts_a_visualizar):
    t = torch.tensor([t_val], device=device)
    noise = torch.randn_like(x_0_demo)
    x_t = q_sample(x_0_demo, t, noise, schedule)

    img_show = (x_t * 0.5 + 0.5).clamp(0, 1)  # des-normalizar
    axes[i].imshow(img_show.cpu().squeeze().numpy(), cmap='gray')
    axes[i].set_title(f"t = {t_val}")
    axes[i].axis('off')

plt.suptitle("Forward process: añadiendo ruido gaussiano a una imagen MNIST", fontsize=12)
plt.tight_layout()
plt.show()
```

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 8.3 [0,2 pts.]:</strong>
Mirando la visualización anterior, explica con tus palabras:
<ul>
  <li>¿Qué representa intuitivamente el forward process?</li>
  <li>¿Por qué es importante que <code>alphas_cumprod[T-1]</code> sea muy pequeño (cercano a 0)?</li>
</ul>
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 8.3:</strong>
  <p><em>El forward process es una cadena de Markov gaussiana que corrompe progresivamente la imagen real x0x_0
x0​ añadiendo ruido en cada paso. A tt
t pequeño la imagen sigue siendo reconocible; a tt
t grande (t ≈ T) la imagen es indistinguible de ruido blanco gaussiano puro. Visualmente, las primeras columnas de la figura muestran el dígito nítido, mientras que las últimas son estática uniforme.</em></p>
  <p><em>Porque αˉT−1≈0\bar\alpha_{T-1} \approx 0
αˉT−1​≈0 implica que αˉT−1≈0\sqrt{\bar\alpha_{T-1}} \approx 0
αˉT−1​​≈0 y 1−αˉT−1≈1\sqrt{1-\bar\alpha_{T-1}} \approx 1
1−αˉT−1​​≈1. En ese límite, xT=αˉT x0+1−αˉT ϵ≈ϵx_T = \sqrt{\bar\alpha_T}\,x_0 + \sqrt{1-\bar\alpha_T}\,\epsilon \approx \epsilon
xT​=αˉT​​x0​+1−αˉT​​ϵ≈ϵ, es decir, la distribución de xTx_T
xT​ es prácticamente N(0,I)\mathcal{N}(0, I)
N(0,I) independientemente del contenido de x0x_0
x0​. Esto es esencial: el punto de partida del sampling (reverse process) debe ser ruido puro, y solo lo es si el forward process ha destruido completamente la información original. Si αˉT−1\bar\alpha_{T-1}
αˉT−1​ fuera mayor que 0, el sampling comenzaría desde una distribución que no es pura Gaussiana y el proceso generativo estaría mal condicionado.</em></p>

</div>


## 9. Modelo predictor de ruido

<p>
El corazón del DDPM es una red neuronal $\epsilon_\theta(x_t, t)$ que recibe una imagen ruidosa $x_t$ y el paso $t$, y predice el ruido $\epsilon$ que se le añadió. Para mantener la simplicidad didáctica, usaremos una <strong>red MLP simple</strong> en lugar de una UNet convolucional. Sigue siendo un DDPM válido, solo con menos capacidad.
</p>

<p>
Hay dos elementos que necesitamos resolver:
</p>

<ol>
  <li><strong>Codificación del paso $t$</strong>: el modelo debe saber en qué paso de difusión está. Usaremos un <strong>embedding sinusoidal</strong>, similar al usado en Transformers para posiciones, que convierte un entero $t$ en un vector denso.</li>
  <li><strong>Combinación de imagen y embedding del tiempo</strong>: aplanaremos la imagen y la concatenaremos con el embedding de $t$ antes de pasarla por el MLP.</li>
</ol>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 9.1 [0,3 pts.]:</strong>
Completa la función <code>sinusoidal_time_embedding(t, dim)</code> que produce un embedding sinusoidal del paso de tiempo. La idea es la misma que en Transformer: para cada par de dimensiones, alternamos $\sin$ y $\cos$ con frecuencias geométricamente decrecientes.
</div>

```python
def sinusoidal_time_embedding(t, dim):
    # t:    (B,) tensor de enteros (los timesteps)
    # dim:  dimensión del embedding (par)

    half = dim // 2
    # frecuencias geométricamente decrecientes
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device).float() / half
    )

    # Calcula el argumento (B, half) como t (B,1) * freqs (1,half)
    args = t[:, None].float() * freqs[None, :]          # (B, half)

    # Concatena sin y cos a lo largo de la última dimensión -> (B, dim)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)

    return emb

# Test rápido
t_test = torch.tensor([0, 50, 150, 299], device=device)
emb_test = sinusoidal_time_embedding(t_test, dim=32)
print(f"Embedding shape: {emb_test.shape}")
print(f"Rango: [{emb_test.min():.3f}, {emb_test.max():.3f}]")
```

Embedding shape: torch.Size([4, 32])
Rango: [-1.000, 1.000]

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 9.2 [0,5 pts.]:</strong>
Completa la clase <code>SimpleDenoiser</code> con la siguiente arquitectura:
<ul>
  <li><strong>Embedding del tiempo:</strong> aplica <code>sinusoidal_time_embedding(t, dim=time_dim)</code> y proyéctalo con un MLP <code>Linear(time_dim, hidden_dim) → SiLU → Linear(hidden_dim, hidden_dim)</code>.</li>
  <li><strong>Proyección de la imagen:</strong> aplana <code>x</code> de <code>(B, 1, 28, 28)</code> a <code>(B, image_dim)</code> y proyecta con <code>Linear(image_dim, hidden_dim)</code>.</li>
  <li><strong>Combinación e inyección del tiempo:</strong> <strong>suma</strong> el embedding del tiempo a la representación de la imagen (broadcasting) y procesa el resultado con un MLP de 3 capas <code>Linear(hidden_dim, hidden_dim)</code> intercaladas con <code>SiLU</code>.</li>
  <li><strong>Salida:</strong> capa final <code>Linear(hidden_dim, image_dim)</code> <strong>sin activación</strong> (la salida es el ruido predicho, valores reales sin restricción), y reshape de vuelta a <code>(B, 1, 28, 28)</code>.</li>
</ul>
<p>Debes completar tanto el constructor (definición de los bloques) como el método <code>forward</code>.</p>
</div>

```python
class SimpleDenoiser(nn.Module):
    def __init__(self, image_dim, time_dim=64, hidden_dim=1024):
        super().__init__()
        self.time_dim = time_dim

        # 9.2.a: MLP para procesar el embedding del tiempo
        # Estructura: Linear(time_dim, hidden_dim) -> SiLU -> Linear(hidden_dim, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 9.2.b: proyección de la imagen aplanada al espacio oculto
        # Una sola capa lineal: Linear(image_dim, hidden_dim)
        self.input_proj = nn.Linear(image_dim, hidden_dim)

        self.main = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, image_dim),
        )

    def forward(self, x, t):
        # x: (B, 1, 28, 28)   imagen ruidosa
        # t: (B,) ints        timesteps de difusión
        B = x.size(0)

        # 9.2.c: calcula el embedding sinusoidal del tiempo y proyéctalo con time_mlp
        # Pista: primero sinusoidal_time_embedding(t, self.time_dim), luego self.time_mlp(...)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)   # (B, time_dim)
        t_emb = self.time_mlp(t_emb)                          # (B, hidden_dim)
        # 9.2.d: aplana x a (B, image_dim) y proyéctala con input_proj
        h = x.view(B, -1)       # (B, image_dim)
        h = self.input_proj(h)  # (B, hidden_dim)

        # 9.2.e: suma el embedding del tiempo a la representación de la imagen
        # y pasa el resultado por el bloque principal
        h = h + t_emb           # (B, hidden_dim)
        h = self.main(h)        # (B, image_dim)

        # Reshape de vuelta a imagen (B, 1, 28, 28)
        return h.view(B, 1, image_size, image_size)


# Test rápido (no modificar)
denoiser_test = SimpleDenoiser(image_dim).to(device)
x_test = torch.randn(4, 1, 28, 28, device=device)
t_test = torch.randint(0, T, (4,), device=device)
out = denoiser_test(x_test, t_test)
print(f"Entrada x: {x_test.shape}")
print(f"Entrada t: {t_test.shape}")
print(f"Salida (ruido predicho): {out.shape}")
```

Entrada x: torch.Size([4, 1, 28, 28])
Entrada t: torch.Size([4])
Salida (ruido predicho): torch.Size([4, 1, 28, 28])

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 9.3 [0,2 pts.]:</strong>
La red predice <strong>el ruido $\epsilon$</strong> que se añadió, en lugar de predecir directamente la imagen limpia $x_0$. ¿Por qué? Da al menos dos razones.
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 9.3:</strong>
  <p><em>La red predice el ruido ε en lugar de la imagen limpia x0x_0
x0​ por dos razones principales:

Escala de valores más estable: el ruido ε es siempre N(0,I)\mathcal{N}(0,I)
N(0,I) independientemente del contenido de x0x_0
x0​, lo que hace que el objetivo de regresión tenga la misma escala para todos los pasos tt
t. Si en cambio predijéramos x0x_0
x0​ directamente, para tt
t pequeños la señal a predecir varía mucho según la imagen real, dificultando el entrenamiento.
Conexión directa con la fórmula del reverse process: la actualización μt\mu_t
μt​ del DDPM se expresa en términos de ϵ^\hat\epsilon
ϵ^ de forma cerrada y matemáticamente limpia (derivada directamente del ELBO variacional del proceso gaussiano). Predecir x0x_0
x0​ requeriría una reparametrización adicional, añadiendo complejidad sin ventaja práctica.</em></p>
</div>


## 10. Entrenamiento del DDPM

<p>
El entrenamiento de un DDPM es notablemente más simple que el de una GAN. Para cada batch:
</p>

<ol>
  <li>Tomar imágenes reales $x_0$.</li>
  <li>Samplear un paso $t$ aleatorio para cada imagen del batch (uniforme en $[0, T)$).</li>
  <li>Samplear ruido $\epsilon \sim N(0, I)$.</li>
  <li>Calcular $x_t = q_{sample}(x_0, t, \epsilon)$.</li>
  <li>Predecir el ruido con la red: $\hat\epsilon = \epsilon_\theta(x_t, t)$.</li>
  <li>Pérdida MSE: $L = \|\hat\epsilon - \epsilon\|^2$.</li>
  <li>Backward y step.</li>
</ol>

<p>
No hay juego adversario, no hay <code>detach()</code>, no hay dos optimizadores. Una sola red, una sola pérdida.
</p>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 10.1 [0,75 pts.]:</strong> Completa el bucle de entrenamiento del DDPM. Solo debes rellenar las líneas marcadas con <code># TODO</code>.
</div>

```python
def train_ddpm(model, train_loader, optimizer, n_epochs, T, schedule, device):
    model.train()
    history = {'loss': []}

    for epoch in range(n_epochs):
        loss_epoch = 0.0

        for batch_idx, (x_0, _) in enumerate(train_loader):
            x_0 = x_0.to(device)
            B = x_0.size(0)

            optimizer.zero_grad(set_to_none=True)

            # TODO 10.1.a: samplear un paso t aleatorio para cada imagen (B,) en [0, T)
            t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

            # TODO 10.1.b: samplear ruido eps con la misma forma que x_0
            noise = torch.randn_like(x_0)

            # TODO 10.1.c: aplicar el forward process para obtener x_t
            x_t = q_sample(x_0, t, noise, schedule)

            # TODO 10.1.d: predecir el ruido con el modelo
            noise_pred = model(x_t, t)

            # TODO 10.1.e: pérdida MSE entre el ruido real y el predicho
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_epoch += loss.item()

        loss_epoch /= len(train_loader)
        history['loss'].append(loss_epoch)
        print(f"Época {epoch+1}/{n_epochs} | loss = {loss_epoch:.4f}")

    return history

# Entrenamiento del DDPM
# Aprox. 05 minutos en GPU, 95% de uso de GPU T4, No ejecutarlo en CPU
N_EPOCHS_DDPM = 25

denoiser = SimpleDenoiser(image_dim, time_dim=128, hidden_dim=3072).to(device)

opt_ddpm = optim.AdamW(
    denoiser.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

n_params_dn = sum(p.numel() for p in denoiser.parameters())
print(f"Parámetros del denoiser: {n_params_dn:,}\n")

history_ddpm = train_ddpm(denoiser, train_loader, opt_ddpm, N_EPOCHS_DDPM, T, schedule, device)
```

Parámetros del denoiser: 33,537,808

Época 1/25 | loss = 0.6247
Época 2/25 | loss = 0.3366
Época 3/25 | loss = 0.2138
Época 4/25 | loss = 0.1648
Época 5/25 | loss = 0.1551
Época 6/25 | loss = 0.1514
Época 7/25 | loss = 0.1547
Época 8/25 | loss = 0.1526
Época 9/25 | loss = 0.1540
Época 10/25 | loss = 0.1523
Época 11/25 | loss = 0.1520
Época 12/25 | loss = 0.1508
Época 13/25 | loss = 0.1515
Época 14/25 | loss = 0.1521
Época 15/25 | loss = 0.1521
Época 16/25 | loss = 0.1510
Época 17/25 | loss = 0.1520
Época 18/25 | loss = 0.1515
Época 19/25 | loss = 0.1535
Época 20/25 | loss = 0.1518
Época 21/25 | loss = 0.1513
Época 22/25 | loss = 0.1494
Época 23/25 | loss = 0.1507
Época 24/25 | loss = 0.1508
Época 25/25 | loss = 0.1498

```python
# Curva de pérdida del DDPM
plt.figure(figsize=(8, 4))
plt.plot(history_ddpm['loss'], color='tab:purple')
plt.xlabel('Época')
plt.ylabel('MSE loss')
plt.title('Curva de pérdida del DDPM')
plt.grid(True, alpha=0.3)
plt.show()
```

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 10.2 [0,25 pts.]:</strong>
Compara la curva de pérdida del DDPM con la de la GAN (sección 6). ¿Qué diferencias observas? ¿Qué te dice cada una sobre la dinámica de entrenamiento de su modelo?
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 10.2:</strong>
  <p><em>La curva del DDPM muestra una pérdida MSE que desciende de forma suave y monótona desde los primeros pasos, con muy poca varianza entre épocas. Esto refleja que el DDPM tiene un único objetivo bien definido (predicción de ruido) y no hay dinámica adversarial.
En contraste, las curvas de la GAN son mucho más inestables: la pérdida de D y la de G fluctúan y se "persiguen" mutuamente, porque el equilibrio del juego minimax es un punto silla y cualquier ventaja momentánea de uno perturba al otro. Es habitual ver la pérdida de G subir cuando D mejora y viceversa.
En la práctica esto significa que entrenar el DDPM es considerablemente más predecible: si la curva baja, el modelo está mejorando. En la GAN una pérdida baja de D puede ser señal de colapso del generador, no de buen entrenamiento.</em></p>
</div>

## 11. Sampling: generación de muestras (reverse process)

<p>
Una vez entrenado el modelo, generamos imágenes nuevas con el siguiente algoritmo iterativo:
</p>

<ol>
  <li>Inicializar $x_T \sim N(0, I)$.</li>
  <li>Para $t = T-1, T-2, \ldots, 0$:
    <ul>
      <li>Predecir el ruido: $\hat\epsilon = \epsilon_\theta(x_t, t)$.</li>
      <li>Calcular la media: $\mu_t = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \hat\epsilon \right)$</li>
      <li>Si $t > 0$: $x_{t-1} = \mu_t + \sqrt{\beta_t}\, z$, con $z \sim N(0, I)$.</li>
      <li>Si $t = 0$: $x_{t-1} = \mu_t$ (no añadimos ruido en el último paso).</li>
    </ul>
  </li>
  <li>Devolver $x_0$ como muestra generada.</li>
</ol>

<p>
Esta es la fórmula del DDPM en su versión más simple. Cada paso es un denoising "un poquito": eliminamos parte del ruido y añadimos un poco más controlado para mantener la estocasticidad del proceso.
</p>

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 11.1 [0,75 pts.]:</strong>
Completa la función <code>p_sample_loop</code> que ejecuta el reverse process completo para generar <code>n_samples</code> imágenes nuevas.
</div>


```python
@torch.no_grad()
def p_sample_loop(model, n_samples, T, schedule, device, image_size=28, return_history=False):
    model.eval()

    betas = schedule['betas']
    alphas = schedule['alphas']
    alphas_cumprod = schedule['alphas_cumprod']
    sqrt_omacp = schedule['sqrt_one_minus_alphas_cumprod']

    # 11.1.a: inicializar x_T como ruido gaussiano puro de forma (n_samples, 1, 28, 28)
    x = torch.randn(n_samples, 1, image_size, image_size, device=device)

    history = [x.clone()] if return_history else None

    # 11.1.b: bucle desde T-1 hasta 0 (incluido)
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)

        # 11.1.c: predecir el ruido con el modelo
        noise_pred = model(x, t_batch)

        # Coeficientes para el paso t
        beta_t = betas[t]
        alpha_t = alphas[t]
        sqrt_omacp_t = sqrt_omacp[t]

        # 11.1.d: calcular la media mu_t segun la fórmula
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / sqrt_omacp_t) * noise_pred
        )

        if t > 0:
            # 11.1.e: añadir ruido sqrt(beta_t) * z, z ~ N(0,I)
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * noise
        else:
            x = mean

        if return_history and t % 30 == 0:
            history.append(x.clone()) # type: ignore

    if return_history:
        return x, history
    return x


def mostrar_grid(imgs, nrow=8, title=""):
    grid = make_grid(imgs.cpu(), nrow=nrow, padding=2)
    plt.figure(figsize=(nrow, imgs.size(0) // nrow))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Generar 64 muestras con el DDPM
print("Sampling DDPM (puede tardar un momento)...")
muestras_ddpm = p_sample_loop(denoiser, n_samples=64, T=T, schedule=schedule, device=device)

# Des-normalizar y mostrar
muestras_ddpm_show = (muestras_ddpm * 0.5 + 0.5).clamp(0, 1)
mostrar_grid(muestras_ddpm_show, nrow=8, title="Muestras generadas por el DDPM")
```

### Visualización del proceso de denoising

<p>
Para entender qué hace el reverse process, mostremos cómo evoluciona <strong>la misma muestra</strong> a lo largo de los pasos de sampling, partiendo de ruido puro hasta una imagen reconocible:
</p>

```python
# Generar 8 muestras guardando el historial
muestras_hist, historia = p_sample_loop(
    denoiser, n_samples=8, T=T, schedule=schedule, device=device, return_history=True
)

# Plot: filas = muestra, columnas = paso del proceso (de ruido a imagen)
n_steps_plot = len(historia)
fig, axes = plt.subplots(8, n_steps_plot, figsize=(2 * n_steps_plot, 16))
for fila in range(8):
    for col in range(n_steps_plot):
        img = (historia[col][fila] * 0.5 + 0.5).clamp(0, 1)
        axes[fila, col].imshow(img.cpu().squeeze().numpy(), cmap='gray')
        axes[fila, col].axis('off')
        if fila == 0:
            t_label = T - col * 30
            axes[fila, col].set_title(f"t≈{max(t_label, 0)}", fontsize=10)

plt.suptitle("Reverse process: de ruido (izq.) a imagen (der.)", fontsize=14)
plt.tight_layout()
plt.show()
```

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 11.2 [0,25 pts.]:</strong>
Compara el coste computacional de generar muestras con la GAN frente al DDPM. Para los mismos 64 ejemplos, ¿cuál es más rápido y por qué? ¿Qué consecuencias prácticas tiene esto?
</div>

## 12. Comparación cualitativa GAN vs Difusión

<p>
Para cerrar la práctica, mostraremos lado a lado las muestras generadas por la GAN entrenada en la Etapa 1 y las generadas por el DDPM entrenado en la Etapa 2, sobre el mismo dataset MNIST.
</p>

```python
# Generamos 32 muestras de cada modelo
muestras_gan_cmp = generar_muestras_gan(G, n_samples=32, latent_dim=latent_dim, device=device)
print("Sampling DDPM (puede tardar)...")
muestras_ddpm_cmp = p_sample_loop(denoiser, n_samples=32, T=T, schedule=schedule, device=device)
muestras_ddpm_cmp = (muestras_ddpm_cmp * 0.5 + 0.5).clamp(0, 1)

# Plot lado a lado
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

grid_gan = make_grid(muestras_gan_cmp.cpu(), nrow=8, padding=2)
axes[0].imshow(grid_gan.permute(1, 2, 0).numpy(), cmap='gray')
axes[0].set_title(f"GAN-MLP ({N_EPOCHS_GAN} épocas)", fontsize=14)
axes[0].axis('off')

grid_ddpm = make_grid(muestras_ddpm_cmp.cpu(), nrow=8, padding=2)
axes[1].imshow(grid_ddpm.permute(1, 2, 0).numpy(), cmap='gray')
axes[1].set_title(f"DDPM ({N_EPOCHS_DDPM} épocas)", fontsize=14)
axes[1].axis('off')

plt.suptitle("Comparación cualitativa: GAN vs Difusión sobre MNIST", fontsize=15)
plt.tight_layout()
plt.show()
```

<div style="background-color: #EDF7FF; border-left: 5px solid #7C9DBF; padding: 0.5em;">
<strong>Pregunta 12 [0,5 pts.]:</strong>
A partir de la comparación visual y de tu experiencia entrenando ambos modelos en este notebook, completa la siguiente tabla en tu respuesta razonando cada criterio:

<table style="margin-top: 1em; width: 100%; border-collapse: collapse;">
  <thead style="background-color: #f0f0f0;">
    <tr>
      <th style="border: 1px solid #ccc; padding: 6px; text-align: left;">Criterio</th>
      <th style="border: 1px solid #ccc; padding: 6px;">GAN-MLP</th>
      <th style="border: 1px solid #ccc; padding: 6px;">DDPM</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="border: 1px solid #ccc; padding: 6px;">Calidad visual</td><td style="border: 1px solid #ccc; padding: 6px;">?</td><td style="border: 1px solid #ccc; padding: 6px;">?</td></tr>
    <tr><td style="border: 1px solid #ccc; padding: 6px;">Diversidad de muestras</td><td style="border: 1px solid #ccc; padding: 6px;">?</td><td style="border: 1px solid #ccc; padding: 6px;">?</td></tr>
    <tr><td style="border: 1px solid #ccc; padding: 6px;">Estabilidad del entrenamiento</td><td style="border: 1px solid #ccc; padding: 6px;">?</td><td style="border: 1px solid #ccc; padding: 6px;">?</td></tr>
    <tr><td style="border: 1px solid #ccc; padding: 6px;">Velocidad de generación</td><td style="border: 1px solid #ccc; padding: 6px;">?</td><td style="border: 1px solid #ccc; padding: 6px;">?</td></tr>
    <tr><td style="border: 1px solid #ccc; padding: 6px;">Interpretabilidad de la pérdida</td><td style="border: 1px solid #ccc; padding: 6px;">?</td><td style="border: 1px solid #ccc; padding: 6px;">?</td></tr>
  </tbody>
</table>

</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">
  <strong>Respuesta a la pregunta 12:</strong>
  <p><em><table>
<thead>
<tr><th>Criterio</th><th>GAN-MLP</th><th>DDPM</th></tr>
</thead>
<tbody>
<tr>
<td><strong>Calidad visual</strong></td>
<td>Media-buena. Los dígitos son reconocibles pero aparecen borrosidades, contornos incompletos y ocasionalmente artefactos. La arquitectura MLP carece de inducción convolucional para capturar estructura espacial fina.</td>
<td>Buena-muy buena. Los dígitos presentan trazo más nítido y natural, con menos ruido residual. El proceso iterativo de denoising permite recuperar detalles locales que el MLP generador de la GAN no alcanza.</td>
</tr>
<tr>
<td><strong>Diversidad de muestras</strong></td>
<td>Variable. Puede aparecer cierto mode collapse suave (sobrerepresentación de dígitos fáciles como el 1). Con 25 épocas y MLP la diversidad es aceptable pero no uniforme en los 10 dígitos.</td>
<td>Alta. Al no haber juego adversario, el modelo no colapsa: genera representantes de todos los dígitos con distribución más equilibrada. El muestreo estocástico en cada paso del reverse process fomenta la diversidad.</td>
</tr>
<tr>
<td><strong>Estabilidad del entrenamiento</strong></td>
<td>Inestable. Las pérdidas de G y D oscilan mutuamente; el equilibrio minimax es frágil. Se requiere ajustar con cuidado el balance entre ambas redes para evitar colapso o dominancia de D.</td>
<td>Muy estable. Una única pérdida MSE que desciende monotónicamente sin oscilaciones. No hay dinámica adversarial ni riesgo de colapso.</td>
</tr>
<tr>
<td><strong>Velocidad de generación</strong></td>
<td>Muy rápida. Un único pase por el generador (~1 ms para 64 imágenes en GPU). Apta para aplicaciones en tiempo real.</td>
<td>Lenta. Requiere T = 1000 pasos del reverse process (~varios segundos para 64 imágenes en GPU). Inapropiada para tiempo real sin samplers acelerados (DDIM, etc.).</td>
</tr>
<tr>
<td><strong>Interpretabilidad de la pérdida</strong></td>
<td>Baja. Las pérdidas BCE de G y D no tienen un significado absoluto fácil de interpretar: un valor de 0.69 (log(2)) corresponde teóricamente al equilibrio, pero en la práctica las curvas fluctúan sin que su nivel indique calidad visual directamente.</td>
<td>Alta. La pérdida MSE mide directamente el error de predicción de ruido. Una pérdida más baja implica que el modelo predice el ruido con mayor precisión, lo que se traduce directamente en mejor calidad generativa.</td>
</tr>
</tbody>
</table></em></p>
</div>

<div style="background-color: #fcf2f2; border-left: 5px solid #dfb5b4; padding: 0.5em;">

# **Referencias**

Fuentes bibliográficas, recursos de aprendizaje, documentación técnica y herramientas de inteligencia artificial utilizadas en la elaboración de esta PEC.

<ul>

### **Recursos de aprendizaje**

<li>

<strong>De la Torre Gallart, J. (2024).</strong> <em> Modelos generativos. [Recurso de aprendizaje textual]. 1.ª ed. Barcelona: Fundació Universitat Oberta de Catalunya (FUOC).

<strong>Utilizado en:</strong> Ejercicios 1, 2, 3, 4, 5, 6 y 7

Sustituye este texto por el que deberia ir en esta practica de modelos generativos:
* Proporciona los <strong>fundamentos teóricos de la arquitectura de los Modelos generativos</strong>, incluyendo el mecanismo de self-attention, la codificación posicional y la técnica de <em>Knowledge Distillation</em>.
* Utilizado como referencia principal para comprender el <strong>diseño de modelos encoder-only tipo BERT</strong>, las estrategias de fine-tuning downstream y los principios que guían la construcción de modelos personalizados como <code>MiniBertForSequenceClassification</code>.

</li>

### **Fuentes Bibliográficas y Documentación Técnica**

<li>

<strong>HuggingFace. (2026).</strong> Documentación oficial: Transformers, Datasets y Hub.<br>
<a href="https://huggingface.co/docs/transformers">https://huggingface.co/docs/transformers</a>

<strong>Utilizado en:</strong> Ejercicios 4, 5 y 7

* Utilizado como referencia técnica para el uso de <strong><code>AutoTokenizer</code>, <code>AutoModelForSequenceClassification</code>, <code>Trainer</code> y <code>TrainingArguments</code></strong> en el fine-tuning de <code>bert-mini-uncased</code> y <code>distilbert-base-uncased</code>.
* Empleado para la carga de los modelos preentrenados <strong><code>gaunernst/bert-mini-uncased</code></strong> y <strong><code>distilbert-base-uncased</code></strong> desde el Hub, así como para la gestión del <code>DataCollatorWithPadding</code> y el scheduler de learning rate.

</li>

<li>

<strong>PyTorch Team. (2026).</strong> Documentación oficial de PyTorch.<br>
<a href="https://pytorch.org">https://pytorch.org</a>

<strong>Utilizado en:</strong> Ejercicios 1, 2, 3, 6 y 7

* Utilizado como referencia para la implementación del <strong>bucle de entrenamiento manual</strong>, incluyendo <code>nn.LSTM</code>, <code>nn.MultiheadAttention</code>, <code>nn.TransformerEncoderLayer</code>, <code>Dataset</code>, <code>DataLoader</code> y optimizadores como <code>AdamW</code>.
* Empleado para confirmar el uso correcto del parámetro <strong><code>need_weights=True</code></strong> en <code>nn.MultiheadAttention</code>, necesario para extraer los pesos de atención en el ejercicio 3, y para implementar la <strong>función de pérdida combinada (KL Divergence + Cross-Entropy)</strong> en el ejercicio 7.

</li>

<li>

<strong>OpenAI. (2026).</strong> <em>ChatGPT (GPT-5.3).</em> /
<strong>Anthropic. (2026).</strong> <em>Claude (Haiku-4.5).</em>

<strong>Utilizado en:</strong> Ejercicios 1, 2, 3, 4, 5, 6 y 7

* Utilizado como <strong>herramienta de apoyo</strong> para revisar explicaciones conceptuales y resolver dudas técnicas puntuales durante el desarrollo de los ejercicios.

</li>

</ul>
</div>