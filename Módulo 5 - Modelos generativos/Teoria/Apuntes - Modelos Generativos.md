# Apuntes: Modelos Generativos

## 1. ¿Qué es un modelo generativo?

Un **modelo discriminativo** aprende a predecir P(y|X): dada una entrada, ¿a qué clase pertenece?

Un **modelo generativo** aprende P(X,y) o simplemente p(x): no clasifica, sino que **genera nuevas instancias** similares a los datos de entrenamiento.

El aprendizaje requerido es **no supervisado** (no se necesitan etiquetas). El reto central es capturar la distribución de probabilidad de los datos reales para luego poder muestrear instancias nuevas.

Un concepto clave es el de **variables latentes**: representaciones de menor dimensión que codifican los factores esenciales de los datos. Sirven para controlar la generación, eliminar sesgos y detectar outliers.

---

## 2. Fundamentos de Probabilidad

### Conceptos básicos

- **Espacio muestral Ω**: todos los resultados posibles de un experimento.
- **Medida de probabilidad P**: función que cumple P(A) ≥ 0, P(Ω) = 1, y aditividad para eventos disjuntos.
- **Probabilidad condicional**: P(A|B) = P(A∩B) / P(B)
- **Regla de la cadena**: P(S₁∩...∩Sₖ) = P(S₁)·P(S₂|S₁)·...·P(Sₖ|S₁∩...∩Sₖ₋₁)
- **Independencia**: A y B son independientes si P(A∩B) = P(A)·P(B)

### Distribuciones

- **CDF** (función de distribución acumulativa): FX(x) = P(X ≤ x)
- **PMF** (función de masa de probabilidad): para variables **discretas**, pX(x) = P(X = x)
- **PDF** (función de densidad de probabilidad): para variables **continuas**, fX(x) = dFX(x)/dx. Importante: fX(x) ≠ P(X = x)

### Esperanza y Varianza

- **Esperanza**: E[g(X)] = media ponderada de g(x) por su probabilidad
- **Varianza**: Var[X] = E[(X - E[X])²] = E[X²] - E[X]²
- **Covarianza**: Cov[X,Y] = E[(X - E[X])(Y - E[Y])]
- Si X e Y son independientes → Cov[X,Y] = 0

### Teorema de Bayes

El **Teorema de Bayes** sirve para **actualizar una probabilidad cuando aparece nueva información**.

La idea principal es:

> “Antes pensaba que algo era poco probable, pero después de ver una evidencia, ahora creo que es más (o menos) probable”.

La fórmula es:
$$P(Y|X) = \frac{P(X|Y) \cdot P(Y)}{P(X)}$$

Términos clave:

- **Prior** P(Y): probabilidad antes de observar X
- **Posterior** P(Y|X): probabilidad actualizada tras observar X
- **Likelihood** P(X|Y): verosimilitud
- **Evidence** P(X): normalización

---

## 3. GANs (Redes Generativas Adversarias)

### Idea fundamental

Dos redes neuronales compiten entre sí:

- **Generador G**: crea instancias falsas a partir de ruido z ~ pz
- **Discriminador D**: intenta distinguir instancias reales de falsas

El entrenamiento es un **juego de suma cero** (min-max): $$\min_G \max_D ; \mathbb{E}_{x \sim p_r}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$
Esto significa:

- El **discriminador** quiere maximizar esa función:
    - dar valores altos a imágenes reales
    - dar valores bajos a imágenes falsas
- El **generador** quiere minimizarla:
    - hacer que las falsas parezcan reales
    
Idealmente convergen a un **equilibrio de Nash**.

### Ventajas

- Imágenes más nítidas que otros modelos generativos
- Tamaño del espacio latente configurable
- Generador versátil (no restringido a distribuciones concretas)

### Desventajas

- **Colapso de modo**: G aprende a engañar a D con un solo tipo de muestra
- **Desvanecimiento de gradientes**: D converge demasiado rápido y los gradientes hacia G se vuelven inútiles
- **Inestabilidad**: los parámetros oscilan sin converger

### Arquitecturas derivadas destacadas

|Arquitectura|Aportación principal|
|---|---|
|**SGAN**|Aprovecha datos etiquetados con un cabezal extra|
|**CGAN**|Condicionamiento con variable auxiliar y|
|**DCGAN**|Convoluciones para mayor resolución y estabilidad|
|**PROGAN**|Entrenamiento progresivo desde 4×4 hasta 1024×1024|
|**SAGAN**|Mecanismo de autoatención para capturar dependencias lejanas|
|**BigGAN**|Escalado masivo (x4 parámetros, x8 batch)|
|**StyleGAN**|Control de estilo por capas mediante AdaIN y red de mapeo|

### f-GANs y distancia entre distribuciones

La GAN original minimiza la **divergencia Jensen-Shannon**. Otras métricas posibles:

- **KL divergence**, **Total Variation**, **Wasserstein** (Earth Mover Distance)...
- Cada f-divergencia Df(Q||P) = ∫ p(x)·f(q(x)/p(x))dx con f convexa y f(1)=0

**WGAN** usa la distancia de Wasserstein, que es más estable cuando las distribuciones están muy separadas y resuelve el desvanecimiento de gradientes.

**SSGAN** usa auto-supervisión (predice rotaciones de imágenes) sin etiquetas externas.

**SNGAN** normaliza espectralmente los pesos del discriminador para garantizar que D sea K-Lipschitz, estabilizando el entrenamiento.

---

## 4. Autocodificadores (AE)

### Estructura general

Dos partes:

- **Codificador** Eφ: X → Z (comprime)
- **Decodificador** Dθ: Z → X (reconstruye)

El cuello de botella del espacio latente obliga a comprimir la información esencial. La función de pérdida minimiza la distancia entre x y su reconstrucción x'.

### Problema: memorización

Si el modelo tiene demasiada capacidad puede memorizar los datos (aprender la función identidad). Para evitarlo, se regulariza:

**Autocodificador poco denso (SAE)**: fuerza a que pocas neuronas latentes estén activas.

**Autocodificador con atenuación de ruido (DAE)**: corrompe la entrada durante el entrenamiento y obliga al modelo a reconstruir el original limpio.

**Autocodificador contractivo (CAE)**: penaliza el jacobiano del codificador para que pequeñas variaciones en la entrada produzcan pequeñas variaciones en el espacio latente.

### Aplicaciones típicas

- Compresión de datos
- Clasificación con atributos latentes
- Detección de anomalías (instancias anómalas tienen mayor error de reconstrucción)
- Eliminación de ruido en imágenes

---

## 5. VAEs (Autocodificadores Variacionales)

### Motivación

Un AE clásico no permite generar datos nuevos fácilmente (el espacio latente no es continuo ni estructurado). El VAE añade una formulación probabilística al espacio latente.

### Inferencia variacional

El objetivo es aproximar el posterior p(z|x), que suele ser intratable, mediante una distribución más sencilla **qφ(z|x)** (normalmente gaussiana): $$\hat{\phi} = \arg\min_\phi ; D_{KL}(q_\phi(z|x) | p(z|x))$$

### ELBO (Límite Inferior de Evidencia)

En lugar de minimizar la KL directamente, se maximiza el ELBO: $$\mathcal{L}_{\theta,\phi}(x) = \underbrace{\mathbb{E}_{z \sim q}[\log p(x|z)]}_{\text{reconstrucción}} - \underbrace{D_{KL}(q_\phi(z|x) | p(z))}_{\text{regularización}}$$

Maximizar el ELBO equivale simultáneamente a:

1. Mejorar el modelo generativo (mayor probabilidad marginal)
2. Hacer qφ(z|x) más parecido al posterior real

### Truco de reparametrización

Para poder propagar gradientes a través del muestreo, en vez de muestrear z ~ qφ(z|x), se hace: $$\epsilon \sim \mathcal{N}(0, I), \quad z = \mu + \sigma \odot \epsilon$$

Así el muestreo queda separado de los parámetros φ.

### Problema típico de los VAEs

Las imágenes generadas tienden a ser **borrosas**, porque qφ no es suficientemente expresiva para aproximar bien el posterior real.

### Extensiones destacadas

- **β-VAE**: controla la separación de variables latentes con un parámetro β
- **IWAE**: estimación más precisa del ELBO con múltiples muestras de ruido
- **VQ-VAE**: espacio latente discreto con "libro de códigos"
- **Normalizing Flows / IAF**: posteriores más flexibles mediante transformaciones invertibles

---

## 6. Modelos de Difusión

### Idea central

Inspirados en termodinámica: primero se corrompe progresivamente una imagen añadiendo ruido gaussiano (proceso directo), y luego se entrena una red para **revertir** ese proceso paso a paso.

### Proceso directo (forward)

En T pasos, se añade ruido: $$q(x_t | x_{t-1}) = \mathcal{N}(x_t;, \sqrt{1-\beta_t},x_{t-1},, \beta_t I)$$

Propiedad importante: se puede obtener xₜ directamente desde x₀ sin iterar: $$q(x_t | x_0) = \mathcal{N}(x_t;, \sqrt{\bar{\alpha}_t},x_0,, (1-\bar{\alpha}_t)I)$$

### Proceso inverso (reverse)

Se aproxima q(xₜ₋₁|xₜ) con una red neuronal parametrizada: $$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1};, \mu_\theta(x_t, t),, \Sigma_\theta(x_t, t))$$

La red aprende a **predecir el ruido** ε añadido en cada paso, y la función de pérdida simplificada es: $$L_{simple} = \mathbb{E}\left[|\epsilon - \epsilon_\theta(x_t, t)|^2\right]$$

### Relación con los VAEs

Los modelos de difusión son un caso especial de **VAE jerárquico**: el codificador está fijo (proceso de ruido), las dimensiones latentes son iguales a las de los datos, y el decodificador comparte pesos en todos los pasos.

### Ventajas sobre las GANs

- No requieren entrenamiento adversario (más estables)
- Generan imágenes de mayor calidad perceptual
- Mejor escalabilidad y paralelización

### Extensión: SDEs

El proceso de difusión se puede expresar como una ecuación diferencial estocástica (SDE): $$dx_t = -\tfrac{1}{2}\beta(t)x_t,dt + \sqrt{\beta(t)},d\omega_t$$

Y su inversión se obtiene añadiendo la **función de puntuación** ∇ₓ log qₜ(xₜ), que también se puede aprender con una red neuronal.

---

## Preguntas de Repaso

**¿Qué diferencia fundamental hay entre un modelo discriminativo y uno generativo?** El discriminativo aprende P(y|X) para clasificar. El generativo aprende P(X) o P(X,y) para crear nuevas muestras similares a los datos de entrenamiento.

**¿Qué es el colapso de modo en una GAN y por qué ocurre?** Ocurre cuando el generador aprende a producir solo un subconjunto de la distribución real (un único "modo") porque así engaña al discriminador, aunque no cubra la diversidad total de los datos.

**¿Qué problema resuelve la distancia de Wasserstein frente a la divergencia KL o JS?** La distancia de Wasserstein es continua y proporciona gradientes útiles incluso cuando las distribuciones del generador y los datos reales no se solapan, lo que evita el desvanecimiento de gradientes.

**¿Por qué los VAEs necesitan el truco de reparametrización?** Porque el muestreo z ~ qφ(z|x) no es diferenciable, lo que impide la retropropagación. Al escribir z = μ + σ⊙ε con ε ~ N(0,I), el muestreo se separa de los parámetros y los gradientes pueden calcularse.

**¿Qué mide el ELBO y por qué se maximiza en vez de la log-verosimilitud directamente?** El ELBO es un límite inferior de log p(x). La log-verosimilitud es intratable por requerir integrar sobre todas las variables latentes z. El ELBO es tratable y maximizarlo equivale a maximizar la verosimilitud y minimizar la distancia entre qφ y el posterior verdadero.

**¿En qué se diferencia un VAE de un autocodificador clásico?** El VAE impone una distribución probabilística sobre el espacio latente (generalmente gaussiana), lo que lo hace continuo y estructurado y permite generar nuevas muestras muestreando del espacio latente. El AE clásico no garantiza esta propiedad.

**¿Por qué los modelos de difusión generan imágenes más nítidas que los VAEs?** Porque no asumen una distribución gaussiana simple para el posterior. Al revertir muchos pasos pequeños de ruido, el modelo puede capturar distribuciones mucho más complejas sin la limitación de producir una media ponderada borrosa.

**¿Qué es la función de puntuación en los modelos de difusión basados en SDEs?** Es el gradiente del logaritmo de la densidad marginal ∇ₓ log qₜ(xₜ), que indica la dirección hacia zonas de mayor probabilidad. Aprendiendo esta función con una red neuronal se puede revertir el proceso de difusión para generar nuevas imágenes.

**¿Qué ventaja aporta PROGAN frente a entrenar una GAN directamente a resolución final?** Al empezar con imágenes de 4×4 e ir añadiendo capas progresivamente, la red aprende primero las estructuras globales y luego los detalles finos, lo que estabiliza el entrenamiento y permite alcanzar resoluciones de 1024×1024.

**¿Qué es un autocodificador contractivo y cuál es su objetivo?** Es un AE que penaliza la norma de Frobenius del jacobiano del codificador, forzando a que pequeñas variaciones en la entrada produzcan pequeñas variaciones en el espacio latente. Hace la representación latente robusta frente a perturbaciones locales de la entrada.