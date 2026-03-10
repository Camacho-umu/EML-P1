# Documentación: Brazos Bernoulli y Binomial

## Índice

1. [Introducción](#1-introducción)
2. [Brazo Bernoulli (`ArmBernoulli`)](#2-brazo-bernoulli-armbernoulli)
3. [Brazo Binomial (`ArmBinomial`)](#3-brazo-binomial-armbinomial)
4. [Relación entre distribuciones](#4-relación-entre-distribuciones)
5. [Decisiones de implementación](#5-decisiones-de-implementación)
6. [Ejemplo de ejecución](#6-ejemplo-de-ejecución)

---

## 1. Introducción

En el problema del bandido de k-brazos, cada brazo (acción) devuelve una recompensa regida por una distribución de probabilidad desconocida para el agente. El tipo de distribución que genera las recompensas condiciona tanto la naturaleza del problema como el comportamiento de los algoritmos de selección.

El proyecto base del profesor proporcionaba únicamente el brazo con distribución **Normal** (`ArmNormal`). Según el enunciado de la práctica (sección 4.1.2), se requiere implementar también brazos con distribuciones **Bernoulli** y **Binomial**, ya que cada una modela un tipo de problema real diferente:

| Distribución | Recompensa | Ejemplo del mundo real |
|---|---|---|
| **Bernoulli** | Binaria: 0 o 1 | Publicidad online (clic vs. no clic) |
| **Binomial** | Entero en [0, n] | Promociones en app de delivery (cuántos de n usuarios usan la oferta) |
| **Normal** | Continua en ℝ | Tiempo de visualización en plataformas de streaming |

Ambas clases heredan de la clase abstracta `Arm` y deben implementar los métodos:
- `pull()`: genera una recompensa aleatoria según la distribución.
- `get_expected_value()`: devuelve el valor esperado teórico (E[X]).
- `generate_arms(k)`: genera k brazos con parámetros aleatorios.

---

## 2. Brazo Bernoulli (`ArmBernoulli`)

### 2.1 Fundamento teórico

La distribución de Bernoulli modela un experimento con exactamente **dos resultados posibles**: éxito (1) o fracaso (0), con una probabilidad de éxito *p*.

$$X \sim \text{Bernoulli}(p)$$

- **Soporte**: $X \in \{0, 1\}$
- **Valor esperado**: $E[X] = p$
- **Varianza**: $\text{Var}[X] = p(1 - p)$

Es el caso más simple de distribución discreta y resulta un caso particular de la distribución Binomial con n=1: $\text{Bernoulli}(p) \equiv B(1, p)$.

### 2.2 Caso de uso

El enunciado de la práctica (sección 5.1) describe el ejemplo de **publicidad online**:

> *Un anunciante elige entre k=3 anuncios diferentes. Cada anuncio tiene una probabilidad desconocida de que un usuario haga clic en él (CTR - Click-Through Rate). Al mostrar un anuncio, se observa si el usuario hizo clic (éxito = 1) o no (fracaso = 0).*

Otros ejemplos:
- Suministro de medicamentos: ¿funciona el tratamiento? (sí/no)
- Test A/B en interfaces web: ¿el usuario convierte? (sí/no)

### 2.3 Implementación

```python
class ArmBernoulli(Arm):
    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."
        self.p = p
```

**Parámetro único**: la probabilidad de éxito `p ∈ [0, 1]`.

#### `pull()`

```python
def pull(self):
    reward = np.random.binomial(1, self.p)
    return reward
```

Se utiliza `np.random.binomial(1, p)` en lugar de, por ejemplo, `np.random.choice([0, 1], p=[1-p, p])`. La razón es que la Bernoulli es formalmente una Binomial con n=1, por lo que esta implementación es:
- **Correcta teóricamente**: respeta la definición formal.
- **Eficiente**: `binomial(1, p)` está altamente optimizado en NumPy.
- **Coherente**: facilita ver la relación con `ArmBinomial`.

#### `get_expected_value()`

```python
def get_expected_value(self) -> float:
    return self.p
```

Devuelve directamente `p`, ya que $E[X] = p$ para una Bernoulli.

#### `generate_arms(k)`

```python
@classmethod
def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
    p_values = []
    while len(p_values) < k:
        p = round(np.random.uniform(p_min, p_max), 3)
        if p not in p_values:
            p_values.append(p)
    arms = [ArmBernoulli(p) for p in p_values]
    return arms
```

- Se generan `k` probabilidades únicas en el rango `[p_min, p_max]`.
- Se evitan extremos (0 y 1) para que todos los brazos tengan variabilidad.
- Las probabilidades se redondean a 3 decimales para legibilidad y unicidad.

---

## 3. Brazo Binomial (`ArmBinomial`)

### 3.1 Fundamento teórico

La distribución Binomial modela el **número de éxitos en n ensayos independientes**, cada uno con probabilidad de éxito *p*.

$$X \sim B(n, p)$$

- **Soporte**: $X \in \{0, 1, 2, \ldots, n\}$
- **Valor esperado**: $E[X] = n \cdot p$
- **Varianza**: $\text{Var}[X] = n \cdot p \cdot (1 - p)$

### 3.2 Caso de uso

El enunciado (sección 5.1) describe el ejemplo de **promociones en una app de delivery**:

> *Una app de delivery tiene k=4 promociones. Cada día, se muestra una promoción a un lote de n=100 usuarios nuevos. La recompensa es el número de usuarios que usaron la promoción. Esto se modela como $B(100, p_i)$, donde $p_i$ es la probabilidad (desconocida) de conversión de la promoción i.*

Otros ejemplos:
- Marketing por email: de n emails enviados, cuántos se abren.
- Control de calidad: de n piezas fabricadas, cuántas son defectuosas.

### 3.3 Implementación

```python
class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."
        self.n = n
        self.p = p
```

**Dos parámetros**: el número de ensayos `n` y la probabilidad de éxito `p`.

#### `pull()`

```python
def pull(self):
    reward = np.random.binomial(self.n, self.p)
    return reward
```

Genera directamente una muestra de la distribución $B(n, p)$ usando NumPy.

#### `get_expected_value()`

```python
def get_expected_value(self) -> float:
    return self.n * self.p
```

Devuelve $n \cdot p$, el valor esperado teórico de la distribución binomial.

#### `generate_arms(k, n=100)`

```python
@classmethod
def generate_arms(cls, k: int, n: int = 100, p_min: float = 0.1, p_max: float = 0.9):
    p_values = []
    while len(p_values) < k:
        p = round(np.random.uniform(p_min, p_max), 3)
        if p not in p_values:
            p_values.append(p)
    arms = [ArmBinomial(n, p) for p in p_values]
    return arms
```

- Todos los brazos comparten el **mismo n** (por defecto 100, como en el ejemplo de la práctica).
- Lo que varía es la probabilidad `p` de cada brazo.
- El valor por defecto `n=100` modela el escenario de lotes de 100 usuarios.

---

## 4. Relación entre distribuciones

Las tres distribuciones implementadas están relacionadas matemáticamente:

```
Bernoulli(p) ≡ B(1, p)    ← caso particular de la Binomial con n=1
```

Además, bajo ciertas condiciones, la Binomial se puede aproximar por una Normal:

```
Si np ≥ 5 y n(1-p) ≥ 5:
    B(n, p) ≈ N(μ = np,  σ² = np(1-p))
```

Esto permite al alumno experimentar con la misma familia de problemas a distintos niveles de abstracción y observar cómo los algoritmos se comportan según la naturaleza de la recompensa (binaria, discreta acotada, o continua).

### Tabla comparativa de propiedades

| Propiedad | Bernoulli(p) | B(n, p) | N(μ, σ²) |
|---|---|---|---|
| **Tipo** | Discreta | Discreta | Continua |
| **Soporte** | {0, 1} | {0, ..., n} | (-∞, +∞) |
| **E[X]** | p | n·p | μ |
| **Var[X]** | p(1-p) | np(1-p) | σ² |
| **Parámetros** | p | n, p | μ, σ |
| **Rango recompensa** | Muy estrecho | Medio | Amplio |

---

## 5. Decisiones de implementación

### 5.1 Uso de lista en lugar de set para `generate_arms`

La implementación original del profesor en `ArmNormal` usaba un `set()` para almacenar valores únicos. Nosotros cambiamos esto a una **lista con comprobación de duplicados**:

```python
# Antes (no reproducible):
p_values = set()
while len(p_values) < k:
    p = round(np.random.uniform(0.1, 0.9), 3)
    p_values.add(p)

# Después (reproducible):
p_values = []
while len(p_values) < k:
    p = round(np.random.uniform(0.1, 0.9), 3)
    if p not in p_values:
        p_values.append(p)
```

**¿Por qué?** Los `set` en Python no garantizan un orden de iteración determinista basado en el orden de inserción (en CPython 3.7+ sí lo hacen los `dict`, pero no los `set`). Esto es relevante porque:

1. La práctica exige **reproducibilidad** ("utilice siempre la misma semilla").
2. El orden de los brazos afecta a cuál es el brazo óptimo (`optimal_arm`), y por tanto a las métricas de selección óptima.
3. Con una lista, el brazo i-ésimo generado siempre será el brazo i-ésimo del bandido.

### 5.2 Parámetros `p_min` y `p_max` configurables

Se añadieron parámetros opcionales para controlar el rango de probabilidades generadas:

```python
def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
```

Esto permite:
- Experimentar con brazos con probabilidades más cercanas (ej: `p_min=0.4, p_max=0.6`) para estudiar cómo los algoritmos discriminan entre opciones similares.
- Experimentar con brazos con probabilidades muy dispersas (ej: `p_min=0.05, p_max=0.95`) para estudiar la velocidad de convergencia.

### 5.3 Implementación de Bernoulli a través de Binomial

Se usa `np.random.binomial(1, p)` en lugar de alternativas como:
- `np.random.choice([0, 1], p=[1-p, p])` — más lento y menos elegante.
- `1 if np.random.random() < p else 0` — correcto pero no explicita la relación teórica.

La implementación elegida **refleja explícitamente** que $\text{Bernoulli}(p) = B(1, p)$, lo cual es coherente con la teoría y facilita la comprensión del código.

### 5.4 Redondeo a 3 decimales

```python
p = round(np.random.uniform(p_min, p_max), 3)
```

Se redondea para:
1. **Legibilidad**: un brazo con `p=0.437` es más interpretable que `p=0.4371829...`.
2. **Unicidad efectiva**: evita que dos valores extremadamente cercanos (ej: 0.43718 y 0.43719) se consideren diferentes cuando en la práctica son indistinguibles.
3. **Consistencia**: sigue la misma convención que `ArmNormal` (que redondea μ a 2 decimales).

### 5.5 Validación con asserts

Ambas clases validan sus parámetros mediante `assert`:

```python
assert 0.0 <= p <= 1.0, "..."    # Bernoulli y Binomial
assert n > 0, "..."               # Solo Binomial
assert k > 0, "..."               # En generate_arms
```

Esto detecta errores de programación de forma temprana (fail-fast) durante el desarrollo y la experimentación.

---

## 6. Ejemplo de ejecución

Se ejecuta el siguiente código con semilla `seed=42` para generar 5 brazos de cada tipo:

```python
import numpy as np
from arms import ArmBernoulli, ArmBinomial, Bandit

np.random.seed(42)

# Generar 5 brazos de Bernoulli
brazos_bernoulli = ArmBernoulli.generate_arms(5)
bandit_bernoulli = Bandit(arms=brazos_bernoulli)

# Generar 5 brazos de Binomial (n=100)
brazos_binomial = ArmBinomial.generate_arms(5)
bandit_binomial = Bandit(arms=brazos_binomial)
```

### Resultado: Brazos Bernoulli

```
=== Bernoulli ===
  ArmBernoulli(p=0.400) -> E[X]=0.400
  ArmBernoulli(p=0.861) -> E[X]=0.861
  ArmBernoulli(p=0.686) -> E[X]=0.686
  ArmBernoulli(p=0.579) -> E[X]=0.579
  ArmBernoulli(p=0.225) -> E[X]=0.225

Bandit Bernoulli: optimal=1, E[opt]=0.861
```

**Interpretación**:
- Se generaron 5 brazos con probabilidades de éxito: 0.400, 0.861, 0.686, 0.579 y 0.225.
- El brazo óptimo** es el brazo 1 (índice desde 0) con p=0.861.
- Esto significa que un algoritmo perfecto elegiría siempre este brazo, obteniendo ~86.1% de recompensas positivas (clics, éxitos, etc.).
- El reto para los algoritmos ε-greedy, UCB1 y Softmax será descubrir que el brazo 1 es el mejor sin conocer previamente las probabilidades.
- Nótese que las recompensas de un brazo Bernoulli son siempre 0 o 1, lo cual genera alta varianza y dificulta la estimación rápida del brazo óptimo.

### Resultado: Brazos Binomial (n=100)

```
=== Binomial (n=100) ===
  ArmBinomial(n=100, p=0.225) -> E[X]=22.5
  ArmBinomial(n=100, p=0.146) -> E[X]=14.6
  ArmBinomial(n=100, p=0.793) -> E[X]=79.3
  ArmBinomial(n=100, p=0.581) -> E[X]=58.1
  ArmBinomial(n=100, p=0.666) -> E[X]=66.6

Bandit Binomial: optimal=2, E[opt]=79.3
```

**Interpretación**:
- Se generaron 5 brazos binomiales, todos con n=100 ensayos pero con diferentes probabilidades.
- El brazo óptimo** es el brazo 2 (índice desde 0) con B(100, 0.793), cuyo valor esperado es 79.3 éxitos de cada 100.
- En el contexto del ejemplo de la práctica: la promoción del brazo 2 logra que ~79 de cada 100 usuarios la aprovechen, siendo la más efectiva.
- Las recompensas binomiales están en el rango [0, 100], con valores esperados entre 14.6 y 79.3. Esta mayor separación numérica entre brazos (comparada con Bernoulli donde todo está en [0, 1]) facilita que los algoritmos identifiquen más rápidamente el brazo óptimo.
- Sin embargo, la varianza también es mayor: $\text{Var} = 100 \times 0.793 \times 0.207 \approx 16.4$, con $\sigma \approx 4.05$, lo que introduce ruido en las estimaciones.

### Comparación entre ambos tipos

| Aspecto | Bernoulli | Binomial (n=100) |
|---|---|---|
| **Rango recompensa** | {0, 1} | {0, ..., 100} |
| **E[X] del óptimo** | 0.861 | 79.3 |
| **Varianza máxima** | p(1-p) ≤ 0.25 | np(1-p) ≤ 25 |
| **Dificultad para el agente** | Alta (señal débil) | Media (señal más fuerte) |
| **Velocidad de convergencia esperada** | Más lenta | Más rápida |

La principal diferencia práctica es que en Bernoulli cada muestra da muy poca información (solo 0 o 1), mientras que en Binomial con n=100, cada muestra da un valor más informativo que permite estimar mejor la calidad del brazo. Esto se traduce en que los algoritmos tienden a converger más rápido con brazos binomiales que con brazos de Bernoulli.