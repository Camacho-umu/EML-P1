# Documentación: Tarea 2 — Completar el estudio de la familia ε-greedy

## Índice

1. [Resumen de cambios](#1-resumen-de-cambios)
2. [Corrección del caso ε=0](#2-corrección-del-caso-ε0)
3. [Ampliación de `run_experiment`](#3-ampliación-de-run_experiment)
4. [Gráfica: Selección del brazo óptimo (`plot_optimal_selections`)](#4-gráfica-selección-del-brazo-óptimo)
5. [Gráfica: Regret acumulado (`plot_regret`)](#5-gráfica-regret-acumulado)
6. [Gráfica: Estadísticas por brazo (`plot_arm_statistics`)](#6-gráfica-estadísticas-por-brazo)
7. [Mejoras complementarias](#7-mejoras-complementarias)
8. [Resultados de los tests](#8-resultados-de-los-tests)

---

## 1. Resumen de cambios

| Archivo | Cambio |
|---|---|
| `algorithms/epsilon_greedy.py` | Corrección del caso ε=0: fase de inicialización para probar todos los brazos |
| `main.py` | `run_experiment` devuelve 4 valores: rewards, optimal_selections, regret_accumulated, arm_stats |
| `main.py` | `main()` invoca las 4 gráficas |
| `plotting/plotting.py` | Implementación de `plot_optimal_selections`, `plot_regret`, `plot_arm_statistics` |
| `plotting/plotting.py` | Mejora de `get_algorithm_label` para soportar futuros algoritmos |
| `plotting/__init__.py` | Exportación de las 4 funciones de plotting |

---

## 2. Corrección del caso ε=0

### El problema


Con la implementación original, cuando ε=0:
1. Nunca se entra en la rama de exploración (`if np.random.random() < 0` es siempre `False`).
2. Se ejecuta siempre `np.argmax(self.values)`.
3. Como `self.values` empieza todo a 0, `argmax` devuelve siempre el índice 0 (el primer brazo).
4. El algoritmo se queda atrapado en el brazo 0 sin probar los demás, independientemente de su calidad.

### La solución

Se añade una fase de inicialización al principio de `select_arm()`:

```python
def select_arm(self) -> int:
    # Fase de inicialización: probar brazos que nunca se han seleccionado
    unexplored = np.where(self.counts == 0)[0]
    if len(unexplored) > 0:
        return np.random.choice(unexplored)

    # ... resto del algoritmo epsilon-greedy
```

### Por qué esta solución

1. **Garantiza exploración mínima**: Cada brazo se prueba al menos una vez antes de que el algoritmo empiece a explotar. Esto es fundamental para que ε=0 (greedy puro) no se quede atrapado.
2. **No altera el comportamiento para ε>0**: Con ε>0, los primeros k pasos igualmente explorarían los brazos no visitados.  La fase de inicialización simplemente asegura que esto ocurra de forma explícita.
3. **Es estándar en la literatura**: Sutton & Barto (2018) asumen esta inicialización en su tratamiento del algorithm greedy.
4. **Selección aleatoria entre los no explorados**: Se usa `np.random.choice(unexplored)` para que no haya sesgo hacia ningún brazo particular durante la inicialización.

### Verificación

```
=== TEST 1: Fix epsilon=0 ===
  Primeras 5 selecciones con eps=0: [2, 0, 4, 3, 1]
  OK: todos los 5 brazos fueron probados al menos una vez
```

---

## 3. Ampliación de `run_experiment`

### Estado original

La función `run_experiment` original devolvía solo 2 valores:
- `rewards`: recompensas promedio (algoritmos × pasos)
- `optimal_selections`: porcentaje de selecciones óptimas

Además, el notebook tenía 2 TODOs sin completar:
1. `#TODO: modificar optimal_selections cuando el brazo elegido se corresponda con el brazo óptimo`
2. `# TODO: calcular el porcentaje de selecciones óptimas`

### Cambios realizados

La función ahora devuelve 4 valores:

```python
return rewards, optimal_selections, regret_accumulated, arm_stats
```

#### a) Selecciones óptimas 

Dentro del bucle de pasos, cuando el brazo elegido coincide con el óptimo:

```python
if chosen_arm == optimal_arm:
    optimal_selections[idx, step] += 1
```

Y al final, se convierte a porcentaje:

```python
optimal_selections = (optimal_selections / runs) * 100
```

Se usa el porcentaje y no la proporción porque es más intuitivo para la visualización. Un 92% de selección óptima es más legible que 0.92.

#### b) Regret acumulado 

El regret mide la pérdida acumulada por no elegir siempre el brazo óptimo. Se define como:

$$R(T) = \sum_{t=1}^{T} \left( \mu^* - \mu_{a_t} \right)$$

donde $\mu^* = \max_i \mu_i$ es el valor esperado del brazo óptimo y $\mu_{a_t}$ es el valor esperado del brazo elegido en el paso $t$.

Implementación:

```python
q_star = bandit.get_expected_value(optimal_arm)  # μ*

# En cada paso:
instant_regret = q_star - current_bandit.get_expected_value(chosen_arm)  # μ* - μ_{a_t}
cumulative_regret[idx] += instant_regret
regret_accumulated[idx, step] += cumulative_regret[idx]
```

**Decisión importante**: Se usa el regret basado en valores esperados (no en recompensas observadas). Esto es porque el regret es una medida teórica de la calidad de la política, no de la suerte en una ejecución particular. Usar la recompensa observada introduciría ruido innecesario.

El regret cumplido se promedia sobre todas las ejecuciones: `regret_accumulated /= runs`.

#### c) Estadísticas por brazo 

Para la gráfica `plot_arm_statistics`, se acumulan:
- **Conteos por brazo**: cuántas veces fue seleccionado cada brazo por cada algoritmo.
- **Recompensas medias por brazo**: la recompensa media estimada para cada brazo.

```python
arm_total_counts[idx] += algo.counts
arm_total_rewards[idx] += algo.values * algo.counts
```

Se promedian sobre todas las ejecuciones y se empaquetan en una lista de diccionarios:

```python
arm_stats.append({
    'counts': avg_counts,
    'avg_rewards': avg_rewards,
    'optimal_arm': optimal_arm
})
```

### Verificación

```
=== TEST 2: run_experiment ===
  rewards.shape = (2, 100)
  optimal_sel range: [6.0%, 92.0%]
  regret_acc is monotonically non-decreasing: True
  regret_acc final: eps=0 -> 47.04, eps=0.1 -> 81.35
  arm_stats[0] counts sum: 100 (expected ~100)
```

El regret es monótonamente no decreciente (correcto, ya que el regret instantáneo siempre es ≥ 0).

---

## 4. Gráfica: Selección del brazo óptimo


### Implementación

```python
def plot_optimal_selections(steps, optimal_selections, algorithms):
```

Genera una gráfica de líneas donde:
- **Eje X**: pasos de tiempo (1 a T).
- **Eje Y**: porcentaje de selección del brazo óptimo (0% a 100%).
- **Cada línea**: un algoritmo ε-greedy con diferente ε.


La recompensa promedio muestra cuánto gana el agente, pero no distingue si el agente está eligiendo el brazo correcto o simplemente tiene suerte. La selección óptima muestra directamente si el algoritmo está aprendiendo a identificar el mejor brazo.

Ejemplo: un algoritmo con ε=0.1 podría tener menor recompensa promedio que ε=0.01 a largo plazo (por el 10% de exploración), pero mayor porcentaje de selección óptima (converge más rápido a saber cuál es el mejor).

### Diseño

- `plt.ylim(0, 100)`: fija el rango para que sea comparable entre experimentos.
- Se usa seaborn con estilo `whitegrid` para consistencia visual con las demás gráficas.

---

## 5. Gráfica: Regret acumulado

### Implementación

```python
def plot_regret(steps, regret_accumulated, algorithms, show_log_bound=True):
```

Genera una gráfica de líneas del regret acumulado, con una referencia logarítmica opcional.

### La cota teórica de Lai y Robbins (1985)

Un resultado fundamental en la teoría de bandidos dice que para cualquier algoritmo consistente:

$$R(T) \geq C \cdot \ln(T)$$

donde $C$ es una constante que depende de las distribuciones de los brazos (específicamente de la divergencia KL entre cada brazo y el óptimo).

La implementación muestra esta cota como línea discontinua gris:

```python
if show_log_bound:
    max_regret = np.max(regret_accumulated[:, -1])
    cte = max_regret / (2 * np.log(steps))  # Escalar para visualización
    log_bound = cte * np.log(t_range)
    plt.plot(range(steps), log_bound, '--', color='gray', alpha=0.7,
             label='Cte·ln(T) (referencia)')
```

### Importancia de la gráfica

1. **Complementa la recompensa promedio**: La recompensa muestra la ganancia, el regret muestra la pérdida acumulada por decisiones subóptimas.
2. **Permite comparar con la cota teórica**: Un buen algoritmo tiene regret que crece como $O(\ln T)$, y la referencia visual permite evaluarlo.
3. **Es la métrica estándar en la literatura**: El regret es la medida formal para comparar algoritmos de bandidos (Sutton & Barto, 2018; Auer et al., 2002).
4. **Discrimina mejor**: Un algoritmo con ε alto tiene regret que crece linealmente (regret constante por paso), mientras que ε-greedy con buen ε tiene regret que crece sublinealmente.

### Ajuste de la constante

La constante $C$ se ajusta visualmente como `max_regret / (2 * ln(T))` para que la curva de referencia sea visible y proporcional al rango del gráfico, sin calcular la divergencia KL exacta (que requiere conocer las distribuciones verdaderas).

---

## 6. Gráfica: Estadísticas por brazo

### Implementación

```python
def plot_arm_statistics(arm_stats, algorithms):
```

Genera un panel de subgráficas (una por algoritmo) donde cada subgráfica es un histograma de barras:

- **Eje X**: cada brazo, con etiqueta que incluye:
  - Número de brazo
  - Número medio de veces seleccionado (n=...)
  - Indicador ★ si es el brazo óptimo
- **Eje Y**: recompensa promedio estimada del brazo.
- **Colores**: verde para el brazo óptimo, azul para el resto.
- **Valores sobre las barras**: recompensa media numérica.

### Utilidad de la gráfica

1. **Visibilidad de la distribución de selecciones**: Muestra qué brazos fueron más seleccionados por cada algoritmo.
2. **Diagnóstico de exploración**: Si un algoritmo con ε-greedy selecciona uniformemente todos los brazos, está explorando demasiado. Si concentra en el óptimo, está explotando bien.
3. **Comparación entre algoritmos**: Al mostrar un panel lado a lado, se ve cómo ε=0 concentra todo en un brazo (posiblemente incorrecto) mientras ε=0.1 distribuye más pero concentra en el óptimo.

### Decisiones de diseño

- **Panel horizontal** (`subplots(1, n_algos)`): permite comparación directa lado a lado.
- **`sharey=True`**: mismo eje Y para todos los subplots, facilitando la comparación visual.
- **Color verde para el óptimo**: permite identificar inmediatamente si el algoritmo "acierta".

---

## 7. Mejoras complementarias

### `get_algorithm_label` mejorado

El original lanzaba un `ValueError` para cualquier algoritmo que no fuera `EpsilonGreedy`. Ahora tiene un fallback genérico:

```python
def get_algorithm_label(algo: Algorithm) -> str:
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (ε={algo.epsilon})"
    else:
        label += f" (k={algo.k})"
    return label
```

De esta forma preparamos el sistema para la Tarea 3 (UCB1, Softmax) sin necesidad de modificar esta función cada vez.

### Actualización de `plotting/__init__.py`

Se exportan las 4 funciones:

```python
from .plotting import plot_average_rewards, plot_optimal_selections, plot_regret, plot_arm_statistics
__all__ = ['plot_average_rewards', 'plot_optimal_selections', 'plot_regret', 'plot_arm_statistics']
```

### Actualización de `main.py`

- Se importan las nuevas funciones de plotting.
- `main()` invoca las 4 gráficas:

```python
rewards, optimal_selections, regret_accumulated, arm_stats = run_experiment(...)
plot_average_rewards(steps, rewards, algorithms)
plot_optimal_selections(steps, optimal_selections, algorithms)
plot_regret(steps, regret_accumulated, algorithms)
plot_arm_statistics(arm_stats, algorithms)
```

---

## 8. Resultados de los tests

Se creó un script `test_task2.py` que verifica 5 aspectos:

```
=== TEST 1: Fix epsilon=0 ===
  Primeras 5 selecciones con eps=0: [2, 0, 4, 3, 1]
  OK: todos los 5 brazos fueron probados al menos una vez

=== TEST 2: run_experiment returns (rewards, optimal_sel, regret, arm_stats) ===
  rewards.shape = (2, 100) (expected: (2, 100))
  optimal_sel.shape = (2, 100)
  regret_acc.shape = (2, 100)
  len(arm_stats) = 2
  optimal_sel range: [6.0%, 92.0%]
  regret_acc is monotonically non-decreasing: True
  regret_acc final: eps=0 -> 47.04, eps=0.1 -> 81.35
  arm_stats[0] keys: ['counts', 'avg_rewards', 'optimal_arm']
  arm_stats[0] counts sum: 100 (expected ~100)

=== TEST 3: plotting functions importable ===
  plot_average_rewards: True
  plot_optimal_selections: True
  plot_regret: True
  plot_arm_statistics: True

=== TEST 4: get_algorithm_label ===
  label eps=0: "EpsilonGreedy (ε=0)"
  label eps=0.1: "EpsilonGreedy (ε=0.1)"

=== TEST 5: Bernoulli & Binomial arms with experiment ===
  Bernoulli: rewards shape=(1, 50), regret final=7.31
  Binomial: rewards shape=(1, 50), regret final=293.55

=============================
=== ALL TESTS PASSED ✓ ===
=============================
```

### Interpretación de resultados clave

| Resultado | Significado |
|---|---|
| `regret_acc is monotonically non-decreasing: True` | El regret acumulado nunca decrece (correcto, ya que el regret instantáneo $\mu^* - \mu_{a_t} \geq 0$) |
| `optimal_sel range: [6.0%, 92.0%]` | Con 100 pasos y 50 runs, el porcentaje va desde 6% (al inicio, exploración) hasta 92% (convergencia) |
| `arm_stats counts sum: 100` | Cada algoritmo selecciona exactamente 100 brazos en 100 pasos (consistencia) |
| `Bernoulli regret: 7.31` | Regret bajo porque las recompensas de Bernoulli están en [0,1] |
| `Binomial regret: 293.55` | Regret mayor porque con n=100 las recompensas esperadas están en [0,100], con diferencias mayores entre brazos |

---

## Archivos modificados

```
eml_k_bandit-main/
├── algorithms/
│   ├── epsilon_greedy.py     ← Corrección fase inicialización (ε=0)
│   └── __init__.py           (sin cambios)
├── plotting/
│   ├── plotting.py           ← 3 funciones nuevas + mejora get_algorithm_label
│   └── __init__.py           ← Exporta 4 funciones
├── main.py                   ← run_experiment devuelve 4 valores, main() usa 4 gráficas
└── test_task2.py             ← Script de verificación (nuevo)
```
