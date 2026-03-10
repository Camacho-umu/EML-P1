# Documentación: Tarea 3 — Implementación de UCB1 y Softmax

## Índice

1. [Resumen de cambios](#1-resumen-de-cambios)
2. [Algoritmo UCB1 (Upper Confidence Bound)](#2-algoritmo-ucb1-upper-confidence-bound)
3. [Algoritmo Softmax (Boltzmann Exploration)](#3-algoritmo-softmax-boltzmann-exploration)
4. [Comparación teórica entre los tres algoritmos](#4-comparación-teórica-entre-los-tres-algoritmos)
5. [Cambios en archivos auxiliares](#5-cambios-en-archivos-auxiliares)
6. [Resultados de los tests](#6-resultados-de-los-tests)

---

## 1. Resumen de cambios

| Archivo | Cambio |
|---|---|
| `algorithms/ucb1.py` | **Nuevo** — Implementación completa del algoritmo UCB1 |
| `algorithms/softmax.py` | **Nuevo** — Implementación completa del algoritmo Softmax |
| `algorithms/__init__.py` | Exportación de `UCB1` y `Softmax` |
| `plotting/plotting.py` | `get_algorithm_label` actualizado para generar etiquetas con los parámetros de UCB1 (c) y Softmax (τ) |

---

## 2. Algoritmo UCB1 (Upper Confidence Bound)

### Fundamento teórico

UCB1 aborda el dilema exploración-explotación de forma determinista (sin aleatoriedad en la selección). La idea clave es el principio de optimismo ante la incertidumbre: se le da a cada brazo el beneficio de la duda, asignándole un bono que refleja cuánto podría valer realmente.

Fórmula de selección (Auer, Cesa-Bianchi & Fischer, 2002):

$$A_t = \arg\max_a \left[ Q_t(a) + c \cdot \sqrt{\frac{\ln t}{N_t(a)}} \right]$$

donde:
- $Q_t(a)$: valor estimado del brazo $a$ en el paso $t$ (media empírica)
- $N_t(a)$: número de veces que se ha seleccionado el brazo $a$
- $t$: número total de pasos hasta el momento
- $c$: parámetro de exploración ($c = \sqrt{2}$ es el valor teórico)

El término $c \cdot \sqrt{\frac{\ln t}{N_t(a)}}$ es el bono de confianza:
- Crece con $t$ (los brazos poco explorados se vuelven más atractivos con el tiempo)
- Decrece con $N_t(a)$ (cuantas más veces probamos un brazo, menos incertidumbre tenemos)

### Propiedad clave: Regret logarítmico

A diferencia de ε-greedy, UCB1 consigue regret logarítmico:

$$R(T) = O\left(\sum_{a: \mu_a < \mu^*} \frac{\ln T}{\Delta_a}\right)$$

donde $\Delta_a = \mu^* - \mu_a$ es la diferencia entre el brazo óptimo y el brazo $a$. Esto se acerca a la cota inferior teórica de Lai y Robbins (1985), lo que lo hace asintóticamente óptimo.

### Implementación

```python
class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 2**0.5):
        super().__init__(k)
        self.c = c
        self.total_counts: int = 0

    def select_arm(self) -> int:
        # Fase de inicialización: probar brazos no explorados
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)

        # Calcular UCB para cada brazo
        t = self.total_counts
        ucb_values = self.values + self.c * np.sqrt(np.log(t) / self.counts)
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float):
        self.total_counts += 1
        super().update(chosen_arm, reward)

    def reset(self):
        super().reset()
        self.total_counts = 0
```

### Decisiones de diseño

| Decisión | Justificación |
|---|---|
| **`total_counts` como atributo** | Más eficiente que recalcular `sum(counts)` en cada paso |
| **Fase de inicialización** | Necesaria porque `sqrt(ln(t)/0)` no está definido; garantiza que todos los brazos se prueban |
| **`c = sqrt(2)` por defecto** | Valor teórico derivado de la desigualdad de Hoeffding para recompensas en [0, 1]. Funciona bien en la práctica también para normales |
| **`argmax` determinista** | Consistente con la filosofía de UCB1: no hay aleatoriedad en la selección |
| **Override de `update` y `reset`** | Para mantener `total_counts` sincronizado |

### Parámetro c: exploración vs explotación

- **c alto** (e.g., c=5): explora mucho → más lento pero más robusto
- **c bajo** (e.g., c=0.5): explora poco → más rápido pero puede quedarse en un brazo subóptimo
- **c = √2 ≈ 1.41**: valor teórico óptimo para recompensas acotadas en [0, 1]

---

## 3. Algoritmo Softmax (Boltzmann Exploration)

### Fundamento teórico

Softmax selecciona brazos con probabilidad proporcional a la exponencial de sus valores estimados, escalados por un parámetro de temperatura τ:

$$P(A_t = a) = \frac{e^{Q_t(a) / \tau}}{\sum_{b=1}^{k} e^{Q_t(b) / \tau}}$$

La temperatura controla el equilibrio exploración-explotación:
- $\tau \to \infty$: distribución uniforme (exploración pura, como selección al azar)
- $\tau \to 0^+$: distribución degenerada en $\arg\max Q$ (explotación pura, como greedy)

### Ventaja sobre ε-greedy

La diferencia fundamental es cómo se explora:
- **ε-greedy**: explora uniformemente — un brazo con μ=0.1 tiene la misma probabilidad de ser explorado que uno con μ=0.9
- **Softmax**: explora proporcionalmente — brazos con mejores estimaciones tienen más probabilidad de ser seleccionados

Esto es especialmente ventajoso cuando hay brazos claramente subóptimos: Softmax los evita naturalmente, mientras que ε-greedy los prueba con la misma frecuencia que el resto.

### Implementación

```python
class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 1.0):
        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        # Fase de inicialización: probar brazos no explorados
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)

        # Calcular probabilidades softmax con truco de estabilidad numérica
        scaled_values = self.values / self.tau
        scaled_values -= np.max(scaled_values)  # Truco de estabilidad
        exp_values = np.exp(scaled_values)
        probabilities = exp_values / np.sum(exp_values)

        return int(np.random.choice(self.k, p=probabilities))
```

### Truco de estabilidad numérica

Un detalle crítico es restar `max(scaled_values)` antes de calcular la exponencial:

```python
scaled_values -= np.max(scaled_values)
```

**¿Por qué?** Si τ es muy pequeña y los valores Q son grandes, `exp(Q/τ)` puede causar overflow (valores mayores que lo que puede representar un float64). Al restar el máximo:
- El mayor valor se convierte en `exp(0) = 1`
- Los demás son `exp(negativo) ∈ (0, 1)`
- Las probabilidades resultantes son idénticas (la constante se cancela en numerador y denominador)

Matemáticamente: $\frac{e^{x_i - c}}{\sum_j e^{x_j - c}} = \frac{e^{x_i} \cdot e^{-c}}{\sum_j e^{x_j} \cdot e^{-c}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$

### Decisiones de diseño

| Decisión | Justificación |
|---|---|
| **Truco max-substraction** | Evita overflow numérico sin cambiar las probabilidades |
| **`np.random.choice` con pesos** | Muestreo eficiente según la distribución de Boltzmann |
| **τ = 1.0 por defecto** | Valor intermedio que equilibra exploración/explotación |
| **Fase de inicialización** | Igual que UCB1 y ε-greedy, para consistencia y para evitar que todos los Q=0 den distribución uniforme trivial |

### Parámetro τ: temperatura

- **τ alto** (e.g., τ=10): casi uniforme → explora mucho, aprende lento
- **τ medio** (e.g., τ=1): equilibrado → explora proporcionalmente a las estimaciones
- **τ bajo** (e.g., τ=0.1): casi greedy → explota agresivamente

---

## 4. Comparación teórica entre los tres algoritmos

| Propiedad | ε-greedy | UCB1 | Softmax |
|---|---|---|---|
| **Tipo de exploración** | Aleatoria uniforme (ε del tiempo) | Determinista (bono de confianza) | Aleatoria ponderada (Boltzmann) |
| **Parámetro** | ε ∈ [0, 1] | c > 0 | τ > 0 |
| **Regret teórico** | O(T) lineal | O(ln T) logarítmico | Depende de τ |
| **Exploración decrece?** | No (ε fijo) | Sí (bono decrece) | No (τ fijo) |
| **Sensible a escala?** | No | Parcialmente (c) | Sí (τ depende de la escala de Q) |
| **Calidad de exploración** | Uniforme (explora brazos malos igual) | Dirigida a brazos inciertos | Dirigida a brazos prometedores |

### ¿Cuándo usar cada uno?

- **ε-greedy**: Cuando la simplicidad es prioritaria y el horizonte es corto. Buena línea base.
- **UCB1**: Cuando se busca la mayor eficiencia teórica y no importa la aleación. Ideal para horizontes largos.
- **Softmax**: Cuando se quiere exploración proporcional a la calidad estimada. Útil cuando el agente no puede permitirse elegir brazos claramente malos.

---

## 5. Cambios en archivos auxiliares

### `algorithms/__init__.py`

Añadidas las importaciones y exportaciones:

```python
from .ucb1 import UCB1
from .softmax import Softmax

__all__ = ['Algorithm', 'EpsilonGreedy', 'UCB1', 'Softmax']
```

### `plotting/plotting.py` — `get_algorithm_label`

Actualizada para generar etiquetas descriptivas para los nuevos algoritmos:

```python
from algorithms import Algorithm, EpsilonGreedy, UCB1, Softmax

def get_algorithm_label(algo: Algorithm) -> str:
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (ε={algo.epsilon})"
    elif isinstance(algo, UCB1):
        label += f" (c={algo.c:.2f})"
    elif isinstance(algo, Softmax):
        label += f" (τ={algo.tau})"
    else:
        label += f" (k={algo.k})"
    return label
```

---

## 6. Resultados de los tests

Se creó `test_task3.py` con 7 tests que cubren los aspectos clave:

```
============================================================
Tests para UCB1 y Softmax
============================================================
✓ test_ucb1_basic: OK
✓ test_ucb1_exploits_best: OK (brazo óptimo 2 seleccionado 1991 veces)
✓ test_ucb1_reset: OK
✓ test_softmax_basic: OK
✓ test_softmax_exploits_best: OK (brazo óptimo 2 seleccionado 1996 veces)
✓ test_softmax_high_tau_explores: OK (proporciones: [0.196 0.202 0.215 0.196 0.191])
✓ test_run_experiment_with_all: OK
  Recompensa final (paso 200): ε-greedy=9.24, UCB1=9.70, Softmax=9.51

============================================================
TODOS LOS TESTS PASARON ✓
============================================================
```

### Interpretación de los tests

| Test | Qué verifica |
|---|---|
| `test_ucb1_basic` | UCB1 prueba los k brazos en la fase de inicialización |
| `test_ucb1_exploits_best` | Tras 2000 pasos, UCB1 converge al brazo óptimo (1991/2000 selecciones) |
| `test_ucb1_reset` | `reset()` limpia counts, values y total_counts |
| `test_softmax_basic` | Softmax prueba los k brazos en la fase de inicialización |
| `test_softmax_exploits_best` | Con τ=0.1 (bajo), Softmax converge al brazo óptimo (1996/2000) |
| `test_softmax_high_tau_explores` | Con τ=100 (alto), distribución casi uniforme (~20% cada brazo) |
| `test_run_experiment_with_all` | Los 3 algoritmos funcionan juntos en `run_experiment` |

### Observación clave de los tests

- **UCB1** con c=√2 selecciona el brazo óptimo 1991/2000 veces → la exploración UCB se reduce naturalmente
- **Softmax** con τ=0.1 selecciona el óptimo 1996/2000 → temperatura baja es casi greedy
- **Softmax** con τ=100 da proporciones ~[0.196, 0.202, 0.215, 0.196, 0.191] → casi uniforme, confirma que τ alto → exploración pura
- En `run_experiment` con 200 pasos y 50 ejecuciones: UCB1 (9.70) > Softmax (9.51) > ε-greedy (9.24) en recompensa final
