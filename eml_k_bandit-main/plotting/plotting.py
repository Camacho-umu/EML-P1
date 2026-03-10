"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos
en el problema del bandido de k-brazos.

Gráficas implementadas:
    - plot_average_rewards: Recompensa promedio vs pasos de tiempo.
    - plot_optimal_selections: Porcentaje de selección del brazo óptimo vs pasos.
    - plot_regret: Regret acumulado vs pasos de tiempo.
    - plot_arm_statistics: Estadísticas por brazo para cada algoritmo.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List, Dict

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, UCB1, Softmax


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
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


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    Muestra cómo evoluciona la probabilidad de que cada algoritmo seleccione el brazo
    con mayor valor esperado a lo largo de los pasos de tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección Brazo Óptimo', fontsize=14)
    plt.title('Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo', fontsize=16)
    plt.ylim(0, 100)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm],
                show_log_bound: bool = True):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.

    El regret acumulado mide la pérdida total por no haber elegido siempre el brazo óptimo:
        R(T) = T·μ* - Σ_{t=1}^{T} μ_{a_t}
    donde μ* es el valor esperado del brazo óptimo y μ_{a_t} es el valor esperado
    del brazo elegido en el paso t.

    Opcionalmente muestra la cota teórica logarítmica Cte·ln(T) de Lai y Robbins (1985).

    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param show_log_bound: Si True, muestra una referencia logarítmica Cte·ln(T).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    # Cota teórica logarítmica de referencia: Lai y Robbins (1985)
    # R(T) ≥ Cte · ln(T), ajustamos la constante visualmente
    if show_log_bound:
        t_range = np.arange(1, steps + 1)
        # Ajustar la constante para que sea visible en la escala del gráfico
        max_regret = np.max(regret_accumulated[:, -1])
        if max_regret > 0:
            cte = max_regret / (2 * np.log(steps))  # Escalar para visualización
            log_bound = cte * np.log(t_range)
            plt.plot(range(steps), log_bound, '--', color='gray', alpha=0.7,
                     label='Cota teórica', linewidth=1.5)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado', fontsize=14)
    plt.title('Regret Acumulado vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats: List[Dict], algorithms: List[Algorithm]):
    """
    Genera gráficas de estadísticas por brazo para cada algoritmo.

    Para cada algoritmo se muestra un histograma donde:
    - Eje X: cada brazo (con etiqueta indicando nº de selecciones y si es óptimo).
    - Eje Y: recompensa promedio obtenida en ese brazo.

    :param arm_stats: Lista de diccionarios con estadísticas por brazo para cada algoritmo.
        Cada diccionario contiene:
            - 'counts': np.ndarray con el promedio de selecciones por brazo.
            - 'avg_rewards': np.ndarray con la recompensa media estimada por brazo.
            - 'optimal_arm': int, índice del brazo óptimo.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

    n_algos = len(algorithms)
    fig, axes = plt.subplots(1, n_algos, figsize=(6 * n_algos, 6), sharey=True)

    # Si solo hay un algoritmo, axes no es una lista
    if n_algos == 1:
        axes = [axes]

    for idx, (stats, algo) in enumerate(zip(arm_stats, algorithms)):
        ax = axes[idx]
        k = len(stats['counts'])
        optimal_arm = stats['optimal_arm']
        counts = stats['counts']
        avg_rewards = stats['avg_rewards']

        # Colores: verde para el brazo óptimo, azul para el resto
        colors = ['#2ecc71' if i == optimal_arm else '#3498db' for i in range(k)]

        bars = ax.bar(range(k), avg_rewards, color=colors, edgecolor='white', linewidth=0.5)

        # Etiquetas del eje X: número de brazo + conteo de selecciones
        x_labels = []
        for i in range(k):
            label = f"Brazo {i + 1}\n(n={counts[i]:.0f})"
            if i == optimal_arm:
                label += "\n★ Óptimo"
            x_labels.append(label)

        ax.set_xticks(range(k))
        ax.set_xticklabels(x_labels, fontsize=8, rotation=0)
        ax.set_ylabel('Recompensa Promedio' if idx == 0 else '', fontsize=12)
        ax.set_title(get_algorithm_label(algo), fontsize=13)

        # Añadir valores sobre las barras
        for bar, reward in zip(bars, avg_rewards):
            if reward > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{reward:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('Estadísticas por Brazo y Algoritmo', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

