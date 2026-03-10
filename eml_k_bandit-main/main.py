"""
Module: main.py
Description: Main script to run comparative experiments between different algorithms.
El experimento compara el rendimiento de algoritmos epsilon-greedy en un problema de k-armed bandit.    
Se generan gráficas de recompensas promedio y selecciones óptimas para cada algoritmo.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np

from algorithms import Algorithm, EpsilonGreedy
from arms import ArmNormal, Bandit
from plotting import plot_average_rewards, plot_optimal_selections, plot_regret, plot_arm_statistics


def run_experiment(bandit: Bandit, algorithms: List[Algorithm], steps: int, runs: int):
    """
    Ejecuta experimentos comparativos entre diferentes algoritmos.

    :param bandit: Instancia de Bandit configurada para el experimento.
    :param algorithms: Lista de instancias de algoritmos a comparar.
    :param steps: Número de pasos de tiempo por ejecución.
    :param runs: Número de ejecuciones independientes.
    :return: Tuple de cuatro elementos:
        - rewards: recompensas promedio (algoritmos x pasos)
        - optimal_selections: porcentaje de selecciones óptimas (algoritmos x pasos)
        - regret_accumulated: regret acumulado promedio (algoritmos x pasos)
        - arm_stats: lista de diccionarios con estadísticas por brazo para cada algoritmo
    :rtype: Tuple of (np.ndarray, np.ndarray, np.ndarray, list)
    """

    k = bandit.k
    optimal_arm = bandit.optimal_arm
    q_star = bandit.get_expected_value(optimal_arm)  # Recompensa esperada del brazo óptimo

    # Inicializar matrices para recompensas, selecciones óptimas y regret
    rewards = np.zeros((len(algorithms), steps))
    optimal_selections = np.zeros((len(algorithms), steps))
    regret_accumulated = np.zeros((len(algorithms), steps))

    # Estadísticas por brazo: acumulamos conteos y recompensas por brazo/algoritmo
    arm_total_counts = np.zeros((len(algorithms), k))
    arm_total_rewards = np.zeros((len(algorithms), k))

    for run in range(runs):
        # Crear una nueva instancia del bandit para cada ejecución
        current_bandit = Bandit(arms=bandit.arms)

        for algo in algorithms:
            algo.reset()

        # Regret acumulado por algoritmo en esta ejecución
        cumulative_regret = np.zeros(len(algorithms))

        for step in range(steps):
            for idx, algo in enumerate(algorithms):
                chosen_arm = algo.select_arm()
                reward = current_bandit.pull_arm(chosen_arm)
                algo.update(chosen_arm, reward)

                rewards[idx, step] += reward

                # Regret instantáneo: diferencia entre la recompensa esperada óptima
                # y la recompensa esperada del brazo elegido
                instant_regret = q_star - current_bandit.get_expected_value(chosen_arm)
                cumulative_regret[idx] += instant_regret
                regret_accumulated[idx, step] += cumulative_regret[idx]

                if chosen_arm == optimal_arm:
                    optimal_selections[idx, step] += 1

        # Acumular estadísticas por brazo de la última ejecución
        for idx, algo in enumerate(algorithms):
            arm_total_counts[idx] += algo.counts
            arm_total_rewards[idx] += algo.values * algo.counts  # recompensa total = media * conteo

    # Promediar sobre todas las ejecuciones
    rewards /= runs
    optimal_selections = (optimal_selections / runs) * 100
    regret_accumulated /= runs

    # Construir estadísticas por brazo promediadas
    arm_stats = []
    for idx in range(len(algorithms)):
        avg_counts = arm_total_counts[idx] / runs
        avg_rewards = np.divide(arm_total_rewards[idx], arm_total_counts[idx],
                                out=np.zeros(k), where=arm_total_counts[idx] != 0)
        arm_stats.append({
            'counts': avg_counts,
            'avg_rewards': avg_rewards,
            'optimal_arm': optimal_arm
        })

    return rewards, optimal_selections, regret_accumulated, arm_stats




def main():
    """
    Main function to set up and execute comparative experiments.
    """

    seed = 42
    np.random.seed(seed)

    k = 10 # Número de brazos
    steps = 1000  # Número de pasos
    runs = 500  # Número de ejecuciones

    bandit = Bandit(arms=ArmNormal.generate_arms(k))  # Bandit(arms=ArmBinomial.generate_arms(k))
    # bandit = Bandit(arms=ArmBernoulli.generate_arms(k))  # Bandit(arms=ArmBinomial.generate_arms(k))
    print(bandit)

    optimal_arm: int = bandit.optimal_arm
    print(f"Optimal arm: {optimal_arm + 1} with expected reward={bandit.get_expected_value(optimal_arm)}")

    algorithms = [EpsilonGreedy(k=k, epsilon=0), EpsilonGreedy(k=k, epsilon=0.01), EpsilonGreedy(k=k, epsilon=0.1)]

    # Ejecutar el experimento y obtener las recompensas promedio y selecciones óptimas
    rewards, optimal_selections, regret_accumulated, arm_stats = run_experiment(bandit, algorithms, steps, runs)

    # Generar las gráficas utilizando las funciones externas
    plot_average_rewards(steps, rewards, algorithms)
    plot_optimal_selections(steps, optimal_selections, algorithms)
    plot_regret(steps, regret_accumulated, algorithms)
    plot_arm_statistics(arm_stats, algorithms)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
