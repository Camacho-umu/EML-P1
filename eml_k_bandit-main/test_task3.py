"""
Test script for UCB1 and Softmax algorithms.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from arms import ArmNormal, Bandit
from algorithms import UCB1, Softmax, EpsilonGreedy
from main import run_experiment

def test_ucb1_basic():
    """UCB1: selecciona todos los brazos en la fase de inicialización."""
    np.random.seed(42)
    k = 5
    algo = UCB1(k=k, c=2**0.5)
    bandit = Bandit(arms=ArmNormal.generate_arms(k))
    
    selected = set()
    for _ in range(k):
        arm = algo.select_arm()
        reward = bandit.pull_arm(arm)
        algo.update(arm, reward)
        selected.add(arm)
    
    assert len(selected) == k, f"UCB1 debería probar todos los brazos: probó {len(selected)}/{k}"
    assert algo.total_counts == k, f"total_counts debería ser {k}, es {algo.total_counts}"
    print("✓ test_ucb1_basic: OK")

def test_ucb1_exploits_best():
    """UCB1: tras suficientes pasos, converge al brazo óptimo."""
    np.random.seed(42)
    k = 5
    algo = UCB1(k=k, c=2**0.5)
    bandit = Bandit(arms=ArmNormal.generate_arms(k))
    optimal = bandit.optimal_arm
    
    for _ in range(2000):
        arm = algo.select_arm()
        reward = bandit.pull_arm(arm)
        algo.update(arm, reward)
    
    # El brazo óptimo debería ser el más seleccionado
    most_selected = np.argmax(algo.counts)
    assert most_selected == optimal, f"UCB1 debería converger al brazo {optimal}, convergió al {most_selected}"
    print(f"✓ test_ucb1_exploits_best: OK (brazo óptimo {optimal+1} seleccionado {algo.counts[optimal]} veces)")

def test_ucb1_reset():
    """UCB1: reset limpia todo el estado."""
    np.random.seed(42)
    algo = UCB1(k=5, c=1.0)
    algo.counts[0] = 10
    algo.values[0] = 5.0
    algo.total_counts = 10
    algo.reset()
    
    assert np.all(algo.counts == 0), "Counts no se resetearon"
    assert np.all(algo.values == 0), "Values no se resetearon"
    assert algo.total_counts == 0, "total_counts no se reseteó"
    print("✓ test_ucb1_reset: OK")

def test_softmax_basic():
    """Softmax: selecciona todos los brazos en la fase de inicialización."""
    np.random.seed(42)
    k = 5
    algo = Softmax(k=k, tau=1.0)
    bandit = Bandit(arms=ArmNormal.generate_arms(k))
    
    selected = set()
    for _ in range(k):
        arm = algo.select_arm()
        reward = bandit.pull_arm(arm)
        algo.update(arm, reward)
        selected.add(arm)
    
    assert len(selected) == k, f"Softmax debería probar todos los brazos: probó {len(selected)}/{k}"
    print("✓ test_softmax_basic: OK")

def test_softmax_exploits_best():
    """Softmax con tau bajo: converge al brazo óptimo."""
    np.random.seed(42)
    k = 5
    algo = Softmax(k=k, tau=0.1)  # Temperatura baja → más explotación
    bandit = Bandit(arms=ArmNormal.generate_arms(k))
    optimal = bandit.optimal_arm
    
    for _ in range(2000):
        arm = algo.select_arm()
        reward = bandit.pull_arm(arm)
        algo.update(arm, reward)
    
    most_selected = np.argmax(algo.counts)
    assert most_selected == optimal, f"Softmax debería converger al brazo {optimal}, convergió al {most_selected}"
    print(f"✓ test_softmax_exploits_best: OK (brazo óptimo {optimal+1} seleccionado {algo.counts[optimal]} veces)")

def test_softmax_high_tau_explores():
    """Softmax con tau alto: distribución más uniforme (más exploración)."""
    np.random.seed(42)
    k = 5
    algo = Softmax(k=k, tau=100.0)  # Temperatura muy alta → casi uniforme
    bandit = Bandit(arms=ArmNormal.generate_arms(k))
    
    for _ in range(5000):
        arm = algo.select_arm()
        reward = bandit.pull_arm(arm)
        algo.update(arm, reward)
    
    # Con tau=100, la distribución debería ser casi uniforme
    proportions = algo.counts / algo.counts.sum()
    expected_uniform = 1.0 / k
    max_deviation = np.max(np.abs(proportions - expected_uniform))
    assert max_deviation < 0.1, f"Con tau=100, la distribución debería ser casi uniforme. Desviación máx: {max_deviation:.3f}"
    print(f"✓ test_softmax_high_tau_explores: OK (proporciones: {np.round(proportions, 3)})")

def test_run_experiment_with_all():
    """run_experiment funciona con los 3 tipos de algoritmos juntos."""
    np.random.seed(42)
    k = 5
    bandit = Bandit(arms=ArmNormal.generate_arms(k))
    
    algorithms = [
        EpsilonGreedy(k=k, epsilon=0.1),
        UCB1(k=k, c=2**0.5),
        Softmax(k=k, tau=0.5),
    ]
    
    rewards, opt_sel, regret, arm_stats = run_experiment(bandit, algorithms, steps=200, runs=50)
    
    assert rewards.shape == (3, 200), f"Shape rewards incorrecto: {rewards.shape}"
    assert opt_sel.shape == (3, 200), f"Shape opt_sel incorrecto: {opt_sel.shape}"
    assert regret.shape == (3, 200), f"Shape regret incorrecto: {regret.shape}"
    assert len(arm_stats) == 3, f"arm_stats debería tener 3 elementos, tiene {len(arm_stats)}"
    
    print("✓ test_run_experiment_with_all: OK")
    print(f"  Recompensa final (paso 200): ε-greedy={rewards[0,-1]:.2f}, UCB1={rewards[1,-1]:.2f}, Softmax={rewards[2,-1]:.2f}")

if __name__ == "__main__":
    print("=" * 60)
    print("Tests para UCB1 y Softmax")
    print("=" * 60)
    
    test_ucb1_basic()
    test_ucb1_exploits_best()
    test_ucb1_reset()
    test_softmax_basic()
    test_softmax_exploits_best()
    test_softmax_high_tau_explores()
    test_run_experiment_with_all()
    
    print("\n" + "=" * 60)
    print("TODOS LOS TESTS PASARON ✓")
    print("=" * 60)
