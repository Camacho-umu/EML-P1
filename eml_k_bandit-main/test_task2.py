"""
Test script for Task 2: Completar el estudio de la familia ε-greedy.
Tests: fix epsilon=0, run_experiment with regret, plotting functions.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from algorithms import Algorithm, EpsilonGreedy
from arms import ArmNormal, Bandit

# ====== TEST 1: epsilon=0 fix ======
print('=== TEST 1: Fix epsilon=0 ===')
np.random.seed(42)
k = 5
algo = EpsilonGreedy(k=k, epsilon=0)
bandit_test = Bandit(arms=ArmNormal.generate_arms(k))
arms_chosen = []
for _ in range(k):
    arm = algo.select_arm()
    reward = bandit_test.pull_arm(arm)
    algo.update(arm, reward)  # update counts so unexplored tracking works
    arms_chosen.append(arm)
print(f'  Primeras {k} selecciones con eps=0: {arms_chosen}')
unique = set(arms_chosen)
assert len(unique) == k, f'ERROR: con eps=0 deberia probar los {k} brazos, pero solo probo {unique}'
print(f'  OK: todos los {k} brazos fueron probados al menos una vez')

# ====== TEST 2: run_experiment ======
print()
print('=== TEST 2: run_experiment returns (rewards, optimal_sel, regret, arm_stats) ===')
np.random.seed(42)
k = 10
steps = 100
runs = 50
bandit = Bandit(arms=ArmNormal.generate_arms(k))
algorithms = [EpsilonGreedy(k=k, epsilon=0), EpsilonGreedy(k=k, epsilon=0.1)]

from main import run_experiment
results = run_experiment(bandit, algorithms, steps, runs)
rewards, optimal_sel, regret_acc, arm_stats = results

print(f'  rewards.shape = {rewards.shape} (expected: ({len(algorithms)}, {steps}))')
assert rewards.shape == (2, steps)
print(f'  optimal_sel.shape = {optimal_sel.shape}')
assert optimal_sel.shape == (2, steps)
print(f'  regret_acc.shape = {regret_acc.shape}')
assert regret_acc.shape == (2, steps)
print(f'  len(arm_stats) = {len(arm_stats)}')
assert len(arm_stats) == 2

# Verify ranges
print(f'  optimal_sel range: [{optimal_sel.min():.1f}%, {optimal_sel.max():.1f}%]')
assert 0 <= optimal_sel.min() and optimal_sel.max() <= 100
print(f'  regret_acc is monotonically non-decreasing: {np.all(np.diff(regret_acc[0]) >= -1e-10)}')
print(f'  regret_acc final: eps=0 -> {regret_acc[0, -1]:.2f}, eps=0.1 -> {regret_acc[1, -1]:.2f}')
print(f'  arm_stats[0] keys: {list(arm_stats[0].keys())}')
print(f'  arm_stats[0] counts sum: {arm_stats[0]["counts"].sum():.0f} (expected ~{steps})')

# ====== TEST 3: plotting import ======
print()
print('=== TEST 3: plotting functions importable ===')
from plotting import plot_average_rewards, plot_optimal_selections, plot_regret, plot_arm_statistics
print(f'  plot_average_rewards: {callable(plot_average_rewards)}')
print(f'  plot_optimal_selections: {callable(plot_optimal_selections)}')
print(f'  plot_regret: {callable(plot_regret)}')
print(f'  plot_arm_statistics: {callable(plot_arm_statistics)}')

# ====== TEST 4: get_algorithm_label ======
print()
print('=== TEST 4: get_algorithm_label ===')
from plotting.plotting import get_algorithm_label
algo0 = EpsilonGreedy(k=10, epsilon=0)
algo1 = EpsilonGreedy(k=10, epsilon=0.1)
print(f'  label eps=0: "{get_algorithm_label(algo0)}"')
print(f'  label eps=0.1: "{get_algorithm_label(algo1)}"')

# ====== TEST 5: Bernoulli & Binomial arms with run_experiment ======
print()
print('=== TEST 5: Bernoulli & Binomial arms with experiment ===')
from arms import ArmBernoulli, ArmBinomial

np.random.seed(42)
bandit_bern = Bandit(arms=ArmBernoulli.generate_arms(5))
algos_bern = [EpsilonGreedy(k=5, epsilon=0.1)]
r, o, reg, stats = run_experiment(bandit_bern, algos_bern, steps=50, runs=20)
print(f'  Bernoulli: rewards shape={r.shape}, regret final={reg[0,-1]:.2f}')

np.random.seed(42)
bandit_bin = Bandit(arms=ArmBinomial.generate_arms(5, n=100))
algos_bin = [EpsilonGreedy(k=5, epsilon=0.1)]
r, o, reg, stats = run_experiment(bandit_bin, algos_bin, steps=50, runs=20)
print(f'  Binomial: rewards shape={r.shape}, regret final={reg[0,-1]:.2f}')

print()
print('=============================')
print('=== ALL TESTS PASSED ✓ ===')
print('=============================')
