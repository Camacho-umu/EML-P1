"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax (Boltzmann exploration) para el problema de los k-brazos.

El algoritmo Softmax asigna probabilidades de selección a cada brazo según una distribución
de Boltzmann sobre los valores estimados, controlada por un parámetro de temperatura τ.

Fórmula de selección:
    P(A_t = a) = exp(Q_t(a) / τ) / Σ_b exp(Q_t(b) / τ)

donde:
    - Q_t(a): valor estimado del brazo a en el paso t
    - τ (tau): parámetro de temperatura
        * τ → ∞: distribución uniforme (exploración pura)
        * τ → 0+: distribución concentrada en argmax Q (explotación pura, greedy)

Ventaja sobre ε-greedy:
    - ε-greedy explora uniformemente entre todos los brazos, incluyendo los claramente subóptimos.
    - Softmax pondera la exploración según la calidad estimada: es más probable explorar
      brazos que parecen prometedores y menos probable elegir brazos claramente malos.

Author: Extensiones de Machine Learning — Práctica RL
Date: 2025
"""

import numpy as np

from algorithms.algorithm import Algorithm


class Softmax(Algorithm):

    def __init__(self, k: int, tau: float = 1.0):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param tau: Parámetro de temperatura. Controla el equilibrio entre exploración y explotación.
                    - tau alto (e.g., 10): distribución casi uniforme → mucha exploración
                    - tau bajo (e.g., 0.01): distribución casi greedy → mucha explotación
                    - tau = 1: equilibrio intermedio
        :raises AssertionError: Si tau no es positivo.
        """
        assert tau > 0, "El parámetro tau debe ser positivo (tau > 0)."

        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la distribución de Boltzmann.

        Fase de inicialización: si hay brazos con counts[i] == 0, se selecciona
        uno aleatoriamente para garantizar que todos se prueban al menos una vez.

        Tras la inicialización, se calcula la probabilidad de selección de cada brazo
        usando softmax y se muestrea según esa distribución.

        Se usa el truco de estabilidad numérica: restar el máximo antes de calcular exp()
        para evitar overflow en la exponencial.

        :return: Índice del brazo seleccionado.
        """
        # Fase de inicialización: probar brazos no explorados
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)

        # Calcular probabilidades softmax con truco de estabilidad numérica
        scaled_values = self.values / self.tau
        # Restar el máximo para evitar overflow en exp()
        scaled_values -= np.max(scaled_values)
        exp_values = np.exp(scaled_values)
        probabilities = exp_values / np.sum(exp_values)

        # Muestrear un brazo según las probabilidades
        return int(np.random.choice(self.k, p=probabilities))
