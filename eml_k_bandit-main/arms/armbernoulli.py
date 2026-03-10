"""
Module: arms/armbernoulli.py
Description: Implementación de la clase ArmBernoulli para el problema del bandido de k-brazos.

Un brazo de Bernoulli modela situaciones binarias de éxito/fracaso (recompensa 1 o 0),
como por ejemplo la publicidad online donde un usuario hace clic (1) o no (0).
La distribución de Bernoulli es un caso particular de la Binomial con n=1:
    X ~ Bernoulli(p)  equivale a  X ~ B(1, p)
    E[X] = p
    Var[X] = p(1 - p)

Referencia: Sección 5.1 de la práctica - Ejemplo de publicidad online (CTR).
"""

import numpy as np
from arms import Arm

class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución de Bernoulli.

        :param p: Probabilidad de éxito.
        """
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."

        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución de Bernoulli (1 o 0).

        :return: Recompensa obtenida del brazo.
        """
        # Una Bernoulli es una binomial con n=1
        reward = np.random.binomial(1, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución de Bernoulli.

        :return: Valor esperado (p).
        """
        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo de Bernoulli.

        :return: Descripción detallada del brazo.
        """
        return f"ArmBernoulli(p={self.p:.3f})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos de Bernoulli con probabilidades p únicas y aleatorias.

        Se evitan valores extremos (0 y 1) para garantizar variabilidad en las recompensas.
        Se usa una lista (no un set) para mantener el orden determinista con la semilla fijada.

        :param k: Número de brazos a generar.
        :param p_min: Probabilidad mínima (por defecto 0.1).
        :param p_max: Probabilidad máxima (por defecto 0.9).
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert 0.0 < p_min < p_max < 1.0, "p_min y p_max deben estar en (0, 1) con p_min < p_max."

        # Generar k valores únicos de probabilidad p (usando lista para reproducibilidad)
        p_values = []
        while len(p_values) < k:
            p = round(np.random.uniform(p_min, p_max), 3)
            if p not in p_values:
                p_values.append(p)

        arms = [ArmBernoulli(p) for p in p_values]
        return arms