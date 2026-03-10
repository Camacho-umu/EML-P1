"""
Module: arms/armbinomial.py
Description: Implementación de la clase ArmBinomial para el problema del bandido de k-brazos.

Un brazo binomial modela situaciones donde se realizan n ensayos independientes,
cada uno con probabilidad de éxito p. La recompensa es el número total de éxitos.
    X ~ B(n, p)
    E[X] = n * p
    Var[X] = n * p * (1 - p)

Ejemplo práctico: Optimización de promociones en una app de delivery.
Se ofrecen distintas promociones a lotes de n=100 usuarios; la recompensa es
el número de usuarios que aprovechan la promoción.

Nota: Si np >= 5 y n(1-p) >= 5, la binomial se puede aproximar por una Normal:
    X ≈ N(μ=np, σ²=np(1-p))

Referencia: Sección 5.1 de la práctica - Ejemplo de promociones en app de delivery.
"""

import numpy as np
from arms import Arm

class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución binomial.

        :param n: Número de ensayos (intentos).
        :param p: Probabilidad de éxito en cada ensayo.
        """
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."

        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución binomial B(n, p).

        :return: Recompensa obtenida del brazo (número de éxitos).
        """
        reward = np.random.binomial(self.n, self.p)
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución binomial.

        :return: Valor esperado (n * p).
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo binomial.

        :return: Descripción detallada del brazo.
        """
        return f"ArmBinomial(n={self.n}, p={self.p:.3f})"

    @classmethod
    def generate_arms(cls, k: int, n: int = 100, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos binomiales con el mismo n pero con probabilidades p únicas y aleatorias.

        Se evitan probabilidades extremas (0 y 1) para garantizar variabilidad.
        Se usa una lista (no un set) para mantener el orden determinista con la semilla fijada.

        :param k: Número de brazos a generar.
        :param n: Número de ensayos (por defecto 100, como en el ejemplo de las promociones).
        :param p_min: Probabilidad mínima de éxito (por defecto 0.1).
        :param p_max: Probabilidad máxima de éxito (por defecto 0.9).
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0.0 < p_min < p_max < 1.0, "p_min y p_max deben estar en (0, 1) con p_min < p_max."

        # Generar k valores únicos de probabilidad p (usando lista para reproducibilidad)
        p_values = []
        while len(p_values) < k:
            p = round(np.random.uniform(p_min, p_max), 3)
            if p not in p_values:
                p_values.append(p)

        arms = [ArmBinomial(n, p) for p in p_values]
        return arms