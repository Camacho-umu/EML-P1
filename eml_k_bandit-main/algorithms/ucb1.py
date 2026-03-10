"""
Module: algorithms/ucb1.py
Description: Implementación del algoritmo UCB1 (Upper Confidence Bound) para el problema de los k-brazos.

UCB1 resuelve el dilema exploración-explotación de forma determinista asignando a cada brazo
un bono de confianza que decrece con el número de selecciones y crece con el tiempo total.

Fórmula de selección (Auer et al., 2002):
    A_t = argmax_a [ Q_t(a) + c · sqrt( ln(t) / N_t(a) ) ]

donde:
    - Q_t(a): valor estimado del brazo a en el paso t
    - N_t(a): número de veces que se ha seleccionado el brazo a
    - t: número total de pasos hasta el momento
    - c: parámetro de exploración (c = sqrt(2) es el valor teórico)

Propiedades teóricas:
    - Regret logarítmico: R(T) = O(sqrt(k · T · ln(T))), y bajo ciertas condiciones O(ln(T))
    - No requiere conocer el horizonte temporal T de antemano
    - Convergencia: el porcentaje de selección del brazo óptimo tiende a 100%

Author: Extensiones de Machine Learning — Práctica RL
Date: 2025
"""

import numpy as np

from algorithms.algorithm import Algorithm


class UCB1(Algorithm):

    def __init__(self, k: int, c: float = 2**0.5):
        """
        Inicializa el algoritmo UCB1.

        :param k: Número de brazos.
        :param c: Parámetro de exploración que controla el peso del término de confianza.
                  c = sqrt(2) es el valor teórico derivado de la desigualdad de Hoeffding.
                  Valores más altos favorecen la exploración; más bajos, la explotación.
        :raises AssertionError: Si c es negativo.
        """
        assert c >= 0, "El parámetro c debe ser no negativo."

        super().__init__(k)
        self.c = c
        # Contador total de pasos (suma de counts, pero lo mantenemos por eficiencia)
        self.total_counts: int = 0

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1.

        Fase de inicialización: si hay brazos con counts[i] == 0, se selecciona
        uno aleatoriamente para garantizar que todos se prueban al menos una vez.
        Esto es necesario porque el término sqrt(ln(t)/0) no está definido.

        Tras la inicialización, se selecciona el brazo que maximiza:
            Q(a) + c · sqrt(ln(t) / N(a))

        :return: Índice del brazo seleccionado.
        """
        # Fase de inicialización: probar brazos no explorados
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)

        # Calcular UCB para cada brazo
        t = self.total_counts
        ucb_values = self.values + self.c * np.sqrt(np.log(t) / self.counts)

        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las estadísticas del brazo elegido y el contador total.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        self.total_counts += 1
        super().update(chosen_arm, reward)

    def reset(self):
        """
        Reinicia el estado del algoritmo.
        """
        super().reset()
        self.total_counts = 0
