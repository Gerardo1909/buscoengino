"""
Utilidades de ranking basadas en similitud vectorial.

Este módulo mantiene una única responsabilidad:
calcular la similitud coseno entre vectores.
"""

from __future__ import annotations

import numpy as np


class Scorer:
    """
    Métodos auxiliares para puntuar similitud entre vectores.
    """

    def cosine_similarity(
        self,
        vector_a: np.ndarray,
        vector_b: np.ndarray,
    ) -> float:
        """
        Calcula la similitud coseno entre dos vectores.

        Args:
            vector_a: Primer vector.
            vector_b: Segundo vector.

        Returns:
            Puntaje de similitud en el rango [0, 1] para vectores no negativos.

        Raises:
            ValueError: Si los vectores no tienen la misma longitud.
        """
        if len(vector_a) != len(vector_b):
            raise ValueError("Los vectores deben tener la misma longitud.")

        a = np.asarray(vector_a, dtype=float)
        b = np.asarray(vector_b, dtype=float)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))
