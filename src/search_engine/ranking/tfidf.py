"""
Módulo de utilidades para cálculo de TF-IDF y vectorización.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


class TFIDF:
    """
    Calcula componentes TF-IDF y genera vectores para documentos y consultas.
    """

    def term_frequency(self, tokens: Iterable[str]) -> dict[str, float]:
        """
        Calcula la frecuencia de término normalizada (TF).

        La fórmula utilizada es:

            TF(t, d) = ocurrencias_de_t_en_d / total_de_términos_en_d

        Se ignoran tokens vacíos.

        Args:
            tokens: Secuencia de tokens de un documento o consulta.

        Returns:
            Diccionario `término -> tf_normalizado`.
            Si no hay tokens válidos, retorna `{}`.
        """
        token_list = [token for token in tokens if token]
        if not token_list:
            return {}

        counts = Counter(token_list)
        total = len(token_list)
        return {term: count / total for term, count in counts.items()}

    def inverse_document_frequency(
        self,
        documents: list[list[str]],
    ) -> dict[str, float]:
        """
        Calcula la frecuencia inversa de documentos (IDF) para todo el corpus.

        Fórmula con suavizado:

            IDF(t) = log((N + 1) / (DF(t) + 1)) + 1

        donde:
            - N es la cantidad total de documentos.
            - DF(t) es la cantidad de documentos que contienen el término t.

        El suavizado evita divisiones por cero y mantiene valores estables.

        Args:
            documents: Lista de documentos tokenizados.

        Returns:
            Diccionario `término -> idf`.
            Si el corpus está vacío, retorna `{}`.
        """
        document_count = len(documents)
        if document_count == 0:
            return {}

        document_frequencies: Counter[str] = Counter()
        for tokens in documents:
            unique_terms = {token for token in tokens if token}
            document_frequencies.update(unique_terms)

        return {
            term: math.log((document_count + 1) / (df + 1)) + 1.0
            for term, df in document_frequencies.items()
        }

    def vectorize(
        self,
        tokens: Iterable[str],
        vocabulary: dict[str, int],
        idf: dict[str, float],
    ) -> np.ndarray:
        """
        Vectoriza una secuencia de tokens en el espacio del vocabulario.

        Cada componente del vector representa:

            TF-IDF(t, d) = TF(t, d) * IDF(t)

        Reglas:
            - Tokens fuera del vocabulario se ignoran.
            - Si falta IDF de un término, se asume `0.0`.
            - Si no hay vocabulario, retorna vector nulo.

        Args:
            tokens: Tokens de documento o consulta.
            vocabulary: Mapeo `término -> índice` en el vector.
            idf: Mapeo `término -> valor_idf`.

        Returns:
            Vector `numpy.ndarray` de tamaño `len(vocabulary)`.
        """
        vector = np.zeros(len(vocabulary), dtype=float)
        if not vocabulary:
            return vector

        tf = self.term_frequency(tokens)
        if not tf:
            return vector

        for term, tf_value in tf.items():
            index = vocabulary.get(term)
            if index is None:
                continue
            vector[index] = tf_value * idf.get(term, 0.0)

        return vector
