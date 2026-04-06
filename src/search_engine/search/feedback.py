"""
Módulo para el manejo del sistema de retroalimentación (feedback) histórico.
Permite almacenar y consultar qué documentos fueron relevantes para consultas pasadas.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from search_engine.config.settings import FEEDBACK_FILE
from search_engine.ingestion.loader import Loader
from search_engine.ranking.scorer import Scorer


class FeedbackStore:
    """
    Almacén de feedback que guarda y consulta las interacciones de los usuarios
    con los resultados de búsqueda.
    """

    def __init__(self, filepath: Path | str = FEEDBACK_FILE):
        """
        Inicializa el almacén de feedback.

        Args:
            filepath: Ruta al archivo JSON donde se guarda el feedback.
        """
        self.filepath = Path(filepath)
        self.loader = Loader()
        self.scorer = Scorer()
        self.data: Dict[str, Any] = {"queries": {}}
        self._load()

    def _load(self) -> None:
        """Carga el estado del feedback desde el disco."""
        data = self.loader.load_json(self.filepath)
        if data and "queries" in data:
            self.data = data
        else:
            self.data = {"queries": {}}

    def _save(self) -> None:
        """Guarda el estado actual del feedback en el disco."""
        self.loader.save_json(self.data, self.filepath)

    def add_feedback(
        self, query: str, query_vector: np.ndarray, doc_path: str, is_relevant: bool
    ) -> None:
        """
        Registra feedback (positivo o negativo) para un documento ante una consulta.

        Args:
            query: La consulta original en texto plano.
            query_vector: Representación vectorial de la consulta.
            doc_path: Ruta del documento calificado.
            is_relevant: True si fue útil, False si no lo fue.
        """
        queries = self.data["queries"]
        if query not in queries:
            queries[query] = {"vector": query_vector.tolist(), "feedback": {}}

        feedback = queries[query]["feedback"]
        if doc_path not in feedback:
            feedback[doc_path] = {"positive": 0, "negative": 0}

        if is_relevant:
            feedback[doc_path]["positive"] += 1
        else:
            feedback[doc_path]["negative"] += 1

        self._save()

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos vectores.
        Aplica padding si los tamaños del vocabulario difieren entre sesiones.
        """
        len1, len2 = len(v1), len(v2)
        if len1 > len2:
            v2 = np.pad(v2, (0, len1 - len2))
        elif len2 > len1:
            v1 = np.pad(v1, (0, len2 - len1))

        return self.scorer.cosine_similarity(v1, v2)

    def get_feedback_scores(
        self, query_vector: np.ndarray, threshold: float = 0.1
    ) -> Dict[str, float]:
        """
        Calcula puntajes basados en feedback histórico encontrando la consulta más similar.

        Args:
            query_vector: Vector de la consulta actual.
            threshold: Similitud mínima requerida para considerar un query histórico.

        Returns:
            Diccionario de rutas de documentos a su puntaje derivado del feedback.
        """
        queries = self.data.get("queries", {})
        if not queries:
            return {}

        best_query = None
        max_similarity = -1.0

        for q_str, q_data in queries.items():
            stored_vector = np.array(q_data["vector"])
            sim = self._cosine_similarity(query_vector, stored_vector)

            if sim > max_similarity:
                max_similarity = sim
                best_query = q_str

        if best_query is None or max_similarity < threshold:
            return {}

        best_feedback = queries[best_query]["feedback"]
        scores = {}

        for doc_path, stats in best_feedback.items():
            pos = stats.get("positive", 0)
            neg = stats.get("negative", 0)
            total = pos + neg
            if total > 0:
                # El puntaje es la proporción de feedback positivo ponderado por la similitud de la consulta
                scores[doc_path] = (pos / total) * max_similarity

        return scores
