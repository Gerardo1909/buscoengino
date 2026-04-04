"""
Módulo que contiene la clase Stopwords, que se encarga de filtrar stopwords de tokens preprocesados.
"""

from pathlib import Path

from search_engine.preprocessing.normalizer import Normalizer


class Stopwords:
    """
    Filtra stopwords de una lista de tokens ya normalizados.

    Al cargar, aplica la misma normalización del corpus a cada stopword,
    garantizando que la comparación sea correcta después del stemming.
    """

    def __init__(self, path: Path, normalizer: Normalizer) -> None:
        self.stopwords = self._load_stopwords(path, normalizer)

    def _load_stopwords(self, path: Path, normalizer: Normalizer) -> set[str]:
        """
        Carga y normaliza las stopwords desde disco.

        Cada stopword pasa por la misma normalización que los tokens del corpus,
        para que la comparación sea válida tras el stemming.
        """
        stopwords: set[str] = set()
        for word in path.read_text(encoding="utf-8").splitlines():
            word = word.strip()
            if not word:
                continue
            normalized = normalizer.normalize_word(word)
            if normalized:
                stopwords.add(normalized)
        return stopwords

    def filter(self, tokens: list[str]) -> list[str]:
        """
        Elimina los tokens que son stopwords.
        """
        return [token for token in tokens if token not in self.stopwords]
