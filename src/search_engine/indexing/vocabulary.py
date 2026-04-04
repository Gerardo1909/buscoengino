"""
Módulo para construir el vocabulario del corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class VocabularyBuilder:
    """
    Construye un vocabulario ordenado alfabéticamente.

    El vocabulario se representa como un diccionario:
    término normalizado -> posición dentro del vocabulario.
    """

    vocabulary: dict[str, int] = field(default_factory=dict)

    def build(self, documents: Iterable[Iterable[str]]) -> dict[str, int]:
        """
        Construye el vocabulario a partir de una colección de documentos tokenizados.

        Args:
            documents: Iterable de documentos, donde cada documento es un iterable de tokens.

        Returns:
            dict[str, int]: Vocabulario ordenado alfabéticamente.
        """
        terms = set()

        for document in documents:
            for token in document:
                if token:
                    terms.add(token)

        self.vocabulary = {term: index for index, term in enumerate(sorted(terms))}
        return self.vocabulary
