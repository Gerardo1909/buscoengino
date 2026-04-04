"""
Módulo que contiene la clase Tokenizer, que se encarga de tokenizar el contenido de los documentos.
"""

from __future__ import annotations

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab", quiet=True)


class Tokenizer:
    """
    Clase que se encarga de tokenizar el contenido de los documentos.
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Tokeniza un texto y devuelve los tokens crudos.
        """
        return word_tokenize(text)
