"""
Módulo que contiene la clase Normalizer, que se encarga de normalizar texto y aplicar stemming.
"""

import re
import string

from nltk.stem.snowball import SnowballStemmer


class Normalizer:
    """
    Clase que se encarga de normalizar palabras dentro de un texto.
    """

    def __init__(self) -> None:
        # Usamos en español por la predominancia del idioma en nuestro corpus
        self.stemmer = SnowballStemmer("spanish")

    def remove_punctuation(self, text: str) -> str:
        """
        Elimina la puntuación del texto.
        """
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    def remove_digits(self, text: str) -> str:
        """
        Elimina los dígitos del texto.
        """
        translator = str.maketrans("", "", string.digits)
        return text.translate(translator)

    def trim_non_alphanumeric(self, text: str) -> str:
        """
        Elimina los caracteres no alfanuméricos al principio y al final del texto.
        """
        return re.sub(
            r"^[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]+|[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ]+$", "", text
        )

    def normalize_word(self, text: str) -> str:
        """
        Normaliza una palabra: limpia caracteres, pasa a minúsculas y aplica stemming.
        """
        text = text.strip().lower()
        text = self.remove_punctuation(text)
        text = self.remove_digits(text)
        text = self.trim_non_alphanumeric(text)
        if not text:
            return ""
        return self.stemmer.stem(text)
