"""
Módulo que contiene la clase Normalizer, que se encarga de normalizar el contenido de los documentos.
"""

import re
import string


class Normalizer:
    """
    Clase que se encarga de normalizar una palabra dentro de un documento tokenizado.
    """

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
        Elimina los caracteres no alfanuméricos al principio del texto.
        """
        return re.sub(r"^[^a-zA-Z0-9]+", "", text)

    def normalize_word(self, text: str) -> str:
        """
        Normaliza una palabra dentro de un documento tokenizado.
        """
        text = self.remove_punctuation(text)
        text = self.remove_digits(text)
        text = self.trim_non_alphanumeric(text)
        return text
