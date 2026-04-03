"""
Módulo que contiene la clase Stopwords, que se encarga de eliminar las stopwords de un texto.
"""

from search_engine.models.documents import Document


class Stopwords:
    """
    Clase que se encarga de eliminar las stopwords de un texto.
    """

    def __init__(self, stopwords: Document):
        self.stopwords = stopwords.content.split()

    def remove_stopwords(self, text: str) -> str:
        """
        Elimina las stopwords del texto.
        """
        return " ".join([word for word in text.split() if word not in self.stopwords])
