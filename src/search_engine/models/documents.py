"""
Módulo que define los modelos de datos para documentos que serán usados en el motor de búsqueda.
"""

from dataclasses import dataclass


@dataclass
class Document:
    """
    Clase que representa un documento a ser ingestado en el motor de búsqueda.
    """

    path: str
    content: str
