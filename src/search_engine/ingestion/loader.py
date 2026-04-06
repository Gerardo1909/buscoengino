"""
Módulo que se encarga de la lectura de documentos para su ingesta en el motor de búsqueda.
"""

import json
from pathlib import Path
from typing import Any, Dict

from search_engine.models.documents import Document


class Loader:
    """
    Clase que se encarga de la lectura de documentos para su ingesta en el motor de búsqueda.
    """

    def load_document(self, path: Path) -> Document:
        """
        Carga un documento de texto plano desde disco.

        Args:
            path: Ruta del archivo a leer.

        Returns:
            Document con la ruta y el contenido cargado.
        """
        content = path.read_text(encoding="utf-8")
        return Document(path=str(path), content=content)

    def load_documents(self, path: Path) -> list[Document]:
        """
        Carga todos los documentos .txt de una carpeta.

        Args:
            path: Ruta del directorio que contiene los documentos.

        Raises:
            ValueError: Si el path es un archivo y no es un .txt.

        Returns:
            Lista de documentos cargados desde archivos .txt.
        """
        if path.is_file():
            if path.suffix.lower() != ".txt":
                raise ValueError(f"Tipo de archivo no soportado: {path.suffix}")
            return [self.load_document(path)]
        documents: list[Document] = []
        for file_path in sorted(path.rglob("*.txt")):
            if file_path.is_file():
                documents.append(self.load_document(file_path))
        return documents

    def load_json(self, path: Path) -> Dict[str, Any]:
        """
        Carga datos desde un archivo JSON.

        Args:
            path: Ruta del archivo JSON a leer.

        Returns:
            Diccionario con los datos cargados. Retorna un diccionario vacío si el archivo no existe o es inválido.
        """
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {}

    def save_json(self, data: Dict[str, Any], path: Path) -> None:
        """
        Guarda un diccionario en formato JSON.

        Args:
            data: Diccionario con los datos a guardar.
            path: Ruta donde se guardará el archivo JSON.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
