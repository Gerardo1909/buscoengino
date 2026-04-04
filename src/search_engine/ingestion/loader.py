"""
Módulo que se encarga de la lectura de documentos para su ingesta en el motor de búsqueda.
"""

from pathlib import Path

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

        Returns:
            Lista de documentos cargados desde archivos .txt.
        """
        if path.is_file():
            if path.suffix.lower() != ".txt":
                return []
            return [self.load_document(path)]

        if not path.exists():
            return []

        documents: list[Document] = []
        for file_path in sorted(path.rglob("*.txt")):
            if file_path.is_file():
                documents.append(self.load_document(file_path))

        return documents
