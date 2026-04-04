"""
Modelos de datos centrales del motor de búsqueda.

Estos modelos hacen explícito el pipeline:

Document -> ProcessedDocument -> QueryRepresentation -> SearchResult

También encapsulan el estado del corpus (`CorpusState`) para evitar pasar
estructuras sueltas entre módulos y reducir la carga cognitiva.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Document:
    """
    Documento crudo cargado desde disco.
    """

    path: str
    content: str


@dataclass(frozen=True)
class ProcessedDocument:
    """
    Documento luego de preprocesamiento textual.
    """

    path: str
    tokens: list[str]


@dataclass
class CorpusState:
    """
    Estado materializado del corpus indexado para ranking TF-IDF.

    Atributos:
        documents: Documentos crudos cargados.
        processed_documents: Documentos preprocesados (tokens normalizados).
        vocabulary: Mapa término -> índice dentro del vector.
        idf: Mapa término -> valor IDF precalculado.
        document_vectors: Vectores TF-IDF de cada documento del corpus.
        inverted_index: Índice invertido (término -> lista de ids/rutas de documento).

    Nota:
        `inverted_index` se mantiene aunque el ranking actual recorra todos los
        documentos, porque habilita una fase futura de recuperación de candidatos
        para escalar mejor.
    """

    documents: list[Document] = field(default_factory=list)
    processed_documents: list[ProcessedDocument] = field(default_factory=list)
    vocabulary: dict[str, int] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    document_vectors: list[np.ndarray] = field(default_factory=list)
    inverted_index: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRepresentation:
    """
    Representación de una consulta en términos del pipeline de ranking.
    """

    original_query: str
    tokens: list[str]
    vector: np.ndarray


@dataclass(frozen=True)
class SearchResult:
    """
    Resultado rankeado de una búsqueda.
    """

    document: Document
    score: float
