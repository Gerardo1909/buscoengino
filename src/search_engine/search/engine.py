"""
Módulo que implementa el motor de búsqueda.
"""

from __future__ import annotations

from search_engine.config.settings import RAW_DOCUMENTS_DIR, STOP_WORDS_FILE
from search_engine.indexing.inverted_index import InvertedIndex, VocabularyBuilder
from search_engine.ingestion.loader import Loader
from search_engine.models.documents import (
    CorpusState,
    Document,
    ProcessedDocument,
    QueryRepresentation,
    SearchResult,
)
from search_engine.preprocessing.normalizer import Normalizer
from search_engine.preprocessing.stopwords import Stopwords
from search_engine.preprocessing.tokenizer import Tokenizer
from search_engine.ranking.scorer import Scorer
from search_engine.ranking.tfidf import TFIDF


class BuscoEngino:
    """
    Motor de búsqueda basado en un pipeline explícito:

    Ingesta → Tokenización → Normalización → Filtrado de stopwords →
    Vocabulario → IDF → Vectores TF-IDF → Similitud coseno.
    """

    def __init__(
        self,
        documents_path=RAW_DOCUMENTS_DIR,
        stopwords_path=STOP_WORDS_FILE,
    ) -> None:
        self.documents_path = documents_path
        self.stopwords_path = stopwords_path

        self.loader = Loader()
        self.tokenizer = Tokenizer()
        self.normalizer = Normalizer()
        self.tfidf = TFIDF()
        self.scorer = Scorer()
        self.vocabulary_builder = VocabularyBuilder()
        self.inverted_index = InvertedIndex()

        self._stopwords_filter: Stopwords | None = None
        self.state = CorpusState()
        self._initialized = False

    def _validate_paths(self) -> None:
        """
        Valida que las rutas críticas del motor existan y sean del tipo esperado.
        """
        if not self.documents_path.exists() or not self.documents_path.is_dir():
            raise FileNotFoundError(
                f"La ruta de documentos no existe o no es un directorio: {self.documents_path}"
            )
        if not self.stopwords_path.exists() or not self.stopwords_path.is_file():
            raise FileNotFoundError(
                f"El archivo de stopwords no existe: {self.stopwords_path}"
            )

    def _preprocess_text(self, text: str) -> list[str]:
        """
        Ejecuta el preprocesamiento completo de un texto:

        1) Tokenización con NLTK
        2) Normalización de cada token (minúsculas, limpieza, stemming)
        3) Filtrado de stopwords sobre los tokens normalizados
        """
        if self._stopwords_filter is None:
            raise RuntimeError("El motor no está inicializado.")

        raw_tokens = self.tokenizer.tokenize(text)
        normalized = [self.normalizer.normalize_word(t) for t in raw_tokens]
        normalized = [t for t in normalized if t]
        return self._stopwords_filter.filter(normalized)

    def _preprocess_documents(
        self, documents: list[Document]
    ) -> list[ProcessedDocument]:
        """
        Preprocesa todos los documentos del corpus.
        """
        return [
            ProcessedDocument(path=doc.path, tokens=self._preprocess_text(doc.content))
            for doc in documents
        ]

    def _build_corpus_state(self) -> None:
        """
        Materializa el estado completo del corpus:
        documentos, preprocesado, vocabulario, IDF, vectores e índice invertido.
        """
        self._validate_paths()

        self._stopwords_filter = Stopwords(self.stopwords_path, self.normalizer)
        documents = self.loader.load_documents(self.documents_path)
        processed_documents = self._preprocess_documents(documents)

        vocabulary = self.vocabulary_builder.build(
            doc.tokens for doc in processed_documents
        )
        idf = self.tfidf.inverse_document_frequency(
            [doc.tokens for doc in processed_documents]
        )
        document_vectors = [
            self.tfidf.vectorize(doc.tokens, vocabulary, idf)
            for doc in processed_documents
        ]
        inverted_index = self.inverted_index.build(
            {doc.path: doc.tokens for doc in processed_documents}
        )

        self.state.documents = documents
        self.state.processed_documents = processed_documents
        self.state.vocabulary = vocabulary
        self.state.idf = idf
        self.state.document_vectors = document_vectors
        self.state.inverted_index = inverted_index

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """
        Inicializa el estado del motor una única vez (lazy initialization).
        """
        if not self._initialized:
            self._build_corpus_state()

    def _build_query_representation(self, query: str) -> QueryRepresentation:
        """
        Construye la representación vectorial de una consulta.
        """
        if not query or not query.strip():
            raise ValueError("La consulta debe ser un texto no vacío.")

        tokens = self._preprocess_text(query)
        vector = self.tfidf.vectorize(tokens, self.state.vocabulary, self.state.idf)
        return QueryRepresentation(original_query=query, tokens=tokens, vector=vector)

    def search(self, query: str) -> list[SearchResult]:
        """
        Ejecuta la búsqueda por similitud coseno sobre vectores TF-IDF.

        Flujo:
            1) Inicializa el corpus (si aplica)
            2) Preprocesa y vectoriza la consulta
            3) Calcula similitud coseno contra todos los documentos
            4) Devuelve resultados con score > 0, ordenados de mayor a menor relevancia

        Args:
            query: Consulta de búsqueda.

        Returns:
            Lista de SearchResult ordenados por relevancia descendente.
        """
        self._ensure_initialized()

        if not self.state.documents:
            return []

        query_representation = self._build_query_representation(query)

        results: list[SearchResult] = []
        for document, document_vector in zip(
            self.state.documents, self.state.document_vectors
        ):
            score = self.scorer.cosine_similarity(
                query_representation.vector,
                document_vector,
            )
            if score > 0.0:
                results.append(SearchResult(document=document, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results


if __name__ == "__main__":
    engine = BuscoEngino()
    results = engine.search("Messi")
    for result in results:
        print(f"{result.score:.4f} - {result.document.path}")
