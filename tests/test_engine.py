"""
Tests unitarios para el motor de búsqueda BuscoEngino.
"""

from unittest.mock import patch

import numpy as np
import pytest

from search_engine.models.documents import (
    Document,
    ProcessedDocument,
    QueryRepresentation,
    SearchResult,
)
from search_engine.search.engine import BuscoEngino


@pytest.fixture
def sample_documents():
    """Proporciona documentos de prueba para tests."""
    return [
        Document(
            path="doc1.txt",
            content="Lionel Messi es un jugador de fútbol argentino famoso.",
        ),
        Document(
            path="doc2.txt",
            content="Cristiano Ronaldo es un futbolista portugués excepcional.",
        ),
        Document(
            path="doc3.txt",
            content="El fútbol es el deporte más popular del mundo.",
        ),
    ]


@pytest.fixture
def engine():
    """Proporciona una instancia del motor de búsqueda."""
    return BuscoEngino()


class TestPreprocessText:
    """Tests para el método _preprocess_text."""

    def test_preprocess_text_should_remove_stopwords_when_processing_common_words(
        self, engine
    ):
        # Arrange
        text = "el gato y el perro están en la casa"

        # Act
        result = engine._preprocess_text(text)

        # Assert
        assert "el" not in result
        assert "y" not in result
        assert "en" not in result
        assert "la" not in result
        assert len(result) > 0

    def test_preprocess_text_should_normalize_tokens_when_processing_text(self, engine):
        # Arrange
        text = "MESSI Football 2024"

        # Act
        result = engine._preprocess_text(text)

        # Assert
        assert all(token == token.lower() for token in result)
        assert "2024" not in result

    def test_preprocess_text_should_return_empty_list_when_text_contains_only_stopwords(
        self, engine
    ):
        # Arrange
        text = "el y la de que"

        # Act
        result = engine._preprocess_text(text)

        # Assert
        assert result == []

    def test_preprocess_text_should_raise_error_when_stopwords_filter_not_initialized(
        self,
    ):
        # Arrange
        engine_broken = BuscoEngino()
        engine_broken._stopwords_filter = None
        text = "test content"

        # Act & Assert
        with pytest.raises(RuntimeError, match="no está inicializado"):
            engine_broken._preprocess_text(text)


class TestPreprocessDocuments:
    """Tests para el método _preprocess_documents."""

    def test_preprocess_documents_should_convert_all_documents_to_tokens(
        self, engine, sample_documents
    ):
        # Arrange
        documents = sample_documents

        # Act
        result = engine._preprocess_documents(documents)

        # Assert
        assert len(result) == len(documents)
        assert all(isinstance(doc, ProcessedDocument) for doc in result)
        assert all(doc.path == orig.path for doc, orig in zip(result, documents))

    def test_preprocess_documents_should_remove_empty_tokens_during_processing(
        self, engine
    ):
        # Arrange
        documents = [Document(path="test.txt", content="")]

        # Act
        result = engine._preprocess_documents(documents)

        # Assert
        assert len(result) == 1
        assert result[0].tokens == []

    def test_preprocess_documents_should_handle_multiple_documents(
        self, engine, sample_documents
    ):
        # Arrange
        documents = sample_documents

        # Act
        result = engine._preprocess_documents(documents)

        # Assert
        assert len(result) == 3
        assert all(isinstance(doc.tokens, list) for doc in result)


class TestBuildQueryRepresentation:
    """Tests para el método _build_query_representation."""

    def test_build_query_representation_should_create_valid_representation_when_valid_query(
        self, engine
    ):
        # Arrange
        engine.state.vocabulary = {"futbol": 0, "jugador": 1}
        engine.state.idf = {"futbol": 1.5, "jugador": 2.0}
        query = "fútbol jugador"

        # Act
        result = engine._build_query_representation(query)

        # Assert
        assert isinstance(result, QueryRepresentation)
        assert result.original_query == query
        assert isinstance(result.tokens, list)
        assert isinstance(result.vector, np.ndarray)

    def test_build_query_representation_should_raise_error_when_empty_query(
        self, engine
    ):
        # Arrange
        query = ""

        # Act & Assert
        with pytest.raises(ValueError, match="debe ser un texto no vacío"):
            engine._build_query_representation(query)

    def test_build_query_representation_should_raise_error_when_whitespace_only_query(
        self, engine
    ):
        # Arrange
        query = "   \t\n"

        # Act & Assert
        with pytest.raises(ValueError, match="debe ser un texto no vacío"):
            engine._build_query_representation(query)

    def test_build_query_representation_should_generate_vector_matching_vocabulary_size(
        self, engine
    ):
        # Arrange
        engine.state.vocabulary = {"a": 0, "b": 1, "c": 2}
        engine.state.idf = {"a": 1.0, "b": 1.0, "c": 1.0}
        query = "test query"

        # Act
        result = engine._build_query_representation(query)

        # Assert
        assert len(result.vector) == len(engine.state.vocabulary)


class TestSearch:
    """Tests para el método search."""

    @patch("search_engine.search.engine.Loader.load_documents")
    def test_search_should_return_empty_list_when_corpus_not_initialized(
        self, mock_load_documents, engine
    ):
        # Arrange
        mock_load_documents.return_value = []
        query = "Messi"

        # Act
        results = engine.search(query)

        # Assert
        assert results == []
        assert isinstance(results, list)

    @patch("search_engine.search.engine.BuscoEngino._build_corpus_state")
    def test_search_should_return_sorted_results_by_score_when_corpus_initialized(
        self, mock_build_corpus, engine, sample_documents
    ):
        # Arrange
        engine.state.documents = sample_documents
        engine.state.vocabulary = {"messi": 0, "futbol": 1, "jugador": 2}
        engine.state.idf = {"messi": 2.0, "futbol": 1.5, "jugador": 1.8}

        engine.state.document_vectors = [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.0, 0.4, 0.1]),
            np.array([0.0, 0.6, 0.0]),
        ]

        query = "Messi futbol"

        # Act
        results = engine.search(query)

        # Assert
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    @patch("search_engine.search.engine.BuscoEngino._build_corpus_state")
    def test_search_should_filter_results_with_zero_score_when_no_matching_terms(
        self, mock_build_corpus, engine, sample_documents
    ):
        # Arrange
        engine.state.documents = sample_documents
        engine.state.vocabulary = {"test": 0}
        engine.state.idf = {"test": 1.0}
        engine.state.document_vectors = [
            np.array([0.0]),
            np.array([0.0]),
            np.array([0.0]),
        ]

        query = "Messi"

        # Act
        results = engine.search(query)

        # Assert
        assert results == []

    @patch("search_engine.search.engine.BuscoEngino._build_corpus_state")
    def test_search_should_return_results_with_positive_scores_only(
        self, mock_build_corpus, engine
    ):
        # Arrange
        engine.state.documents = [
            Document(path="doc1.txt", content="test"),
            Document(path="doc2.txt", content="test"),
        ]
        engine.state.vocabulary = {"test": 0}
        engine.state.idf = {"test": 1.0}
        engine.state.document_vectors = [
            np.array([0.5]),
            np.array([0.0]),
        ]

        query = "test"

        # Act
        results = engine.search(query)

        # Assert
        assert all(r.score > 0.0 for r in results)
        assert len(results) == 1


class TestBuildCorpusState:
    """Tests para el método _build_corpus_state."""

    @patch("search_engine.search.engine.Loader.load_documents")
    def test_build_corpus_state_should_populate_all_state_fields_when_successful(
        self, mock_load_documents, engine, sample_documents
    ):
        # Arrange
        mock_load_documents.return_value = sample_documents

        # Act
        engine._build_corpus_state()

        # Assert
        assert engine.state.documents == sample_documents
        assert len(engine.state.processed_documents) == len(sample_documents)
        assert isinstance(engine.state.vocabulary, dict)
        assert isinstance(engine.state.idf, dict)
        assert len(engine.state.document_vectors) == len(sample_documents)

    @patch("search_engine.search.engine.Loader.load_documents")
    def test_build_corpus_state_should_handle_empty_document_list(
        self, mock_load_documents, engine
    ):
        # Arrange
        mock_load_documents.return_value = []

        # Act
        engine._build_corpus_state()

        # Assert
        assert engine.state.documents == []
        assert engine.state.processed_documents == []
        assert engine.state.vocabulary == {}
        assert engine.state.idf == {}
        assert engine.state.document_vectors == []

    @patch("search_engine.search.engine.Loader.load_documents")
    def test_build_corpus_state_should_create_vocabulary_from_all_tokens(
        self, mock_load_documents, engine, sample_documents
    ):
        # Arrange
        mock_load_documents.return_value = sample_documents

        # Act
        engine._build_corpus_state()

        # Assert
        assert isinstance(engine.state.vocabulary, dict)
        assert len(engine.state.vocabulary) > 0
        assert all(isinstance(idx, int) for idx in engine.state.vocabulary.values())

    @patch("search_engine.search.engine.Loader.load_documents")
    def test_build_corpus_state_should_compute_idf_scores(
        self, mock_load_documents, engine, sample_documents
    ):
        # Arrange
        mock_load_documents.return_value = sample_documents

        # Act
        engine._build_corpus_state()

        # Assert
        assert isinstance(engine.state.idf, dict)
        assert all(isinstance(score, float) for score in engine.state.idf.values())


class TestEngineInitialization:
    """Tests para la inicialización del motor."""

    def test_engine_should_initialize_with_default_paths(self):
        # Arrange & Act
        engine = BuscoEngino()

        # Assert
        assert engine.documents_path is not None
        assert engine.stopwords_path is not None
        assert engine._initialized is False

    def test_engine_should_initialize_all_components(self):
        # Arrange & Act
        engine = BuscoEngino()

        # Assert
        assert engine.loader is not None
        assert engine.tokenizer is not None
        assert engine.normalizer is not None
        assert engine.tfidf is not None
        assert engine.scorer is not None
        assert engine.vocabulary_builder is not None
        assert engine._stopwords_filter is not None

    def test_engine_should_initialize_empty_corpus_state(self):
        # Arrange & Act
        engine = BuscoEngino()

        # Assert
        assert engine.state.documents == []
        assert engine.state.processed_documents == []
        assert engine.state.vocabulary == {}
        assert engine.state.idf == {}
        assert engine.state.document_vectors == []


@pytest.mark.smoke
class TestSearchIntegration:
    """Tests de integración para flujos completos de búsqueda."""

    @patch("search_engine.search.engine.Loader.load_documents")
    def test_search_integration_should_return_documents_matching_query(
        self, mock_load_documents, engine
    ):
        # Arrange
        docs = [
            Document(path="doc1.txt", content="Messi es un jugador de fútbol"),
            Document(path="doc2.txt", content="El fútbol es un deporte"),
        ]
        mock_load_documents.return_value = docs
        engine._build_corpus_state()

        # Act
        results = engine.search("fútbol")

        # Assert
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @patch("search_engine.search.engine.Loader.load_documents")
    def test_search_integration_should_rank_documents_by_relevance(
        self, mock_load_documents, engine
    ):
        # Arrange
        docs = [
            Document(path="doc1.txt", content="Messi futbol jugador argentino"),
            Document(path="doc2.txt", content="futbol deportes entretenimiento"),
        ]
        mock_load_documents.return_value = docs
        engine._build_corpus_state()

        # Act
        results = engine.search("Messi futbol")

        # Assert
        assert len(results) > 0
        assert results[0].document.path == "doc1.txt"
