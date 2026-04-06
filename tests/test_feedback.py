import json
import os

import numpy as np
import pytest

from search_engine.search.feedback import FeedbackStore


@pytest.fixture
def temp_feedback_file(tmp_path):
    return str(tmp_path / "test_feedback.json")


class TestFeedbackStore:
    def test_init_should_create_empty_data_when_file_not_exists(
        self, temp_feedback_file
    ):
        store = FeedbackStore(temp_feedback_file)
        assert store.data == {"queries": {}}

    def test_add_feedback_should_store_new_query_and_feedback(self, temp_feedback_file):
        store = FeedbackStore(temp_feedback_file)
        vector = np.array([1.0, 0.5])

        store.add_feedback("test query", vector, "doc1.txt", True)

        assert "test query" in store.data["queries"]
        assert store.data["queries"]["test query"]["vector"] == [1.0, 0.5]
        assert (
            store.data["queries"]["test query"]["feedback"]["doc1.txt"]["positive"] == 1
        )
        assert (
            store.data["queries"]["test query"]["feedback"]["doc1.txt"]["negative"] == 0
        )

        assert os.path.exists(temp_feedback_file)
        with open(temp_feedback_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert (
            saved_data["queries"]["test query"]["feedback"]["doc1.txt"]["positive"] == 1
        )

    def test_add_feedback_should_increment_counts_when_query_exists(
        self, temp_feedback_file
    ):
        store = FeedbackStore(temp_feedback_file)
        vector = np.array([1.0, 0.5])

        store.add_feedback("test query", vector, "doc1.txt", True)
        store.add_feedback("test query", vector, "doc1.txt", False)
        store.add_feedback("test query", vector, "doc1.txt", True)

        feedback = store.data["queries"]["test query"]["feedback"]["doc1.txt"]
        assert feedback["positive"] == 2
        assert feedback["negative"] == 1

    def test_cosine_similarity_should_handle_different_length_vectors(
        self, temp_feedback_file
    ):
        store = FeedbackStore(temp_feedback_file)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.5])

        sim = store._cosine_similarity(v1, v2)
        assert sim > 0

    def test_cosine_similarity_should_return_zero_when_norm_is_zero(
        self, temp_feedback_file
    ):
        store = FeedbackStore(temp_feedback_file)
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 1.0])

        sim = store._cosine_similarity(v1, v2)
        assert sim == 0.0

    def test_get_feedback_scores_should_return_empty_when_no_data(
        self, temp_feedback_file
    ):
        store = FeedbackStore(temp_feedback_file)
        scores = store.get_feedback_scores(np.array([1.0]))
        assert scores == {}

    def test_get_feedback_scores_should_return_empty_when_below_threshold(
        self, temp_feedback_file
    ):
        store = FeedbackStore(temp_feedback_file)
        store.add_feedback("test query", np.array([1.0, 0.0]), "doc1.txt", True)

        scores = store.get_feedback_scores(np.array([0.0, 1.0]), threshold=0.1)
        assert scores == {}

    def test_get_feedback_scores_should_return_proportional_scores(
        self, temp_feedback_file
    ):
        store = FeedbackStore(temp_feedback_file)
        vector = np.array([1.0, 0.0])

        store.add_feedback("test query", vector, "doc1.txt", True)
        store.add_feedback("test query", vector, "doc1.txt", True)
        store.add_feedback("test query", vector, "doc1.txt", False)
        store.add_feedback("test query", vector, "doc2.txt", False)

        scores = store.get_feedback_scores(vector)

        assert abs(scores["doc1.txt"] - (2 / 3)) < 1e-5
        assert scores["doc2.txt"] == 0.0
