
import pytest

from src.vectorstore.embeddings import (
    FakeEmbedder,
    Embedder,
    APIEmbedder,
)


def test_fake_embedder_returns_vectors():
    embedder = FakeEmbedder(dim=64)

    result = embedder.embed(["machine learning"])

    assert len(result) == 1
    assert len(result[0]) == 64


def test_embedding_values_are_normalized():
    embedder = FakeEmbedder(dim=64)

    vector = embedder.embed(["machine learning"])[0]

    norm = sum(value ** 2 for value in vector) ** 0.5

    assert norm == pytest.approx(1.0)


def test_same_text_produces_same_embedding():
    embedder = FakeEmbedder(dim=64)

    first = embedder.embed(["cardiovascular disease"])
    second = embedder.embed(["cardiovascular disease"])

    assert first == second


def test_different_text_produces_embeddings():
    embedder = FakeEmbedder(dim=64)

    first = embedder.embed(["machine learning"])
    second = embedder.embed(["natural language processing"])

    assert first != second


def test_multiple_texts_produce_multiple_vectors():
    embedder = FakeEmbedder(dim=32)

    result = embedder.embed([
        "machine learning",
        "deep learning",
        "data science",
    ])

    assert len(result) == 3
    assert all(len(vector) == 32 for vector in result)


def test_empty_text_produces_valid_vector():
    embedder = FakeEmbedder(dim=64)

    vector = embedder.embed([""])[0]

    assert len(vector) == 64
    assert vector[0] == 1.0


def test_embedder_name():
    embedder = FakeEmbedder(dim=64)

    assert embedder.name == "fake-64"


def test_api_embedder_not_implemented():
    with pytest.raises(NotImplementedError):
        APIEmbedder()


def test_embedder_is_abstract():
    with pytest.raises(TypeError):
        Embedder()
