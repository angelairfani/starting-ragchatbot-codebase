"""
Integration tests for the RAG query pipeline.

Uses a real temporary ChromaDB (via tmp_path) and a mocked Anthropic client.
The embedding function is stubbed out so no model is downloaded during tests.

Bug demonstration:
  test_add_course_content_with_none_lesson_number_raises
      EXPECTED TO FAIL before the fix is applied — ChromaDB 1.0.15 rejects
      None values in metadata with ValueError. After applying the fix in
      vector_store.py (filter out None lesson_number), this test will pass.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from models import CourseChunk
from vector_store import VectorStore


# ---------------------------------------------------------------------------
# Minimal embedding stub — avoids downloading a real sentence-transformer model
# ---------------------------------------------------------------------------

class _FixedEF:
    """
    Minimal embedding function stub for ChromaDB 1.0.x.

    ChromaDB calls name() during collection validation and checks is_legacy
    when serialising the embedding function config. Both must be present.
    """

    is_legacy = False

    def name(self) -> str:  # ChromaDB 1.0.x calls this on every collection open
        return "default"

    def __call__(self, input):  # noqa: A002
        return [[float(i % 10) / 10.0 for i in range(384)] for _ in input]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_store(tmp_path):
    """VectorStore backed by a fresh temp ChromaDB with a stub embedding fn."""
    with patch(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
        return_value=_FixedEF(),
    ):
        store = VectorStore(str(tmp_path), "all-MiniLM-L6-v2", max_results=5)
    return store


def _build_rag(tmp_path):
    """
    Construct a RAGSystem with:
      - real temp ChromaDB (stub embeddings)
      - mocked Anthropic client returning a canned end_turn response
    """
    config = SimpleNamespace(
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHROMA_PATH=str(tmp_path),
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="claude-test",
        MAX_RESULTS=5,
        MAX_HISTORY=2,
    )

    mock_anthropic = MagicMock()
    mock_msg = MagicMock()
    mock_msg.stop_reason = "end_turn"
    mock_msg.content = [MagicMock(text="I don't have specific information about that.")]
    mock_anthropic.messages.create.return_value = mock_msg

    with (
        patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
            return_value=_FixedEF(),
        ),
        patch("anthropic.Anthropic", return_value=mock_anthropic),
    ):
        from rag_system import RAGSystem
        rag = RAGSystem(config)

    return rag


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_add_course_content_with_none_lesson_number_raises(tmp_store):
    """
    EXPECTED TO FAIL before the fix is applied.

    ChromaDB 1.0.15 raises ValueError when metadata contains None:
        "Expected metadata value to be str, int, float or bool, got None"

    After the fix (filter None from metadata in add_course_content), this
    call must complete without raising and this test will pass.
    """
    chunk = CourseChunk(
        content="Content from a document with no lesson markers.",
        course_title="Python Basics",
        lesson_number=None,  # triggers the bug
        chunk_index=0,
    )
    # After the fix this should not raise.
    # Before the fix ChromaDB raises — causing this test to fail.
    tmp_store.add_course_content([chunk])


def test_add_course_content_succeeds_with_valid_lesson_number(tmp_store):
    """Control case: valid integer lesson_number should always succeed."""
    chunk = CourseChunk(
        content="Variables are containers for storing data values.",
        course_title="Python Basics",
        lesson_number=1,
        chunk_index=0,
    )
    tmp_store.add_course_content([chunk])  # must not raise


def test_query_with_empty_collection_does_not_raise(tmp_path):
    """rag_system.query() must return a string even when 0 docs are indexed."""
    rag = _build_rag(tmp_path)
    result = rag.query("What is Python?")

    assert isinstance(result, tuple)
    response, sources = result
    assert isinstance(response, str)
    assert isinstance(sources, list)


def test_query_returns_tuple_of_response_and_sources(tmp_path):
    """app.py unpacks rag_system.query() as (response, sources) — verify the shape."""
    rag = _build_rag(tmp_path)
    result = rag.query("Tell me about variables")

    assert isinstance(result, tuple)
    assert len(result) == 2
    response, sources = result
    assert isinstance(response, str)
    assert isinstance(sources, list)


def test_sources_cleared_between_queries(tmp_path):
    """
    reset_sources() is called after every query so sources from one call
    don't bleed into the next.
    """
    rag = _build_rag(tmp_path)

    rag.query("Question 1")
    assert rag.tool_manager.get_last_sources() == []

    rag.query("Question 2")
    assert rag.tool_manager.get_last_sources() == []
