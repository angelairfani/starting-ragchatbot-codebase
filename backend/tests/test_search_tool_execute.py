"""
Unit tests for CourseSearchTool.execute() using a mocked VectorStore.
All tests here should pass — they test the tool's contract in isolation.
"""
import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(results: SearchResults = None, lesson_link: str = None) -> MagicMock:
    store = MagicMock()
    store.search.return_value = results if results is not None else SearchResults.empty("no results")
    store.get_lesson_link.return_value = lesson_link
    return store


def _make_results(docs=None, meta=None) -> SearchResults:
    docs = docs or ["Sample content about Python"]
    meta = meta or [{"course_title": "Python Basics", "lesson_number": 2, "chunk_index": 0}]
    return SearchResults(documents=docs, metadata=meta, distances=[0.1] * len(docs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_formatted_content_with_header():
    results = _make_results(
        docs=["Learn about variables."],
        meta=[{"course_title": "Python Basics", "lesson_number": 3, "chunk_index": 0}],
    )
    tool = CourseSearchTool(_make_store(results=results))
    output = tool.execute(query="variables")

    assert "[Python Basics - Lesson 3]" in output
    assert "Learn about variables." in output


def test_empty_results_returns_not_found_message():
    empty = SearchResults(documents=[], metadata=[], distances=[])
    tool = CourseSearchTool(_make_store(results=empty))
    output = tool.execute(query="something")

    assert "No relevant content found" in output


def test_empty_results_with_filters_includes_filter_info():
    empty = SearchResults(documents=[], metadata=[], distances=[])
    tool = CourseSearchTool(_make_store(results=empty))
    output = tool.execute(query="something", course_name="Python Basics", lesson_number=1)

    assert "Python Basics" in output
    assert "1" in output


def test_error_from_store_returned_as_string():
    error_results = SearchResults.empty("Search error: connection refused")
    tool = CourseSearchTool(_make_store(results=error_results))
    output = tool.execute(query="test")

    assert "Search error" in output
    # Must not raise — errors come back as strings


def test_query_forwarded_to_store():
    empty = SearchResults(documents=[], metadata=[], distances=[])
    store = _make_store(results=empty)
    tool = CourseSearchTool(store)

    tool.execute(query="my specific query")

    store.search.assert_called_once()
    call_kwargs = store.search.call_args
    # query may be positional or keyword
    actual_query = call_kwargs.kwargs.get("query") or (call_kwargs.args[0] if call_kwargs.args else None)
    assert actual_query == "my specific query"


def test_course_name_filter_forwarded():
    empty = SearchResults(documents=[], metadata=[], distances=[])
    store = _make_store(results=empty)
    tool = CourseSearchTool(store)

    tool.execute(query="test", course_name="Python Basics")

    store.search.assert_called_once_with(
        query="test", course_name="Python Basics", lesson_number=None
    )


def test_lesson_number_filter_forwarded():
    empty = SearchResults(documents=[], metadata=[], distances=[])
    store = _make_store(results=empty)
    tool = CourseSearchTool(store)

    tool.execute(query="test", lesson_number=5)

    store.search.assert_called_once_with(
        query="test", course_name=None, lesson_number=5
    )


def test_last_sources_populated_after_search():
    results = _make_results()
    tool = CourseSearchTool(_make_store(results=results))

    assert tool.last_sources == []
    tool.execute(query="test")
    assert len(tool.last_sources) > 0


def test_lesson_link_included_in_source():
    results = _make_results(
        docs=["Content"],
        meta=[{"course_title": "Python Basics", "lesson_number": 2, "chunk_index": 0}],
    )
    store = _make_store(results=results, lesson_link="https://example.com/lesson2")
    tool = CourseSearchTool(store)

    tool.execute(query="test")

    assert len(tool.last_sources) == 1
    assert "https://example.com/lesson2" in tool.last_sources[0]
