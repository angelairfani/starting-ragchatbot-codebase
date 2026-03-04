"""
Unit tests for AIGenerator tool-call flow using a mocked Anthropic client.
All tests here should pass — they test AIGenerator's two-call pattern in isolation.
"""
import pytest
from unittest.mock import MagicMock

from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_block(
    name: str = "search_course_content",
    input_data: dict = None,
    tool_id: str = "tu_123",
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data if input_data is not None else {"query": "what is Python?"}
    block.id = tool_id
    return block


def _response(stop_reason: str, content: list) -> MagicMock:
    r = MagicMock()
    r.stop_reason = stop_reason
    r.content = content
    return r


@pytest.fixture
def generator():
    """AIGenerator with a mock Anthropic client, no network calls."""
    gen = AIGenerator.__new__(AIGenerator)
    gen.client = MagicMock()
    gen.model = "claude-test"
    gen.base_params = {"model": "claude-test", "temperature": 0, "max_tokens": 800}
    return gen


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tool_use_triggers_second_api_call(generator):
    first = _response("tool_use", [_tool_block()])
    second = _response("end_turn", [_text_block("Final answer")])
    generator.client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    result = generator.generate_response(
        query="test query",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert generator.client.messages.create.call_count == 2
    assert result == "Final answer"


def test_correct_tool_name_passed_to_manager(generator):
    first = _response("tool_use", [_tool_block(name="search_course_content")])
    second = _response("end_turn", [_text_block("Done")])
    generator.client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    call_args = tool_manager.execute_tool.call_args
    assert call_args[0][0] == "search_course_content"


def test_tool_inputs_unpacked_as_kwargs(generator):
    inputs = {"query": "variables", "course_name": "Python", "lesson_number": 3}
    first = _response("tool_use", [_tool_block(input_data=inputs)])
    second = _response("end_turn", [_text_block("Done")])
    generator.client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content",
        query="variables",
        course_name="Python",
        lesson_number=3,
    )


def test_tool_result_in_second_call_messages(generator):
    first = _response("tool_use", [_tool_block(tool_id="tu_abc")])
    second = _response("end_turn", [_text_block("Done")])
    generator.client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search result text"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_kwargs = generator.client.messages.create.call_args_list[1][1]
    messages = second_kwargs["messages"]
    # Last message is the tool result user message
    tool_msg = messages[-1]
    assert tool_msg["role"] == "user"
    content = tool_msg["content"]
    assert isinstance(content, list)
    tool_result = next(r for r in content if r.get("type") == "tool_result")
    assert tool_result["tool_use_id"] == "tu_abc"
    assert tool_result["content"] == "search result text"


def test_second_call_includes_tools(generator):
    """Round 0 (second API call) is NOT the last round, so tools must be present."""
    first = _response("tool_use", [_tool_block()])
    second = _response("end_turn", [_text_block("Done")])
    generator.client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "results"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    second_kwargs = generator.client.messages.create.call_args_list[1][1]
    assert "tools" in second_kwargs


def test_two_sequential_tool_calls_makes_three_api_calls(generator):
    first = _response("tool_use", [_tool_block(tool_id="tu_1")])
    second = _response("tool_use", [_tool_block(tool_id="tu_2")])
    third = _response("end_turn", [_text_block("Final answer")])
    generator.client.messages.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    result = generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert generator.client.messages.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
    assert result == "Final answer"


def test_final_call_has_no_tools_after_two_rounds(generator):
    first = _response("tool_use", [_tool_block(tool_id="tu_1")])
    second = _response("tool_use", [_tool_block(tool_id="tu_2")])
    third = _response("end_turn", [_text_block("Done")])
    generator.client.messages.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    third_kwargs = generator.client.messages.create.call_args_list[2][1]
    assert "tools" not in third_kwargs


def test_second_round_messages_include_both_tool_results(generator):
    first = _response("tool_use", [_tool_block(tool_id="tu_1")])
    second = _response("tool_use", [_tool_block(tool_id="tu_2")])
    third = _response("end_turn", [_text_block("Done")])
    generator.client.messages.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "result"

    generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    third_kwargs = generator.client.messages.create.call_args_list[2][1]
    messages = third_kwargs["messages"]
    # user, assistant(tu1), user(tr1), assistant(tu2), user(tr2)
    assert len(messages) == 5
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[3]["role"] == "assistant"
    assert messages[4]["role"] == "user"


def test_tool_execution_error_doesnt_crash(generator):
    first = _response("tool_use", [_tool_block(tool_id="tu_1")])
    second = _response("end_turn", [_text_block("Graceful answer")])
    generator.client.messages.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = Exception("tool failed")

    result = generator.generate_response(
        query="test",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert generator.client.messages.create.call_count == 2
    assert result == "Graceful answer"
    # Verify the error string was passed as tool_result content
    second_kwargs = generator.client.messages.create.call_args_list[1][1]
    tool_msg = second_kwargs["messages"][-1]
    content = tool_msg["content"]
    assert any("tool failed" in str(r.get("content", "")) for r in content)


def test_direct_response_returned_without_tool_call(generator):
    text_response = _response("end_turn", [_text_block("Direct answer")])
    generator.client.messages.create.return_value = text_response

    tool_manager = MagicMock()

    result = generator.generate_response(
        query="what is 2+2?",
        tools=[{"name": "search_course_content"}],
        tool_manager=tool_manager,
    )

    assert result == "Direct answer"
    assert generator.client.messages.create.call_count == 1
    tool_manager.execute_tool.assert_not_called()
