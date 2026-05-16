"""Unit tests for app/llm.py."""

from unittest.mock import MagicMock, patch

from app.llm import build_prompt, generate_answer

# ████████████████████████████████████████████████████████████████████
# ██ BUILD PROMPT
# ████████████████████████████████████████████████████████████████████

def test_build_prompt_includes_question():
    result = build_prompt(["some context"], "what is AI?")
    assert "what is AI?" in result


def test_build_prompt_includes_all_chunks():
    result = build_prompt(["chunk one", "chunk two"], "question")
    assert "chunk one" in result
    assert "chunk two" in result


def test_build_prompt_joins_chunks_with_newlines():
    result = build_prompt(["first", "second"], "question")
    assert "first\n\nsecond" in result


def test_build_prompt_returns_string():
    result = build_prompt(["context"], "question")
    assert isinstance(result, str)


# ████████████████████████████████████████████████████████████████████
# ██ GENERATE ANSWER
# ████████████████████████████████████████████████████████████████████

def _make_mock_client(answer_text: str) -> MagicMock:
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = answer_text
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


def test_generate_answer_returns_text():
    mock_client = _make_mock_client("This is the answer.")
    with patch("app.llm._get_client", return_value=mock_client):
        result = generate_answer(["context"], "question")
    assert result == "This is the answer."


def test_generate_answer_returns_string():
    mock_client = _make_mock_client("answer")
    with patch("app.llm._get_client", return_value=mock_client):
        result = generate_answer(["context"], "question")
    assert isinstance(result, str)


def test_generate_answer_calls_chat_completions():
    mock_client = _make_mock_client("answer")
    with patch("app.llm._get_client", return_value=mock_client):
        generate_answer(["context"], "question")
    mock_client.chat.completions.create.assert_called_once()


def test_generate_answer_sends_question_in_prompt():
    mock_client = _make_mock_client("answer")
    with patch("app.llm._get_client", return_value=mock_client):
        generate_answer(["context"], "my specific question")
    call_kwargs = str(mock_client.chat.completions.create.call_args)
    assert "my specific question" in call_kwargs
