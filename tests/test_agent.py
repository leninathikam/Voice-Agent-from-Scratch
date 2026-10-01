from app.agent import VoiceAgent
from app.providers import MockASR, MockLLM, MockTTS
from app.tools import calculate, run_tool


def _agent() -> VoiceAgent:
    return VoiceAgent(MockASR(), MockLLM(), MockTTS())


def test_cascade_remembers_the_turn():
    agent = _agent()
    result = agent.respond(b"hello there", mime="text/plain")
    assert result.transcript == "hello there"
    assert "hello there" in result.reply
    assert result.audio == result.reply.encode()
    assert [t.role for t in agent.memory.turns] == ["user", "assistant"]


def test_empty_audio_does_not_call_the_model_history():
    agent = _agent()
    result = agent.respond(b"   ", mime="text/plain")
    assert result.reply == ""
    assert agent.memory.turns == []


def test_cancel_before_reply_drops_the_user_turn():
    agent = _agent()
    agent.cancel()
    result = agent.respond(b"stop talking", mime="text/plain")
    assert result.cancelled
    assert result.reply == ""
    assert agent.memory.turns == []


def test_calculate_allows_arithmetic_only():
    assert calculate("(2+3)*4") == "20.0"
    assert run_tool("calculate", {"expression": "2+2"}) == "4.0"
    blocked = run_tool("calculate", {"expression": "__import__('os').system('x')"})
    assert blocked.startswith("Tool error")


def test_notes_round_trip():
    agent = _agent()
    saved = agent.respond(b"remember buy milk", mime="text/plain")
    listed = agent.respond(b"what are my notes", mime="text/plain")
    assert "Saved" in saved.reply
    assert "buy milk" in listed.reply


def test_notes_stay_on_one_agent():
    first = _agent()
    second = _agent()
    first.respond(b"remember buy milk", mime="text/plain")
    listed = second.respond(b"what are my notes", mime="text/plain")
    assert "buy milk" not in listed.reply
    assert "No notes" in listed.reply
