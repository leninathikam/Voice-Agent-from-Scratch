"""Speech and language providers.

`mock` runs fully offline so the pipeline can be checked without API keys.
`openai` uses Whisper, Chat Completions, and TTS.
"""

from __future__ import annotations

import json
from typing import Protocol

import httpx

from app.config import Settings
from app.tools import TOOL_SPECS, run_tool


class SpeechToText(Protocol):
    def transcribe(self, audio: bytes, mime: str) -> str: ...


class LanguageModel(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> bytes: ...


class MockASR:
    def transcribe(self, audio: bytes, mime: str) -> str:
        # Tests and the browser demo send UTF-8 text labeled as text/plain.
        if mime.startswith("text/") or mime == "application/json":
            return audio.decode("utf-8").strip()
        if not audio:
            return ""
        return audio.decode("utf-8", errors="ignore").strip()


class MockLLM:
    def complete(self, messages: list[dict]) -> str:
        user = ""
        for message in reversed(messages):
            if message["role"] == "user":
                user = message["content"]
                break
        lowered = user.lower()
        if "time" in lowered:
            return f"The time is {run_tool('current_time', {})}."
        if lowered.startswith("calculate "):
            expr = user.split(" ", 1)[1]
            return f"The result is {run_tool('calculate', {'expression': expr})}."
        if lowered.startswith("remember "):
            return run_tool("remember", {"note": user.split(" ", 1)[1]})
        if "notes" in lowered:
            return run_tool("list_notes", {})
        return f"I heard you say: {user}"


class MockTTS:
    def synthesize(self, text: str) -> bytes:
        # Not real audio. The client treats the mock provider as text-only.
        return text.encode("utf-8")


class OpenAIASR:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def transcribe(self, audio: bytes, mime: str) -> str:
        ext = "webm" if "webm" in mime else "wav"
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
                files={"file": (f"audio.{ext}", audio, mime or "audio/webm")},
                data={"model": self.settings.asr_model},
            )
            response.raise_for_status()
            return response.json()["text"].strip()


class OpenAILLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict]) -> str:
        working = list(messages)
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60) as client:
            for _ in range(4):
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": self.settings.chat_model,
                        "messages": working,
                        "tools": TOOL_SPECS,
                    },
                )
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                tool_calls = message.get("tool_calls") or []
                if not tool_calls:
                    return (message.get("content") or "").strip()
                working.append(message)
                for call in tool_calls:
                    name = call["function"]["name"]
                    raw = call["function"].get("arguments") or "{}"
                    result = run_tool(name, json.loads(raw))
                    working.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result,
                        }
                    )
        return "I could not finish that tool call."


class OpenAITTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def synthesize(self, text: str) -> bytes:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self.settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.tts_model,
                    "voice": self.settings.tts_voice,
                    "input": text,
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()
            return response.content


def build_stack(settings: Settings) -> tuple[SpeechToText, LanguageModel, TextToSpeech]:
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when PROVIDER=openai")
        return OpenAIASR(settings), OpenAILLM(settings), OpenAITTS(settings)
    return MockASR(), MockLLM(), MockTTS()
