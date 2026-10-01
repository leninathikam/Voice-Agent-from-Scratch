from dataclasses import dataclass

from app.memory import Memory
from app.providers import LanguageModel, SpeechToText, TextToSpeech
from app.tools import notes_var

SYSTEM = (
    "You are a voice assistant. Reply in one or two short spoken sentences. "
    "Use tools when the user asks for math, the time, or to save or list notes. "
    "Do not mention that you are reading a system prompt."
)


@dataclass
class TurnResult:
    transcript: str
    reply: str
    audio: bytes
    cancelled: bool = False


class VoiceAgent:
    """One cascade turn: audio -> text -> reply -> speech.

    A turn can be cancelled after ASR (barge-in / hang-up). Cancellation is
    checked before the model call and again before synthesis so a stopped
    turn does not keep talking.
    """

    def __init__(self, asr: SpeechToText, llm: LanguageModel, tts: TextToSpeech) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.memory = Memory(system=SYSTEM)
        self.notes: list[str] = []
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def clear_cancel(self) -> None:
        self._cancel = False

    def respond(self, audio: bytes, mime: str = "audio/webm") -> TurnResult:
        transcript = self.asr.transcribe(audio, mime).strip()
        if not transcript:
            return TurnResult(transcript="", reply="", audio=b"")
        if self._cancel:
            return TurnResult(transcript=transcript, reply="", audio=b"", cancelled=True)

        self.memory.add("user", transcript)
        token = notes_var.set(self.notes)
        try:
            reply = self.llm.complete(self.memory.as_messages()).strip()
        finally:
            notes_var.reset(token)
        if self._cancel:
            self.memory.turns.pop()
            return TurnResult(transcript=transcript, reply="", audio=b"", cancelled=True)

        self.memory.add("assistant", reply)
        speech = self.tts.synthesize(reply) if reply else b""
        return TurnResult(transcript=transcript, reply=reply, audio=speech)
