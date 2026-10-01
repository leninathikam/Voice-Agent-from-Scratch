import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    provider: str
    openai_api_key: str
    chat_model: str
    asr_model: str
    tts_model: str
    tts_voice: str


def load_settings() -> Settings:
    return Settings(
        provider=os.getenv("PROVIDER", "mock").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        asr_model=os.getenv("OPENAI_ASR_MODEL", "whisper-1"),
        tts_model=os.getenv("OPENAI_TTS_MODEL", "tts-1"),
        tts_voice=os.getenv("OPENAI_TTS_VOICE", "alloy"),
    )
