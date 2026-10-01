import asyncio
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import VoiceAgent
from app.config import load_settings
from app.providers import build_stack

settings = load_settings()
asr, llm, tts = build_stack(settings)

app = FastAPI(title="Voice Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def voice_socket(socket: WebSocket) -> None:
    await socket.accept()
    agent = VoiceAgent(asr, llm, tts)
    try:
        while True:
            message = await socket.receive_json()
            kind = message.get("type")
            if kind == "cancel":
                agent.cancel()
                await socket.send_json({"type": "cancelled"})
                continue
            if kind != "audio":
                await socket.send_json({"type": "error", "message": "unknown message"})
                continue
            agent.clear_cancel()
            raw = base64.b64decode(message.get("data") or "")
            mime = message.get("mime") or "audio/webm"
            result = await asyncio.to_thread(agent.respond, raw, mime)
            if result.cancelled:
                await socket.send_json({"type": "cancelled", "transcript": result.transcript})
                continue
            payload = {
                "type": "turn",
                "transcript": result.transcript,
                "reply": result.reply,
                "audio_b64": base64.b64encode(result.audio).decode("ascii") if result.audio else "",
                "audio_mime": "text/plain" if settings.provider == "mock" else "audio/mpeg",
            }
            await socket.send_json(payload)
    except WebSocketDisconnect:
        return
