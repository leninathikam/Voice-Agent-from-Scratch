# Voice Agent from Scratch

A small cascade voice agent: microphone audio becomes text, a language model writes a short reply (with a few tools), and that reply is spoken back. You can cut the agent off mid-answer.

Offline by default (`PROVIDER=mock`). Set `PROVIDER=openai` and `OPENAI_API_KEY` when you want real speech recognition and speech synthesis.

## Run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.server:app --reload
```

Open http://127.0.0.1:8000 . Hold **Hold to talk**, then release. Press **Stop** while audio is playing to barge in.

```bash
pytest
```

## Architecture

One browser tab holds one WebSocket. That socket owns one `VoiceAgent`: its own memory, its own notes, and a cancel flag. Speech recognition, the language model, and speech synthesis are stateless and shared.

```mermaid
flowchart LR
  subgraph browser [Browser]
    mic[Hold to talk]
    play[Audio playback]
    stop[Stop]
  end
  subgraph server [FastAPI]
    ws[WebSocket /ws]
    agent[VoiceAgent]
    mem[Memory]
    notes[Session notes]
  end
  subgraph cascade [Cascade]
    asr[Speech to text]
    llm[Language model]
    tools[calculate / time / notes]
    tts[Text to speech]
  end
  mic -->|audio frame| ws --> agent
  agent --> asr --> agent
  agent --> mem
  agent --> llm
  llm <--> tools
  tools --> notes
  llm --> agent
  agent --> tts --> ws
  ws -->|mp3| play
  stop -->|cancel| ws
```

## How a turn works

```mermaid
flowchart TD
  mic[Microphone] --> rec[Browser records until you release]
  rec --> ws[WebSocket audio frame]
  ws --> asr[Speech to text]
  asr --> gate{Cancelled?}
  gate -->|yes| drop[Drop the turn]
  gate -->|no| mem[Append user turn to memory]
  mem --> llm[Language model]
  llm --> tools{Tool call?}
  tools -->|yes| exec[Run calculate, time, or notes]
  exec --> llm
  tools -->|no| reply[Short spoken reply]
  reply --> gate2{Cancelled?}
  gate2 -->|yes| rollback[Remove the unfinished user turn]
  gate2 -->|no| tts[Text to speech]
  tts --> play[Play audio in the browser]
  play --> barge{User presses Stop}
  barge -->|yes| cancel[Set the cancel flag]
  barge -->|no| idle[Idle, wait for the next utterance]
```

Each box is one module:

| Step | Module | Job |
| --- | --- | --- |
| Capture and playback | `static/index.html` | Records one utterance, plays MP3, sends `cancel` on barge-in |
| Session | `app/server.py` | One agent per WebSocket |
| Turn | `app/agent.py` | Orders ASR, memory, model, TTS, and checks cancellation |
| Memory | `app/memory.py` | Keeps the recent user and assistant turns |
| Tools | `app/tools.py` | Arithmetic, UTC time, session notes |
| Models | `app/providers.py` | Mock stack, or OpenAI transcription, chat, and speech |
