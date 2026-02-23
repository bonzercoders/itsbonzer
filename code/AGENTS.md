# Project Overview

Low-latency voice chat application with a FastAPI (Python) backend and JavaScript (React TypeScript) frontend. 
Single-user application. Do not over-engineer for enterprise scale or multi-tenancy.

## Stack/Libraries

- Backend: Python, FastAPI, Supabase.
- Frontend: JavaScript (React TypeScript), shadcn ui, Tailwind CSS.
- STT: RealtimeSTT (faster-whisper).
- TTS: Higgs Audio, Chatterbox.
- stream2sentence — sentence boundary detection.

## Principles/Style

- Adhere to KISS, YAGNI principles.
- Write code a human can read and maintain.

## Patterns

  - Producer/consumer with `asyncio.Queue`
  - `asyncio.create_task()` with `Queue.task_done()`
