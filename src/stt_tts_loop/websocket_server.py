"""
MediAssist WebSocket Server
Handles:
  • {"type":"voice",  "audio":"<base64>", "reqId": N}  → STT → RAG → LLM → TTS
  • {"type":"chat",   "message":"...",    "reqId": N}  → RAG → LLM → text reply
  • {"type":"ping"}                                    → pong
  • {"type":"clear_history"}                           → reset conversation
"""

import asyncio
import websockets
import tempfile
import os
import uuid
import shutil
import base64
import json
import logging

from src.stt_tts_loop.transcriber import transcribe_audio
from src.stt_tts_loop.tts_creator import generate_tts_webm
from src.stt_tts_loop.response_generator.medical_response import (
    get_generator, initialize_generator,
)
from src.rag.rag_service import get_rag_service, initialize_rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = "0.0.0.0"
PORT = 8000


async def handle_connection(websocket):
    logger.info("Client connected")
    generator = get_generator()

    try:
        async for message in websocket:

            # ── binary legacy fallback ───────────────────────────────────────
            if isinstance(message, bytes):
                await _handle_voice(websocket, message, generator, req_id=None)
                continue

            # ── JSON messages ────────────────────────────────────────────────
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                # base64 string (legacy)
                try:
                    audio_bytes = base64.b64decode(message)
                    await _handle_voice(websocket, audio_bytes, generator, req_id=None)
                except Exception:
                    pass
                continue

            msg_type = payload.get("type", "chat")
            req_id   = payload.get("reqId")

            if msg_type == "ping":
                await websocket.send(json.dumps({"type": "pong"}))
                continue

            if msg_type == "clear_history":
                if generator:
                    generator.clear_history()
                await websocket.send(json.dumps({"type": "history_cleared"}))
                continue

            if msg_type == "voice":
                b64 = payload.get("audio", "")
                if not b64:
                    continue
                try:
                    audio_bytes = base64.b64decode(b64)
                except Exception:
                    continue
                await _handle_voice(websocket, audio_bytes, generator, req_id=req_id)
                continue

            if msg_type == "chat":
                text = payload.get("message", "").strip()
                if not text:
                    continue
                await websocket.send(json.dumps({"type": "processing", "reqId": req_id}))
                response = await generator.generate(text, is_voice=False)
                await websocket.send(json.dumps({
                    "type": "text_response",
                    "response": response,
                    "reqId": req_id,
                }))
                continue

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")


async def _handle_voice(websocket, audio_bytes: bytes, generator, req_id):
    temp_dir  = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.webm")
    tts_path   = None

    try:
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        await websocket.send(json.dumps({"type": "processing", "reqId": req_id}))

        # 1. STT
        user_text = transcribe_audio(audio_path)
        logger.info(f"Transcribed: {user_text[:80]}")

        if not user_text.strip():
            await websocket.send(json.dumps({
                "type": "voice_response",
                "transcription": "",
                "response": "",
                "reqId": req_id,
            }))
            return

        # 2. LLM + RAG
        response_text = await generator.generate(user_text, is_voice=True)
        logger.info(f"Response: {response_text[:80]}")

        # 3. TTS
        tts_path = await generate_tts_webm(response_text)

        with open(tts_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        await websocket.send(json.dumps({
            "type":          "voice_response",
            "transcription": user_text,
            "response":      response_text,
            "audio_base64":  audio_b64,
            "reqId":         req_id,
        }))

    except Exception as e:
        logger.exception(f"Voice pipeline error: {e}")
        try:
            await websocket.send(json.dumps({"type": "error", "message": str(e), "reqId": req_id}))
        except Exception:
            pass
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if tts_path and os.path.exists(tts_path):
            os.remove(tts_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "")
    db_url   = os.getenv("DATABASE_URL", "")

    rag_ok = False
    if db_url:
        rag_ok = await initialize_rag(db_url)
        if not rag_ok:
            logger.warning("RAG unavailable — running without knowledge base")
    else:
        logger.warning("DATABASE_URL not set — RAG disabled")

    rag_service = get_rag_service() if rag_ok else None
    initialize_generator(groq_key, rag_service)

    async with websockets.serve(handle_connection, HOST, PORT):
        logger.info(f"MediAssist WebSocket running on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
