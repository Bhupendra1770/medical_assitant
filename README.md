# 🏥 MediAssist — AI Medical Voice & Chat Assistant

A voice + text medical information chatbot with **RAG (pgvector on Neon PostgreSQL)**, **Groq Whisper STT**, **Edge TTS**, and a modern React UI.

---

## Architecture

```
Browser (React + Vite)
    │
    ├── WebSocket ws://host:8000  ← voice (binary audio in/out)
    │                               text (JSON messages)
    │
    └── HTTP http://host:8080     ← text chat REST API
                                    document upload / RAG management

Python Backend
    ├── WebSocket Server  (src/stt_tts_loop/websocket_server.py)
    │     ├── Groq Whisper  → Speech-to-Text
    │     ├── RAG Service   → pgvector semantic search
    │     ├── Groq LLaMA    → Medical LLM response
    │     └── Edge TTS      → Text-to-Speech
    │
    └── HTTP API Server    (src/chat_api/server.py)
          ├── POST /chat
          ├── POST /upload-document
          ├── GET  /documents
          └── DELETE /documents/{id}

PostgreSQL (Neon)
    └── medical_documents table (pgvector 384-dim embeddings)
```

---

## Quick Start

### 1. Clone / copy this project

```bash
cd /root
cp -r /path/to/medical_assistant voice_medical
cd voice_medical
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your keys:
#   GROQ_API_KEY=gsk_...
#   DATABASE_URL=postgresql://...@...neon.tech/neondb?sslmode=require
#   CHAT_API_KEY=your-secret-key
nano .env
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
# Also ensure ffmpeg is installed:
apt-get install -y ffmpeg   # Ubuntu/Debian
```

### 4. Start the WebSocket server (voice + text)

```bash
python -m src.stt_tts_loop
# Listens on ws://0.0.0.0:8000
```

### 5. Start the HTTP API (text chat + document upload)

```bash
uvicorn src.chat_api.server:app --host 0.0.0.0 --port 8080
```

### 6. Start the frontend

```bash
cd my-audio-app
cp .env.example .env
# Edit .env with your server URLs
npm install
npm run dev
```

---

## Features

### 🎤 Voice Chat
- Push-to-talk or auto VAD (voice activity detection)
- Groq Whisper transcription
- Edge TTS Indian English voice (NeerjaNeural)
- Audio plays back automatically; interrupts if you speak

### 💬 Text Chat
- Full conversation history maintained per session
- RAG context automatically injected into every response
- Markdown formatted responses in text mode

### 📚 RAG Knowledge Base
- Upload plain text medical documents (.txt files or paste text)
- Documents are chunked, embedded (all-MiniLM-L6-v2), and stored in pgvector
- Top-4 semantically relevant chunks retrieved per query
- Manage documents from the Knowledge Base tab

### 🏥 Medical Intelligence
- Symptom analysis and condition explanation
- Medication information (usage, dosage, side effects)
- Clinical guidelines from uploaded documents
- Always includes appropriate medical disclaimer

---

## pgvector Table Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE medical_documents (
    id         SERIAL PRIMARY KEY,
    title      TEXT,
    content    TEXT NOT NULL,
    embedding  vector(384),
    metadata   JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

The table is auto-created on startup if `DATABASE_URL` is set.

---

## Adding Medical Knowledge

Via the **Knowledge Base** tab in the UI:
- Paste drug information, clinical guidelines, disease descriptions
- Upload `.txt` files

Via API:
```bash
curl -X POST http://localhost:8080/upload-document \
  -H "x-api-key: dev-key" \
  -F "title=Paracetamol Info" \
  -F "content=Paracetamol (acetaminophen) is used to treat mild to moderate pain..."
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (STT + LLM) |
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `CHAT_API_KEY` | Secret key for HTTP API authentication |

Frontend (my-audio-app/.env):

| Variable | Default |
|----------|---------|
| `VITE_WS_URL` | `ws://localhost:8000` |
| `VITE_API_URL` | `http://localhost:8080` |
| `VITE_API_KEY` | `dev-key` |
