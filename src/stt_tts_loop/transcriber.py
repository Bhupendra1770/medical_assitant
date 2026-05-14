

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY", "").strip())

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file using Groq Whisper model.
    Returns transcribed text.
    """
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file
        )
    return transcription.text
